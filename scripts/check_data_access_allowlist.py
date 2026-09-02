#!/usr/bin/env python3
"""
scripts/check_data_access_allowlist.py —— data_access 禁用 API 静态检查

目的
----
团队规范：业务代码一律走 data_access.store，不得直接 import duckdb /
pd.read_parquet / pq.read_table / DataFrame.to_parquet。
豁免靠 `.data_access_allowlist.yaml` 显式登记。

本脚本扫描仓库里所有 .py 文件，用 AST 识别下列「禁用 API」：
    import duckdb                           (Import / ImportFrom)
    xxx.connect(...) where xxx 来自 `import duckdb as xxx` 的绑定
    xxx.read_parquet / xxx.to_parquet
    xxx.read_table / xxx.write_table / xxx.ParquetFile / xxx.ParquetDataset
    ——只要是 Attribute 访问 + 属性名命中，就报；不下推上下文证明调用者一定是 pandas/pyarrow。
      AST 检测比正则精确：不会误伤 docstring / 注释 / 正则字符串里的片段。

如果违反者的路径没出现在 allowlist 里，以非零退出码报错并列出违规点。

用法
----
    python scripts/check_data_access_allowlist.py                 # 扫描整个仓库
    python scripts/check_data_access_allowlist.py path/to/file.py # 只扫指定文件（pre-commit 走这条）

退出码
------
    0 = 无违规
    1 = 有违规
    2 = 脚本自身错误（allowlist 缺失、解析失败等）

维护
----
新增豁免：编辑 `.data_access_allowlist.yaml`，在合适的分节下加 `- path: xxx` 并写 reason。
路径支持 glob：`data_access/tests/**`、`**/tests/**`、`strategy_layer/**/tests/**` 都行。
"""

from __future__ import annotations

import ast
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    sys.stderr.write("ERROR: pyyaml 未安装。pip install pyyaml 后再试。\n")
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / ".data_access_allowlist.yaml"


# ---- 禁用属性名 -------------------------------------------------------------
# 这些属性名一旦出现在 Attribute 访问里就算违规。覆盖的调用：
#   pd.read_parquet / pandas.read_parquet
#   pq.read_table   / pyarrow.parquet.read_table
#   pq.write_table  / pyarrow.parquet.write_table
#   pq.ParquetFile  / pq.ParquetDataset
#   df.to_parquet   / series.to_parquet
# 精度策略：不追 alias → 直接看「属性名」。优点是误伤极少（哪个 pandas 外的库会
# 叫自己的方法 read_parquet？）；缺点是会漏掉 `getattr(pd, "read_parquet")` 这种
# 动态调用——但那是反模式，报不报都没影响。
_BANNED_ATTR_NAMES: dict[str, str] = {
    "read_parquet":    "pd.read_parquet",
    "read_table":      "pq.read_table",
    "write_table":     "pq.write_table",
    "ParquetFile":     "pq.ParquetFile",
    "ParquetDataset":  "pq.ParquetDataset",
    "to_parquet":      ".to_parquet",
}
# `duckdb.connect` 单独处理：属性名 `connect` 太通用（asyncio.connect / mongo.connect 都用）。
# 需要 value 是 Name `duckdb` 或 Attribute `duckdb.xxx` 才算。


# 扫描时忽略的顶层目录（都是第三方 / 临时产物 / 构建产物）
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "workspace_data",
    "dist",
    "build",
    ".claude",
    ".vscode",
    ".idea",
    "site-packages",
    ".backend-runtime",
    ".pykx-runtime",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    col: int
    label: str
    snippet: str


def load_allowlist() -> list[str]:
    """读取 allowlist。只抽顶层 list 里的 `- path: xxx` 条目。"""
    if not ALLOWLIST_PATH.exists():
        sys.stderr.write(f"ERROR: 找不到 allowlist: {ALLOWLIST_PATH}\n")
        sys.exit(2)
    try:
        raw = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        sys.stderr.write(f"ERROR: allowlist YAML 解析失败: {exc}\n")
        sys.exit(2)

    if raw is None:
        return []
    if isinstance(raw, list):
        entries: list[object] = list(raw)
    elif isinstance(raw, dict):
        entries = []
        for value in raw.values():
            if isinstance(value, list):
                entries.extend(value)
    else:
        sys.stderr.write(
            f"ERROR: allowlist 顶层类型不识别: {type(raw).__name__}\n"
        )
        sys.exit(2)

    patterns: list[str] = []
    for item in entries:
        if isinstance(item, dict) and "path" in item:
            patterns.append(str(item["path"]))
    return patterns


def is_allowed(rel_path: str, patterns: list[str]) -> bool:
    """判断相对路径是否在 allowlist 里。`**` 视作任意深度通配。"""
    candidate = PurePosixPath(rel_path)
    for pattern in patterns:
        if pattern == rel_path:
            return True
        try:
            if candidate.match(pattern):
                return True
        except (ValueError, TypeError):
            pass
        if fnmatch.fnmatch(rel_path, pattern.replace("**", "*")):
            return True
    return False


def _is_duckdb_access(node: ast.Attribute) -> bool:
    """判断 Attribute 是否形如 `duckdb.xxx` 或 `xxx.duckdb.yyy`（罕见）。"""
    cur: ast.AST = node.value
    while isinstance(cur, ast.Attribute):
        if cur.attr == "duckdb":
            return True
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id == "duckdb"


def _attr_snippet(source_lines: list[str], node: ast.Attribute) -> str:
    """尽量给出一行上下文：属性所在那一行 strip 后的内容。"""
    if 1 <= node.lineno <= len(source_lines):
        return source_lines[node.lineno - 1].strip()
    return f"<attr:{node.attr}>"


def scan_file(path: Path, rel: str) -> list[Violation]:
    """扫单个 .py 文件。纯 AST 路径。"""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    violations: list[Violation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "duckdb" or alias.name.startswith("duckdb."):
                    violations.append(
                        Violation(
                            path=rel,
                            line=node.lineno,
                            col=node.col_offset,
                            label="import duckdb",
                            snippet=f"import {alias.name}"
                            + (f" as {alias.asname}" if alias.asname else ""),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "duckdb" or (
                node.module and node.module.startswith("duckdb.")
            ):
                names = ", ".join(alias.name for alias in node.names)
                violations.append(
                    Violation(
                        path=rel,
                        line=node.lineno,
                        col=node.col_offset,
                        label="from duckdb import",
                        snippet=f"from {node.module} import {names}",
                    )
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTR_NAMES:
                violations.append(
                    Violation(
                        path=rel,
                        line=node.lineno,
                        col=node.col_offset,
                        label=_BANNED_ATTR_NAMES[node.attr],
                        snippet=_attr_snippet(source_lines, node),
                    )
                )
            elif node.attr == "connect" and _is_duckdb_access(node):
                violations.append(
                    Violation(
                        path=rel,
                        line=node.lineno,
                        col=node.col_offset,
                        label="duckdb.connect",
                        snippet=_attr_snippet(source_lines, node),
                    )
                )

    return violations


def iter_py_files(targets: list[Path]) -> list[Path]:
    """给定若干路径（文件或目录），返回所有 .py 文件。"""
    out: list[Path] = []
    for target in targets:
        if target.is_file():
            if target.suffix == ".py":
                out.append(target)
            continue
        if target.is_dir():
            # 用 pathlib.walk-like 遍历，手动跳过 SKIP_DIRS
            for py in _walk_py(target):
                out.append(py)
    return out


def _walk_py(root: Path):
    """walk with dir pruning：比 rglob 省时间（遇到 .venv 立刻剪）。"""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        # in-place filter out skipped dirs
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def main(argv: list[str]) -> int:
    patterns = load_allowlist()

    if len(argv) > 1:
        targets = [Path(arg).resolve() for arg in argv[1:] if arg]
    else:
        targets = [REPO_ROOT]

    files = iter_py_files(targets)
    if not files:
        return 0

    all_violations: list[Violation] = []
    for py in files:
        try:
            rel = py.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        # 豁免检查：整个文件在 allowlist 里就跳过扫描
        if is_allowed(rel, patterns):
            continue
        all_violations.extend(scan_file(py, rel))

    if not all_violations:
        return 0

    sys.stderr.write(
        "\n❌ 检测到 data_access 禁用 API（不在 .data_access_allowlist.yaml 里）:\n\n"
    )
    by_file: dict[str, list[Violation]] = {}
    for v in all_violations:
        by_file.setdefault(v.path, []).append(v)
    for file_path in sorted(by_file):
        sys.stderr.write(f"  {file_path}\n")
        for v in by_file[file_path]:
            sys.stderr.write(f"    line {v.line:>4}: [{v.label}] {v.snippet}\n")
        sys.stderr.write("\n")

    sys.stderr.write(
        "——————\n"
        "要么换成 data_access.get_store() / get_shared_engine() 的正式 API，\n"
        "要么在 .data_access_allowlist.yaml 里加 `- path: xxx` + reason 登记豁免。\n"
        "细节参考 data_access/docs/用户使用手册.md 与 data_access/README.md\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
