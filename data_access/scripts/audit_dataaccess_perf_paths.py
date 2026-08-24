#!/usr/bin/env python3
"""R30-P1-027 / R30-§67 —— 静态性能路径审计（pandas materialization 检测）。

对 production **热路径**源码做静态正则扫描（**不执行业务**）：

    scan roots: read/ runtime/ write/ core/engine.py store.py

检测三类会绕开 governed 读管线 / 造成大结果集 pandas 物化的模式：

    PATTERN_TO_PANDAS         ``.to_pandas()``        —— 读路径大结果集转 pandas
                                                         （内存膨胀 / 类型丢失 / 绕过 Arrow）
    PATTERN_PD_READ_PARQUET   ``pd/pandas.read_parquet`` —— 绕过 registry/预算直读 parquet
    PATTERN_DUCKDB_READ_PARQUET ``duckdb.read_parquet(``  —— 绕过 engine 直连 DuckDB 读
    PATTERN_RAW_SQL_PARQUET   行内同时出现 ``.execute*`` 与 ``read_parquet(``
                                                         —— 绕过 snapshot/verifier 的
                                                         裸 DuckDB SQL 读

每一处命中如果不在 **allowlist**（显式允许的 reference/debug/写路径/受控原语），
计为违规。allowlist 只放真正合法的地方；``quality/ scripts/ tests/ benchmarks/``
是 reference/debug 区域，**不在扫描范围内**。

只做静态检查，不 import 业务包、不连接数据源。退出码：0 = 0 违规；1 = 有违规。

用法：
    python3 scripts/audit_dataaccess_perf_paths.py [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

# ---- 扫描范围：production 热路径 ----
SCAN_ROOTS = ("read", "runtime", "write")
SCAN_FILES = ("core/engine.py", "store.py")

# ---- 检测模式（只匹配单行）----
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("to_pandas", re.compile(r"\.to_pandas\s*\(")),
    ("pd_read_parquet", re.compile(r"\b(?:pd|pandas)\.read_parquet\s*\(")),
    ("duckdb_read_parquet", re.compile(r"\bduckdb\.read_parquet\s*\(")),
    # 裸 DuckDB SQL 读：同一行既调用 execute* 又出现 read_parquet(
    (
        "raw_sql_read_parquet",
        re.compile(r"\.execute(?:_arrow|_df|_table|_stream|_relation)?\s*\([^)]*read_parquet\s*\("),
    ),
)


@dataclass(frozen=True)
class AllowlistEntry:
    """一条允许豁免：文件后缀 + 行内容子串（对行号变化稳健）。"""

    file_suffix: str
    line_substring: str
    reason: str


# 显式 allowlist——只放真正合法的 reference/debug/写路径/受控原语。
ALLOWLIST: tuple[AllowlistEntry, ...] = (
    AllowlistEntry(
        "read/resample.py",
        "table.to_pandas()",
        "standalone frequency resample：文档化 pandas 工具（groupby+Grouper），"
        "不经过 governed 读热路径",
    ),
    AllowlistEntry(
        "write/upsert.py",
        "table.to_pandas()",
        "写路径 upsert 分区拆分：批次小（一次 flush 几百~几万行），性能差距可忽略",
    ),
    AllowlistEntry(
        "write/upsert.py",
        "pq.read_table",
        "写路径 delete_rows 读旧 parquet：写治理场景，非读热路径",
    ),
    AllowlistEntry(
        "store.py",
        "df = current.to_pandas()",
        "delete_rows 写/变更路径读取当前代：非读热路径",
    ),
    AllowlistEntry(
        "core/engine.py",
        "read_parquet",
        "DuckDBEngine 底层文件加载原语：受控引擎层，非绕过治理",
    ),
    # ---- 有意的 pandas 输出层转换（governed 读完成后才物化给调用方）----
    AllowlistEntry(
        "core/engine.py",
        "return table.to_pandas(self_destruct=True, split_blocks=True)",
        "execute_df 公共 API：SQL 执行完成后按契约返回 pandas DataFrame",
    ),
    AllowlistEntry(
        "read/adapters.py",
        "to_pandas(self_destruct=True, split_blocks=True)",
        "Arrow→MultiIndex Series 输出适配器（load_columns 终端转换）",
    ),
    AllowlistEntry(
        "read/read_handle.py",
        "self.to_arrow().to_pandas(split_blocks=True)",
        "ReadHandle.to_pandas 终端消费方法（公共 API 契约）",
    ),
    AllowlistEntry(
        "read/relation_handle.py",
        "return self.arrow().to_pandas(split_blocks=True)",
        "RelationHandle.pandas 终端消费方法（公共 API 契约）",
    ),
    AllowlistEntry(
        "store.py",
        "return table.to_pandas(self_destruct=True, split_blocks=True)",
        "read_frame 公共 API：governed 读完成后按契约返回 pandas DataFrame",
    ),
)


def _iter_code_lines(path: str):
    """逐行 yield (lineno, line)；跳过注释与 docstring（简单三引号状态机）。"""
    in_triple = False
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            n_triple = stripped.count('"""') + stripped.count("'''")
            if in_triple:
                if n_triple % 2 == 1:
                    in_triple = False
                continue
            if stripped.startswith(('"""', "'''")):
                if n_triple % 2 == 1:
                    in_triple = True
                continue
            if stripped.startswith("#"):
                continue
            yield lineno, line


def _collect_files(repo_root: str) -> list[str]:
    files: list[str] = []
    for root in SCAN_ROOTS:
        base = os.path.join(repo_root, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith(".py"):
                    files.append(os.path.join(dirpath, fn))
    for rel in SCAN_FILES:
        p = os.path.join(repo_root, rel)
        if os.path.isfile(p):
            files.append(p)
    return sorted(files)


def _is_allowed(file_path: str, line: str) -> str | None:
    """命中 allowlist 返回 reason，否则 None。"""
    norm = file_path.replace(os.sep, "/")
    for entry in ALLOWLIST:
        if norm.endswith(entry.file_suffix) and entry.line_substring in line:
            return entry.reason
    return None


def audit_perf_paths(repo_root: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = _collect_files(repo_root)
    hits: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "files_scanned": len(files),
        "pattern_counts": {name: 0 for name, _ in PATTERNS},
        "allowlisted": 0,
        "violations": 0,
    }

    for path in files:
        try:
            code_lines = list(_iter_code_lines(path))
        except OSError as exc:
            print(f"[WARN] 读取 {path} 失败: {exc}", file=sys.stderr)
            continue
        rel = os.path.relpath(path, repo_root).replace(os.sep, "/")
        for lineno, line in code_lines:
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    counts["pattern_counts"][name] += 1
                    reason = _is_allowed(rel, line)
                    hits.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "pattern": name,
                            "code": line.strip(),
                            "allowed": reason is not None,
                            "reason": reason,
                        }
                    )
                    if reason is not None:
                        counts["allowlisted"] += 1
                    else:
                        counts["violations"] += 1

    violations = [h for h in hits if not h["allowed"]]
    return violations, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="静态性能路径审计（pandas materialization 检测）"
    )
    parser.add_argument("--root", default=None, help="仓库根目录（缺省=脚本上级）")
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果（含 counts + violations）",
    )
    args = parser.parse_args()

    repo_root = args.root
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    violations, counts = audit_perf_paths(repo_root)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": len(violations) == 0,
                    "counts": counts,
                    "violations": violations,
                    "violation_count": len(violations),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not violations else 1

    pc = ", ".join(f"{k}={v}" for k, v in counts["pattern_counts"].items())
    print(
        f"性能路径静态审计：{counts['files_scanned']} 个热路径文件 "
        f"(模式命中: {pc})"
    )
    print(f"  allowlisted（合法豁免）: {counts['allowlisted']} 处")

    if not violations:
        print("结果: 0 违规")
        return 0

    print(f"结果: {len(violations)} 违规")
    for v in violations:
        suffix = ""
        if v["reason"]:
            suffix = f"  [allowlisted: {v['reason']}]"
        print(
            f"  - [{v['pattern']}] {v['file']}:{v['line']}: "
            f"{v['code']}{suffix}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
