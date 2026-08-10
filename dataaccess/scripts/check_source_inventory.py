#!/usr/bin/env python3
"""R26 §9.2 / P0-001 —— clean-tree source inventory 检查。

用途：保证 ``git clean`` / GitHub clean checkout 后所有 Python 源码都存在。
否则「本地 untracked/ignored 模块才能跑」会骗过本地产物测试。

检查：
    1. 所有 *.py 源文件必须被 Git 跟踪（除明确允许的 generated/backup 目录）。
    2. import graph：``data_access`` 下被 import 的模块文件必须存在于磁盘，
       且非 ignored（除 allowed 目录）。若缺，CI fail。

用法：
    python -m data_access.scripts.check_source_inventory

退出码：0 = OK；1 = 有 blocker。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_IGNORED_DIRS = (
    ".replace_backup_",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    ".egg-info",
    ".tmp",
    ".pytest_cache",
)


def _git(args: list[str]) -> list[str]:
    out = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _is_allowed(path: str) -> bool:
    return any(part in path for part in ALLOWED_IGNORED_DIRS)


def check_ignored_source() -> list[str]:
    problems: list[str] = []
    ignored = _git(["ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py"])
    for path in ignored:
        if not _is_allowed(path):
            problems.append(
                f"production 源码被 gitignore：{path}（R26 §9.2 / T-R26-CLEAN-004 fail）"
            )
    return problems


def _module_targets() -> set[str]:
    """收集 data_access 内被 import 的模块路径。"""
    targets: set[str] = set()
    for py in (REPO_ROOT / "dataaccess").rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            for kw in ("import ", "from "):
                if not line.startswith(kw):
                    continue
                # 取前 1-2 段（data_access.x / data_access.x.y）
                rest = line[len(kw):]
                for part in rest.split(";"):
                    head = part.strip()
                    if head.startswith("import "):
                        head = head[len("import "):]
                    elif head.startswith("from "):
                        head = head[len("from "):]
                    mod = head.split()[0].split(",")[0].strip()
                    if not mod.startswith("data_access."):
                        continue
                    targets.add(mod)
    return targets


def _target_to_path(mod: str) -> Path:
    # package-dir 映射：data_access = dataaccess/（磁盘目录无下划线）
    rel = mod[len("data_access."):].replace(".", "/")
    p = REPO_ROOT / "dataaccess" / rel
    if p.is_dir():
        return p / "__init__.py"
    if p.suffix == "":
        return p.with_suffix(".py")
    return p


def check_import_targets() -> list[str]:
    problems: list[str] = []
    for mod in sorted(_module_targets()):
        p = _target_to_path(mod)
        if p.is_dir():
            p = p / "__init__.py"
        if not p.exists():
            problems.append(f"import target 不存在：{mod} -> {p}")
            continue
        # 源码文件被 gitignore → clean clone 缺模块（等价 P0-001 blocker）
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if _is_allowed(str(rel)):
            continue
        r = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(rel)],
            cwd=str(REPO_ROOT),
            capture_output=True,
        )
        if r.returncode == 0:
            problems.append(f"import target 被 gitignore：{mod}（{rel}）")
    return problems


def main() -> int:
    problems = check_ignored_source() + check_import_targets()
    if problems:
        print("R26 source inventory FAILED:")
        for p in sorted(set(problems)):
            print(f"  - {p}")
        return 1
    print("R26 source inventory OK (all production .py tracked, import targets present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
