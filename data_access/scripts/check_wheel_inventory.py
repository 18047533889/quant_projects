#!/usr/bin/env python3
"""R26-P0-002 / §9.1 —— wheel 包清单完整性检查。

本项目是「物理目录 = data_access.X」的非标准布局（package-dir 映射），
setuptools 无法自动 discovery，必须显式列包。为防止后续新增子包漏进
pyproject.toml，此脚本构建 wheel 并校验：

    - 每个带 __init__.py 的顶层物理目录必须出现在 wheel 的 data_access.* 下
      （允许集合：core/registry/read/write/cos/service/clickhouse/quality/
      contract/security/runtime/snapshot/r30/export/telemetry）；
    - 顶层 data_access/__init__.py 必须存在。

用法：
    python -m data_access.scripts.check_wheel_inventory [--wheel-dir DIR]

退出码：0 = OK；1 = 有缺包（CI failure）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PKG_DIRS = {
    "core",
    "registry",
    "read",
    "write",
    "cos",
    "service",
    "clickhouse",
    "quality",
    "contract",
    "security",
    "runtime",
    "snapshot",
    "r30",
    "export",
    "telemetry",
}


def _build_wheel(workdir: Path) -> list[Path]:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(workdir)],
        cwd=str(PKG_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(workdir.glob("*.whl"))


def _physical_pkg_dirs() -> set[str]:
    return {
        d.name
        for d in PKG_ROOT.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name in ALLOWED_PKG_DIRS
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, default=None)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="r26-wheel-") if args.wheel_dir is None else None
    wheel_dir = args.wheel_dir or Path(tmp)
    try:
        wheels = _build_wheel(wheel_dir)
        if not wheels:
            print("R26 wheel inventory FAILED: 没有生成 wheel")
            return 1
        whl = wheels[0]
        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())

        expected = _physical_pkg_dirs()
        missing = []
        for pkg in sorted(expected):
            if not any(f"data_access/{pkg}/" in n for n in names):
                missing.append(f"data_access.{pkg}")
        if not any(n == "data_access/__init__.py" for n in names):
            missing.append("data_access/__init__.py")

        if missing:
            print(f"R26 wheel inventory FAILED: {whl.name} 缺 {len(missing)} 个子包：")
            for m in sorted(missing):
                print(f"  - {m}（R26-P0-002：pyproject.toml 显式包列表漏配）")
            return 1
        print(
            f"R26 wheel inventory OK: {whl.name} 含全部 {len(expected)} 个物理子包"
        )
        return 0
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
