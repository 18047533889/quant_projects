#!/usr/bin/env python3
# -*- coding: utf-8
"""因子湖 Parquet metadata 列迁移 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from factor_engine.storage.schema_migration import migrate_factor_lake_tree
from factor_engine.util.workspace_paths import default_factor_lake_root


def main() -> int:
    """补全因子湖 Parquet metadata 列（默认 dry-run，``--apply`` 实际写入）。"""
    parser = argparse.ArgumentParser(description="补全因子湖 Parquet metadata 列")
    parser.add_argument(
        "--lake-root",
        default=str(default_factor_lake_root()),
        help="因子湖根目录（含 factors/ 子目录）",
    )
    parser.add_argument("--factor-id", default=None, help="仅迁移指定 factor_id")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际写入（默认 dry-run 仅统计）",
    )
    args = parser.parse_args()
    report = migrate_factor_lake_tree(
        args.lake_root,
        factor_id=args.factor_id,
        dry_run=not args.apply,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
