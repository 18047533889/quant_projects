#!/usr/bin/env python3
"""因子湖 publish 归档回滚 CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "factor_engine"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from storage.lake_version import (
    diff_factor_trees,
    list_publish_archives,
    rollback_factor_publish,
)
from workspace_paths import default_factor_lake_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor lake version diff / rollback")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list-archives", help="List publish archives")
    list_cmd.add_argument("--factor-id", required=True)
    list_cmd.add_argument("--lake-root", default=None)

    diff_cmd = sub.add_parser("diff", help="Diff current vs archive")
    diff_cmd.add_argument("--factor-id", required=True)
    diff_cmd.add_argument("--archive", default=None)
    diff_cmd.add_argument("--lake-root", default=None)

    rb = sub.add_parser("rollback", help="Rollback to archive")
    rb.add_argument("--factor-id", required=True)
    rb.add_argument("--archive", default=None)
    rb.add_argument("--lake-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lake_root = Path(args.lake_root or default_factor_lake_root())

    if args.command == "list-archives":
        archives = list_publish_archives(args.factor_id, lake_root)
        payload = {"factor_id": args.factor_id, "archives": [str(p) for p in archives]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "diff":
        current = lake_root / "factors" / args.factor_id
        if args.archive:
            archive = Path(args.archive)
        else:
            archives = list_publish_archives(args.factor_id, lake_root)
            if not archives:
                print(json.dumps({"ok": False, "error": "no archives"}, ensure_ascii=False))
                return 1
            archive = archives[-1]
        report = diff_factor_trees(archive, current)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "rollback":
        result = rollback_factor_publish(
            factor_id=args.factor_id,
            lake_root=lake_root,
            archive_path=args.archive,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
