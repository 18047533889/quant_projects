#!/usr/bin/env python3
"""因子湖 staging → published 发布 CLI（需审批）。"""
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

from runtime.config import load_config
from storage.lake_publish import PublishNotApprovedError, publish_factor_lake
from workspace_paths import default_factor_lake_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish factor lake staging to published")
    parser.add_argument("--factor-id", required=True)
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--config", default=None, help="Optional YAML for snapshot reconcile")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicit approval (or set QUANT_PUBLISH_APPROVED=1)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not copy local lake partitions to staging before publish",
    )
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Skip data_snapshot_id reconciliation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_source_config = None
    if args.config:
        cfg = load_config(args.config)
        data_source_config = {"type": cfg.data_source.type, **cfg.data_source.options}

    try:
        result = publish_factor_lake(
            factor_id=args.factor_id,
            lake_root=args.lake_root or default_factor_lake_root(),
            approve=args.approve,
            sync_from_local=not args.skip_sync,
            reconcile=not args.skip_reconcile,
            data_source_config=data_source_config,
        )
    except PublishNotApprovedError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
