#!/usr/bin/env python3
"""全链路 data_snapshot_id 对账 CLI。"""
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
from runtime.snapshot_reconcile import reconcile_data_snapshot
from workspace_paths import default_factor_lake_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile data_snapshot_id across catalog/run/parquet")
    parser.add_argument("--factor-id", required=True)
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--config", default=None, help="Optional YAML config for expected data_source hash")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lake_root = Path(args.lake_root or default_factor_lake_root())

    data_source_config = None
    if args.config:
        cfg = load_config(args.config)
        data_source_config = {"type": cfg.data_source.type, **cfg.data_source.options}

    report = reconcile_data_snapshot(
        factor_id=args.factor_id,
        lake_root=lake_root,
        data_source_config=data_source_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
