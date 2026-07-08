#!/usr/bin/env python3
"""factor_run 与 data_access audit 关联查询 CLI。"""
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

from runtime.run_audit_correlate import correlate_runs_with_audit, read_audit_log
from storage.catalog import FactorCatalog
from workspace_paths import default_factor_lake_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate factor_run with data_access audit")
    parser.add_argument("--factor-id", default=None)
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--audit-log", default=None)
    parser.add_argument("--max-delta-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lake_root = Path(args.lake_root or default_factor_lake_root())
    catalog = FactorCatalog(lake_root / "_catalog.sqlite")
    audits = read_audit_log(args.audit_log)
    report = correlate_runs_with_audit(
        catalog,
        audits,
        factor_id=args.factor_id,
        max_delta_seconds=args.max_delta_seconds,
    )
    print(json.dumps({"count": len(report), "items": report}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
