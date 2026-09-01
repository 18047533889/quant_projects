#!/usr/bin/env python3
"""Audit every report factor for usable (not merely indexed) full-window values."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT / "jobs"))
from factor_report_sources import resolve_raw_matrix

REPORT_DIR = ROOT / "factor_engine/docs/reports/2026-08-23/factors"
OUT = ROOT / "weekly_backtest_output/full_window_value_audit.json"


def main() -> int:
    records = []
    pages = sorted(p.stem.removeprefix("factor_") for p in REPORT_DIR.glob("factor_*.html"))
    for i, page in enumerate(pages, 1):
        source = resolve_raw_matrix(page)
        record = {
            "page": page,
            "status": "ready" if source.is_full_window else "needs_landing",
            "source": source.source,
            "path": str(source.path) if source.path else None,
            "index_start": str(source.start.date()) if source.start is not None else None,
            "index_end": str(source.end.date()) if source.end is not None else None,
            "valid_start": str(source.valid_start.date()) if source.valid_start is not None else None,
            "valid_end": str(source.valid_end.date()) if source.valid_end is not None else None,
            "valid_days": source.valid_days,
            "reason": source.reason,
        }
        records.append(record)
        print(f"[{i}/{len(pages)}] {record['status']} {page} {record['valid_days']}", flush=True)
    summary = {
        "total": len(records),
        "ready": sum(r["status"] == "ready" for r in records),
        "needs_landing": sum(r["status"] == "needs_landing" for r in records),
    }
    OUT.write_text(json.dumps({"summary": summary, "factors": records}, ensure_ascii=False, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
