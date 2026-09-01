#!/usr/bin/env python3
"""Rebuild stale detail pages that claim full-window data is unavailable.

The marker was written by an earlier renderer even where a 2016--2026 raw
matrix already exists.  This job deliberately selects only those stale pages,
uses the canonical resolver through ``stage_page_inject``, and refuses to
report success unless the rebuilt HTML states the verified full window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT / "jobs"))

import incremental_factor_intake as intake
from factor_report_sources import FULL_WINDOW_END, FULL_WINDOW_START, resolve_raw_matrix

REPORTS = ROOT / "factor_engine/docs/reports/2026-08-23/factors"
STALE_MARKER = "尚未完成全窗回测"


def pages_with_stale_marker() -> list[str]:
    return sorted(
        p.stem.removeprefix("factor_")
        for p in REPORTS.glob("factor_*.html")
        if STALE_MARKER in p.read_text(encoding="utf-8", errors="ignore")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated page names")
    args = parser.parse_args()
    selected = [x for x in args.only.split(",") if x] if args.only else pages_with_stale_marker()
    meta = json.loads((ROOT / "weekly_backtest_output/optimized_meta.json").read_text())
    failures: list[str] = []
    for index, page in enumerate(selected, 1):
        source = resolve_raw_matrix(page)
        if not source.is_full_window:
            failures.append(f"{page}: no verified full-window raw matrix ({source.reason})")
            continue
        factor = {
            "page_name": page,
            "factor_name": page,
            "is_flipped": bool((meta.get(page) or {}).get("is_flipped", False)),
            "local_formula": "",
        }
        outcome = intake.stage_page_inject(factor, {})
        output = REPORTS / f"factor_{page}.html"
        html = output.read_text(encoding="utf-8", errors="ignore") if output.exists() else ""
        expected_window = f"{FULL_WINDOW_START.date()} ~ {FULL_WINDOW_END.date()}"
        if outcome.get("mode") != "full" or STALE_MARKER in html or expected_window not in html:
            failures.append(f"{page}: incomplete rebuild {outcome}")
            continue
        print(f"[{index}/{len(selected)}] rebuilt {page}: {source.path.name}", flush=True)
    if failures:
        print("FAILED", *failures, sep="\n", flush=True)
        return 1
    print(f"PASS rebuilt={len(selected)} full_window={FULL_WINDOW_START.date()}..{FULL_WINDOW_END.date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
