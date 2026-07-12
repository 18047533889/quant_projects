#!/usr/bin/env python3
"""Restore RankIC / decile / LS charts on factor HTML reports (TopK metrics kept from progress)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb, build_fwd_returns_cache
from scripts.cogalpha_lqtp.report_html import render_factor_report
from scripts.cogalpha_lqtp.run_production_batch import _apply_flip_state_to_catalog


def _has_panel_charts(html: str) -> bool:
    return (
        'alt="Daily RankIC"' in html
        and 'alt="Group PnL"' in html
        and 'alt="Cumulative LS"' in html
        and "src=\"data:image/png;base64," in html
    )


def _topk_summary_from_row(row: dict) -> dict:
    return {
        "total_return": row.get("backtest_total_ret"),
        "sharpe": row.get("backtest_sharpe"),
        "max_drawdown": row.get("backtest_max_drawdown"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair factor HTML panel charts")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--all-with-topk", action="store_true", default=True)
    args = parser.parse_args()

    work = args.work_dir
    progress = json.loads((work / "production_progress.json").read_text(encoding="utf-8"))
    catalog = {e["function_name"]: e for e in json.loads((work / "dsl_catalog.json").read_text())}
    parsed = {r["function_name"]: r["python_code"] for r in json.loads((work / "parsed_factors.json").read_text())}
    flip = progress.get("flip_state", {})
    _apply_flip_state_to_catalog(
        json.loads((work / "dsl_catalog.json").read_text()), flip, parsed
    )

    fwd = build_fwd_returns_cache(
        work / "lqtp_close_returns_cache.parquet",
        work / "lqtp_fwd_close_returns_cache.parquet",
    )
    report_dir = work / "reports"
    fixed = 0

    for row in progress.get("index_rows", []):
        name = row["factor_name"]
        if args.only and name not in set(args.only):
            continue
        bt = row.get("backtest_sharpe")
        if bt is None or (isinstance(bt, float) and math.isnan(bt)):
            continue
        report_path = report_dir / row.get("report", f"{name}.html")
        if not report_path.exists():
            continue
        html = report_path.read_text(encoding="utf-8", errors="replace")
        if _has_panel_charts(html):
            continue

        pq = work / "factor_lake" / name / "values.parquet"
        if not pq.exists():
            continue
        entry = catalog.get(name, {})
        py = parsed.get(name, "")
        if flip.get(name, {}).get("python_code"):
            py = flip[name]["python_code"]
        dsl = (entry.get("dsl") or "").strip() or "(python only)"
        print(f"repair charts: {name}")
        analysis = analyze_factor_parquet_duckdb(factor_path=pq, fwd_returns_path=fwd)
        render_factor_report(
            factor_name=name,
            dsl=dsl,
            analysis=analysis,
            backtest_rows=[],
            out_path=report_path,
            eval_mode=row.get("eval_mode", ""),
            topk_summary=_topk_summary_from_row(row),
            python_code=py,
            work_dir=work,
        )
        fixed += 1

    print(f"done repaired={fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
