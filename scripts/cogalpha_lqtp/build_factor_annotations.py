#!/usr/bin/env python3
"""Generate factor_annotations.json and optionally refresh HTML reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import analyze_factor_parquet_duckdb, build_fwd_returns_cache  # noqa: E402
from scripts.cogalpha_lqtp.factor_annotations import build_all_annotations, save_annotations  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report  # noqa: E402


def _refresh_reports(work_dir: Path, names: list[str] | None = None) -> int:
    progress = json.loads((work_dir / "production_progress.json").read_text(encoding="utf-8"))
    catalog = {e["function_name"]: e for e in json.loads((work_dir / "dsl_catalog.json").read_text())}
    parsed = {r["function_name"]: r for r in json.loads((work_dir / "parsed_factors.json").read_text())}
    flip = progress.get("flip_state", {})
    fwd = build_fwd_returns_cache(
        work_dir / "lqtp_close_returns_cache.parquet",
        work_dir / "lqtp_fwd_close_returns_cache.parquet",
    )
    report_dir = work_dir / "reports"
    rows = progress.get("index_rows", [])
    if names:
        allow = set(names)
        rows = [r for r in rows if r["factor_name"] in allow]
    n = 0
    for row in rows:
        name = row["factor_name"]
        pq = work_dir / "factor_lake" / name / "values.parquet"
        if not pq.exists():
            continue
        entry = catalog.get(name, {})
        py = parsed.get(name, {}).get("python_code", "")
        if flip.get(name, {}).get("python_code"):
            py = flip[name]["python_code"]
        dsl = (entry.get("dsl") or "").strip() or "(python only)"
        analysis = analyze_factor_parquet_duckdb(factor_path=pq, fwd_returns_path=fwd)
        render_factor_report(
            factor_name=name,
            dsl=dsl,
            analysis=analysis,
            backtest_rows=None,
            out_path=report_dir / row.get("report", f"{name}.html"),
            eval_mode=row.get("eval_mode", ""),
            python_code=py,
            work_dir=work_dir,
        )
        n += 1
        if n % 20 == 0:
            print(f"  refreshed {n} reports...")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Build factor formula/rationale/lookahead annotations")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--print-top", type=int, default=0, help="print top-N by rank ic from progress")
    parser.add_argument(
        "--refresh-reports",
        action="store_true",
        help="re-render HTML (RankIC/LS only; TopK NAV cleared unless re-backtested)",
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    data = build_all_annotations(args.work_dir)
    path = save_annotations(args.work_dir, data)
    meta = data["meta"]
    print(f"wrote {path}")
    print(
        f"  factors={meta['generated_count']} python_only={meta['python_only']} "
        f"lookahead_pending_fix={meta['lookahead_pending_fix']} high={meta['lookahead_high']}"
    )

    if args.refresh_reports:
        n = _refresh_reports(args.work_dir, args.only)
        print(f"refreshed {n} factor reports")

    if args.print_top:
        progress = json.loads((args.work_dir / "production_progress.json").read_text())
        rows = sorted(progress.get("index_rows", []), key=lambda r: r.get("mean_rank_ic", 0), reverse=True)
        for r in rows[: args.print_top]:
            ann = data["factors"].get(r["factor_name"], {})
            print(f"\n=== {r['factor_name']} ic={r.get('mean_rank_ic')} risk={ann.get('lookahead_risk')} ===")
            print(ann.get("formula_display", "")[:200])
            print(ann.get("rationale_zh", "")[:400])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
