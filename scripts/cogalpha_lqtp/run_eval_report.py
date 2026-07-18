#!/usr/bin/env python3
"""End-to-end: factor_engine DSL → local materialize → LQTP evaluate → HTML reports.

Architecture:
  1. Parse CogAlpha Python factors → factor_engine DSL (our operator naming)
  2. Materialize with factor_engine locally (never LQTP RunFactor for our factors)
  3. Upload daily_values to LQTP for IC / group / coverage analysis
  4. Build top-quantile weights from uploaded values → LQTP Backtest
  5. Generate per-factor HTML reports with all FactorAnalysis + Backtest fields
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    evaluate_factor,
    factor_values_to_long_df,
    fetch_lqtp_universe,
    login,
    long_df_to_daily_values,
    run_backtest_from_weights,
    top_quantile_weights,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report, render_index  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import _run_py, _yyyymmdd  # noqa: E402


def _load_smoke_pass(work_dir: Path) -> list[str] | None:
    report = work_dir / "dsl_smoke_test_report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    return data.get("passed") or None


def main() -> int:
    parser = argparse.ArgumentParser(description="factor_engine materialize → LQTP evaluate → reports")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_batch")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--only", nargs="*", help="explicit factor function_name subset")
    parser.add_argument("--smoke-pass-only", action="store_true", help="use dsl_smoke_test_report passed list")
    parser.add_argument("--limit", type=int, default=0, help="max factors (0 = all)")
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="abort unless data/cogalpha_lqtp_preflight/preflight_report.json all_passed=true",
    )
    args = parser.parse_args()

    if args.require_preflight:
        pf = ROOT / "data/cogalpha_lqtp_preflight/preflight_report.json"
        if not pf.exists():
            raise SystemExit(f"preflight report missing: {pf}; run run_preflight.py first")
        pf_data = json.loads(pf.read_text(encoding="utf-8"))
        if not pf_data.get("all_passed"):
            raise SystemExit(
                f"preflight not passed: failed={pf_data.get('failed')}; fix factors before bulk run"
            )

    work = args.work_dir
    catalog = work / "dsl_catalog.json"
    lake = work / "factor_lake"
    report_dir = work / "reports"
    work.mkdir(parents=True, exist_ok=True)

    if not catalog.exists():
        parsed = work / "parsed_factors.json"
        _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(parsed))
        _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog))

    only = list(args.only) if args.only else None
    if only is None and args.smoke_pass_only:
        only = _load_smoke_pass(work)
        if not only:
            raise SystemExit("no smoke_pass list; run run_dsl_test.py first")

    if not args.skip_materialize:
        data_root = args.data_root
        if data_root is None:
            smoke_dir = work / "smoke_StockDailyBar"
            if not smoke_dir.exists():
                _run_py(
                    SCRIPTS / "smoke_data.py",
                    "--out-dir",
                    str(smoke_dir),
                    "--start",
                    args.start,
                    "--end",
                    args.end,
                    "--n-symbols",
                    "300",
                )
            data_root = smoke_dir

        mat_args = [
            SCRIPTS / "materialize.py",
            "--catalog",
            str(catalog),
            "--data-root",
            str(data_root),
            "--lake-root",
            str(lake),
            "--start",
            args.start,
            "--end",
            args.end,
        ]
        if only:
            mat_args.extend(["--only", *only])
        else:
            mat_args.extend(["--status", "ready"])
        _run_py(*mat_args)

    manifest = json.loads((lake / "materialize_manifest.json").read_text(encoding="utf-8"))
    if only:
        only_set = set(only)
        manifest = [x for x in manifest if x["function_name"] in only_set]
    if args.limit > 0:
        manifest = manifest[: args.limit]

    auth = login(args.server, args.username, args.password)
    token = auth.access_token
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    lqtp_universe = fetch_lqtp_universe(
        token=token,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
    )

    index_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for item in manifest:
        name = item["function_name"]
        try:
            long_df = pd.read_parquet(item["values_path"])
            daily_values = long_df_to_daily_values(long_df)
            analysis, eval_mode = evaluate_factor(
                token=token,
                daily_values=daily_values,
                begin_date=begin_i,
                end_date=end_i,
                server=args.server,
            )
            weights = top_quantile_weights(
                factor_values_to_long_df(daily_values),
                allowed_symbols=lqtp_universe,
            )
            backtest_rows, backtest_id = run_backtest_from_weights(
                token=token,
                weights=weights,
                begin_date=begin_i,
                end_date=end_i,
                server=args.server,
            )
            report_path = report_dir / f"{name}.html"
            render_factor_report(
                factor_name=name,
                dsl=item["dsl"],
                analysis=analysis,
                backtest_rows=backtest_rows,
                out_path=report_path,
                eval_mode=eval_mode,
                materialize_meta={
                    "engine": "factor_engine",
                    "rows": item.get("rows"),
                    "values_path": item["values_path"],
                },
            )
            index_rows.append(
                {
                    "factor_name": name,
                    "report": report_path.name,
                    "eval_mode": eval_mode,
                    "mean_ic": f"{analysis.get('mean_ic', float('nan')):.6f}",
                    "icir": f"{analysis.get('icir', float('nan')):.6f}",
                    "backtest_id": backtest_id,
                }
            )
            print(f"ok {name} mode={eval_mode} ic={analysis.get('mean_ic'):.4f}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"factor_name": name, "error": str(exc)})
            print(f"fail {name}: {exc}")

    render_index(index_rows, report_dir / "index.html")
    summary = {
        "pipeline": "factor_engine_materialize → lqtp_evaluate → html_report",
        "eval_note": "DSL uses factor_engine operators; LQTP never re-computes our factor formula.",
        "ok": index_rows,
        "failures": failures,
    }
    (work / "eval_report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": len(index_rows), "fail": len(failures), "index": str(report_dir / "index.html")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
