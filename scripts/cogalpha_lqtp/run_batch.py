#!/usr/bin/env python3
"""Batch runner for full CogAlpha catalog: validate DSL, materialize, LQTP evaluate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import validate_factor_engine_dsl  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import _run_py, _yyyymmdd  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    evaluate_factor,
    factor_values_to_long_df,
    login,
    long_df_to_daily_values,
    run_backtest_from_weights,
    top_quantile_weights,
)
from scripts.cogalpha_lqtp.report_html import render_factor_report, render_index  # noqa: E402

import pandas as pd


def validate_catalog(catalog_path: Path) -> dict[str, int]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    stats = {"ready": 0, "python": 0, "dsl_invalid": 0}
    for item in catalog:
        status = item.get("status", "python")
        if status == "ready":
            stats["ready"] += 1
        else:
            stats["python"] += 1
        dsl = item.get("dsl", "")
        if dsl:
            ok, _ = validate_factor_engine_dsl(dsl)
            if not ok:
                stats["dsl_invalid"] += 1
                item["status"] = "python"
                item["dsl"] = ""
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch CogAlpha LQTP pipeline")
    parser.add_argument("--factors-md", type=Path, default=ROOT / "factors(1).md")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_batch")
    parser.add_argument("--data-root", type=Path, default=None, help="A-share StockDailyBar root")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--status", default="ready", help="catalog status filter")
    args = parser.parse_args()

    work = args.work_dir
    parsed = work / "parsed_factors.json"
    catalog = work / "dsl_catalog.json"
    lake = work / "factor_lake"
    report_dir = work / "reports"
    work.mkdir(parents=True, exist_ok=True)

    _run_py(SCRIPTS / "parse_factors_md.py", str(args.factors_md), "--out", str(parsed))
    _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog))
    stats = validate_catalog(catalog)
    print("catalog stats:", stats)

    if args.validate_only:
        return 0

    data_root = args.data_root
    if data_root is None:
        smoke_dir = work / "smoke_StockDailyBar"
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

    _run_py(
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
        "--status",
        args.status,
    )

    auth = login(args.server, args.username, args.password)
    token = auth.access_token
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)

    manifest = json.loads((lake / "materialize_manifest.json").read_text(encoding="utf-8"))
    index_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for item in manifest:
        factor_name = item["function_name"]
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
            lqtp_long = factor_values_to_long_df(daily_values)
            weights = top_quantile_weights(lqtp_long)
            backtest_rows, _ = run_backtest_from_weights(
                token=token,
                weights=weights,
                begin_date=begin_i,
                end_date=end_i,
                server=args.server,
            )
            report_path = report_dir / f"{factor_name}.html"
            render_factor_report(
                factor_name=factor_name,
                dsl=item["dsl"],
                analysis=analysis,
                backtest_rows=backtest_rows,
                out_path=report_path,
                eval_mode=eval_mode,
                materialize_meta={"rows": item.get("rows"), "engine": "factor_engine"},
            )
            index_rows.append(
                {
                    "factor_name": factor_name,
                    "report": report_path.name,
                    "eval_mode": eval_mode,
                    "mean_ic": f"{analysis.get('mean_ic', float('nan')):.6f}",
                    "icir": f"{analysis.get('icir', float('nan')):.6f}",
                }
            )
            print(f"ok {factor_name}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"factor_name": factor_name, "error": str(exc)})
            print(f"fail {factor_name}: {exc}")

    render_index(index_rows, report_dir / "index.html")
    summary = {"ok": index_rows, "failures": failures, "catalog_stats": stats}
    (work / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": len(index_rows), "fail": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
