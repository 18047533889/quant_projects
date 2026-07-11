#!/usr/bin/env python3
"""Light end-to-end test: parse → DSL → materialize → LQTP analyze/backtest → HTML."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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

DEFAULT_FACTORS = [
    "factor_persistence",
    "factor_persistence_ewma",
    "factor_price_impact_stable_5d",
]


def _run_py(script: Path, *args: str) -> None:
    cmd = [sys.executable, str(script), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _yyyymmdd(date_str: str) -> int:
    return int(date_str.replace("-", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="CogAlpha LQTP light test")
    parser.add_argument(
        "--factors-md",
        type=Path,
        default=ROOT / "factors(1).md",
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_smoke")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-03-29")
    parser.add_argument("--only", nargs="*", default=DEFAULT_FACTORS)
    args = parser.parse_args()

    work = args.work_dir
    parsed = work / "parsed_factors.json"
    catalog = work / "dsl_catalog.json"
    smoke_dir = work / "smoke_StockDailyBar"
    lake = work / "factor_lake"
    report_dir = work / "reports"

    work.mkdir(parents=True, exist_ok=True)

    _run_py(SCRIPTS / "parse_factors_md.py", str(args.factors_md), "--out", str(parsed))
    _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog))
    _run_py(
        SCRIPTS / "smoke_data.py",
        "--out-dir",
        str(smoke_dir),
        "--start",
        args.start,
        "--end",
        args.end,
    )
    _run_py(
        SCRIPTS / "materialize.py",
        "--catalog",
        str(catalog),
        "--data-root",
        str(smoke_dir),
        "--lake-root",
        str(lake),
        "--start",
        args.start,
        "--end",
        args.end,
        "--only",
        *args.only,
    )

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
    print(f"LQTP universe size: {len(lqtp_universe)}")

    manifest = json.loads((lake / "materialize_manifest.json").read_text(encoding="utf-8"))
    index_rows: list[dict[str, str]] = []

    for item in manifest:
        factor_name = item["function_name"]
        dsl = item["dsl"]
        values_path = Path(item["values_path"])
        long_df = pd.read_parquet(values_path)
        daily_values = long_df_to_daily_values(long_df)

        print(f"LQTP evaluate (uploaded values): {factor_name} ({len(daily_values)} days)")
        analysis, eval_mode = evaluate_factor(
            token=token,
            daily_values=daily_values,
            begin_date=begin_i,
            end_date=end_i,
            server=args.server,
        )
        print(f"LQTP eval mode: {eval_mode}")

        lqtp_long = factor_values_to_long_df(daily_values)
        weights = top_quantile_weights(
            lqtp_long,
            allowed_symbols=lqtp_universe,
        )
        backtest_rows, backtest_id = run_backtest_from_weights(
            token=token,
            weights=weights,
            begin_date=begin_i,
            end_date=end_i,
            server=args.server,
        )
        print(f"LQTP backtest: {factor_name} backtest_id={backtest_id} days={len(backtest_rows)}")

        report_path = report_dir / f"{factor_name}.html"
        render_factor_report(
            factor_name=factor_name,
            dsl=dsl,
            analysis=analysis,
            backtest_rows=backtest_rows,
            out_path=report_path,
            eval_mode=eval_mode,
            materialize_meta={
                "rows": item.get("rows"),
                "values_path": item["values_path"],
                "engine": "factor_engine",
            },
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

    render_index(index_rows, report_dir / "index.html")
    summary = {
        "server": args.server,
        "factors": index_rows,
        "report_index": str(report_dir / "index.html"),
    }
    (work / "light_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
