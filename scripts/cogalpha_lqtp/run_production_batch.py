#!/usr/bin/env python3
"""Production batch: 2019-2026 materialize (DSL + Python) → LQTP eval → HTML reports.

Memory model (~30G RAM, no swap):
  - one factor at a time
  - explicit gc + data_access store reset between factors
  - symbol-chunked Python fallback
  - resume via production_progress.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
FE_ROOT = ROOT / "factor_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FE_ROOT) not in sys.path:
    sys.path.insert(0, str(FE_ROOT))

from api.mining_integration import default_ashare_pv_data_source_config, validate_factor_engine_dsl  # noqa: E402

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
from scripts.cogalpha_lqtp.materialize import materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import ensure_memory_floor, read_mem_available_gb, release_memory  # noqa: E402
from scripts.cogalpha_lqtp.python_materialize import materialize_python_factor  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report, render_index  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import _run_py, _yyyymmdd  # noqa: E402


def _ashare_data_source(start: str, end: str) -> dict[str, Any]:
    return default_ashare_pv_data_source_config(start_date=start, end_date=end)


def _count_bar_files(data_root: Path) -> int:
    bar = data_root / "StockDailyBar"
    if not bar.is_dir():
        return 0
    return sum(1 for _ in bar.glob("*.parquet"))


def _has_market_data(data_root: Path, *, min_files: int = 20) -> bool:
    return _count_bar_files(data_root) >= min_files


def _load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "failed": {}, "skipped": []}


def _save_progress(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _route(entry: dict[str, Any]) -> str:
    st = entry.get("status", "")
    dsl = entry.get("dsl", "")
    if st == "hard" or not dsl:
        return "python"
    if st == "ready":
        return "factor_engine"
    return "factor_engine_then_python"


def _materialize_entry(
    *,
    entry: dict[str, Any],
    python_code: str,
    data_cfg: dict[str, Any],
    lake_root: Path,
    start: str,
    end: str,
    symbol_chunk: int,
) -> tuple[Path, str, str]:
    name = entry["function_name"]
    dsl = entry.get("dsl", "")
    route = _route(entry)

    if route == "python":
        out = materialize_python_factor(
            function_name=name,
            python_code=python_code,
            start=start,
            end=end,
            lake_root=lake_root,
            symbol_chunk=symbol_chunk,
        )
        return out, "python", dsl or "(python only)"

    try:
        if route == "factor_engine_then_python":
            ok, msg = validate_factor_engine_dsl(dsl)
            if not ok:
                raise RuntimeError(msg)
        out = materialize_factor(
            factor_id=name,
            dsl=dsl,
            data_source_cfg=data_cfg,
            lake_root=lake_root,
        )
        return out, "factor_engine", dsl
    except Exception:
        if route != "factor_engine_then_python":
            raise
        out = materialize_python_factor(
            function_name=name,
            python_code=python_code,
            start=start,
            end=end,
            lake_root=lake_root,
            symbol_chunk=symbol_chunk,
        )
        return out, "python_fallback", dsl


def main() -> int:
    parser = argparse.ArgumentParser(description="Production CogAlpha batch 2019-2026")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--bootstrap-lqtp", action="store_true", help="fetch StockDailyBar from LQTP if missing")
    parser.add_argument("--symbol-chunk", type=int, default=300)
    parser.add_argument("--min-mem-gb", type=float, default=4.0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    parsed = work / "parsed_factors.json"
    catalog_path = work / "dsl_catalog.json"
    lake = work / "factor_lake"
    report_dir = work / "reports"
    progress_path = work / "production_progress.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    lake.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root or Path(
        os.getenv("ASHARE_PARQUET_ROOT", str(ROOT / "data/a_share/lqtp_data"))
    )

    _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(parsed))
    _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog_path))

    if args.bootstrap_lqtp or not _has_market_data(data_root):
        print("bootstrapping StockDailyBar from LQTP ...")
        _run_py(
            SCRIPTS / "bootstrap_lqtp_market_data.py",
            "--out-root",
            str(data_root),
            "--start",
            args.start,
            "--end",
            args.end,
            "--server",
            args.server,
        )

    if not _has_market_data(data_root):
        raise SystemExit(
            f"StockDailyBar missing under {data_root}; "
            "install clean-cos-ro or run with --bootstrap-lqtp"
        )

    catalog: list[dict[str, Any]] = json.loads(catalog_path.read_text(encoding="utf-8"))
    python_map = {r["function_name"]: r["python_code"] for r in json.loads(parsed.read_text(encoding="utf-8"))}
    if args.only:
        only = set(args.only)
        catalog = [x for x in catalog if x["function_name"] in only]

    progress = _load_progress(progress_path) if args.resume else {"completed": [], "failed": {}, "skipped": []}
    done = set(progress.get("completed", []))

    auth = login(args.server, args.username, args.password)
    token = auth.access_token
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    lqtp_universe = fetch_lqtp_universe(token=token, begin_date=begin_i, end_date=end_i, server=args.server)
    print(f"LQTP universe={len(lqtp_universe)} mem_avail={read_mem_available_gb():.1f}G")

    data_cfg = _ashare_data_source(args.start, args.end)
    index_rows: list[dict[str, str]] = []
    t0 = time.time()

    for i, entry in enumerate(catalog, 1):
        name = entry["function_name"]
        if name in done:
            print(f"[{i}/{len(catalog)}] skip done {name}")
            continue

        ensure_memory_floor(args.min_mem_gb)
        print(f"[{i}/{len(catalog)}] {name} route={_route(entry)} mem={read_mem_available_gb():.1f}G")

        try:
            py_code = python_map.get(name, "")
            if _route(entry) == "python" and not py_code:
                raise RuntimeError("missing python_code")

            out_path, engine, dsl_used = _materialize_entry(
                entry=entry,
                python_code=py_code,
                data_cfg=data_cfg,
                lake_root=lake,
                start=args.start,
                end=args.end,
                symbol_chunk=args.symbol_chunk,
            )
            rows = int(pd.read_parquet(out_path, columns=["value"]).shape[0])
            meta = {
                "engine": engine,
                "rows": rows,
                "values_path": str(out_path),
                "date_range": [args.start, args.end],
            }

            eval_mode = "skipped"
            analysis: dict[str, Any] = {}
            backtest_rows: list[dict[str, Any]] = []
            backtest_id = ""

            if not args.skip_eval:
                long_df = pd.read_parquet(out_path)
                daily_values = long_df_to_daily_values(long_df)
                release_memory(long_df)
                analysis, eval_mode = evaluate_factor(
                    token=token,
                    daily_values=daily_values,
                    begin_date=begin_i,
                    end_date=end_i,
                    server=args.server,
                    returns_cache=work / "lqtp_returns_cache.parquet",
                )
                weights = top_quantile_weights(
                    factor_values_to_long_df(daily_values),
                    allowed_symbols=lqtp_universe,
                )
                release_memory(daily_values)
                backtest_rows, backtest_id = run_backtest_from_weights(
                    token=token,
                    weights=weights,
                    begin_date=begin_i,
                    end_date=end_i,
                    server=args.server,
                )
                release_memory(weights)

            report_path = report_dir / f"{name}.html"
            render_factor_report(
                factor_name=name,
                dsl=dsl_used,
                analysis=analysis,
                backtest_rows=backtest_rows,
                out_path=report_path,
                eval_mode=eval_mode,
                materialize_meta=meta,
            )
            progress.setdefault("completed", []).append(name)
            progress.get("failed", {}).pop(name, None)
            _save_progress(progress_path, progress)

            index_rows.append(
                {
                    "factor_name": name,
                    "report": report_path.name,
                    "engine": engine,
                    "eval_mode": eval_mode,
                    "mean_ic": f"{analysis.get('mean_ic', float('nan')):.6f}",
                    "icir": f"{analysis.get('icir', float('nan')):.6f}",
                    "rows": str(rows),
                    "backtest_id": backtest_id,
                }
            )
            print(f"  ok rows={rows} ic={analysis.get('mean_ic', 'n/a')} mode={eval_mode}")
        except Exception as exc:  # noqa: BLE001
            progress.setdefault("failed", {})[name] = str(exc)
            _save_progress(progress_path, progress)
            print(f"  FAIL {name}: {exc}")
            traceback.print_exc()
        finally:
            release_memory()

    render_index(index_rows, report_dir / "index.html")
    summary = {
        "work_dir": str(work),
        "date_range": [args.start, args.end],
        "elapsed_sec": time.time() - t0,
        "completed": progress.get("completed", []),
        "failed": progress.get("failed", {}),
        "reports_index": str(report_dir / "index.html"),
        "latest": index_rows,
    }
    (work / "production_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": len(progress.get("completed", [])), "fail": len(progress.get("failed", {}))}, ensure_ascii=False))
    return 0 if not progress.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
