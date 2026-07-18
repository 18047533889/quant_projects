#!/usr/bin/env python3
"""Recompute RankIC / 分层 / 多空 for factors that used broken LQTP platform analyze.

LQTP RunFactor(analyze=True) returns daily_ls / group_pnls that are not usable as
close-to-close research metrics (mean daily LS can be ~3%+, G10 cum → 1e26).
This script re-fetches factor values (analyze=False), saves parquet, DuckDB-panels,
re-renders HTML, and updates production_progress.json.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_WORKER: dict[str, Any] = {}

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import (  # noqa: E402
    _json_float,
    analyze_factor_long_df_duckdb,
    build_fwd_returns_cache,
)
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    factor_values_to_long_df,
    run_factor_formula,
)
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    _enrich_topk_summary,
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.run_light_test import _yyyymmdd  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _apply_flip_state_to_catalog,
    _save_progress,
    _upsert_index_row,
)


def _is_platform_row(row: dict[str, Any]) -> bool:
    kind = str(row.get("return_kind") or "")
    mode = str(row.get("eval_mode") or "")
    return kind == "lqtp_platform_analyze" or "lqtp_run_factor" in mode


def _init_worker(
    fwd_path: str,
    server: str,
    username: str,
    password: str,
    token_refresh_minutes: float,
) -> None:
    _WORKER.clear()
    _WORKER["fwd_path"] = fwd_path
    _WORKER["token_mgr"] = LqtpTokenManager.login(
        server,
        username,
        password,
        refresh_interval_seconds=int(token_refresh_minutes * 60),
    )
    _WORKER["server"] = server


def _repair_one(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload["name"]
    t0 = time.time()
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        work = Path(payload["work_dir"])
        token_mgr: LqtpTokenManager = _WORKER["token_mgr"]
        token_mgr.maybe_refresh()
        dsl = payload["dsl"]
        py = payload.get("python_code") or ""
        prev = payload.get("prev_row") or {}
        print(f"[{payload['index']}/{payload['total']}] {name} RunFactor…")
        resp = run_factor_formula(
            token=token_mgr.token,
            formula=dsl,
            begin_date=int(payload["begin_i"]),
            end_date=int(payload["end_i"]),
            warmup=int(payload.get("warmup", 60)),
            analyze=False,
            server=_WORKER.get("server", DEFAULT_SERVER),
            factor_name="",
        )
        if resp.error:
            raise RuntimeError(str(resp.error))
        long_df = factor_values_to_long_df(resp.values)
        if long_df.empty:
            raise RuntimeError("empty values")
        lake_dir = work / "factor_lake" / name
        lake_dir.mkdir(parents=True, exist_ok=True)
        out_pq = lake_dir / "values.parquet"
        long_df.to_parquet(out_pq, index=False)

        analysis = analyze_factor_long_df_duckdb(
            long_df,
            fwd_returns_path=Path(payload["fwd_path"]),
        )
        analysis["value_source"] = "lqtp_run_factor"
        analysis["signal_lag_note"] = (
            "Factor values from LQTP RunFactor; RankIC / 分层 / 多空 recomputed locally "
            "(DuckDB close-to-close). Platform analyze LS/groups discarded."
        )

        topk_summary = _enrich_topk_summary(
            {
                "total_return": prev.get("backtest_total_ret"),
                "annualized_return": prev.get("backtest_ann_ret"),
                "max_drawdown": prev.get("backtest_max_drawdown"),
                "sharpe": prev.get("backtest_sharpe"),
                "volatility": prev.get("backtest_volatility"),
                "calmar": prev.get("backtest_calmar"),
                "win_rate": prev.get("backtest_win_rate"),
                "avg_turnover": prev.get("backtest_avg_turnover"),
                "total_commission": prev.get("backtest_total_commission"),
                "trading_days": prev.get("backtest_trading_days"),
                "final_nav": prev.get("backtest_final_nav"),
            }
        )
        eval_mode = "lqtp_values_duckdb_panel"
        bt = prev.get("backtest_sharpe")
        if bt is not None and not (isinstance(bt, float) and math.isnan(bt)):
            eval_mode = f"{eval_mode}+topk_open"

        report_path = work / "reports" / f"{name}.html"
        render_factor_report(
            factor_name=name,
            dsl=dsl,
            analysis=analysis,
            backtest_rows=[],
            out_path=report_path,
            eval_mode=eval_mode,
            materialize_meta={
                "engine": "lqtp_dsl",
                "eval_route": "lqtp_dsl",
                "rows": int(len(long_df)),
                "values_path": str(out_pq),
                "date_range": [payload["start"], payload["end"]],
                "note": "panel repaired: DuckDB close-to-close (platform LS discarded)",
            },
            topk_summary=topk_summary,
            python_code=py,
            work_dir=work,
            engine="lqtp_dsl",
            eval_route="lqtp_dsl",
        )

        row = dict(prev)
        row.update(
            {
                "report": report_path.name,
                "eval_mode": eval_mode,
                "mean_ic": _json_float(analysis.get("mean_rank_ic")),
                "mean_rank_ic": _json_float(analysis.get("mean_rank_ic")),
                "icir": _json_float(analysis.get("rank_icir")),
                "rank_icir": _json_float(analysis.get("rank_icir")),
                "rank_ic_positive_ratio": _json_float(analysis.get("rank_ic_positive_ratio")),
                "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
                "long_short_return": _json_float(analysis.get("long_short_return")),
                "rows": int(len(long_df)),
                "return_kind": str(analysis.get("return_kind", "")),
            }
        )
        elapsed = time.time() - t0
        print(
            f"  ok {name} rank_ic={row['mean_rank_ic']:.4f} "
            f"ls_cum={row['long_short_return']:.3f} "
            f"ls_sh={row['long_short_sharpe']:.3f} {elapsed:.1f}s"
        )
        return {"ok": True, "name": name, "row": row, "elapsed": elapsed}
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name}: {exc}")
        return {"ok": False, "name": name, "error": str(exc), "elapsed": time.time() - t0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair LQTP platform LS/decile charts")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--min-rank-ic",
        type=float,
        default=None,
        help="only factors with |mean_rank_ic| >= threshold (zero-IC broken DSL always included)",
    )
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument(
        "--token-refresh-minutes",
        type=float,
        default=20.0,
        help="refresh LQTP access token every N minutes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="parallel LQTP RunFactor workers (default 3)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    work = args.work_dir
    progress_path = work / "production_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    catalog = json.loads((work / "dsl_catalog.json").read_text(encoding="utf-8"))
    parsed = {r["function_name"]: r.get("python_code", "") for r in json.loads((work / "parsed_factors.json").read_text())}
    flip = progress.get("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip, parsed)
    entry_by = {e["function_name"]: e for e in catalog}

    fwd = build_fwd_returns_cache(
        work / "lqtp_close_returns_cache.parquet",
        work / "lqtp_fwd_close_returns_cache.parquet",
    )
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)

    def _rank_ic(row: dict[str, Any]) -> float:
        try:
            return float(row.get("mean_rank_ic", row.get("mean_ic", float("-inf"))))
        except (TypeError, ValueError):
            return float("-inf")

    targets = []
    for row in progress.get("index_rows", []):
        name = row["factor_name"]
        if args.only and name not in set(args.only):
            continue
        if not _is_platform_row(row):
            continue
        entry = entry_by.get(name)
        if not entry or not (entry.get("dsl") or "").strip():
            continue
        ic = _rank_ic(row)
        if args.min_rank_ic is not None and abs(ic) < float(args.min_rank_ic) and abs(ic) >= 1e-9:
            continue
        targets.append(name)

    targets.sort(key=lambda n: abs(_rank_ic(next(r for r in progress["index_rows"] if r["factor_name"] == n))), reverse=True)

    print(
        f"repair platform panel: {len(targets)} factors "
        f"min_rank_ic={args.min_rank_ic} workers={args.workers} order=rankic_desc"
    )
    if targets[:3]:
        top = ", ".join(
            f"{n}({ _rank_ic(next(r for r in progress['index_rows'] if r['factor_name']==n)):.4f})"
            for n in targets[:3]
        )
        print(f"  top: {top}")

    index_rows = list(progress.get("index_rows", []))
    index_by_name = {r["factor_name"]: r for r in index_rows}
    payloads: list[dict[str, Any]] = []
    for i, name in enumerate(targets, 1):
        entry = entry_by[name]
        dsl = (entry.get("dsl") or "").strip()
        if flip.get(name, {}).get("dsl"):
            dsl = flip[name]["dsl"]
        py = parsed.get(name, "")
        if flip.get(name, {}).get("python_code"):
            py = flip[name]["python_code"]
        payloads.append(
            {
                "index": i,
                "total": len(targets),
                "name": name,
                "work_dir": str(work),
                "dsl": dsl,
                "python_code": py,
                "prev_row": index_by_name.get(name, {}),
                "fwd_path": str(fwd),
                "begin_i": begin_i,
                "end_i": end_i,
                "start": args.start,
                "end": args.end,
                "warmup": args.warmup,
            }
        )

    fixed = 0
    failed: dict[str, str] = {}
    t0_batch = time.time()
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(
            str(fwd),
            args.server,
            args.username,
            args.password,
            args.token_refresh_minutes,
        ),
    ) as pool:
        futures = {pool.submit(_repair_one, p): p["name"] for p in payloads}
        done_n = 0
        for fut in as_completed(futures):
            result = fut.result()
            name = result["name"]
            done_n += 1
            if result.get("ok"):
                disk = json.loads(progress_path.read_text(encoding="utf-8"))
                disk_rows = list(disk.get("index_rows", []))
                _upsert_index_row(disk_rows, result["row"])
                disk["index_rows"] = disk_rows
                disk["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save_progress(progress_path, disk)
                index_rows = disk_rows
                fixed += 1
            else:
                failed[name] = result.get("error") or "unknown"
            if done_n % 3 == 0 or done_n == len(payloads):
                render_index(index_rows, work / "reports" / "index.html")
                render_production_summary(
                    rows=index_rows,
                    out_path=work / "reports" / "production_summary.html",
                    meta={
                        "catalog_total": 176,
                        "date_range": [args.start, args.end],
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "note": f"LQTP panel repair in progress ({done_n}/{len(payloads)})",
                    },
                    failed=failed,
                )
                print(f"  checkpoint {done_n}/{len(payloads)} fixed={fixed}")
    render_index(index_rows, work / "reports" / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=work / "reports" / "production_summary.html",
        meta={
            "catalog_total": 176,
            "date_range": [args.start, args.end],
            "note": "Repaired LQTP platform panel → DuckDB close-to-close",
        },
    )
    print(
        f"done fixed={fixed}/{len(targets)} failed={len(failed)} "
        f"elapsed={time.time()-t0_batch:.1f}s"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
