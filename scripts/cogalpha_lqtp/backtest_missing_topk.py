#!/usr/bin/env python3
"""Run LQTP TopK backtest for factors that only have RankIC (no TopK yet)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.eval_lake_fast import (  # noqa: E402
    _eval_route,
    _json_float,
    analyze_factor_parquet_duckdb,
    build_fwd_returns_cache,
)
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    factor_values_to_long_df,
    fetch_lqtp_universe,
    long_df_to_daily_values,
    summarize_backtest,
    top_quantile_weights,
)
from scripts.cogalpha_lqtp.regenerate_reports_fast import _normalize_factor_long  # noqa: E402
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    LOOKAHEAD_DEFERRED_FACTORS,
    _apply_flip_state_to_catalog,
    _safe_backtest,
    _save_progress,
    _upsert_index_row,
)
from scripts.cogalpha_lqtp.run_light_test import _yyyymmdd  # noqa: E402


def _needs_backtest(row: dict[str, Any]) -> bool:
    bt = row.get("backtest_sharpe")
    if bt is None:
        return True
    try:
        return math.isnan(float(bt))
    except (TypeError, ValueError):
        return True


_WORKER: dict[str, Any] = {}


def _init_worker(fwd_path: str, server: str, username: str, password: str) -> None:
    from scripts.cogalpha_lqtp.eval_lake_fast import _init_worker as _init_fwd

    _init_fwd(fwd_path)
    auth = LqtpTokenManager.login(server, username, password)
    _WORKER["token"] = auth.token
    _WORKER["server"] = server


def _backtest_one(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload["name"]
    t0 = time.time()
    try:
        work = Path(payload["work_dir"])
        lake = work / "factor_lake"
        report_dir = work / "reports"
        out_path = lake / name / "values.parquet"
        if not out_path.exists():
            raise RuntimeError(f"missing values: {out_path}")

        entry = dict(payload["entry"])
        py_code = payload.get("python_code") or ""
        prev = payload.get("prev_row") or {}

        dsl_used = (entry.get("dsl") or "").strip() or "(python only)"
        route = _eval_route(entry)
        engine = prev.get("engine") or ("python" if route == "local_python" else "factor_engine")
        if route == "lqtp_dsl":
            engine = "lqtp_dsl"

        fwd = Path(payload["fwd_path"])
        analysis = analyze_factor_parquet_duckdb(
            factor_path=out_path,
            fwd_returns_path=fwd,
            con=_WORKER.get("con"),
        )

        long_df = _normalize_factor_long(pd.read_parquet(out_path))
        daily_values = long_df_to_daily_values(long_df)
        lqtp_long = factor_values_to_long_df(daily_values)
        allowed = set(payload.get("backtest_symbols") or [])
        if allowed:
            lqtp_long = lqtp_long[lqtp_long["symbol"].isin(allowed)]
        weights = top_quantile_weights(lqtp_long, allowed_symbols=allowed or None)
        backtest_rows, backtest_id = _safe_backtest(
            token=_WORKER["token"],
            weights=weights,
            begin_date=int(payload["begin_i"]),
            end_date=int(payload["end_i"]),
            server=_WORKER.get("server", DEFAULT_SERVER),
            max_quote_retries=150,
        )
        topk_summary = summarize_backtest(backtest_rows)

        eval_mode = prev.get("eval_mode") or "duckdb_panel_close_to_close"
        if backtest_rows:
            eval_mode = f"{eval_mode}+topk_open"

        meta = {
            "engine": engine,
            "eval_route": route,
            "rows": int(len(long_df)),
            "values_path": str(out_path),
            "date_range": [payload["start"], payload["end"]],
            "formula": dsl_used if dsl_used != "(python only)" else "",
            "eval_engine": "duckdb_panel+lqtp_topk",
        }

        report_path = report_dir / f"{name}.html"
        render_factor_report(
            factor_name=name,
            dsl=dsl_used,
            analysis=analysis,
            backtest_rows=backtest_rows,
            out_path=report_path,
            eval_mode=eval_mode,
            materialize_meta=meta,
            topk_summary=topk_summary,
            python_code=py_code,
            work_dir=work,
        )

        row = {
            "factor_name": name,
            "report": report_path.name,
            "engine": engine,
            "eval_route": route,
            "eval_mode": eval_mode,
            "ic_sign_flipped": bool(entry.get("ic_sign_flipped") or prev.get("ic_sign_flipped")),
            "mean_ic": _json_float(analysis.get("mean_rank_ic")),
            "mean_rank_ic": _json_float(analysis.get("mean_rank_ic")),
            "icir": _json_float(analysis.get("rank_icir")),
            "rank_icir": _json_float(analysis.get("rank_icir")),
            "rank_ic_positive_ratio": _json_float(analysis.get("rank_ic_positive_ratio")),
            "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
            "long_short_return": _json_float(analysis.get("long_short_return")),
            "backtest_total_ret": _json_float(topk_summary.get("total_return")),
            "backtest_sharpe": _json_float(topk_summary.get("sharpe")),
            "backtest_max_drawdown": _json_float(topk_summary.get("max_drawdown")),
            "rows": int(len(long_df)),
            "backtest_id": backtest_id,
            "return_kind": str(analysis.get("return_kind", "")),
        }
        dt = time.time() - t0
        bt_flag = "yes" if backtest_rows else "skip"
        bt_sh = row["backtest_sharpe"]
        bt_sh_s = f"{bt_sh:.3f}" if bt_sh == bt_sh else "nan"
        print(
            f"ok {name} rank_ic={row['mean_rank_ic']:.4f} "
            f"topk_sharpe={bt_sh_s} bt={bt_flag} {dt:.1f}s"
        )
        return {"ok": True, "name": name, "row": row, "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
        return {"ok": False, "name": name, "row": None, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="LQTP TopK backtest for factors missing TopK")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    work = args.work_dir
    progress_path = work / "production_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    catalog = json.loads((work / "dsl_catalog.json").read_text(encoding="utf-8"))
    parsed = json.loads((work / "parsed_factors.json").read_text(encoding="utf-8"))
    python_map = {r["function_name"]: r["python_code"] for r in parsed}
    flip_state = progress.setdefault("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip_state, python_map)
    entry_by_name = {e["function_name"]: e for e in catalog}

    skipped = set(progress.get("skipped", [])) | set(LOOKAHEAD_DEFERRED_FACTORS)
    index_by_name = {r["factor_name"]: r for r in progress.get("index_rows", [])}

    targets: list[str] = []
    for name, row in index_by_name.items():
        if name in skipped:
            continue
        if not (work / "factor_lake" / name / "values.parquet").exists():
            continue
        if _needs_backtest(row):
            targets.append(name)
    if args.only:
        only = set(args.only)
        targets = [n for n in targets if n in only]
    targets.sort()

    if not targets:
        print("no factors need TopK backtest")
        return 0

    auth = LqtpTokenManager.login(args.server, args.username, args.password)
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    backtest_symbols = sorted(
        fetch_lqtp_universe(token=auth.token, begin_date=begin_i, end_date=end_i, server=args.server)
    )
    fwd = build_fwd_returns_cache(
        work / "lqtp_close_returns_cache.parquet",
        work / "lqtp_fwd_close_returns_cache.parquet",
    )
    print(f"LQTP ok universe={len(backtest_symbols)} targets={len(targets)} workers={args.workers}")

    payloads = []
    for name in targets:
        entry = entry_by_name.get(name)
        if entry is None:
            continue
        py = python_map.get(name, "")
        if flip_state.get(name, {}).get("python_code"):
            py = flip_state[name]["python_code"]
        payloads.append(
            {
                "name": name,
                "work_dir": str(work),
                "entry": entry,
                "python_code": py,
                "prev_row": index_by_name.get(name, {}),
                "fwd_path": str(fwd),
                "begin_i": begin_i,
                "end_i": end_i,
                "start": args.start,
                "end": args.end,
                "backtest_symbols": backtest_symbols,
            }
        )

    index_rows: list[dict[str, Any]] = list(progress.get("index_rows", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))
    completed: list[str] = list(progress.get("completed", []))
    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(str(fwd), args.server, args.username, args.password),
    ) as pool:
        futures = {pool.submit(_backtest_one, p): p["name"] for p in payloads}
        done_n = 0
        for fut in as_completed(futures):
            result = fut.result()
            name = result["name"]
            done_n += 1
            if result["ok"]:
                _upsert_index_row(index_rows, result["row"])
                if name not in completed:
                    completed.append(name)
                failed.pop(name, None)
            else:
                failed[name] = result.get("error") or "unknown"
            if done_n % 5 == 0 or done_n == len(payloads):
                progress["completed"] = completed
                progress["failed"] = failed
                progress["index_rows"] = index_rows
                progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save_progress(progress_path, progress)
                render_index(index_rows, work / "reports" / "index.html")
                render_production_summary(
                    rows=index_rows,
                    out_path=work / "reports" / "production_summary.html",
                    meta={
                        "catalog_total": 176,
                        "date_range": [args.start, args.end],
                        "generated_at": progress["regen_at"],
                        "note": "TopK backtest batch in progress",
                    },
                    failed=failed,
                )
                print(f"  checkpoint {done_n}/{len(payloads)}")

    progress["completed"] = completed
    progress["failed"] = failed
    progress["index_rows"] = index_rows
    progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_progress(progress_path, progress)
    render_index(index_rows, work / "reports" / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=work / "reports" / "production_summary.html",
        meta={
            "catalog_total": 176,
            "date_range": [args.start, args.end],
            "generated_at": progress["regen_at"],
            "note": "TopK OPEN backtest via LQTP",
            "elapsed_sec": round(time.time() - t0, 1),
        },
        failed=failed,
    )
    ok_bt = sum(
        1
        for r in index_rows
        if r.get("backtest_sharpe") == r.get("backtest_sharpe")
        and not (isinstance(r.get("backtest_sharpe"), float) and math.isnan(r.get("backtest_sharpe")))
    )
    print(f"done targets={len(payloads)} with_topk={ok_bt} failed={len(failed)} elapsed={time.time()-t0:.1f}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
