#!/usr/bin/env python3
"""Fast report regeneration from existing factor_lake parquet.

Uses cached close-to-close returns (no LQTP login required for RankIC/groups/LS).
Optionally runs TopK backtest when LQTP is reachable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.factor_eval import (  # noqa: E402
    analyze_values_with_lqtp_returns,
    fetch_lqtp_close_returns,
)
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    _to_lqtp_symbol,
    factor_values_to_long_df,
    long_df_to_daily_values,
    run_topk_backtest_for_long_df,
    save_backtest_rows,
    summarize_backtest,
)
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _upsert_index_row,
)
from scripts.cogalpha_lqtp.lqtp_dsl_compat import eval_route_for_entry  # noqa: E402


def _eval_route(entry: dict[str, Any]) -> str:
    return eval_route_for_entry(status=entry.get("status", ""), dsl=entry.get("dsl", ""))


def _normalize_factor_long(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "trade_date" not in work.columns:
        if "datetime" in work.columns:
            work["trade_date"] = pd.to_datetime(work["datetime"]).dt.strftime("%Y%m%d").astype(int)
        else:
            raise KeyError("factor parquet missing trade_date/datetime")
    if "symbol" not in work.columns:
        if "asset" in work.columns:
            work["symbol"] = work["asset"].astype(str).map(_to_lqtp_symbol)
        else:
            raise KeyError("factor parquet missing symbol/asset")
    return work[["trade_date", "symbol", "value"]]
from scripts.cogalpha_lqtp.report_html import (  # noqa: E402
    render_factor_report,
    render_index,
    render_production_summary,
)
from scripts.cogalpha_lqtp.run_light_test import _yyyymmdd  # noqa: E402
from scripts.cogalpha_lqtp.run_production_batch import (  # noqa: E402
    _apply_flip_state_to_catalog,
    _mean_ic_value,
    _negate_formula,
    _negate_python_code,
    _negate_values_parquet,
    _save_progress,
    _upsert_index_row,
)


def _json_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


_TOKEN_MGR: LqtpTokenManager | None = None


def _token_mgr_for(payload: dict[str, Any]) -> LqtpTokenManager:
    global _TOKEN_MGR
    if _TOKEN_MGR is None:
        _TOKEN_MGR = LqtpTokenManager.login(
            payload.get("server", DEFAULT_SERVER),
            payload["username"],
            payload["password"],
        )
    return _TOKEN_MGR


def _regen_one(payload: dict[str, Any]) -> dict[str, Any]:
    name = payload["name"]
    try:
        work = Path(payload["work_dir"])
        lake = work / "factor_lake"
        report_dir = work / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        returns_cache = work / "lqtp_close_returns_cache.parquet"
        if not returns_cache.exists():
            raise RuntimeError(f"missing returns cache: {returns_cache}")

        entry = dict(payload["entry"])
        py_code = payload.get("python_code") or entry.get("python_code") or ""
        flip_state = payload.get("flip_state") or {}
        state = flip_state.get(name, {})
        if state.get("dsl"):
            entry["dsl"] = state["dsl"]
        if state.get("python_code"):
            py_code = state["python_code"]
            entry["python_code"] = py_code
        if state.get("ic_sign_flipped"):
            entry["ic_sign_flipped"] = True

        out_path = lake / name / "values.parquet"
        if not out_path.exists():
            raise RuntimeError(f"missing values: {out_path}")

        dsl_used = (entry.get("dsl") or "").strip() or "(python only)"
        route = _eval_route(entry)
        engine = "python" if route == "local_python" else "factor_engine"
        if route == "lqtp_dsl":
            engine = "lqtp_dsl"

        begin_i = int(payload["begin_i"])
        end_i = int(payload["end_i"])

        long_df = _normalize_factor_long(pd.read_parquet(out_path))
        returns_long = fetch_lqtp_close_returns(
            token="",
            begin_date=begin_i,
            end_date=end_i,
            server=payload.get("server", DEFAULT_SERVER),
            cache_path=returns_cache,
        )
        analysis = analyze_values_with_lqtp_returns(
            factor_long=long_df,
            returns_long=returns_long,
            return_mode="close_to_close",
        )
        eval_mode = "local_values_lqtp_returns"
        flip_patch: dict[str, Any] | None = None

        ic = _mean_ic_value(analysis)
        if payload.get("flip_negative_ic") and ic < 0 and ic == ic and not entry.get("ic_sign_flipped"):
            flip_patch = {"ic_sign_flipped": True}
            entry["ic_sign_flipped"] = True
            if dsl_used and dsl_used != "(python only)":
                dsl_used = _negate_formula(dsl_used)
                entry["dsl"] = dsl_used
                flip_patch["dsl"] = dsl_used
            if py_code.strip():
                py_code = _negate_python_code(py_code)
                entry["python_code"] = py_code
                flip_patch["python_code"] = py_code
            _negate_values_parquet(out_path)
            flip_patch["values_negated"] = True
            long_df = _normalize_factor_long(pd.read_parquet(out_path))
            analysis = analyze_values_with_lqtp_returns(
                factor_long=long_df,
                returns_long=returns_long,
                return_mode="close_to_close",
            )

        meta: dict[str, Any] = {
            "engine": engine,
            "eval_route": route,
            "rows": int(len(long_df)),
            "values_path": str(out_path),
            "date_range": [payload["start"], payload["end"]],
            "formula": dsl_used if dsl_used != "(python only)" else "",
        }
        if flip_patch:
            meta["ic_sign_flipped"] = True
            meta["ic_before_flip"] = ic
            meta["mean_ic_after_flip"] = _mean_ic_value(analysis)

        backtest_rows: list[dict[str, Any]] = []
        backtest_id = ""
        topk_summary: dict[str, Any] = {}
        if payload.get("with_backtest"):
            lqtp_long = factor_values_to_long_df(long_df_to_daily_values(long_df))
            allowed = set(payload.get("backtest_symbols") or [])
            work = Path(payload["work_dir"])
            open_cache = work / "lqtp_open_returns_cache.parquet"
            open_returns = pd.read_parquet(open_cache) if open_cache.exists() else None
            backtest_rows, backtest_id, topk_summary, excluded = run_topk_backtest_for_long_df(
                _token_mgr_for(payload),
                lqtp_long,
                begin_date=begin_i,
                end_date=end_i,
                server=payload.get("server", DEFAULT_SERVER),
                allowed_symbols=allowed or None,
                open_returns_long=open_returns,
                pre_excluded=set(),
            )
            if backtest_rows:
                save_backtest_rows(lake / name / "backtest_topk.json", backtest_rows)

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
            engine=engine,
            eval_route=route,
        )
        row = {
            "factor_name": name,
            "report": report_path.name,
            "engine": engine,
            "eval_route": route,
            "eval_mode": eval_mode,
            "ic_sign_flipped": bool(entry.get("ic_sign_flipped")),
            "mean_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
            "mean_rank_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
            "icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
            "rank_icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
            "rank_ic_positive_ratio": _json_float(
                analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
            ),
            "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
            "long_short_return": _json_float(analysis.get("long_short_return")),
            "backtest_total_ret": _json_float(topk_summary.get("total_return")),
            "backtest_ann_ret": _json_float(topk_summary.get("annualized_return")),
            "backtest_sharpe": _json_float(topk_summary.get("sharpe")),
            "backtest_max_drawdown": _json_float(topk_summary.get("max_drawdown")),
            "backtest_volatility": _json_float(topk_summary.get("volatility")),
            "backtest_calmar": _json_float(topk_summary.get("calmar")),
            "backtest_win_rate": _json_float(topk_summary.get("win_rate")),
            "backtest_avg_turnover": _json_float(topk_summary.get("avg_turnover")),
            "backtest_total_commission": _json_float(topk_summary.get("total_commission")),
            "backtest_trading_days": int(topk_summary.get("trading_days") or 0),
            "backtest_final_nav": _json_float(topk_summary.get("final_nav")),
            "rows": int(len(long_df)),
            "backtest_id": backtest_id,
            "return_kind": str(analysis.get("return_kind", "")),
        }
        print(f"ok {name} rank_ic={row['mean_rank_ic']:.4f} bt={'yes' if backtest_rows else 'skip'}")
        return {"ok": True, "name": name, "row": row, "flip_patch": flip_patch, "error": None}
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
        return {"ok": False, "name": name, "row": None, "flip_patch": None, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast regen HTML from factor_lake parquet")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", "james.gd.luo@gmail.com"))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", "3213709208"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--with-backtest", action="store_true", help="run TopK backtest if LQTP reachable")
    parser.add_argument(
        "--flip-negative-ic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    work = args.work_dir
    catalog_path = work / "dsl_catalog.json"
    parsed_path = work / "parsed_factors.json"
    progress_path = work / "production_progress.json"
    lake = work / "factor_lake"
    report_dir = work / "reports"

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    python_map = {r["function_name"]: r["python_code"] for r in json.loads(parsed_path.read_text(encoding="utf-8"))}
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    flip_state: dict[str, Any] = progress.setdefault("flip_state", {})
    _apply_flip_state_to_catalog(catalog, flip_state, python_map)

    entry_by_name = {e["function_name"]: e for e in catalog}
    names = sorted(p.name for p in lake.iterdir() if (p / "values.parquet").exists())
    skipped = set(progress.get("skipped", []))
    if skipped:
        names = [n for n in names if n not in skipped]
    if args.only:
        only = set(args.only)
        names = [n for n in names if n in only]

    token = ""
    backtest_symbols: list[str] = []
    if args.with_backtest:
        try:
            auth = LqtpTokenManager.login(args.server, args.username, args.password)
            token = auth.token
            from scripts.cogalpha_lqtp.lqtp_client import fetch_lqtp_universe

            begin_i = _yyyymmdd(args.start)
            end_i = _yyyymmdd(args.end)
            backtest_symbols = sorted(fetch_lqtp_universe(token=token, begin_date=begin_i, end_date=end_i, server=args.server))
            print(f"LQTP ok, backtest universe={len(backtest_symbols)}")
        except Exception as exc:  # noqa: BLE001
            print(f"LQTP unavailable, skip backtest: {exc}")

    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    payloads = []
    for name in names:
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
                "flip_state": flip_state,
                "begin_i": begin_i,
                "end_i": end_i,
                "start": args.start,
                "end": args.end,
                "server": args.server,
                "username": args.username,
                "password": args.password,
                "backtest_symbols": backtest_symbols,
                "with_backtest": bool(token),
                "flip_negative_ic": args.flip_negative_ic,
            }
        )

    print(f"regen {len(payloads)} factors workers={args.workers} backtest={'on' if token else 'off'}")
    index_rows: list[dict[str, Any]] = list(progress.get("index_rows", []))
    failed: dict[str, str] = dict(progress.get("failed", {}))
    completed: list[str] = list(progress.get("completed", []))

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_regen_one, p): p["name"] for p in payloads}
        for fut in as_completed(futures):
            result = fut.result()
            name = result["name"]
            if result["ok"]:
                _upsert_index_row(index_rows, result["row"])
                if name not in completed:
                    completed.append(name)
                failed.pop(name, None)
                fp = result.get("flip_patch")
                if fp:
                    flip_state[name] = {**flip_state.get(name, {}), **fp}
            else:
                failed[name] = result.get("error") or "unknown"

    progress["completed"] = completed
    progress["failed"] = failed
    progress["index_rows"] = index_rows
    progress["flip_state"] = flip_state
    progress["eval_methodology"] = "rankic_close_to_close_Tplus1_topk_open_v3"
    progress["regen_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_progress(progress_path, progress)

    summary_meta = {
        "generated_at": progress["regen_at"],
        "eval_methodology": progress["eval_methodology"],
        "note": "Fast regen from factor_lake (cached returns)",
    }
    render_index(index_rows, report_dir / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=report_dir / "production_summary.html",
        meta=summary_meta,
        failed=failed,
    )
    print(f"done ok={len(completed)} failed={len(failed)} reports={report_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
