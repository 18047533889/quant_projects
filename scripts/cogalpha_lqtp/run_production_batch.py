#!/usr/bin/env python3
"""Production batch: 2019-2026 materialize (DSL + Python) → LQTP eval → HTML reports.

Memory model (~30G RAM, no swap):
  - bounded parallel: lqtp_dsl + local_engine capped by RAM (default total <= 3)
  - explicit gc + data_access store reset between factors
  - resume via production_progress.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
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

from scripts.cogalpha_lqtp.factor_eval import evaluate_lqtp_formula, fetch_lqtp_close_returns, fetch_lqtp_open_returns  # noqa: E402
from scripts.cogalpha_lqtp.ast_translator import dsl_to_lqtp, lqtp_to_fe_dsl  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_dsl_compat import eval_route_for_entry, is_lqtp_native_dsl  # noqa: E402
from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    LqtpTokenManager,
    evaluate_factor,
    factor_values_to_long_df,
    fetch_lqtp_universe,
    is_lqtp_auth_error,
    long_df_to_daily_values,
    long_short_decile_weights,
    run_backtest_auth_safe,
    run_backtest_from_weights,
    run_topk_backtest_for_long_df,
    safe_backtest,
    summarize_backtest,
    top_quantile_weights,
)
from scripts.cogalpha_lqtp.materialize import materialize_factor  # noqa: E402
from scripts.cogalpha_lqtp.memory_utils import (  # noqa: E402
    DEFAULT_MAX_TOTAL_PARALLEL,
    DEFAULT_MEM_RESERVE_GB,
    LOCAL_MATERIALIZE_MEM_GB,
    LOCAL_REUSE_MEM_GB,
    LQTP_JOB_MEM_GB,
    cap_combined_parallel,
    ensure_memory_floor,
    estimate_local_parallel_workers,
    estimate_lqtp_parallel_workers,
    read_mem_available_gb,
    release_memory,
    wait_for_memory,
)
from scripts.cogalpha_lqtp.python_materialize import materialize_python_factor  # noqa: E402
from scripts.cogalpha_lqtp.report_html import render_factor_report, render_index, render_production_summary  # noqa: E402
from scripts.cogalpha_lqtp.run_light_test import _run_py, _yyyymmdd  # noqa: E402

# Per-symbol full-history .rank() fixes deferred until main catalog is complete.
LOOKAHEAD_DEFERRED_FACTORS: tuple[str, ...] = (
    "factor_adaptive_vol_volume_asym",
    "factor_drawdown_vol_simple",
    "factor_drawdown_volume_complexity",
    "factor_drawdown_volume_geometry",
    "factor_drawdown_volume_geometry_rolling",
    "factor_drawdown_volume_modulated",
    "factor_fear_adjusted_dollar_pressure_short_ema",
    "factor_herding_pressure",
    "factor_pressure_ema_mutation",
    "factor_simplified_fear_pressure",
    "factor_stability_crash_guard",
    "factor_volatility_adjusted_dollar_pressure",
)

def _ashare_data_source(
    start: str,
    end: str,
    *,
    instrument_filter: list[str] | None = None,
) -> dict[str, Any]:
    cfg = default_ashare_pv_data_source_config(start_date=start, end_date=end)
    if instrument_filter:
        cfg["instrument_filter"] = sorted(instrument_filter)
    return cfg


def _count_bar_files(data_root: Path) -> int:
    bar = data_root / "StockDailyBar"
    if not bar.is_dir():
        return 0
    return sum(1 for _ in bar.glob("*.parquet"))


def _has_market_data(data_root: Path, *, min_files: int = 20) -> bool:
    return _count_bar_files(data_root) >= min_files


def _load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("index_rows", [])
        data.setdefault("flip_state", {})
        return data
    return {"completed": [], "failed": {}, "skipped": [], "index_rows": [], "flip_state": {}}


def _save_progress(path: Path, data: dict[str, Any]) -> None:
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float):
            return float(obj)
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:  # noqa: BLE001
                return str(obj)
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        return str(obj)

    path.write_text(json.dumps(_sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _route(entry: dict[str, Any]) -> str:
    st = entry.get("status", "")
    dsl = entry.get("dsl", "")
    if st in {"python", "hard"} or not dsl:
        return "python"
    if st == "ready":
        return "factor_engine"
    return "python"


def _eval_route(entry: dict[str, Any]) -> str:
    route = entry.get("eval_route", "")
    if route:
        return route
    # Prefer LQTP formula when rename-compatible.
    lqtp = (entry.get("lqtp_formula") or entry.get("dsl") or "").strip()
    if entry.get("status") == "ready" and is_lqtp_native_dsl(lqtp):
        return "lqtp_dsl"
    return eval_route_for_entry(status=entry.get("status", ""), dsl=entry.get("dsl", ""))


def _lqtp_run_formula(entry: dict[str, Any]) -> str:
    """Formula sent to LQTP RunFactor (LQTP operator naming)."""
    for key in ("lqtp_formula", "dsl"):
        text = (entry.get(key) or "").strip()
        if text and is_lqtp_native_dsl(text):
            return text
    dsl = (entry.get("dsl") or "").strip()
    if not dsl:
        return ""
    converted = dsl_to_lqtp(dsl)
    return converted if is_lqtp_native_dsl(converted) else dsl


def _normalize_lqtp_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if "." in text:
        return text
    if text.isdigit():
        return f"{text.zfill(6)}.SZ"
    return text


def _filter_to_universe(long_df: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    if long_df.empty or not universe:
        return long_df
    work = long_df.copy()
    work["asset"] = work["asset"].map(_normalize_lqtp_symbol)
    return work[work["asset"].isin(universe)].copy()


def _mean_ic_value(analysis: dict[str, Any]) -> float:
    try:
        return float(analysis.get("mean_rank_ic", analysis.get("mean_ic", float("nan"))))
    except (TypeError, ValueError):
        return float("nan")


def _negate_formula(dsl: str) -> str:
    """Permanently rewrite DSL as -(expr). Idempotent for our own wrap."""
    text = (dsl or "").strip()
    if not text or text == "(python only)":
        return text
    if text.startswith("-(") and text.endswith(")"):
        return text
    return f"-({text})"


def _negate_python_code(code: str) -> str:
    """Permanently rewrite Python factor so every return is negated."""
    text = (code or "").rstrip()
    if not text:
        return text
    if "IC_SIGN_FLIPPED" in text:
        return text if text.endswith("\n") else text + "\n"
    out = ["# IC_SIGN_FLIPPED: definition permanently negated (orig RankIC < 0)"]
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("return ") and not stripped.startswith("return -("):
            indent = line[: len(line) - len(stripped)]
            expr = stripped[len("return ") :]
            out.append(f"{indent}return -({expr})")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _negate_values_parquet(path: Path) -> None:
    frame = pd.read_parquet(path)
    frame["value"] = -pd.to_numeric(frame["value"], errors="coerce")
    frame.to_parquet(path, index=False)


def _apply_flip_state_to_catalog(
    catalog: list[dict[str, Any]],
    flip_state: dict[str, Any],
    python_map: dict[str, str] | None = None,
) -> None:
    for entry in catalog:
        name = entry["function_name"]
        state = flip_state.get(name, {})
        if not state:
            continue
        dsl = state.get("dsl")
        if dsl:
            converted = dsl_to_lqtp(dsl)
            if is_lqtp_native_dsl(converted):
                entry["dsl"] = converted
                entry["lqtp_formula"] = converted
                entry["eval_route"] = "lqtp_dsl"
                entry["lqtp_native"] = True
            else:
                entry["dsl"] = lqtp_to_fe_dsl(dsl)
                entry["lqtp_formula"] = state.get("lqtp_formula") or converted
                # Keep catalog route unless flip forced a non-native formula.
                if entry.get("eval_route") == "lqtp_dsl":
                    entry["eval_route"] = "local_dsl"
                    entry["lqtp_native"] = False
            state["dsl"] = entry["dsl"]
            state["lqtp_formula"] = entry.get("lqtp_formula", "")
        elif state.get("lqtp_formula"):
            entry["lqtp_formula"] = state["lqtp_formula"]
        if state.get("ic_sign_flipped"):
            entry["ic_sign_flipped"] = True
        py = state.get("python_code")
        if py and python_map is not None:
            python_map[name] = py
            entry["python_code"] = py


def _ensure_flipped_python_definitions(
    *,
    flip_state: dict[str, Any],
    python_map: dict[str, str],
    progress: dict[str, Any],
) -> list[str]:
    """Backfill negated python into flip_state; re-queue those factors to refresh reports."""
    queued: list[str] = []
    completed = set(progress.get("completed", []))
    for name, state in list(flip_state.items()):
        if not state.get("ic_sign_flipped"):
            continue
        py = state.get("python_code") or ""
        if "IC_SIGN_FLIPPED" in py:
            python_map[name] = py
            continue
        orig = python_map.get(name, "")
        if not orig.strip():
            continue
        negated = _negate_python_code(orig)
        state["python_code"] = negated
        python_map[name] = negated
        if name in completed:
            completed.discard(name)
            queued.append(name)
    if queued:
        progress["completed"] = [x for x in progress.get("completed", []) if x not in queued]
        progress["index_rows"] = [
            r for r in progress.get("index_rows", []) if r.get("factor_name") not in queued
        ]
        print(f"flip python definition refresh queued: {len(queued)} factors")
    return queued


def _queue_negative_ic_retests(
    *,
    catalog: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    lake: Path,
    flip_negative_ic: bool,
    python_map: dict[str, str] | None = None,
) -> list[str]:
    """Permanently negate factor DSL/python/values when mean_ic<0 and re-test once."""
    if not flip_negative_ic:
        return []
    ic_map = {row["factor_name"]: _mean_ic_value(row) for row in index_rows}
    completed = set(progress.get("completed", []))
    flip_state: dict[str, Any] = progress.setdefault("flip_state", {})
    queued: list[str] = []
    py_map = python_map if python_map is not None else {}

    entry_by_name = {entry["function_name"]: entry for entry in catalog}
    for name in sorted(completed):
        ic = ic_map.get(name)
        if ic is None or ic >= 0 or ic != ic:
            continue
        if flip_state.get(name, {}).get("ic_sign_flipped"):
            continue
        entry = entry_by_name.get(name)
        if entry is None:
            continue

        patch: dict[str, Any] = {"ic_sign_flipped": True}
        dsl = (entry.get("dsl") or "").strip()
        if dsl and dsl != "(python only)":
            negated = _negate_formula(dsl)
            entry["dsl"] = negated
            entry["lqtp_formula"] = negated if is_lqtp_native_dsl(negated) else dsl_to_lqtp(negated)
            patch["dsl"] = negated
            patch["lqtp_formula"] = entry["lqtp_formula"]
        orig_py = py_map.get(name, "") or entry.get("python_code", "")
        if orig_py.strip():
            negated_py = _negate_python_code(orig_py)
            py_map[name] = negated_py
            entry["python_code"] = negated_py
            patch["python_code"] = negated_py
        flip_state[name] = patch

        out_path = lake / name / "values.parquet"
        if out_path.exists():
            _negate_values_parquet(out_path)
            patch["values_negated"] = True

        completed.discard(name)
        queued.append(name)

    if queued:
        progress["completed"] = [x for x in progress.get("completed", []) if x not in queued]
        progress["index_rows"] = [
            r for r in progress.get("index_rows", []) if r.get("factor_name") not in queued
        ]
        print(f"negative IC sign flip queued for re-test: {len(queued)} factors")
    return queued


def _eval_local_parquet(
    *,
    out_path: Path,
    lqtp_universe: set[str],
    backtest_symbols: set[str],
    auth: LqtpTokenManager,
    job: "FactorJob",
    returns_cache: Path,
) -> tuple[dict[str, Any], str, Any, list[dict[str, Any]], str]:
    long_df = pd.read_parquet(out_path)
    tradable = lqtp_universe & backtest_symbols
    long_df = _filter_to_universe(long_df, tradable)
    if long_df.empty:
        raise RuntimeError("no factor values overlap LQTP tradable universe")
    daily_values = long_df_to_daily_values(long_df)
    release_memory(long_df)
    analysis, eval_mode = evaluate_factor(
        token=auth.token,
        daily_values=daily_values,
        begin_date=job.begin_i,
        end_date=job.end_i,
        server=job.server,
        returns_cache=returns_cache,
    )
    lqtp_long = factor_values_to_long_df(daily_values)
    backtest_rows, backtest_id, topk_summary, ls_rows, ls_id, ls_summary = _run_platform_backtests(
        auth=auth,
        lqtp_long=lqtp_long,
        backtest_symbols=backtest_symbols,
        begin_date=job.begin_i,
        end_date=job.end_i,
        server=job.server,
        work_dir=Path(job.work_dir),
    )
    release_memory(lqtp_long)
    # stash extras on analysis for report path via caller
    analysis = dict(analysis)
    analysis["_topk_summary"] = topk_summary
    analysis["_ls_backtest_rows"] = ls_rows
    analysis["_ls_summary"] = ls_summary
    analysis["_ls_backtest_id"] = ls_id
    return analysis, eval_mode, daily_values, backtest_rows, backtest_id


def _eval_lqtp_and_backtest(
    *,
    dsl: str,
    auth: LqtpTokenManager,
    job: "FactorJob",
    backtest_symbols: set[str],
) -> tuple[dict[str, Any], str, Any, list[dict[str, Any]], str, int]:
    from scripts.cogalpha_lqtp.eval_lake_fast import build_fwd_returns_cache

    work = Path(job.work_dir)
    fwd = build_fwd_returns_cache(
        work / "lqtp_close_returns_cache.parquet",
        work / "lqtp_fwd_close_returns_cache.parquet",
    )
    analysis, eval_mode, daily_values = evaluate_lqtp_formula(
        token=auth.token,
        formula=dsl,
        begin_date=job.begin_i,
        end_date=job.end_i,
        server=job.server,
        fwd_returns_path=fwd,
    )
    rows = sum(len(point.values) for point in daily_values)
    lqtp_long = factor_values_to_long_df(daily_values)
    backtest_rows, backtest_id, topk_summary, ls_rows, ls_id, ls_summary = _run_platform_backtests(
        auth=auth,
        lqtp_long=lqtp_long,
        backtest_symbols=backtest_symbols,
        begin_date=job.begin_i,
        end_date=job.end_i,
        server=job.server,
        work_dir=Path(job.work_dir),
    )
    release_memory(lqtp_long)
    analysis = dict(analysis)
    analysis["_topk_summary"] = topk_summary
    analysis["_ls_backtest_rows"] = ls_rows
    analysis["_ls_summary"] = ls_summary
    analysis["_ls_backtest_id"] = ls_id
    return analysis, eval_mode, daily_values, backtest_rows, backtest_id, rows


def _maybe_flip_negative_ic_and_retest(
    *,
    job: "FactorJob",
    auth: LqtpTokenManager,
    analysis: dict[str, Any],
    eval_mode: str,
    daily_values: Any,
    backtest_rows: list[dict[str, Any]],
    backtest_id: str,
    dsl_used: str,
    meta: dict[str, Any],
    rows: int,
    ran_via_lqtp: bool,
    actual_eval_route: str,
    eval_route: str,
    lake: Path,
    name: str,
    lqtp_universe: set[str],
    backtest_symbols: set[str],
    returns_cache: Path,
) -> tuple[
    dict[str, Any],
    str,
    Any,
    list[dict[str, Any]],
    str,
    str,
    dict[str, Any],
    int,
    dict[str, Any] | None,
]:
    """If mean_ic<0, negate formula/values on the original factor and re-test once."""
    flip_patch: dict[str, Any] | None = None
    if job.skip_eval or not job.flip_negative_ic:
        return analysis, eval_mode, daily_values, backtest_rows, backtest_id, dsl_used, meta, rows, flip_patch
    if job.entry.get("ic_sign_flipped"):
        return analysis, eval_mode, daily_values, backtest_rows, backtest_id, dsl_used, meta, rows, flip_patch

    ic = _mean_ic_value(analysis)
    if ic >= 0 or ic != ic:
        return analysis, eval_mode, daily_values, backtest_rows, backtest_id, dsl_used, meta, rows, flip_patch

    print(f"  mean_ic={ic:.4f}<0, permanently negate factor definition and retest")
    flip_patch = {"ic_sign_flipped": True}
    job.entry["ic_sign_flipped"] = True

    out_path = lake / name / "values.parquet"
    if dsl_used and dsl_used not in {"", "(python only)"}:
        dsl_used = _negate_formula(dsl_used)
        job.entry["dsl"] = dsl_used
        job.entry["lqtp_formula"] = dsl_used
        flip_patch["dsl"] = dsl_used
        flip_patch["lqtp_formula"] = dsl_used
        meta["formula"] = dsl_used
    orig_py = (job.python_code or "").strip()
    if orig_py:
        negated_py = _negate_python_code(orig_py)
        job.entry["python_code"] = negated_py
        flip_patch["python_code"] = negated_py
    if out_path.exists():
        _negate_values_parquet(out_path)
        flip_patch["values_negated"] = True

    release_memory(daily_values)
    daily_values = None

    if ran_via_lqtp or (eval_route == "lqtp_dsl" and actual_eval_route == "lqtp_dsl" and dsl_used):
        analysis, eval_mode, daily_values, backtest_rows, backtest_id, rows = _eval_lqtp_and_backtest(
            dsl=dsl_used,
            auth=auth,
            job=job,
            backtest_symbols=backtest_symbols,
        )
    elif out_path.exists():
        analysis, eval_mode, daily_values, backtest_rows, backtest_id = _eval_local_parquet(
            out_path=out_path,
            lqtp_universe=lqtp_universe,
            backtest_symbols=backtest_symbols,
            auth=auth,
            job=job,
            returns_cache=returns_cache,
        )
        rows = int(pd.read_parquet(out_path, columns=["value"]).shape[0])
    else:
        raise RuntimeError(f"{name}: cannot retest after sign flip without DSL or values.parquet")

    meta["ic_sign_flipped"] = True
    meta["ic_before_flip"] = ic
    meta["mean_ic_after_flip"] = _mean_ic_value(analysis)
    return analysis, eval_mode, daily_values, backtest_rows, backtest_id, dsl_used, meta, rows, flip_patch


def _backtest_total_ret(rows: list[dict[str, Any]]) -> float:
    return float(summarize_backtest(rows).get("total_return", float("nan")))


def _load_missing_quote_cache(work_dir: Path) -> set[str]:
    path = work_dir / "backtest_missing_quote_symbols.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return set()


def _save_missing_quote_cache(work_dir: Path, excluded: set[str]) -> None:
    if not excluded:
        return
    path = work_dir / "backtest_missing_quote_symbols.json"
    path.write_text(json.dumps(sorted(excluded), ensure_ascii=False, indent=2), encoding="utf-8")


def _run_platform_backtests(
    *,
    auth: LqtpTokenManager,
    lqtp_long: pd.DataFrame,
    backtest_symbols: set[str],
    begin_date: int,
    end_date: int,
    server: str,
    work_dir: Path | None = None,
    open_returns_long: pd.DataFrame | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    """TopK long-only + optional long-short platform backtests at OPEN."""
    if open_returns_long is None and work_dir is not None:
        open_cache = work_dir / "lqtp_open_returns_cache.parquet"
        if open_cache.exists():
            open_returns_long = pd.read_parquet(open_cache)
    backtest_rows, backtest_id, topk_summary, excluded = run_topk_backtest_for_long_df(
        auth,
        lqtp_long,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        allowed_symbols=backtest_symbols,
        open_returns_long=open_returns_long,
        pre_excluded=set(),
    )

    ls_rows: list[dict[str, Any]] = []
    ls_id = ""
    ls_summary: dict[str, Any] = {}
    try:
        ls_weights = long_short_decile_weights(lqtp_long, allowed_symbols=backtest_symbols)
        ls_rows, ls_id = run_backtest_auth_safe(
            auth,
            weights=ls_weights,
            begin_date=begin_date,
            end_date=end_date,
            server=server,
            enable_short_selling=True,
        )
        release_memory(ls_weights)
        ls_summary = summarize_backtest(ls_rows)
    except Exception as exc:  # noqa: BLE001
        if is_lqtp_auth_error(exc):
            raise
        print(f"  long-short platform backtest skipped: {exc}")
        ls_rows, ls_id, ls_summary = [], "", {}

    return backtest_rows, backtest_id, topk_summary, ls_rows, ls_id, ls_summary


def _materialize_entry(
    *,
    entry: dict[str, Any],
    python_code: str,
    data_cfg: dict[str, Any],
    lake_root: Path,
    start: str,
    end: str,
    symbol_chunk: int,
    python_workers: int,
    allowed_symbols: set[str] | None = None,
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
            workers=python_workers,
            allowed_symbols=allowed_symbols,
        )
        return out, "python", dsl or "(python only)"

    ok, msg = validate_factor_engine_dsl(lqtp_to_fe_dsl(dsl))
    if not ok:
        raise RuntimeError(f"DSL invalid: {msg}")
    out = materialize_factor(
        factor_id=name,
        dsl=dsl,
        data_source_cfg=data_cfg,
        lake_root=lake_root,
        allowed_symbols=allowed_symbols,
    )
    return out, "factor_engine", dsl


def _write_route_summary(catalog: list[dict[str, Any]], out_path: Path) -> None:
    groups: dict[str, list[dict[str, str]]] = {
        "lqtp_dsl": [],
        "local_dsl": [],
        "local_python": [],
    }
    for entry in catalog:
        route = _eval_route(entry)
        groups.setdefault(route, []).append(
            {
                "function_name": entry["function_name"],
                "dsl": entry.get("dsl", ""),
                "fe_only_ops": entry.get("fe_only_ops", ""),
                "status": entry.get("status", ""),
            }
        )
    payload = {
        "summary": {route: len(items) for route, items in groups.items()},
        "routes": groups,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _backtest_symbol_set(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str,
    returns_cache: Path,
    lqtp_universe: set[str],
) -> set[str]:
    """Symbols with LQTP open-to-open return bars (safe for BacktestService)."""
    returns_long = fetch_lqtp_open_returns(
        token=token,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=returns_cache,
    )
    symbols = set(returns_long["symbol"].astype(str).map(_normalize_lqtp_symbol))
    if lqtp_universe:
        symbols &= lqtp_universe
    return symbols


def _upsert_index_row(index_rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    name = row["factor_name"]
    for i, existing in enumerate(index_rows):
        if existing.get("factor_name") == name:
            index_rows[i] = row
            return
    index_rows.append(row)


def _write_reports(
    *,
    index_rows: list[dict[str, Any]],
    report_dir: Path,
    meta: dict[str, Any],
    failed: dict[str, str],
) -> None:
    render_index(index_rows, report_dir / "index.html")
    render_production_summary(
        rows=index_rows,
        out_path=report_dir / "production_summary.html",
        meta=meta,
        failed=failed,
    )


@dataclass(frozen=True)
class FactorJob:
    index: int
    total: int
    entry: dict[str, Any]
    python_code: str
    work_dir: str
    start: str
    end: str
    server: str
    username: str
    password: str
    begin_i: int
    end_i: int
    symbol_chunk: int
    python_workers: int
    skip_eval: bool
    token_refresh_minutes: float
    min_mem_gb: float
    mem_reserve_gb: float
    lqtp_job_mem_gb: float
    local_materialize_mem_gb: float
    local_reuse_mem_gb: float
    lqtp_universe: tuple[str, ...]
    backtest_symbols: tuple[str, ...]
    data_cfg: dict[str, Any]
    flip_negative_ic: bool = True


def _local_needs_materialize(job: FactorJob) -> bool:
    out_path = Path(job.work_dir) / "factor_lake" / job.entry["function_name"] / "values.parquet"
    return not out_path.exists()


def _lqtp_run_factor_should_fallback(exc: BaseException) -> bool:
    msg = str(exc)
    return "公式完全重复" in msg or "未知算子" in msg


def _safe_backtest(**kwargs: Any) -> tuple[list[dict[str, Any]], str]:
    """Run backtest; auto-refresh token when *token_mgr* is provided."""
    return safe_backtest(**kwargs)


def _job_mem_need_gb(job: FactorJob) -> float:
    route = _eval_route(job.entry)
    if route == "lqtp_dsl":
        return job.lqtp_job_mem_gb
    if _local_needs_materialize(job):
        return job.local_materialize_mem_gb
    return job.local_reuse_mem_gb


def _run_parallel_batch(
    *,
    pending: list[FactorJob],
    progress: dict[str, Any],
    index_rows: list[dict[str, Any]],
    progress_path: Path,
    report_dir: Path,
    summary_meta: dict[str, Any],
    lqtp_workers: int,
    local_workers: int,
    mem_reserve_gb: float,
) -> None:
    """Submit jobs incrementally; never exceed worker caps; gate on MemAvailable."""
    lqtp_jobs = [j for j in pending if _eval_route(j.entry) == "lqtp_dsl"]
    local_jobs = [j for j in pending if _eval_route(j.entry) != "lqtp_dsl"]
    progress_lock = threading.Lock()
    mp_ctx = multiprocessing.get_context("spawn")

    lqtp_workers = lqtp_workers if lqtp_jobs else 0
    local_workers = local_workers if local_jobs else 0

    print(
        f"parallel dispatch (memory-safe): lqtp_dsl={len(lqtp_jobs)}x{lqtp_workers} "
        f"local={len(local_jobs)}x{local_workers} reserve={mem_reserve_gb:.1f}G"
    )

    lqtp_pool = ProcessPoolExecutor(max_workers=max(1, lqtp_workers), mp_context=mp_ctx) if lqtp_workers else None
    local_pool = ProcessPoolExecutor(max_workers=max(1, local_workers), mp_context=mp_ctx) if local_workers else None

    lqtp_pending = list(lqtp_jobs)
    local_pending = list(local_jobs)
    in_flight: dict[Future, FactorJob] = {}
    running_lqtp = 0
    running_local = 0

    def _handle_result(result: dict[str, Any]) -> None:
        _apply_factor_result(
            result=result,
            progress=progress,
            index_rows=index_rows,
            progress_path=progress_path,
            report_dir=report_dir,
            summary_meta=summary_meta,
            lock=progress_lock,
        )
        if not result["ok"] and result.get("traceback"):
            print(result["traceback"])

    try:
        while lqtp_pending or local_pending or in_flight:
            avail = read_mem_available_gb()

            if local_pending and running_local < local_workers:
                job = local_pending[0]
                need = _job_mem_need_gb(job)
                if avail >= need + mem_reserve_gb:
                    wait_for_memory(required_gb=need, reserve_gb=mem_reserve_gb, timeout_sec=30.0)
                    local_pending.pop(0)
                    assert local_pool is not None
                    fut = local_pool.submit(_process_one_factor, job)
                    in_flight[fut] = job
                    running_local += 1
                    print(
                        f"  +local {job.entry['function_name']} "
                        f"(running_lqtp={running_lqtp} running_local={running_local} mem={read_mem_available_gb():.1f}G)"
                    )

            avail = read_mem_available_gb()
            if lqtp_pending and running_lqtp < lqtp_workers:
                job = lqtp_pending[0]
                need = _job_mem_need_gb(job)
                if avail >= need + mem_reserve_gb:
                    wait_for_memory(required_gb=need, reserve_gb=mem_reserve_gb, timeout_sec=30.0)
                    lqtp_pending.pop(0)
                    assert lqtp_pool is not None
                    fut = lqtp_pool.submit(_process_one_factor, job)
                    in_flight[fut] = job
                    running_lqtp += 1
                    print(
                        f"  +lqtp {job.entry['function_name']} "
                        f"(running_lqtp={running_lqtp} running_local={running_local} mem={read_mem_available_gb():.1f}G)"
                    )

            if not in_flight:
                if lqtp_pending or local_pending:
                    print(
                        f"  memory wait mem={read_mem_available_gb():.1f}G reserve={mem_reserve_gb:.1f}G "
                        f"pending_lqtp={len(lqtp_pending)} pending_local={len(local_pending)}"
                    )
                    release_memory()
                    time.sleep(5.0)
                continue

            done, _ = wait(in_flight, return_when=FIRST_COMPLETED, timeout=10.0)
            for fut in done:
                job = in_flight.pop(fut)
                route = _eval_route(job.entry)
                if route == "lqtp_dsl":
                    running_lqtp -= 1
                else:
                    running_local -= 1
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "ok": False,
                        "name": job.entry["function_name"],
                        "row": None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                try:
                    _handle_result(result)
                except Exception:  # noqa: BLE001
                    print(f"  progress update failed for {job.entry['function_name']}:")
                    print(traceback.format_exc())
                release_memory()
    finally:
        if lqtp_pool is not None:
            lqtp_pool.shutdown(wait=True, cancel_futures=False)
        if local_pool is not None:
            local_pool.shutdown(wait=True, cancel_futures=False)


def _process_one_factor(job: FactorJob) -> dict[str, Any]:
    """Worker entry: one factor end-to-end in an isolated process."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(FE_ROOT) not in sys.path:
        sys.path.insert(0, str(FE_ROOT))

    name = job.entry["function_name"]
    work = Path(job.work_dir)
    lake = work / "factor_lake"
    report_dir = work / "reports"
    returns_cache = work / "lqtp_close_returns_cache.parquet"
    lqtp_universe = set(job.lqtp_universe)
    backtest_symbols = set(job.backtest_symbols)
    eval_route = _eval_route(job.entry)

    try:
        need_gb = _job_mem_need_gb(job)
        wait_for_memory(required_gb=need_gb, reserve_gb=job.mem_reserve_gb, timeout_sec=900.0)
        ensure_memory_floor(job.min_mem_gb)
        auth = LqtpTokenManager.login(
            job.server,
            job.username,
            job.password,
            refresh_interval_seconds=int(job.token_refresh_minutes * 60),
        )
        auth.maybe_refresh()

        py_code = job.python_code
        if eval_route == "local_python" and not py_code:
            raise RuntimeError("missing python_code")

        eval_mode = "skipped"
        analysis: dict[str, Any] = {}
        backtest_rows: list[dict[str, Any]] = []
        backtest_id = ""
        ls_backtest_rows: list[dict[str, Any]] = []
        topk_summary: dict[str, Any] = {}
        ls_summary: dict[str, Any] = {}
        daily_values = None
        rows = 0
        dsl_used = ""
        engine = ""
        actual_eval_route = eval_route
        ran_via_lqtp = False
        meta: dict[str, Any] = {}
        flip_patch: dict[str, Any] | None = None

        if eval_route == "lqtp_dsl":
            dsl_used = _lqtp_run_formula(job.entry)
            if not dsl_used:
                raise RuntimeError("lqtp_dsl route missing dsl")
            engine = "lqtp_dsl"
            meta = {
                "engine": engine,
                "eval_route": eval_route,
                "formula": dsl_used,
                "date_range": [job.start, job.end],
            }
            if not job.skip_eval:
                print(f"[{job.index}/{job.total}] {name} → LQTP RunFactor (native DSL)")
                try:
                    from scripts.cogalpha_lqtp.eval_lake_fast import build_fwd_returns_cache

                    fwd = build_fwd_returns_cache(
                        work / "lqtp_close_returns_cache.parquet",
                        work / "lqtp_fwd_close_returns_cache.parquet",
                    )
                    analysis, eval_mode, daily_values = evaluate_lqtp_formula(
                        token=auth.token,
                        formula=dsl_used,
                        begin_date=job.begin_i,
                        end_date=job.end_i,
                        server=job.server,
                        fwd_returns_path=fwd,
                    )
                    rows = sum(len(point.values) for point in daily_values)
                    meta["rows"] = rows
                    lqtp_long = factor_values_to_long_df(daily_values)
                    (
                        backtest_rows,
                        backtest_id,
                        topk_summary,
                        ls_backtest_rows,
                        _ls_id,
                        ls_summary,
                    ) = _run_platform_backtests(
                        auth=auth,
                        lqtp_long=lqtp_long,
                        backtest_symbols=backtest_symbols,
                        begin_date=job.begin_i,
                        end_date=job.end_i,
                        server=job.server,
                        work_dir=Path(job.work_dir),
                    )
                    release_memory(lqtp_long)
                    analysis = dict(analysis)
                    analysis["_topk_summary"] = topk_summary
                    analysis["_ls_backtest_rows"] = ls_backtest_rows
                    analysis["_ls_summary"] = ls_summary
                    ran_via_lqtp = True
                except RuntimeError as exc:
                    if not _lqtp_run_factor_should_fallback(exc):
                        raise
                    print(f"  LQTP RunFactor fallback to local materialize: {exc}")
                    meta["lqtp_fallback_reason"] = str(exc)
                    actual_eval_route = "local_dsl"
                    release_memory(daily_values)
                    daily_values = None

        if not ran_via_lqtp and (eval_route != "lqtp_dsl" or not job.skip_eval):
            out_path = lake / name / "values.parquet"
            if out_path.exists():
                engine = "python" if _route(job.entry) == "python" else "factor_engine"
                dsl_used = job.entry.get("dsl", "") or "(python only)"
                print(f"[{job.index}/{job.total}] {name} reuse {out_path.name}")
            else:
                print(f"[{job.index}/{job.total}] {name} materialize ...")
                out_path, engine, dsl_used = _materialize_entry(
                    entry=job.entry,
                    python_code=py_code,
                    data_cfg=job.data_cfg,
                    lake_root=lake,
                    start=job.start,
                    end=job.end,
                    symbol_chunk=job.symbol_chunk,
                    python_workers=job.python_workers,
                    allowed_symbols=lqtp_universe,
                )
            rows = int(pd.read_parquet(out_path, columns=["value"]).shape[0])
            meta = {
                "engine": engine,
                "eval_route": actual_eval_route,
                "rows": rows,
                "values_path": str(out_path),
                "date_range": [job.start, job.end],
            }
            if not job.skip_eval:
                long_df = pd.read_parquet(out_path)
                tradable = lqtp_universe & backtest_symbols
                long_df = _filter_to_universe(long_df, tradable)
                if long_df.empty:
                    raise RuntimeError("no factor values overlap LQTP tradable universe")
                daily_values = long_df_to_daily_values(long_df)
                release_memory(long_df)
                analysis, eval_mode = evaluate_factor(
                    token=auth.token,
                    daily_values=daily_values,
                    begin_date=job.begin_i,
                    end_date=job.end_i,
                    server=job.server,
                    returns_cache=returns_cache,
                )
                lqtp_long = factor_values_to_long_df(daily_values)
                (
                    backtest_rows,
                    backtest_id,
                    topk_summary,
                    ls_backtest_rows,
                    _ls_id,
                    ls_summary,
                ) = _run_platform_backtests(
                    auth=auth,
                    lqtp_long=lqtp_long,
                    backtest_symbols=backtest_symbols,
                    begin_date=job.begin_i,
                    end_date=job.end_i,
                    server=job.server,
                    work_dir=Path(job.work_dir),
                )
                release_memory(lqtp_long)
                analysis = dict(analysis)
                analysis["_topk_summary"] = topk_summary
                analysis["_ls_backtest_rows"] = ls_backtest_rows
                analysis["_ls_summary"] = ls_summary

        if not job.skip_eval:
            (
                analysis,
                eval_mode,
                daily_values,
                backtest_rows,
                backtest_id,
                dsl_used,
                meta,
                rows,
                flip_patch,
            ) = _maybe_flip_negative_ic_and_retest(
                job=job,
                auth=auth,
                analysis=analysis,
                eval_mode=eval_mode,
                daily_values=daily_values,
                backtest_rows=backtest_rows,
                backtest_id=backtest_id,
                dsl_used=dsl_used,
                meta=meta,
                rows=rows,
                ran_via_lqtp=ran_via_lqtp,
                actual_eval_route=actual_eval_route,
                eval_route=eval_route,
                lake=lake,
                name=name,
                lqtp_universe=lqtp_universe,
                backtest_symbols=backtest_symbols,
                returns_cache=returns_cache,
            )
            # Permanent definition after flip (DSL + Python both rewritten with a leading minus).
            if flip_patch and flip_patch.get("python_code"):
                py_code = flip_patch["python_code"]
            elif job.entry.get("python_code"):
                py_code = str(job.entry["python_code"])

        release_memory(daily_values)
        topk_summary = analysis.pop("_topk_summary", topk_summary) or summarize_backtest(backtest_rows)
        ls_backtest_rows = analysis.pop("_ls_backtest_rows", ls_backtest_rows) or []
        ls_summary = analysis.pop("_ls_summary", ls_summary) or {}
        analysis.pop("_ls_backtest_id", None)

        report_path = report_dir / f"{name}.html"
        render_factor_report(
            factor_name=name,
            dsl=dsl_used,
            analysis=analysis,
            backtest_rows=backtest_rows,
            out_path=report_path,
            eval_mode=eval_mode,
            materialize_meta=meta,
            ls_backtest_rows=ls_backtest_rows,
            topk_summary=topk_summary,
            ls_summary=ls_summary,
            python_code=py_code,
            work_dir=Path(job.work_dir),
            engine=engine,
            eval_route=actual_eval_route,
        )
        row = {
            "factor_name": name,
            "report": report_path.name,
            "engine": engine,
            "eval_route": actual_eval_route,
            "eval_mode": eval_mode,
            "ic_sign_flipped": bool(job.entry.get("ic_sign_flipped") or (flip_patch or {}).get("ic_sign_flipped")),
            "mean_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
            "mean_rank_ic": _json_float(analysis.get("mean_rank_ic", analysis.get("mean_ic"))),
            "icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
            "rank_icir": _json_float(analysis.get("rank_icir", analysis.get("icir"))),
            "ic_positive_ratio": _json_float(
                analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
            ),
            "rank_ic_positive_ratio": _json_float(
                analysis.get("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
            ),
            "coverage": _json_float(analysis.get("coverage")),
            "long_short_sharpe": _json_float(analysis.get("long_short_sharpe")),
            "long_short_return": _json_float(analysis.get("long_short_return")),
            "backtest_total_ret": _json_float(topk_summary.get("total_return")),
            "backtest_ann_ret": _json_float(topk_summary.get("annualized_return")),
            "backtest_sharpe": _json_float(topk_summary.get("sharpe")),
            "backtest_max_drawdown": _json_float(topk_summary.get("max_drawdown")),
            "backtest_win_rate": _json_float(topk_summary.get("win_rate")),
            "backtest_avg_turnover": _json_float(topk_summary.get("avg_turnover")),
            "rows": int(rows),
            "backtest_id": backtest_id,
            "return_kind": str(analysis.get("return_kind", "")),
        }
        print(
            f"[{job.index}/{job.total}] ok {name} rows={rows} "
            f"rank_ic={analysis.get('mean_rank_ic', analysis.get('mean_ic', 'n/a'))} "
            f"topk_sharpe={topk_summary.get('sharpe', 'n/a')} mode={eval_mode}"
        )
        return {"ok": True, "name": name, "row": row, "error": None, "flip_patch": flip_patch}
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[{job.index}/{job.total}] FAIL {name}: {exc}")
        return {"ok": False, "name": name, "row": None, "error": str(exc), "traceback": tb}
    finally:
        release_memory()


def _json_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_factor_result(
    *,
    result: dict[str, Any],
    progress: dict[str, Any],
    index_rows: list[dict[str, Any]],
    progress_path: Path,
    report_dir: Path,
    summary_meta: dict[str, Any],
    lock: threading.Lock,
) -> None:
    name = result["name"]
    with lock:
        if result["ok"]:
            _upsert_index_row(index_rows, result["row"])
            if name not in progress.setdefault("completed", []):
                progress["completed"].append(name)
            progress.get("failed", {}).pop(name, None)
            flip_patch = result.get("flip_patch")
            if flip_patch:
                fs = progress.setdefault("flip_state", {})
                fs[name] = {**fs.get(name, {}), **flip_patch}
        else:
            progress.setdefault("failed", {})[name] = result["error"] or "unknown error"
        progress["index_rows"] = index_rows
        _save_progress(progress_path, progress)
        summary_meta["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_reports(
            index_rows=index_rows,
            report_dir=report_dir,
            meta=summary_meta,
            failed=progress.get("failed", {}),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Production CogAlpha batch 2019-2026")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--bootstrap-lqtp", action="store_true", help="fetch StockDailyBar from LQTP if missing")
    parser.add_argument("--symbol-chunk", type=int, default=250)
    parser.add_argument("--python-workers", type=int, default=2)
    parser.add_argument("--min-mem-gb", type=float, default=3.0)
    parser.add_argument("--mem-reserve-gb", type=float, default=DEFAULT_MEM_RESERVE_GB)
    parser.add_argument(
        "--flip-negative-ic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if mean_ic<0, permanently rewrite factor as -(defn) (DSL+Python+values) and re-test once (default on)",
    )
    parser.add_argument("--max-total-parallel", type=int, default=DEFAULT_MAX_TOTAL_PARALLEL)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=None, help="factor names to skip this run")
    parser.add_argument(
        "--skip-lookahead",
        action="store_true",
        help="defer LOOKAHEAD_DEFERRED_FACTORS (12 rank look-ahead fixes) until later",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true", help="ignore production_progress.json and rerun all factors")
    parser.add_argument("--token-refresh-minutes", type=float, default=20.0, help="refresh LQTP access token every N minutes")
    parser.add_argument(
        "--parallel-lqtp",
        type=int,
        default=3,
        help="max concurrent lqtp_dsl RunFactor jobs (default 3; each still pulls ~5G locally)",
    )
    parser.add_argument(
        "--parallel-local",
        type=int,
        default=0,
        help="max concurrent local-engine jobs (0=auto from available memory)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="deprecated alias for --parallel-local when --parallel-local is 0",
    )
    parser.add_argument(
        "--local-mem-gb",
        type=float,
        default=LOCAL_MATERIALIZE_MEM_GB,
        help="estimated RAM per local materialize job for auto parallel-local sizing",
    )
    parser.add_argument(
        "--lqtp-mem-gb",
        type=float,
        default=LQTP_JOB_MEM_GB,
        help="estimated RAM per lqtp_dsl job (RunFactor response + backtest)",
    )
    args = parser.parse_args()
    if args.parallel_lqtp < 1:
        raise SystemExit("--parallel-lqtp must be >= 1")
    if args.parallel_local < 0:
        raise SystemExit("--parallel-local must be >= 0")

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
            "install COS mirror or run with --bootstrap-lqtp"
        )

    catalog: list[dict[str, Any]] = json.loads(catalog_path.read_text(encoding="utf-8"))
    progress = (
        {"completed": [], "failed": {}, "skipped": [], "index_rows": [], "flip_state": {}}
        if args.fresh
        else (_load_progress(progress_path) if args.resume else {"completed": [], "failed": {}, "skipped": [], "index_rows": [], "flip_state": {}})
    )
    index_rows: list[dict[str, Any]] = list(progress.get("index_rows", []))
    python_map = {r["function_name"]: r["python_code"] for r in json.loads(parsed.read_text(encoding="utf-8"))}
    _apply_flip_state_to_catalog(catalog, progress.get("flip_state", {}), python_map)
    refresh_q = _ensure_flipped_python_definitions(
        flip_state=progress.setdefault("flip_state", {}),
        python_map=python_map,
        progress=progress,
    )
    queued = _queue_negative_ic_retests(
        catalog=catalog,
        index_rows=index_rows,
        progress=progress,
        lake=lake,
        flip_negative_ic=args.flip_negative_ic,
        python_map=python_map,
    )
    if queued or refresh_q:
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_progress(progress_path, progress)
    route_stats: dict[str, int] = {}
    for entry in catalog:
        route = _eval_route(entry)
        route_stats[route] = route_stats.get(route, 0) + 1
    print(
        "eval routes: "
        + " ".join(f"{k}={v}" for k, v in sorted(route_stats.items()))
    )
    _write_route_summary(catalog, work / "eval_routes.json")
    catalog_total = len(catalog)
    if args.only:
        only = set(args.only)
        catalog = [x for x in catalog if x["function_name"] in only]

    skip_names: set[str] = set(progress.get("skipped", []))
    if args.skip:
        skip_names.update(args.skip)
    if args.skip_lookahead:
        skip_names.update(LOOKAHEAD_DEFERRED_FACTORS)
    if skip_names:
        progress["skipped"] = sorted(skip_names)
        _save_progress(progress_path, progress)
        print(f"skip {len(skip_names)} factors: " + ", ".join(sorted(skip_names)[:6]) + (" ..." if len(skip_names) > 6 else ""))

    done = set(progress.get("completed", []))

    auth = LqtpTokenManager.login(
        args.server,
        args.username,
        args.password,
        refresh_interval_seconds=int(args.token_refresh_minutes * 60),
    )
    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    lqtp_universe = fetch_lqtp_universe(token=auth.token, begin_date=begin_i, end_date=end_i, server=args.server)
    close_returns_cache = work / "lqtp_close_returns_cache.parquet"
    open_returns_cache = work / "lqtp_open_returns_cache.parquet"
    fetch_lqtp_close_returns(
        token=auth.token,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
        cache_path=close_returns_cache,
    )
    backtest_symbols = _backtest_symbol_set(
        token=auth.token,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
        returns_cache=open_returns_cache,
        lqtp_universe=lqtp_universe,
    )
    lqtp_symbols = sorted(lqtp_universe)
    data_cfg = _ashare_data_source(
        args.start,
        args.end,
        instrument_filter=lqtp_symbols,
    )
    print(f"local materialize instrument_filter={len(lqtp_symbols)} (LQTP universe)")
    summary_meta = {
        "date_range": [args.start, args.end],
        "catalog_total": catalog_total,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_reports(
        index_rows=index_rows,
        report_dir=report_dir,
        meta=summary_meta,
        failed=progress.get("failed", {}),
    )

    t0 = time.time()
    pending: list[FactorJob] = []
    for i, entry in enumerate(catalog, 1):
        name = entry["function_name"]
        if name in skip_names:
            print(f"[{i}/{len(catalog)}] skip deferred {name}")
            continue
        if name in done:
            print(f"[{i}/{len(catalog)}] skip done {name}")
            continue
        pending.append(
            FactorJob(
                index=i,
                total=len(catalog),
                entry=entry,
                python_code=python_map.get(name, ""),
                work_dir=str(work),
                start=args.start,
                end=args.end,
                server=args.server,
                username=args.username,
                password=args.password,
                begin_i=begin_i,
                end_i=end_i,
                symbol_chunk=args.symbol_chunk,
                python_workers=args.python_workers,
                skip_eval=args.skip_eval,
                token_refresh_minutes=args.token_refresh_minutes,
                min_mem_gb=args.min_mem_gb,
                mem_reserve_gb=args.mem_reserve_gb,
                lqtp_job_mem_gb=args.lqtp_mem_gb,
                local_materialize_mem_gb=args.local_mem_gb,
                local_reuse_mem_gb=LOCAL_REUSE_MEM_GB,
                lqtp_universe=tuple(sorted(lqtp_universe)),
                backtest_symbols=tuple(sorted(backtest_symbols)),
                data_cfg=data_cfg,
                flip_negative_ic=args.flip_negative_ic,
            )
        )

    pending_lqtp = [j for j in pending if _eval_route(j.entry) == "lqtp_dsl"]
    pending_local = [j for j in pending if _eval_route(j.entry) != "lqtp_dsl"]
    mem_avail = read_mem_available_gb()
    lqtp_workers = estimate_lqtp_parallel_workers(
        pending_count=len(pending_lqtp),
        mem_avail_gb=mem_avail,
        mem_reserve_gb=args.mem_reserve_gb,
        mem_per_job_gb=args.lqtp_mem_gb,
        default=args.parallel_lqtp,
        max_workers=min(args.parallel_lqtp, 4),
    )
    if args.parallel_local > 0:
        local_workers = args.parallel_local
    elif args.parallel_workers is not None and args.parallel_workers > 0:
        local_workers = args.parallel_workers
    else:
        local_workers = estimate_local_parallel_workers(
            mem_avail_gb=mem_avail,
            mem_reserve_gb=args.mem_reserve_gb,
            n_materialize=sum(1 for j in pending_local if _local_needs_materialize(j)),
            n_reuse=sum(1 for j in pending_local if not _local_needs_materialize(j)),
            mem_per_materialize_gb=args.local_mem_gb,
            mem_per_reuse_gb=LOCAL_REUSE_MEM_GB,
            max_workers=3,
        )
    if pending_local and local_workers < 1:
        local_workers = 1
    lqtp_workers, local_workers = cap_combined_parallel(
        lqtp_workers=lqtp_workers,
        local_workers=local_workers,
        has_lqtp=bool(pending_lqtp),
        has_local=bool(pending_local),
        max_total=args.max_total_parallel,
    )

    print(
        f"LQTP universe={len(lqtp_universe)} backtest_symbols={len(backtest_symbols)} "
        f"pending_lqtp={len(pending_lqtp)} pending_local={len(pending_local)} "
        f"parallel_lqtp={lqtp_workers} parallel_local={local_workers} "
        f"mem_avail={mem_avail:.1f}G reserve={args.mem_reserve_gb:.1f}G max_total={args.max_total_parallel}"
    )

    if not pending:
        print("nothing to do")
    elif lqtp_workers <= 1 and local_workers <= 1 and len(pending) == 1:
        progress_lock = threading.Lock()
        result = _process_one_factor(pending[0])
        _apply_factor_result(
            result=result,
            progress=progress,
            index_rows=index_rows,
            progress_path=progress_path,
            report_dir=report_dir,
            summary_meta=summary_meta,
            lock=progress_lock,
        )
        if not result["ok"] and result.get("traceback"):
            print(result["traceback"])
    elif not pending_lqtp:
        _run_parallel_batch(
            pending=pending_local,
            progress=progress,
            index_rows=index_rows,
            progress_path=progress_path,
            report_dir=report_dir,
            summary_meta=summary_meta,
            lqtp_workers=0,
            local_workers=local_workers,
            mem_reserve_gb=args.mem_reserve_gb,
        )
    elif not pending_local:
        _run_parallel_batch(
            pending=pending_lqtp,
            progress=progress,
            index_rows=index_rows,
            progress_path=progress_path,
            report_dir=report_dir,
            summary_meta=summary_meta,
            lqtp_workers=lqtp_workers,
            local_workers=0,
            mem_reserve_gb=args.mem_reserve_gb,
        )
    else:
        _run_parallel_batch(
            pending=pending,
            progress=progress,
            index_rows=index_rows,
            progress_path=progress_path,
            report_dir=report_dir,
            summary_meta=summary_meta,
            lqtp_workers=lqtp_workers,
            local_workers=local_workers,
            mem_reserve_gb=args.mem_reserve_gb,
        )

    summary = {
        "work_dir": str(work),
        "date_range": [args.start, args.end],
        "elapsed_sec": time.time() - t0,
        "completed": progress.get("completed", []),
        "failed": progress.get("failed", {}),
        "reports_index": str(report_dir / "index.html"),
        "production_summary": str(report_dir / "production_summary.html"),
        "index_rows": index_rows,
    }
    (work / "production_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": len(progress.get("completed", [])),
                "fail": len(progress.get("failed", {})),
                "summary_html": str(report_dir / "production_summary.html"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not progress.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
