#!/usr/bin/env python3
"""Evaluate factor_engine materialized values on LQTP (values upload only)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lqtp-python-grpc-examples" / "protos"))

import Factor_pb2  # noqa: E402

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    analysis_to_dict,
    analyze_factor_values,
    factor_values_to_long_df,
    run_factor_formula,
)

LQTP_DAILY_RETURN_FORMULA = "close / delay(close, 1) - 1"


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return float("nan")
    return float(x.rank(method="average").corr(y.rank(method="average")))


def analyze_values_with_lqtp_returns(
    *,
    factor_long: pd.DataFrame,
    returns_long: pd.DataFrame,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Compute FactorAnalysis-shaped metrics from uploaded values + LQTP returns."""
    factor = factor_long.copy()
    rets = returns_long.copy()
    factor["trade_date"] = factor["trade_date"].astype(int)
    rets["trade_date"] = rets["trade_date"].astype(int)

    dates = sorted(factor["trade_date"].unique())
    if len(dates) < 2:
        raise RuntimeError("need at least 2 trade dates for factor analysis")

    daily_ic: list[float] = []
    daily_ls: list[float] = []
    trade_dates_out: list[int] = []
    sample_counts: list[int] = []
    coverages: list[float] = []
    group_pnls: list[dict[str, Any]] = [{f"group": g + 1, "pnl": []} for g in range(n_groups)]

    universe_sizes: list[int] = []
    for idx, td in enumerate(dates[:-1]):
        next_td = dates[idx + 1]
        f_day = factor[factor["trade_date"] == td].dropna(subset=["value"])
        r_day = rets[rets["trade_date"] == next_td].dropna(subset=["value"])
        merged = f_day.merge(
            r_day[["symbol", "value"]],
            on="symbol",
            how="inner",
            suffixes=("_factor", "_ret"),
        )
        if merged.empty:
            continue

        ic = _spearman(merged["value_factor"], merged["value_ret"])
        daily_ic.append(ic)
        trade_dates_out.append(int(td))

        n = len(merged)
        sample_counts.append(n)
        universe_sizes.append(max(n, len(r_day)))
        coverages.append(n / max(len(r_day), 1))

        try:
            merged["group"] = pd.qcut(
                merged["value_factor"].rank(method="first"),
                n_groups,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            merged["group"] = 0

        grp_ret = merged.groupby("group", observed=True)["value_ret"].mean()
        bottom = float(grp_ret.min()) if not grp_ret.empty else 0.0
        top = float(grp_ret.max()) if not grp_ret.empty else 0.0
        daily_ls.append(top - bottom)

        for g in range(n_groups):
            g_ret = float(grp_ret.get(g, 0.0))
            prev = group_pnls[g]["pnl"][-1] if group_pnls[g]["pnl"] else 0.0
            group_pnls[g]["pnl"].append(prev + g_ret)

    if not daily_ic:
        raise RuntimeError("no overlapping factor/return rows for analysis")

    ic_arr = np.array(daily_ic, dtype=float)
    ls_arr = np.array(daily_ls, dtype=float)
    mean_ic = float(np.nanmean(ic_arr))
    std_ic = float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    ic_pos = float(np.mean(ic_arr > 0))

    group_mean_returns: list[float] = []
    for item in group_pnls:
        pnl = item["pnl"]
        group_mean_returns.append(float(pnl[-1] / len(pnl)) if pnl else 0.0)

    ls_std = float(np.nanstd(ls_arr, ddof=1)) if len(ls_arr) > 1 else 0.0
    ls_mean = float(np.nanmean(ls_arr))
    ls_sharpe = ls_mean / ls_std * np.sqrt(252) if ls_std > 1e-12 else 0.0

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "ic_positive_ratio": ic_pos,
        "coverage": float(np.mean(coverages)),
        "long_short_return": float(np.nansum(ls_arr)),
        "long_short_sharpe": ls_sharpe,
        "daily_ic": daily_ic,
        "daily_ls_returns": daily_ls,
        "trade_dates": trade_dates_out,
        "quote_times": [0] * len(trade_dates_out),
        "sample_counts": sample_counts,
        "coverages": coverages,
        "group_mean_returns": group_mean_returns,
        "group_pnls": group_pnls,
    }


def fetch_lqtp_daily_returns(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch next-day return proxy from LQTP market data (not our factor DSL)."""
    if cache_path is not None and cache_path.exists():
        frame = pd.read_parquet(cache_path)
        frame["trade_date"] = frame["trade_date"].astype(int)
        return frame

    def _year_ranges(b: int, e: int) -> list[tuple[int, int]]:
        sy, ey = b // 10000, e // 10000
        out: list[tuple[int, int]] = []
        for year in range(sy, ey + 1):
            y0 = year * 10000 + 101
            y1 = year * 10000 + 1231
            lo = max(b, y0)
            hi = min(e, y1)
            if lo <= hi:
                out.append((lo, hi))
        return out

    parts: list[pd.DataFrame] = []
    for lo, hi in _year_ranges(begin_date, end_date):
        resp = run_factor_formula(
            token=token,
            formula=LQTP_DAILY_RETURN_FORMULA,
            begin_date=lo,
            end_date=hi,
            warmup=1,
            analyze=False,
            server=server,
        )
        if resp.error:
            raise RuntimeError(resp.error)
        parts.append(factor_values_to_long_df(resp.values))

    if not parts:
        raise RuntimeError("no LQTP returns fetched")
    frame = pd.concat(parts, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def evaluate_uploaded_values(
    *,
    token: str,
    daily_values: list[Factor_pb2.FactorDailyValues],
    begin_date: int,
    end_date: int,
    server: str,
    returns_cache: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Evaluate factor_engine materialized values on LQTP.

    Primary: AnalyzeFactor(daily_values) — official upload path.
    Fallback: same metrics computed locally using LQTP returns only (never re-runs our DSL).
    """
    try:
        resp = analyze_factor_values(
            token=token,
            daily_values=daily_values,
            begin_date=begin_date,
            end_date=end_date,
            server=server,
        )
        if resp.error:
            raise RuntimeError(resp.error)
        return analysis_to_dict(resp.analysis), "analyze_factor_upload"
    except grpc.RpcError as error:
        if error.code() != grpc.StatusCode.UNIMPLEMENTED:
            raise

    factor_long = factor_values_to_long_df(daily_values)
    returns_long = fetch_lqtp_daily_returns(
        token=token,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=returns_cache,
    )
    analysis = analyze_values_with_lqtp_returns(
        factor_long=factor_long,
        returns_long=returns_long,
    )
    return analysis, "local_values_lqtp_returns"
