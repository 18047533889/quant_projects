#!/usr/bin/env python3
"""Evaluate factor values: RankIC / groups / long-short with no look-ahead.

Signal timing:
  - Factor is known only after close on day T.
  - RankIC / groups / LS: standard close-to-close forward return close(T+1)/close(T)-1.
  - TopK / platform backtest (separate): OPEN execution, signal T → trade open(T+1).
"""
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

# open[D] / open[D-1] - 1  →  labeled on date D (used for OPEN backtest symbol filter)
LQTP_OPEN_TO_OPEN_RETURN = "open / delay(open, 1) - 1"
# close[D] / close[D-1] - 1  →  labeled on date D (standard RankIC forward return)
LQTP_CLOSE_TO_CLOSE_RETURN = "close / delay(close, 1) - 1"
LQTP_DAILY_RETURN_FORMULA = LQTP_CLOSE_TO_CLOSE_RETURN

# Round-trip commission each side, percent points: 0.01 = 0.01%
DEFAULT_COMMISSION_BPS = 0.01


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return float("nan")
    return float(x.rank(method="average").corr(y.rank(method="average")))


def _next_map(dates: list[int]) -> dict[int, int]:
    """Map each date to the next trading date in *dates*."""
    ordered = sorted(dates)
    return {ordered[i]: ordered[i + 1] for i in range(len(ordered) - 1)}


def analyze_values_with_lqtp_returns(
    *,
    factor_long: pd.DataFrame,
    returns_long: pd.DataFrame,
    n_groups: int = 10,
    commission_buy: float = DEFAULT_COMMISSION_BPS,
    commission_sell: float = DEFAULT_COMMISSION_BPS,
    return_mode: str = "close_to_close",
) -> dict[str, Any]:
    """Compute RankIC / decile groups / G10−G1 long-short.

    *returns_long* daily return series labeled on exit date D.

    return_mode ``close_to_close`` (default, standard RankIC):
      factor on T vs return on T+1 where value = close(T+1)/close(T)-1.

    return_mode ``open_to_open`` (legacy research path):
      factor on T vs return on T+2 where value = open(T+2)/open(T+1)-1.
    """
    factor = factor_long.copy()
    rets = returns_long.copy()
    factor["trade_date"] = factor["trade_date"].astype(int)
    rets["trade_date"] = rets["trade_date"].astype(int)
    factor["symbol"] = factor["symbol"].astype(str)
    rets["symbol"] = rets["symbol"].astype(str)

    cal = sorted(set(factor["trade_date"].unique()) | set(rets["trade_date"].unique()))
    nxt = _next_map(cal)
    if return_mode == "open_to_open":
        forward_map = {d: nxt[nxt[d]] for d in cal if d in nxt and nxt[d] in nxt}
        min_dates = 3
        min_dates_msg = "need at least 3 trade dates for open-to-open RankIC (T → T+2)"
        return_kind = "open_to_open_T_plus_2"
        signal_lag_note = (
            "Factor after close T → open(T+1) to open(T+2) return; "
            "LS groups use same return window"
        )
    else:
        forward_map = nxt
        min_dates = 2
        min_dates_msg = "need at least 2 trade dates for close-to-close RankIC (T → T+1)"
        return_kind = "close_to_close_T_plus_1"
        signal_lag_note = (
            "Factor after close T → forward return close(T+1)/close(T)-1 "
            "(standard RankIC); TopK backtest uses OPEN execution separately"
        )

    factor_dates = sorted(factor["trade_date"].unique())
    if len(factor_dates) < min_dates:
        raise RuntimeError(min_dates_msg)

    daily_rank_ic: list[float] = []
    daily_ls_gross: list[float] = []
    daily_ls_net: list[float] = []
    trade_dates_out: list[int] = []
    sample_counts: list[int] = []
    coverages: list[float] = []
    group_pnls: list[dict[str, Any]] = [{"group": g + 1, "pnl": []} for g in range(n_groups)]
    group_equity = [1.0] * n_groups

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    roundtrip = float(commission_buy) + float(commission_sell)

    rets_by_date = {int(td): g for td, g in rets.groupby("trade_date", sort=False)}

    for td in factor_dates:
        exit_td = forward_map.get(int(td))
        if exit_td is None:
            continue
        f_day = factor[factor["trade_date"] == td].dropna(subset=["value"])
        r_day = rets_by_date.get(int(exit_td))
        if r_day is None or r_day.empty:
            continue
        r_day = r_day.dropna(subset=["value"])
        merged = f_day.merge(
            r_day[["symbol", "value"]],
            on="symbol",
            how="inner",
            suffixes=("_factor", "_ret"),
        )
        if len(merged) < max(30, n_groups * 3):
            continue

        rank_ic = _spearman(merged["value_factor"], merged["value_ret"])
        daily_rank_ic.append(rank_ic)
        trade_dates_out.append(int(td))
        n = len(merged)
        sample_counts.append(n)
        # coverage vs return universe that day
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
        # Fixed deciles: G1 = lowest factor, G10 = highest (labels 0..n-1)
        g_lo = 0
        g_hi = int(merged["group"].max())
        bottom = float(grp_ret.get(g_lo, 0.0))
        top = float(grp_ret.get(g_hi, 0.0))
        ls_gross = top - bottom
        daily_ls_gross.append(ls_gross)

        long_set = set(merged.loc[merged["group"] == g_hi, "symbol"].astype(str))
        short_set = set(merged.loc[merged["group"] == g_lo, "symbol"].astype(str))
        # One-way turnover of each book (0..1); charge buy+sell on changed names
        if prev_long or prev_short:
            long_to = 1.0 - (len(long_set & prev_long) / max(len(long_set), 1))
            short_to = 1.0 - (len(short_set & prev_short) / max(len(short_set), 1))
        else:
            long_to = 1.0
            short_to = 1.0
        # Long and short each pay round-trip on turned portion
        cost = (long_to + short_to) * roundtrip / 100.0  # bps → fraction (0.01 means 0.01%)
        # commission_buy unit is percent: 0.01 = 0.01%, so /100 → 0.0001 = 1bp
        daily_ls_net.append(ls_gross - cost)
        prev_long, prev_short = long_set, short_set

        for g in range(n_groups):
            g_ret = float(grp_ret.get(g, 0.0))
            group_equity[g] *= 1.0 + g_ret
            group_pnls[g]["pnl"].append(group_equity[g] - 1.0)

    if not daily_rank_ic:
        raise RuntimeError("no overlapping factor/return rows for RankIC")

    ic_arr = np.array(daily_rank_ic, dtype=float)
    ls_net = np.array(daily_ls_net, dtype=float)
    ls_gross = np.array(daily_ls_gross, dtype=float)
    mean_rank_ic = float(np.nanmean(ic_arr))
    std_rank_ic = float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
    rank_icir = mean_rank_ic / std_rank_ic if std_rank_ic > 1e-12 else 0.0
    rank_ic_pos = float(np.mean(ic_arr > 0))

    group_mean_returns = []
    for item in group_pnls:
        pnl = item["pnl"]
        # average daily group return approximation from equity
        if len(pnl) >= 2:
            eq = np.array([1.0] + [1.0 + x for x in pnl])
            daily = np.diff(eq) / eq[:-1]
            group_mean_returns.append(float(np.nanmean(daily)))
        else:
            group_mean_returns.append(0.0)

    def _sharpe(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr, ddof=1))
        return m / s * np.sqrt(252) if s > 1e-12 else 0.0

    # Cumulative LS via cumprod of net daily LS (real portfolio-style)
    ls_cum = float(np.prod(1.0 + ls_net) - 1.0) if len(ls_net) else 0.0

    return {
        # Canonical RankIC names
        "mean_rank_ic": mean_rank_ic,
        "std_rank_ic": std_rank_ic,
        "rank_icir": rank_icir,
        "rank_ic_positive_ratio": rank_ic_pos,
        # Backward-compatible aliases (old report fields)
        "mean_ic": mean_rank_ic,
        "std_ic": std_rank_ic,
        "icir": rank_icir,
        "ic_positive_ratio": rank_ic_pos,
        "coverage": float(np.mean(coverages)) if coverages else 0.0,
        "long_short_return": ls_cum,
        "long_short_return_sum": float(np.nansum(ls_net)),
        "long_short_sharpe": _sharpe(ls_net),
        "long_short_return_gross_sum": float(np.nansum(ls_gross)),
        "long_short_sharpe_gross": _sharpe(ls_gross),
        "daily_ic": daily_rank_ic,
        "daily_rank_ic": daily_rank_ic,
        "daily_ls_returns": ls_net.tolist(),
        "daily_ls_gross": ls_gross.tolist(),
        "trade_dates": trade_dates_out,
        "quote_times": [0] * len(trade_dates_out),
        "sample_counts": sample_counts,
        "coverages": coverages,
        "group_mean_returns": group_mean_returns,
        "group_pnls": group_pnls,
        "return_kind": return_kind,
        "return_mode": return_mode,
        "signal_lag_note": signal_lag_note,
        "commission_buy": commission_buy,
        "commission_sell": commission_sell,
        "n_groups": n_groups,
    }


def _fetch_lqtp_returns_by_formula(
    *,
    token: str,
    formula: str,
    begin_date: int,
    end_date: int,
    server: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
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
            formula=formula,
            begin_date=lo,
            end_date=hi,
            warmup=2,
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


def fetch_lqtp_close_returns(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch close-to-close returns (value on D = close[D]/close[D-1]-1)."""
    return _fetch_lqtp_returns_by_formula(
        token=token,
        formula=LQTP_CLOSE_TO_CLOSE_RETURN,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=cache_path,
    )


def fetch_lqtp_open_returns(
    *,
    token: str,
    begin_date: int,
    end_date: int,
    server: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch open-to-open returns (value on D = open[D]/open[D-1]-1)."""
    return _fetch_lqtp_returns_by_formula(
        token=token,
        formula=LQTP_OPEN_TO_OPEN_RETURN,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=cache_path,
    )


# Standard daily return for RankIC
fetch_lqtp_daily_returns = fetch_lqtp_close_returns


def evaluate_lqtp_formula(
    *,
    token: str,
    formula: str,
    begin_date: int,
    end_date: int,
    server: str,
    warmup: int = 1,
    factor_name: str = "",
) -> tuple[dict[str, Any], str, list[Factor_pb2.FactorDailyValues]]:
    """Run LQTP-native DSL on platform; return analysis + daily values for backtest."""
    resp = run_factor_formula(
        token=token,
        formula=formula,
        begin_date=begin_date,
        end_date=end_date,
        warmup=warmup,
        analyze=True,
        server=server,
        factor_name="",
    )
    if resp.error:
        raise RuntimeError(resp.error)
    if not resp.analysis:
        raise RuntimeError("RunFactor returned no analysis")
    analysis = analysis_to_dict(resp.analysis)
    # Platform analysis uses its own timing; alias RankIC names for report consistency.
    analysis.setdefault("mean_rank_ic", analysis.get("mean_ic"))
    analysis.setdefault("std_rank_ic", analysis.get("std_ic"))
    analysis.setdefault("rank_icir", analysis.get("icir"))
    analysis.setdefault("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
    analysis["daily_rank_ic"] = analysis.get("daily_ic", [])
    analysis["return_kind"] = "lqtp_platform_analyze"
    analysis["signal_lag_note"] = (
        "LQTP RunFactor(analyze=True) platform metrics (platform-defined timing)"
    )
    return analysis, "lqtp_run_factor", list(resp.values)


def evaluate_uploaded_values(
    *,
    token: str,
    daily_values: list[Factor_pb2.FactorDailyValues],
    begin_date: int,
    end_date: int,
    server: str,
    returns_cache: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Evaluate materialized values.

    Prefer LQTP AnalyzeFactor for small payloads; production-size payloads use local
    RankIC with LQTP close-to-close returns (standard cross-vendor comparison).
    """
    total_rows = sum(len(point.values) for point in daily_values)
    use_local = total_rows > 400_000

    if not use_local:
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
            analysis = analysis_to_dict(resp.analysis)
            analysis.setdefault("mean_rank_ic", analysis.get("mean_ic"))
            analysis.setdefault("std_rank_ic", analysis.get("std_ic"))
            analysis.setdefault("rank_icir", analysis.get("icir"))
            analysis.setdefault("rank_ic_positive_ratio", analysis.get("ic_positive_ratio"))
            analysis["daily_rank_ic"] = analysis.get("daily_ic", [])
            analysis["return_kind"] = "lqtp_analyze_factor"
            return analysis, "analyze_factor_upload"
        except grpc.RpcError as error:
            if error.code() not in {
                grpc.StatusCode.UNIMPLEMENTED,
                grpc.StatusCode.OUT_OF_RANGE,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            }:
                raise

    factor_long = factor_values_to_long_df(daily_values)
    close_cache = returns_cache
    if returns_cache is not None and "close" not in returns_cache.name:
        close_cache = returns_cache.with_name("lqtp_close_returns_cache.parquet")
    returns_long = fetch_lqtp_close_returns(
        token=token,
        begin_date=begin_date,
        end_date=end_date,
        server=server,
        cache_path=close_cache,
    )
    analysis = analyze_values_with_lqtp_returns(
        factor_long=factor_long,
        returns_long=returns_long,
        return_mode="close_to_close",
    )
    return analysis, "local_values_lqtp_returns"
