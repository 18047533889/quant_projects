from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EvalConfig


# ---------------------------------------------------------------------------
# Input loading helpers
# ---------------------------------------------------------------------------
def load_factor_series(factor_input, cfg: EvalConfig) -> pd.Series:
    """Convert factor input (Series or DataFrame) to a clean pd.Series with
    DatetimeIndex.

    Accepts:
      - pd.DataFrame with columns [cfg.factor_timestamp_col, cfg.factor_col]
      - pd.Series with DatetimeIndex
    """
    if isinstance(factor_input, pd.DataFrame):
        ts_col = cfg.factor_timestamp_col
        val_col = cfg.factor_col
        df = factor_input[[ts_col, val_col]].copy()
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.drop_duplicates(subset=[ts_col], keep="last")
        df = df.sort_values(ts_col)
        return pd.Series(
            df[val_col].to_numpy(), index=df[ts_col], name="factor",
        )
    elif isinstance(factor_input, pd.Series):
        s = factor_input.copy()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = "factor"
        return s
    else:
        raise TypeError(
            f"factor_input must be pd.Series or pd.DataFrame, got {type(factor_input)}"
        )


def load_vwap_series(market_df: pd.DataFrame, cfg: EvalConfig) -> pd.Series:
    """Extract VWAP series from market DataFrame."""
    ts_col = cfg.timestamp_col
    vwap_col = cfg.vwap_col
    df = market_df[[ts_col, vwap_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.sort_values(ts_col)
    return pd.Series(df[vwap_col].to_numpy(), index=df[ts_col], name="vwap")


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------
def align_on_overlap_reindex(
    factor: pd.Series, vwap: pd.Series,
) -> pd.DataFrame:
    """Align factor and vwap on their overlapping time range using union
    index with reindex."""
    if factor.empty or vwap.empty:
        raise ValueError("Input factor/vwap series is empty.")

    overlap_start = max(factor.index.min(), vwap.index.min())
    overlap_end = min(factor.index.max(), vwap.index.max())
    if overlap_start > overlap_end:
        raise ValueError(
            "No overlapping timestamp range between factor and vwap."
        )

    factor_overlap = factor.loc[
        (factor.index >= overlap_start) & (factor.index <= overlap_end)
    ]
    vwap_overlap = vwap.loc[
        (vwap.index >= overlap_start) & (vwap.index <= overlap_end)
    ]

    overlap_index = factor_overlap.index.union(vwap_overlap.index).sort_values()

    aligned = pd.DataFrame(index=overlap_index)
    aligned["factor_raw"] = factor_overlap.reindex(overlap_index)
    aligned["vwap"] = vwap_overlap.reindex(overlap_index)
    aligned.index.name = "timestamp"
    return aligned


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score normalisation (ddof=0 for std)."""
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    z = (series - rolling_mean) / rolling_std
    return z.where(rolling_std > 0)


def make_forward_vwap_return(vwap: pd.Series, horizon: int) -> pd.Series:
    """Forward VWAP return: factor at t → return over [t+1, t+1+N]."""
    return vwap.shift(-(horizon + 1)) / vwap.shift(-1) - 1.0


def eligibility_mask_by_day(
    df: pd.DataFrame, horizon: int,
) -> pd.Series:
    """Mask out tail bars per day where horizon exceeds remaining bars."""
    g = df.groupby("date")
    pos_in_day = g.cumcount()
    day_size = g["date"].transform("size")
    return pos_in_day < (day_size - horizon)


def winsorize_by_day(
    series: pd.Series, date_index: pd.Series, q: float,
) -> pd.Series:
    """Per-day winsorisation at quantile *q* and *1-q*.

    Optimised: computes quantiles via ``groupby.quantile`` (C-level)
    then maps back, avoiding per-group Python lambdas.
    """
    grouped = series.groupby(date_index)
    lo_q = grouped.quantile(q)
    hi_q = grouped.quantile(1.0 - q)
    lower = date_index.map(lo_q)
    upper = date_index.map(hi_q)
    return series.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# High-level prepare functions
# ---------------------------------------------------------------------------
def prepare_base(
    factor_input, market_df: pd.DataFrame, cfg: EvalConfig,
) -> pd.DataFrame:
    """Common preprocessing shared by all downstream modules.

    Steps:
      1. Extract factor series and vwap series from inputs
      2. Align on overlapping timestamps
      3. Apply rolling z-score to raw factor

    Returns a DataFrame with columns:
      factor_raw, vwap, factor_z, date
    Index: timestamp (DatetimeIndex)
    """
    factor = load_factor_series(factor_input, cfg)
    vwap = load_vwap_series(market_df, cfg)

    aligned = align_on_overlap_reindex(factor, vwap)
    aligned["factor_z"] = rolling_zscore(
        aligned["factor_raw"], window=cfg.zscore_window,
    )
    aligned["date"] = aligned.index.date
    return aligned


def prepare_horizon(
    base_df: pd.DataFrame, horizon: int, cfg: EvalConfig,
) -> pd.DataFrame:
    """Per-horizon preprocessing for the full evaluation path.

    Steps:
      1. Compute forward VWAP return for this horizon
      2. Apply eligibility mask (exclude day-tail bars)
      3. Winsorise factor_z within each day

    Returns a copy of *base_df* extended with:
      target_ret_n, factor_eval, target_eval
    """
    df = base_df.copy()
    df["target_ret_n"] = make_forward_vwap_return(df["vwap"], horizon)

    eligible = eligibility_mask_by_day(df, horizon)
    df["factor_eval"] = df["factor_z"].where(eligible)
    df["target_eval"] = df["target_ret_n"].where(eligible)

    df["factor_eval"] = winsorize_by_day(
        df["factor_eval"], df["date"], cfg.winsor_quantile,
    )
    return df


def prepare_status(
    base_df: pd.DataFrame, cfg: EvalConfig,
) -> pd.DataFrame:
    """Preprocessing for the status-analysis path (horizon=1 only,
    no eligibility mask).

    Steps:
      1. Compute single-period forward return: vwap[t+2]/vwap[t+1] − 1
      2. Winsorise factor_z within each day

    Returns a copy of *base_df* extended with:
      fwd_ret, factor_eval
    """
    df = base_df.copy()
    # single-period forward return: factor_t → ret from t+1 to t+2
    df["fwd_ret"] = df["vwap"].shift(-2) / df["vwap"].shift(-1) - 1.0
    df["factor_eval"] = winsorize_by_day(
        df["factor_z"], df["date"], cfg.winsor_quantile,
    )
    return df
