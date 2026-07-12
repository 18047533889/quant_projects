#!/usr/bin/env python3
"""Shared runtime helpers for executing CogAlpha Python factor code."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp import talib_compat  # noqa: E402


class _VolumeRegimeResult:
    """Supports both ``regime == 'high'`` and ``_, _, vol_ratio = ...`` patterns."""

    __slots__ = ("is_high", "is_low", "vol_ratio")

    def __init__(self, is_high: pd.Series, is_low: pd.Series, vol_ratio: pd.Series) -> None:
        self.is_high = is_high
        self.is_low = is_low
        self.vol_ratio = vol_ratio

    def __eq__(self, other: object) -> Any:
        if other == "high":
            return self.is_high > 0
        if other == "low":
            return self.is_low > 0
        if other == "normal":
            return (self.is_high <= 0) & (self.is_low <= 0)
        return False

    def __iter__(self):
        yield self.is_high
        yield self.is_low
        yield self.vol_ratio


class _AlphaToolsFacade:
    @staticmethod
    def classify_volume_regime(
        volume: pd.Series,
        window: int = 20,
        high_threshold: float = 1.5,
        low_threshold: float = 0.6,
    ) -> _VolumeRegimeResult:
        from toolkit.alpha_tools.library import classify_volume_regime as _cvr

        is_high, is_low, vol_ratio = _cvr(
            volume,
            window=window,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        return _VolumeRegimeResult(is_high, is_low, vol_ratio)

    @staticmethod
    def decompose_overnight_intraday(open_price: pd.Series, close: pd.Series):
        from toolkit.alpha_tools.library import decompose_overnight_intraday

        return decompose_overnight_intraday(open_price, close)


def _style_gate_resvol_high(close: pd.Series) -> pd.Series:
    """Proxy residual-volatility gate from price returns only."""
    ret = close.pct_change(fill_method=None)
    resvol = ret.rolling(20, min_periods=10).std()
    baseline = resvol.rolling(60, min_periods=20).median()
    return (resvol > baseline).astype(float).fillna(0.0)


def _style_gate_liquidity_high(df: pd.DataFrame) -> pd.Series:
    """Proxy liquidity gate from relative volume intensity."""
    volume = df["volume"]
    baseline = volume.rolling(20, min_periods=10).median()
    ratio = volume / baseline.replace(0, np.nan)
    return (ratio > 1.2).astype(float).fillna(0.0)


def causal_ts_rank(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 20,
    pct: bool = True,
) -> pd.Series:
    """Rolling time-series rank on one symbol's history (no full-sample look-ahead)."""
    return series.rolling(window, min_periods=min_periods).rank(pct=pct)


def prepare_factor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure optional CogAlpha columns exist before executing factor code."""
    work = df.copy()
    if "close" in work.columns and "style_gate_resvol_high" not in work.columns:
        work["style_gate_resvol_high"] = _style_gate_resvol_high(work["close"])
    if "volume" in work.columns and "style_gate_liquidity_high" not in work.columns:
        work["style_gate_liquidity_high"] = _style_gate_liquidity_high(work)
    return work


def build_python_exec_namespace() -> dict[str, Any]:
    """Namespace for ``exec`` of CogAlpha factor definitions."""
    try:
        import talib  # type: ignore
    except ImportError:
        talib = talib_compat

    alpha_tools = _AlphaToolsFacade()
    return {
        "pd": pd,
        "np": np,
        "talib": talib,
        "alpha_tools": alpha_tools,
        "causal_ts_rank": causal_ts_rank,
    }


def column_from_source(data_source: Any, name: str) -> Any:
    if hasattr(data_source, "load_column"):
        return data_source.load_column(name)
    if hasattr(data_source, "get_column"):
        return data_source.get_column(name)
    raise AttributeError(f"data source has no load_column/get_column: {type(data_source)}")
