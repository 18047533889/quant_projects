#!/usr/bin/env python3
"""Pure-pandas TA-Lib substitutes for CogAlpha Python factor materialization."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(values, index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values, index=index, dtype="float64")


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def TRANGE(high, low, close) -> pd.Series:
    return _true_range(_as_series(high, None), _as_series(low, None), _as_series(close, None))


def ATR(high, low, close, timeperiod: int = 14) -> pd.Series:
    high = _as_series(high, None)
    low = _as_series(low, None)
    close = _as_series(close, None)
    tr = _true_range(high, low, close)
    return tr.rolling(timeperiod, min_periods=timeperiod).mean()


def SMA(series, timeperiod: int) -> pd.Series:
    series = _as_series(series, None)
    return series.rolling(timeperiod, min_periods=timeperiod).mean()


def EMA(series, timeperiod: int) -> pd.Series:
    series = _as_series(series, None)
    return series.ewm(span=timeperiod, min_periods=1, adjust=False).mean()


def ROC(series, timeperiod: int) -> pd.Series:
    series = _as_series(series, None)
    return series.pct_change(timeperiod)


def RSI(series, timeperiod: int = 14) -> pd.Series:
    series = _as_series(series, None)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / timeperiod, min_periods=timeperiod, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / timeperiod, min_periods=timeperiod, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ADX(high, low, close, timeperiod: int = 14) -> pd.Series:
    high = _as_series(high, None)
    low = _as_series(low, None)
    close = _as_series(close, None)
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(high, low, close)
    atr = tr.rolling(timeperiod, min_periods=timeperiod).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(timeperiod).sum() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(timeperiod).sum() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(timeperiod, min_periods=timeperiod).mean()
