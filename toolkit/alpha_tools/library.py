"""Active seed alpha tools disclosed by the reference report.

Seed tools are fixed baseline modules. Generated tools may extend the library,
but automated maintenance must not replace or delete these report-disclosed
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_overnight_intraday(
    close: pd.Series,
    open_price: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Decompose daily return into overnight and intraday components.

    Economic meaning:
        Separates overnight information diffusion from intraday trading-session
        price discovery.
    Args:
        close: Backward-adjusted close price Series for one stock.
        open_price: Backward-adjusted open price Series for one stock, aligned to
            ``close``.
    Returns:
        Tuple ``(overnight_ret, intraday_ret)`` aligned to ``close.index``.
        ``overnight_ret = open_t / close_{t-1} - 1`` and
        ``intraday_ret = close_t / open_t - 1``.
    Required columns:
        None; pass close and open Series directly.
    Leakage notes:
        Uses previous close, current open, and current close only.
    """

    open_price = open_price.reindex(close.index)
    prev_close = close.shift(1)
    overnight_ret = open_price / prev_close.replace(0, np.nan) - 1.0
    intraday_ret = close / open_price.replace(0, np.nan) - 1.0
    overnight_ret = pd.Series(overnight_ret, index=close.index, dtype="float64").replace(
        [np.inf, -np.inf], np.nan
    )
    intraday_ret = pd.Series(intraday_ret, index=close.index, dtype="float64").replace(
        [np.inf, -np.inf], np.nan
    )
    overnight_ret.name = "overnight_ret"
    intraday_ret.name = "intraday_ret"
    return overnight_ret, intraday_ret


def classify_volume_regime(
    volume: pd.Series,
    window: int = 20,
    high_threshold: float = 1.5,
    low_threshold: float = 0.6,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classify volume state as high, low, or normal.

    Economic meaning:
        Uses volume relative to its EMA baseline to identify whether the trading
        environment is expansionary, contracted, or normal.
    Args:
        volume: Volume Series for one stock.
        window: EMA span for the historical volume baseline.
        high_threshold: Mark high-volume regime when ``volume / EMA`` is above
            this threshold.
        low_threshold: Mark low-volume regime when ``volume / EMA`` is below this
            threshold.
    Returns:
        Tuple ``(is_high_vol, is_low_vol, vol_ratio)`` aligned to ``volume.index``.
        High/low flags are float Series with values 0 or 1.
    Required columns:
        None; pass volume Series directly.
    Leakage notes:
        Uses causal EMA with current and past volume observations only.
    """

    vol_ema = volume.ewm(span=window).mean()
    vol_ratio = volume / vol_ema.replace(0, np.nan)
    vol_ratio = pd.Series(vol_ratio, index=volume.index, dtype="float64").replace(
        [np.inf, -np.inf], np.nan
    )
    is_high_vol = (vol_ratio > high_threshold).astype(float)
    is_low_vol = (vol_ratio < low_threshold).astype(float)
    is_high_vol[vol_ratio.isna()] = np.nan
    is_low_vol[vol_ratio.isna()] = np.nan
    is_high_vol.name = "is_high_vol"
    is_low_vol.name = "is_low_vol"
    vol_ratio.name = "vol_ratio"
    return is_high_vol, is_low_vol, vol_ratio
