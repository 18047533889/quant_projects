# -*- coding: utf-8
"""A-share minute features shared by the intraday daily runtime."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def ashare_price_limit_rate(instrument: str, trade_date: object) -> float:
    """Return the regular A-share daily price-limit rate.

    Instrument prefixes cover the commonly traded Shanghai/Shenzhen/Beijing
    boards.  ST flags are reference-data attributes and therefore intentionally
    remain an explicit runtime parameter instead of being inferred from prices.
    """
    code = str(instrument).split(".")[0]
    date = pd.Timestamp(trade_date)
    if code.startswith(("300", "301")) and date >= pd.Timestamp("2020-08-24"):
        return 0.20
    if code.startswith("688") and date >= pd.Timestamp("2019-07-22"):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30
    return 0.10


def ashare_limit_prices(
    previous_close: float,
    instrument: str,
    trade_date: object,
    *,
    is_st: bool = False,
) -> tuple[float, float]:
    """Theoretical upper/lower limits using exchange tick rounding."""
    if not np.isfinite(previous_close) or previous_close <= 0:
        return np.nan, np.nan
    rate = 0.05 if is_st else ashare_price_limit_rate(instrument, trade_date)
    return round(previous_close * (1.0 + rate) + _EPS, 2), round(
        previous_close * (1.0 - rate) + _EPS, 2
    )


def limit_touch_fraction(
    bars: pd.DataFrame,
    limit_price: float,
    *,
    direction: str,
) -> float:
    """Fraction of bars whose high/low touches the daily price limit."""
    if not np.isfinite(limit_price) or bars.empty:
        return np.nan
    if direction == "up":
        touched = pd.to_numeric(bars["high"], errors="coerce") >= limit_price - _EPS
    elif direction == "down":
        touched = pd.to_numeric(bars["low"], errors="coerce") <= limit_price + _EPS
    else:
        raise ValueError("direction must be up or down")
    return float(touched.mean())


def return_path_features(close: np.ndarray) -> dict[str, float]:
    """Reusable return-path geometry for an ordered intraday close path."""
    values = np.asarray(close, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"path_length": np.nan, "path_efficiency": np.nan, "reversal_count": np.nan}
    changes = np.diff(values)
    length = float(np.sum(np.abs(changes)))
    signs = np.sign(changes)
    nonzero = signs[signs != 0]
    reversals = float(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0.0
    efficiency = float(abs(values[-1] - values[0]) / length) if length > _EPS else 0.0
    return {"path_length": length, "path_efficiency": efficiency, "reversal_count": reversals}


def trade_structure_features(
    returns: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
) -> dict[str, float]:
    """Volume/turnover concentration and signed-flow structure."""
    ret = np.nan_to_num(np.asarray(returns, dtype=float), nan=0.0)
    vol = np.where(np.isfinite(volume), np.asarray(volume, dtype=float), 0.0)
    amt = np.where(np.isfinite(amount), np.asarray(amount, dtype=float), 0.0)
    total_volume, total_amount = float(vol.sum()), float(amt.sum())
    signed = float(np.sum(np.sign(ret) * vol) / total_volume) if total_volume > _EPS else np.nan
    average_trade_price = total_amount / total_volume if total_volume > _EPS else np.nan
    active_share = float(np.sum(vol[np.abs(ret) > 0]) / total_volume) if total_volume > _EPS else np.nan
    return {
        "signed_volume_imbalance": signed,
        "average_trade_price": float(average_trade_price),
        "active_volume_share": active_share,
    }
