"""Current observations represented against strictly preceding asset history."""
from __future__ import annotations

import numbers
import numpy as np
import pandas as pd


def _apply(values, window, *, method=None, cap=None):
    if (isinstance(window, (bool, np.bool_)) or not isinstance(window, numbers.Integral)
            or not 2 <= window <= 252):
        raise ValueError("window must be an integer in [2, 252]")
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values must be a DataFrame")
    if not values.columns.is_unique or not {"date", "asset_id", "value"}.issubset(values.columns):
        raise ValueError("unique date, asset_id and value columns are required")
    if values[["date", "asset_id"]].isna().any().any():
        raise ValueError("date and asset_id cannot be missing")
    if values.duplicated(["date", "asset_id"]).any():
        raise ValueError("duplicate date/asset identities")
    grouped = values.groupby("asset_id", sort=False)
    if not grouped["date"].is_monotonic_increasing.all():
        raise ValueError("dates must be monotone per asset")
    output = np.full(len(values), np.nan)
    raw = values["value"].to_numpy(dtype=float)
    for positions in grouped.indices.values():
        x = raw[positions]
        if len(x) <= window:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(x, window+1)
        # Bound temporary matrices even for long histories.
        for start in range(0, len(windows), 1024):
            block = windows[start:start+1024]
            valid = np.isfinite(block).all(axis=1)
            if not valid.any():
                continue
            observed = block[valid]
            history, current = observed[:, :-1], observed[:, -1]
            if method is not None:
                lower = np.sum(history < current[:, None], axis=1)
                equal = np.sum(history == current[:, None], axis=1)
                result = (lower + (.5*equal if method == "average" else 0.))/window
            else:
                scale = np.max(np.abs(history), axis=1)
                scale = np.where(scale > 0, scale, 1.)
                normalized = history/scale[:, None]
                spread = np.std(normalized, axis=1, ddof=1)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    result = np.clip((current/scale-np.mean(normalized, axis=1))/spread, -cap, cap)
                result = np.where(spread > 0, result, np.nan)
            output[positions[window+start+np.flatnonzero(valid)]] = result
    return pd.Series(output, index=values.index, name="value")


def time_series_rank(values, *, window, method="average"):
    """Rank current value against w previous observations; full finite history.

    average: (# previous < current + .5 * # previous == current) / w.
    min: # previous < current / w. The current value never enters history.
    Windows count each asset's observed rows, not elapsed calendar days.
    """
    if method not in {"average", "min"}:
        raise ValueError("method must be average or min")
    return _apply(values, window, method=method)


def capped_time_series_zscore(values, *, window, cap=3.):
    """Normalize current value with the previous w mean/sample std, then cap.

    Full finite history is required; zero history variance returns NaN.
    Scaling history before its moments prevents finite-magnitude overflow.
    """
    if (isinstance(cap, (bool, np.bool_)) or not isinstance(cap, numbers.Real)
            or not np.isfinite(cap) or cap <= 0):
        raise ValueError("cap must be finite and positive")
    return _apply(values, window, cap=float(cap))
