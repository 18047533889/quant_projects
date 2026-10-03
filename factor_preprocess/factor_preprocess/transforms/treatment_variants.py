"""
Treatment variant transforms that the eligibility engine may propose.

These are *real, executable* transforms so that every transform id the
eligibility engine advertises resolves to an actual implementation in the
canonical :class:`TransformRegistry` (DLIB-FP-014). They live in this focused
module rather than a shared ``common/utils`` package.

Freshness-aware fill is current-inclusive: a finite value at ``t`` is emitted
at ``t`` and resets the age; missing rows carry the latest finite value.
"""
import numpy as np
import pandas as pd
from numbers import Integral
from typing import Optional


def _check_sort(values, asset_col, time_col):
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")


def freshness_aware_fill(
    values: pd.DataFrame,
    max_lag: Optional[int] = None,
    decay_halflife: float = 10.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Freshness-aware forward fill.

    Propagates the most recent valid observation forward, but *decays* the
    carried value by an exponential freshness weight as it ages. A value that
    has been stale for ``decay_halflife`` periods is halved. ``max_lag``
    bounds how many consecutive missing periods a value may be carried (None
    = unbounded). This is the recommended missingness treatment for
    FUNDAMENTAL / SPARSE_UPDATE factors.

    Timing: finite observations are emitted immediately at their timestamp.
    Missing or non-finite rows carry the last finite value with exponential
    age decay. With integer ``max_lag=L``, at most L consecutive missing rows
    are carried.
    """
    if isinstance(decay_halflife, (bool, np.bool_)):
        raise ValueError("decay_halflife must be finite and > 0")
    try:
        decay_halflife = float(decay_halflife)
    except (TypeError, ValueError) as exc:
        raise ValueError("decay_halflife must be finite and > 0") from exc
    if not np.isfinite(decay_halflife) or decay_halflife <= 0:
        raise ValueError(f"decay_halflife must be finite and > 0, got {decay_halflife}")
    if max_lag is not None and (isinstance(max_lag, (bool, np.bool_))
                                or not isinstance(max_lag, Integral) or max_lag < 1):
        raise ValueError(f"max_lag must be a positive integer or None, got {max_lag}")
    if max_lag is not None:
        max_lag = int(max_lag)

    _check_sort(values, asset_col, time_col)
    decay = np.log(2.0) / decay_halflife
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        vals = values.iloc[positions][value_col].to_numpy()
        n = len(vals)
        out = np.full(n, np.nan)
        last_valid = np.nan
        age = 0
        for i in range(n):
            x = vals[i]
            if np.isfinite(x):
                last_valid = x
                age = 0
                out[i] = x
            else:
                if not np.isfinite(last_valid):
                    out[i] = np.nan
                    continue
                if max_lag is not None and age >= max_lag:
                    out[i] = np.nan
                    continue
                age += 1
                out[i] = last_valid * np.exp(-decay * age)
        result.iloc[positions] = out
    return result
