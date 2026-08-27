"""
Treatment variant transforms that the eligibility engine may propose.

These are *real, executable* transforms so that every transform id the
eligibility engine advertises resolves to an actual implementation in the
canonical :class:`TransformRegistry` (DLIB-FP-014). They live in this focused
module rather than a shared ``common/utils`` package.

Causality
---------
All transforms in this module are strictly one-sided: each output depends only
on observations ``<= t - 1`` (the current observation is excluded via
``shift(1)``), so they are prefix-invariant and production-causal.
"""
import numpy as np
import pandas as pd
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

    Causality: each output uses only the most recent past valid observation
    and the elapsed age; the current observation is not consumed.
    """
    if decay_halflife <= 0:
        raise ValueError(f"decay_halflife must be > 0, got {decay_halflife}")
    if max_lag is not None and max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

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
