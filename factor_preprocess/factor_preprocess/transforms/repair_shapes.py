"""Stateless value-repair primitives used by research repair plans.

Every cross-sectional operation is evaluated independently per date and
returns a Series aligned to the original long-frame index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import math
import numbers

from .cross_sectional import cs_rank, cs_zscore


def _finite_number(value) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def _per_date(values: pd.DataFrame, operation) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float, name="value")
    for positions in values.groupby("date", sort=False).indices.values():
        positions = list(positions)
        raw = values.iloc[positions]["value"].to_numpy(dtype=float)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        result.iloc[positions] = operation(raw)
    return result


def cross_sectional_rank(values: pd.DataFrame, *, method: str = "average") -> pd.Series:
    if method not in {"average", "min"}:
        raise ValueError("method must be 'average' or 'min'")
    return _per_date(values, lambda x: cs_rank(x, method=method, pct=True))


def rank_shape(values: pd.DataFrame, *, center: float, power: float,
               inverted: bool = False, asymmetric: bool = False) -> pd.Series:
    if not _finite_number(center) or not 0.0 <= center <= 1.0:
        raise ValueError("center must be finite in [0, 1]")
    if not _finite_number(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    if type(inverted) is not bool:
        raise ValueError("inverted must be a strict bool")
    if type(asymmetric) is not bool:
        raise ValueError("asymmetric must be a strict bool")
    rank = cross_sectional_rank(values)
    distance = (rank - center).abs()
    if asymmetric:
        left = rank < center
        if center > 0:
            distance.loc[left] = distance.loc[left] / center
        if center < 1:
            distance.loc[~left] = distance.loc[~left] / (1.0 - center)
    shaped = distance.pow(power)
    return -shaped if inverted else shaped


def capped_zscore(values: pd.DataFrame, *, cap: float) -> pd.Series:
    if not _finite_number(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    return _per_date(values, lambda x: np.clip(cs_zscore(x, ddof=1), -cap, cap))


def tail_hinge(values: pd.DataFrame, *, hinge: str, hinge_value: float) -> pd.Series:
    if not _finite_number(hinge_value):
        raise ValueError("hinge_value must be finite")
    clean = values["value"].where(np.isfinite(values["value"]))
    if hinge == "top":
        return clean.sub(hinge_value).clip(lower=0.0).rename("value")
    if hinge == "bottom":
        return clean.sub(hinge_value).clip(upper=0.0).rename("value")
    raise ValueError("hinge must be 'top' or 'bottom'")


def tail_saturation(values: pd.DataFrame, *, quantile: float, side: str) -> pd.Series:
    if not _finite_number(quantile) or not .5 < quantile < 1.0:
        raise ValueError("quantile must be finite in (0.5, 1)")
    if side not in {"top", "bottom", "both"}:
        raise ValueError("side must be top, bottom, or both")
    def one(x):
        finite = x[np.isfinite(x)]
        if not len(finite):
            return np.full_like(x, np.nan)
        lower = np.quantile(finite, 1.0 - quantile) if side in {"bottom", "both"} else -np.inf
        upper = np.quantile(finite, quantile) if side in {"top", "both"} else np.inf
        return np.clip(x, lower, upper)
    return _per_date(values, one)


def robust_scale(values: pd.DataFrame, *, scale: str, center: str) -> pd.Series:
    if scale not in {"mad", "iqr", "std"}:
        raise ValueError("scale must be mad, iqr, or std")
    if center not in {"median", "mean"}:
        raise ValueError("center must be median or mean")
    def one(x):
        finite = x[np.isfinite(x)]
        if not len(finite):
            return np.full_like(x, np.nan)
        normalizer = np.max(np.abs(finite))
        normalized = finite if normalizer == 0 else finite / normalizer
        location = np.median(normalized) if center == "median" else np.mean(normalized)
        if scale == "mad":
            denominator = np.median(np.abs(normalized - np.median(normalized)))
        elif scale == "iqr":
            denominator = np.quantile(normalized, .75) - np.quantile(normalized, .25)
        elif scale == "std":
            denominator = np.std(normalized, ddof=1) if len(normalized) > 1 else 0.0
        else:
            raise ValueError("unknown robust scale")
        # A zero robust denominator can occur in a non-constant, highly tied
        # slice. Fall back to sample std; only a truly constant slice maps 0.
        if (not np.isfinite(denominator) or denominator == 0) and len(finite) > 1:
            denominator = np.std(normalized, ddof=1)
        out = np.full_like(x, np.nan)
        mask = np.isfinite(x)
        out[mask] = 0.0 if not np.isfinite(denominator) or denominator == 0 else (
            (x[mask] if normalizer == 0 else x[mask] / normalizer) - location
        ) / denominator
        return out
    return _per_date(values, one)
