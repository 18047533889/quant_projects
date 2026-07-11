"""Same-date cross-sectional transforms for assembled factor panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from toolkit.registry import is_allowed_transform


def cs_rank(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Return same-date percentile ranks in the 0 to 1 interval."""

    frame = _as_float_frame(factor_values)
    out = frame.rank(axis=1, pct=True, method="average")
    return _nan_no_signal_rows(out, frame)


def cs_zscore(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Return same-date cross-sectional z-scores."""

    frame = _as_float_frame(factor_values)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    out = frame.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return _nan_no_signal_rows(out, frame)


def cs_robust_zscore(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Return same-date median/MAD robust z-scores."""

    frame = _as_float_frame(factor_values)
    median = frame.median(axis=1)
    centered = frame.sub(median, axis=0)
    mad = centered.abs().median(axis=1) * 1.4826
    out = centered.div(mad.replace(0.0, np.nan), axis=0)
    return _nan_no_signal_rows(out, frame)


def cs_winsorize(
    factor_values: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip same-date values at cross-sectional quantile bounds."""

    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("winsorize bounds must satisfy 0 <= lower < upper <= 1")
    frame = _as_float_frame(factor_values)
    lower_bounds = frame.quantile(lower, axis=1)
    upper_bounds = frame.quantile(upper, axis=1)
    return frame.clip(lower=lower_bounds, upper=upper_bounds, axis=0)


def cs_demean(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Subtract the same-date cross-sectional mean."""

    frame = _as_float_frame(factor_values)
    return frame.sub(frame.mean(axis=1), axis=0)


def cs_scale(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Scale each date by the sum of absolute same-date values."""

    frame = _as_float_frame(factor_values)
    denom = frame.abs().sum(axis=1).replace(0.0, np.nan)
    out = frame.div(denom, axis=0)
    return _nan_no_signal_rows(out, frame)


def cs_minmax(factor_values: pd.DataFrame) -> pd.DataFrame:
    """Scale each date's cross section into the 0 to 1 range."""

    frame = _as_float_frame(factor_values)
    row_min = frame.min(axis=1)
    row_max = frame.max(axis=1)
    denom = (row_max - row_min).replace(0.0, np.nan)
    out = frame.sub(row_min, axis=0).div(denom, axis=0)
    return _nan_no_signal_rows(out, frame)


def cs_rank_gauss(
    factor_values: pd.DataFrame,
    *,
    clip: float = 1e-4,
) -> pd.DataFrame:
    """Map same-date ranks to approximately Gaussian scores."""

    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5")
    ranks = cs_rank(factor_values).clip(lower=clip, upper=1.0 - clip)
    return pd.DataFrame(
        stats.norm.ppf(ranks.to_numpy(dtype=float)),
        index=ranks.index,
        columns=ranks.columns,
    )


def cs_quantile_bucket(
    factor_values: pd.DataFrame,
    *,
    quantiles: int = 10,
) -> pd.DataFrame:
    """Assign same-date values to integer buckets from 1 to ``quantiles``."""

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    ranks = cs_rank(factor_values)
    buckets = np.ceil(ranks * quantiles).clip(1, quantiles)
    return pd.DataFrame(buckets, index=ranks.index, columns=ranks.columns).where(
        ranks.notna()
    )


def cs_signed_power(
    factor_values: pd.DataFrame,
    *,
    exponent: float = 0.5,
) -> pd.DataFrame:
    """Compress magnitude while preserving sign."""

    if exponent <= 0:
        raise ValueError("exponent must be positive")
    frame = _as_float_frame(factor_values)
    return np.sign(frame) * np.power(frame.abs(), exponent)


def apply_cross_sectional_transform(
    factor_values: pd.DataFrame,
    transform: str = "none",
) -> pd.DataFrame:
    """Apply one registered fixed cross-sectional transform."""

    if not isinstance(factor_values, pd.DataFrame):
        raise TypeError("factor_values must be a pandas DataFrame")
    if transform == "none":
        return factor_values.copy()
    if not is_allowed_transform(transform, exposed_only=True):
        raise ValueError(f"unknown or unavailable cross-sectional transform: {transform!r}")
    if transform == "cs_rank":
        return cs_rank(factor_values)
    if transform == "cs_zscore":
        return cs_zscore(factor_values)
    if transform == "cs_robust_zscore":
        return cs_robust_zscore(factor_values)
    if transform == "cs_winsorize_rank":
        return cs_rank(cs_winsorize(factor_values))
    if transform == "cs_winsorize_zscore":
        return cs_zscore(cs_winsorize(factor_values))
    if transform == "cs_rank_gauss":
        return cs_rank_gauss(factor_values)
    if transform == "cs_quantile_bucket":
        return cs_quantile_bucket(factor_values)
    raise ValueError(f"unhandled cross-sectional transform: {transform!r}")


def _as_float_frame(factor_values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(factor_values, pd.DataFrame):
        raise TypeError("factor_values must be a pandas DataFrame")
    return factor_values.astype(float)


def _nan_no_signal_rows(transformed: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    out = transformed.copy()
    no_signal_rows = _no_cross_section_signal_rows(original)
    if no_signal_rows.any():
        out.loc[no_signal_rows] = np.nan
    return out


def _no_cross_section_signal_rows(frame: pd.DataFrame) -> pd.Series:
    valid_count = frame.notna().sum(axis=1)
    row_min = frame.min(axis=1)
    row_max = frame.max(axis=1)
    return valid_count.lt(2) | row_min.eq(row_max)
