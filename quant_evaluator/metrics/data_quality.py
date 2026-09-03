"""Data-quality metrics for a factor batch (spec §35).

These metrics consume the raw factor panel (``factor_batch``) and, where
relevant, the label panel (``label_bundle``).  They quantify how much of the
universe is usable, how stale / tied / outlier-heavy the factor values are,
and how much the tradable universe churns over time.

All functions return a per-factor scalar array of shape ``(F,)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.label_panel import normalize_label_panel

__all__ = [
    "compute_missing_ratio",
    "compute_missing_timeline",
    "compute_staleness",
    "compute_outlier_ratio",
    "compute_effective_n",
    "compute_distinct_level_ratio",
    "compute_tie_ratio",
    "compute_cross_section_cardinality",
    "compute_label_maturity",
    "compute_tradable_coverage",
    "compute_universe_churn",
]


def _factor_values(factor_batch: FactorBatch) -> np.ndarray:
    values = np.asarray(factor_batch.values, dtype=np.float64)
    if factor_batch.validity is not None:
        values = np.where(factor_batch.validity, values, np.nan)
    return values


def _labels(label_bundle: LabelBundle, num_assets: int) -> np.ndarray:
    labels, _ = normalize_label_panel(label_bundle, num_assets)
    return labels


def compute_missing_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Fraction of (T, N) cells with non-finite factor values, (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    missing = np.sum(~np.isfinite(values), axis=(0, 1))
    return missing / (T * N)


def compute_missing_timeline(factor_batch: FactorBatch) -> np.ndarray:
    """Fraction of time periods with any missing factor value, (F,).

    A value of 1.0 means every day has at least one missing asset; 0.0 means
    the factor is fully populated on every day.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    any_missing = np.any(~np.isfinite(values), axis=1)  # (T, F)
    return np.sum(any_missing, axis=0) / T


def compute_staleness(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of assets whose value is unchanged from the prior day, (F,).

    A high staleness ratio indicates the factor is slow-moving / sticky.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    if T < 2:
        return np.full(F, np.nan)
    prev = values[:-1, :, :]
    curr = values[1:, :, :]
    both = np.isfinite(prev) & np.isfinite(curr)
    unchanged = (prev == curr) & both
    n = np.sum(both, axis=(0, 1))
    same = np.sum(unchanged, axis=(0, 1))
    return same / np.maximum(n, 1)


def compute_outlier_ratio(
    factor_batch: FactorBatch,
    threshold: float = 3.0,
    min_obs: int = 10,
) -> np.ndarray:
    """Fraction of finite factor values that are z-score outliers, (F,).

    Outliers are defined per factor over the full (T, N) panel using a
    z-score threshold (default 3.0).
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        col = values[:, :, f]
        finite = np.isfinite(col)
        n = np.sum(finite)
        if n < min_obs:
            continue
        mean = np.nanmean(col)
        std = np.nanstd(col, ddof=1)
        if not np.isfinite(std) or std == 0:
            continue
        z = np.abs((col - mean) / std)
        out[f] = np.sum((z > threshold) & finite) / n
    return out


def compute_effective_n(factor_batch: FactorBatch) -> np.ndarray:
    """Mean number of finite factor values per day, (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    n = np.sum(np.isfinite(values), axis=1)  # (T, F)
    return np.mean(n, axis=0)


def compute_distinct_level_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of distinct values among finite factor values per day, (F,).

    A low ratio indicates heavy duplication / coarse factor values.
    """
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        ratios = []
        for t in range(T):
            row = values[t, :, f]
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            ratios.append(np.unique(finite).size / finite.size)
        if ratios:
            out[f] = float(np.mean(ratios))
    return out


def compute_tie_ratio(factor_batch: FactorBatch) -> np.ndarray:
    """Mean fraction of finite factor values that are tied with another value, (F,).

    Computed as ``1 - distinct_level_ratio`` (the complement of the distinct
    level ratio).
    """
    return 1.0 - compute_distinct_level_ratio(factor_batch)


def compute_cross_section_cardinality(factor_batch: FactorBatch) -> np.ndarray:
    """Mean number of distinct values per day (cross-section cardinality), (F,)."""
    values = _factor_values(factor_batch)
    T, N, F = values.shape
    out = np.full(F, np.nan)
    for f in range(F):
        counts = []
        for t in range(T):
            row = values[t, :, f]
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            counts.append(np.unique(finite).size)
        if counts:
            out[f] = float(np.mean(counts))
    return out


def compute_label_maturity(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Fraction of (T, N) cells with a finite forward-return label, (F,).

    Measures how much of the factor's universe has a mature (available)
    label for evaluation.
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    label_finite = np.isfinite(labels)  # (T, N)
    factor_finite = np.isfinite(values)  # (T, N, F)
    mature = factor_finite & label_finite[:, :, None]
    return np.sum(mature, axis=(0, 1)) / (T * N)


def compute_tradable_coverage(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
    min_assets: int = 10,
) -> np.ndarray:
    """Fraction of days with at least ``min_assets`` jointly valid cells, (F,).

    A tradable day is one where the factor and label are both available for
    at least ``min_assets`` assets.
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    label_finite = np.isfinite(labels)
    factor_finite = np.isfinite(values)
    valid = factor_finite & label_finite[:, :, None]  # (T, N, F)
    per_day = np.sum(valid, axis=1)  # (T, F)
    return np.sum(per_day >= min_assets, axis=0) / T


def compute_universe_churn(
    factor_batch: FactorBatch,
    label_bundle: LabelBundle,
) -> np.ndarray:
    """Mean fraction of the tradable universe that changes membership per day, (F,).

    Churn = fraction of assets that are tradable on exactly one of two
    adjacent days (entering or leaving the tradable set).
    """
    values = _factor_values(factor_batch)
    labels = _labels(label_bundle, factor_batch.num_assets)
    T, N, F = values.shape
    if T < 2:
        return np.full(F, np.nan)
    label_finite = np.isfinite(labels)
    factor_finite = np.isfinite(values)
    tradable = factor_finite & label_finite[:, :, None]  # (T, N, F)
    prev = tradable[:-1, :, :]
    curr = tradable[1:, :, :]
    churn = np.sum(prev != curr, axis=1)  # (T-1, F)
    return np.mean(churn / N, axis=0)
