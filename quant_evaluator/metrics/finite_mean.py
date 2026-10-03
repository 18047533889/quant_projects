"""Numerically guarded means over finite samples along axis zero."""
import numbers

import numpy as np

from quant_evaluator.metrics.quantile_numeric import (
    label_sum_error_bounds, stable_finite_mean,
)


def finite_mean_axis0(values, min_periods=1):
    """Return finite-sample means and counts over axis zero.

    Ordinary columns use vectorized Float64 summation. Only columns whose
    conservative sequential-sum error bound exceeds the established 1e-12
    per-mean threshold, or whose vectorized mean is nonfinite, are repaired
    with ``stable_finite_mean``. All NaN and infinite samples are excluded;
    ``min_periods`` is applied to finite counts and leaves insufficient means
    as NaN.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("values must have a sample axis")
    if (isinstance(min_periods, (bool, np.bool_))
            or not isinstance(min_periods, (numbers.Integral, np.integer))
            or min_periods <= 0):
        raise ValueError("min_periods must be a positive integer")
    min_periods = int(min_periods)

    finite = np.isfinite(array)
    counts = np.sum(finite, axis=0, dtype=np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        totals = np.sum(np.where(finite, array, 0.0), axis=0)
    means = np.divide(
        totals, counts,
        out=np.full(np.shape(totals), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    sufficient = counts >= min_periods
    means = np.where(sufficient, means, np.nan)
    if array.shape[0] == 0 or not np.any(sufficient):
        return means, counts

    columns = array.reshape(array.shape[0], -1)
    column_counts = np.asarray(counts).reshape(-1)
    column_means = np.asarray(means).reshape(-1)
    column_sufficient = np.asarray(sufficient).reshape(-1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error_bounds = label_sum_error_bounds(columns.T)
        risky = column_sufficient & (
            ~np.isfinite(column_means)
            | (error_bounds / np.maximum(column_counts, 1) > 1e-12)
        )
    for column in np.flatnonzero(risky):
        samples = columns[:, column]
        finite_samples = samples[np.isfinite(samples)]
        column_means[column] = stable_finite_mean(finite_samples)
    return means, counts
