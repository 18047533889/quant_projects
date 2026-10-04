"""Independent bounded oracle for the source-profile quantile/turnover family.

The oracle uses scalar percentile, tie-rank and bucket-mean arithmetic. It
retains the caller-owned factor tile plus ``(T, Q, F)`` daily bucket means and
counts; it never constructs a ``(T, N, F)`` working copy.
"""
from __future__ import annotations

import math

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_linear_shape_oracle import (
    _fractional_average,
    _independent_boundaries,
    _six_profiles,
)

N_QUANTILES = 5
MIN_BUCKET_ASSETS = 10
MIN_PERIODS = 20
TURNOVER_MIN_PERIODS = 2
FACTOR_TURNOVER_MIN_PERIODS = 30

METRIC_IDS = (
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
)
_LINEAR_METRICS = METRIC_IDS[6:]


def _rank_weights(row: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Average-tie rank weights normalized to one; None below 10 finite cells."""
    positions = np.flatnonzero(valid)
    if positions.size < MIN_BUCKET_ASSETS:
        return None
    order = positions[np.argsort(row[positions], kind="mergesort")]
    weights = np.zeros(row.size, dtype=np.float64)
    first = 0
    while first < order.size:
        last = first + 1
        while last < order.size and row[order[last]] == row[order[first]]:
            last += 1
        average_rank = ((first + 1) + last) / 2.0
        weights[order[first:last]] = average_rank
        first = last
    weights /= order.size * (order.size + 1) / 2.0
    return weights


def _linear_threshold(values: np.ndarray, quantile: float) -> float:
    """Scalar NumPy-style ``method='linear'`` quantile for finite values."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = quantile * (ordered.size - 1)
    lower = int(position)
    fraction = position - lower
    if lower >= ordered.size - 1:
        return float(ordered[-1])
    left, right = float(ordered[lower]), float(ordered[lower + 1])
    delta = right - left
    if math.isfinite(delta):
        return left + fraction * delta
    return (1.0 - fraction) * left + fraction * right


def reference_shape_tile(batch: FactorBatch, labels: LabelBundle) -> BatchEvaluationBundle:
    """Return independent values and conventional per-factor observation counts.

    Fixed quantile settings are Q=5, minimum bucket size 10, and 20-period
    profile gates. Registry turnover gates remain 2 periods for rank-proxy
    turnover and 30 valid transitions for top-membership turnover.
    """
    if not isinstance(batch, FactorBatch):
        raise TypeError("batch must be FactorBatch")
    if not isinstance(labels, LabelBundle):
        raise TypeError("labels must be LabelBundle")
    T, N, F = batch.values.shape
    if labels.values.shape != (T, N):
        raise ValueError("label shape must match factor tile time/asset axes")

    result = BatchEvaluationBundle(tuple(batch.factor_ids), labels.target_id)
    for metric in METRIC_IDS:
        result.scalar_metrics[metric] = np.full(F, np.nan, dtype=np.float64)
        result.observation_counts[metric] = np.zeros(F, dtype=np.int64)

    # These are the only retained daily quantile work arrays. No factor cube
    # copy or asset-expanded assignment tensor is created.
    daily_means = np.full((T, N_QUANTILES, F), np.nan, dtype=np.float64)
    daily_counts = np.zeros((T, N_QUANTILES, F), dtype=np.int64)
    label_values = labels.values
    label_validity = labels.validity
    factor_values = batch.values
    factor_validity = batch.validity
    pair_counts = np.zeros(F, dtype=np.int64)

    for t in range(T):
        labels_t = np.asarray(label_values[t], dtype=np.float64)
        valid_labels = np.isfinite(labels_t)
        if label_validity is not None:
            valid_labels &= label_validity[t]
        for f in range(F):
            values_t = np.asarray(factor_values[t, :, f], dtype=np.float64)
            valid_factor = np.isfinite(values_t)
            if factor_validity is not None:
                valid_factor &= factor_validity[t, :, f]
            pair_counts[f] += int(np.count_nonzero(valid_factor & valid_labels))

            factor_assets = np.flatnonzero(valid_factor)
            if factor_assets.size < N_QUANTILES:
                continue
            edges = _independent_boundaries(values_t[factor_assets], N_QUANTILES)
            # Factor-only edges; edge ties enter the higher bucket.
            assignments = np.searchsorted(edges, values_t[factor_assets], side="right")
            selected_labels = valid_labels[factor_assets]
            selected_bins = assignments[selected_labels]
            selected_returns = labels_t[factor_assets[selected_labels]]
            for q in range(N_QUANTILES):
                bucket_returns = selected_returns[selected_bins == q]
                count = int(bucket_returns.size)
                daily_counts[t, q, f] = count
                if count >= MIN_BUCKET_ASSETS:
                    daily_means[t, q, f] = _fractional_average(bucket_returns)

    denominator = T * N
    result.scalar_metrics["coverage"] = np.divide(
        pair_counts.astype(np.float64), denominator,
        out=np.zeros(F, dtype=np.float64), where=denominator > 0,
    )
    result.observation_counts["coverage"] = pair_counts

    profiles = np.full((N_QUANTILES, F), np.nan, dtype=np.float64)
    for q in range(N_QUANTILES):
        for f in range(F):
            finite = np.isfinite(daily_means[:, q, f])
            if np.count_nonzero(finite) >= MIN_PERIODS:
                profiles[q, f] = _fractional_average(daily_means[finite, q, f])

    for f in range(F):
        with np.errstate(over="ignore", invalid="ignore"):
            spread = daily_means[:, -1, f] - daily_means[:, 0, f]
        finite_spread = spread[np.isfinite(spread)]
        result.observation_counts["quantile_spread"][f] = finite_spread.size
        if finite_spread.size >= MIN_PERIODS:
            result.scalar_metrics["quantile_spread"][f] = _fractional_average(finite_spread)

        profile = profiles[:, f]
        adjacent = np.isfinite(profile[:-1]) & np.isfinite(profile[1:])
        n_adjacent = int(np.count_nonzero(adjacent))
        result.observation_counts["quantile_monotonicity"][f] = n_adjacent
        if n_adjacent:
            increasing = profile[1:][adjacent] > profile[:-1][adjacent]
            result.scalar_metrics["quantile_monotonicity"][f] = np.count_nonzero(increasing) / n_adjacent

        daily_pair = (np.isfinite(daily_means[:, :-1, f])
                      & np.isfinite(daily_means[:, 1:, f]))
        daily_pair_counts = np.count_nonzero(daily_pair, axis=1)
        daily_shape = np.full(T, np.nan, dtype=np.float64)
        for t in np.flatnonzero(daily_pair_counts):
            pair_mask = daily_pair[t]
            increases = (daily_means[t, 1:, f][pair_mask]
                         > daily_means[t, :-1, f][pair_mask])
            daily_shape[t] = np.count_nonzero(increases) / daily_pair_counts[t]
        valid_daily_shape = daily_shape[np.isfinite(daily_shape)]
        result.observation_counts["daily_quantile_monotonicity_rate"][f] = valid_daily_shape.size
        if valid_daily_shape.size >= MIN_PERIODS:
            result.scalar_metrics["daily_quantile_monotonicity_rate"][f] = (
                _fractional_average(valid_daily_shape))

        shape_values = _six_profiles(profile)
        for metric, value in zip(_LINEAR_METRICS, shape_values, strict=True):
            result.scalar_metrics[metric][f] = value
            result.observation_counts[metric][f] = int(math.isfinite(value))

    # Rank-proxy turnover. Count reporting follows the evaluator's facade;
    # the value uses its >=10-finite average-rank-weight eligibility.
    ordinary_turnover_counts = np.zeros(F, dtype=np.int64)
    membership_counts = np.zeros(F, dtype=np.int64)
    ordinary_turnover_values: list[list[float]] = [[] for _ in range(F)]
    membership_values: list[list[float]] = [[] for _ in range(F)]
    previous_rows: list[tuple[np.ndarray | None, np.ndarray, np.ndarray] | None] = [None] * F
    for t in range(T):
        for f in range(F):
            row = np.asarray(factor_values[t, :, f], dtype=np.float64)
            valid = np.isfinite(row)
            if factor_validity is not None:
                valid &= factor_validity[t, :, f]
            valid_count = int(np.count_nonzero(valid))
            current_weights = _rank_weights(row, valid)
            current_top = np.zeros(N, dtype=bool)
            if valid_count >= MIN_BUCKET_ASSETS:
                current_top = valid & (row >= _linear_threshold(row[valid], 0.9))

            previous = previous_rows[f]
            if previous is not None:
                previous_weights, previous_valid, previous_top = previous
                if previous_weights is not None and current_weights is not None:
                    ordinary_turnover_counts[f] += 1
                    ordinary_turnover_values[f].append(
                        0.5 * float(np.sum(np.abs(current_weights - previous_weights))))

                eligible = (min(int(np.count_nonzero(previous_valid)), valid_count)
                            >= MIN_BUCKET_ASSETS
                            and not np.any(previous_top & ~valid))
                if eligible:
                    union_valid = previous_valid | valid
                    denom = int(np.count_nonzero(union_valid))
                    if denom:
                        changed = previous_top != current_top
                        membership_values[f].append(int(np.count_nonzero(changed)) / denom)
                        membership_counts[f] += 1
            previous_rows[f] = (current_weights, valid.copy(), current_top)

    for f in range(F):
        result.observation_counts["turnover"][f] = ordinary_turnover_counts[f]
        values = ordinary_turnover_values[f]
        if len(values) >= TURNOVER_MIN_PERIODS - 1:
            result.scalar_metrics["turnover"][f] = _fractional_average(values)
        result.observation_counts["factor_turnover_rate"][f] = membership_counts[f]
        membership = membership_values[f]
        if len(membership) >= FACTOR_TURNOVER_MIN_PERIODS:
            result.scalar_metrics["factor_turnover_rate"][f] = _fractional_average(membership)

    result.metadata = {
        "reference_method": "independent_factor_only_quantile_rank_and_membership_v1",
        "n_quantiles": N_QUANTILES,
        "min_assets": MIN_BUCKET_ASSETS,
        "min_periods": MIN_PERIODS,
        "turnover_min_periods": TURNOVER_MIN_PERIODS,
        "factor_turnover_min_periods": FACTOR_TURNOVER_MIN_PERIODS,
        "bucket_assignment": "finite_factor_only; linear_percentile; tie_max_searchsorted_right",
        "coverage_denominator": "T_times_N_per_factor",
        "rank_weights": "average_ties_normalized_to_sum_one",
        "membership_threshold": "linear_90th_percentile_inclusive",
    }
    return result


__all__ = ["METRIC_IDS", "reference_shape_tile"]
