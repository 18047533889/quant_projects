"""Independent, bounded factor-source oracle for six linear shape metrics.

This module intentionally does not call the public quantile builder or any
production shape metric. It independently assigns factor-only buckets, scores
valid forward labels within those buckets, builds time-mean quantile profiles,
and evaluates the six linear formulas.
"""
from __future__ import annotations

from fractions import Fraction
import math
import sys

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.factor_source import _source_request_fingerprint
from quant_evaluator.contracts.factor_tile_source import (
    admitted_source_tile_limit,
    capture_factor_tile_source,
    read_validated_factor_tile,
)
from quant_evaluator.contracts.label_bundle import LabelBundle

METRIC_IDS = (
    "quantile_curvature",
    "quantile_tail_asymmetry",
    "quantile_adjacent_spread",
    "quantile_extreme_cliff",
    "top_quantile_cliff",
    "bottom_quantile_cliff",
)


def _validate_request(source, labels, *, max_tile_size, max_result_bytes,
                      n_quantiles, min_assets, min_periods):
    for name, value in (("max_tile_size", max_tile_size),
                        ("max_result_bytes", max_result_bytes),
                        ("n_quantiles", n_quantiles),
                        ("min_assets", min_assets),
                        ("min_periods", min_periods)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive builtin integer")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    if not isinstance(labels, LabelBundle):
        raise TypeError("labels must be LabelBundle")
    meta = capture_factor_tile_source(source)
    T, N, F = meta.time_axis.size, meta.asset_axis.size, len(meta.factor_ids)
    if labels.values.shape != (T, N):
        raise ValueError("label shape must match complete source time/asset axes")
    if labels.values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("linear-shape oracle requires Float32/Float64 labels")
    time_values = np.asarray(labels.decision_time)
    if (time_values.dtype != meta.time_axis.values.dtype
            or not np.array_equal(time_values, meta.time_axis.values)):
        raise ValueError("label decision-time axis must exactly match source time axis")
    axis = labels.asset_axis
    if (axis is None or axis.values is None
            or (axis.name, axis.dtype, axis.size) !=
               (meta.asset_axis.name, meta.asset_axis.dtype, meta.asset_axis.size)
            or axis.values.dtype != meta.asset_axis.values.dtype
            or not np.array_equal(axis.values, meta.asset_axis.values)):
        raise ValueError("label asset axis must exactly match typed source asset axis")
    # Six float64 result vectors plus six int64 0/1 evidence vectors. Admission
    # occurs before the first source read; this is returned-array budget only.
    result_bytes = 6 * F * (np.dtype(np.float64).itemsize
                            + np.dtype(np.int64).itemsize)
    if result_bytes > max_result_bytes:
        raise MemoryError("linear-shape oracle result exceeds max_result_bytes")
    try:
        admitted = admitted_source_tile_limit(
            meta.max_tile_size, getattr(source, "admitted_max_tile_size", None))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    width = min(max_tile_size, admitted)
    return meta, T, N, F, width


def _independent_boundaries(values: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Linear percentile edges, scalar reference arithmetic, for finite values."""
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    n = sorted_values.size
    if n < n_quantiles:
        return np.empty(max(n_quantiles - 1, 0), dtype=np.float64)
    edges = np.empty(max(n_quantiles - 1, 0), dtype=np.float64)
    for index in range(n_quantiles - 1):
        position = (index + 1) / n_quantiles * (n - 1)
        lower = int(position)
        fraction = position - lower
        if lower >= n - 1:
            edge = sorted_values[-1]
        elif fraction < 1e-9:
            edge = sorted_values[lower]
        elif fraction > 1.0 - 1e-9:
            edge = sorted_values[lower + 1]
        else:
            left, right = float(sorted_values[lower]), float(sorted_values[lower + 1])
            delta = right - left
            edge = (left + fraction * delta if math.isfinite(delta)
                    else (1.0 - fraction) * left + fraction * right)
        edges[index] = edge
    return edges


def _fractional_average(values) -> float:
    """Accurately average finite float inputs; reserve Fraction for risky scale."""
    seq = [float(value) for value in values]
    if not seq:
        return math.nan
    largest = max(abs(value) for value in seq)
    has_subnormal = any(
        0.0 < abs(value) < sys.float_info.min for value in seq)
    if has_subnormal or largest > sys.float_info.max / max(2, len(seq)):
        exact = sum((Fraction.from_float(value) for value in seq), Fraction())
        return float(exact / len(seq))
    try:
        return math.fsum(seq) / len(seq)
    except OverflowError:
        exact = sum((Fraction.from_float(value) for value in seq), Fraction())
        return float(exact / len(seq))


def _fraction_to_float(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def _exact_linear(values, coefficients, divisor=1) -> float:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    exact = sum(
        (Fraction.from_float(float(value)) * int(coefficient)
         for value, coefficient in zip(values, coefficients, strict=True)),
        Fraction(),
    )
    return _fraction_to_float(exact / divisor)


def _six_profiles(profile: np.ndarray) -> tuple[float, ...]:
    q = profile.size
    result = [math.nan] * len(METRIC_IDS)
    finite = np.isfinite(profile)
    if q >= 3:
        second = [
            (profile[index - 1], profile[index], profile[index + 1])
            for index in range(1, q - 1)
            if finite[index - 1] and finite[index] and finite[index + 1]
        ]
        if second:
            exact_curvature_sum = sum(
                (
                    Fraction.from_float(float(upper))
                    - 2 * Fraction.from_float(float(middle))
                    + Fraction.from_float(float(lower))
                    for lower, middle, upper in second
                ),
                Fraction(),
            )
            result[0] = _fraction_to_float(exact_curvature_sum / len(second))
        middle_index = q // 2
        if finite[0] and finite[middle_index] and finite[-1]:
            result[1] = _exact_linear(
                (profile[-1], profile[middle_index], profile[0]), (1, -2, 1))
    pairs = [
        (profile[index], profile[index + 1])
        for index in range(q - 1) if finite[index] and finite[index + 1]
    ]
    if pairs:
        differences = [
            abs(Fraction.from_float(float(right)) - Fraction.from_float(float(left)))
            for left, right in pairs
        ]
        result[2] = _fraction_to_float(sum(differences, Fraction()) / len(differences))
    if q >= 2:
        if finite[0] and finite[1] and finite[-2] and finite[-1]:
            result[3] = _exact_linear(
                (profile[-1], profile[-2], profile[1], profile[0]),
                (1, -1, 1, -1), divisor=2)
        if finite[-2] and finite[-1]:
            result[4] = _exact_linear((profile[-1], profile[-2]), (1, -1))
        if finite[0] and finite[1]:
            result[5] = _exact_linear((profile[1], profile[0]), (1, -1))
    return tuple(result)


def reference_source_linear_shape(
    source,
    labels: LabelBundle,
    *,
    max_tile_size: int = 1,
    max_result_bytes: int = 64 * 1024**2,
    n_quantiles: int = 5,
    min_assets: int = 10,
    min_periods: int = 20,
) -> BatchEvaluationBundle:
    """Independent factor-only Q-bucket oracle, retaining at most one tile.

    Per decision date/factor, bucket edges use finite factor values (and the
    factor validity mask) only. Forward-label validity enters only the
    per-bucket return count and mean. A quantile's temporal profile is the
    unweighted mean of its finite daily bucket means, subject to min_periods.
    """
    meta, T, N, F, width = _validate_request(
        source, labels, max_tile_size=max_tile_size,
        max_result_bytes=max_result_bytes, n_quantiles=n_quantiles,
        min_assets=min_assets, min_periods=min_periods)
    # Bound output storage before processing and never allocate T x N x F.
    out = BatchEvaluationBundle(meta.factor_ids, labels.target_id)
    for metric in METRIC_IDS:
        out.scalar_metrics[metric] = np.full(F, np.nan, dtype=np.float64)
        out.observation_counts[metric] = np.zeros(F, dtype=np.int64)
    fingerprint = _source_request_fingerprint(meta, labels, METRIC_IDS)
    for start in range(0, F, width):
        end = min(start + width, F)
        tile = read_validated_factor_tile(source, meta, start, end)
        factor_values = None
        batch = tile.batch
        daily = np.full((T, n_quantiles, end - start), np.nan, dtype=np.float64)
        for t in range(T):
            row_labels = np.asarray(labels.values[t], dtype=np.float64)
            label_valid = np.isfinite(row_labels)
            if labels.validity is not None:
                label_valid &= labels.validity[t]
            for local_f in range(end - start):
                factor_values = np.asarray(batch.values[t, :, local_f], dtype=np.float64)
                factor_valid = np.isfinite(factor_values)
                if batch.validity is not None:
                    factor_valid &= batch.validity[t, :, local_f]
                assets = np.flatnonzero(factor_valid)
                if assets.size < n_quantiles:
                    continue
                edges = _independent_boundaries(
                    factor_values[assets], n_quantiles)
                assignments = np.searchsorted(
                    edges, factor_values[assets], side="right")
                label_assets = assets[label_valid[assets]]
                if not label_assets.size:
                    continue
                label_bins = assignments[label_valid[assets]]
                label_values = row_labels[label_assets]
                for bucket in range(n_quantiles):
                    bucket_values = label_values[label_bins == bucket]
                    if bucket_values.size >= min_assets:
                        daily[t, bucket, local_f] = _fractional_average(bucket_values)
        # Row views can otherwise retain the previous tile until the next
        # read creates its replacement.
        factor_values = None
        profiles = np.full((n_quantiles, end - start), np.nan, dtype=np.float64)
        for q in range(n_quantiles):
            for local_f in range(end - start):
                finite_daily = daily[:, q, local_f]
                finite_daily = finite_daily[np.isfinite(finite_daily)]
                if finite_daily.size >= min_periods:
                    profiles[q, local_f] = _fractional_average(finite_daily)
        for local_f in range(end - start):
            values = _six_profiles(profiles[:, local_f])
            global_f = start + local_f
            for metric, value in zip(METRIC_IDS, values, strict=True):
                out.scalar_metrics[metric][global_f] = value
                out.observation_counts[metric][global_f] = int(math.isfinite(value))
        del profiles, daily, batch, tile
    out.metadata = {
        "source_request_fingerprint": fingerprint,
        "source_snapshot_id": meta.snapshot_id,
        "reference_method": "independent_factor_only_percentile_fsum_fraction_shape_v1",
        "coverage_scope": "every_factor; bounded_factor_tiles",
        "factor_tile_size": width,
        "n_quantiles": n_quantiles,
        "min_assets": min_assets,
        "min_periods": min_periods,
        "bucket_assignment": "finite_factor_only; linear_percentile; tie_max_searchsorted_right",
        "label_validity_scope": "bucket_return_aggregation_only",
    }
    return out


__all__ = ["METRIC_IDS", "reference_source_linear_shape"]
