"""Independent, tile-bounded reference for the source all-24 IC metrics."""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import pearsonr, rankdata

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

IC_TILE_METRICS = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
)

_MIN_ASSETS = 20
_MIN_SUMMARY_PERIODS = 20
_ROLLING_WINDOW = 60
_RECENT_WINDOW = 63
_ROLLING_STD_EPSILON = 1e-12


def _mean(values):
    return math.fsum(float(value) for value in values) / len(values)


def _sample_std(values, mean):
    if len(values) < 2:
        return math.nan
    return math.sqrt(math.fsum((float(value) - mean) ** 2 for value in values)
                     / (len(values) - 1))


def _correlation(x, y, *, spearman):
    if len(x) < _MIN_ASSETS or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan
    if spearman:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
        # Equal or complementary average ranks prove +/-1 independently;
        # do not turn normalization roundoff into enormous summary ICIR.
        if np.array_equal(x, y):
            return 1.0
        if np.all(x + y == len(x) + 1):
            return -1.0
    else:
        # Independent, scale-safe row oracle with Decimal fallback near constant.
        from quant_evaluator.scripts.source_pearson_oracle import reference_row_pearson

        return reference_row_pearson(x, y, min_assets=_MIN_ASSETS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(pearsonr(x, y).statistic)
    return value if math.isfinite(value) else math.nan


def _validate_inputs(batch, labels):
    if not isinstance(batch, FactorBatch):
        raise TypeError("batch must be FactorBatch")
    if not isinstance(labels, LabelBundle):
        raise TypeError("labels must be LabelBundle")
    if batch.values.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("IC tile oracle supports Float32/Float64 factors only")
    raw_labels = np.asarray(labels.values)
    if raw_labels.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("IC tile oracle supports Float32/Float64 labels only")
    T, N, _ = batch.values.shape
    if raw_labels.ndim == 1:
        if raw_labels.shape != (T,):
            raise ValueError("label time shape does not match factor tile")
        y = np.broadcast_to(raw_labels[:, None], (T, N))
    elif raw_labels.ndim == 2 and raw_labels.shape == (T, N):
        y = raw_labels
    else:
        raise ValueError("label shape does not match factor tile")
    if labels.decision_time and batch.time_axis.values is not None and not np.array_equal(
            np.asarray(labels.decision_time), batch.time_axis.values):
        raise ValueError("label decision times do not match factor tile")
    axis = labels.asset_axis
    if axis is not None and (axis.values is None or axis.name != batch.asset_axis.name
            or axis.dtype != batch.asset_axis.dtype
            or not np.array_equal(axis.values, batch.asset_axis.values)):
        raise ValueError("label asset coordinates do not match factor tile")
    label_validity = labels.validity
    if label_validity is not None:
        label_validity = np.asarray(label_validity, dtype=bool)
        if label_validity.ndim == 1 and label_validity.shape == (T,):
            label_validity = np.broadcast_to(label_validity[:, None], (T, N))
        elif label_validity.shape != (T, N):
            raise ValueError("label validity shape does not match factor tile")
    return y, label_validity


def _summarize(series, *, method):
    T, F = series.shape
    if method == "rank":
        names = ("rank_ic", "ic_ir", "ic_std", "ic_median",
                 "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir")
    else:
        names = ("pearson_ic", "pearson_ic_std", "pearson_ic_ir")
    out = {name: np.full(F, np.nan, dtype=np.float64) for name in names}
    for factor in range(F):
        values = series[:, factor]
        finite = [float(value) for value in values if math.isfinite(float(value))]
        count = len(finite)
        mean = _mean(finite) if finite else math.nan
        if count:
            out["rank_ic" if method == "rank" else "pearson_ic"][factor] = mean
        if count >= _MIN_SUMMARY_PERIODS:
            std = _sample_std(finite, mean)
            if method == "rank":
                out["ic_std"][factor] = std
                ordered = sorted(finite)
                out["ic_median"][factor] = (
                    ordered[(count - 1) // 2] + ordered[count // 2]) / 2.0
                out["rank_ic_positive_ratio"][factor] = sum(v > 0 for v in finite) / count
                tail = [float(v) for v in values[max(0, T - _RECENT_WINDOW):]
                        if math.isfinite(float(v))]
                if tail:
                    out["recent_3m_rank_ic"][factor] = _mean(tail)
            else:
                out["pearson_ic_std"][factor] = std
            # Whole-series ICIR excludes exact constants, without an epsilon
            # cutoff that would erase genuinely small daily IC dispersion.
            if any(value != finite[0] for value in finite) and std > 0:
                ratio = mean / std
                if math.isfinite(ratio):
                    out["ic_ir" if method == "rank" else "pearson_ic_ir"][factor] = ratio
        if method == "rank":
            rolling = []
            for end in range(T):
                window = [float(v) for v in values[max(0, end - _ROLLING_WINDOW + 1):end + 1]
                          if math.isfinite(float(v))]
                if len(window) < _MIN_SUMMARY_PERIODS:
                    continue
                window_mean = _mean(window)
                window_std = _sample_std(window, window_mean)
                # The documented rolling IR contract has a strict 1e-12 gate.
                if window_std > _ROLLING_STD_EPSILON:
                    ratio = window_mean / window_std
                    if math.isfinite(ratio):
                        rolling.append(ratio)
            if rolling:
                out["rolling_rank_ic_ir"][factor] = _mean(rolling)
    return out


def reference_ic_tile(batch: FactorBatch, labels: LabelBundle) -> BatchEvaluationBundle:
    """Return independent values for the twelve all-24 IC metrics.

    Daily Pearson and Spearman coefficients are computed row by row with SciPy,
    never through QE's IC or summary kernels. The only new numeric outputs are
    two ``(T, tile_F)`` series and scalar summaries; no full factor cube is
    allocated here.
    """
    y, label_validity = _validate_inputs(batch, labels)
    T, _, F = batch.values.shape
    rank_series = np.full((T, F), np.nan, dtype=np.float64)
    pearson_series = np.full((T, F), np.nan, dtype=np.float64)
    factor_validity = batch.validity
    for t in range(T):
        label_mask = np.isfinite(y[t])
        if label_validity is not None:
            label_mask &= label_validity[t]
        for factor in range(F):
            x = batch.values[t, :, factor]
            mask = label_mask & np.isfinite(x)
            if factor_validity is not None:
                mask &= factor_validity[t, :, factor]
            if int(np.count_nonzero(mask)) < _MIN_ASSETS:
                continue
            x_valid = np.asarray(x[mask], dtype=np.float64)
            y_valid = np.asarray(y[t, mask], dtype=np.float64)
            rank_series[t, factor] = _correlation(x_valid, y_valid, spearman=True)
            pearson_series[t, factor] = _correlation(x_valid, y_valid, spearman=False)
    result = BatchEvaluationBundle(batch.factor_ids, labels.target_id)
    result.series_metrics["rank_ic_series"] = rank_series
    result.series_metrics["pearson_ic_series"] = pearson_series
    result.scalar_metrics.update(_summarize(rank_series, method="rank"))
    result.scalar_metrics.update(_summarize(pearson_series, method="pearson"))
    result.observation_counts.update(_observation_counts(rank_series, pearson_series))
    result.metadata = {
        "reference_method": "scipy_rowwise_spearman_pearson_independent_v1",
        "coverage_scope": "every_time_factor_row",
    }
    return result


def _observation_counts(rank_series, pearson_series):
    """Bind each metric family to finite days in its own computed series."""
    rank_counts = np.count_nonzero(np.isfinite(rank_series), axis=0).astype(np.int64)
    pearson_counts = np.count_nonzero(np.isfinite(pearson_series), axis=0).astype(np.int64)
    return {
        metric: (pearson_counts if metric.startswith("pearson_ic") else rank_counts).copy()
        for metric in IC_TILE_METRICS
    }
