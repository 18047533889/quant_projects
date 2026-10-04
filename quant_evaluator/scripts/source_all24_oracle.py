"""Bounded independent reference for the canonical 15/24 source requests."""
from __future__ import annotations

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.factor_source import _source_request_fingerprint
from quant_evaluator.contracts.factor_tile_source import (
    admitted_source_tile_limit, capture_factor_tile_source, read_validated_factor_tile,
)
from quant_evaluator.contracts.label_bundle import LabelBundle

_SERIES_METRICS = frozenset({"rank_ic_series", "pearson_ic_series"})

REFERENCE_METRICS = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
)


def reference_source_all24(source, labels, *, metrics=REFERENCE_METRICS,
                           max_tile_size=1, max_result_bytes=64 * 1024**2):
    """Read one admitted tile at a time and independently reference each metric.

    Only canonical historical15/full24 requests are supported. The byte gate
    bounds returned arrays, not total RSS or the source's own scratch memory.
    The caller owns source lifecycle; this function never closes it.
    """
    for name, value in (("max_tile_size", max_tile_size),
                        ("max_result_bytes", max_result_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive builtin integer")
    if type(metrics) is not tuple or metrics not in (REFERENCE_METRICS[:15], REFERENCE_METRICS):
        raise ValueError("metrics must be the canonical historical15 or full24 tuple")
    if not isinstance(labels, LabelBundle):
        raise TypeError("labels must be LabelBundle")
    meta = capture_factor_tile_source(source)
    T, N, F = meta.time_axis.size, meta.asset_axis.size, len(meta.factor_ids)
    if (np.dtype(meta.dtype) not in (np.dtype("float32"), np.dtype("float64"))
            or labels.values.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise TypeError("reference requires float32/float64 source and labels")
    if labels.values.shape != (T, N):
        raise ValueError("label shape differs from source time/asset shape")
    time_values = np.asarray(labels.decision_time)
    if (time_values.dtype != meta.time_axis.values.dtype
            or not np.array_equal(time_values, meta.time_axis.values)):
        raise ValueError("label decision-time coordinates differ from source")
    axis = labels.asset_axis
    if (axis is None or axis.values is None
            or (axis.name, axis.dtype, axis.size) !=
               (meta.asset_axis.name, meta.asset_axis.dtype, meta.asset_axis.size)
            or axis.values.dtype != meta.asset_axis.values.dtype
            or not np.array_equal(axis.values, meta.asset_axis.values)):
        raise ValueError("label asset coordinates differ from source")
    result_bytes = 8 * F * (sum(T if name in _SERIES_METRICS else 1 for name in metrics)
                            + len(metrics))
    if result_bytes > max_result_bytes:
        raise MemoryError("independent reference result exceeds returned-array budget")
    width = min(max_tile_size, admitted_source_tile_limit(
        meta.max_tile_size, getattr(source, "admitted_max_tile_size", None)))
    out = BatchEvaluationBundle(meta.factor_ids, labels.target_id)
    for metric in metrics:
        if metric in _SERIES_METRICS:
            out.series_metrics[metric] = np.full((T, F), np.nan, dtype=np.float64)
        else:
            out.scalar_metrics[metric] = np.full(F, np.nan, dtype=np.float64)
        out.observation_counts[metric] = np.zeros(F, dtype=np.int64)
    for start in range(0, F, width):
        end = min(start + width, F)
        tile = read_validated_factor_tile(source, meta, start, end)
        parts = _reference_tile_parts(tile.batch, labels)
        if type(parts) is not tuple or not 1 <= len(parts) <= 2:
            raise ValueError("independent tile reference must return one or two bundles")
        seen = set()
        for part in parts:
            if (type(part) is not BatchEvaluationBundle
                    or part.factor_ids != meta.factor_ids[start:end]
                    or part.label_id != labels.target_id):
                raise ValueError("independent component factor/target identity differs")
            scalar_names, series_names = set(part.scalar_metrics), set(part.series_metrics)
            names = scalar_names | series_names
            if (scalar_names & series_names or names & seen
                    or not names <= set(REFERENCE_METRICS)
                    or set(part.observation_counts) != names or part.vector_metrics):
                raise ValueError("independent component metric domain is malformed")
            seen.update(names)
            for metric in names:
                is_series = metric in _SERIES_METRICS
                if is_series != (metric in series_names):
                    raise ValueError("independent component artifact kind differs")
                values = (part.series_metrics if is_series else part.scalar_metrics)[metric]
                expected_shape = (T, end-start) if is_series else (end-start,)
                if (not isinstance(values, np.ndarray) or values.shape != expected_shape
                        or values.dtype.kind not in "fiu"):
                    raise ValueError("independent component values have invalid shape/type")
                counts = part.observation_counts[metric]
                if (not isinstance(counts, np.ndarray) or counts.shape != (end-start,)
                        or counts.dtype.kind not in "iu" or np.any(counts < 0)
                        or (counts.dtype.kind == "u"
                            and np.any(counts > np.iinfo(np.int64).max))):
                    raise ValueError("independent component observation counts are invalid")
                if metric in metrics:
                    if is_series:
                        out.series_metrics[metric][:, start:end] = values
                    else:
                        out.scalar_metrics[metric][start:end] = values
                    out.observation_counts[metric][start:end] = counts
        if not set(metrics) <= seen:
            raise ValueError("independent component metrics do not cover the request")
        # Explicitly release all loop aliases before the next source read.
        del counts, values, part, parts, tile
    out.metadata = {
        "source_request_fingerprint": _source_request_fingerprint(meta, labels, metrics),
        "source_snapshot_id": meta.snapshot_id,
        "reference_method": "independent_all_source_metric_chain_v1",
        "coverage_scope": "every_time_factor; bounded_factor_tiles",
        "factor_tile_size": width,
        "returned_array_bytes": result_bytes,
    }
    return out


def _reference_tile_parts(batch, labels):
    # Imported only after complete admission, before computation on a valid tile.
    from quant_evaluator.scripts.source_all24_ic_oracle import reference_ic_tile
    from quant_evaluator.scripts.source_all24_shape_oracle import reference_shape_tile
    return (reference_ic_tile(batch, labels), reference_shape_tile(batch, labels))
