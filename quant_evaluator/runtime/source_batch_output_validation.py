"""Structural validation for columnar factor-source executor results."""
from __future__ import annotations

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.source_metric_catalog import SOURCE_SERIES_METRICS


def validate_source_batch_output(output, *, factor_ids, label_id, metrics, time_size):
    """Reject malformed executor output without changing or normalizing it."""
    def invalid():
        raise InvalidContractError("source batch executor returned an invalid output bundle")

    if not isinstance(output, BatchEvaluationBundle):
        invalid()
    if type(output.metadata) is not dict:
        invalid()
    if type(output.factor_ids) is not tuple:
        invalid()
    try:
        ids_match = output.factor_ids == tuple(factor_ids)
    except Exception:
        invalid()
    if not ids_match:
        invalid()
    if type(output.label_id) is not str or output.label_id != label_id:
        invalid()

    selected = tuple(metrics)
    expected_series = {metric for metric in selected if metric in SOURCE_SERIES_METRICS}
    expected_scalar = set(selected) - expected_series
    if (type(output.scalar_metrics) is not dict
            or type(output.series_metrics) is not dict
            or type(output.vector_metrics) is not dict
            or type(output.observation_counts) is not dict):
        invalid()
    if (set(output.scalar_metrics) != expected_scalar
            or set(output.series_metrics) != expected_series
            or output.vector_metrics):
        invalid()

    factor_size = len(factor_ids)
    for metric, values in output.scalar_metrics.items():
        if (not isinstance(values, np.ndarray) or values.dtype.kind != "f"
                or values.shape != (factor_size,)):
            invalid()
    for metric, values in output.series_metrics.items():
        if (not isinstance(values, np.ndarray) or values.dtype.kind != "f"
                or values.shape != (time_size, factor_size)):
            invalid()

    if set(output.observation_counts) != set(selected):
        invalid()
    int64_max = np.iinfo(np.int64).max
    for counts in output.observation_counts.values():
        if (not isinstance(counts, np.ndarray) or counts.dtype.kind not in "iu"
                or counts.shape != (factor_size,)):
            invalid()
        if counts.dtype.kind == "u" and np.any(counts > int64_max):
            invalid()
        if np.any(counts < 0):
            invalid()
    return output

