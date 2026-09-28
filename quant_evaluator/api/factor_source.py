"""One-call bounded factor-source evaluation for factor-separable core metrics.

This is a columnar BatchEvaluationBundle API, not the richer EvaluationBundle
artifact/probe contract. Unsupported metrics and special inputs fail closed.
"""
from __future__ import annotations

from uuid import uuid4

import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.contracts.errors import InvalidContractError, UnsupportedMetricError
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.factor_tile_source import (
    capture_factor_tile_source, read_validated_factor_tile,
)


_SOURCE_METRICS = frozenset({"rank_ic", "rank_ic_series", "coverage", "quantile_spread"})
_F8_RANK_PAIR = frozenset({"rank_ic", "rank_ic_series"})
_F8_SHAPE = (2586, 5461, 8)
_F8_MIN_EFFECTIVE_VRAM_BYTES = 14 * 1024 ** 3


def _validate_source_label(metadata, label_bundle):
    T, N = metadata.time_axis.size, metadata.asset_axis.size
    if label_bundle.values.shape != (T, N):
        raise InvalidContractError("label shape must match the complete factor source axes")
    axis = label_bundle.asset_axis
    if (axis is None or axis.values is None or axis.name != metadata.asset_axis.name
            or axis.dtype != metadata.asset_axis.dtype
            or not np.array_equal(axis.values, metadata.asset_axis.values)):
        raise InvalidContractError("label asset coordinates must match the factor source")
    if (len(label_bundle.decision_time) != T
            or not np.array_equal(np.asarray(label_bundle.decision_time), metadata.time_axis.values)):
        raise InvalidContractError("label decision times must match the factor source")


def _cpu_source_batch(source, metadata, label, metrics, policy, max_tile_size):
    from quant_evaluator.runtime.evaluator import evaluate

    total = len(metadata.factor_ids)
    out = BatchEvaluationBundle(metadata.factor_ids, label.target_id)
    reserved = 0
    count = 0
    width = min(metadata.max_tile_size, max_tile_size or metadata.max_tile_size)
    for start in range(0, total, width):
        tile = read_validated_factor_tile(source, metadata, start, min(start + width, total))
        bundle = evaluate(tile.batch, label, metrics=metrics, backend="cpu")
        start, end = tile.start, tile.end
        for metric in metrics:
            artifact = bundle.artifacts.get(metric)
            if artifact is None or artifact.artifact_kind not in {"scalar", "series"}:
                raise InvalidContractError(f"source CPU metric {metric!r} has no scalar/series artifact")
            values = artifact.values
            if values.shape[-1] != end - start:
                raise InvalidContractError(f"source CPU metric {metric!r} lost its factor axis")
            group = out.series_metrics if artifact.artifact_kind == "series" else out.scalar_metrics
            if metric not in group:
                bytes_needed = int(np.prod(values.shape[:-1])) * total * values.dtype.itemsize
                reserved += bytes_needed
                if reserved > policy.max_host_result_bytes:
                    raise MemoryError("source batch results exceed max_host_result_bytes")
                group[metric] = np.empty((*values.shape[:-1], total), dtype=values.dtype)
            group[metric][..., start:end] = values
            if metric not in out.observation_counts:
                bytes_needed = total * np.dtype(np.int64).itemsize
                reserved += bytes_needed
                if reserved > policy.max_host_result_bytes:
                    raise MemoryError("source batch counts exceed max_host_result_bytes")
                out.observation_counts[metric] = np.empty(total, dtype=np.int64)
            for index, factor_id in enumerate(tile.batch.factor_ids):
                if metric == "rank_ic_series":
                    observations = int(np.isfinite(values[:, index]).sum())
                else:
                    group_metrics = bundle.grouped_metrics or {}
                    observations = group_metrics[factor_id][metric].observation_count
                out.observation_counts[metric][start + index] = observations
        count += 1
    out.metadata = {
        "factor_tiles_processed": count,
        "source_snapshot_id": metadata.snapshot_id,
        "host_result_bytes_reserved": reserved,
        "host_result_budget_bytes": policy.max_host_result_bytes,
    }
    return out


def evaluate_factor_source_batch(
    source, label_bundle, *, metrics=("rank_ic", "rank_ic_series"),
    backend="auto", max_tile_size=None, gpu_policy=None,
) -> BatchEvaluationBundle:
    """Evaluate every source factor once in bounded tiles.

    Supported options: backend is auto/cpu/cuda_strict; max_tile_size caps
    source reads; gpu_policy controls VRAM fraction, OOM retiling and output
    budget. The caller owns source.close(). The result is columnar and omits
    the richer EvaluationBundle diagnostics/probe artifacts.
    """
    if isinstance(metrics, (str, bytes)):
        raise InvalidContractError("metrics must be a sequence of metric ids")
    try:
        selected = tuple(metrics)
    except TypeError as exc:
        raise InvalidContractError("metrics must be a nonempty sequence") from exc
    if (not selected or any(not isinstance(metric, str) or not metric.strip()
                            for metric in selected)
            or len(set(selected)) != len(selected)):
        raise InvalidContractError("metrics must contain nonempty unique strings")
    unsupported = set(selected) - _SOURCE_METRICS
    if unsupported:
        raise UnsupportedMetricError(f"source batch does not implement {sorted(unsupported)}")
    if backend is None:
        backend = "auto"
    if backend not in {"auto", "cpu", "cuda_strict"}:
        raise InvalidContractError("source backend must be auto, cpu or cuda_strict")
    if max_tile_size is not None and (type(max_tile_size) is not int or max_tile_size <= 0):
        raise InvalidContractError("max_tile_size must be a positive integer")
    policy = GPUExecutionPolicy() if gpu_policy is None else gpu_policy
    if not isinstance(policy, GPUExecutionPolicy):
        raise InvalidContractError("gpu_policy must be GPUExecutionPolicy")
    if not isinstance(label_bundle, LabelBundle):
        raise InvalidContractError("label_bundle must be a typed LabelBundle")
    metadata = capture_factor_tile_source(source)
    _validate_source_label(metadata, label_bundle)
    route = backend
    reason = "explicit"
    if backend == "auto":
        shape = (metadata.time_axis.size, metadata.asset_axis.size, len(metadata.factor_ids))
        if (shape == _F8_SHAPE and metadata.dtype == "float64"
                and label_bundle.values.dtype == np.float64
                and len(selected) == 2 and frozenset(selected) == _F8_RANK_PAIR):
            from quant_evaluator.runtime.evaluator import _auto_batch_cuda_rejection

            if policy.precision_policy != GPUExecutionPolicy().precision_policy:
                rejection = "gpu_precision_policy_outside_certified_range"
            else:
                rejection = _auto_batch_cuda_rejection(policy, _F8_MIN_EFFECTIVE_VRAM_BYTES)
            route = "cuda_strict" if rejection is None else "cpu"
            reason = "bounded_f8_rank_pair_gpu" if rejection is None else rejection
        else:
            route, reason = "cpu", "source_shape_or_metrics_not_certified"
    if route == "cuda_strict":
        from quant_evaluator.runtime.device_session import DeviceEvaluationSession
        from quant_evaluator.runtime.gpu_executor import GPUExecutor

        with DeviceEvaluationSession(policy) as session:
            out = GPUExecutor(session).run_source_tiled(
                source, label_bundle, selected, max_tile_size=max_tile_size,
                source_metadata=metadata)
    else:
        out = _cpu_source_batch(source, metadata, label_bundle, selected, policy, max_tile_size)
    out.metadata.update({
        "request_id": str(uuid4()),
        "backend_requested": backend,
        "backend_used": "cuda" if route == "cuda_strict" else "cpu",
        "auto_backend_reason": reason if backend == "auto" else None,
        "metric_backends": {
            metric: "cuda" if route == "cuda_strict" else "cpu" for metric in selected
        },
        "source_snapshot_id": metadata.snapshot_id,
        "source_api": "columnar_factor_source_v1",
        "source_identity_trust": "caller_supplied_snapshot_id",
    })
    return out
