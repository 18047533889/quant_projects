"""Rebind structural source-profile fixtures to real small CPU API outputs.

The CUDA side is an explicit numerical clone for CPU-only routing tests. It is
not evidence of real GPU execution or performance.
"""

from dataclasses import replace
import hashlib
import json

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.scripts.source_profile_measurement import (
    _metric_array,
    build_backend_profile_measurement, build_metric_comparison_receipts,
)


def _backend_bundle(bundle, backend):
    metadata = dict(bundle.metadata)
    metadata["backend_used"] = backend
    return BatchEvaluationBundle(
        factor_ids=bundle.factor_ids, label_id=bundle.label_id,
        scalar_metrics={name: values.copy() for name, values in bundle.scalar_metrics.items()},
        series_metrics={name: values.copy() for name, values in bundle.series_metrics.items()},
        vector_metrics={name: values.copy() for name, values in bundle.vector_metrics.items()},
        metadata=metadata,
        observation_counts={name: values.copy()
                            for name, values in bundle.observation_counts.items()},
    )


def _run_receipt(bundle, profile, context, backend):
    ranges = profile.source_ranges
    source_width = profile.source_tile_size
    schedule_sha = stable_content_hex(
        tag="SourceExecutionSchedule.v1",
        fields={"backend": backend, "factor_count": len(bundle.factor_ids),
                "source_width": source_width, "source_ranges": ranges,
                "compute_ranges": profile.compute_ranges, "oom_retries": 0},
    )
    ids_sha = hashlib.sha256(json.dumps(
        list(bundle.factor_ids), separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    return {
        "backend_used": backend,
        "source_request_fingerprint": context.request_content_sha256,
        "factor_ids_sha256": ids_sha,
        "timing_scope": context.timing_scope,
        "seconds": float(profile.seconds),
        "total_wall_seconds": float(profile.seconds),
        "oom_retries": 0,
        "actual_source_tile_size": source_width,
        "actual_gpu_factor_tile_size": source_width if backend == "cuda" else None,
        "effective_max_tile_size": (context.live_source_admitted_max_tile_size
                                    if backend == "cpu"
                                    else context.gpu_admitted_max_tile_size),
        "execution_schedule_scope": "zero_oom_source_equals_compute_v1",
        "execution_schedule_sha256": schedule_sha,
        "factor_tiles_processed": len(ranges),
        "tile_ranges": ranges,
        "compute_tile_ranges": profile.compute_ranges,
    }


def rebind_pair_to_cpu_outputs(records, actual_cpu_bundle):
    """Rebuild both backend receipts from one real CPU output bundle.

    The cloned CUDA values are for tests that exercise CPU fallback routing or
    CUDA dispatch plumbing with a patched executor. No GPU truth is implied.
    """
    context = records[0].context
    cpu_bundle = _backend_bundle(actual_cpu_bundle, "cpu")
    simulated_cuda_bundle = _backend_bundle(actual_cpu_bundle, "cuda")
    rebound = []
    for record in records:
        cpu_profile = build_backend_profile_measurement(
            cpu_bundle, _run_receipt(cpu_bundle, record.cpu, context, "cpu"),
            context, correctness_validated=True)
        cuda_profile = build_backend_profile_measurement(
            simulated_cuda_bundle,
            _run_receipt(simulated_cuda_bundle, record.cuda, context, "cuda"),
            context, correctness_validated=True)
        metric_report = {"pass": True, "metrics": {}}
        for metric in context.metric_ids:
            left, left_counts = _metric_array(cpu_bundle, metric, context)
            right, right_counts = _metric_array(simulated_cuda_bundle, metric, context)
            cpu_output = next(item for item in cpu_profile.outputs
                              if item.metric_id == metric)
            cuda_output = next(item for item in cuda_profile.outputs
                               if item.metric_id == metric)
            metric_report["metrics"][metric] = {
                "pass": True, "compared_value_count": left.size,
                "artifact_kind": "series" if left.ndim == 2 else "scalar",
                "cpu_shape": list(left.shape), "cuda_shape": list(right.shape),
                "shape_valid": left.shape == right.shape,
                "factor_count": len(cpu_bundle.factor_ids),
                "observation_counts_shape_valid": True,
                "cpu_observation_counts_sha256": cpu_output.observation_counts_sha256,
                "cuda_observation_counts_sha256": cuda_output.observation_counts_sha256,
                "finite_mask_equal": True,
                "observation_counts_equal": bool(
                    (left_counts == right_counts).all()),
                "max_abs_error": 0.0,
                "cpu_values_sha256": cpu_output.values_sha256,
                "cuda_values_sha256": cuda_output.values_sha256,
            }
        comparisons = build_metric_comparison_receipts(
            cpu_bundle, simulated_cuda_bundle, context, metric_report,
            cpu_profile=cpu_profile, cuda_profile=cuda_profile)
        rebound.append(replace(record, cpu=cpu_profile, cuda=cuda_profile,
                               comparison_evidence=comparisons))
    return tuple(rebound)
