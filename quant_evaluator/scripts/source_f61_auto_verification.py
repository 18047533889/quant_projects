"""Verify one F61 public-auto run against its qualified source profile."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, SourceRouteProfileContext,
    SourceRouteProfileQualification,
)
from quant_evaluator.scripts.source_profile_abba import _oracle_report
from quant_evaluator.scripts.source_profile_measurement import (
    build_backend_profile_measurement,
)


def verify_source_profile_auto(
    bundle, receipt: Mapping[str, Any], qualification, context, oracle_bundle, *,
    selected_profile, run_index: int, require_cache_hit: bool,
):
    """Validate auto routing, execution identity, cache expectations, and oracle output.

    The returned mapping is a reader-compatible receipt for one qualified-only
    auto verification run. ``oracle_bundle`` is independently produced evidence;
    the caller-provided receipt itself is not a signed attestation.
    Callers must supply the validator-produced qualification and its selected
    backend measurement. This helper checks the live run against those objects;
    it does not authenticate them or revalidate their original ABBA records.
    """
    if type(context) is not SourceRouteProfileContext:
        raise ValueError("context must be an exact source profile context")
    if type(qualification) is not SourceRouteProfileQualification:
        raise ValueError("qualification must be a validated source profile result")
    if qualification.context != context:
        raise ValueError("qualification context differs from auto verification context")
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be a mapping")
    if (type(run_index) is not int or run_index not in (4, 5)
            or type(require_cache_hit) is not bool
            or require_cache_hit is not (run_index == 5)):
        raise ValueError("auto verification run index/cache expectation is invalid")
    if type(bundle) is not BatchEvaluationBundle:
        raise ValueError("bundle must be an exact BatchEvaluationBundle")
    if type(oracle_bundle) is not BatchEvaluationBundle:
        raise ValueError("oracle_bundle must be an exact BatchEvaluationBundle")
    if type(selected_profile) is not BackendRouteProfileMeasurement:
        raise ValueError("selected_profile must be an exact backend profile measurement")

    winner = qualification.winning_backend
    if winner not in ("cpu", "cuda") or selected_profile.backend != winner:
        raise ValueError("selected profile backend differs from qualification winner")
    if (receipt.get("backend_requested") != "auto"
            or receipt.get("source_auto_policy") != "qualified_only"):
        raise ValueError("run receipt is not qualified-only auto")
    if (receipt.get("context_before") != context
            or receipt.get("context_after") != context):
        raise ValueError("auto verification context differs from measured profile")

    metadata = bundle.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("bundle metadata must be a mapping")
    if (metadata.get("source_qualification_status") != "qualified_current_source"
            or metadata.get("source_qualification_applied") is not True
            or metadata.get("source_qualification_winner") != winner
            or receipt.get("backend_used") != winner
            or metadata.get("backend_used") != winner
            or metadata.get("source_auto_policy") != "qualified_only"):
        raise ValueError("auto run did not apply the qualified winning backend")

    expected_source_width = dict(qualification.source_tile_sizes).get(winner)
    expected_actual_width = dict(qualification.actual_tile_sizes).get(winner)
    expected_config = dict(qualification.execution_config_sha256).get(winner)
    if (type(expected_source_width) is not int
            or type(expected_actual_width) is not int
            or type(expected_config) is not str):
        raise ValueError("qualification lacks winner execution identity")
    if (selected_profile.source_tile_size != expected_source_width
            or selected_profile.actual_tile_size != expected_actual_width
            or selected_profile.execution_config_sha256 != expected_config
            or expected_source_width != expected_actual_width):
        raise ValueError("selected profile differs from qualified winner execution config")
    if (type(receipt.get("oom_retries")) is not int
            or receipt["oom_retries"] != 0
            or receipt.get("effective_max_tile_size") != expected_actual_width
            or receipt.get("actual_source_tile_size") != expected_source_width):
        raise ValueError("auto execution width or OOM count differs from qualification")
    compute_width = receipt.get("actual_gpu_factor_tile_size")
    if ((winner == "cuda" and compute_width != expected_actual_width)
            or (winner == "cpu" and compute_width is not None)):
        raise ValueError("auto compute width differs from the qualified profile")

    if require_cache_hit:
        cache_status = metadata.get("source_qualification_cache_status")
        if cache_status != "cache_hit":
            raise ValueError("sixth default-auto verification run did not hit the qualification cache")
    else:
        cache_status = None

    oracle_metadata = oracle_bundle.metadata
    if (bundle.factor_ids != oracle_bundle.factor_ids
            or bundle.label_id != oracle_bundle.label_id
            or metadata.get("source_request_fingerprint") != context.request_content_sha256
            or not isinstance(oracle_metadata, Mapping)
            or oracle_metadata.get("source_request_fingerprint")
            != context.request_content_sha256):
        raise ValueError("independent oracle bundle differs from the live source request")

    oracle = _oracle_report(
        bundle, receipt, context, backend=winner, run_index=run_index,
        oracle=lambda **_: oracle_bundle)
    measurement = build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=expected_actual_width)
    if (measurement.backend != winner
            or measurement.source_tile_size != expected_source_width
            or measurement.actual_tile_size != expected_actual_width
            or measurement.execution_config_sha256 != expected_config):
        raise ValueError("measured execution differs from qualified profile")
    profile_fields = (
        "source_tile_size", "source_ranges", "actual_tile_size", "compute_ranges",
        "execution_config_sha256", "execution_schedule_sha256",
        "execution_schedule_scope", "correctness_validated", "coverage_expected",
        "coverage_observed", "outputs",
    )
    if any(getattr(measurement, field) != getattr(selected_profile, field)
           for field in profile_fields):
        raise ValueError("auto measurement differs from selected qualified profile")

    result = {
        "backend_requested": "auto",
        "source_auto_policy": "qualified_only",
        "backend_used": winner,
        "qualification_status": "qualified_current_source",
        "qualification_applied": True,
        "qualification_winner": winner,
        "output_matches_independent_oracle": True,
        "oracle_report": oracle,
        "values_sha256": {item.metric_id: item.values_sha256
                           for item in measurement.outputs},
        "metric_outputs": [asdict(item) for item in measurement.outputs],
    }
    if require_cache_hit:
        result["cache_status"] = cache_status
    return result
