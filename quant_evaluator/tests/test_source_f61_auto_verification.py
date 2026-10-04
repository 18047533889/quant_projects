"""Live auto receipts must bind a qualified winner, not merely claimed success."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib

import pytest

from quant_evaluator.tests.test_source_profile_measurement_oct04 import sample
from quant_evaluator.runtime.source_route_profiles import validate_source_route_profile_qualification
from quant_evaluator.scripts.source_profile_measurement import build_counterbalanced_profile_record
from quant_evaluator.scripts.source_f61_auto_verification import verify_source_profile_auto


@pytest.fixture(params=("cpu", "cuda"))
def evidence(sample, request):
    context, cpu, cuda, cpu_run, cuda_run, comparison = sample
    if request.param == "cuda":
        cpu_run["seconds"] = cpu_run["total_wall_seconds"] = 2.0
    first = build_counterbalanced_profile_record(
        cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
        cuda_run_receipt=cuda_run, context=context, comparison_report=comparison,
        execution_order=("cpu", "cuda"), cpu_correctness_validated=True,
        cuda_correctness_validated=True,
    )
    second = replace(first, execution_order=("cuda", "cpu"))
    qualification = validate_source_route_profile_qualification(
        (first, second), expected_context=context)
    assert qualification.winning_backend == request.param
    selected = first.cpu if request.param == "cpu" else first.cuda
    actual = deepcopy(cpu if request.param == "cpu" else cuda)
    oracle = deepcopy(cpu)
    run = deepcopy(cpu_run if request.param == "cpu" else cuda_run)
    run.update(context_before=context, context_after=context,
               backend_requested="auto", source_auto_policy="qualified_only")
    actual.metadata.update(
        source_qualification_status="qualified_current_source",
        source_qualification_applied=True,
        source_qualification_winner=request.param,
        source_auto_policy="qualified_only",
    )
    return dict(bundle=actual, receipt=run, qualification=qualification,
                context=context, oracle_bundle=oracle, selected_profile=selected,
                run_index=4, require_cache_hit=False)


def call(evidence):
    return verify_source_profile_auto(**evidence)


@pytest.mark.parametrize("run_index", (4, 5))
def test_actual_scalar_and_series_receipts_match_winner_and_recompute_oracle(evidence, run_index):
    evidence["run_index"] = run_index
    evidence["require_cache_hit"] = run_index == 5
    if run_index == 5:
        evidence["bundle"].metadata["source_qualification_cache_status"] = "cache_hit"
    result = call(evidence)
    selected = evidence["selected_profile"]
    assert result["metric_outputs"] == [asdict(x) for x in selected.outputs]
    assert result["values_sha256"]["rank_ic_series"] == hashlib.sha256(
        evidence["bundle"].series_metrics["rank_ic_series"].tobytes()).hexdigest()
    assert result["metric_outputs"][0]["coverage_observed"] == 5
    assert result["metric_outputs"][1]["coverage_observed"] == 15
    assert result["oracle_report"]["pass"] is True
    assert result["oracle_report"]["run_index"] == run_index
    assert result["source_auto_policy"] == "qualified_only"
    assert result["backend_requested"] == "auto"
    if run_index == 5:
        assert result["cache_status"] == "cache_hit"
    else:
        assert "cache_status" not in result


@pytest.mark.parametrize("run_index", (True, 4.0, -1, 6))
def test_run_index_is_exact_integer_four_or_five(evidence, run_index):
    assert call(evidence)["oracle_report"]["pass"] is True
    evidence["run_index"] = run_index
    with pytest.raises(ValueError, match="run index"):
        call(evidence)


@pytest.mark.parametrize("mutation", (
    "backend_requested", "backend_used", "policy", "context_before", "context_after",
    "qualification_context", "oom", "width", "schedule", "missing_profile",
    "wrong_profile_backend", "metadata_winner", "metadata_applied",
    "oracle_ids", "oracle_label", "oracle_request",
))
def test_rejects_one_drift_from_a_demonstrably_valid_execution(evidence, mutation):
    # A broken baseline must fail here, not make every negative falsely green.
    assert call(evidence)["oracle_report"]["pass"] is True
    run, bundle = evidence["receipt"], evidence["bundle"]
    context = evidence["context"]
    if mutation == "backend_requested":
        run["backend_requested"] = "cpu"
    elif mutation == "backend_used":
        run["backend_used"] = "cuda" if run["backend_used"] == "cpu" else "cpu"
    elif mutation == "policy":
        run["source_auto_policy"] = "legacy_measured"
    elif mutation in ("context_before", "context_after"):
        run[mutation] = replace(context, request_content_sha256="0" * 64)
    elif mutation == "qualification_context":
        evidence["qualification"] = replace(
            evidence["qualification"], context=replace(context, request_content_sha256="0" * 64))
    elif mutation == "oom":
        run["oom_retries"] = 1
    elif mutation == "width":
        run["actual_source_tile_size"] += 1
    elif mutation == "schedule":
        run["execution_schedule_sha256"] = "0" * 64
    elif mutation == "missing_profile":
        evidence["selected_profile"] = None
    elif mutation == "wrong_profile_backend":
        evidence["selected_profile"] = replace(
            evidence["selected_profile"], backend="cuda" if run["backend_used"] == "cpu" else "cpu")
    elif mutation == "metadata_winner":
        bundle.metadata["source_qualification_winner"] = "other"
    elif mutation == "metadata_applied":
        bundle.metadata["source_qualification_applied"] = 1
    elif mutation == "oracle_ids":
        evidence["oracle_bundle"].factor_ids = tuple(reversed(bundle.factor_ids))
    elif mutation == "oracle_label":
        evidence["oracle_bundle"].label_id = "different-label"
    elif mutation == "oracle_request":
        evidence["oracle_bundle"].metadata["source_request_fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        call(evidence)


@pytest.mark.parametrize("metric", ("rank_ic", "rank_ic_series"))
def test_within_tolerance_value_drift_still_rejects_changed_profile_identity(evidence, metric):
    assert call(evidence)["oracle_report"]["pass"] is True
    values = (evidence["bundle"].scalar_metrics if metric == "rank_ic"
              else evidence["bundle"].series_metrics)[metric]
    values.flat[0] += 1e-12  # Still within 1e-10 oracle tolerance.
    with pytest.raises(ValueError, match="selected qualified profile"):
        call(evidence)


@pytest.mark.parametrize("cache_status", ("supplied", None))
def test_default_auto_requires_real_metadata_cache_hit(evidence, cache_status):
    assert call(evidence)["oracle_report"]["pass"] is True
    evidence.update(run_index=5, require_cache_hit=True)
    evidence["bundle"].metadata["source_qualification_cache_status"] = cache_status
    with pytest.raises(ValueError, match="cache"):
        call(evidence)


def test_independent_reference_failure_is_not_a_claimed_receipt_pass(evidence):
    assert call(evidence)["oracle_report"]["pass"] is True
    evidence["receipt"]["oracle_report"] = {"pass": True}
    evidence["oracle_bundle"].scalar_metrics["rank_ic"][0] += 1.0
    with pytest.raises(ValueError, match="independent oracle mismatch"):
        call(evidence)


@pytest.mark.parametrize("mutation", ("counts", "finite_mask"))
def test_matching_oracle_cannot_override_measured_output_identity(evidence, mutation):
    assert call(evidence)["oracle_report"]["pass"] is True
    actual, oracle = evidence["bundle"], evidence["oracle_bundle"]
    if mutation == "counts":
        actual.observation_counts["rank_ic"][0] += 1
        oracle.observation_counts["rank_ic"][0] += 1
    else:
        actual.scalar_metrics["rank_ic"][0] = float("nan")
        oracle.scalar_metrics["rank_ic"][0] = float("nan")
    with pytest.raises(ValueError, match="selected qualified profile"):
        call(evidence)


@pytest.mark.parametrize("require_cache_hit", (None, 0, 1, True))
def test_cache_requirement_is_exact_boolean_matching_run_index(evidence, require_cache_hit):
    assert call(evidence)["oracle_report"]["pass"] is True
    evidence["require_cache_hit"] = require_cache_hit
    with pytest.raises(ValueError, match="run index"):
        call(evidence)


@pytest.mark.parametrize("run_index", (4, 5))
def test_generated_receipt_passes_the_readers_full_auto_identity_gate(evidence, run_index):
    from quant_evaluator.scripts.source_profile_report_reader import _check_auto_receipt

    evidence.update(run_index=run_index, require_cache_hit=run_index == 5)
    if run_index == 5:
        evidence["bundle"].metadata["source_qualification_cache_status"] = "cache_hit"
    result = call(evidence)
    outputs = evidence["selected_profile"].outputs
    _check_auto_receipt(
        result, evidence["context"], evidence["qualification"].winning_backend,
        run_index, "default_auto_verification" if run_index == 5 else "auto_verification",
        default_auto=run_index == 5,
        expected_value_hashes={item.metric_id: item.values_sha256 for item in outputs},
        expected_outputs=outputs,
    )


def test_qualified_auto_width_below_admitted_ceiling_is_not_rejected(sample):
    context, cpu, cuda, cpu_run, cuda_run, comparison = sample
    context = replace(context, gpu_admitted_max_tile_size=5)
    cpu_run["seconds"] = cpu_run["total_wall_seconds"] = 2.0
    cuda_run["effective_max_tile_size"] = 5  # Explicit run: admission cap, not actual width 4.
    first = build_counterbalanced_profile_record(
        cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
        cuda_run_receipt=cuda_run, context=context, comparison_report=comparison,
        execution_order=("cpu", "cuda"), cpu_correctness_validated=True,
        cuda_correctness_validated=True,
    )
    qualification = validate_source_route_profile_qualification(
        (first, replace(first, execution_order=("cuda", "cpu"))), expected_context=context)
    assert qualification.winning_backend == "cuda"
    assert first.cuda.actual_tile_size == 4
    actual, oracle = deepcopy(cuda), deepcopy(cpu)
    actual.metadata.update(
        source_qualification_status="qualified_current_source",
        source_qualification_applied=True, source_qualification_winner="cuda",
        source_auto_policy="qualified_only")
    run = deepcopy(cuda_run)
    run.update(backend_requested="auto", source_auto_policy="qualified_only",
               context_before=context, context_after=context, effective_max_tile_size=4)
    result = verify_source_profile_auto(
        actual, run, qualification, context, oracle, selected_profile=first.cuda,
        run_index=4, require_cache_hit=False)
    assert result["backend_used"] == "cuda"
    assert result["metric_outputs"] == [asdict(x) for x in first.cuda.outputs]
