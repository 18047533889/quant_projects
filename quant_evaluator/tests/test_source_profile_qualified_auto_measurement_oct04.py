"""Qualified auto measurements may run below their admitted ceiling."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pytest

from quant_evaluator.tests.test_source_profile_measurement_oct04 import sample as actualsample
from quant_evaluator.scripts.source_profile_measurement import build_backend_profile_measurement
from quant_evaluator.runtime.source_route_profiles import route_profile_execution_config_sha256


def _auto_evidence(sample, backend):
    context, cpu, cuda, cpu_run, cuda_run, _ = sample
    if backend == "cuda":
        context = replace(context, gpu_admitted_max_tile_size=5)
    bundle, receipt = (cpu, cpu_run) if backend == "cpu" else (cuda, cuda_run)
    bundle, receipt = deepcopy(bundle), dict(receipt)
    bundle.metadata.update(
        source_qualification_status="qualified_current_source",
        source_qualification_applied=True,
        source_qualification_winner=backend,
        source_auto_policy="qualified_only",
    )
    receipt.update(
        backend_requested="auto", source_auto_policy="qualified_only",
        context_before=context, context_after=context,
    )
    return context, bundle, receipt


@pytest.mark.parametrize("backend,width", [("cpu", 5), ("cuda", 4)])
def test_qualified_auto_cap_binds_real_measurement_below_context_ceiling(
        actualsample, backend, width):
    context, bundle, receipt = _auto_evidence(actualsample, backend)
    measured = build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=width)
    assert measured.backend == backend
    assert measured.source_tile_size == measured.actual_tile_size == width
    assert measured.execution_config_sha256 == route_profile_execution_config_sha256(
        context, backend, width)
    assert measured.outputs and measured.coverage_observed == context.expected_coverage_count


@pytest.mark.parametrize("backend,width", [("cpu", 5), ("cuda", 4)])
@pytest.mark.parametrize("cap", [True, 0, -1, 6])
def test_qualified_auto_rejects_invalid_cap_after_valid_measurement(
        actualsample, backend, width, cap):
    context, bundle, receipt = _auto_evidence(actualsample, backend)
    baseline = build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=width)
    assert baseline.actual_tile_size == width
    with pytest.raises(ValueError):
        build_backend_profile_measurement(
            bundle, receipt, context, correctness_validated=True,
            qualified_execution_cap=cap)


@pytest.mark.parametrize("backend,width,bad_cap", [("cpu", 5, 4), ("cuda", 4, 3)])
def test_qualified_auto_cap_must_match_measured_width(
        actualsample, backend, width, bad_cap):
    context, bundle, receipt = _auto_evidence(actualsample, backend)
    baseline = build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=width)
    assert baseline.actual_tile_size == width
    with pytest.raises(ValueError):
        build_backend_profile_measurement(
            bundle, receipt, context, correctness_validated=True,
            qualified_execution_cap=bad_cap)


@pytest.mark.parametrize("backend,width", [("cpu", 5), ("cuda", 4)])
@pytest.mark.parametrize("mutation", [
    "policy", "status", "applied", "winner", "requested", "receipt_policy",
    "receipt_context", "receipt_cap",
])
def test_qualified_auto_rejects_unbound_qualification_evidence(
        actualsample, backend, width, mutation):
    context, bundle, receipt = _auto_evidence(actualsample, backend)
    baseline = build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=width)
    assert baseline.actual_tile_size == width
    if mutation == "policy":
        bundle.metadata["source_auto_policy"] = "legacy_measured"
    elif mutation == "status":
        bundle.metadata["source_qualification_status"] = "unqualified"
    elif mutation == "applied":
        bundle.metadata["source_qualification_applied"] = False
    elif mutation == "winner":
        bundle.metadata["source_qualification_winner"] = "cpu" if backend == "cuda" else "cuda"
    elif mutation == "requested":
        receipt["backend_requested"] = backend
    elif mutation == "receipt_policy":
        receipt["source_auto_policy"] = "legacy_measured"
    elif mutation == "receipt_context":
        receipt["context_after"] = replace(context, requested_tile_size=17)
    elif mutation == "receipt_cap":
        receipt["effective_max_tile_size"] = width + 1
    with pytest.raises(ValueError):
        build_backend_profile_measurement(
            bundle, receipt, context, correctness_validated=True,
            qualified_execution_cap=width)


def test_default_none_keeps_strict_context_cap_behavior(actualsample):
    context, bundle, receipt = _auto_evidence(actualsample, "cuda")
    receipt["backend_requested"] = "cuda_strict"
    receipt["effective_max_tile_size"] = 5
    assert build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True).actual_tile_size == 4
    receipt["effective_max_tile_size"] = 4
    with pytest.raises(ValueError, match="effective source cap"):
        build_backend_profile_measurement(
            bundle, receipt, context, correctness_validated=True)
    with pytest.raises(ValueError, match="qualified-only auto"):
        build_backend_profile_measurement(
            bundle, receipt, context, correctness_validated=True, qualified_execution_cap=4)
    receipt["backend_requested"] = "auto"
    assert build_backend_profile_measurement(
        bundle, receipt, context, correctness_validated=True,
        qualified_execution_cap=4).actual_tile_size == 4
