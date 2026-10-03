"""Typed-reader coverage for the fixed six-metric F48 linear-shape report."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_profile_report_schema import (
    LINEAR_SHAPE_METRICS, LINEAR_SHAPE_ORACLE, LINEAR_SHAPE_REPORT_KIND,
    LINEAR_SHAPE_SCHEMA, REPORT_SHAPE, REPORT_TILE_CAP)
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    route_profile_execution_config_sha256,
    validate_source_route_profile_qualification)
from quant_evaluator.scripts.source_profile_abba import _oracle_report
from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report

SHAPE = REPORT_SHAPE
METRICS = LINEAR_SHAPE_METRICS
COVERAGE = (48,) * 6


def _h(char):
    return char * 64


def _records():
    context = SourceRouteProfileContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        common_config_fingerprint_sha256=_h("3"), request_shape=SHAPE,
        metric_ids=METRICS, expected_coverage_count=sum(COVERAGE),
        requested_tile_size=REPORT_TILE_CAP, timing_scope="evaluate_factor_source_batch_wall_v1",
        metric_coverage=tuple(zip(METRICS, COVERAGE)),
        metric_error_tolerances=tuple((metric, 1e-10) for metric in METRICS),
        live_source_admitted_max_tile_size=REPORT_TILE_CAP, gpu_admitted_max_tile_size=4)
    outputs = tuple(MetricOutputReceipt(
        metric, _h("a"), _h("b"), _h("c"), COVERAGE[i], COVERAGE[i])
        for i, metric in enumerate(METRICS))
    comparisons = tuple(MetricComparisonReceipt(
        metric, outputs[i].values_sha256, outputs[i].values_sha256,
        outputs[i].finite_mask_sha256, outputs[i].finite_mask_sha256,
        outputs[i].observation_counts_sha256, outputs[i].observation_counts_sha256,
        COVERAGE[i], True, True, 0.0, 1e-10, _h("d"))
        for i, metric in enumerate(METRICS))

    def measurement(backend, width, seconds):
        ranges = tuple((start, min(start + width, SHAPE[-1]))
                       for start in range(0, SHAPE[-1], width))
        scope = "zero_oom_source_equals_compute_v1"
        return BackendRouteProfileMeasurement(
            backend, seconds, 0, width, ranges, width, ranges,
            route_profile_execution_config_sha256(context, backend, width),
            stable_content_hex(tag="RouteProfileExecutionSchedule.v2", fields={
                "scope": scope, "source_ranges": ranges, "compute_ranges": ranges}),
            scope, True, sum(COVERAGE), sum(COVERAGE), outputs)

    cpu = measurement("cpu", REPORT_TILE_CAP, 1.0)
    cuda = measurement("cuda", 4, 2.0)
    records = (
        CounterbalancedRouteProfileRecord(context, ("cpu", "cuda"), cpu, cuda, comparisons),
        CounterbalancedRouteProfileRecord(context, ("cuda", "cpu"), cpu, cuda, comparisons),
    )
    validate_source_route_profile_qualification(records, expected_context=context)
    return records


def _oracle_receipt(context, backend, run_index):
    count = SHAPE[-1]
    bundle = BatchEvaluationBundle(
        factor_ids=tuple(f"f-{i}" for i in range(count)), label_id="fixture",
        scalar_metrics={metric: np.zeros(count, dtype=np.float64) for metric in METRICS},
        series_metrics={},
        metadata={"source_request_fingerprint": context.request_content_sha256},
        observation_counts={metric: np.ones(count, dtype=np.int64) for metric in METRICS})
    expected = {metric: {"values": bundle.scalar_metrics[metric],
                         "observation_counts": bundle.observation_counts[metric]}
                for metric in METRICS}
    return _oracle_report(
        bundle, {}, context, backend=backend, run_index=run_index,
        oracle=lambda **_: expected)


def _payload():
    records = _records()
    context = records[0].context
    per_run = tuple(_oracle_receipt(context, backend, index)
                    for index, backend in enumerate(("cpu", "cuda", "cuda", "cpu")))
    auto = {
        "backend_used": "cpu", "qualification_status": "qualified_current_source",
        "qualification_applied": True, "qualification_winner": "cpu",
        "output_matches_independent_oracle": True,
        "oracle_report": _oracle_receipt(context, "cpu", 4),
        "values_sha256": {metric: _h("a") for metric in METRICS},
    }
    default_auto = {**auto, "cache_status": "cache_hit",
                    "oracle_report": _oracle_receipt(context, "cpu", 5)}
    return {        "kind": LINEAR_SHAPE_REPORT_KIND, "status": "complete", "run_started": True,
        "shape": list(SHAPE), "metric_ids": list(METRICS), "manifest_sha256": _h("9"),
        "requested_tile_cap": REPORT_TILE_CAP,
        "preflight": {"pass": True, "available_ram_bytes": 50_000_000_000,
            "minimum_available_ram_bytes": 34_359_738_368,
            "cos_cache_disk_free_bytes": 100_000_000_000,
            "required_disk_bytes": 5_368_709_120},
        "request_shape": list(SHAPE), "oracle": LINEAR_SHAPE_ORACLE,
        "profile_records": [asdict(record) for record in records],
        "run_order": ["cpu", "cuda_strict", "cuda_strict", "cpu"],
        "oracle_reports": list(per_run), "qualification_winner": "cpu",
        "auto_verification": auto, "default_auto_verification": default_auto,
    }


def _parse(payload):
    return parse_source_profile_report(json.dumps(payload, allow_nan=False))


def test_linear_shape_report_roundtrips_typed_records_and_real_qualifier():
    result = _parse(_payload())
    assert result.kind == LINEAR_SHAPE_REPORT_KIND
    assert result.qualification.winning_backend == "cpu"
    assert type(result.records[0]) is CounterbalancedRouteProfileRecord
    assert result.records[0].context.metric_ids == METRICS
    assert len(result.records[0].cpu.outputs) == 6
    assert result.records[0].cpu.outputs[0].coverage_observed == 48
    assert LINEAR_SHAPE_SCHEMA.default_auto_required is True
    assert validate_source_route_profile_qualification(
        result.records, expected_context=result.records[0].context) == result.qualification


def test_linear_shape_report_requires_default_auto_receipt():
    payload = _payload()
    del payload["default_auto_verification"]
    with pytest.raises(ValueError, match="missing=.*default_auto_verification"):
        _parse(payload)


@pytest.mark.parametrize("mutation", ["order", "metric_set"])
def test_linear_shape_report_rejects_noncanonical_metric_order_or_set(mutation):
    payload = _payload()
    if mutation == "order":
        payload["metric_ids"][0], payload["metric_ids"][1] = (
            payload["metric_ids"][1], payload["metric_ids"][0])
    else:
        payload["metric_ids"][0] = "quantile_monotonicity"
    with pytest.raises(ValueError):
        _parse(payload)


@pytest.mark.parametrize("field,value", [
    ("shape", [2586, 5461, 47]), ("requested_tile_cap", 8),
])
def test_linear_shape_report_rejects_nonfixed_shape_and_cap(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError):
        _parse(payload)


def test_linear_shape_report_rejects_oracle_missing_coverage_count():
    payload = _payload()
    payload["oracle_reports"][0]["metrics"][METRICS[0]]["observed_coverage"] = 47
    with pytest.raises(ValueError, match="coverage"):
        _parse(payload)


@pytest.mark.parametrize("section,index", [
    ("auto_verification", 3), ("default_auto_verification", 4),
])
def test_linear_shape_report_requires_auto_oracle_run_indices_four_and_five(section, index):
    payload = _payload()
    payload[section]["oracle_report"]["run_index"] = index
    with pytest.raises(ValueError, match="misbound"):
        _parse(payload)


def test_linear_shape_report_rejects_malformed_receipt_hash():
    payload = _payload()
    payload["profile_records"][0]["cpu"]["outputs"][0]["values_sha256"] = "bad-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        _parse(payload)


def test_linear_shape_report_auto_hash_map_must_cover_exact_schema_metrics():
    payload = _payload()
    del payload["auto_verification"]["values_sha256"][METRICS[-1]]
    with pytest.raises(ValueError, match="fields mismatch"):
        _parse(payload)


@pytest.mark.parametrize("section", ["auto_verification", "default_auto_verification"])
def test_linear_shape_auto_rejects_well_formed_hash_from_different_output(section):
    payload = _payload()
    # Do not let the two receipts alias the same hash map in this fixture.
    payload[section]["values_sha256"] = dict(payload[section]["values_sha256"])
    payload[section]["values_sha256"][METRICS[0]] = _h("f")
    with pytest.raises(ValueError, match="selected backend profile"):
        _parse(payload)


def test_linear_shape_auto_binding_does_not_require_cross_backend_bitwise_equality():
    payload = _payload()
    for record in payload["profile_records"]:
        for output in record["cuda"]["outputs"]:
            output["values_sha256"] = _h("f")
        for comparison in record["comparison_evidence"]:
            comparison["cuda_values_sha256"] = _h("f")
            comparison["max_abs_error"] = 1e-12
    # CPU wins and matches its own profile; CUDA need only satisfy the tolerance.
    assert _parse(payload).qualification.winning_backend == "cpu"
