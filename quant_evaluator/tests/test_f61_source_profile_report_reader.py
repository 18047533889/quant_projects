"""Public reader contract for the fixed F61 source-profile report families.

These tests catch a reader that accepts an F61 kind without enforcing its
metric domain, coverage, qualification, oracle receipts, or auto-output binding.
"""
import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    route_profile_execution_config_sha256,
    validate_source_route_profile_qualification,
)
from quant_evaluator.scripts.source_profile_report_reader import parse_source_profile_report


METRICS_15 = (
    "rank_ic", "rank_ic_series", "ic_ir", "ic_std", "ic_median",
    "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir",
    "coverage", "quantile_spread", "quantile_monotonicity",
    "daily_quantile_monotonicity_rate", "turnover", "factor_turnover_rate",
)
METRICS_24 = METRICS_15 + (
    "quantile_curvature", "quantile_tail_asymmetry", "quantile_adjacent_spread",
    "quantile_extreme_cliff", "top_quantile_cliff", "bottom_quantile_cliff",
    "rank_ic_positive_ratio", "recent_3m_rank_ic", "rolling_rank_ic_ir",
)
KINDS = {
    "15": ("real_cos_profile_abba_f61_all15.v1", METRICS_15),
    "24": ("real_cos_profile_abba_f61_all24.v1", METRICS_24),
}
SHAPE = (2586, 5461, 61)
SERIES_METRICS = {"rank_ic_series", "pearson_ic_series"}
ORACLE = "independent all-source metric chain"
RUN_BACKENDS = ("cpu", "cuda", "cuda", "cpu")
RUN_ORDER = ("cpu", "cuda_strict", "cuda_strict", "cpu")


def _h(char):
    return char * 64


def _fixture(metric_ids, *, coverage_override=None, tolerance=1e-10,
             divergent_cuda=False):
    coverage = tuple((coverage_override or {}).get(
        metric, 2586 * 61 if metric in SERIES_METRICS else 61)
        for metric in metric_ids)
    context = SourceRouteProfileContext(
        request_content_sha256=_h("a"), source_identity_sha256=_h("b"),
        source_content_sha256=_h("c"), executable_source_sha256=_h("d"),
        runtime_fingerprint_sha256=_h("e"), package_fingerprint_sha256=_h("f"),
        thread_fingerprint_sha256=_h("1"), device_fingerprint_sha256=_h("2"),
        common_config_fingerprint_sha256=_h("3"), request_shape=SHAPE,
        metric_ids=metric_ids, expected_coverage_count=sum(coverage),
        requested_tile_size=16, timing_scope="evaluate_factor_source_batch_wall_v1",
        metric_coverage=tuple(zip(metric_ids, coverage)),
        metric_error_tolerances=tuple((metric, tolerance) for metric in metric_ids),
        live_source_admitted_max_tile_size=16, gpu_admitted_max_tile_size=16)
    outputs = tuple(MetricOutputReceipt(
        metric, _h("a"), _h("b"), _h("c"), coverage[index], coverage[index])
        for index, metric in enumerate(metric_ids))
    cuda_outputs = tuple(MetricOutputReceipt(
        item.metric_id, _h("e") if divergent_cuda else item.values_sha256,
        item.finite_mask_sha256, item.observation_counts_sha256,
        item.coverage_expected, item.coverage_observed) for item in outputs)
    comparisons = tuple(MetricComparisonReceipt(
        metric, output.values_sha256, cuda_outputs[index].values_sha256,
        output.finite_mask_sha256, cuda_outputs[index].finite_mask_sha256,
        output.observation_counts_sha256, cuda_outputs[index].observation_counts_sha256,
        coverage[index], True, True, 1e-12 if divergent_cuda else 0.0,
        tolerance, _h("d"))
        for index, (metric, output) in enumerate(zip(metric_ids, outputs)))

    def profile(backend, width, seconds):
        ranges = tuple((start, min(start + width, 61)) for start in range(0, 61, width))
        scope = "zero_oom_source_equals_compute_v1"
        backend_outputs = outputs if backend == "cpu" else cuda_outputs
        return BackendRouteProfileMeasurement(
            backend, seconds, 0, width, ranges, width, ranges,
            route_profile_execution_config_sha256(context, backend, width),
            stable_content_hex(tag="RouteProfileExecutionSchedule.v2", fields={
                "scope": scope, "source_ranges": ranges, "compute_ranges": ranges}),
            scope, True, sum(coverage), sum(coverage), backend_outputs)

    cpu, cuda = profile("cpu", 16, 1.0), profile("cuda", 4, 2.0)
    records = (
        CounterbalancedRouteProfileRecord(context, ("cpu", "cuda"), cpu, cuda, comparisons),
        CounterbalancedRouteProfileRecord(context, ("cuda", "cpu"), cpu, cuda, comparisons),
    )
    validate_source_route_profile_qualification(records, expected_context=context)
    return records


def _oracle_report(metrics, backend, run_index, *, coverage_override=None,
                   tolerance=1e-10, corrupt=None):
    receipt = {}
    for metric in metrics:
        count = (coverage_override or {}).get(
            metric, 2586 * 61 if metric in SERIES_METRICS else 61)
        item = {
            "pass": True, "expected_coverage": count, "observed_coverage": count,
            "shape_equal": True, "finite_mask_equal": True, "nan_mask_equal": True,
            "positive_infinity_mask_equal": True, "negative_infinity_mask_equal": True,
            "observation_counts_equal": True, "max_abs_error": 0.0,
            "tolerance": tolerance,
        }
        if metric == corrupt:
            item["observed_coverage"] -= 1
        receipt[metric] = item
    return {"pass": True, "backend": backend, "run_index": run_index, "metrics": receipt}


def _payload(version="24", *, coverage_override=None, tolerance=1e-10,
             divergent_cuda=False):
    kind, metrics = KINDS[version]
    records = _fixture(metrics, coverage_override=coverage_override,
                       tolerance=tolerance, divergent_cuda=divergent_cuda)
    winner_outputs = records[0].cpu.outputs
    metric_outputs = [asdict(output) for output in winner_outputs]
    oracle_reports = [_oracle_report(metrics, backend, index,
                                     coverage_override=coverage_override,
                                     tolerance=tolerance)
                      for index, backend in enumerate(RUN_BACKENDS)]
    auto = {
        "backend_requested": "auto", "source_auto_policy": "qualified_only",
        "backend_used": "cpu", "qualification_status": "qualified_current_source",
        "qualification_applied": True, "qualification_winner": "cpu",
        "output_matches_independent_oracle": True,
        "oracle_report": _oracle_report(metrics, "cpu", 4,
                                         coverage_override=coverage_override,
                                         tolerance=tolerance),
        "metric_outputs": metric_outputs,
        "values_sha256": {metric: _h("a") for metric in metrics},
    }
    return {
        "kind": kind, "status": "complete", "run_started": True,
        "shape": list(SHAPE), "metric_ids": list(metrics), "manifest_sha256": _h("9"),
        "requested_tile_cap": 16,
        "preflight": {"pass": True, "available_ram_bytes": 50_000_000_000,
            "minimum_available_ram_bytes": 34_359_738_368,
            "cos_cache_disk_free_bytes": 100_000_000_000,
            "required_disk_bytes": 5_368_709_120},
        "request_shape": list(SHAPE), "oracle": ORACLE,
        "profile_records": [asdict(record) for record in records],
        "run_order": list(RUN_ORDER), "oracle_reports": oracle_reports,
        "qualification_winner": "cpu", "auto_verification": auto,
        "default_auto_verification": {
            **deepcopy(auto), "cache_status": "cache_hit",
            "oracle_report": _oracle_report(metrics, "cpu", 5,
                                             coverage_override=coverage_override,
                                             tolerance=tolerance),
        },
    }


def _json(payload):
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


@pytest.mark.parametrize("version", ["15", "24"])
def test_f61_report_is_typed_and_requalified_through_public_parser(version):
    payload = _payload(version)
    result = parse_source_profile_report(_json(payload))
    assert result.kind == payload["kind"]
    assert type(result.records[0]) is CounterbalancedRouteProfileRecord
    assert type(result.records[0].context) is SourceRouteProfileContext
    assert type(result.records[0].cpu.outputs[0]) is MetricOutputReceipt
    assert result.qualification.winning_backend == "cpu"
    assert validate_source_route_profile_qualification(
        result.records, expected_context=result.records[0].context) == result.qualification


def test_oracle_metric_object_key_order_is_not_semantic():
    payload = _payload("24")
    for report in [*payload["oracle_reports"],
                   payload["auto_verification"]["oracle_report"],
                   payload["default_auto_verification"]["oracle_report"]]:
        report["metrics"] = dict(reversed(list(report["metrics"].items())))
    result = parse_source_profile_report(_json(payload))
    assert result.kind == KINDS["24"][0]


def test_auto_output_binds_the_winner_without_requiring_cpu_cuda_hash_identity():
    payload = _payload("24", divergent_cuda=True)
    result = parse_source_profile_report(_json(payload))
    assert result.qualification.winning_backend == "cpu"
    assert result.records[0].cpu.outputs[0].values_sha256 != result.records[0].cuda.outputs[0].values_sha256


@pytest.mark.parametrize("version", ["15", "24"])
@pytest.mark.parametrize("mutation", [
    "oracle_missing_metric", "oracle_extra_metric",
    "coverage", "tolerance", "auto_backend_requested", "auto_policy",
    "auto_metric_missing", "auto_metric_extra", "auto_metric_order",
    "auto_output_hash", "auto_mask_hash", "auto_count_hash", "auto_coverage",
    "values_map_missing", "values_map_extra", "values_map_hash",
    "default_output_hash", "default_mask_hash", "default_count_hash",
    "default_coverage",
    "default_index", "default_cache", "cross_kind",
])
def test_f61_reader_rejects_domain_or_receipt_drift(version, mutation):
    payload = _payload(version)
    metrics = KINDS[version][1]
    if mutation == "oracle_missing_metric":
        del payload["oracle_reports"][0]["metrics"][metrics[0]]
    elif mutation == "oracle_extra_metric":
        payload["oracle_reports"][0]["metrics"]["unrequested"] = payload["oracle_reports"][0]["metrics"][metrics[0]]
    elif mutation == "coverage":
        payload["oracle_reports"][0]["metrics"]["rank_ic_series"]["observed_coverage"] -= 1
    elif mutation == "tolerance":
        payload["oracle_reports"][0]["metrics"][metrics[0]]["tolerance"] = 1e-9
    elif mutation == "auto_backend_requested":
        payload["auto_verification"]["backend_requested"] = "cpu"
    elif mutation == "auto_policy":
        payload["auto_verification"]["source_auto_policy"] = "legacy_measured"
    elif mutation.startswith("auto_metric_") and mutation in {
            "auto_metric_missing", "auto_metric_extra", "auto_metric_order"}:
        outputs = payload["auto_verification"]["metric_outputs"]
        if mutation == "auto_metric_missing":
            outputs.pop()
        elif mutation == "auto_metric_extra":
            outputs.append({**outputs[0], "metric_id": "unexpected"})
        else:
            outputs.reverse()
    elif mutation in {"auto_output_hash", "auto_mask_hash", "auto_count_hash", "auto_coverage",
                      "default_output_hash", "default_mask_hash", "default_count_hash",
                      "default_coverage"}:
        surface = "default_auto_verification" if mutation.startswith("default_") else "auto_verification"
        receipt = payload[surface]["metric_outputs"][0]
        suffix = mutation.removeprefix("default_") if mutation.startswith("default_") else mutation
        field = {"auto_output_hash": "values_sha256", "output_hash": "values_sha256",
                 "auto_mask_hash": "finite_mask_sha256", "mask_hash": "finite_mask_sha256",
                 "auto_count_hash": "observation_counts_sha256", "count_hash": "observation_counts_sha256",
                 "auto_coverage": "coverage_observed", "coverage": "coverage_observed"}[suffix]
        receipt[field] = 1 if field == "coverage_observed" else _h("f")
    elif mutation == "default_index":
        payload["default_auto_verification"]["oracle_report"]["run_index"] = 4
    elif mutation == "default_cache":
        payload["default_auto_verification"]["cache_status"] = "cache_miss"
    elif mutation == "values_map_missing":
        del payload["auto_verification"]["values_sha256"][metrics[0]]
    elif mutation == "values_map_extra":
        payload["auto_verification"]["values_sha256"]["unexpected"] = _h("a")
    elif mutation == "values_map_hash":
        payload["auto_verification"]["values_sha256"][metrics[0]] = _h("f")
    else:
        payload["kind"] = KINDS["15" if version == "24" else "24"][0]
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))


@pytest.mark.parametrize("version", ["15", "24"])
@pytest.mark.parametrize("field,value", [
    ("shape", [2586, 5461, 60]), ("requested_tile_cap", 8),
])
def test_f61_reader_rejects_wrong_fixed_shape_or_cap(version, field, value):
    payload = _payload(version)
    payload[field] = value
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))


@pytest.mark.parametrize("version,metric,bad_count", [
    ("15", "rank_ic_series", 61), ("24", "rank_ic_series", 61),
    ("15", "rank_ic", 2586 * 61), ("24", "rank_ic", 2586 * 61),
])
def test_f61_reader_rejects_noncanonical_but_internally_consistent_coverage(
        version, metric, bad_count):
    records = _fixture(KINDS[version][1], coverage_override={metric: bad_count})
    assert validate_source_route_profile_qualification(
        records, expected_context=records[0].context).winning_backend == "cpu"
    payload = _payload(version, coverage_override={metric: bad_count})
    with pytest.raises(ValueError, match="metric coverage"):
        parse_source_profile_report(_json(payload))


@pytest.mark.parametrize("version", ["15", "24"])
def test_f61_reader_rejects_consistent_noncanonical_error_tolerance(version):
    records = _fixture(KINDS[version][1], tolerance=1e-9)
    assert validate_source_route_profile_qualification(
        records, expected_context=records[0].context).winning_backend == "cpu"
    payload = _payload(version, tolerance=1e-9)
    with pytest.raises(ValueError, match="error tolerances"):
        parse_source_profile_report(_json(payload))


@pytest.mark.parametrize("version", ["15", "24"])
def test_f61_provider_returns_unverified_candidate_for_exact_request(version, tmp_path):
    # Parsing a current-kind candidate is not live source qualification.
    from quant_evaluator.runtime.source_qualification_provider import FileSourceQualificationProvider

    path = tmp_path / "f61.json"
    path.write_text(_json(_payload(version)), encoding="utf-8")
    provider = FileSourceQualificationProvider({_h("a"): path})
    lookup = provider.lookup(_h("a"))
    assert lookup.reason_code == "candidate_found"
    assert lookup.candidate is not None
    assert lookup.candidate.records[0].context.metric_ids == KINDS[version][1]
    assert provider.lookup(_h("f")).reason_code == "candidate_not_found"


@pytest.mark.parametrize("version", ["15", "24"])
def test_invalid_f61_candidate_is_not_absence_or_legacy_authority(version, tmp_path):
    # A well-formed but different mask identity must fail closed in the provider.
    from quant_evaluator.runtime.source_qualification_provider import FileSourceQualificationProvider
    from quant_evaluator.runtime.source_auto_authority import legacy_auto_authorized

    payload = _payload(version)
    payload["default_auto_verification"]["metric_outputs"][0]["finite_mask_sha256"] = _h("f")
    path = tmp_path / "invalid-f61.json"
    path.write_text(_json(payload), encoding="utf-8")
    provider = FileSourceQualificationProvider({_h("a"): path})
    lookup = provider.lookup(_h("a"))
    assert lookup.reason_code == "report_invalid"
    assert lookup.candidate is None
    assert legacy_auto_authorized(
        policy="legacy_measured", qualification_status="not_available_legacy_fallback",
        qualification_reason="qualified_cache_miss", provider_status=lookup.reason_code,
        qualification_supplied=False,
    ) is False


@pytest.mark.parametrize("version", ["15", "24"])
@pytest.mark.parametrize("surface", ["oracle_reports", "auto_verification", "default_auto_verification"])
def test_f61_oracle_negative_absolute_error_is_rejected(version, surface):
    payload = _payload(version)
    oracle = payload[surface][0] if surface == "oracle_reports" else payload[surface]["oracle_report"]
    oracle["metrics"][KINDS[version][1][0]]["max_abs_error"] = -5e-324
    with pytest.raises(ValueError, match="nonnegative"):
        parse_source_profile_report(_json(payload))
