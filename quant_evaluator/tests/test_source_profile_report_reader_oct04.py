import json
from dataclasses import asdict

import numpy as np
import pytest

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.runtime.source_route_profiles import (
    BackendRouteProfileMeasurement, CounterbalancedRouteProfileRecord,
    MetricComparisonReceipt, MetricOutputReceipt, SourceRouteProfileContext,
    route_profile_execution_config_sha256,
    validate_source_route_profile_qualification,
)
from quant_evaluator.scripts.source_profile_abba import _oracle_report
from quant_evaluator.scripts.source_profile_report_reader import (
    MAX_REPORT_BYTES, load_source_profile_report, parse_source_profile_report,
)

KIND = "real_cos_profile_abba_f48_pearson.v1"
METRICS = ("pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir")
SHAPE = (2586, 5461, 48)
COVERAGE = (48, 2586 * 48, 48, 48)
RUN_ORDER = ("cpu", "cuda_strict", "cuda_strict", "cpu")


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
        requested_tile_size=16, timing_scope="evaluate_factor_source_batch_wall_v1",
        metric_coverage=tuple(zip(METRICS, COVERAGE)),
        metric_error_tolerances=tuple((m, 1e-10) for m in METRICS),
        live_source_admitted_max_tile_size=16, gpu_admitted_max_tile_size=4)
    outputs = tuple(MetricOutputReceipt(
        m, _h("a"), _h("b"), _h("c"), COVERAGE[i], COVERAGE[i])
        for i, m in enumerate(METRICS))
    comparisons = tuple(MetricComparisonReceipt(
        m, outputs[i].values_sha256, outputs[i].values_sha256,
        outputs[i].finite_mask_sha256, outputs[i].finite_mask_sha256,
        outputs[i].observation_counts_sha256, outputs[i].observation_counts_sha256,
        COVERAGE[i], True, True, 0.0, 1e-10, _h("d"))
        for i, m in enumerate(METRICS))

    def profile(name, width, seconds):
        ranges = tuple((i, min(i+width, SHAPE[-1])) for i in range(0, SHAPE[-1], width))
        scope = "zero_oom_source_equals_compute_v1"
        return BackendRouteProfileMeasurement(
            name, seconds, 0, width, ranges, width, ranges,
            route_profile_execution_config_sha256(context, name, width),
            stable_content_hex(tag="RouteProfileExecutionSchedule.v2",
                fields={"scope": scope, "source_ranges": ranges, "compute_ranges": ranges}),
            scope, True, sum(COVERAGE), sum(COVERAGE), outputs)
    cpu, cuda = profile("cpu", 16, 1.0), profile("cuda", 4, 2.0)
    records = (
        CounterbalancedRouteProfileRecord(context, ("cpu", "cuda"), cpu, cuda, comparisons),
        CounterbalancedRouteProfileRecord(context, ("cuda", "cpu"), cpu, cuda, comparisons),
    )
    validate_source_route_profile_qualification(records, expected_context=context)
    return records


def _oracle_receipts(context):
    factors = SHAPE[-1]
    factor_ids = tuple(f"f-{index}" for index in range(factors))
    scalars = {m: np.zeros(factors, dtype=np.float64)
               for m in METRICS if m != "pearson_ic_series"}
    series = {"pearson_ic_series": np.zeros((SHAPE[0], factors), dtype=np.float64)}
    counts = {m: np.ones(factors, dtype=np.int64) for m in METRICS}
    bundle = BatchEvaluationBundle(
        factor_ids=factor_ids, label_id="fixture", scalar_metrics=scalars,
        series_metrics=series,
        metadata={"source_request_fingerprint": context.request_content_sha256},
        observation_counts=counts)
    expected = {m: {"values": series[m] if m in series else scalars[m],
                    "observation_counts": counts[m]} for m in METRICS}
    oracle = lambda **_: expected
    return tuple(_oracle_report(
        bundle, {}, context, backend=backend, run_index=index, oracle=oracle)
        for index, backend in enumerate(("cpu", "cuda", "cuda", "cpu"))) + (
        _oracle_report(bundle, {}, context, backend="cpu", run_index=4, oracle=oracle),
        _oracle_report(bundle, {}, context, backend="cpu", run_index=5, oracle=oracle),)


def _payload():
    records = _records()
    oracle_reports = _oracle_receipts(records[0].context)
    return {
        "kind": KIND, "status": "complete", "run_started": True,
        "shape": list(SHAPE), "metric_ids": list(METRICS), "manifest_sha256": _h("9"),
        "requested_tile_cap": 16,
        "preflight": {"pass": True, "available_ram_bytes": 50_000_000_000,
            "minimum_available_ram_bytes": 34_359_738_368,
            "cos_cache_disk_free_bytes": 100_000_000_000, "required_disk_bytes": 5_368_709_120},
        "request_shape": list(SHAPE), "oracle": "independent scipy/decimal Pearson chain",
        "profile_records": [asdict(r) for r in records], "run_order": list(RUN_ORDER),
        "oracle_reports": list(oracle_reports[:4]),
        "qualification_winner": "cpu",
        "auto_verification": {"backend_used": "cpu",
            "qualification_status": "qualified_current_source",
            "qualification_applied": True, "qualification_winner": "cpu",
            "output_matches_independent_oracle": True,
            "oracle_report": oracle_reports[4],
            "values_sha256": {m: _h("e") for m in METRICS}},
    }


def _payload_v2():
    payload = _payload()
    payload["kind"] = "real_cos_profile_abba_f48_pearson.v2"
    payload["default_auto_verification"] = dict(payload["auto_verification"])
    payload["default_auto_verification"]["cache_status"] = "cache_hit"
    payload["default_auto_verification"]["oracle_report"] = _oracle_receipts(
        _records()[0].context)[5]
    return payload


def _json(payload):
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def test_roundtrip_typed_records_and_real_qualifier():
    result = parse_source_profile_report(_json(_payload()))
    assert len(result.records) == 2
    assert type(result.records[0]) is CounterbalancedRouteProfileRecord
    assert type(result.records[0].context) is SourceRouteProfileContext
    assert type(result.records[0].cpu) is BackendRouteProfileMeasurement
    assert type(result.records[0].cpu.outputs[0]) is MetricOutputReceipt
    assert type(result.records[0].comparison_evidence[0]) is MetricComparisonReceipt
    assert type(result.records[0].context.request_shape) is tuple
    assert type(result.records[0].cpu.source_ranges) is tuple
    assert result.qualification.winning_backend == "cpu"
    assert validate_source_route_profile_qualification(
        result.records, expected_context=result.records[0].context) == result.qualification


def test_loader_rejects_one_byte_over_limit(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b"x" * (MAX_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        load_source_profile_report(path)


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong_type"])
def test_top_level_schema_is_exact(mutation):
    payload = _payload()
    if mutation == "extra":
        payload["unknown"] = 1
    elif mutation == "missing":
        del payload["qualification_winner"]
    else:
        payload["run_started"] = 1
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))


def test_duplicate_keys_and_nonfinite_constants_rejected():
    raw = _json(_payload()).replace('{"kind":', '{"kind":"bad","kind":', 1)
    with pytest.raises(ValueError, match="duplicate"):
        parse_source_profile_report(raw)
    raw = _json(_payload()).replace('"max_abs_error":0.0', '"max_abs_error":NaN', 1)
    with pytest.raises(ValueError):
        parse_source_profile_report(raw)


@pytest.mark.parametrize("mutation", ["failed_auto", "winner", "shape", "pair_count"])
def test_completion_and_qualification_drift_rejected(mutation):
    payload = _payload()
    if mutation == "failed_auto":
        payload["auto_verification"]["output_matches_independent_oracle"] = False
    elif mutation == "winner":
        payload["qualification_winner"] = "cuda"
    elif mutation == "shape":
        payload["shape"][-1] = 47
    else:
        payload["profile_records"] = payload["profile_records"][:1]
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))


@pytest.mark.parametrize("where", ["context", "measurement", "record", "wrong_type"])
def test_nested_schemas_reject_missing_and_extra_fields(where):
    payload = _payload()
    record = payload["profile_records"][0]
    if where == "context":
        del record["context"]["request_content_sha256"]
    elif where == "measurement":
        record["cpu"]["unknown"] = "x"
    elif where == "wrong_type":
        record["cpu"]["seconds"] = True
    else:
        record["unknown"] = "x"
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))


def test_module_docs_explain_caller_trust_requalification_and_warmup():
    from quant_evaluator.scripts import source_profile_report_reader as reader
    doc = reader.__doc__.lower()
    assert "not signed" in doc
    assert "warm" in doc
    assert "re-capture" in doc


def test_loader_reads_a_valid_json_report(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(_json(_payload()), encoding="utf-8")
    assert load_source_profile_report(path).qualification.winning_backend == "cpu"



def test_v2_report_accepts_sixth_cache_hit_default_auto_receipt():
    result = parse_source_profile_report(_json(_payload_v2()))
    assert result.kind == "real_cos_profile_abba_f48_pearson.v2"
    assert result.qualification.winning_backend == "cpu"


@pytest.mark.parametrize("mutation", ["cache", "status", "oracle"])
def test_v2_report_rejects_bad_sixth_default_auto_receipt(mutation):
    payload = _payload_v2()
    verification = payload["default_auto_verification"]
    if mutation == "cache":
        verification["cache_status"] = "cache_miss"
    elif mutation == "status":
        verification["qualification_status"] = "stale_source"
    else:
        verification["oracle_report"]["run_index"] = 4
    with pytest.raises(ValueError):
        parse_source_profile_report(_json(payload))
