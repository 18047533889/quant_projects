"""CPU-only exact-envelope and historical-receipt tests for F48 Pearson auto."""
from pathlib import Path
import json

import pytest

from quant_evaluator.runtime.source_auto_evidence import select_source_auto_route
from quant_evaluator.scripts.f48_pearson_auto_references import METRICS, SHAPE, validate_references

def query(**changes):
    args = dict(shape=SHAPE, metrics=METRICS, source_dtype="float64",
                label_dtype="float64", requested_tile_width=4)
    args.update(changes)
    return select_source_auto_route(**args)

@pytest.mark.parametrize("cap,evidence", [
    (4, "real_cos_f48_pearson_chain_cap4_tile4"),
    (16, "real_cos_f48_pearson_chain_cap16_tile4"),
])
def test_f48_pearson_routes_are_exact(cap, evidence):
    route = query(requested_tile_width=cap)
    assert route.evidence_id == evidence
    assert route.effective_tile_width == 4
    assert route.evidence_status == "measured_source_ab"

@pytest.mark.parametrize("changes", [
    {"requested_tile_width": n} for n in (1, 3, 5, 8, 15, 17, 32)
] + [
    {"shape": (2585, 5461, 48)}, {"shape": (2586, 5461, 47)},
    {"shape": (2586, 5461, 49)}, {"source_dtype": "float32"},
    {"label_dtype": "float32"}, {"metrics": METRICS[:-1]},
    {"metrics": METRICS + ("coverage",)},
])
def test_f48_pearson_route_does_not_generalize(changes):
    assert query(**changes) is None

def test_width4_reference_pair_is_valid_and_opposite_order():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json",
             root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json"]
    reference = validate_references(paths)
    assert reference["historical_source_aggregate_sha256"] == "b13bb54505d6da85b29726dd4736e4da71369a4f2074774282f9e0388a97046f"

@pytest.mark.parametrize("mutation", [
    ("comparison.compared_metric_count", 1),
    ("source_adapter", "legacy"),
    ("runs.1.backend_used", "cpu"),
    ("runs.1.cos_prefetch", "off"),
    ("runs.1.prefetch_window", 1),
    ("runs.1.max_source_memory_bytes", 1),
    ("caller_settings.cos_prefetch_workers", 1),
    ("caller_settings.max_prefetch_memory_mib", 0),
    ("preflight_before_cpu.available_ram_bytes", 1),
    ("preflight_before_cuda.required_disk_bytes", 1),
    ("runs.1.seconds", float("nan")),
    ("runs.1.seconds", -1),
    ("runs.1.seconds", 1000),
    ("runs.1.source_snapshot_id", "bad"),
    ("comparison.metrics.pearson_ic_series.artifact_kind", "scalar"),
    ("comparison.metrics.pearson_ic_series.cpu_shape", [48]),
    ("comparison.metrics.pearson_ic_series.max_abs_error", 1.0),
    ("comparison.metrics.pearson_ic_series.cuda_observation_counts_sha256", "0" * 64),
    ("source_provenance_verification.pass", False),
    ("source_provenance_verification.after_aggregate_sha256", "bad"),
    ("second.comparison.metrics.pearson_ic.cuda_values_sha256", "0" * 64),
])
def test_reference_validator_rejects_corrupted_receipt_field(tmp_path, mutation):
    root = Path(__file__).resolve().parents[2]
    source_paths = [root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cpu_first_20261003.json",
                    root / "quant_evaluator/docs/benchmarks/real_cos_f48_pearson_chain_width4_cuda_first_20261003.json"]
    data = [json.loads(p.read_text()) for p in source_paths]
    field, value = mutation
    if field.startswith("second."):
        target = data[1]
        field = field[len("second."):]
    else:
        target = data[0]
    parts = field.split(".")
    if parts[0] == "runs":
        target["runs"][int(parts[1])][parts[2]] = value
    else:
        node = target
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    paths = []
    for i, report in enumerate(data):
        path = tmp_path / f"report-{i}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    with pytest.raises(ValueError):
        validate_references(paths)


def test_reference_validator_rejects_oversized_and_non_object_reports(tmp_path):
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        validate_references((oversized, oversized))
    non_object = tmp_path / "array.json"
    non_object.write_text("[]")
    with pytest.raises(ValueError):
        validate_references((non_object, non_object))

def _install_driver_cpu_stubs(monkeypatch, *, fault=None, second_gate_ok=True, initial_gate_ok=True):
    from types import SimpleNamespace
    from quant_evaluator.scripts import benchmark_f48_pearson_default_auto as driver

    valid_gate = {"pass": True, "available_ram_bytes": 40 * 1024**3,
                  "minimum_available_ram_bytes": 32 * 1024**3,
                  "cos_cache_disk_free_bytes": 8 * 1024**3,
                  "required_disk_bytes": 5 * 1024**3}
    low_gate = dict(valid_gate, available_ram_bytes=20 * 1024**3)
    gates = [valid_gate if initial_gate_ok else low_gate, valid_gate if second_gate_ok else low_gate]
    monkeypatch.setattr(driver, "validate_references", lambda _paths: {
        "verified_source_request_fingerprint": "f" * 64,
        "historical_source_aggregate_sha256": "b" * 64})
    monkeypatch.setattr(driver, "capture_source_tree", lambda *_a, **_k: {"before": True})
    def finalize(report, *_a, **_k):
        report["source_provenance_verification"] = {"pass": fault != "provenance"}
        report["source_provenance"] = {"aggregate_sha256": "c" * 64}
    monkeypatch.setattr(driver, "finalize_source_tree", finalize)
    monkeypatch.setattr(driver.harness, "preflight", lambda *_a, **_k: gates.pop(0))
    monkeypatch.setattr(driver.harness, "emit_report", lambda report, _path: emitted.append(report))
    monkeypatch.setattr(driver.tiles, "read_manifest", lambda _sha: {})
    records = [("factor-" + str(i),) for i in range(48)]
    monkeypatch.setattr(driver.tiles, "select_source_records", lambda *_a, **_k: records)
    dates = list(range(driver.SHAPE[0]))
    assets = list(range(driver.SHAPE[1]))
    source_rows = [("source",)]
    monkeypatch.setattr(driver.tiles, "intersect_axes",
                        lambda *_a, **_k: (dates, assets, source_rows))
    labels = SimpleNamespace(values=SimpleNamespace(dtype="float64"))
    monkeypatch.setattr(driver.tiles, "load_labels", lambda *_a, **_k: (dates, assets, labels))
    monkeypatch.setattr(driver, "GPUExecutionPolicy", lambda: object())
    monkeypatch.setattr(driver, "_auto_batch_cuda_rejection", lambda *_a, **_k: None)
    metadata = {"auto_backend_evidence_id": driver.ROUTE_ID,
                "auto_backend_reason": driver.ROUTE_REASON,
                "source_request_fingerprint": "f" * 64}
    api_receipt = {
        "backend_requested": "auto", "backend_used": "cuda",
        "auto_backend_reason": driver.ROUTE_REASON,
        "auto_backend_evidence_id": driver.ROUTE_ID,
        "auto_backend_evidence_version": driver.SOURCE_AUTO_EVIDENCE_VERSION,
        "auto_backend_evidence_artifacts": ("measured_cpu_first.json",
                                            "measured_cuda_first.json"),
        "auto_backend_evidence_status": "measured_source_ab",
        "source_request_fingerprint": "f" * 64,
        "effective_max_tile_size": 4, "admitted_source_tile_size": 4,
        "metric_backends": {metric: "cuda" for metric in driver.METRICS},
    }
    api_receipt["receipt_hash"] = driver.stable_content_hex(
        tag="FactorSourceBatchExecutionReceipt.v1", fields=api_receipt)
    metadata["execution_receipt"] = api_receipt
    if fault == "route":
        metadata["auto_backend_evidence_id"] = "wrong"
    if fault == "api_receipt_missing_receipt":
        metadata.pop("execution_receipt")
    if fault and fault.startswith("api_receipt_missing_"):
        api_receipt.pop(fault.removeprefix("api_receipt_missing_"), None)
    if fault and fault.startswith("api_receipt_wrong_"):
        api_receipt[fault.removeprefix("api_receipt_wrong_")] = "wrong"
    if fault and fault.startswith("api_receipt_empty_"):
        api_receipt[fault.removeprefix("api_receipt_empty_")] = ""
    if fault and fault != "api_receipt_empty_receipt_hash":
        api_receipt.pop("receipt_hash", None)
        api_receipt["receipt_hash"] = driver.stable_content_hex(
            tag="FactorSourceBatchExecutionReceipt.v1", fields=api_receipt)
    if fault == "api_receipt_tampered_receipt_hash":
        api_receipt["receipt_hash"] = "0" * 64
    auto = SimpleNamespace(metadata=metadata)
    receipt = {"api_default_tile_size": True, "declared_source_tile_size": 16,
               "admitted_source_tile_size": 4, "effective_max_tile_size": 4,
               "tile_ranges": [(i, i + 4) for i in range(0, 48, 4)],
               "factor_tiles_processed": 12, "oom_retries": 0,
               "backend_used": "cuda"}
    if fault == "coverage":
        receipt["tile_ranges"] = [(0, 48)]
    monkeypatch.setattr(driver.harness, "run_backend", lambda *_a, **_k: (auto, receipt))
    parity = {"pass": fault != "parity"}
    monkeypatch.setattr(driver.harness, "verify_auto_against_reference",
                        lambda *_a, **_k: parity)
    emitted = []
    args = SimpleNamespace(
        references=(Path("cpu.json"), Path("cuda.json")), axis_index=None,
        output=Path("unused.json"), max_object_mib=128, max_total_mib=4096,
        max_source_memory_mib=4096)
    return driver, args, emitted, receipt


def test_default_auto_driver_mock_positive_control(monkeypatch):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(monkeypatch)
    driver.run(args)
    report = emitted[-1]
    assert report["status"] == "complete"
    assert report["route_pass"] and report["coverage_pass"]
    api_receipt = report["run"]["api_execution_receipt"]
    assert api_receipt["auto_backend_evidence_id"] == driver.ROUTE_ID
    assert api_receipt["auto_backend_evidence_version"] == driver.SOURCE_AUTO_EVIDENCE_VERSION
    assert api_receipt["auto_backend_evidence_artifacts"] == (
        "measured_cpu_first.json", "measured_cuda_first.json")
    assert driver._valid_api_execution_receipt(
        json.loads(json.dumps(api_receipt)), request_fingerprint="f" * 64,
        admitted_source_tile_size=4)
    assert report["preflight_initial"]["pass"]
    assert report["preflight_before_evaluation"]["pass"]
    assert report["source_provenance_verification"]["pass"]


@pytest.mark.parametrize("fault", [
    "api_receipt_missing_receipt",
    "api_receipt_missing_auto_backend_evidence_version",
    "api_receipt_empty_auto_backend_evidence_version",
    "api_receipt_empty_auto_backend_evidence_artifacts",
    "api_receipt_empty_receipt_hash",
    "api_receipt_wrong_auto_backend_evidence_id",
    "api_receipt_wrong_auto_backend_reason",
    "api_receipt_wrong_auto_backend_evidence_version",
    "api_receipt_wrong_source_request_fingerprint",
    "api_receipt_wrong_effective_max_tile_size",
    "api_receipt_wrong_admitted_source_tile_size",
    "api_receipt_wrong_metric_backends",
    "api_receipt_tampered_receipt_hash",
])
def test_default_auto_driver_mock_rejects_incomplete_api_receipt(monkeypatch, fault):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(monkeypatch, fault=fault)
    with pytest.raises(SystemExit) as exc:
        driver.run(args)
    assert exc.value.code == 1
    report = emitted[-1]
    assert report["status"] == "verification_failed"
    assert report["route_pass"] is False
    assert "api_execution_receipt" in report["run"]

@pytest.mark.parametrize("fault,failed_field", [
    ("route", "route_pass"), ("coverage", "coverage_pass"),
    ("parity", "reference_comparison"),
])
def test_default_auto_driver_mock_rejects_specific_qualification_failure(
        monkeypatch, fault, failed_field):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(monkeypatch, fault=fault)
    with pytest.raises(SystemExit) as exc:
        driver.run(args)
    assert exc.value.code == 1
    report = emitted[-1]
    assert report["status"] == "verification_failed"
    if failed_field == "reference_comparison":
        assert report[failed_field]["pass"] is False
    else:
        assert report[failed_field] is False


def test_default_auto_driver_mock_rejects_provenance_drift(monkeypatch):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(monkeypatch, fault="provenance")
    with pytest.raises(SystemExit) as exc:
        driver.run(args)
    assert exc.value.code == 1
    report = emitted[-1]
    assert report["status"] == "source_tree_changed"
    assert report["source_provenance_verification"]["pass"] is False


def test_default_auto_driver_mock_initial_preflight_stops_before_preparation(monkeypatch):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(
        monkeypatch, initial_gate_ok=False)
    monkeypatch.setattr(driver.tiles, "select_source_records",
                        lambda *_a, **_k: pytest.fail("source selection must not start"))
    monkeypatch.setattr(driver.tiles, "load_labels",
                        lambda *_a, **_k: pytest.fail("label/source preparation must not start"))
    monkeypatch.setattr(driver.harness, "run_backend",
                        lambda *_a, **_k: pytest.fail("evaluation must not start"))
    with pytest.raises(SystemExit) as exc:
        driver.run(args)
    assert exc.value.code == 1
    report = emitted[-1]
    assert report["failure_stage"] == "initial"
    assert report["evaluation_started"] is False
    assert report["preflight_before_evaluation"] is None


def test_default_auto_driver_mock_second_preflight_stops_before_evaluation(monkeypatch):
    driver, args, emitted, _receipt = _install_driver_cpu_stubs(
        monkeypatch, second_gate_ok=False)
    evaluation_calls = []
    monkeypatch.setattr(driver.harness, "run_backend",
                        lambda *_a, **_k: evaluation_calls.append(True))
    with pytest.raises(SystemExit) as exc:
        driver.run(args)
    assert exc.value.code == 1
    report = emitted[-1]
    assert report["status"] == "preflight_rejected"
    assert report["failure_stage"] == "before_evaluation"
    assert report["evaluation_started"] is False
    assert report["preflight_initial"]["available_ram_bytes"] >= 32 * 1024**3
    assert report["preflight_before_evaluation"]["available_ram_bytes"] < 32 * 1024**3
    assert not evaluation_calls
