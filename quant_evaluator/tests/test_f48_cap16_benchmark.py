"""Scope and harness contract tests for the F48 cap16 default benchmark."""
from types import SimpleNamespace
from pathlib import Path
import threading

import pytest

from quant_evaluator.scripts import benchmark_f48_cap16_auto as candidate
from quant_evaluator.scripts.f48_auto_references import HASH_FIELDS
from quant_evaluator.api import factor_source
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness
from quant_evaluator.scripts.benchmark_f48_cap16_auto import (
    CANDIDATE_STATUS, cap16_candidate_scope,
    CANDIDATE_REASON, EVIDENCE_ID, REFERENCE_EVIDENCE_ID,
)
from quant_evaluator.scripts.f48_auto_references import F48_METRICS, F48_SHAPE


def _args(**changes):
    result = dict(shape=F48_SHAPE, metrics=F48_METRICS,
                  source_dtype="float64", label_dtype="float64",
                  requested_tile_width=16)
    result.update(changes)
    return result


def test_scope_is_exact_and_restores_selector_after_error():
    original = factor_source.select_source_auto_route
    with pytest.raises(RuntimeError, match="synthetic"):
        with cap16_candidate_scope(**_args()) as provenance:
            assert provenance["effective_tile_width"] == 2
            assert provenance["evidence_status"] == CANDIDATE_STATUS
            raise RuntimeError("synthetic")
    assert factor_source.select_source_auto_route is original


@pytest.mark.parametrize("override", [
    {"shape": (2586, 5461, 47)},
    {"source_dtype": "float32"}, {"label_dtype": "float32"},
    {"requested_tile_width": 2},
])
def test_scope_rejects_profile_drift(override):
    with pytest.raises(ValueError, match="exact F48"):
        with cap16_candidate_scope(**_args(**override)):
            pytest.fail("accepted profile drift")


def test_scope_accepts_reordered_unique_metric_set_and_rejects_duplicates():
    reordered = tuple(reversed(F48_METRICS))
    with cap16_candidate_scope(**_args(metrics=reordered)) as provenance:
        assert provenance["effective_tile_width"] == 2
    duplicate = (F48_METRICS[0], F48_METRICS[0], F48_METRICS[2])
    with pytest.raises(ValueError, match="exact F48"):
        with cap16_candidate_scope(**_args(metrics=duplicate)):
            pytest.fail("duplicate metrics admitted")
def test_candidate_identity_is_distinct_from_published_cap2_evidence():
    assert EVIDENCE_ID == "real_cos_f48_mixed_three_cap16_tile2_candidate"
    assert REFERENCE_EVIDENCE_ID == "real_cos_f48_mixed_three_tile2"
    assert CANDIDATE_REASON == "bounded_f48_mixed_three_gpu_cap16_tile2"


def test_scope_rejects_nested_entry():
    with cap16_candidate_scope(**_args()):
        with pytest.raises(RuntimeError, match="non-reentrant"):
            with cap16_candidate_scope(**_args()):
                pytest.fail("nested scope admitted")


def test_scope_rejects_worker_thread():
    errors = []
    def enter():
        try:
            with cap16_candidate_scope(**_args()):
                pass
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=enter)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "main thread" in str(errors[0])


def test_default_tile_flag_is_forwarded_and_reported(monkeypatch):
    class Source:
        factor_ids = ("f",)
        reads = [(0, 1)]
        snapshot_id = "snapshot"
        max_tile_size = 16
        max_object_mib = 128
        prefetch_objects = False
        tile_read_timings = []
        def close(self): pass
    source = Source()
    monkeypatch.setattr(harness, "RealCosSource", lambda *a, **k: source)
    seen = {}
    class Result:
        metadata = {"effective_max_tile_size": 2, "admitted_source_tile_size": 5,
                    "factor_tiles_processed": 1, "backend_used": "cuda"}
    def evaluate(*args, **kwargs):
        seen.update(kwargs)
        return Result()
    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)
    dates = type("Axis", (), {"values": []})()
    result, receipt = harness.run_backend(
        "auto", [("f",)], [("f",)], dates, (), None, "a" * 64, 16, 128,
        object(), ("rank_ic",), expected_auto_cuda=True,
        use_default_tile_size=True)
    assert seen["max_tile_size"] is None
    assert receipt["api_default_tile_size"] is True
    assert receipt["declared_source_tile_size"] == 16
    assert receipt["admitted_source_tile_size"] == 5
    assert receipt["effective_max_tile_size"] == 2
def _mock_full_run(monkeypatch, defect=None):
    dates, assets = range(2586), range(5461)
    labels = SimpleNamespace(values=SimpleNamespace(dtype="float64"),
                             asset_axis=object())
    rows = [tuple([f"row-{i}"]) for i in range(48)]
    records = [f"factor-{i}" for i in range(48)]
    monkeypatch.setattr(candidate, "validate_f48_auto_references",
                        lambda *args: {"verified_source_request_fingerprint": "f" * 64})
    monkeypatch.setattr(candidate, "capture_source_tree", lambda *a, **k: {"hash": "a"})
    monkeypatch.setattr(candidate, "finalize_source_tree", lambda *a, **k: True)
    monkeypatch.setattr(candidate.tiles, "read_manifest", lambda *a: {})
    monkeypatch.setattr(candidate.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(candidate.tiles, "iter_frames", lambda *a, **k: ())
    monkeypatch.setattr(candidate.tiles, "intersect_axes", lambda *a, **k: (dates, assets, rows))
    monkeypatch.setattr(candidate.tiles, "load_labels", lambda *a: (dates, assets, labels))
    monkeypatch.setattr(harness, "preflight", lambda *a: {
        "pass": True, "available_ram_bytes": 64 * 1024**3,
        "minimum_available_ram_bytes": 32 * 1024**3,
        "cos_cache_disk_free_bytes": 64 * 1024**3,
        "required_disk_bytes": 5 * 1024**3})
    monkeypatch.setattr(candidate, "_auto_batch_cuda_rejection", lambda *a: None)
    output = []
    monkeypatch.setattr(harness, "emit_report", lambda report, path: output.append(report))
    fingerprints = {key: "a" * 64 for key in HASH_FIELDS}
    fingerprints["source_request_fingerprint"] = "f" * 64
    calls = []
    ranges = [(start, start + 2) for start in range(0, 48, 2)]
    def fake_backend(backend, *args, **kwargs):
        calls.append((backend, args, kwargs))
        tile_size = args[6]
        metadata = {"backend_used": "cuda", "effective_max_tile_size": 2,
                    **fingerprints}
        if backend == "auto":
            metadata.update(auto_backend_reason=candidate.CANDIDATE_REASON,
                            auto_backend_evidence_id=candidate.EVIDENCE_ID,
                            auto_backend_evidence_status=candidate.CANDIDATE_STATUS)
        if defect == "fingerprint" and backend == "auto":
            metadata["source_request_fingerprint"] = "b" * 64
        receipt = {**fingerprints,
                   "api_default_tile_size": backend == "auto" and kwargs.get("use_default_tile_size") is True,
                   "declared_source_tile_size": tile_size,
                   "admitted_source_tile_size": 5 if backend == "auto" else 2,
                   "effective_max_tile_size": 2,
                   "auto_backend_reason": candidate.CANDIDATE_REASON if backend == "auto" else None,
                   "tile_ranges": ranges,
                   "factor_tiles_processed": 24, "oom_retries": 0}
        if defect == "route" and backend == "auto":
            metadata["effective_max_tile_size"] = 4
            receipt["effective_max_tile_size"] = 4
        if defect == "coverage" and backend == "auto":
            receipt["tile_ranges"] = ranges[:-1]
        if defect == "identity" and backend == "cuda_strict":
            receipt["source_snapshot_id"] = "c" * 64
        return SimpleNamespace(metadata=metadata, factor_ids=tuple(records)), receipt
    monkeypatch.setattr(harness, "run_backend", fake_backend)
    monkeypatch.setattr(harness, "verify_auto_against_reference", lambda *a: {"pass": True})
    monkeypatch.setattr(harness, "compare", lambda *a, **k: {"pass": True})
    args = SimpleNamespace(references=[Path("cpu.json"), Path("cuda.json")],
                           axis_index=None, output=Path("receipt.json"),
                           max_object_mib=128, max_total_mib=4096,
                           max_source_memory_mib=4096)
    return args, calls, output


def test_actual_f48_memory_profile_admits_hint5_but_rejects_declared16():
    memory, admitted = candidate._memory_profile(F48_SHAPE)
    assert admitted == 5
    assert memory["16"]["admitted"] is False
    assert memory["2"]["admitted"] is True


def test_mocked_full_run_separates_default_cap16_and_strict_cap2(monkeypatch):
    args, calls, output = _mock_full_run(monkeypatch)
    candidate.run(args)
    assert output[0]["status"] == "complete"
    auto_call, cuda_call = calls
    assert auto_call[0] == "auto"
    assert auto_call[1][6] == 16
    assert auto_call[2]["use_default_tile_size"] is True
    assert auto_call[2]["expected_auto_cuda"] is True
    assert cuda_call[0] == "cuda_strict"
    assert cuda_call[1][6] == 2
    assert output[0]["memory_admitted_tile_size"] == 5
    assert output[0]["coverage_pass"] and output[0]["identity_pass"]
    assert output[0]["runs"]["auto_default"]["effective_max_tile_size"] == 2


@pytest.mark.parametrize("defect,flag", [
    ("route", "route_pass"), ("fingerprint", "route_pass"),
    ("coverage", "coverage_pass"), ("identity", "identity_pass"),
])
def test_mocked_full_run_fails_closed_on_route_identity_or_reads(monkeypatch, defect, flag):
    args, _, output = _mock_full_run(monkeypatch, defect)
    with pytest.raises(SystemExit):
        candidate.run(args)
    assert output[0]["status"] == "verification_failed"
    assert output[0][flag] is False


def test_preflight_rejection_happens_before_factor_selection(monkeypatch):
    args, _, _ = _mock_full_run(monkeypatch)
    monkeypatch.setattr(harness, "preflight", lambda *a: {"pass": False})
    def unexpected(*args, **kwargs):
        pytest.fail("factor selection ran before preflight passed")
    monkeypatch.setattr(candidate.tiles, "select_source_records", unexpected)
    with pytest.raises(RuntimeError, match="preflight"):
        candidate.run(args)
