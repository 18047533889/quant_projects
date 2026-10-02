"""End-to-end orchestration contracts without allocating the real F48 panel."""
from types import SimpleNamespace
import numpy as np
import pytest
from quant_evaluator.scripts import benchmark_f48_cap16_auto as runner


@pytest.mark.parametrize("fault", ["none", "ordinary", "hint", "ranges", "fingerprint", "provenance"])
def test_default_cap16_workflow_separates_declared_admitted_and_effective_width(monkeypatch, tmp_path, fault):
    fingerprint = "f" * 64
    reference = {"verified_source_request_fingerprint": fingerprint}
    monkeypatch.setattr(runner, "validate_f48_auto_references", lambda *a: reference)
    monkeypatch.setattr(runner, "capture_source_tree", lambda *a, **k: {"hash": "before"})
    monkeypatch.setattr(runner.harness, "preflight", lambda *a: {"pass": True})
    monkeypatch.setattr(runner.tiles, "read_manifest", lambda *a: {})
    records = tuple((f"f{i}",) for i in range(48))
    monkeypatch.setattr(runner.tiles, "select_source_records", lambda *a: records)
    monkeypatch.setattr(runner.tiles, "iter_frames", lambda *a, **k: iter(()))
    monkeypatch.setattr(runner.tiles, "intersect_axes",
                        lambda *a: (range(2586), range(5461), records))
    labels = SimpleNamespace(values=np.ones(1, dtype=np.float64))
    monkeypatch.setattr(runner.tiles, "load_labels", lambda dates, assets, *a: (dates, assets, labels))
    monkeypatch.setattr(runner, "_auto_batch_cuda_rejection", lambda *a: None)
    monkeypatch.setattr(runner.harness, "verify_auto_against_reference", lambda *a: {"pass": True})
    monkeypatch.setattr(runner.harness, "compare", lambda *a, **k: {"pass": True})
    calls, reports = [], []

    def fake_backend(backend, *args, **kwargs):
        calls.append((backend, args[6], kwargs))
        is_auto = backend == "auto"
        if is_auto:
            assert args[6] == 16
            assert kwargs["use_default_tile_size"] is True
            assert kwargs["expected_auto_cuda"] is True
            route = runner.factor_source.select_source_auto_route(
                shape=(2586, 5461, 48), metrics=runner.F48_METRICS,
                source_dtype="float64", label_dtype="float64",
                requested_tile_width=16)
            assert route.evidence_id == (
                runner.REGISTERED_EVIDENCE_ID if fault == "ordinary" else runner.EVIDENCE_ID)
            assert route.effective_tile_width == 2
        else:
            assert args[6] == 2
            assert not kwargs.get("use_default_tile_size", False)
        metadata = {
            "auto_backend_reason": runner.CANDIDATE_REASON,
            "auto_backend_evidence_status": (
                "measured_source_ab" if fault == "ordinary" else runner.CANDIDATE_STATUS),
            "auto_backend_evidence_id": (
                runner.REGISTERED_EVIDENCE_ID if fault == "ordinary" else runner.EVIDENCE_ID),
            "source_request_fingerprint": fingerprint,
            "effective_max_tile_size": 2,
        }
        receipt = {
            "api_default_tile_size": is_auto,
            "declared_source_tile_size": 16 if is_auto else 2,
            "admitted_source_tile_size": 5 if is_auto else 2,
            "effective_max_tile_size": 2,
            "tile_ranges": [(i, i + 2) for i in range(0, 48, 2)],
            "factor_tiles_processed": 24, "oom_retries": 0,
        }
        receipt.update({key: fingerprint for key in runner.HASH_FIELDS})
        if is_auto and fault == "hint":
            receipt["admitted_source_tile_size"] = 2
        if fault == "ranges":
            receipt["tile_ranges"] = [(0, 48)]
        if is_auto and fault == "fingerprint":
            metadata["source_request_fingerprint"] = "bad"
        return SimpleNamespace(metadata=metadata), receipt

    monkeypatch.setattr(runner.harness, "run_backend", fake_backend)
    def finalize(report, *a, **k):
        if fault == "provenance":
            report["status"] = "source_tree_changed"
        return fault != "provenance"
    monkeypatch.setattr(runner, "finalize_source_tree", finalize)
    monkeypatch.setattr(runner.harness, "emit_report", lambda report, output: reports.append(report))
    args = SimpleNamespace(references=(tmp_path / "cpu.json", tmp_path / "cuda.json"),
                           axis_index=None, output=tmp_path / "out.json",
                           max_object_mib=128, max_total_mib=4096,
                           max_source_memory_mib=4096)
    args.ordinary = fault == "ordinary"
    if args.ordinary:
        monkeypatch.setattr(runner, "cap16_candidate_scope",
                            lambda **kw: pytest.fail("ordinary auto injected a candidate"))
    original = runner.factor_source.select_source_auto_route
    if fault in {"none", "ordinary"}:
        runner.run(args)
        assert reports[0]["status"] == "complete"
        assert reports[0]["benchmark_only"] is (fault != "ordinary")
        assert reports[0]["registered_route_verification"] is (fault == "ordinary")
    else:
        with pytest.raises(SystemExit):
            runner.run(args)
        assert reports[0]["status"] != "complete"
    assert runner.factor_source.select_source_auto_route is original
    assert [(backend, width) for backend, width, _ in calls] == [("auto", 16), ("cuda_strict", 2)]
    assert reports[0]["memory_admitted_tile_size"] == 5
    assert reports[0]["source_memory_preflight"]["16"]["admitted"] is False
    assert reports[0]["source_memory_preflight"]["2"]["admitted"] is True
