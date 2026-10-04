from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.source_auto_authority import validate_source_auto_policy
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness


def _install_lightweight_main_boundaries(monkeypatch, argv):
    records = tuple((f"factor-{i}", f"cos://fixture/{i}", f"{i:064x}", 1)
                    for i in range(61))
    source_rows = tuple((*row, f"etag-{i}", f"{i + 100:064x}")
                        for i, row in enumerate(records))
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    assets = np.asarray(["A", "B", "C"])
    labels = SimpleNamespace(values=np.zeros((2, 3), dtype=np.float64))
    monkeypatch.setattr("sys.argv", ["source-bench", *argv])
    monkeypatch.setattr(harness, "preflight", lambda *args: {"pass": True})
    monkeypatch.setattr(harness.tiles, "read_manifest", lambda *_: object())
    monkeypatch.setattr(harness.tiles, "select_source_records",
                        lambda *_args, **_kwargs: records)
    monkeypatch.setattr(harness.tiles, "intersect_axes",
                        lambda *_args, **_kwargs: (dates, assets, source_rows))
    monkeypatch.setattr(harness.tiles, "load_labels",
                        lambda *_args, **_kwargs: (dates, assets, labels))
    monkeypatch.setattr(harness, "capture_source_tree", lambda *_: {})
    monkeypatch.setattr(harness, "finalize_source_tree", lambda *_: None)
    reports = []
    monkeypatch.setattr(harness, "emit_report", lambda report, _path: reports.append(report))
    monkeypatch.setattr(harness, "_auto_batch_cuda_rejection", lambda *_: None)

    class FakeSource:
        def __init__(self, received_records, _rows, _dates, _assets, _labels,
                     manifest_sha, tile_size, _max_object_mib, **_kwargs):
            self.factor_ids = tuple(row[0] for row in received_records)
            self.snapshot_id = "fixture-snapshot"
            self.manifest_sha256 = manifest_sha
            self.max_tile_size = tile_size
            self.prefetch_objects = False
            self.prefetch_mode = "off"
            self.prefetch_window = 1
            self.reads = []
            self.tile_read_timings = []
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(harness, "RealCosSource", FakeSource)

    return records, source_rows, reports


def _install_fake_evaluator(monkeypatch, *, returned_policy=None, calls=None):
    def evaluate(source, _labels, *, metrics, backend, max_tile_size, gpu_policy,
                 source_auto_policy):
        requested_policy = validate_source_auto_policy(source_auto_policy)
        if calls is not None:
            calls.append((backend, requested_policy))
        width = max_tile_size or source.max_tile_size
        for start in range(0, len(source.factor_ids), width):
            source.reads.append((start, min(start + width, len(source.factor_ids))))
        actual_backend = "cuda" if backend in ("cuda_strict", "auto") else "cpu"
        reason = ("bounded_f61_all_source_15_gpu_tile16" if backend == "auto" else None)
        return SimpleNamespace(metadata={
            "factor_tiles_processed": len(source.reads),
            "backend_used": actual_backend,
            "effective_max_tile_size": width,
            "admitted_source_tile_size": width,
            "factor_tile_size": width if actual_backend == "cuda" else None,
            "oom_retries": 0,
            "source_auto_policy": requested_policy if returned_policy is None else returned_policy,
            "auto_backend_reason": reason,
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)


def _verified_auto_argv():
    return ["--factors", "61", "--tile-size", "16", "--metrics", "all_source",
            "--days", "2", "--assets", "3", "--verify-auto"]


def test_cli_policy_choices_reject_invalid_value_before_preflight_or_manifest_io(
        monkeypatch, capsys):
    # Missing choices would accept or defer a bad policy until after expensive source admission.
    monkeypatch.setattr("sys.argv", ["source-bench", "--factors", "61",
                                      "--source-auto-policy", "surprise"])
    monkeypatch.setattr(harness, "preflight",
                        lambda *_: pytest.fail("policy validation must precede preflight"))
    monkeypatch.setattr(harness.tiles, "read_manifest",
                        lambda *_: pytest.fail("policy validation must precede manifest reads"))

    with pytest.raises(SystemExit) as exc:
        harness.main()

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "policy_arg,expected",
    [(None, "qualified_only"), ("legacy_measured", "legacy_measured")],
)
def test_verify_auto_cli_forwards_default_or_explicit_policy_to_each_real_run_backend(
        monkeypatch, policy_arg, expected):
    argv = _verified_auto_argv()
    if policy_arg is not None:
        argv.extend(["--source-auto-policy", policy_arg])
    _, _, reports = _install_lightweight_main_boundaries(monkeypatch, argv)
    calls = []
    _install_fake_evaluator(monkeypatch, calls=calls)
    # This test isolates policy forwarding; metric parity has dedicated coverage.
    monkeypatch.setattr(harness, "compare", lambda *_args, **_kwargs: {"pass": True})

    harness.main()

    assert calls == [("cpu", expected), ("cuda_strict", expected), ("auto", expected)]
    assert reports[0]["auto_run"]["source_auto_policy"] == expected


def test_run_backend_validates_policy_before_constructing_source(monkeypatch):
    constructed = []
    monkeypatch.setattr(harness, "RealCosSource",
                        lambda *_args, **_kwargs: constructed.append(True))
    with pytest.raises(InvalidContractError):
        harness.run_backend(
            "cpu", (), (), pd.date_range("2024-01-01", periods=1), (), object(),
            "a" * 64, 1, 16, harness.GPUExecutionPolicy(), harness.DEFAULT_METRICS,
            source_auto_policy=True,
        )
    assert constructed == []


def test_run_backend_receipt_uses_and_checks_actual_bundle_policy(monkeypatch):
    records = tuple((f"factor-{i}", f"cos://fixture/{i}", f"{i:064x}", 1)
                    for i in range(2))
    source_rows = tuple((*row, "etag", "b" * 64) for row in records)
    dates = pd.date_range("2024-01-01", periods=1)
    assets = np.asarray(["A"])
    source_seen = []
    reports = []
    _install_lightweight_main_boundaries(monkeypatch, _verified_auto_argv())
    # run_backend's source is small and deterministic; replace just the data source boundary.
    class Source:
        factor_ids = tuple(row[0] for row in records)
        snapshot_id = "snapshot"
        manifest_sha256 = "a" * 64
        max_tile_size = 1
        prefetch_objects = False
        prefetch_mode = "off"
        prefetch_window = 1
        reads = []
        tile_read_timings = []

        def close(self):
            pass

    source = Source()
    monkeypatch.setattr(harness, "RealCosSource", lambda *_args, **_kwargs: source)

    def evaluate(_source, _labels, *, metrics, backend, max_tile_size, gpu_policy,
                 source_auto_policy):
        source_seen.append(source_auto_policy)
        source.reads.extend([(0, 1), (1, 2)])
        return SimpleNamespace(metadata={
            "factor_tiles_processed": 2, "backend_used": "cpu",
            "effective_max_tile_size": 1, "admitted_source_tile_size": 1,
            "source_auto_policy": source_auto_policy,
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)
    _, receipt = harness.run_backend(
        "cpu", records, source_rows, dates, assets, object(), "a" * 64, 1, 16,
        harness.GPUExecutionPolicy(), harness.DEFAULT_METRICS,
    )
    assert source_seen == ["qualified_only"]
    assert receipt["source_auto_policy"] == "qualified_only"

    source.reads.clear()
    def mismatched_evaluate(*_args, **_kwargs):
        source.reads.extend([(0, 1), (1, 2)])
        return SimpleNamespace(metadata={
            "factor_tiles_processed": 2,
            "backend_used": "cpu", "effective_max_tile_size": 1,
            "admitted_source_tile_size": 1,
            "source_auto_policy": "legacy_measured",
        })

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", mismatched_evaluate)
    with pytest.raises(ValueError, match="source API reported a different auto policy"):
        harness.run_backend(
            "cpu", records, source_rows, dates, assets, object(), "a" * 64, 1, 16,
            harness.GPUExecutionPolicy(), harness.DEFAULT_METRICS,
        )
