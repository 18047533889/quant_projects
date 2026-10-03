"""Safety contracts for the explicit real-COS Pearson profile CLI."""
import json
import numpy as np

import pytest

from quant_evaluator.scripts import benchmark_real_cos_profile_abba as driver


def test_default_cli_is_f48_pearson_dry_run_without_cos_reads(monkeypatch, capsys):
    monkeypatch.setattr(driver, "preflight", lambda *args, **kwargs: {
        "pass": True, "available_ram_bytes": 48 * 1024**3,
        "minimum_available_ram_bytes": 32 * 1024**3,
    })

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not read COS or create a source")

    monkeypatch.setattr(driver.tiles, "read_manifest", forbidden)
    monkeypatch.setattr(driver, "_make_cos_source", forbidden)
    assert driver.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "preflight_only"
    assert report["shape"] == [2586, 5461, 48]
    assert report["metric_ids"] == [
        "pearson_ic", "pearson_ic_series", "pearson_ic_std", "pearson_ic_ir"
    ]
    assert report["run_started"] is False


def test_real_run_requires_explicit_output_and_user_axis_index(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CLI must fail before preflight or COS")

    monkeypatch.setattr(driver, "preflight", forbidden)
    monkeypatch.setattr(driver.tiles, "read_manifest", forbidden)
    with pytest.raises(SystemExit) as exc:
        driver.main(["--run"])
    assert exc.value.code == 2


def test_real_run_preflight_rejection_never_reads_manifest_or_opens_source(
    monkeypatch, tmp_path,
):
    gate_calls = []

    def reject(*args, **kwargs):
        gate_calls.append((args, kwargs))
        return {"pass": False, "available_ram_bytes": 1,
                "minimum_available_ram_bytes": 32 * 1024**3}

    def forbidden(*args, **kwargs):
        raise AssertionError("resource rejection must precede COS access")

    monkeypatch.setattr(driver, "preflight", reject)
    monkeypatch.setattr(driver.tiles, "read_manifest", forbidden)
    monkeypatch.setattr(driver, "_make_cos_source", forbidden)
    with pytest.raises(SystemExit, match="preflight"):
        driver.main([
            "--run", "--axis-index", str(tmp_path / "axes.json"),
            "--output", str(tmp_path / "report.json"),
        ])
    assert len(gate_calls) == 1
def test_cpu_warmup_runs_real_api_on_small_typed_tiles_and_closes_source(monkeypatch):
    reads = []
    closes = []
    original_read = driver._WarmSource.read_tile
    original_close = driver._WarmSource.close

    def read(self, start, end):
        reads.append((start, end))
        return original_read(self, start, end)

    def close(self):
        closes.append(self.snapshot_id)
        return original_close(self)

    monkeypatch.setattr(driver._WarmSource, "read_tile", read)
    monkeypatch.setattr(driver._WarmSource, "close", close)
    driver._warm(driver.GPUExecutionPolicy(), "cpu")
    assert reads
    assert all(end - start <= 2 for start, end in reads)
    assert closes == ["warmup-only"]


def test_checked_runner_rechecks_preflight_for_each_backend(monkeypatch):
    gate_calls = []
    backend_calls = []
    rejection_calls = []

    def gate():
        gate_calls.append(len(gate_calls))
        return {"pass": True}

    monkeypatch.setattr(driver, "preflight", gate)
    monkeypatch.setattr(driver, "_auto_batch_cuda_rejection",
                        lambda policy, minimum: rejection_calls.append(minimum) or None)
    monkeypatch.setattr(driver.source_batch, "run_backend",
                        lambda backend, **kwargs: backend_calls.append(backend) or (None, {}))
    policy = driver.GPUExecutionPolicy()
    driver._checked_run_backend("cpu", policy=policy)
    driver._checked_run_backend("cuda_strict", policy=policy)
    assert len(gate_calls) == 2
    assert backend_calls == ["cpu", "cuda_strict"]
    assert rejection_calls == [driver.MIN_EFFECTIVE_VRAM_BYTES]


def test_checked_runner_stops_before_source_when_per_run_gate_or_vram_fails(monkeypatch):
    gate_calls = []
    source_calls = []

    def gate():
        gate_calls.append(None)
        return {"pass": len(gate_calls) == 1}

    monkeypatch.setattr(driver, "preflight", gate)
    monkeypatch.setattr(driver.source_batch, "run_backend",
                        lambda *args, **kwargs: source_calls.append(None))
    policy = driver.GPUExecutionPolicy()
    driver._checked_run_backend("cpu", policy=policy)
    with pytest.raises(SystemExit, match="preflight"):
        driver._checked_run_backend("cpu", policy=policy)
    assert source_calls == [None]

    monkeypatch.setattr(driver, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(driver, "_auto_batch_cuda_rejection",
                        lambda *args: "insufficient_vram")
    with pytest.raises(SystemExit, match="CUDA source run rejected"):
        driver._checked_run_backend("cuda_strict", policy=policy)
    assert source_calls == [None]


def test_checked_runner_gates_cuda_auto_only_when_cuda_is_the_expected_winner(monkeypatch):
    checks = []
    calls = []
    monkeypatch.setattr(driver, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(driver, "_auto_batch_cuda_rejection",
                        lambda *args: checks.append(None) or None)
    monkeypatch.setattr(driver.source_batch, "run_backend",
                        lambda backend, **kwargs: calls.append(backend) or (None, {}))
    policy = driver.GPUExecutionPolicy()
    driver._checked_run_backend("auto", policy=policy, expected_auto_cuda=False)
    assert checks == []
    driver._checked_run_backend("auto", policy=policy, expected_auto_cuda=True)
    assert checks == [None]
    assert calls == ["auto", "auto"]
from test_source_profile_measurement_oct04 import sample as profile_sample


@pytest.mark.parametrize("mutation", ["finite_to_inf", "inf_sign", "count"])
def test_auto_oracle_checker_rejects_mask_sign_and_count_mismatches(
    profile_sample, mutation,
):
    from copy import deepcopy
    from quant_evaluator.scripts.source_profile_abba import _oracle_report

    context, expected, *_ = profile_sample
    actual = deepcopy(expected)
    metric = "rank_ic_series"
    if mutation == "finite_to_inf":
        actual.series_metrics[metric][0, 0] = np.inf
    elif mutation == "inf_sign":
        actual.series_metrics[metric][2, 3] = np.inf
    else:
        actual.observation_counts[metric][0] += 1

    with pytest.raises(ValueError, match="independent oracle mismatch"):
        _oracle_report(actual, {}, context, backend="cpu", run_index=4,
                       oracle=lambda **kwargs: expected)


@pytest.mark.parametrize("cache_status", ["cache_hit", "cache_miss", "producer_fail"])
def test_live_driver_verifies_sixth_default_auto_without_supplied_records(
    monkeypatch, tmp_path, profile_sample, cache_status,
):
    from copy import deepcopy
    from dataclasses import replace
    from types import SimpleNamespace
    from quant_evaluator.scripts.source_profile_measurement import build_counterbalanced_profile_record

    context, cpu, cuda, cpu_run, cuda_run, comparison = profile_sample
    row = build_counterbalanced_profile_record(
        cpu_bundle=cpu, cuda_bundle=cuda, cpu_run_receipt=cpu_run,
        cuda_run_receipt=cuda_run, context=context, comparison_report=comparison,
        execution_order=("cpu", "cuda"), cpu_correctness_validated=True,
        cuda_correctness_validated=True)
    records = (row, replace(row, execution_order=("cuda", "cpu")))
    monkeypatch.setattr(driver, "SHAPE", context.request_shape)
    monkeypatch.setattr(driver, "METRICS", context.metric_ids)
    monkeypatch.setattr(driver, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(driver, "_auto_batch_cuda_rejection", lambda *args: None)
    monkeypatch.setattr(driver.tiles, "read_manifest", lambda *args: None)
    monkeypatch.setattr(driver.tiles, "select_source_records", lambda *args: list(range(5)))
    monkeypatch.setattr(driver.tiles, "read_axis_index", lambda *args: (None, None, []))
    monkeypatch.setattr(driver.tiles, "load_labels", lambda *args: (list(range(3)), list(range(4)), None))
    closes = []
    monkeypatch.setattr(driver, "_make_cos_source", lambda *args, **kwargs:
                        SimpleNamespace(close=lambda: closes.append(True)))
    monkeypatch.setattr(driver, "reference_source_pearson_chain", lambda *args, **kwargs: cpu)
    monkeypatch.setattr(driver, "_runtime_ready", lambda: True)
    monkeypatch.setattr(driver, "_warm", lambda *args: None)
    monkeypatch.setattr(driver, "produce_source_route_profile_abba", lambda **kwargs:
                        SimpleNamespace(records=records, run_order=("cpu", "cuda_strict", "cuda_strict", "cpu"), oracle_reports=[]))
    calls = []
    def run(backend, **kwargs):
        calls.append(kwargs)
        bundle = deepcopy(cpu)
        bundle.metadata.update(source_qualification_applied=True,
            source_qualification_status="qualified_current_source",
            source_qualification_winner="cpu",
            source_qualification_cache_status="supplied" if len(calls) == 1 else cache_status)
        receipt = dict(cpu_run, context_before=context, context_after=context)
        return bundle, receipt
    monkeypatch.setattr(driver, "_checked_run_backend", run)
    reports = []
    monkeypatch.setattr(driver, "_dump", lambda body, path=None: reports.append(body))
    argv = ["--run", "--axis-index", str(tmp_path / "axes"), "--output", str(tmp_path / "report")]
    if cache_status == "producer_fail":
        original = tmp_path / "report"
        original.write_text("previous verified report", encoding="utf-8")
        def fail(**kwargs):
            kwargs["progress_observer"]({"phase": "run_validated", "run_index": 0,
                                         "backend_used": "cpu", "seconds": 1.0})
            raise ValueError("do not expose private source locator")
        monkeypatch.setattr(driver, "produce_source_route_profile_abba", fail)
        with pytest.raises(ValueError, match="private source locator"):
            driver.main(argv)
        assert original.read_text() == "previous verified report"
        diagnostic = json.loads((tmp_path / "report.progress.json").read_text())
        assert diagnostic["status"] == "failed"
        assert diagnostic["error_type"] == "ValueError"
        assert len(diagnostic["validated_runs"]) == 1
        assert "private source locator" not in json.dumps(diagnostic)
        assert "profile_records" not in diagnostic
        assert calls == [] and reports == [] and closes == [True]
        return
    if cache_status == "cache_miss":
        with pytest.raises(ValueError, match="default auto did not reuse"):
            driver.main(argv)
        assert not reports
    else:
        assert driver.main(argv) == 0
        assert reports[0]["kind"].endswith(".v2")
        check = reports[0]["default_auto_verification"]
        assert check["cache_status"] == "cache_hit"
        assert check["oracle_report"]["run_index"] == 5
    assert closes == [True]
    assert calls[0]["source_qualification"] == records
    assert "source_qualification" not in calls[1]


def test_partial_progress_is_bounded_diagnostic_not_a_qualification(tmp_path):
    from quant_evaluator.scripts.source_profile_report_reader import load_source_profile_report
    path = tmp_path / "partial.json"
    events = [{"run_index": 0, "backend_used": "cpu", "seconds": 2.0}]
    driver._save_progress({"kind": "real_cos_profile_abba_f48_pearson.v2"}, path, events,
                          status="failed", error_type="ValueError")
    body = json.loads(path.read_text())
    assert body["status"] == "failed"
    assert body["qualification_available"] is False
    assert body["validated_runs"] == events
    assert "profile_records" not in body
    with pytest.raises(ValueError):
        load_source_profile_report(path)


@pytest.mark.parametrize("events", [[{"seconds": float("nan")}], [{"extra": "x" * 1024**2}]])
def test_progress_rejects_nonfinite_or_oversized_reports_before_writing(tmp_path, events):
    path = tmp_path / "partial.json"
    with pytest.raises(ValueError):
        driver._save_progress({}, path, events)
    assert not path.exists()


def test_live_observer_accepts_real_typed_source_and_keyword_contract(monkeypatch):
    from quant_evaluator.scripts import source_profile_abba as profiles
    source = driver._WarmSource()
    times = tuple(source.time_axis.values)
    labels = driver.LabelBundle("observer-contract", np.zeros((24, 32)), 1,
        decision_time=times, observation_time=times, signal_available_time=times,
        execution_time=times, label_start_time=times,
        label_end_time=tuple(source.time_axis.values + np.timedelta64(1, "D")),
        asset_axis=source.asset_axis)
    captured = []
    token = object()
    monkeypatch.setattr(profiles.source_profile_router, "capture_source_route_profile_context",
                        lambda **kwargs: captured.append(kwargs) or token)
    policy = driver.GPUExecutionPolicy()
    assert profiles.live_source_profile_context_observer(
        phase="before", source=source, labels=labels, metrics=driver.METRICS,
        requested_tile_size=16, policy=policy) is token
    assert captured[0]["source"] is source
    assert captured[0]["metadata"].factor_ids == source.factor_ids
    assert captured[0]["requested_tile_size"] == 16
    assert captured[0]["policy"] is policy
    assert len(captured[0]["request_fingerprint"]) == 64
