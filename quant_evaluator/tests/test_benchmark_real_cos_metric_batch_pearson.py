"""CLI contract for the bounded F32 Pearson-chain public A/B."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_factor_batch as profile
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


from quant_evaluator.tests.benchmark_repeat_receipt_fixture import fake_repeat_receipts

def _artifact():
    return {
        "descriptor": {"type": "ScalarMetricArtifact", "artifact_kind": "scalar"},
        "values": [0.125], "finite_mask": [True], "valid_mask": [True],
        "counts": [32], "provenance_observation_counts": [32],
        "provenance": {"source": "bound_cos"},
        "metric_values": [{"metric_id": "pearson_ic", "value": 0.125,
                           "valid": True, "observation_count": 32,
                           "metric_version": "1", "warnings": [],
                           "sample_unit": "date"}],
    }


def test_pearson_chain_comparison_checks_complete_public_artifact(monkeypatch):
    monkeypatch.setattr(benchmark, "METRICS", benchmark.PEARSON_CHAIN)
    left = {"config_hash": "same", "artifacts": {
        metric: _artifact() for metric in benchmark.PEARSON_CHAIN}}
    right = {"config_hash": "same", "artifacts": {
        metric: _artifact() for metric in benchmark.PEARSON_CHAIN}}
    assert benchmark._compare(left, right)["pass"]
    for field, changed in (("descriptor", {"type": "wrong"}),
                           ("values", [0.5]), ("finite_mask", [False]),
                           ("valid_mask", [False]), ("counts", [31]),
                           ("provenance_observation_counts", [31]),
                           ("provenance", {"source": "wrong"}),
                           ("metric_values", [])):
        right["artifacts"]["pearson_ic"][field] = changed
        assert not benchmark._compare(left, right)["metrics"]["pearson_ic"]["pass"]
        right["artifacts"]["pearson_ic"][field] = _artifact()[field]


def test_f32_pearson_preflight_then_six_order_run(monkeypatch, capsys):
    batch = SimpleNamespace(num_times=2586, num_assets=5461, num_factors=32,
                            values=SimpleNamespace(dtype=np.dtype("float64")))
    calls = []
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: calls.append(("preflight", args)) or {"status": "ready"})
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: calls.append(("load", kwargs)) or (batch, object(), {}))

    def fake_run_one(context, backend, repeats, timeout_s):
        calls.append(("run", backend, benchmark.METRICS))
        return {"backend_requested": backend, "backend_used": backend,
                "auto_backend_reason": None, "cold_s": 0.1,
                "warm_median_s": 0.1, "peak_vram": None, "peak_rss_kib": 100,
                "config_hash": "same",
                "artifacts": {metric: _artifact() for metric in benchmark.METRICS},
                "repeat_receipts": fake_repeat_receipts(backend, benchmark.METRICS)}

    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py",
                                      "--factors", "32", "--metrics",
                                      ",".join(benchmark.PEARSON_CHAIN)])
    benchmark.main()
    assert [kind for kind, *_ in calls] == ["preflight"]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "preflight_only"

    calls.clear()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--run"])
    benchmark.main()
    assert [kind for kind, *_ in calls] == ["preflight", "load"] + ["run"] * 6
    assert [row[1] for row in calls if row[0] == "run"] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu"]
    assert all(row[2] == benchmark.PEARSON_CHAIN for row in calls if row[0] == "run")
    assert calls[1][1]["max_total_mib"] == 2048
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parity_pass"]


def test_pearson_chain_rejects_other_factor_counts_before_loading(monkeypatch):
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: pytest.fail("unexpected COS load"))
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py",
                                      "--factors", "12", "--metrics",
                                      ",".join(benchmark.PEARSON_CHAIN)])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2

@pytest.mark.parametrize("metric", benchmark.PEARSON_CHAIN)
def test_f32_pearson_singleton_preflights_without_loading(monkeypatch, capsys, metric):
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: pytest.fail("preflight must precede COS loading"))
    calls = []
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: calls.append(args) or {"status": "ready"})
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py",
                                      "--factors", "32", "--metrics", metric])
    benchmark.main()
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "preflight_only"


def test_f32_pearson_singleton_rejects_low_memory_before_loading(monkeypatch):
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: pytest.fail("low-memory request must not load COS"))
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: {"status": "insufficient_resources"})
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py",
                                      "--factors", "32", "--metrics", "pearson_ic", "--run"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


def test_f32_pearson_singleton_runs_six_interleaved_workers_after_preflight(monkeypatch, capsys):
    metric = "pearson_ic"
    batch = SimpleNamespace(num_times=2586, num_assets=5461, num_factors=32,
                            values=SimpleNamespace(dtype=np.dtype("float64")))
    calls = []
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: calls.append(("preflight",)) or {"status": "ready"})
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: calls.append(("load",)) or (batch, object(), {}))

    def fake_run_one(context, backend, repeats, timeout_s):
        calls.append(("run", backend, benchmark.METRICS))
        return {"backend_requested": backend, "backend_used": backend,
                "auto_backend_reason": None, "cold_s": 0.1,
                "warm_median_s": 0.1, "peak_vram": None, "peak_rss_kib": 100,
                "config_hash": "same", "artifacts": {metric: _artifact()},
                "repeat_receipts": fake_repeat_receipts(backend, benchmark.METRICS)}

    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py",
                                      "--factors", "32", "--metrics", metric, "--run", "--compact"])
    benchmark.main()
    assert [item[0] for item in calls] == ["preflight", "load"] + ["run"] * 6
    assert [item[1] for item in calls if item[0] == "run"] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu"]
    assert all(item[2] == (metric,) for item in calls if item[0] == "run")
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parity_pass"]
