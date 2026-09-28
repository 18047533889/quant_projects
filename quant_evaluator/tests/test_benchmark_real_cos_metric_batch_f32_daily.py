"""Offline contract for the bounded F32 daily-quantile public A/B."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_factor_batch as profile
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


def _artifact():
    return {
        "descriptor": {"type": "DailyQuantileReturnArtifact",
                       "artifact_kind": "daily_quantile_return"},
        "values": [[[0.125]]],
        "finite_mask": [[[True]]],
        "valid_mask": [[[True]]],
        "counts": [[[32]]],
        "provenance_observation_counts": [[32]],
        "provenance": {"source": "bound_cos"},
        "metric_values": [{"metric_id": "quantile_returns_daily", "value": 0.125,
                           "valid": True, "observation_count": 32,
                           "metric_version": "1", "warnings": [],
                           "sample_unit": "date"}],
    }


def test_daily_comparison_checks_complete_public_artifact(monkeypatch):
    monkeypatch.setattr(benchmark, "METRICS", benchmark.QUANTILE_DAILY_SINGLE)
    left = {"config_hash": "same", "artifacts": {"quantile_returns_daily": _artifact()}}
    right = {"config_hash": "same", "artifacts": {"quantile_returns_daily": _artifact()}}
    assert benchmark._compare(left, right)["pass"]
    changes = (
        ("descriptor", {"type": "wrong"}),
        ("values", [[[0.5]]]),
        ("finite_mask", [[[False]]]),
        ("valid_mask", [[[False]]]),
        ("counts", [[[31]]]),
        ("provenance_observation_counts", [[31]]),
        ("provenance", {"source": "wrong"}),
        ("metric_values", []),
    )
    for field, changed in changes:
        right["artifacts"]["quantile_returns_daily"][field] = changed
        assert not benchmark._compare(left, right)["metrics"]["quantile_returns_daily"]["pass"]
        right["artifacts"]["quantile_returns_daily"][field] = _artifact()[field]
    right["config_hash"] = "wrong"
    assert not benchmark._compare(left, right)["pass"]


def test_f32_daily_preflight_then_six_order_run(monkeypatch, capsys, tmp_path):
    batch = SimpleNamespace(num_times=2586, num_assets=5461, num_factors=32,
                            values=SimpleNamespace(dtype=np.dtype("float64")))
    calls = []
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: calls.append(("preflight", args)) or {"status": "ready"})
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: calls.append(("load", kwargs)) or (batch, object(), {}))

    def fake_run_one(context, backend, repeats, timeout_s):
        calls.append(("run", backend, benchmark.METRICS, repeats, timeout_s))
        return {"backend_requested": backend,
                "backend_used": "cpu" if backend == "auto" else backend,
                "auto_backend_reason": "metric_not_certified" if backend == "auto" else None,
                "cold_s": 0.1, "warm_median_s": 0.1, "peak_vram": None,
                "peak_rss_kib": 100, "config_hash": "same",
                "artifacts": {"quantile_returns_daily": _artifact()}}

    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    argv = ["benchmark_real_cos_metric_batch.py", "--factors", "32",
            "--metrics", "quantile_returns_daily", "--timeout-s", "300",
            "--compact", "--output", str(tmp_path / "daily.json")]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()
    assert [row[0] for row in calls] == ["preflight"]
    assert calls[0][1].manifest_sha256 == benchmark.MANIFEST_SHA256
    assert calls[0][1].profile_max_total_mib == 2048
    assert calls[0][1].max_working_gib == 50
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "preflight_only"
    assert not (tmp_path / "daily.json").exists()

    calls.clear()
    monkeypatch.setattr(sys, "argv", [*argv, "--run"])
    benchmark.main()
    assert [row[0] for row in calls] == ["preflight", "load"] + ["run"] * 6
    assert [row[1] for row in calls if row[0] == "run"] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu"]
    assert all(row[2:] == (benchmark.QUANTILE_DAILY_SINGLE, 2, 300.0)
               for row in calls if row[0] == "run")
    assert calls[1][1]["factors"] == 32
    assert calls[1][1]["max_total_mib"] == 2048
    report = json.loads((tmp_path / "daily.json").read_text())
    assert report["request"]["shape"] == [2586, 5461, 32]
    assert report["parity_pass"]
    assert all(item["metrics"]["quantile_returns_daily"]["pass"]
               for group in report["comparisons_to_first_cpu"].values() for item in group)
    assert all("artifacts" not in run and "artifact_sha256" in run for run in report["runs"])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parity_pass"]


def test_f32_daily_preflight_failure_and_other_shape_never_load(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: pytest.fail("unexpected COS load"))
    monkeypatch.setattr(profile, "preflight_factor_count_profile",
                        lambda args: calls.append(args) or {"status": "insufficient_resources"})
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py", "--factors", "32",
                                      "--metrics", "quantile_returns_daily", "--run"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert len(calls) == 1

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["benchmark_real_cos_metric_batch.py", "--factors", "12",
                                      "--metrics", "quantile_returns_daily", "--run"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert calls == []
