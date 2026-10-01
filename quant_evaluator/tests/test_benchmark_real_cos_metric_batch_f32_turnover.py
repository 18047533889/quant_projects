"""CLI admission tests for the exact F32 factor-turnover profile."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_factor_batch as profile
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


from quant_evaluator.tests.benchmark_repeat_receipt_fixture import fake_repeat_receipts

def _setup_run(monkeypatch, calls):
    batch = SimpleNamespace(
        num_times=2586,
        num_assets=5461,
        num_factors=32,
        values=SimpleNamespace(dtype=np.dtype("float64")),
    )
    labels = object()

    def fake_preflight(args):
        calls.append(("preflight", args))
        return {"status": "ready"}

    def fake_load(**kwargs):
        calls.append(("load", kwargs))
        return batch, labels, {}

    def fake_run_one(context, backend, repeats, timeout_s):
        calls.append(("run", backend, benchmark.METRICS, repeats, timeout_s))
        actual_backend = "cuda" if backend == "auto" else backend
        return {
            "backend_requested": backend,
            "backend_used": actual_backend,
            "auto_backend_reason": None,
            "cold_s": 0.1,
            "warm_median_s": 0.1,
            "peak_vram": None,
            "peak_rss_kib": 100,
            "config_hash": "same",
            "artifacts": {},
            "repeat_receipts": fake_repeat_receipts(actual_backend, benchmark.METRICS),
        }

    monkeypatch.setattr(profile, "preflight_factor_count_profile", fake_preflight)
    monkeypatch.setattr(benchmark, "load_real_batch", fake_load)
    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(benchmark, "_compare", lambda reference, candidate: {"pass": True})
    monkeypatch.setattr(benchmark, "_BATCH", None)
    monkeypatch.setattr(benchmark, "_LABELS", None)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)


def test_f32_turnover_preflight_only_and_run_requires_output(
    monkeypatch, capsys, tmp_path
):
    calls = []
    _setup_run(monkeypatch, calls)
    output = tmp_path / "turnover.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
        "--metrics", "factor_turnover_rate",
        "--output", str(output),
    ])

    benchmark.main()

    assert [entry[0] for entry in calls] == ["preflight"]
    preflight = calls[0][1]
    assert preflight.profile_max_object_mib == 128
    assert preflight.profile_max_total_mib == 2048
    assert preflight.max_working_gib == 50
    output_lines = capsys.readouterr().out.splitlines()
    assert json.loads(output_lines[0])["preflight"]["status"] == "ready"
    result = json.loads(output_lines[-1])
    assert result["status"] == "preflight_only"
    assert "factor turnover" in result["note"]
    assert not output.exists()

    calls.clear()
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
        "--metrics", "factor_turnover_rate",
        "--run",
    ])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert calls == []


def test_f32_turnover_uses_shared_preflight_and_six_order_run(
    monkeypatch, capsys, tmp_path
):
    calls = []
    _setup_run(monkeypatch, calls)
    output = tmp_path / "turnover.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
        "--metrics", "factor_turnover_rate",
        "--run",
        "--output", str(output),
    ])

    benchmark.main()

    assert [entry[0] for entry in calls] == [
        "preflight", "load", "run", "run", "run", "run", "run", "run"
    ]
    preflight = calls[0][1]
    assert preflight.manifest_sha256 == benchmark.MANIFEST_SHA256
    assert preflight.profile_max_object_mib == 128
    assert preflight.profile_max_total_mib == 2048
    assert preflight.max_working_gib == 50
    assert calls[1][1] == {
        "factors": 32,
        "days": 0,
        "assets": 5500,
        "max_object_mib": 128,
        "max_total_mib": 2048,
        "manifest_sha256": benchmark.MANIFEST_SHA256,
    }
    runs = [entry for entry in calls if entry[0] == "run"]
    assert [entry[1] for entry in runs] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu"
    ]
    assert all(entry[2:] == (benchmark.TURNOVER_SINGLE, 2, 180.0)
               for entry in runs)

    report = json.loads(output.read_text())
    assert report["request"]["metrics"] == ["factor_turnover_rate"]
    assert report["request"]["shape"] == [2586, 5461, 32]
    assert report["parity_pass"] is True
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parity_pass"]


def test_f32_turnover_preflight_failure_stops_before_cos_load(monkeypatch, tmp_path):
    calls = []
    _setup_run(monkeypatch, calls)
    monkeypatch.setattr(
        profile, "preflight_factor_count_profile",
        lambda args: calls.append(("preflight", args))
        or {"status": "insufficient_resources"},
    )
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
        "--metrics", "factor_turnover_rate",
        "--run",
        "--output", str(tmp_path / "turnover.json"),
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 2
    assert [entry[0] for entry in calls] == ["preflight"]


def test_existing_f13_turnover_profile_keeps_its_unpreflighted_cli(
    monkeypatch, capsys
):
    calls = []
    _setup_run(monkeypatch, calls)
    batch = SimpleNamespace(
        num_times=2586,
        num_assets=5461,
        num_factors=13,
        values=SimpleNamespace(dtype=np.dtype("float64")),
    )

    def fake_load(**kwargs):
        calls.append(("load", kwargs))
        return batch, object(), {}

    monkeypatch.setattr(
        profile, "preflight_factor_count_profile",
        lambda args: pytest.fail("existing F13 profile must not enter F32 preflight"),
    )
    monkeypatch.setattr(benchmark, "load_real_batch", fake_load)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "13",
        "--metrics", "factor_turnover_rate",
    ])

    benchmark.main()

    assert [entry[0] for entry in calls] == ["load", "run", "run", "run", "run", "run", "run"]
    assert calls[0][1]["factors"] == 13
    assert calls[0][1]["max_object_mib"] == 64
    assert calls[0][1]["max_total_mib"] == 256
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parity_pass"]
