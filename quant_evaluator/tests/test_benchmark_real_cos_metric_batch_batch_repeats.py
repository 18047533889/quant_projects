import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


def _mock_normal_run(monkeypatch, tmp_path, extra_args=()):
    monkeypatch.setattr(benchmark, "_BATCH", None)
    monkeypatch.setattr(benchmark, "_LABELS", None)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    batch = SimpleNamespace(
        num_times=2586, num_assets=5461, num_factors=8,
        values=SimpleNamespace(dtype=np.dtype("float64")),
    )
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: (batch, object(), {}))
    calls = []

    def run_one(_context, backend, repeats, timeout_s):
        calls.append((backend, repeats))
        return {
            "backend_requested": backend, "backend_used": backend,
            "auto_backend_reason": None, "cold_s": 0.1,
            "warm_median_s": 0.1, "peak_vram": None, "peak_rss_kib": 100,
            "config_hash": "same", "artifacts": {},
            "repeat_receipts": [
                {"backend_used": backend, "repeat": index}
                for index in range(1, repeats + 1)
            ],
        }

    monkeypatch.setattr(benchmark, "_run_one", run_one)
    monkeypatch.setattr(benchmark, "_compare", lambda *_args: {"pass": True})
    monkeypatch.setattr(benchmark, "_compare_repeat_receipt",
                        lambda *_args: {"pass": True})
    monkeypatch.setattr(benchmark, "_compare_repeat_numeric",
                        lambda *_args: {"pass": True})
    output = tmp_path / "batch-repeats.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--output", str(output), *extra_args,
    ])
    return calls, output


def test_batch_repeats_three_is_passed_to_workers_and_reported(monkeypatch, tmp_path):
    calls, output = _mock_normal_run(monkeypatch, tmp_path, ("--batch-repeats", "3"))

    benchmark.main()

    report = json.loads(output.read_text())
    assert [repeats for _backend, repeats in calls] == [3] * 6
    assert report["request"]["repeats_per_child"] == 3


def test_batch_repeats_defaults_to_two_for_compatibility(monkeypatch, tmp_path):
    calls, output = _mock_normal_run(monkeypatch, tmp_path)

    benchmark.main()

    report = json.loads(output.read_text())
    assert [repeats for _backend, repeats in calls] == [2] * 6
    assert report["request"]["repeats_per_child"] == 2


def test_invalid_batch_repeats_is_rejected_before_cos_load(monkeypatch):
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: pytest.fail("invalid CLI input must precede COS load"))
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--batch-repeats", "5",
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 2


def test_reference_repeats_option_remains_independent(monkeypatch, tmp_path):
    calls, output = _mock_normal_run(monkeypatch, tmp_path, ("--repeats", "1"))

    benchmark.main()

    report = json.loads(output.read_text())
    assert [repeats for _backend, repeats in calls] == [2] * 6
    assert report["request"]["repeats_per_child"] == 2
