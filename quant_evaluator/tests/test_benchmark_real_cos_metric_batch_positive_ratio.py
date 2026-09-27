"""CLI contract test for the bounded F13 positive-ratio run."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


def test_f13_accepts_exact_single_positive_ratio_request(monkeypatch, capsys):
    batch = SimpleNamespace(
        num_times=2586,
        num_assets=5461,
        num_factors=13,
        values=SimpleNamespace(dtype=np.dtype("float64")),
    )
    labels = object()
    load_calls = []
    runs = []

    def fake_load_real_batch(**kwargs):
        load_calls.append(kwargs)
        return batch, labels, {"sources": []}

    def fake_run_one(context, backend, repeats, timeout_s):
        runs.append((backend, repeats, timeout_s, benchmark.METRICS))
        return {
            "backend_requested": backend,
            "backend_used": "cpu" if backend == "auto" else backend,
            "auto_backend_reason": "shape_outside_certified_range" if backend == "auto" else None,
            "cold_s": 0.1,
            "warm_median_s": 0.1,
            "peak_vram": None,
            "peak_rss_kib": 100,
            "config_hash": "same",
            "artifacts": {},
        }

    monkeypatch.setattr(benchmark, "_BATCH", None)
    monkeypatch.setattr(benchmark, "_LABELS", None)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    monkeypatch.setattr(benchmark, "load_real_batch", fake_load_real_batch)
    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(benchmark, "_compare", lambda reference, candidate: {"pass": True})
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "13",
        "--metrics", "rank_ic_positive_ratio",
    ])

    benchmark.main()

    assert load_calls == [{
        "factors": 13,
        "days": 0,
        "assets": 5500,
        "max_object_mib": 64,
        "max_total_mib": 256,
        "manifest_sha256": benchmark.MANIFEST_SHA256,
    }]
    assert [run[0] for run in runs] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu",
    ]
    assert all(run[3] == ("rank_ic_positive_ratio",) for run in runs)
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["parity_pass"] is True


def test_f13_rejects_multi_metric_positive_ratio_before_loading(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid F13 request must be rejected before loading COS data")

    monkeypatch.setattr(benchmark, "load_real_batch", fail)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "13",
        "--metrics", "rank_ic_positive_ratio,rank_ic",
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
