"""CLI contract tests for bounded F32 single-metric runs."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


from quant_evaluator.tests.benchmark_repeat_receipt_fixture import fake_repeat_receipts

@pytest.mark.parametrize("metric", ["rank_ic_series", "quantile_returns_full"])
def test_f32_accepts_only_exact_single_metric_request(monkeypatch, capsys, metric):
    batch = SimpleNamespace(
        num_times=2586,
        num_assets=5461,
        num_factors=32,
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
            "backend_used": backend,
            "auto_backend_reason": None,
            "cold_s": 0.1,
            "warm_median_s": 0.1,
            "peak_vram": None,
            "peak_rss_kib": 100,
            "config_hash": "same",
            "artifacts": {},
            "repeat_receipts": fake_repeat_receipts(backend, benchmark.METRICS),
        }

    monkeypatch.setattr(benchmark, "_BATCH", None)
    monkeypatch.setattr(benchmark, "_LABELS", None)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    monkeypatch.setattr(benchmark, "load_real_batch", fake_load_real_batch)
    monkeypatch.setattr(benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(benchmark, "_compare", lambda reference, candidate: {"pass": True})
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
        "--metrics", metric,
    ])

    benchmark.main()

    assert load_calls == [{
        "factors": 32,
        "days": 0,
        "assets": 5500,
        "max_object_mib": 128,
        "max_total_mib": 2048,
        "manifest_sha256": benchmark.MANIFEST_SHA256,
    }]
    assert [run[0] for run in runs] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu",
    ]
    assert all(run[3] == (metric,) for run in runs)
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["parity_pass"] is True


def test_f32_rejects_default_multi_metric_request_before_loading(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("invalid F32 request must be rejected before loading COS data")

    monkeypatch.setattr(benchmark, "load_real_batch", fail)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py",
        "--factors", "32",
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("metrics", [
    "coverage",
    "rank_ic,quantile_spread,factor_turnover_rate,coverage",
    "rank_ic,pearson_ic,ic_ir,quantile_spread,coverage",
])
def test_f32_coverage_preflight_only_never_loads_or_writes(monkeypatch, capsys, metrics):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as factor_batch

    preflights = []

    def fake_preflight(args):
        preflights.append(args)
        return {"status": "ready", "pass": True}

    def fail_load(*args, **kwargs):
        pytest.fail("preflight-only request must not load COS data")

    monkeypatch.setattr(factor_batch, "preflight_factor_count_profile", fake_preflight)
    monkeypatch.setattr(benchmark, "load_real_batch", fail_load)
    monkeypatch.setattr(benchmark, "METRICS", benchmark.DEFAULT_METRICS)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--factors", "32", "--metrics", metrics,
    ])

    benchmark.main()

    assert len(preflights) == 1
    assert preflights[0].profile_max_object_mib == 128
    assert preflights[0].profile_max_total_mib == 2048
    assert preflights[0].max_working_gib == 50
    assert preflights[0].metric_ids == tuple(metrics.split(","))
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["status"] == "preflight_only"


@pytest.mark.parametrize("metrics", [
    "coverage",
    "rank_ic,pearson_ic,ic_ir,quantile_spread,coverage",
])
def test_f32_coverage_run_requires_output_before_preflight_or_load(monkeypatch, metrics):
    from quant_evaluator.scripts import benchmark_real_cos_factor_batch as factor_batch

    def fail(*args, **kwargs):
        pytest.fail("missing-output guard must run before preflight or COS load")

    monkeypatch.setattr(factor_batch, "preflight_factor_count_profile", fail)
    monkeypatch.setattr(benchmark, "load_real_batch", fail)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--factors", "32",
        "--metrics", metrics, "--run",
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(("factors", "metrics"), [
    (32, "coverage,rank_ic,quantile_spread,factor_turnover_rate"),
    (24, "coverage"),
    (32, "coverage,quantile_spread,ic_ir,pearson_ic,rank_ic"),
    (24, "rank_ic,pearson_ic,ic_ir,quantile_spread,coverage"),
])
def test_f32_rejects_wrong_order_or_factor_count_before_loading(
        monkeypatch, factors, metrics):
    def fail_load(*args, **kwargs):
        pytest.fail("invalid coverage request must be rejected before COS load")

    monkeypatch.setattr(benchmark, "load_real_batch", fail_load)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--factors", str(factors),
        "--metrics", metrics,
    ])

    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
