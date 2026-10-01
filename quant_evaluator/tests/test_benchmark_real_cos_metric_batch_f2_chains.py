"""Bounded F2 whole-batch benchmark CLI admission tests."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quant_evaluator.scripts import benchmark_real_cos_metric_batch as benchmark


from quant_evaluator.tests.benchmark_repeat_receipt_fixture import fake_repeat_receipts

@pytest.mark.parametrize("metrics", [benchmark.RANK_CHAIN, benchmark.QUANTILE_CHAIN])
def test_f2_exact_chain_admitted_before_cos_load(monkeypatch, capsys, metrics):
    loaded = []
    runs = []
    batch = SimpleNamespace(num_times=2586, num_assets=5461, num_factors=2,
                            values=SimpleNamespace(dtype=np.dtype("float64")))
    monkeypatch.setattr(benchmark, "load_real_batch",
                        lambda **kwargs: (loaded.append(kwargs) or (batch, object(), {"sources": []})))
    monkeypatch.setattr(benchmark, "_run_one",
                        lambda context, backend, repeats, timeout_s: (
                            runs.append((backend, benchmark.METRICS)) or {
                                "backend_requested": backend,
                                "backend_used": backend,
                                "auto_backend_reason": None,
                                "cold_s": 0.1,
                                "warm_median_s": 0.1,
                                "peak_vram": None,
                                "peak_rss_kib": 100,
                                "config_hash": "same",
                                "artifacts": {},
                                "repeat_receipts": fake_repeat_receipts(
                                    backend, benchmark.METRICS),
                            }))
    monkeypatch.setattr(benchmark, "_compare",
                        lambda reference, candidate: {"pass": True})
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--factors", "2",
        "--metrics", ",".join(metrics),
    ])
    benchmark.main()
    assert loaded and loaded[0]["factors"] == 2
    assert [backend for backend, _ in runs] == [
        "cpu", "cuda_strict", "auto", "auto", "cuda_strict", "cpu"]
    assert all(requested == metrics for _, requested in runs)
    assert '"parity_pass": true' in capsys.readouterr().out


def test_f2_rejects_uncertified_metric_set_before_cos_load(monkeypatch):
    monkeypatch.setattr(
        benchmark, "load_real_batch",
        lambda **kwargs: pytest.fail("invalid request must not load COS data"))
    monkeypatch.setattr(sys, "argv", [
        "benchmark_real_cos_metric_batch.py", "--factors", "2",
        "--metrics", "rank_ic,ic_ir",
    ])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
