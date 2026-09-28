"""Offline contracts for the verified F61 tail-five public A/B harness."""
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_evaluator.scripts import benchmark_real_cos_f61_tail5 as tail
from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles
from quant_evaluator.scripts import benchmark_real_cos_metric_batch as full


def _mapping():
    return {f"f{i:02d}": {"verified": True,
            "status": "materialized_not_evaluated", "bytes": i + 1,
            "sha256": f"{i+1:064x}",
            "uri": f"{tiles.POOL}/{i+1:064x}/f{i:02d}.parquet"}
            for i in range(61)}


def test_select_tail_uses_last_five_of_verified_f61_axis_index(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=3)
    def fake_index(path, sha, records):
        assert path == tail.AXIS_INDEX
        assert sha == full.MANIFEST_SHA256
        assert len(records) == 61
        return dates, {"A.SZ"}, tuple((*row, "etag", "axis") for row in records)
    monkeypatch.setattr(tiles, "read_axis_index", fake_index)
    selected, sources, common_dates, common_assets = tail.select_tail(_mapping())
    assert [row[0] for row in selected] == [f"f{i:02d}" for i in range(56, 61)]
    assert [row[0] for row in sources] == [row[0] for row in selected]
    assert common_dates.equals(dates) and common_assets == {"A.SZ"}


def test_receipt_checks_requested_actual_and_all_metrics():
    run = {"backend_used": "cuda", "config_hash": "config",
           "metric_backends": {metric: "cuda" for metric in tail.METRICS},
           "execution_receipt": {"backend_requested": "cuda_strict",
                                 "backend_used": "cuda", "config_hash": "config",
                                 "receipt_hash": "a" * 64,
                                 "metric_backends": {metric: "cuda" for metric in tail.METRICS}}}
    assert tail.receipt_check(run, "cuda_strict")
    run["execution_receipt"]["metric_backends"].pop("factor_turnover_rate")
    assert not tail.receipt_check(run, "cuda_strict")


def test_no_run_preflights_without_loading_factor_or_labels(monkeypatch, capsys):
    rows = tuple((f"f{i}", "uri", "sha", 1) for i in range(5))
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: _mapping())
    monkeypatch.setattr(tail, "select_tail", lambda mapping: (
        rows, tuple((*row, "etag", "axis") for row in rows),
        pd.date_range("2024-01-01", periods=3), {"A.SZ"}))
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": True})
    monkeypatch.setattr(tiles, "load_labels",
                        lambda *args: pytest.fail("labels loaded without --run"))
    monkeypatch.setattr(tiles, "iter_frames",
                        lambda *args: pytest.fail("factors loaded without --run"))
    monkeypatch.setattr(sys, "argv", ["f61-tail5"])
    tail.main()
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["status"] == "ready"
    assert result["factor_ids"] == [row[0] for row in rows]


def test_six_order_ab_uses_full_artifact_comparator(monkeypatch):
    calls = []
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": True})
    def fake_run(context, backend, repeats, timeout_s):
        calls.append((backend, repeats, timeout_s, full.METRICS))
        actual = "cuda" if backend == "cuda_strict" else "cpu"
        metric_backends = {metric: actual for metric in tail.METRICS}
        return {"backend_requested": backend, "backend_used": actual,
                "cold_s": 1.0, "warm_median_s": 0.5, "peak_vram": 1,
                "config_hash": "same", "metric_backends": metric_backends,
                "execution_receipt": {"backend_requested": backend,
                                      "backend_used": actual, "config_hash": "same",
                                      "receipt_hash": "a" * 64,
                                      "metric_backends": metric_backends}}
    monkeypatch.setattr(full, "_run_one", fake_run)
    comparisons = []
    monkeypatch.setattr(full, "_compare",
                        lambda a, b: comparisons.append((a, b)) or {"pass": True})
    result = tail.run_ab(object(), object(), 300)
    assert [row[0] for row in calls] == list(tail.ORDER)
    assert all(row[1:] == (2, 300, tail.METRICS) for row in calls)
    assert len(comparisons) == 6
    assert result["parity_pass"] is True


def test_failed_preflight_blocks_explicit_run(monkeypatch):
    monkeypatch.setattr(tiles, "read_manifest", lambda sha: _mapping())
    monkeypatch.setattr(tail, "select_tail",
                        lambda mapping: ((), (), (), ()))
    monkeypatch.setattr(tiles, "memory_preflight", lambda *args: {"pass": False})
    monkeypatch.setattr(tiles, "load_labels",
                        lambda *args: pytest.fail("labels loaded after failed preflight"))
    monkeypatch.setattr(sys, "argv", ["f61-tail5", "--run"])
    with pytest.raises(SystemExit) as exc:
        tail.main()
    assert exc.value.code == 2
