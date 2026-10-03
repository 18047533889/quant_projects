"""Small-array contract tests for the bounded FE rank comparison harness."""
from __future__ import annotations

import numpy as np
import pytest


def test_external_transposed_tile_runs_paired_abba_with_full_value_oracle(monkeypatch):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    # A caller-owned (T, N, F) tile becomes a non-copying (F, T, N) view.
    backing = np.array(
        [
            [[2.0], [2.0], [5.0]],
            [[np.nan], [4.0], [8.0]],
        ]
    )
    values = backing.transpose(2, 0, 1)
    values.setflags(write=False)
    before = backing.copy()
    call_order = []
    fp = bench._fp_cs_rank
    fe = bench._fe_cs_rank

    def observed_fp(*args, **kwargs):
        call_order.append("FP")
        return fp(*args, **kwargs)

    def observed_fe(*args, **kwargs):
        call_order.append("FE")
        return fe(*args, **kwargs)

    monkeypatch.setattr(bench, "_fp_cs_rank", observed_fp)
    monkeypatch.setattr(bench, "_fe_cs_rank", observed_fe)

    report = bench.benchmark_fe_native_rank(values, max_result_bytes=4096)

    assert report["abba_order"] == ["FP", "FE", "FE", "FP"]
    assert [run["backend"] for run in report["runs"]] == report["abba_order"]
    assert call_order[:2] == ["FP", "FE"]  # bounded parity pass; timed runs follow ABBA
    assert [run["context_fingerprint"] for run in report["runs"]] == [
        report["context_fingerprint"]
    ] * 4
    assert report["oracle_check"]["status"] == "passed"
    assert report["oracle_check"]["verified_values"] == values.size
    assert report["oracle_check"]["nan_count"] == 1
    assert report["output_fingerprints"]["FP"] == report["output_fingerprints"]["FE"]
    assert report["input_fingerprint"]["shape"] == [1, 2, 3]
    assert report["input_fingerprint"]["nan_count"] == 1
    assert report["production_admitted"] is False
    assert report["performance_claim"] is None
    assert "speedup" not in report
    assert report["oracle_check"]["oracle"] == "actual_FP_cs_rank_parity_baseline"
    assert "scipy" in report["runtime_fingerprint"]
    assert "threadpoolctl" in report["runtime_fingerprint"]
    assert report["runtime_fingerprint"]["polars_thread_pool_size"] > 0
    assert "not a transitive" in report["evidence_scope"]
    assert all(isinstance(run["result_digest"], str) for run in report["runs"])
    np.testing.assert_array_equal(backing, before)
    assert not values.flags.writeable


def test_full_array_oracle_detects_any_fe_value_or_nan_mask_mismatch(monkeypatch):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    original = bench._fe_cs_rank

    def wrong_one_value(values, **kwargs):
        result = original(values, **kwargs)
        result.reshape(-1)[0] = 0.125
        return result

    monkeypatch.setattr(bench, "_fe_cs_rank", wrong_one_value)
    with pytest.raises(AssertionError, match="FP oracle mismatch"):
        bench.benchmark_fe_native_rank(np.array([[1.0, 2.0, 3.0]]))



@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_timed_elapsed_fails_closed(monkeypatch, elapsed):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    clock = iter([10.0, 10.0 + elapsed])
    monkeypatch.setattr(bench, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="finite and strictly positive"):
        bench.benchmark_fe_native_rank(np.array([[1.0, 2.0, 3.0]]))


def test_transient_context_change_during_timed_run_is_detected(monkeypatch):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    original = bench._measurement_context
    calls = 0

    def transient_change(values):
        nonlocal calls
        calls += 1
        context = original(values)
        if calls == 4:  # initial, post-oracle, pre-run, post-run
            context = {**context, "runtime": {**context["runtime"], "scipy": "transient-change"}}
        return context

    monkeypatch.setattr(bench, "_measurement_context", transient_change)
    with pytest.raises(RuntimeError, match="changed during FP run"):
        bench.benchmark_fe_native_rank(np.array([[1.0, 2.0, 3.0]]))
def test_result_memory_budget_rejects_before_either_backend_runs(monkeypatch):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    def forbidden(*args, **kwargs):
        raise AssertionError("budget rejection must precede backend execution")

    monkeypatch.setattr(bench, "_fp_cs_rank", forbidden)
    monkeypatch.setattr(bench, "_fe_cs_rank", forbidden)
    with pytest.raises(MemoryError, match="result allocation"):
        bench.benchmark_fe_native_rank(np.ones((2, 3)), max_result_bytes=47)


@pytest.mark.parametrize("budget", [True, False, 0, -1, 1.5, np.nan])
def test_invalid_result_budgets_fail_closed(budget):
    from factor_preprocess.scripts.benchmark_fe_native_rank import benchmark_fe_native_rank

    with pytest.raises((TypeError, ValueError)):
        benchmark_fe_native_rank(np.ones((2, 3)), max_result_bytes=budget)


def test_full_cross_section_chunk_cap_rejects_before_backend_execution(monkeypatch):
    import factor_preprocess.scripts.benchmark_fe_native_rank as bench

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid group width must reject before execution")

    monkeypatch.setattr(bench, "_fp_cs_rank", forbidden)
    monkeypatch.setattr(bench, "_fe_cs_rank", forbidden)
    with pytest.raises(ValueError, match="complete cross-section"):
        bench.benchmark_fe_native_rank(np.ones((2, 5)), max_chunk_cells=4)


def test_bad_input_shapes_types_and_empty_data_fail_closed():
    from factor_preprocess.scripts.benchmark_fe_native_rank import benchmark_fe_native_rank

    with pytest.raises(TypeError, match="NumPy ndarray"):
        benchmark_fe_native_rank([[1.0, 2.0]])
    with pytest.raises(ValueError, match="at least one axis"):
        benchmark_fe_native_rank(np.array(3.0))
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_fe_native_rank(np.empty((2, 0)))
    with pytest.raises(TypeError, match="real numeric"):
        benchmark_fe_native_rank(np.array([[1.0 + 1.0j, 2.0 + 0.0j]]))
