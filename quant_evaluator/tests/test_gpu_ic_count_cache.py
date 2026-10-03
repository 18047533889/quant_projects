"""Focused GPUExecutor IC observation-count cache regression tests."""
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _cuda_devices = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA driver unavailable", allow_module_level=True)
if _cuda_devices < 1:
    pytest.skip("CUDA device required", allow_module_level=True)

import quant_evaluator.runtime.gpu_executor as gpu_executor


def test_ic_count_reduction_transfer_cache_is_scoped_and_copy_safe(monkeypatch):
    # Tiny real CUDA arrays keep this a bounded check; correlations are mocked
    # so the test measures only executor count-reduction/transfer behavior.
    t, f, n = 3, 2, 8
    factors = cp.zeros((t, f, n), dtype=cp.float64)
    labels = cp.zeros((t, n), dtype=cp.float64)
    session = SimpleNamespace(
        _staged_factors={"__all__": factors},
        _staged_labels={"next_ret": labels},
        metadata=lambda: {},
    )
    executor = gpu_executor.GPUExecutor(session)
    metrics = (
        "rank_ic_series", "rank_ic", "ic_std",
        "pearson_ic_series", "pearson_ic", "pearson_ic_std",
    )
    executor.metric_parameters = {
        "rank_ic_series": {"min_assets": 2},
        "rank_ic": {"min_assets": 2},
        "ic_std": {"min_assets": 5},
        "pearson_ic_series": {"min_assets": 2},
        "pearson_ic": {"min_assets": 2},
        "pearson_ic_std": {"min_assets": 5},
    }

    rank_calls = []
    pearson_calls = []
    rank_base = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, np.nan]])
    rank_strict = np.array([[0.5, np.nan], [np.nan, 0.7], [np.nan, np.nan]])
    pearson_base = np.array([[0.1, 0.2], [np.nan, 0.3], [0.3, 0.4]])
    pearson_strict = np.array([[0.5, 0.5], [0.6, np.nan], [0.7, np.nan]])

    def rank_kernel(_factors, _labels, *, min_obs):
        rank_calls.append(min_obs)
        return cp.asarray(rank_base if min_obs == 2 else rank_strict), None

    def pearson_kernel(_factors, _labels, *, min_obs):
        pearson_calls.append(min_obs)
        return cp.asarray(pearson_base if min_obs == 2 else pearson_strict), None

    import quant_evaluator.kernels.gpu.correlation as correlation
    monkeypatch.setattr(correlation, "batched_spearman_ic", rank_kernel)
    monkeypatch.setattr(correlation, "batched_pearson_ic", pearson_kernel)

    reductions = []
    class CupyProxy:
        def __getattr__(self, name):
            return getattr(cp, name)

        def sum(self, values, *args, **kwargs):
            reductions.append((values.shape, kwargs.get("axis", args[0] if args else None)))
            return cp.sum(values, *args, **kwargs)

    transfers = []
    original_to_cpu = gpu_executor._to_cpu
    def record_to_cpu(value):
        result = original_to_cpu(value)
        if np.asarray(result).dtype.kind in "iu" and np.asarray(result).ndim == 1:
            transfers.append(np.asarray(result).copy())
        return result

    monkeypatch.setattr(gpu_executor, "_import_cp", lambda: CupyProxy())
    monkeypatch.setattr(gpu_executor, "_to_cpu", record_to_cpu)
    result = executor.run(("f0", "f1"), metrics)

    # One kernel and one count reduction/transfer per family and threshold.
    assert rank_calls == [2, 5]
    assert pearson_calls == [2, 5]
    assert len(reductions) == 4
    assert len(transfers) == 4

    # Each count agrees with a direct CPU finite-value reference.
    expected_rank_2 = np.array([3, 2])
    expected_rank_5 = np.array([1, 1])
    expected_pearson_2 = np.array([2, 3])
    expected_pearson_5 = np.array([3, 1])
    np.testing.assert_array_equal(np.isfinite(rank_base).sum(axis=0), expected_rank_2)
    np.testing.assert_array_equal(np.isfinite(rank_strict).sum(axis=0), expected_rank_5)
    np.testing.assert_array_equal(np.isfinite(pearson_base).sum(axis=0), expected_pearson_2)
    np.testing.assert_array_equal(np.isfinite(pearson_strict).sum(axis=0), expected_pearson_5)
    expected = {
        "rank_ic_series": expected_rank_2,
        "rank_ic": expected_rank_2,
        "ic_std": expected_rank_5,
        "pearson_ic_series": expected_pearson_2,
        "pearson_ic": expected_pearson_2,
        "pearson_ic_std": expected_pearson_5,
    }
    for metric, count in expected.items():
        np.testing.assert_array_equal(result.observation_counts[metric], count)

    # Same-key metrics get equal values but independent writable host arrays.
    count_series = result.observation_counts["rank_ic_series"]
    count_scalar = result.observation_counts["rank_ic"]
    assert count_series is not count_scalar
    count_series[0] = -99
    assert count_scalar[0] == expected_rank_2[0]
    pearson_series = result.observation_counts["pearson_ic_series"]
    pearson_scalar = result.observation_counts["pearson_ic"]
    assert pearson_series is not pearson_scalar
    pearson_series[0] = -99
    assert pearson_scalar[0] == expected_pearson_2[0]
