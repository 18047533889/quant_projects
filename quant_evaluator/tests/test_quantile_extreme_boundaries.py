import numpy as np
import pytest
from quant_evaluator.metrics.quantile import assign_quantiles, assign_quantiles_fast, assign_quantiles_batch

@pytest.mark.parametrize("function", [assign_quantiles, assign_quantiles_fast, assign_quantiles_batch])
@pytest.mark.parametrize("method", ["min", "max"])
@pytest.mark.parametrize("dtype,magnitude", [(np.float64, 1e308), (np.float32, 3e38), (np.int64, 9_000_000_000_000_000_000)])
def test_opposite_extremes_keep_both_quantile_legs(function, method, dtype, magnitude):
    values = np.array([[-magnitude, magnitude]], dtype=dtype)
    with np.errstate(over="raise", invalid="raise"):
        actual = function(values, n_quantiles=2, method=method)
    np.testing.assert_array_equal(actual, [[0, 1]])

@pytest.mark.parametrize("method", ["min", "max"])
def test_numba_opposite_extremes_match_hand_calculated_median(method):
    from quant_evaluator.metrics.quantile_numba import _assign_quantiles_jit
    values = np.array([[[-1e308], [1e308]]])
    np.testing.assert_array_equal(_assign_quantiles_jit(values, 2, method), [[[0], [1]]])

@pytest.mark.parametrize("method", ["min", "max"])
def test_gpu_array_algorithm_extreme_boundaries_on_numpy_backend(monkeypatch, method):
    # Exercise the real array algorithm, not CUDA device/runtime behavior.
    from quant_evaluator.kernels.gpu import rank
    monkeypatch.setattr(rank, "_import_cp", lambda: np)
    values = np.array([[-1e308, 1e308], [-1e308, 1e308]])
    with np.errstate(over="ignore", invalid="ignore"):
        actual = rank.batched_quantile_assignment(values, n_quantiles=2, method=method)
    np.testing.assert_array_equal(actual, [[0, 1], [0, 1]])

@pytest.mark.parametrize("method,expected", [("min", [0]*10+[10]*10), ("max", [9]*10+[19]*10)])
def test_twenty_layers_preserve_extreme_plateau_tie_policy(method, expected):
    values = np.array([[-1e308]*10+[1e308]*10])
    with np.errstate(over="raise", invalid="raise"):
        actual = assign_quantiles_batch(values, n_quantiles=20, method=method)
    np.testing.assert_array_equal(actual, [expected])

