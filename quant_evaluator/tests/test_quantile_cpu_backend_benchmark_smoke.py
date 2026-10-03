"""Tiny correctness and bound checks for the opt-in CPU backend harness."""
import numpy as np
import pytest
from quant_evaluator.scripts.benchmark_quantile_cpu_backends_oct03 import (
    MAX_FACTOR_CELLS, _assignment_valid_mask, _check_rounds, _check_shape, is_numba_available, smoke_check,
)

@pytest.mark.skipif(not is_numba_available(), reason="Numba backend is optional")
@pytest.mark.parametrize("scenario", ["dense", "ties_missing"])
def test_quantile_cpu_backend_smoke_checks_numpy_numba_parity(scenario):
    report = smoke_check(scenario=scenario)
    assert report["scenario"] == scenario
    assert len(report["input_fingerprint"]) == 64
    assert report["validity_path"] == "finite-value masks; no explicit validity arrays"
    assert report["quantiles"] == [5, 20]
    assert all(item["assignments"] == "exact" for item in report["correctness"].values())
    assert all(item["counts"] == "exact" for item in report["correctness"].values())

def test_quantile_cpu_backend_benchmark_rejects_unbounded_shapes():
    with pytest.raises(ValueError, match="capped"):
        _check_shape(1000, 5461, 48)
    with pytest.raises(ValueError, match="positive builtin integer"):
        _check_shape(1, 1, 1, max_estimated_bytes=True)
    with pytest.raises(ValueError, match="positive builtin integer"):
        _check_shape(1, 1, 1, max_estimated_bytes=0)
    assert MAX_FACTOR_CELLS == 70_000_000

def test_assignment_mask_respects_minimum_finite_population():
    values = np.array([[[1.0], [2.0], [np.nan]],
                       [[np.nan], [np.inf], [-np.inf]]])
    actual = _assignment_valid_mask(values, n_quantiles=5)
    np.testing.assert_array_equal(actual, np.zeros_like(values, dtype=bool))

@pytest.mark.parametrize("rounds", [True, 2, 21, 3.0])
def test_benchmark_rejects_invalid_round_limits(rounds):
    with pytest.raises(ValueError, match="between 3 and 20"):
        _check_rounds(rounds)

def test_benchmark_accepts_round_bounds():
    _check_rounds(3)
    _check_rounds(20)
