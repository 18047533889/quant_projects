"""Independent finite-slice oracle tests for rolling constancy detection."""
import numpy as np
import pytest

from quant_evaluator.metrics.rolling_window_constancy import finite_constant_windows


def _oracle(values, window):
    count = max(0, len(values) - window + 1)
    constant = np.zeros(count, dtype=bool)
    common = np.full(count, np.nan)
    for start in range(count):
        observed = values[start:start + window]
        observed = observed[np.isfinite(observed)]
        if observed.size and np.all(observed == observed[0]):
            constant[start] = True
            common[start] = observed[0]
    return constant, common


def _cases():
    rng = np.random.default_rng(20261003)
    values = np.full(3000, 0.01)
    values[::17] = np.nan
    values[8::31] = np.inf
    values[12::47] = -np.inf
    nearflat = 1e12 + (np.arange(3000) % 13) * 0.25
    nearflat[[2, 7, 55, 56, 1001]] = [np.nan, np.inf, -np.inf, np.nan, np.inf]
    step = np.repeat(np.array([-0.01, 0.0, 0.01, 0.02]), 750)
    step[[0, 19, 20, 999, 1000, 1499, 1500, 2999]] = np.nan
    signed_zero = np.resize(np.array([-0.0, 0.0, np.nan, np.inf]), 3000)
    random_returns = rng.normal(0.0, 0.01, 3000)
    return {
        "ordinary": random_returns,
        "flat": np.full(3000, 0.01),
        "missing_flat": values,
        "near_flat": nearflat,
        "step": step,
        "step_random": np.repeat(rng.normal(0.0, 0.01, 30), 100),
        "signed_zero": signed_zero,
        "all_missing": np.resize(np.array([np.nan, np.inf, -np.inf]), 3000),
    }


@pytest.mark.parametrize("window", [1, 2, 20, 252])
@pytest.mark.parametrize("case_name", list(_cases()))
def test_finite_constant_windows_matches_independent_slice_oracle(case_name, window):
    values = _cases()[case_name]
    before = values.copy()
    actual_mask, actual_values = finite_constant_windows(values, window)
    expected_mask, expected_values = _oracle(values, window)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_array_equal(actual_values, expected_values)
    np.testing.assert_array_equal(values, before)


def test_change_crossing_window_boundary_changes_finite_constant_classification():
    values = np.array([0.0, 0.0, np.nan, 0.0, 1.0, np.inf, 1.0, 1.0])
    actual, _ = finite_constant_windows(values, 4)
    expected, _ = _oracle(values, 4)
    np.testing.assert_array_equal(actual, expected)
    assert actual.tolist() == [True, False, False, False, True]


@pytest.mark.parametrize("values,window", [
    (np.zeros((2, 2)), 2),
    (np.ones(4), 0),
    (np.ones(4), -1),
])
def test_invalid_constancy_inputs_fail_closed(values, window):
    with pytest.raises(ValueError):
        finite_constant_windows(values, window)
