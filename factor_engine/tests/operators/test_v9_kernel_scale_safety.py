"""Extreme-unit regressions for M35 kernel diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.research_spectral import (
    _kernel_granger_score,
    _residualized_hsic,
    _safe_train_standardize,
)


@pytest.mark.parametrize("scale", [1e-300, -1e-300, 1e300, -1e300])
def test_kernel_granger_is_unit_equivariant_at_extreme_scales(scale):
    rng = np.random.default_rng(3535)
    y = rng.normal(size=120)
    x = rng.normal(size=120)
    expected = _kernel_granger_score(y, x, 2)
    actual = _kernel_granger_score(y * scale, x * scale, 2)
    assert np.isfinite(expected) and np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)


@pytest.mark.parametrize("scale", [1e-300, -1e-300, 1e300, -1e300])
def test_residualized_hsic_is_unit_equivariant_at_extreme_scales(scale):
    rng = np.random.default_rng(3536)
    x, y, z = rng.normal(size=(3, 96))
    expected = _residualized_hsic(x, y, z, 3)
    actual = _residualized_hsic(x * scale, y * scale, z * scale, 3)
    assert np.isfinite(expected) and np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-9, abs=2e-11)


def test_large_offsets_preserve_representable_deviations():
    rng = np.random.default_rng(3537)
    spacing = np.spacing(1e300)
    y = 1e300 + np.round(rng.normal(size=120) * 16.0) * spacing
    x = -1e300 + np.round(rng.normal(size=120) * 16.0) * spacing
    y_dev, x_dev = y - y[0], x - x[0]
    assert _kernel_granger_score(y, x, 2) == pytest.approx(
        _kernel_granger_score(y_dev, x_dev, 2), rel=2e-10, abs=2e-10
    )

    z = 1e300 + np.round(rng.normal(size=120) * 16.0) * spacing
    assert _residualized_hsic(x, y, z, 3) == pytest.approx(
        _residualized_hsic(x_dev, y_dev, z - z[0], 3), rel=2e-9, abs=2e-11
    )


def test_exact_constant_training_coordinates_are_explicitly_degenerate():
    train = np.full(30, 1e300)
    test = np.linspace(-1e300, 1e300, 8)
    train_z, test_z = _safe_train_standardize(train, test)
    assert np.array_equal(train_z, np.zeros_like(train_z))
    assert np.array_equal(test_z, np.zeros_like(test_z))
    target_train, target_test = _safe_train_standardize(
        train, test, preserve_constant_test=True
    )
    assert np.array_equal(target_train, np.zeros_like(target_train))
    assert np.any(target_test != 0.0)

    rng = np.random.default_rng(3538)
    y = rng.normal(size=100)
    assert _kernel_granger_score(y, np.ones(100) * 1e-300, 2) == pytest.approx(0.0, abs=1e-12)
    assert _residualized_hsic(np.ones(96) * 1e300, rng.normal(size=96), rng.normal(size=96), 3) == pytest.approx(0.0, abs=1e-12)


def test_holdout_values_never_change_fitted_training_coordinates():
    rng = np.random.default_rng(3539)
    train = rng.normal(size=(40, 3)) * 1e-300
    ordinary_test = rng.normal(size=(12, 3)) * 1e-300
    poisoned_test = ordinary_test.copy()
    poisoned_test[0] = np.array([1e300, -1e300, 1e300])
    fitted_ordinary, _ = _safe_train_standardize(train, ordinary_test)
    fitted_poisoned, _ = _safe_train_standardize(train, poisoned_test)
    np.testing.assert_array_equal(fitted_ordinary, fitted_poisoned)


def test_constant_training_target_variable_holdout_has_equal_model_loss():
    rng = np.random.default_rng(3540)
    n, lag = 100, 2
    train_n = int(0.7 * (n - lag))
    y = np.ones(n)
    y[lag + train_n :] += rng.normal(size=n - lag - train_n)
    x = rng.normal(size=n)
    assert _kernel_granger_score(y, x, lag) == pytest.approx(0.0, abs=0.0)

    y[:] = 1.0
    assert np.isnan(_kernel_granger_score(y, x, lag))


def test_mixed_block_train_constant_coordinate_has_explicit_domain_policy():
    varying = np.linspace(-2.0, 3.0, 30)
    train = np.column_stack((np.full(30, 5.0), varying))
    test_varying = np.array([-1.5, 0.25, 2.5])
    unchanged = np.column_stack((np.full(3, 5.0), test_varying))
    train_z, test_z = _safe_train_standardize(train, unchanged)

    expected_train = (varying - varying.mean()) / varying.std()
    expected_test = (test_varying - varying.mean()) / varying.std()
    np.testing.assert_allclose(train_z[:, 0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(test_z[:, 0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(train_z[:, 1], expected_train, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(test_z[:, 1], expected_test, rtol=2e-15, atol=2e-15)

    moved = unchanged.copy()
    moved[1, 0] = 6.0
    moved_train_z, moved_test_z = _safe_train_standardize(train, moved)
    np.testing.assert_array_equal(moved_train_z, train_z)
    assert np.isnan(moved_test_z[1, 0])
    assert np.isfinite(moved_test_z[[0, 2]]).all()


def test_nonfinite_kernel_inputs_fail_closed_before_constant_target_shortcut():
    x = np.linspace(-1.0, 1.0, 80)
    for bad in (np.nan, np.inf, -np.inf):
        y = np.ones(80)
        y[-1] = bad
        assert np.isnan(_kernel_granger_score(y, x, 2))


def test_reachable_mixed_constant_lag_extrapolation_fails_closed():
    n, lag = 100, 2
    train_n = int(0.7 * (n - lag))
    x = np.ones(n)
    x[0] = 0.0  # second lag varies in training; first lag stays constant
    x[lag + train_n - 1] = 2.0  # first lag moves only in the holdout block
    y = np.random.default_rng(3541).normal(size=n)
    assert np.isnan(_kernel_granger_score(y, x, lag))


def test_public_winner_extreme_units_are_finite_and_prefix_stable():
    ensure_cleaned_loaded()
    op = OperatorRegistry.get("ts_kernel_granger_score", "pandas_numpy", mode="research")
    rng = np.random.default_rng(3542)
    y_values = rng.normal(size=95)
    x_values = rng.normal(size=95)
    params = {"window": 60, "lag": 2}
    reference = None
    for scale in (1.0, 1e-300, -1e-300, 1e300, -1e300):
        y = pd.DataFrame({"A": y_values * scale})
        x = pd.DataFrame({"A": x_values * scale})
        full = op.calculate(y=y, x=x, **params)
        prefix = op.calculate(y=y.iloc[:80], x=x.iloc[:80], **params)
        pd.testing.assert_series_equal(full["A"].iloc[:80], prefix["A"])
        terminal = float(full["A"].iloc[-1])
        assert np.isfinite(terminal)
        if reference is None:
            reference = terminal
        else:
            assert terminal == pytest.approx(reference, rel=2e-9, abs=2e-10)


def test_public_hsic_extreme_units_are_finite_and_prefix_stable():
    ensure_cleaned_loaded()
    op = OperatorRegistry.get("ts_residualized_hsic", "pandas_numpy", mode="research")
    rng = np.random.default_rng(3543)
    x_values, y_values, z_values = rng.normal(size=(3, 85))
    params = {"window": 48, "purge_gap": 3}
    reference = None
    for scale in (1.0, 1e-300, 1e300):
        x = pd.DataFrame({"A": x_values * scale})
        y = pd.DataFrame({"A": y_values * scale})
        z = pd.DataFrame({"A": z_values * scale})
        full = op.calculate(x=x, y=y, z=z, **params)
        prefix = op.calculate(x=x.iloc[:72], y=y.iloc[:72], z=z.iloc[:72], **params)
        pd.testing.assert_series_equal(full["A"].iloc[:72], prefix["A"])
        terminal = float(full["A"].iloc[-1])
        assert np.isfinite(terminal)
        if reference is None:
            reference = terminal
        else:
            assert terminal == pytest.approx(reference, rel=2e-9, abs=2e-11)
