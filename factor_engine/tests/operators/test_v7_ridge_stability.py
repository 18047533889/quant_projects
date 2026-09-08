import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    last_fit_status,
    ols_fit,
    ridge_fit,
)

# Importing the module registers the real canonical dependants used below.
import factor_engine.cleaned_operators.ts_model.dynamic_regression  # noqa: F401,E402


def _closed_form_one_feature(x, y, alpha):
    centered_x = x - np.mean(x)
    slope = float(centered_x @ (y - np.mean(y)) / (centered_x @ centered_x + alpha))
    return np.array([float(np.mean(y) - np.mean(x) * slope), slope])


def test_high_offset_matches_closed_form_and_sklearn_svd():
    x = 1.0e12 + np.linspace(-2.0, 2.0, 41)
    y = 11.0 + 3.0 * (x - 1.0e12)
    design = np.column_stack((np.ones(x.size), x))

    actual = ridge_fit(design, y, alpha=0.1, has_intercept=True)
    closed = _closed_form_one_feature(x, y, alpha=0.1)
    reference = Ridge(alpha=0.1, fit_intercept=True, solver="svd").fit(x[:, None], y)

    np.testing.assert_allclose(actual[1], closed[1], rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(actual[1], reference.coef_[0], rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(design @ actual, reference.predict(x[:, None]), rtol=0.0, atol=1e-3)


@pytest.mark.parametrize("with_intercept", [True, False])
def test_original_unit_penalty_matches_sklearn_under_units_and_column_permutation(with_intercept):
    rng = np.random.default_rng(20260908)
    features = rng.normal(size=(80, 3)) * np.array([1.0e-4, 3.0, 2.0e3])
    y = features @ np.array([2.5e3, -0.7, 4.0e-4]) + rng.normal(scale=0.02, size=80)
    if with_intercept:
        y += 4.0
    order = np.array([2, 0, 1])
    permuted = features[:, order]
    design = np.column_stack((np.ones(len(y)), permuted)) if with_intercept else permuted

    actual = ridge_fit(design, y, alpha=0.35, has_intercept=with_intercept)
    reference = Ridge(alpha=0.35, fit_intercept=with_intercept, solver="svd").fit(permuted, y)
    expected = np.concatenate(([reference.intercept_], reference.coef_)) if with_intercept else reference.coef_

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(design @ actual, reference.predict(permuted), rtol=2e-11, atol=2e-10)


def test_alpha_zero_preserves_ols_policy_and_positive_ridge_supports_p_greater_n():
    rng = np.random.default_rng(7)
    regular = np.column_stack((np.ones(20), rng.normal(size=(20, 2))))
    y_regular = rng.normal(size=20)
    np.testing.assert_array_equal(
        ridge_fit(regular, y_regular, alpha=0.0, has_intercept=True),
        ols_fit(regular, y_regular),
    )

    wide_features = rng.normal(size=(4, 7))
    wide_design = np.column_stack((np.ones(4), wide_features))
    y_wide = rng.normal(size=4)
    assert ridge_fit(wide_design, y_wide, alpha=0.0, has_intercept=True) is None

    actual = ridge_fit(wide_design, y_wide, alpha=0.8, has_intercept=True)
    reference = Ridge(alpha=0.8, fit_intercept=True, solver="svd").fit(wide_features, y_wide)
    expected = np.concatenate(([reference.intercept_], reference.coef_))
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    collinear = np.column_stack((np.ones(20), regular[:, 1], 2.0 * regular[:, 1]))
    assert ridge_fit(collinear, y_regular, alpha=0.0, has_intercept=True) is None
    actual_collinear = ridge_fit(collinear, y_regular, alpha=0.8, has_intercept=True)
    reference_collinear = Ridge(alpha=0.8, fit_intercept=True, solver="svd").fit(
        collinear[:, 1:], y_regular
    )
    np.testing.assert_allclose(
        actual_collinear,
        np.concatenate(([reference_collinear.intercept_], reference_collinear.coef_)),
        rtol=3e-13,
        atol=3e-13,
    )


def test_constant_penalized_columns_and_invalid_inputs_fail_or_resolve_explicitly():
    x = np.arange(12.0)
    design = np.column_stack((np.ones(12), x, np.full(12, 5.0)))
    y = 2.0 + 0.4 * x
    beta = ridge_fit(design, y, alpha=0.2, has_intercept=True)
    assert beta[2] == 0.0
    assert last_fit_status() == {"converged": True, "reason": "converged"}

    bad = design.copy()
    bad[-1, 1] = np.inf
    assert ridge_fit(bad, y, alpha=0.2, has_intercept=True) is None
    assert last_fit_status()["reason"] == "invalid_params"
    assert ridge_fit(design, y, alpha=-1.0, has_intercept=True) is None
    assert ridge_fit(design, y, alpha=np.nan, has_intercept=True) is None


def test_registered_ridge_dependants_execute_stable_kernel_on_high_offset_panel():
    dates = pd.date_range("2024-01-01", periods=42, freq="D")
    x_values = 1.0e12 + np.linspace(-2.05, 2.05, len(dates))
    y_values = 17.0 + 2.25 * (x_values - 1.0e12)
    x = pd.DataFrame({"A": x_values}, index=dates)
    y = pd.DataFrame({"A": y_values}, index=dates)
    args = dict(window=41, min_periods=10, add_intercept=True, coefficient_index=1)

    fitted = OperatorRegistry.get("ts_ridge_regression_coeff").calculate(y, x, **args)
    prior = OperatorRegistry.get("ts_ridge_regression_coeff_prior").calculate(y, x, **args)
    forecast = OperatorRegistry.get("ts_ridge_regression_forecast_error").calculate(y, x, **args)
    diagnostic_z = OperatorRegistry.get("ts_ridge_regression_resid_z").calculate(y, x, **args)
    forecast_z = OperatorRegistry.get("ts_ridge_regression_forecast_error_z").calculate(y, x, **args)

    expected_fitted = _closed_form_one_feature(x_values[-41:], y_values[-41:], 0.1)[1]
    prior_model = Ridge(alpha=0.1, fit_intercept=True, solver="svd").fit(
        x_values[:41, None], y_values[:41]
    )
    expected_forecast = y_values[-1] - prior_model.predict(x_values[-1:, None])[0]
    assert fitted.iloc[-1, 0] == pytest.approx(expected_fitted, rel=2e-13, abs=1e-13)
    assert prior.iloc[-1, 0] == pytest.approx(prior_model.coef_[0], rel=2e-13, abs=1e-13)
    assert forecast.iloc[-1, 0] == pytest.approx(expected_forecast, abs=1e-3)
    assert np.isfinite(diagnostic_z.iloc[-1, 0])
    assert np.isfinite(forecast_z.iloc[-1, 0])


def test_registered_predictive_and_in_sample_ridge_residuals_are_stable_at_high_offset():
    dates = pd.date_range("2024-01-01", periods=42, freq="D")
    x_values = 1.0e12 + np.linspace(-2.05, 2.05, len(dates))
    y_values = 9.0 + 1.75 * (x_values - 1.0e12)
    # Make the current point a known innovation so the oracle is non-vacuous.
    y_values[-1] += 0.25
    x = pd.DataFrame({"A": x_values}, index=dates)
    y = pd.DataFrame({"A": y_values}, index=dates)

    predictive = OperatorRegistry.get("ts_ridge_regression_predictive_resid").calculate(
        y, x, window=41, alpha=0.1, min_periods=10)
    prior = Ridge(alpha=0.1, fit_intercept=True, solver="svd").fit(
        x_values[:41, None], y_values[:41])
    expected_predictive = y_values[-1] - prior.predict(x_values[-1:, None])[0]
    assert predictive.iloc[-1, 0] == pytest.approx(expected_predictive, abs=1e-3)

    in_sample = OperatorRegistry.get("ts_ridge_regression_in_sample_resid").calculate(
        y, x, window=41, alpha=0.1, min_periods=10)
    current = Ridge(alpha=0.1, fit_intercept=True, solver="svd").fit(
        x_values[-41:, None], y_values[-41:])
    expected_in_sample = y_values[-1] - current.predict(x_values[-1:, None])[0]
    assert in_sample.iloc[-1, 0] == pytest.approx(expected_in_sample, abs=1e-3)
