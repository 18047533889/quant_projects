from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(values) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def test_registered_multi_forecast_error_is_current_oos_innovation():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    x = np.linspace(-2.0, 2.0, 40)
    y = 1.5 + 2.25 * x
    y[-1] += 7.0
    yp, xp = _panel(y), _panel(x)
    prior = OperatorRegistry.get("ts_multi_regression_forecast_error")
    diagnostic = OperatorRegistry.get("ts_multi_regression_resid")
    assert type(prior).__module__.endswith("ts_model.dynamic_regression")
    got = prior.calculate(yp, xp, window=30, min_periods=10)["A"].iloc[-1]
    in_sample = diagnostic.calculate(yp, xp, window=30, min_periods=10)["A"].iloc[-1]
    np.testing.assert_allclose(got, 7.0, rtol=1e-13, atol=1e-13)
    assert abs(in_sample) < abs(got)


def test_current_target_changes_innovation_not_prior_fit():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(1707)
    x = rng.normal(size=50)
    y = -0.4 + 1.7 * x + rng.normal(scale=0.03, size=50)
    op = OperatorRegistry.get("ts_multi_regression_forecast_error")
    base = op.calculate(_panel(y), _panel(x), window=35, min_periods=10)["A"].iloc[-1]
    changed = y.copy()
    changed[-1] += 5.0
    after = op.calculate(_panel(changed), _panel(x), window=35, min_periods=10)["A"].iloc[-1]
    np.testing.assert_allclose(after - base, 5.0, rtol=1e-13, atol=1e-13)


def test_future_suffix_preserves_registered_residual_prefix():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(1708)
    x = rng.normal(size=70)
    y = 0.8 - 0.6 * x + rng.normal(scale=0.08, size=70)
    op = OperatorRegistry.get("ts_multi_regression_forecast_error_z")
    base = op.calculate(_panel(y), _panel(x), window=30, min_periods=10)["A"]
    x2 = np.r_[x, [100.0, np.nan, -100.0]]
    y2 = np.r_[y, [-500.0, np.inf, 500.0]]
    extended = op.calculate(_panel(y2), _panel(x2), window=30, min_periods=10)["A"]
    np.testing.assert_allclose(extended.iloc[: len(base)], base, equal_nan=True)


def test_regression_models_predictive_and_in_sample_roles_are_distinct():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    x = np.linspace(-1.0, 1.0, 45)
    y = 2.0 + 0.75 * x
    y[-1] += 4.0
    predictive = OperatorRegistry.get("ts_huber_regression_predictive_resid")
    diagnostic = OperatorRegistry.get("ts_huber_regression_in_sample_resid")
    assert type(predictive).__module__.endswith("regression_models")
    pred = predictive.calculate(_panel(y), _panel(x), window=30, min_periods=10)["A"].iloc[-1]
    ins = diagnostic.calculate(_panel(y), _panel(x), window=30, min_periods=10)["A"].iloc[-1]
    np.testing.assert_allclose(pred, 4.0, rtol=1e-12, atol=1e-12)
    assert abs(ins) < abs(pred)
    assert "diagnostic_only" in diagnostic.metadata.tags
    assert "diagnostic_only" not in predictive.metadata.tags
