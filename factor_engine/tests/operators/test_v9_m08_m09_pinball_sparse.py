import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from factor_engine.cleaned_operators.extreme_tail import (
    TsQuantileRegressionBeta,
    _quantile_beta,
)
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    last_fit_status,
    pinball_quantile_fit,
)


def _pinball(residual, q):
    return np.sum(np.where(residual >= 0, q * residual, (q - 1) * residual))


def _line_oracle(x, y, q):
    losses = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            if x[i] != x[j]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                intercept = y[i] - slope * x[i]
                losses.append(_pinball(y - intercept - slope * x, q))
    return min(losses)


def test_sparse_lp_matches_independent_vertex_oracle():
    rng = np.random.default_rng(81)
    x = rng.normal(size=30)
    y = 1.5 + 2.25 * x + rng.standard_t(3, size=30)
    q = 0.2
    beta = pinball_quantile_fit(np.c_[np.ones(30), x], y, q)
    assert beta is not None
    loss = _pinball(y - np.c_[np.ones(30), x] @ beta, q)
    np.testing.assert_allclose(loss, _line_oracle(x, y, q), atol=1e-9)


def test_rank_deficient_coefficients_fail_even_if_loss_is_defined():
    x = np.linspace(-2, 2, 20)
    design = np.c_[np.ones(20), x, 2 * x]
    assert pinball_quantile_fit(design, 1 + 3 * x, 0.5) is None
    status = last_fit_status()
    assert status["reason"] == "singular"
    assert status["rank"] == 2


def test_constant_feature_plus_intercept_is_unidentifiable():
    design = np.c_[np.ones(12), np.ones(12)]
    assert pinball_quantile_fit(design, np.arange(12.0), 0.5) is None
    assert last_fit_status()["reason"] == "singular"


def test_sparse_shape_and_finite_solver_budget(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    real = core._linprog
    seen = {}

    def capture(*args, **kwargs):
        seen["sparse"] = issparse(kwargs["A_eq"])
        seen["shape"] = kwargs["A_eq"].shape
        seen["nnz"] = kwargs["A_eq"].nnz
        seen["options"] = kwargs["options"]
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "_linprog", capture)
    x = np.linspace(-1, 1, 50)
    assert pinball_quantile_fit(np.c_[np.ones(50), x], 2 + x, 0.4) is not None
    assert seen["sparse"] and seen["shape"] == (50, 102)
    assert seen["nnz"] <= 4 * 50
    assert seen["options"] == {"maxiter": 10_000, "time_limit": 5.0}


def test_resource_rejection_precedes_sparse_allocation(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    monkeypatch.setattr(core, "_linprog",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    x = np.linspace(-1, 1, 10_001)
    assert pinball_quantile_fit(np.c_[np.ones(x.size), x], 2 * x, 0.5) is None
    assert last_fit_status()["reason"] == "resource_limit"


def test_solver_timeout_fails_closed(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    class Failed:
        success = False
        status = 1
        x = None
        fun = None
    monkeypatch.setattr(core, "_linprog", lambda *a, **k: Failed())
    x = np.arange(10.0)
    assert pinball_quantile_fit(np.c_[np.ones(10), x], x, 0.5) is None
    assert last_fit_status()["reason"] == "non_converged"


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_invalid_quantile_endpoint(q):
    x = np.arange(10.0)
    assert pinball_quantile_fit(np.c_[np.ones(10), x], x, q) is None
    assert last_fit_status()["reason"] == "invalid_params"


def test_large_translation_and_signed_extreme_unit_scaling():
    x = np.linspace(-3, 2, 31)
    y = 1e14 + 1 + 2 * x + 0.2 * np.sin(np.arange(x.size))
    base = pinball_quantile_fit(np.c_[np.ones(x.size), x], y, 0.3)
    assert base is not None
    base_loss = _pinball(y - np.c_[np.ones(x.size), x] @ base, 0.3)
    unit_y = 1 + 2 * x + 0.2 * np.sin(np.arange(x.size))
    unit_base = pinball_quantile_fit(np.c_[np.ones(x.size), x], unit_y, 0.3)
    unit_loss = _pinball(unit_y - np.c_[np.ones(x.size), x] @ unit_base, 0.3)
    unit_high = pinball_quantile_fit(np.c_[np.ones(x.size), x], unit_y, 0.7)
    unit_high_loss = _pinball(unit_y - np.c_[np.ones(x.size), x] @ unit_high, 0.7)
    for factor in (1e-300, -1e-300, 1e300, -1e300):
        bx = pinball_quantile_fit(np.c_[np.ones(x.size), x * factor], y, 0.3)
        assert bx is not None
        np.testing.assert_allclose(
            _pinball(y - np.c_[np.ones(x.size), x * factor] @ bx, 0.3),
            base_loss, atol=0.02)
        by = pinball_quantile_fit(np.c_[np.ones(x.size), x], unit_y * factor, 0.3)
        assert by is not None
        scaled_loss = _pinball(
            (unit_y * factor - np.c_[np.ones(x.size), x] @ by) / abs(factor), 0.3)
        expected_loss = unit_loss if factor > 0 else unit_high_loss
        np.testing.assert_allclose(scaled_loss, expected_loss, atol=0.02)


def test_extreme_tail_caller_uses_shared_identifiability_policy():
    x = np.linspace(-2, 2, 20)
    y = 3 + 4 * x
    np.testing.assert_allclose(_quantile_beta(x, y, 0.5), 4.0, atol=1e-12)
    assert np.isnan(_quantile_beta(np.ones(20), y, 0.5))


def test_public_extreme_tail_prefix_causality():
    x = pd.DataFrame({"a": np.linspace(-2, 2, 25)})
    y = pd.DataFrame({"a": 1 + 3 * x["a"] + 0.1 * np.sin(np.arange(25))})
    op = TsQuantileRegressionBeta()
    full = op.calculate(y, x, window=10, quantile=0.5)
    prefix = op.calculate(y.iloc[:-1], x.iloc[:-1], window=10, quantile=0.5)
    pd.testing.assert_frame_equal(full.iloc[:-1], prefix)


def test_public_huber_exact_zero_has_coefficients_but_undefined_standardized_residual():
    import factor_engine.cleaned_operators.ts_model.dynamic_regression
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    x = pd.DataFrame({"a": np.linspace(-2, 2, 20)})
    y = pd.DataFrame({"a": np.zeros(20)})
    coefficient = OperatorRegistry.get(
        "ts_huber_regression_coeff", "pandas_numpy", mode="research")
    standardized = OperatorRegistry.get(
        "ts_huber_regression_resid_z", "pandas_numpy", mode="research")
    coeff_result = coefficient.calculate(
        y, x, window=20, coefficient_index=1, min_periods=5,
        add_intercept=True, warmup_policy="expanding")
    z_result = standardized.calculate(
        y, x, window=20, coefficient_index=1, min_periods=5,
        add_intercept=True, warmup_policy="expanding")
    np.testing.assert_allclose(coeff_result.iloc[-1, 0], 0.0, atol=0.0)
    assert np.isnan(z_result.iloc[-1, 0])


def test_full_rank_can_still_have_nonunique_quantile_coefficient():
    design = np.ones((4, 1))
    target = np.array([0.0, 0.0, 1.0, 1.0])
    assert np.linalg.matrix_rank(design) == 1
    assert pinball_quantile_fit(design, target, 0.5) is None
    status = last_fit_status()
    assert status["reason"] == "coefficient_not_identified"
    assert status["strict_active_rank"] < 1
    assert abs(status["duality_gap"]) <= 1e-7


def test_no_intercept_tiny_response_keeps_relative_lp_scale():
    x = np.linspace(1.0, 3.0, 20)
    target = 2.0e-300 * x
    beta = pinball_quantile_fit(x.reshape(-1, 1), target, 0.5)
    assert beta is not None
    np.testing.assert_allclose(beta, [2.0e-300], rtol=1e-12, atol=0.0)
    residual = target - x * beta[0]
    assert np.max(np.abs(residual)) <= np.finfo(float).smallest_subnormal


def test_fake_success_dual_bounds_and_gap_do_not_replace_stationarity(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    real = core._linprog

    def fake(*args, **kwargs):
        result = real(*args, **kwargs)
        n = kwargs["b_eq"].size
        result.eqlin.marginals = np.full(n, 0.1)
        result.fun = float(kwargs["b_eq"] @ result.eqlin.marginals)
        return result

    monkeypatch.setattr(core, "_linprog", fake)
    x = np.linspace(-2, 2, 10)
    assert pinball_quantile_fit(np.c_[np.ones(10), x], 1 + 2 * x, 0.5) is None
    status = last_fit_status()
    assert status["reason"] == "non_converged"
    assert status["stationarity"] > 1e-7
    assert abs(status["duality_gap"]) <= 1e-7


def test_fake_success_negative_residual_variables_fail_primal_certificate(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    real = core._linprog

    def fake(*args, **kwargs):
        result = real(*args, **kwargs)
        p = kwargs["A_eq"].shape[1] - 2 * kwargs["b_eq"].size
        n = kwargs["b_eq"].size
        result.x[p : p + n] -= 1e-3
        result.x[p + n :] -= 1e-3
        return result

    monkeypatch.setattr(core, "_linprog", fake)
    x = np.linspace(-2, 2, 10)
    assert pinball_quantile_fit(np.c_[np.ones(10), x], 1 + 2 * x, 0.5) is None
    assert last_fit_status()["reason"] == "non_converged"


def test_malformed_dual_shape_fails_closed_before_dot_product(monkeypatch):
    import factor_engine.cleaned_operators.ts_model._rolling_core as core
    real = core._linprog

    def fake(*args, **kwargs):
        result = real(*args, **kwargs)
        result.eqlin.marginals = np.zeros(kwargs["b_eq"].size - 1)
        return result

    monkeypatch.setattr(core, "_linprog", fake)
    x = np.linspace(-2, 2, 10)
    assert pinball_quantile_fit(np.c_[np.ones(10), x], 1 + 2 * x, 0.5) is None
    assert last_fit_status()["reason"] == "coefficient_not_identified"
