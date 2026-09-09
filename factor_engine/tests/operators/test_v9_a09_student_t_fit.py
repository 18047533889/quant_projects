from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import t as student_t

from factor_engine.cleaned_operators.advanced_topology import (
    _DF_GRID,
    _t_fit,
    _t_logpdf_vec,
    last_t_fit_status,
    TsFisherInformationShift,
)


def _ll(values, fit):
    df, mu, scale = fit
    return float(np.sum(student_t.logpdf(values, df=df, loc=mu, scale=scale)))


def test_t_fit_small_units_preserve_profile_and_scale():
    values = np.linspace(-1.0, 1.0, 20)
    base = _t_fit(values)
    tiny = _t_fit(values * 1e-8)
    assert base is not None and tiny is not None
    assert tiny[0] == base[0]
    assert tiny[1] == pytest.approx(base[1] * 1e-8, abs=1e-16)
    assert tiny[2] == pytest.approx(base[2] * 1e-8, rel=2e-9)
    assert tiny[2] < 1e-8


def test_t_logpdf_matches_independent_scipy_reference():
    values = np.array([-3.0, -0.5, 0.2, 1.0, 7.0])
    actual = _t_logpdf_vec(values, 4.0, 0.3, 1.7)
    expected = student_t.logpdf(values, df=4.0, loc=0.3, scale=1.7)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_t_fit_has_no_fixed_scale_improvement_and_matches_profile_reference():
    values = np.r_[np.zeros(12), np.linspace(1.0, 15.0, 8)]
    fit = _t_fit(values)
    assert fit is not None
    df, mu, scale = fit
    fitted_ll = _ll(values, fit)
    assert _ll(values, (df, mu, 0.99 * scale)) <= fitted_ll + 1e-10
    assert _ll(values, (df, mu, 1.01 * scale)) <= fitted_ll + 1e-10

    profile = []
    for candidate_df in _DF_GRID:
        result = minimize(
            lambda p: -np.sum(
                student_t.logpdf(values, df=candidate_df, loc=p[0], scale=np.exp(p[1]))
            ),
            np.array([np.median(values), np.log(np.std(values))]),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-11, "fatol": 1e-11},
        )
        assert result.success
        profile.append((-float(result.fun), candidate_df))
    reference_ll, reference_df = max(profile)
    assert df == reference_df
    assert fitted_ll == pytest.approx(reference_ll, abs=2e-7)


def test_t_fit_extreme_outlier_near_constant_and_order_invariance():
    values = np.r_[np.linspace(-1e-14, 1e-14, 30), 1e-9]
    fit = _t_fit(values)
    reversed_fit = _t_fit(values[::-1])
    assert fit is not None and reversed_fit is not None
    np.testing.assert_allclose(fit, reversed_fit, rtol=2e-7, atol=1e-25)
    assert fit[2] > 0.0
    assert _t_fit(np.ones(20)) is None
    assert last_t_fit_status()["reason"] == "degenerate_scale"


def test_t_fit_iteration_limit_reports_non_convergence():
    values = np.r_[np.zeros(12), np.linspace(1.0, 15.0, 8)]
    assert _t_fit(values, max_iter=0) is None
    assert last_t_fit_status()["reason"] == "NON_CONVERGED"


def test_fisher_shift_is_prefix_causal_and_uses_disjoint_prior_block():
    rng = np.random.default_rng(909)
    values = rng.standard_t(df=4.0, size=70)
    index = pd.date_range("2020-01-01", periods=70)
    base = pd.DataFrame({"A": values}, index=index)
    op = TsFisherInformationShift()
    full = op.calculate(base, recent_window=20, prior_window=30)
    extended_values = np.r_[values, [1e9, -1e9]]
    extended_index = pd.date_range("2020-01-01", periods=72)
    extended = op.calculate(
        pd.DataFrame({"A": extended_values}, index=extended_index),
        recent_window=20,
        prior_window=30,
    )
    np.testing.assert_array_equal(full["A"].to_numpy(), extended["A"].iloc[:70].to_numpy())
