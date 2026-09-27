"""Independent witnesses for the public VaR and expected shortfall contract."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_evaluator.metrics.risk.var_cvar import (
    compute_cvar,
    compute_var,
    compute_var_cvar,
    empirical_expected_shortfall,
)


def test_historical_tail_boundary_and_interpolated_var_are_distinct():
    returns = np.array([-0.12, -0.04, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    var, cvar = compute_var_cvar(returns, 0.75, min_periods=8)
    assert var == pytest.approx(0.01)
    assert cvar == pytest.approx(0.08)
    assert empirical_expected_shortfall(returns, 0.75) == pytest.approx(0.08)


def test_finite_filter_and_min_periods_apply_per_column():
    good = np.array([-0.12, -0.04, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    panel = np.column_stack((good, good))
    panel[0, 0] = np.nan
    panel[1, 0] = np.inf
    panel[7, 1] = -np.inf
    var, cvar = compute_var_cvar(panel, 0.75, min_periods=7)
    assert np.isnan(var[0]) and np.isnan(cvar[0])
    assert var[1] == pytest.approx(0.02)
    assert cvar[1] == pytest.approx((0.12 + 0.75 * 0.04) / 1.75)


def test_parametric_normal_closed_form_and_constant_series():
    returns = np.array([-0.03, -0.01, 0.01, 0.03])
    sample_std = np.sqrt(0.002 / 3)
    z = norm.ppf(0.1)
    expected_var = -z * sample_std
    expected_es = sample_std * norm.pdf(z) / 0.1
    var, cvar = compute_var_cvar(returns, 0.9, method="parametric", min_periods=4)
    assert var == pytest.approx(expected_var)
    assert cvar == pytest.approx(expected_es)
    assert np.isnan(compute_var(np.zeros(4), 0.9, "parametric", 4))
    assert np.isnan(compute_cvar(np.zeros(4), 0.9, "parametric", 4))
