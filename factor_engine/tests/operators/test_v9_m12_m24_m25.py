import math

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.group_ext import (
    _cs_robust_resid,
    _huber_irls_fit,
)
from factor_engine.cleaned_operators.cross_section.peer_ops import (
    _group_peer_beta_deviation,
    _peer_weighted_mean_ex_self_row,
)


def _panel(values):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame([values], index=[pd.Timestamp("2025-01-02")],
                        columns=[f"s{i}" for i in range(values.size)])


def test_huber_intercept_slope_contract_and_exact_linear_residual():
    x = np.arange(30.0)
    y = 3.0 + 2.0 * x
    intercept, slope = _huber_irls_fit(x, y, True)
    np.testing.assert_allclose([intercept, slope], [3.0, 2.0], atol=1e-10)
    result = _cs_robust_resid(_panel(y), _panel(x), _huber_irls_fit, True)
    np.testing.assert_allclose(result.to_numpy(), 0.0, atol=1e-10)


def test_huber_no_intercept_control_and_outlier_robustness():
    x = np.arange(1.0, 31.0)
    y = 2.0 * x
    np.testing.assert_allclose(_huber_irls_fit(x, y, False), (0.0, 2.0), atol=1e-12)
    contaminated = 3.0 + 2.0 * x
    contaminated[-1] += 500.0
    intercept, slope = _huber_irls_fit(x, contaminated, True)
    ols_slope = float(np.polyfit(x, contaminated, 1)[0])
    assert abs(slope - 2.0) < abs(ols_slope - 2.0)
    assert np.isfinite(intercept)


def test_robust_residual_preserves_axes_and_masks_nonfinite_queries():
    x = np.arange(20.0)
    y = 1.0 + 2.0 * x
    for target, value in (("y", np.nan), ("y", np.inf),
                          ("y", -np.inf), ("x", np.nan),
                          ("x", np.inf), ("x", -np.inf)):
        xx, yy = x.copy(), y.copy()
        (yy if target == "y" else xx)[-1] = value
        result = _cs_robust_resid(_panel(yy), _panel(xx), _huber_irls_fit, True)
        assert result.shape == (1, 20)
        assert list(result.columns) == [f"s{i}" for i in range(20)]
        assert np.isnan(result.iloc[0, -1])
        np.testing.assert_allclose(result.iloc[0, :-1], 0.0, atol=1e-9)


def test_robust_residual_masks_finite_query_when_prediction_overflows():
    x = np.linspace(1e307, 1.1e307, 20)
    y = np.ones(20)
    result = _cs_robust_resid(
        _panel(y), _panel(x), lambda *_: (0.0, 1e308), True,
    )
    assert result.isna().to_numpy().all()


def test_huber_row_prefix_and_translation_scale_equivariance():
    x = np.arange(20.0)
    y = 3.0 + 2.0 * x
    x_panel = pd.concat([_panel(x), _panel(x + 1.0)], ignore_index=True)
    y_panel = pd.concat([_panel(y), _panel(y + 2.0)], ignore_index=True)
    full = _cs_robust_resid(y_panel, x_panel, _huber_irls_fit, True)
    prefix = _cs_robust_resid(y_panel.iloc[:1], x_panel.iloc[:1], _huber_irls_fit, True)
    np.testing.assert_allclose(full.iloc[:1], prefix, atol=1e-10)
    transformed = _cs_robust_resid(
        y_panel * 7.0 - 11.0, x_panel + 13.0, _huber_irls_fit, True,
    )
    np.testing.assert_allclose(transformed, full * 7.0, atol=1e-9)


def test_peer_leave_one_out_avoids_large_self_cancellation():
    actual = _peer_weighted_mean_ex_self_row(
        np.array([1e16, 1.0, 1.0]), np.array(["g", "g", "g"]),
        np.ones(3),
    )
    assert actual[0] == 1.0
    np.testing.assert_allclose(actual[1:], [5e15, 5e15], rtol=2e-16)


def test_peer_weight_scale_permutation_and_invalid_member_contracts():
    x = np.array([1e16, 1.0, 2.0, 7.0])
    group = np.array(["g", "g", "g", "solo"])
    weight = np.array([1.0, 2.0, 3.0, 1.0])
    base = _peer_weighted_mean_ex_self_row(x, group, weight)
    scaled = _peer_weighted_mean_ex_self_row(x, group, weight * 1e8)
    np.testing.assert_allclose(base, scaled, equal_nan=True, rtol=2e-15)
    permutation = np.array([2, 0, 3, 1])
    permuted = _peer_weighted_mean_ex_self_row(
        x[permutation], group[permutation], weight[permutation],
    )
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(base, permuted[inverse], equal_nan=True, rtol=2e-15)
    assert np.isnan(base[3])
    invalid = _peer_weighted_mean_ex_self_row(
        np.array([1.0, 2.0, 3.0]), np.array(["g", "g", "g"]),
        np.array([1.0, 0.0, np.nan]),
    )
    assert np.isnan(invalid).all()


def test_peer_matches_independent_direct_exclusion_reference():
    x = np.array([1e16, -3.0, 1.0, 2.0, 9.0])
    group = np.array(["g"] * 5)
    weight = np.array([1.0, 0.5, 2.0, 3.0, 4.0])
    actual = _peer_weighted_mean_ex_self_row(x, group, weight)
    expected = []
    for excluded in range(x.size):
        peers = [i for i in range(x.size) if i != excluded]
        expected.append(
            math.fsum(weight[i] * x[i] for i in peers)
            / math.fsum(weight[i] for i in peers)
        )
    np.testing.assert_allclose(actual, expected, rtol=2e-15)


def test_peer_public_kernel_keeps_axes_and_matches_direct_exclusion():
    beta = _panel([1e16, 1.0, 1.0])
    group = pd.DataFrame([["g", "g", "g"]], index=beta.index, columns=beta.columns)
    weight = _panel([1.0, 1.0, 1.0])
    result = _group_peer_beta_deviation(beta, group, weight)
    assert result.index.equals(beta.index) and result.columns.equals(beta.columns)
    assert result.iloc[0, 0] == 1e16 - 1.0
    np.testing.assert_allclose(result.iloc[0, 1:], [-5e15, -5e15], rtol=2e-16)
