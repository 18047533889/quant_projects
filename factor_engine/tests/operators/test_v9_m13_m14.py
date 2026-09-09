import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.technical.convex_trend_filter import (
    TrendFilterConvergenceError,
    difference_l1_filter,
    l1_trend_filter,
    total_variation_filter,
)
from factor_engine.cleaned_operators.technical.denoise_filter import (
    TsL1TrendFilterTrailing,
    TsTotalVariationFilterTrailing,
    _l1_trend_filter,
    _total_variation_filter,
)
from factor_engine.cleaned_operators.technical.polars_denoise_filter import (
    TsL1TrendFilterTrailingPolars,
    TsTotalVariationFilterTrailingPolars,
    _l1_trend_filter as _polars_l1_helper,
    _total_variation_filter as _polars_tv_helper,
)


def _objective(x, y, penalty, order):
    return 0.5 * np.sum((y - x) ** 2) + penalty * np.sum(np.abs(np.diff(y, n=order)))


@pytest.mark.parametrize("order,solver", [(1, total_variation_filter), (2, l1_trend_filter)])
def test_lambda_zero_identity_and_objective_improves(order, solver):
    x = np.r_[np.zeros(10), np.full(10, 10.0)]
    assert np.array_equal(solver(x, 0.0), x)
    y = solver(x, 1.0)
    assert _objective(x, y, 1.0, order) <= _objective(x, x, 1.0, order) + 1e-10


def test_two_point_tv_has_analytic_solution():
    np.testing.assert_allclose(total_variation_filter(np.array([0.0, 10.0]), 2.0), [2.0, 8.0], atol=2e-6)
    np.testing.assert_allclose(total_variation_filter(np.array([0.0, 1.0]), 2.0), [.5, .5], atol=2e-6)


def test_l1_trend_preserves_every_affine_line():
    x = 7.0 - 2.5 * np.arange(40.0)
    for penalty in (0.1, 1.0, 100.0):
        np.testing.assert_allclose(l1_trend_filter(x, penalty), x, atol=2e-8)


@pytest.mark.parametrize("order", [1, 2])
def test_solution_satisfies_independent_kkt_conditions(order):
    x = np.array([0.2, 2.0, -1.0, 3.5, 1.1, -0.4])
    penalty = 0.7
    y = difference_l1_filter(x, penalty, order=order)
    d = np.diff(np.eye(x.size), n=order, axis=0)
    dual = np.linalg.lstsq(d.T, x - y, rcond=None)[0]
    np.testing.assert_allclose(y - x + d.T @ dual, 0.0, atol=3e-6)
    assert np.max(np.abs(dual)) <= penalty + 3e-6
    active = np.abs(d @ y) > 3e-6
    np.testing.assert_allclose(dual[active], penalty * np.sign((d @ y)[active]), atol=3e-6)


@pytest.mark.parametrize("order", [1, 2])
def test_matches_independent_small_auxiliary_qp(order):
    from scipy.optimize import LinearConstraint, minimize

    x = np.array([0.0, 2.0, -1.0, 3.0, 1.0])
    penalty = 0.6
    d = np.diff(np.eye(x.size), n=order, axis=0)
    m = d.shape[0]
    # q=(y,t), t >= +/- D y turns the nonsmooth objective into a smooth QP.
    constraint_matrix = np.block([[-d, np.eye(m)], [d, np.eye(m)]])
    constraint = LinearConstraint(constraint_matrix, 0.0, np.inf)
    initial = np.r_[x, np.abs(d @ x) + 1e-3]
    reference = minimize(
        lambda q: 0.5 * np.sum((q[:x.size] - x) ** 2) + penalty * np.sum(q[x.size:]),
        initial,
        jac=lambda q: np.r_[q[:x.size] - x, np.full(m, penalty)],
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert reference.success
    actual = difference_l1_filter(x, penalty, order=order)
    np.testing.assert_allclose(actual, reference.x[:x.size], atol=4e-6)


@pytest.mark.parametrize("solver", [total_variation_filter, l1_trend_filter])
def test_joint_data_penalty_scaling(solver):
    x = np.array([0.0, 2.0, -1.0, 4.0, 3.0, 8.0])
    baseline = solver(x, 0.4)
    np.testing.assert_allclose(solver(x * 11.0, 4.4), baseline * 11.0, atol=2e-5)


def test_shared_helpers_have_exact_backend_parity():
    x = np.r_[np.zeros(8), np.full(8, 5.0)]
    np.testing.assert_array_equal(_total_variation_filter(x, 1.0), _polars_tv_helper(x, 1.0))
    np.testing.assert_array_equal(_l1_trend_filter(x, 1.0), _polars_l1_helper(x, 1.0))


def test_iteration_budget_fails_closed():
    with pytest.raises(TrendFilterConvergenceError):
        difference_l1_filter(np.array([0.0, 5.0, -2.0, 4.0]), 1.0, order=2, max_iter=1)


@pytest.mark.parametrize("order", [1, 2])
def test_huge_finite_input_cannot_be_certified_by_infinite_norms(order):
    x = np.array([1e308, -1e308, 1e308, -1e308])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(TrendFilterConvergenceError):
            difference_l1_filter(x, 1.0, order=order)


@pytest.mark.parametrize("operator,penalty_name", [
    (TsTotalVariationFilterTrailing(), "lambda_tv"),
    (TsL1TrendFilterTrailing(), "lambda_l1"),
])
def test_trailing_operator_missing_policy_axes_and_prefix(operator, penalty_name):
    values = np.linspace(0.0, 4.0, 16)
    values[4] = np.nan
    frame = pd.DataFrame({"a": values, "b": values + 3.0})
    kwargs = {penalty_name: 0.3, "window": 8}
    full = operator._calculate_series(frame, **kwargs)
    prefix = operator._calculate_series(frame.iloc[:12], **kwargs)
    assert full.index.equals(frame.index) and full.columns.equals(frame.columns)
    np.testing.assert_allclose(full.iloc[:12], prefix, equal_nan=True, atol=2e-6)


@pytest.mark.parametrize("operator,penalty_name", [
    (TsTotalVariationFilterTrailingPolars(), "lambda_tv"),
    (TsL1TrendFilterTrailingPolars(), "lambda_l1"),
])
def test_polars_wrapper_prefix_and_shared_missing_policy(operator, penalty_name):
    pl = pytest.importorskip("polars")
    values = np.linspace(0.0, 4.0, 16)
    values[4] = np.nan
    frame = pl.DataFrame({"a": values, "b": values + 3.0})
    kwargs = {penalty_name: 0.3, "window": 8}
    full = operator._calculate_series(frame, **kwargs).to_numpy()
    prefix = operator._calculate_series(frame.head(12), **kwargs).to_numpy()
    np.testing.assert_allclose(full[:12], prefix, equal_nan=True, atol=2e-6)
