import numpy as np
import pandas as pd
from scipy.optimize import linprog as reference_linprog
from scipy.sparse import issparse

from factor_engine.cleaned_operators.group_ext import (
    CsLadResid,
    _huber_irls_fit,
    _lad_fit,
)


AUDIT_X = np.array([
    0.30471707975443135, -1.0399841062404955, 0.7504511958064572,
    0.9405647163912139, -1.9510351886538364, -1.302179506862318,
    0.12784040316728537, -0.3162425923435822, -0.016801157504288795,
    -0.85304392757358, 0.8793979748628286, 0.7777919354289483,
    0.06603069756121605, 1.1272412069680329, 0.4675093422520456,
    -0.8592924628832382, 0.36875078408249884, -0.9588826008289989,
    0.8784503013072725, -0.049925910986252896, -0.18486236354526056,
    -0.6809295444039414, 1.2225413386740303, -0.15452948206880215,
    -0.4283278221631072, -0.3521335504882296, 0.5323091855533487,
    0.36544406436407834, 0.4127326115959884, 0.43082100300788273])
AUDIT_Y = np.array([
    -11.772898876024291, -4.8752176349638505, 5.3666415093978905,
    9.870144734518544, 3.8751396545425365, -0.7775662278660627,
    2.269573751846981, 0.21111574600672522, 1.125115311795894,
    0.09146100510396105, 5.381448095791927, 4.876530074592041,
    1.5325823853949538, 5.6138849439708185, 3.519213835896865,
    -0.35918879192070163, 3.9776811301956863, -0.6530522537123143,
    5.314264466993713, 1.9178013365301327, 1.7345323080542023,
    0.5884995926267163, 4.210468196166424, 1.216740337436292,
    0.2446438792178829, 0.3047215002919692, 3.3217853054333624,
    4.591273504326631, 2.372366719094722, 4.260741363615129])


def _pairwise_line_oracle(x, y):
    candidates = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            if x[i] != x[j]:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                intercept = y[i] - slope * x[i]
                candidates.append(np.sum(np.abs(y - intercept - slope * x)))
    return min(candidates)


def test_audit_seed42_reaches_global_l1_objective():
    fit = _lad_fit(AUDIT_X, AUDIT_Y, True)
    assert fit is not None
    intercept, slope = fit
    objective = np.sum(np.abs(AUDIT_Y - intercept - slope * AUDIT_X))
    np.testing.assert_allclose(objective, 46.8536370915632, rtol=0, atol=1e-9)
    np.testing.assert_allclose(objective, _pairwise_line_oracle(AUDIT_X, AUDIT_Y),
                               rtol=0, atol=1e-9)


def test_outlier_and_high_leverage_global_oracle():
    x = np.array([-100.0, -2, -1, 0, 1, 2, 100.0])
    y = 3 + 2 * x
    y[[1, 5]] += [30, -40]
    fit = _lad_fit(x, y, True)
    assert fit is not None
    got = np.sum(np.abs(y - fit[0] - fit[1] * x))
    np.testing.assert_allclose(got, _pairwise_line_oracle(x, y), atol=1e-10)


def test_no_intercept_and_zero_design_policy():
    x = np.array([-3.0, -1, 0, 2, 4])
    y = 2 * x
    assert _lad_fit(x, y, False) == (0.0, 2.0)
    assert _lad_fit(np.zeros(5), np.arange(5.0), False) is None


def test_nonunique_coefficients_are_judged_by_objective():
    x = np.array([-1.0, 0.0, 0.0, 1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    fit = _lad_fit(x, y, True)
    assert fit is not None
    objective = np.sum(np.abs(y - fit[0] - fit[1] * x))
    np.testing.assert_allclose(objective, _pairwise_line_oracle(x, y), atol=1e-10)


def test_solver_failure_has_no_coordinate_fallback(monkeypatch):
    import scipy.optimize

    def failed(*args, **kwargs):
        class Result:
            success = False
            status = 1
            x = None
        return Result()

    monkeypatch.setattr(scipy.optimize, "linprog", failed)
    assert _lad_fit(AUDIT_X, AUDIT_Y, True) is None


def test_sparse_constraints_and_finite_solver_budget(monkeypatch):
    import scipy.optimize
    seen = {}
    real = scipy.optimize.linprog

    def capture(*args, **kwargs):
        seen["sparse"] = issparse(kwargs["A_ub"])
        seen["options"] = kwargs["options"]
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", capture)
    assert _lad_fit(AUDIT_X, AUDIT_Y, True) is not None
    assert seen["sparse"]
    assert seen["options"]["maxiter"] == 10_000
    assert seen["options"]["time_limit"] == 5.0


def test_resource_cap_rejects_before_solver(monkeypatch):
    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "linprog",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    x = np.linspace(-1, 1, 10_001)
    assert _lad_fit(x, 2 * x, True) is None


def test_public_residual_and_m24_physical_scatter():
    x = pd.DataFrame([np.linspace(-2, 2, 12)], columns=list("abcdefghijkl"))
    y = 1.5 + 2.0 * x
    y.iloc[0, 4] = np.nan
    result = CsLadResid().calculate(y, x, add_intercept=True)
    assert np.isnan(result.iloc[0, 4])
    np.testing.assert_allclose(result.drop(columns="e"), 0.0, atol=1e-12)


def test_m12_huber_intercept_slope_order_preserved():
    x = np.arange(30.0)
    y = 3.0 + 2.0 * x
    fit = _huber_irls_fit(x, y, True)
    assert fit is not None
    np.testing.assert_allclose(fit, (3.0, 2.0), atol=1e-10)


def test_large_y_translation_preserves_representable_lad_variation():
    x = np.linspace(-3.0, 2.0, 41)
    perturbation = 0.25 * np.sin(np.arange(x.size))
    y = 1.0 + 2.0 * x + perturbation
    base = _lad_fit(x, y, True)
    shifted_y = y + 1e14
    shifted = _lad_fit(x, shifted_y, True)
    assert base is not None and shifted is not None
    base_objective = np.sum(np.abs(y - base[0] - base[1] * x))
    shifted_objective = np.sum(
        np.abs(shifted_y - shifted[0] - shifted[1] * x))
    # Quantize the reference through the actual shifted input; its ulp is part
    # of the supplied data, while the 1e14 location must not erase variation.
    reference = _pairwise_line_oracle(x, shifted_y)
    np.testing.assert_allclose(shifted_objective, reference, rtol=0, atol=0.01)
    assert shifted_objective > 0.0
    assert abs(shifted[1] - base[1]) < 0.02
    assert base_objective > 0.0


def test_independent_signed_extreme_x_and_y_unit_scaling():
    x = np.linspace(-2.0, 3.0, 31)
    y = 4.0 + 2.0 * x + 0.1 * np.sin(np.arange(x.size))
    base = _lad_fit(x, y, True)
    assert base is not None
    base_objective = np.sum(np.abs(y - base[0] - base[1] * x))

    for factor in (1e-300, -1e-300, 1e300, -1e300):
        x_fit = _lad_fit(x * factor, y, True)
        assert x_fit is not None
        x_objective = np.sum(np.abs(y - x_fit[0] - x_fit[1] * (x * factor)))
        np.testing.assert_allclose(x_objective, base_objective, rtol=2e-12, atol=1e-12)

        y_fit = _lad_fit(x, y * factor, True)
        assert y_fit is not None
        y_residual = y * factor - y_fit[0] - y_fit[1] * x
        normalized_objective = np.sum(np.abs(y_residual / abs(factor)))
        np.testing.assert_allclose(
            normalized_objective, base_objective, rtol=2e-12, atol=1e-12)


def test_returned_original_unit_objective_matches_independent_oracle():
    x = np.array([-5.0, -2.0, -1.0, 0.5, 1.0, 4.0, 8.0])
    y = np.array([-9.0, -2.0, -1.5, 2.0, 3.2, 10.0, 14.0])
    fit = _lad_fit(x, y, True)
    assert fit is not None
    restored_objective = np.sum(np.abs(y - fit[0] - fit[1] * x))
    np.testing.assert_allclose(
        restored_objective, _pairwise_line_oracle(x, y), rtol=0, atol=1e-10)
