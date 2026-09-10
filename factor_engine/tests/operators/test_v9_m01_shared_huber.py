import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.regression_models import (
    TsHuberRegressionInSampleResid,
    TsHuberRegressionPredictiveResid,
    _huber_fit,
    _regression_resid,
)
from factor_engine.cleaned_operators.ts_model._rolling_core import (
    _returned_fit_backward_error,
    huber_fit,
    last_fit_status,
    ridge_fit,
)


def _seed8():
    rng = np.random.default_rng(8)
    x = rng.normal(size=20)
    y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=20)
    y[rng.choice(20, 5, replace=False)] += rng.normal(0.0, 10.0, 5)
    return np.c_[np.ones(20), x], y


def test_seed8_shared_kernel_converges_with_small_final_score():
    x, y = _seed8()
    beta = _huber_fit(x, y)
    status = last_fit_status()
    assert beta is not None
    assert status["converged"] and status["reason"] == "converged"
    assert status["normalized_score"] <= 1e-6
    assert status["iterations"] > 0
    np.testing.assert_allclose(beta, huber_fit(x, y), rtol=0, atol=0)


def test_one_iteration_cap_fails_closed_and_refreshes_status():
    x, y = _seed8()
    assert huber_fit(x, y) is not None
    assert huber_fit(x, y, iterations=1) is None
    status = last_fit_status()
    assert status["reason"] == "non_converged"
    assert status["iterations"] == 1
    assert status["normalized_score"] > 1e-6


def test_x_and_y_large_translation_equivariance():
    x, y = _seed8()
    b = huber_fit(x, y)
    by = huber_fit(x, y + 1e8)
    xx = x.copy()
    xx[:, 1] += 1e8
    bx = huber_fit(xx, y)
    assert b is not None and by is not None and bx is not None
    np.testing.assert_allclose(by[1], b[1], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(x @ b + 1e8, x @ by, rtol=0, atol=3e-7)
    np.testing.assert_allclose(x @ b, xx @ bx, rtol=0, atol=3e-7)


def test_unit_scaling_preserves_predictions_and_coefficients():
    x, y = _seed8()
    b = huber_fit(x, y)
    scaled_x = x.copy()
    scaled_x[:, 1] *= 1e6
    bx = huber_fit(scaled_x, y)
    by = huber_fit(x, y * 1e-6)
    assert b is not None and bx is not None and by is not None
    np.testing.assert_allclose(scaled_x @ bx, x @ b, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(by, b * 1e-6, rtol=2e-7, atol=2e-9)


def test_exact_zero_target_is_exact_fit_not_singular():
    x = np.c_[np.ones(12), np.linspace(-2, 3, 12)]
    beta = huber_fit(x, np.zeros(12))
    np.testing.assert_array_equal(beta, np.zeros(2))
    status = last_fit_status()
    assert status["converged"] and status["reason"] == "exact_fit"
    assert status["scale"] == 0.0


def test_rank_shape_and_parameter_failures_are_explicit():
    assert huber_fit(np.ones((5, 2)), np.arange(5.0)) is None
    assert last_fit_status()["reason"] == "singular"
    assert huber_fit(np.ones((2, 3)), np.ones(2)) is None
    assert last_fit_status()["reason"] == "insufficient_sample"
    assert huber_fit(np.ones((3, 1)), np.ones((3, 1))) is None
    assert last_fit_status()["reason"] == "invalid_params"
    assert huber_fit(np.ones((3, 1)), np.ones(3), delta=0) is None
    assert last_fit_status()["reason"] == "invalid_params"


def test_public_in_sample_predictive_boundary_and_prefix_invariance():
    idx = pd.RangeIndex(30)
    x = pd.DataFrame({"a": np.linspace(-1, 1, 30)}, index=idx)
    y = pd.DataFrame({"a": 2 + 3 * x["a"] + 0.01 * np.sin(np.arange(30))}, index=idx)
    y.iloc[-1, 0] += 25
    ins = TsHuberRegressionInSampleResid().calculate(y, x, window=20, min_periods=5)
    pred = TsHuberRegressionPredictiveResid().calculate(y, x, window=20, min_periods=5)
    assert np.isfinite(ins.iloc[-1, 0]) and np.isfinite(pred.iloc[-1, 0])
    assert abs(pred.iloc[-1, 0]) > abs(ins.iloc[-1, 0])
    prefix = TsHuberRegressionPredictiveResid().calculate(
        y.iloc[:-1], x.iloc[:-1], window=20, min_periods=5)
    np.testing.assert_allclose(prefix.to_numpy(), pred.iloc[:-1].to_numpy(),
                               equal_nan=True, rtol=0, atol=0)


def test_predictive_failure_becomes_nan_not_half_fit():
    x, y = _seed8()
    got = _regression_resid(y, x[:, 1], method="huber", min_periods=3, predictive=True)
    assert np.isfinite(got)
    assert huber_fit(x, y, iterations=1) is None


def test_context_local_status_thread_and_async_isolation():
    x, y = _seed8()
    import threading
    barrier = threading.Barrier(2)
    def good():
        huber_fit(x, y)
        barrier.wait()
        return last_fit_status()["reason"]
    def bad():
        huber_fit(x, y, iterations=1)
        barrier.wait()
        return last_fit_status()["reason"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(good), pool.submit(bad))
        assert {future.result() for future in futures} == {"converged", "non_converged"}

    async def task(max_iter):
        await asyncio.sleep(0)
        huber_fit(x, y, iterations=max_iter)
        await asyncio.sleep(0)
        return last_fit_status()["reason"]
    async def run():
        return await asyncio.gather(task(100), task(1))
    assert asyncio.run(run()) == ["converged", "non_converged"]


def test_ridge_stable_solve_regression_unchanged():
    x = np.linspace(-2, 2, 30)
    design = np.c_[np.ones(30), x]
    y = 4 + 2.5 * x
    beta = ridge_fit(design, y, 0.1, has_intercept=True)
    assert beta is not None
    assert abs(beta[0] - 4.0) < 1e-12
    expected_slope = np.dot(x, y - y.mean()) / (np.dot(x, x) + 0.1)
    np.testing.assert_allclose(beta[1], expected_slope, rtol=1e-12, atol=1e-12)


def test_final_solution_matches_independent_fixed_scale_convex_reference():
    from scipy.optimize import minimize

    x, y = _seed8()
    beta = huber_fit(x, y)
    status = last_fit_status()
    assert beta is not None and status["reason"] == "converged"
    scale = status["scale"]
    delta = 1.345

    def objective(candidate):
        u = (y - x @ candidate) / scale
        absolute = np.abs(u)
        loss = np.where(absolute <= delta, 0.5 * u * u,
                        delta * absolute - 0.5 * delta * delta)
        return float(np.sum(loss))

    reference = minimize(objective, beta + np.array([0.05, -0.05]), method="BFGS",
                         options={"gtol": 1e-10, "maxiter": 1000})
    assert objective(beta) <= reference.fun + 1e-8
    np.testing.assert_allclose(beta, reference.x, rtol=0, atol=2e-6)


def test_solver_exception_replaces_prior_success_status(monkeypatch):
    x, y = _seed8()
    assert huber_fit(x, y) is not None

    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(np.linalg, "lstsq", broken)
    assert huber_fit(x, y) is None
    status = last_fit_status()
    assert not status["converged"]
    assert status["reason"] == "singular"


def test_extreme_finite_units_do_not_underflow_or_overflow_intermediates():
    base = np.linspace(-2.0, 3.0, 40)
    design = np.c_[np.ones(40), base * 1e-300]
    target = (4.0 + 2.0 * base) * 1e-300
    beta = huber_fit(design, target)
    assert beta is not None
    prediction = design @ beta
    np.testing.assert_allclose(prediction, target, rtol=2e-14, atol=1e-315)
    assert last_fit_status()["reason"] == "exact_fit"

    design2 = np.c_[np.ones(40), base * 1e200]
    target2 = (4.0 + 2.0 * base) * 1e200
    beta2 = huber_fit(design2, target2)
    assert beta2 is not None
    np.testing.assert_allclose(design2 @ beta2, target2, rtol=2e-14, atol=0)


def test_iteration_budget_requires_strict_integer():
    x, y = _seed8()
    assert huber_fit(x, y, iterations=1.5) is None
    assert last_fit_status()["reason"] == "invalid_params"
    assert huber_fit(x, y, iterations=True) is None
    assert last_fit_status()["reason"] == "invalid_params"


def test_exact_audit_seed8_counterexample_reaches_fixed_point():
    x = np.array([-1.738266398496882, -1.3366427931811324, -1.361106708564987, -0.35161713127840977, -2.3125815796967033, -0.18889719608460778, -0.957229228096346, 0.8936001849299788, 0.956847237579234, 1.3922582291390866, 0.7674701130947078, -0.053029778757267734, 0.8597939889439096, 1.5054811563838433, -0.6535945334170485, 0.610351145830187, -0.042673827710852374, 1.4400167254152394, -0.8368950200968434, -0.3015466095655266])
    y = np.array([-2.404065078359876, 24.53968391433355, 27.621396104386907, 0.3687967838906577, -3.648862699296801, 0.5742560379904721, -0.9455187885931506, 2.8309947108756015, -2.1523456186334133, 4.095009771877546, 2.362651891531702, 0.44566687084800705, 2.703193079120851, -23.665012278599768, -0.410909261054745, 2.530957415841508, 0.44278623290795927, 3.7074870674082034, -1.1668142032491577, 0.14987022950737705])
    design = np.c_[np.ones(x.size), x]
    beta = huber_fit(design, y)
    assert beta is not None
    np.testing.assert_allclose(beta, [0.8917680642, 1.8960249151], rtol=0, atol=1e-6)
    status = last_fit_status()
    assert status["iterations"] > 5
    assert status["normalized_score"] <= 1e-6


def test_large_target_translation_does_not_create_false_exact_fit():
    x = np.linspace(-3.0, 2.0, 40)
    design = np.c_[np.ones(x.size), x]
    residual_pattern = 0.25 * np.sin(np.arange(x.size))
    y = 1.0 + 2.0 * x + residual_pattern
    base = huber_fit(design, y)
    assert base is not None
    shifted = huber_fit(design, y + 1e14)
    shifted_status = last_fit_status()
    assert shifted is None
    assert shifted_status["reason"] == "non_converged"
    assert shifted_status["reason"] != "exact_fit"
    assert shifted_status["normalized_score"] > 1e-6


def test_numpy_integer_iteration_budget_is_accepted():
    x, y = _seed8()
    assert huber_fit(x, y, iterations=np.int64(100)) is not None


def test_reported_diagnostics_are_recomputed_from_returned_original_coefficients():
    x, y = _seed8()
    beta = huber_fit(x, y)
    status = last_fit_status()
    assert beta is not None
    residual = y - x @ beta
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if scale == 0.0:
        centered = residual - residual.mean()
        magnitude = np.max(np.abs(centered))
        scale = 0.0 if magnitude == 0.0 else magnitude * np.linalg.norm(
            centered / magnitude) / np.sqrt(residual.size)
    psi = np.clip(residual / scale, -1.345, 1.345)
    normalized_x = x.copy()
    normalized_x[:, 1] = (
        x[:, 1] / np.max(np.abs(x[:, 1]))
        - np.mean(x[:, 1] / np.max(np.abs(x[:, 1]))))
    normalized_x[:, 1] /= np.sqrt(np.mean(normalized_x[:, 1] ** 2))
    col_norm = np.sqrt(np.mean(normalized_x ** 2, axis=0))
    score = np.max(np.abs(normalized_x.T @ psi / residual.size) / col_norm)
    np.testing.assert_allclose(status["scale"], scale, rtol=0, atol=1e-15)
    np.testing.assert_allclose(status["normalized_score"], score, rtol=0, atol=1e-14)


def test_original_unit_exact_gate_rejects_large_offset_cancellation_error():
    centered_x = np.linspace(-3.0, 3.0, 41)
    shifted_x = centered_x + 1e14
    design = np.c_[np.ones(centered_x.size), shifted_x]
    target = 3.0 + 2.0 * centered_x
    inadequate_beta = np.array([3.0 - 2.0e14, 2.0])
    response_max = float(np.max(np.abs(target)))
    normalized = target / response_max
    spread = float(np.max(np.abs(normalized - normalized.mean())))
    error = _returned_fit_backward_error(
        design, target, inadequate_beta, response_max, spread)
    assert error > 64 * np.finfo(float).eps
