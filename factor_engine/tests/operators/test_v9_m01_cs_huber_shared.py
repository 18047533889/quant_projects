from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.group_ext import (
    CsHuberResid,
    _cs_robust_resid,
    _huber_irls_fit,
)
from factor_engine.cleaned_operators.ts_model._rolling_core import last_fit_status


def _seed8():
    rng = np.random.default_rng(8)
    x = rng.normal(size=20)
    y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=20)
    y[rng.choice(20, 5, replace=False)] += rng.normal(0.0, 10.0, 5)
    return x, y


def _panel(values):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        [values],
        index=[pd.Timestamp("2025-01-02")],
        columns=[f"s{i}" for i in range(values.size)],
    )


def test_fixed_outlier_translation_preserves_fit_and_residuals():
    x, y = _seed8()
    base = _huber_irls_fit(x, y, True)
    shifted = _huber_irls_fit(x + 1e8, y, True)
    assert base is not None and shifted is not None
    np.testing.assert_allclose(shifted[1], base[1], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(
        base[0] + base[1] * x,
        shifted[0] + shifted[1] * (x + 1e8),
        rtol=0,
        atol=3e-7,
    )
    residual = _cs_robust_resid(_panel(y), _panel(x), _huber_irls_fit, True)
    translated = _cs_robust_resid(
        _panel(y), _panel(x + 1e8), _huber_irls_fit, True,
    )
    np.testing.assert_allclose(translated, residual, rtol=0, atol=3e-7)


def test_intercept_translation_and_unit_scaling_equivariance():
    x, y = _seed8()
    base = _huber_irls_fit(x, y, True)
    y_shifted = _huber_irls_fit(x, y + 1e8, True)
    x_scaled = _huber_irls_fit(x * 1e6, y, True)
    y_scaled = _huber_irls_fit(x, y * 1e-6, True)
    assert all(fit is not None for fit in (base, y_shifted, x_scaled, y_scaled))
    np.testing.assert_allclose(y_shifted[1], base[1], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(x_scaled[1] * 1e6, base[1], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(y_scaled, np.asarray(base) * 1e-6, rtol=2e-7, atol=2e-9)


def test_exact_linear_and_no_intercept_contracts():
    x = np.linspace(-3.0, 4.0, 20)
    np.testing.assert_allclose(
        _huber_irls_fit(x, 3.0 + 2.0 * x, True), (3.0, 2.0), atol=1e-12,
    )
    assert last_fit_status()["reason"] == "exact_fit"
    x_positive = np.arange(1.0, 21.0)
    np.testing.assert_allclose(
        _huber_irls_fit(x_positive, 2.0 * x_positive, False),
        (0.0, 2.0),
        atol=1e-12,
    )


def test_missing_rank_failure_and_iteration_cap_fail_closed():
    x, y = _seed8()
    x[-1] = np.nan
    result = CsHuberResid().calculate(_panel(y), _panel(x), add_intercept=True)
    assert np.isnan(result.iloc[0, -1])
    assert np.isfinite(result.iloc[0, :-1]).all()

    constant = np.ones(20)
    rank_failed = CsHuberResid().calculate(
        _panel(np.arange(20.0)), _panel(constant), add_intercept=True,
    )
    assert rank_failed.isna().to_numpy().all()

    x, y = _seed8()
    assert _huber_irls_fit(x, y, True, max_iter=1) is None
    status = last_fit_status()
    assert status["reason"] == "non_converged" and status["iterations"] == 1


def test_exact_inlier_majority_has_no_positive_adaptive_mad_scale():
    x = np.arange(1.0, 31.0)
    y = 3.0 + 2.0 * x
    y[-1] += 500.0
    assert _huber_irls_fit(x, y, True) is None
    status = last_fit_status()
    assert status["reason"] == "scale_degenerate"
    assert status["mad"] <= 64.0 * np.finfo(float).eps * status["residual_extent"]


def test_real_registry_winner_uses_stable_kernel():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    winner = OperatorRegistry.get("cs_huber_resid")
    assert isinstance(winner, CsHuberResid)
    x, y = _seed8()
    base = winner.calculate(_panel(y), _panel(x), add_intercept=True)
    shifted = winner.calculate(_panel(y), _panel(x + 1e8), add_intercept=True)
    np.testing.assert_allclose(shifted, base, rtol=0, atol=3e-7)


def test_default_pandas_batch_bridge_preserves_translation():
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.tests.helpers import InMemorySeriesSource

    x, y = _seed8()
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], [f"s{i}" for i in range(x.size)]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(data={
        "y": pd.Series(y, index=index),
        "x": pd.Series(x, index=index),
        "x_shifted": pd.Series(x + 1e8, index=index),
    })
    huber = make_cleaned_call_factory("cs_huber_resid")
    factors = [
        Factor("base", huber(col("y"), col("x"), add_intercept=True)),
        Factor("shifted", huber(col("y"), col("x_shifted"), add_intercept=True)),
    ]
    engine = FactorEngine(backend=build_backend(), data_source=source, run_mode="research")
    result = engine.run_many(factors)["results"]
    np.testing.assert_allclose(result["shifted"], result["base"], rtol=0, atol=3e-7)
