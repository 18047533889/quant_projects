from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_exact_inlier_majority_is_reported_as_exact_consensus():
    x = np.arange(1.0, 31.0)
    y = 3.0 + 2.0 * x
    y[-1] += 500.0
    beta = _huber_irls_fit(x, y, True)
    assert beta == pytest.approx((3.0, 2.0), abs=1e-12)
    status = last_fit_status()
    assert status["converged"] and status["reason"] == "exact_consensus"


def test_rank_deficient_exact_majority_remains_scale_degenerate():
    x = np.r_[np.zeros(21), np.arange(1.0, 10.0)]
    y = np.r_[np.zeros(21), 100.0 * np.arange(1.0, 10.0) + np.linspace(0.0, 3.0, 9)]
    assert _huber_irls_fit(x, y, True) is None
    status = last_fit_status()
    assert not status["converged"] and status["reason"] == "scale_degenerate"


def test_high_leverage_minority_is_not_certified_as_exact_consensus():
    inlier_x = np.arange(21.0)
    x = np.r_[inlier_x, np.full(10, 100.0)]
    y = np.r_[3.0 + 2.0 * inlier_x, np.full(10, -1e6)]
    _huber_irls_fit(x, y, True)
    # The high-leverage minority violates the bounded-subgradient certificate;
    # it must never be relabelled as a zero-scale exact-consensus solution.
    assert last_fit_status()["reason"] != "exact_consensus"


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


@pytest.mark.parametrize("isolated_budget", [False, True], ids=["default-broker", "bounded-fixture"])
def test_default_pandas_batch_bridge_preserves_translation(monkeypatch, isolated_budget):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    from factor_engine.tests.helpers import InMemorySeriesSource

    if isolated_budget:
        # Keep a deterministic mathematical/translation lane alongside the
        # unmodified default-broker integration lane.
        monkeypatch.setattr(AdaptiveBatchScheduler, "_dynamic_wave_budget", lambda _self: 64 * 1024 * 1024)

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
    expected = _cs_robust_resid(_panel(y), _panel(x), _huber_irls_fit, True).stack()
    assert result["base"].index.equals(expected.index)
    assert np.isfinite(result["base"].to_numpy()).all()
    assert np.isfinite(result["shifted"].to_numpy()).all()
    np.testing.assert_allclose(result["base"], expected, rtol=0, atol=3e-7)
    np.testing.assert_allclose(result["shifted"], result["base"], rtol=0, atol=3e-7)
