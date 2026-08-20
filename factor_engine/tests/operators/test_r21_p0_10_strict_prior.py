# -*- coding: utf-8 -*-
"""Regression tests for the strict-prior family (R21-P0-10).

Targets:
- ts_ridge_regression_coeff_prior  (Ridge, fit_lag=1)
- ts_quantile_regression_coeff_prior (Quantile, fit_lag=1)
- ts_mean_reversion_ou_approx_half_life_prior (OU approx, fit_lag=1)

Verifies:
1. Prior-lag invariant (fit_lag=1 excludes current obs)
2. Parameter selection (window/order/q propagation)
3. Family completeness (timing contracts, lane assignments)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.model_timing import get_model_timing_contract, model_timing_contract_is_explicit
from cleaned_operators.model_lane import assign_model_lane

ensure_cleaned_loaded()

# ---------------------------------------------------------------------------
# Helper: synthetic data
# ---------------------------------------------------------------------------

def _panel(n: int = 120, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=["A", "B"])


def _ou_series(n: int = 200, phi: float = 0.92, sigma: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Generate an OU process: x_t = phi * x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal() * sigma
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(x, index=idx, columns=["A"])


# ---------------------------------------------------------------------------
# 1. Prior-lag invariant: fit on t-1 only, not t
# ---------------------------------------------------------------------------

class TestPriorLagInvariant:
    """The *_prior operators must fit strictly on rows <= t-1 and be invariant to
    perturbations of row t (current observation)."""

    def test_ridge_prior_excludes_current_obs(self) -> None:
        """Perturbing y[t] should NOT change the fitted beta (only the residual)."""
        rng = np.random.default_rng(100)
        n, w = 100, 30
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
        y = 2.0 * x + 0.5 + rng.standard_normal((n, 1)) * 0.1

        op = OperatorRegistry.get("ts_ridge_regression_coeff_prior")
        beta_base = op.calculate(y, x, window=w, coefficient_index=1, min_periods=5)

        # Perturb y[t] by +1000
        y_high = y.copy()
        y_high.iloc[-1, 0] += 1000.0
        beta_high = op.calculate(y_high, x, window=w, coefficient_index=1, min_periods=5)

        # Beta must NOT change (fit excludes current row)
        np.testing.assert_allclose(
            beta_base.iloc[-1].to_numpy(),
            beta_high.iloc[-1].to_numpy(),
            rtol=1e-9,
            err_msg="Ridge prior coeff changed when y[t] was perturbed (fit_lag != 1)",
        )

    def test_quantile_prior_excludes_current_obs(self) -> None:
        """Perturbing y[t] should NOT change the fitted quantile beta."""
        rng = np.random.default_rng(101)
        n, w = 100, 30
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
        y = 1.5 * x + 0.3 + rng.standard_normal((n, 1)) * 0.1

        op = OperatorRegistry.get("ts_quantile_regression_coeff_prior")
        beta_base = op.calculate(y, x, window=w, q=0.5, min_periods=5)

        y_high = y.copy()
        y_high.iloc[-1, 0] += 1000.0
        beta_high = op.calculate(y_high, x, window=w, q=0.5, min_periods=5)

        np.testing.assert_allclose(
            beta_base.iloc[-1].to_numpy(),
            beta_high.iloc[-1].to_numpy(),
            rtol=1e-9,
            err_msg="Quantile prior coeff changed when y[t] was perturbed (fit_lag != 1)",
        )

    def test_ou_prior_excludes_current_obs(self) -> None:
        """Perturbing x[t] should NOT change the OU half-life at t."""
        x = _ou_series(n=200, seed=200)
        op = OperatorRegistry.get("ts_mean_reversion_ou_approx_half_life_prior")
        hl_base = op.calculate(x, window=100, min_periods=20)

        x_high = x.copy()
        x_high.iloc[-1, 0] += 1000.0
        hl_high = op.calculate(x_high, window=100, min_periods=20)

        np.testing.assert_allclose(
            hl_base.iloc[-1].to_numpy(),
            hl_high.iloc[-1].to_numpy(),
            rtol=1e-9,
            err_msg="OU prior half-life changed when x[t] was perturbed (fit_lag != 1)",
        )


# ---------------------------------------------------------------------------
# 2. Parameter selection: window/order/q propagation
# ---------------------------------------------------------------------------

class TestParameterSelection:
    """Verify that window, order, q, and other params are correctly propagated."""

    def test_ridge_prior_window_effect(self) -> None:
        """Larger window should produce a different (smoother) beta trajectory."""
        rng = np.random.default_rng(110)
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
        y = 1.0 * x + rng.standard_normal((n, 1)) * 0.5

        op = OperatorRegistry.get("ts_ridge_regression_coeff_prior")
        out_w20 = op.calculate(y, x, window=20, coefficient_index=1, min_periods=5)
        out_w60 = op.calculate(y, x, window=60, coefficient_index=1, min_periods=5)

        # Both should have finite values at the end
        assert np.isfinite(out_w20.iloc[-1, 0])
        assert np.isfinite(out_w60.iloc[-1, 0])
        # They should differ (different windows)
        assert out_w20.iloc[-1, 0] != pytest.approx(out_w60.iloc[-1, 0], abs=1e-6)

    def test_quantile_prior_q_propagation(self) -> None:
        """Different q values should produce different betas (unless data is symmetric)."""
        rng = np.random.default_rng(111)
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
        # Asymmetric noise to make quantiles differ
        noise = rng.standard_normal((n, 1)) * (1 + np.abs(x.to_numpy()))
        y = 1.0 * x + noise

        op = OperatorRegistry.get("ts_quantile_regression_coeff_prior")
        out_q10 = op.calculate(y, x, window=60, q=0.1, min_periods=10)
        out_q90 = op.calculate(y, x, window=60, q=0.9, min_periods=10)

        assert np.isfinite(out_q10.iloc[-1, 0])
        assert np.isfinite(out_q90.iloc[-1, 0])
        assert out_q10.iloc[-1, 0] != pytest.approx(out_q90.iloc[-1, 0], abs=1e-4)

    def test_ou_prior_window_effect(self) -> None:
        """Larger window should yield a more stable half-life estimate."""
        x = _ou_series(n=200, seed=112)
        op = OperatorRegistry.get("ts_mean_reversion_ou_approx_half_life_prior")
        out_w50 = op.calculate(x, window=50, min_periods=20)
        out_w100 = op.calculate(x, window=100, min_periods=20)

        assert np.isfinite(out_w50.iloc[-1, 0])
        assert np.isfinite(out_w100.iloc[-1, 0])


# ---------------------------------------------------------------------------
# 3. Family completeness: timing contracts + lane assignments
# ---------------------------------------------------------------------------

class TestFamilyCompleteness:
    """Every *_prior variant must have an explicit timing contract and be in
    FAST_NATIVE_ALPHA lane."""

    @pytest.mark.parametrize("name", [
        "ts_ridge_regression_coeff_prior",
        "ts_quantile_regression_coeff_prior",
        "ts_mean_reversion_ou_approx_half_life_prior",
    ])
    def test_explicit_timing_contract(self, name: str) -> None:
        assert model_timing_contract_is_explicit(name), (
            f"{name} has no explicit MODEL_TIMING_CONTRACTS entry"
        )
        c = get_model_timing_contract(name)
        assert c.fit_cutoff_offset >= 1, (
            f"{name} fit_cutoff_offset={c.fit_cutoff_offset} < 1 (not strict prior)"
        )

    @pytest.mark.parametrize("name", [
        "ts_ridge_regression_coeff_prior",
        "ts_quantile_regression_coeff_prior",
        "ts_mean_reversion_ou_approx_half_life_prior",
    ])
    def test_fast_native_alpha_lane(self, name: str) -> None:
        lane = assign_model_lane(name)
        assert lane == "FAST_NATIVE_ALPHA", (
            f"{name} lane={lane}, expected FAST_NATIVE_ALPHA"
        )

    @pytest.mark.parametrize("name", [
        "ts_ridge_regression_coeff_prior",
        "ts_quantile_regression_coeff_prior",
        "ts_mean_reversion_ou_approx_half_life_prior",
    ])
    def test_registered_and_not_diagnostic(self, name: str) -> None:
        op = OperatorRegistry.get(name)
        assert op is not None, f"{name} not registered"
        meta = op.metadata
        assert not getattr(meta, "diagnostic_only", False), (
            f"{name} is marked diagnostic_only (should be production alpha)"
        )


# ---------------------------------------------------------------------------
# 4. Determinism: same input -> same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    """All prior operators must be deterministic."""

    @pytest.mark.parametrize("name, args_fn", [
        ("ts_ridge_regression_coeff_prior", lambda: (_panel(seed=200), _panel(seed=201))),
        ("ts_quantile_regression_coeff_prior", lambda: (_panel(seed=202), _panel(seed=203))),
        ("ts_mean_reversion_ou_approx_half_life_prior", lambda: (_ou_series(seed=204),)),
    ])
    def test_deterministic(self, name: str, args_fn) -> None:
        op = OperatorRegistry.get(name)
        args = args_fn()
        out1 = op.calculate(*args)
        out2 = op.calculate(*args)
        pd.testing.assert_frame_equal(out1, out2, check_dtype=False)
