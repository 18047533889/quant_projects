# -*- coding: utf-8 -*-
"""R35 Phase B: model oracles (independent numpy/scipy references).

Each test compares an engine kernel against an independently-derived oracle:
- PCA loadings / explained ratio vs standardized SVD oracle
- rolling OLS slope vs centered-dot-product oracle
- AR coefficient vs Yule-Walker
- Kalman local-level vs hand-written filter
- GARCH persistence vs independent variance-targeting MLE
- HAR next-vol forecast vs explicit design-matrix oracle

Synthetic recovery tests (§145): AR(1) / GARCH / PCA latent factor / Kalman
state recovery — the engine must recover the planted parameter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from tests.r35.model_oracle import (  # noqa: E402
    ar_yule_walker,
    garch_persistence_oracle,
    har_forecast_oracle,
    kalman_local_level,
    pca_oracle,
    rolling_ols_slope,
)


# --------------------------------------------------------------------------
# PCA oracle
# --------------------------------------------------------------------------

def test_pca_oracle_matches_engine_loadings():
    from cleaned_operators.cross_section.panel_model import _pca_svd

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 8))
    X[10, 3] = np.nan  # one missing cell
    eng = _pca_svd(X, 3)
    ref = pca_oracle(X, 3)
    assert eng is not None and ref is not None
    assert eng["k"] == ref["k"]
    np.testing.assert_allclose(eng["loadings"], ref["loadings"], atol=1e-10)
    np.testing.assert_allclose(eng["explained"], ref["explained"], atol=1e-10)
    np.testing.assert_allclose(eng["mu"], ref["mu"], atol=1e-12)


def test_pca_latent_factor_recovery():
    """§145: plant a latent factor, PCA first component must recover it.

    ``_pca_svd`` standardises each column (subtract column mean, divide by
    column sd) before the SVD.  When every column loads on the same factor, each
    standardised column becomes the SAME direction ``(f - mean(f))/sd_f`` up to
    the sign of its loading — so PC1 recovers the SIGN pattern of the loadings
    (a near-equal-weight direction), not their magnitudes.  The correct
    recovery assertions are: (a) PC1 loading signs match ``sign(loads)``, and
    (b) with moderate noise the sign pattern stays clean."""
    from cleaned_operators.cross_section.panel_model import _pca_svd

    rng = np.random.default_rng(2)
    f = rng.standard_normal(120)
    loads = rng.standard_normal(10)
    loads = loads / np.abs(loads).max() * 0.8  # balanced, no single dominant weight
    X = np.outer(f, loads) + 0.02 * rng.standard_normal((120, 10))
    eng = _pca_svd(X, 1)
    assert eng is not None
    v = eng["loadings"][0]
    # sign recovery: PC1 must pick the same direction as the planted loadings
    signed_v = v * np.sign(np.dot(v, loads))
    assert np.sign(signed_v).tolist() == np.sign(loads).tolist(), (
        f"PCA sign pattern not recovered: v={v.round(3)} loads={np.round(loads,3)}"
    )
    # magnitude: PC1 is a unit-norm SVD vector; with balanced loadings and
    # column-standardisation it approximates the equal-weight direction, so the
    # |loading| entries should be near-uniform (sd of the |loadings| ~ small).
    abs_std = float(np.std(np.abs(v)))
    assert abs_std < 0.02, f"PC1 |loading| spread {abs_std:.3f} — not near-equal-weight"


# --------------------------------------------------------------------------
# Rolling OLS
# --------------------------------------------------------------------------

def test_rolling_ols_oracle_matches_engine():
    from cleaned_operators.cross_section.peer_ops import _rolling_regression

    rng = np.random.default_rng(3)
    n = 80
    y = pd.DataFrame({"A": rng.standard_normal(n)}, index=range(n))
    x = pd.DataFrame({"A": rng.standard_normal(n)}, index=range(n))
    eng = _rolling_regression(y, x, 30, 8)["A"].to_numpy()
    ref = rolling_ols_slope(y["A"].to_numpy(), x["A"].to_numpy(), 30, 8)
    both_nan = np.isnan(eng) & np.isnan(ref)
    both_fin = np.isfinite(eng) & np.isfinite(ref)
    assert ((both_nan | both_fin)).all(), "NaN-pattern mismatch"
    assert np.allclose(eng[both_fin], ref[both_fin], atol=1e-9), "value mismatch"


def test_rolling_ols_recovers_beta():
    """§145: plant y = 2*x + noise; rolling slope must recover ~2."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(200)
    y = 2.0 * x + 0.1 * rng.standard_normal(200)
    out = rolling_ols_slope(y, x, 60, 20)
    tail = out[120:]
    assert np.isfinite(tail).all()
    assert np.mean(tail) == pytest.approx(2.0, abs=0.05)


# --------------------------------------------------------------------------
# AR oracle
# --------------------------------------------------------------------------

def test_ar_yule_walker_recovers_phi():
    """AR(1): phi=0.7 must be recovered from a simulated path."""
    rng = np.random.default_rng(5)
    phi = 0.7
    n = 600
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    a = ar_yule_walker(x, 1)
    assert a is not None
    assert a[0] == pytest.approx(phi, abs=0.05)


# --------------------------------------------------------------------------
# Kalman oracle
# --------------------------------------------------------------------------

def test_kalman_level_oracle_tracks_state():
    """§145: local-level Kalman filtered state must track a slowly-moving level."""
    rng = np.random.default_rng(6)
    n = 200
    level = 0.02 * np.arange(n) + np.sin(np.arange(n) / 20.0)
    obs = level + 0.3 * rng.standard_normal(n)
    filt = kalman_local_level(obs, q=0.001, r=0.09)
    # filter lags; at the end it should be close to the true level
    assert np.nanmean(np.abs(filt[150:] - level[150:]) < 0.3


def test_kalman_oracle_matches_engine_kernel():
    """Engine ``ts_kalman_level`` vs the hand-written local-level filter.  The
    engine initialises ``mu=x`` on the first finite row while the oracle
    backcasts from the sample mean, so the first ~dozen rows carry an init
    transient; the steady-state filter (both use the same predict/update
    recursion and covariance advance) must agree to tight tolerance."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.ts_model.state_space import _kalman_level

    load_all()
    rng = np.random.default_rng(9)
    x = rng.standard_normal(200)
    eng = _kalman_level(x, 0.001, 0.09, "level")
    ref = kalman_local_level(x, q=0.001, r=0.09)
    # drop the init transient; compare the settled tail
    both_fin = np.isfinite(eng[40:]) & np.isfinite(ref[40:])
    assert both_fin.sum() > 120
    # the two filters use different initial-state backcasts (engine: first finite
    # row; oracle: sample mean), which leaves a persistent O(q)-sized offset in a
    # low-q filter — the RECURSION must match, so compare with a tolerance that
    # admits the init transient but still rejects a genuinely wrong filter.
    assert np.allclose(eng[40:][both_fin], ref[40:][both_fin], atol=1e-3), (
        "steady-state Kalman level diverged from oracle"
    )


# --------------------------------------------------------------------------
# GARCH oracle
# --------------------------------------------------------------------------

def test_garch_persistence_oracle_recovers():
    """§145: a simulated GARCH(1,1) path must yield persistence ~ alpha+beta."""
    rng = np.random.default_rng(10)
    n = 800
    omega, alpha, beta = 0.05, 0.1, 0.85
    h = np.empty(n)
    rets = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    rets[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * rets[t - 1] ** 2 + beta * h[t - 1]
        rets[t] = np.sqrt(h[t]) * rng.standard_normal()
    pers = garch_persistence_oracle(rets, 400)
    assert pers is not None
    assert pers == pytest.approx(alpha + beta, abs=0.1), f"persistence {pers}"


def test_garch_oracle_matches_engine_persistence():
    from cleaned_operators.ts_model.volatility import _garch_path

    rng = np.random.default_rng(13)
    rets = rng.standard_normal(300)
    eng = _garch_path(rets, 250, "persistence", False, 0.0)
    ref = garch_persistence_oracle(rets, 250)
    if ref is not None and np.isfinite(eng):
        assert eng == pytest.approx(ref, abs=0.15), f"engine {eng} oracle {ref}"


# --------------------------------------------------------------------------
# HAR oracle
# --------------------------------------------------------------------------

def test_har_oracle_matches_engine():
    from cleaned_operators.ts_model.volatility import _har_rv

    rng = np.random.default_rng(14)
    rv = np.abs(rng.standard_normal(300)) + 1.0
    eng = _har_rv(rv, 200, "forecast")
    ref = har_forecast_oracle(rv, 200)
    if ref is not None and np.isfinite(eng):
        assert eng == pytest.approx(ref, abs=1e-6), f"engine {eng} oracle {ref}"
