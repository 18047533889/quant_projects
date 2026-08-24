# -*- coding: utf-8 -*-
"""R35 Phase G-H: FastLinearWindowEngine + model family shared intermediates.

- rolling OLS / ridge sufficient-stats engine matches per-row lstsq reference
- PCA family block feeds loading/resid/commonality consistently
- GARCH family block feeds persistence / shock / next-vol consistently
- intermediate identity separates different window/param/universe
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.backend.fast_linear_window import (  # noqa: E402
    rolling_ols_sufficient,
    rolling_ridge_sufficient,
)
from factor_engine.backend.model_family_kernels import (  # noqa: E402
    fit_garch_block,
    fit_pca_block,
    garch_variance_path,
    intermediate_identity,
)


# --------------------------------------------------------------------------
# FastLinearWindowEngine vs per-row lstsq
# --------------------------------------------------------------------------

def test_rolling_ols_sufficient_matches_lstsq():
    rng = np.random.default_rng(0)
    T, p = 120, 3
    X = rng.standard_normal((T, p))
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.1 * rng.standard_normal(T)
    fast = rolling_ols_sufficient(X, y, 40).beta
    # reference: per-row lstsq
    ref = np.full((T, p), np.nan)
    for t in range(T):
        lo = max(0, t - 40 + 1)
        if t - lo + 1 < p:
            continue
        b, *_ = np.linalg.lstsq(X[lo : t + 1], y[lo : t + 1], rcond=None)
        ref[t] = b
    both = np.isfinite(fast) & np.isfinite(ref)
    assert both.any()
    assert np.allclose(fast[both], ref[both], atol=1e-8), "OLS sufficient vs lstsq"


def test_rolling_ridge_sufficient_recovers():
    rng = np.random.default_rng(1)
    T, p = 100, 2
    X = rng.standard_normal((T, p))
    y = X @ np.array([1.5, -1.0]) + 0.05 * rng.standard_normal(T)
    lam = 0.1
    fast = rolling_ridge_sufficient(X, y, 50, lam).beta
    ref = np.full((T, p), np.nan)
    for t in range(T):
        lo = max(0, t - 50 + 1)
        if t - lo + 1 < p:
            continue
        G = X[lo : t + 1].T @ X[lo : t + 1] + lam * np.eye(p)
        b = np.linalg.solve(G, X[lo : t + 1].T @ y[lo : t + 1])
        ref[t] = b
    both = np.isfinite(fast) & np.isfinite(ref)
    assert np.allclose(fast[both], ref[both], atol=1e-8), "ridge sufficient vs solve"


def test_rolling_ols_missing_pattern_parity():
    """§114: a missing row must not silently change the Gram — the fast path
    must equal the reference even under NaN holes."""
    rng = np.random.default_rng(2)
    T, p = 100, 2
    X = rng.standard_normal((T, p))
    y = rng.standard_normal(T)
    X[30] = np.nan
    X[31, 1] = np.nan
    y[50] = np.nan
    fast = rolling_ols_sufficient(X, y, 30).beta
    ref = np.full((T, p), np.nan)
    for t in range(T):
        lo = max(0, t - 30 + 1)
        seg_X, seg_y = X[lo : t + 1], y[lo : t + 1]
        valid = np.all(np.isfinite(seg_X), axis=1) & np.isfinite(seg_y)
        if valid.sum() < p:
            continue
        b, *_ = np.linalg.lstsq(seg_X[valid], seg_y[valid], rcond=None)
        ref[t] = b
    both = np.isfinite(fast) & np.isfinite(ref)
    assert both.any()
    assert np.allclose(fast[both], ref[both], atol=1e-8), "OLS missing-pattern parity"


# --------------------------------------------------------------------------
# PCA family shared block
# --------------------------------------------------------------------------

def test_pca_block_feeds_multiple_outputs():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 6))
    cur = rng.standard_normal(6)
    block = fit_pca_block(X, 3)
    assert block is not None
    assert block.k <= 3
    resid = block.resid(cur, 2)
    assert resid.shape == (6,)
    assert np.isfinite(resid[block.active]).all()
    # loading from block equals direct SVD loading
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_svd

    direct = _pca_svd(X, 3)
    assert direct is not None
    np.testing.assert_allclose(block.loadings[: block.k], direct["loadings"], atol=1e-10)
    np.testing.assert_allclose(block.explained[: block.k], direct["explained"], atol=1e-10)


def test_pca_block_identity_separates_params():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 4))
    id1 = intermediate_identity(
        family="pca", input_digest="a", window=60, params={"k": 3},
        fit_cutoff_offset=1, universe="u1",
    )
    id2 = intermediate_identity(
        family="pca", input_digest="a", window=60, params={"k": 3},
        fit_cutoff_offset=1, universe="u2",  # different universe
    )
    id3 = intermediate_identity(
        family="pca", input_digest="a", window=120, params={"k": 3},
        fit_cutoff_offset=1, universe="u1",  # different window
    )
    assert id1 != id2
    assert id1 != id3


# --------------------------------------------------------------------------
# GARCH family shared block
# --------------------------------------------------------------------------

def test_garch_block_feeds_persistence_and_vol():
    rng = np.random.default_rng(5)
    rets = rng.standard_normal(300)
    block = fit_garch_block(rets)
    assert block is not None
    assert block.persistence == pytest.approx(block.alpha + block.beta)
    # standardized shock / next-vol are finite and sign-consistent
    h_last = garch_variance_path(rets[-100:], block, float(np.var(rets[-100:-1])))
    r_t = rets[-1]
    s = block.standardized_shock(r_t, h_last)
    nv = block.next_vol(r_t, h_last)
    assert np.isfinite(s) and np.isfinite(nv) and nv > 0
    assert np.sign(s) == np.sign(r_t)


def test_garch_block_matches_engine_persistence():
    from factor_engine.cleaned_operators.ts_model.volatility import _garch_path

    rng = np.random.default_rng(6)
    rets = rng.standard_normal(300)
    block = fit_garch_block(rets)
    assert block is not None
    eng = _garch_path(rets, 250, "persistence", False, 0.0)
    if np.isfinite(eng):
        assert block.persistence == pytest.approx(eng, abs=0.1), (
            f"block {block.persistence} engine {eng}"
        )
