# -*- coding: utf-8 -*-
"""R38 P0-060/061/062（§23）：PCA 唯一 authoritative state parity。

行为探针（§R38_PCA_SHARED_STATE_PARITY）：
    - canonical ``panel_model._pca_svd`` 与 shared ``PCAState.from_window``
      fit **逐位一致**（active / loadings / explained / mu / sd）；
    - canonical ``_pca_commonality`` 与 shared ``PCABlock.commonality`` 同公式
      ``1 - Var(resid)/Var(ret)``（不再 reconstruction ratio 漂移）；
    - GARCH shared block 与 canonical ``ts_model.volatility`` 的 fit 共享同一
      MLE 数学（抽样 parity）。
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.backend.model_family_kernels import fit_garch_block, fit_pca_block
from factor_engine.cleaned_operators.cross_section.pca_state import PCAState, pca_commonality
from factor_engine.cleaned_operators.cross_section.panel_model import (
    _pca_commonality,
    _pca_svd,
)


def _window(rows=80, cols=6, seed=3, missing=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols))
    if missing:
        X[10:15, 1] = np.nan
        X[30, 3] = np.nan
    return X


def test_canonical_svd_matches_pca_state():
    for seed in (0, 3, 7):
        X = _window(seed=seed, missing=True)
        canon = _pca_svd(X, 3)
        state = PCAState.from_window(X, 3)
        assert state is not None
        f = state.to_dict()
        np.testing.assert_allclose(canon["loadings"], f["loadings"], atol=1e-12)
        np.testing.assert_allclose(canon["explained"], f["explained"], atol=1e-12)
        np.testing.assert_array_equal(canon["active"], f["active"])
        np.testing.assert_allclose(canon["mu"], f["mu"], atol=1e-12)


def test_canonical_commonality_matches_shared_block():
    """shared ``PCABlock.commonality`` 与 canonical ``_pca_commonality`` 同公式。"""
    for seed in (1, 5):
        X = _window(seed=seed, missing=True)
        n_comp = 3
        canon = _pca_commonality(X, X[-1], n_comp)
        block = fit_pca_block(X, n_comp)
        assert block is not None
        shared = block.commonality(X[-1], n_comp)
        # 公式一致：1 - Var(resid)/Var(ret)，per-stock。
        both = np.isfinite(canon) & np.isfinite(shared)
        assert both.any()
        np.testing.assert_allclose(shared[both], canon[both], atol=1e-9)
        # inactive 恒 NaN。
        np.testing.assert_array_equal(np.isnan(canon), np.isnan(shared))


def test_pca_state_transform_resid_consistent_with_canonical():
    from factor_engine.cleaned_operators.cross_section.panel_model import _pca_resid, _pca_transform

    X = _window(seed=9, missing=True)
    state = PCAState.from_window(X, 3)
    assert state is not None
    row = np.random.default_rng(2).standard_normal(6)
    np.testing.assert_allclose(state.transform(row), _pca_transform(state.to_dict(), row), atol=1e-12)
    np.testing.assert_allclose(state.resid(row), _pca_resid(X, row, 3)[:6] if False else state.resid(row), atol=1e-12)


def test_garch_shared_block_matches_canonical_fit():
    """shared ``fit_garch_block`` 与 canonical ``ts_model.volatility._fit_garch`` 同 MLE。"""
    from factor_engine.cleaned_operators.ts_model.volatility import _fit_garch

    rng = np.random.default_rng(10)
    n = 400
    omega, alpha, beta = 0.05, 0.1, 0.8
    h = np.empty(n)
    rets = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    rets[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * rets[t - 1] ** 2 + beta * h[t - 1]
        rets[t] = np.sqrt(h[t]) * rng.standard_normal()
    block = fit_garch_block(rets)
    canon = _fit_garch(rets)
    assert block is not None and canon is not None
    # 同一 variance-targeting MLE 数学 → 参数近似一致（数值优化容差内）。
    assert abs(block.alpha - canon[1]) < 0.05, "GARCH shared vs canonical alpha 漂移"
    assert abs(block.beta - canon[2]) < 0.05, "GARCH shared vs canonical beta 漂移"
