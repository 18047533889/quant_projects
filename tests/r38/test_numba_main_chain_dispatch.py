# -*- coding: utf-8 -*-
"""R38 P0-057/058/059（§22/§37）：Numba 主链端到端 dispatch。

行为探针：
    - ``_kalman_level(vals, q, r, "level")``（算子主链 per-column kernel 路径）
      **实际** dispatch 到 certified Numba kernel（``numba_dispatch_stats()``
      计数增加，§R38_NUMBA_END_TO_END_DISPATCH）；
    - numba 结果与独立 reference（``kalman_level_reference``）逐值 parity
      （§R38_NUMBA_END_TO_END_PARITY）；
    - 准入 gate：只有逐算子 parity 验证过的组合 dispatch（``out_stat=="level"``）；
      ``innovation_z`` / trend / beta 走 reference（与 kernel 语义漂移，honest）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.ts_model import state_space as ss


def _reset_counter():
    ss._NUMBA_DISPATCH_COUNTER.clear()


def test_kalman_level_dispatches_to_numba_main_chain():
    _reset_counter()
    rng = np.random.default_rng(0)
    vals = rng.normal(size=500)
    vals[40:50] = np.nan
    vals[250] = np.nan
    out = ss._kalman_level(vals.copy(), 1e-4, 1.0, "level")
    assert out is not None and len(out) == 500
    # 主链 kernel 路径 dispatch 到 Numba。
    stats = ss.numba_dispatch_stats()
    assert stats.get("kalman_level", 0) > 0, "kalman_level 必须 dispatch 到 Numba kernel"


def test_kalman_level_numba_matches_reference():
    from factor_engine.backend.numba_kernels.kalman import kalman_level_reference

    rng = np.random.default_rng(0)
    vals = rng.normal(size=500)
    vals[40:50] = np.nan
    vals[250] = np.nan
    ref = ss._kalman_level(vals.copy(), 1e-4, 1.0, "level")
    ref2 = kalman_level_reference(vals.copy(), 1e-4, 1.0)
    both = np.isfinite(ref) & np.isfinite(ref2)
    assert (np.isnan(ref) == np.isnan(ref2)).all()
    assert np.allclose(ref[both], ref2[both], atol=1e-12)


def test_innovation_z_uses_reference_not_kernel():
    # ``out_stat != "level"`` → 保持 reference（kernel 只认证 level 输出）。
    _reset_counter()
    rng = np.random.default_rng(1)
    vals = rng.normal(size=200)
    out = ss._kalman_level(vals.copy(), 1e-4, 1.0, "innovation_z")
    assert out is not None and len(out) == 200
    assert ss.numba_dispatch_stats().get("kalman_level", 0) == 0


def test_trend_beta_not_dispatched_due_to_parity_drift():
    # trend/beta kernel 与算子 reference 存在 NaN 语义漂移（hostile fixture 实测
    # max diff>0）→ 准入 gate 保持 reference，不 dispatch（honest）。
    _reset_counter()
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    x[100:105] = np.nan
    ss._kalman_trend_slope(x.copy(), 1e-5, 1e-5, 1.0)
    y = rng.normal(size=400)
    xx = rng.normal(size=400)
    y[300] = np.nan
    ss._kalman_beta(y.copy(), xx.copy(), 1e-3, 1.0, "beta")
    stats = ss.numba_dispatch_stats()
    assert stats.get("kalman_trend", 0) == 0
    assert stats.get("kalman_beta", 0) == 0


def test_numba_parity_gate_returns_kernel_for_float64():
    from factor_engine.cleaned_operators.ts_model.state_space import _numba_kernel

    assert _numba_kernel("kalman_level") is not None  # float64 认证 kernel 可用
