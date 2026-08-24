# -*- coding: utf-8 -*-
"""R38 P1-052..056（§21）：FastLinearWindowEngine 真 sliding。

行为探针：
    - sliding vs authoritative reference（逐行 lstsq）parity（§R38_FAST_LINEAR_REFERENCE_PARITY）；
    - 大 T 下 sliding 明显快于 rescan（复杂度回归，§R38_FAST_LINEAR_COMPLEXITY）；
    - missing pattern：joint-finite 滑动等价（5% / block missing）；
    - 近奇异 → lstsq fallback（P1-055，不偷偷变 Ridge）；
    - Ridge intercept 不正则化（P1-056）。
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.backend.fast_linear_window import (
    rolling_ols_reference,
    rolling_ols_sufficient,
    rolling_ridge_sufficient,
    sliding_parity_check,
)


def _data(T: int, p: int, missing: str = "none") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(T, p))
    beta = rng.normal(size=p)
    y = X @ beta + rng.normal(scale=0.1, size=T)
    if missing == "5pct":
        mask = rng.random((T, p)) < 0.05
        X[mask] = np.nan
        ymask = rng.random(T) < 0.05
        y[ymask] = np.nan
    elif missing == "block":
        X[T // 3 : T // 3 + 10, :] = np.nan
        y[T // 2 : T // 2 + 5] = np.nan
    return X, y


@pytest.mark.parametrize("window", [5, 20, 120])
@pytest.mark.parametrize("p", [1, 4])
@pytest.mark.parametrize("missing", ["none", "5pct", "block"])
def test_sliding_parity(window, p, missing):
    X, y = _data(400, p, missing)
    res = sliding_parity_check(X, y, window, rtol=1e-5, atol=1e-7)
    assert res["status"] == "PASS", res
    assert res["match"] is True


def test_sliding_complexity_beats_rescan():
    # 复杂度回归：大 T 下 sliding 比逐行重扫快得多（rescan O(T*window)）。
    import time

    T, p, window = 2000, 4, 120
    X, y = _data(T, p, "none")
    t0 = time.perf_counter()
    sliding_parity_check(X, y, window)
    fast_ms = (time.perf_counter() - t0) * 1000
    # 只测 sliding 本身（reference 单独算）。
    t0 = time.perf_counter()
    rolling_ols_sufficient(X, y, window)
    fast_only_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    rolling_ols_reference(X, y, window)
    ref_ms = (time.perf_counter() - t0) * 1000
    assert fast_ms < 1e6  # 不超时
    assert ref_ms > 0
    # 诚实：sliding-only 必须比 reference 快（rescan 会逐窗口重扫）。
    assert fast_only_ms < ref_ms, f"sliding {fast_only_ms:.1f}ms must beat reference {ref_ms:.1f}ms"


def test_near_singular_uses_lstsq_not_silent_ridge():
    # 重复列 → 近奇异。sliding OLS 必须 fallback lstsq（fallback_mask 有行），
    # 而不是偷偷加 ridge 输出错误。
    rng = np.random.default_rng(7)
    T, window = 300, 30
    x1 = rng.normal(size=T)
    x2 = x1 + 0.0 * rng.normal(size=T)  # 完全共线
    X = np.column_stack([x1, x2])
    y = 2.0 * x1 + rng.normal(scale=0.05, size=T)
    fast = rolling_ols_sufficient(X, y, window)
    ref = rolling_ols_reference(X, y, window)
    assert fast.lstsq_fallback_mask is not None and fast.lstsq_fallback_mask.any()
    # fallback 结果仍与 reference（lstsq）一致（近奇异区段）。
    both = np.isfinite(fast.beta) & np.isfinite(ref.beta)
    assert np.allclose(fast.beta[both], ref.beta[both], atol=1e-4)


def test_ridge_intercept_not_regularized():
    # 无截距模型 + add_intercept：intercept 不应被 lam 拉向 0。
    rng = np.random.default_rng(3)
    T, p, window = 500, 3, 60
    X = rng.normal(size=(T, p))
    Xc = np.concatenate([np.ones((T, 1)), X], axis=1)
    beta_true = np.array([5.0, 1.0, -2.0, 0.5])
    y = Xc @ beta_true + rng.normal(scale=0.1, size=T)
    ols = rolling_ols_sufficient(X, y, window, add_intercept=True)
    ridge = rolling_ridge_sufficient(X, y, window, lam=10.0, add_intercept=True)
    # 大 lam 下拉 slope 明显；intercept 因不正则化应保持接近 OLS（不被 shrink）。
    slope_shrink = abs(ridge.beta[:, 1] - ols.beta[:, 1]).mean()
    intercept_diff = abs(ridge.beta[:, 0] - ols.beta[:, 0]).mean()
    # 末段（稳定窗口）：slope 被 shrink 的量应显著大于 intercept 变化。
    tail = slice(window, T)
    slope_shrink_tail = abs(ridge.beta[tail, 1] - ols.beta[tail, 1]).mean()
    intercept_diff_tail = abs(ridge.beta[tail, 0] - ols.beta[tail, 0]).mean()
    assert slope_shrink_tail > intercept_diff_tail * 3, (
        f"intercept should not be regularized: slope_shrink={slope_shrink_tail:.4f} "
        f"intercept_diff={intercept_diff_tail:.4f}"
    )
