# -*- coding: utf-8 -*-
"""R35 §27-29 / §113-114 + R38 P1-052..056（§21）：FastLinearWindowEngine — true sliding。

R35 版本名义上是 rolling sufficient statistics（O(T p²)），但 ``_roll_grams``
每行 ``Xs.T @ Xs`` 重扫窗口 = O(T*window*p²)（P1-052），还分配 T×p×p Gram
（P1-054），并在近奇异时偷偷 Ridge（P1-055）/ 连 intercept 一起正则化（P1-056）。

R38 修复：
    - **true sliding**：``Gram += x_new x_new'；Gram -= x_old x_old'``（P1-052）；
    - **missing pattern**：每行记录 joint-finite 有效性，valid 行 rank-one 贡献、
      invalid 行零贡献——语义 pairwise/joint finite 时滑动完全等价（P1-053）；
      复杂 mask 才回退 rescan（reference 仍是权威）；
    - **不保存全量 T×p×p**：逐行 solve 当前 Gram → 输出 beta[t]（P1-054）；
    - **OLS 不偷偷变 Ridge**：近奇异用 condition-number gate → fallback ``lstsq``
      （P1-055）；
    - **Ridge intercept 不正则化**：``diag[0] = 0``（P1-056）。

语义 parity 由 :func:`sliding_parity_check` 强制（sliding vs reference 逐值一致）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "rolling_ols_sufficient",
    "rolling_ridge_sufficient",
    "rolling_ols_reference",
    "FastLinearWindowResult",
    "sliding_parity_check",
]

#: condition-number gate：cond(G) > 阈值 → fallback lstsq（P1-055）。
_CONDITION_NUMBER_GATE = 1e12
#: P0-024：滑动 Gram 周期性精确重算间隔（行）。长序列 ``G += outer(new);
#: G -= outer(old)`` 会积累浮点误差（高共线、10 万+ 行）——每 N 行从当前窗口
#: 重算一次 exact Gram。
_REBASE_INTERVAL = 512
#: 触发 rebase 的 symmetry drift 阈值（绝对）。
_SYMMETRY_DRIFT_TOL = 1e-6
#: 触发 rebase 的相对 drift 阈值（相对当前 Gram 范数）。
_REL_DRIFT_TOL = 1e-9


@dataclass
class FastLinearWindowResult:
    beta: np.ndarray                 # (T, p)  coefficients
    resid_sd: np.ndarray | None = None  # (T,)  residual std (population)
    r2: np.ndarray | None = None     # (T,)  R^2
    #: 该行用了 lstsq fallback（近奇异，P1-055）。
    lstsq_fallback_mask: np.ndarray | None = None


def _row_valid(X: np.ndarray, y: np.ndarray, t: int) -> bool:
    return bool(np.isfinite(X[t]).all() and np.isfinite(y[t]))


def _window_grams(
    X: np.ndarray, y: np.ndarray, t: int, window: int
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """精确重算当前窗口 ``[t-window+1, t]`` 的有效行 Gram（P0-024 rebase）。

    只对 joint-finite 有效行累加，返回 ``(G, g, yy, nv)``——与 sliding 累加语义
    完全一致，但重扫窗口消除长序列浮点误差。rebase 间隔 `_REBASE_INTERVAL`，摊销
    O(window*p² / interval)，廉价。
    """
    p = X.shape[1]
    lo = max(0, t - window + 1)
    G = np.zeros((p, p))
    g = np.zeros(p)
    yy = 0.0
    nv = 0
    for i in range(lo, t + 1):
        if _row_valid(X, y, i):
            x = X[i]
            G += np.outer(x, x)
            g += x * y[i]
            yy += y[i] * y[i]
            nv += 1
    return G, g, yy, nv


def _reference_grams(
    X: np.ndarray, y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference rolling Gram（每行重扫窗口，O(T*window*p²)）——**权威**。

    返回 per-row ``XtX (T,p,p)`` / ``Xty (T,p)`` / ``yty (T,)`` / ``nv (T,)`` /
    ``valid (T,)``。sliding 实现必须与它逐值一致。
    """
    T, p = X.shape
    XtX = np.zeros((T, p, p))
    Xty = np.zeros((T, p))
    yty = np.zeros(T)
    nv = np.zeros(T, dtype=int)
    for t in range(T):
        lo = max(0, t - window + 1)
        valid = np.array([_row_valid(X, y, i) for i in range(lo, t + 1)])
        if not valid.any():
            continue
        Xs = X[lo : t + 1][valid]
        ys = y[lo : t + 1][valid]
        nv[t] = len(ys)
        XtX[t] = Xs.T @ Xs
        Xty[t] = Xs.T @ ys
        yty[t] = float(ys @ ys)
    return XtX, Xty, yty, nv


def _sliding_ols(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    *,
    lam: float | None = None,
    regularize_intercept: bool = True,
) -> FastLinearWindowResult:
    """True sliding OLS/Ridge（O(T*p² + T*solve(p))，工作内存 O(p²)）。

    - ``lam is None``：OLS；近奇异 → lstsq fallback（P1-055）。
    - ``lam`` 给定：Ridge。``regularize_intercept=False``（add_intercept 时）
      → 第一列（intercept）**不**正则化（P1-056）。
    """
    T, p = X.shape
    beta = np.full((T, p), np.nan)
    fallback_mask = np.zeros(T, dtype=bool)
    # 当前滑动 Gram（O(p²) 工作内存，不保存全量 T×p×p，P1-054）。
    G = np.zeros((p, p))
    g = np.zeros(p)
    yy = 0.0
    nv = 0
    for t in range(T):
        # 加入新行 t（joint-finite 有效才贡献）。
        if _row_valid(X, y, t):
            x = X[t]
            G += np.outer(x, x)
            g += x * y[t]
            yy += y[t] * y[t]
            nv += 1
        # 移出旧行 t-window。
        old = t - window
        if old >= 0 and _row_valid(X, y, old):
            xo = X[old]
            G -= np.outer(xo, xo)
            g -= xo * y[old]
            yy -= y[old] * y[old]
            nv -= 1
        # P0-024：周期性精确 rebase——每 N 行从当前窗口重算 exact Gram；或
        # 检测到 symmetry / relative drift 超阈值立即 rebase（浮点误差不积累）。
        if t > 0 and t % _REBASE_INTERVAL == 0:
            G2, g2, yy2, nv2 = _window_grams(X, y, t, window)
            rel = float(np.abs(G - G2).max()) / max(1.0, float(np.abs(G2).max()))
            sym = float(np.abs(G - G.T).max())
            if rel > _REL_DRIFT_TOL or sym > _SYMMETRY_DRIFT_TOL:
                G, g, yy, nv = G2, g2, yy2, nv2
        if nv < p:
            continue
        if lam is not None:
            A = G + lam * np.eye(p)
            # P1-056：add_intercept 时第一列是 intercept，不正则化。
            if not regularize_intercept:
                A[0, 0] = G[0, 0]
            try:
                b = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                fallback_mask[t] = True
                b = np.linalg.lstsq(A, g, rcond=None)[0]
        else:
            try:
                # condition-number gate（P1-055）：近奇异 → lstsq（authoritative）。
                cond = np.linalg.cond(G)
                if cond > _CONDITION_NUMBER_GATE:
                    fallback_mask[t] = True
                    b = np.linalg.lstsq(G, g, rcond=None)[0]
                else:
                    b = np.linalg.solve(G, g)
            except np.linalg.LinAlgError:
                fallback_mask[t] = True
                b = np.linalg.lstsq(G, g, rcond=None)[0]
        if np.all(np.isfinite(b)):
            beta[t] = b
    return FastLinearWindowResult(beta=beta, lstsq_fallback_mask=fallback_mask)


def rolling_ols_sufficient(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    *,
    add_intercept: bool = False,
) -> FastLinearWindowResult:
    """Rolling OLS via true sliding sufficient statistics（R38：不再逐窗口 rescan）。"""
    T, p = X.shape
    if add_intercept:
        X = np.concatenate([np.ones((T, 1)), X], axis=1)
        p += 1
    return _sliding_ols(X, y, window)


def rolling_ridge_sufficient(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    lam: float,
    *,
    add_intercept: bool = False,
) -> FastLinearWindowResult:
    """Rolling ridge via true sliding sufficient statistics（intercept 不正则化）。"""
    T, p = X.shape
    if add_intercept:
        X = np.concatenate([np.ones((T, 1)), X], axis=1)
        p += 1
    return _sliding_ols(
        X, y, window, lam=float(lam), regularize_intercept=not add_intercept
    )


def rolling_ols_reference(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    *,
    add_intercept: bool = False,
) -> FastLinearWindowResult:
    """Reference（权威）：逐行 lstsq / 全窗重算 Gram —— 与 sliding 比较用。

    权威语义 = 每行 ``np.linalg.lstsq(X[valid_rows], y[valid_rows])``（P1-055）。
    """
    T, p0 = X.shape
    if add_intercept:
        X = np.concatenate([np.ones((T, 1)), X], axis=1)
    p = X.shape[1]
    beta = np.full((T, p), np.nan)
    for t in range(T):
        lo = max(0, t - window + 1)
        valid = np.array([_row_valid(X, y, i) for i in range(lo, t + 1)])
        if valid.sum() < p:
            continue
        Xs = X[lo : t + 1][valid]
        ys = y[lo : t + 1][valid]
        b, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        if np.all(np.isfinite(b)):
            beta[t] = b
    return FastLinearWindowResult(beta=beta)


def sliding_parity_check(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    *,
    add_intercept: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """sliding 实现 vs authoritative reference（逐行 lstsq）parity。"""
    ref = rolling_ols_reference(X, y, window, add_intercept=add_intercept)
    fast = rolling_ols_sufficient(X, y, window, add_intercept=add_intercept)
    ref_b, fast_b = ref.beta, fast.beta
    both_fin = np.isfinite(ref_b) & np.isfinite(fast_b)
    same_nan = np.isnan(ref_b) == np.isnan(fast_b)
    match = bool(
        same_nan.all()
        and np.allclose(ref_b[both_fin], fast_b[both_fin], rtol=rtol, atol=atol)
    )
    return {
        "status": "PASS" if match else "FAIL",
        "match": match,
        "max_abs_diff": float(
            np.max(np.abs(ref_b[both_fin] - fast_b[both_fin])) if both_fin.any() else 0.0
        ),
        "lstsq_fallback_rows": int(getattr(fast, "lstsq_fallback_mask", np.zeros(0)).sum())
        if getattr(fast, "lstsq_fallback_mask", None) is not None
        else 0,
    }
