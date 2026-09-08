# -*- coding: utf-8 -*-
"""R35 §40-43 / §106-108: model family shared intermediates.

One fit/decomposition serves multiple outputs:
- ``PCABlock``  : one rolling SVD feeds loading / resid / commonality /
                  explained-ratio canonicals that share the same fit window.
- ``GARCHFitBlock`` : one GARCH fit feeds persistence / shock / next-vol /
                  vol-surprise outputs.

Each block carries the timing identity (§44): input semantic identity + window +
params + fit cutoff + source snapshot + universe.  The external canonicals stay
independent; the planner shares the intermediate (the identity key prevents
cross-universe / cross-param reuse).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PCABlock",
    "GARCHFitBlock",
    "intermediate_identity",
]


def intermediate_identity(
    *,
    family: str,
    input_digest: str,
    window: int,
    params: dict[str, Any],
    fit_cutoff_offset: int,
    universe: str,
) -> str:
    """Deterministic identity of a model intermediate (§44).  Two blocks share
    an identity only when input + window + params + cutoff + universe match."""
    key = "|".join(
        [
            family,
            str(input_digest),
            str(window),
            str(sorted((k, str(v)) for k, v in params.items())),
            str(fit_cutoff_offset),
            str(universe),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _digest_panel(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


@dataclass
class PCABlock:
    """One standardized SVD PCA fit + derived outputs, evaluated for a rolling
    window.  Consumed by panel_rolling_pca_* canonicals sharing the fit.

    R38 P0-060（§23）：fit 与 commonality 走唯一 authoritative ``PCAState``——
    ``commonality`` 返回 canonical 公式 ``1 - Var(resid_i)/Var(ret_i)``（per-stock
    训练窗方差分解，不再用 reconstruction ratio）。
    """

    loadings: np.ndarray            # (k, n_active)
    explained: np.ndarray           # (k,)
    active: np.ndarray              # (n,)
    mu: np.ndarray
    sd: np.ndarray
    k: int
    #: R38 P0-060：训练窗 per-stock 方差分解（canonical commonality 的分子/分母）。
    var_resid: np.ndarray | None = None
    var_ret: np.ndarray | None = None

    # ---- derived outputs ----
    def resid(self, row: np.ndarray, n_components: int) -> np.ndarray:
        """Residual of the current cross-section under the block's model."""
        out = np.full(len(row), np.nan)
        active = self.active
        row_a = row[active]
        cur_valid = np.isfinite(row_a)
        z = np.where(cur_valid, (row_a - self.mu) / self.sd, 0.0)
        k = int(min(self.k, n_components))
        if k < 1:
            return out
        score = self.loadings[:k] @ z
        recon = self.mu + self.sd * (self.loadings[:k].T @ score)
        out[active] = np.where(cur_valid, row_a - recon, np.nan)
        return out

    def commonality(self, row: np.ndarray, n_components: int) -> np.ndarray:
        """Canonical commonality ``1 - Var(resid_i)/Var(ret_i)``（per-stock）。

        该值是**模型属性**（训练窗方差分解，与 canonical ``_pca_commonality`` 同
        公式）；``row`` 只决定输出长度与 NaN mask（inactive 恒 NaN）。
        """
        out = np.full(len(row), np.nan)
        if self.var_resid is None or self.var_ret is None:
            return out
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                np.asarray(self.var_ret) > 1e-12,
                1.0 - np.asarray(self.var_resid) / np.asarray(self.var_ret),
                np.nan,
            )
        out[self.active] = ratio
        return out


def fit_pca_block(X: np.ndarray, n_components: int, *, min_obs: int = 2) -> PCABlock | None:
    """Standardized SVD PCA on a training window, returning a shared block.

    R38 P0-060（§23）：fit 走唯一 authoritative ``PCAState``（coverage-gated
    active + 标准化 SVD），与 canonical ``panel_model._pca_svd`` 逐位一致。
    """
    from factor_engine.cleaned_operators.cross_section.pca_state import PCAState, pca_commonality

    state = PCAState.from_window(X, n_components, absolute_min_obs=int(min_obs))
    if state is None:
        return None
    fit = state.to_dict()
    # 训练窗 per-stock 方差分解（canonical commonality 公式）。
    ratio = pca_commonality(X, fit)
    active_values = X[:, fit["active"]]
    active_values = np.where(np.isfinite(active_values), active_values, np.nan)
    var_ret = np.nanvar(active_values, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_resid = np.where(var_ret > 1e-12, var_ret * (1.0 - ratio[fit["active"]]), np.nan)
    return PCABlock(
        loadings=fit["loadings"],
        explained=fit["explained"],
        active=fit["active"],
        mu=fit["mu"],
        sd=fit["sd"],
        k=fit["k"],
        var_resid=var_resid,
        var_ret=var_ret,
    )


@dataclass
class GARCHFitBlock:
    """One GARCH(1,1) variance-targeting MLE fit + derived outputs."""

    omega: float
    alpha: float
    beta: float
    long_var: float
    persistence: float

    def next_vol(self, r_t: float, h_last: float) -> float:
        h_next = self.omega + self.alpha * r_t * r_t + self.beta * h_last
        return float(np.sqrt(max(h_next, 1e-12)))

    def standardized_shock(self, r_t: float, h_last: float) -> float:
        if not np.isfinite(r_t):
            return np.nan
        return float(r_t / np.sqrt(max(h_last, 1e-12)))


def fit_garch_block(rets: np.ndarray) -> GARCHFitBlock | None:
    """Variance-targeting GARCH(1,1) MLE (matches ts_model.volatility._fit_garch)."""
    from scipy.optimize import minimize

    if len(rets) < 10 or np.std(rets) <= 1e-12:
        return None
    long_var = float(np.var(rets))

    def _nll(p):
        a, b = p
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - b), 1e-12)
        h = np.full(len(rets), long_var)
        for t in range(1, len(rets)):
            h[t] = w + a * rets[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + rets**2 / h))

    try:
        res = minimize(_nll, np.array([0.05, 0.9]), method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
        if not getattr(res, "success", False):
            return None
        a, b = float(res.x[0]), float(res.x[1])
        if not (np.isfinite(a) and np.isfinite(b)) or a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return None
        w = max(long_var * (1 - a - b), 1e-12)
        return GARCHFitBlock(omega=w, alpha=a, beta=b, long_var=long_var, persistence=a + b)
    except Exception:
        return None


def garch_variance_path(
    seg: np.ndarray, block: GARCHFitBlock, h_init: float
) -> float:
    """Variance recursion over ``seg`` (strictly-prior info) using the block."""
    h = h_init
    for i in range(1, len(seg)):
        h = block.omega + block.alpha * seg[i - 1] ** 2 + block.beta * h
    return h
