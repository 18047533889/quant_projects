# -*- coding: utf-8 -*-
"""R38 P0-060..062（§23）：PCA 唯一 authoritative state —— 不再在 backend 复制一套。

R36/R35 的 ``model_family_kernels.PCABlock.commonality`` 是 reconstruction ratio，
而真实 panel PCA canonical 的 commonality 是 ``1 - Var(resid_i)/Var(ret_i)``——
实质语义差异（P0-060）。修复：提取**唯一** authoritative state：

    - :meth:`PCAState.from_window`：coverage-gated active 集 + 标准化 SVD
      （与 canonical ``_pca_svd`` 逐位一致，active-space only）；
    - :func:`pca_commonality`：per-stock ``1 - Var(resid)/Var(ret)``（canonical）；
    - :func:`pca_resid` / :func:`pca_transform`：当前行投影 / 残差。

canonical ``panel_model`` 与 shared ``PCABlock`` **都调用本模块**——单点数学
（P0-060），active coverage / rank cap / missing imputation / commonality 公式
（P0-061）与 canonical 完全一致。sign orientation（P0-062）由 SVD 本身决定；
canonical 的稳定 instrument-ID tie-break 由 ``_pca_loading`` 单独实现（本模块
只负责 fit + commonality/resid/transform 的 shared math）。

``rank_policy``（M-021）：fit 的 rank cap 拆成两种内部 semantic policy——
reconstruction 用 ``k <= p-1``（避免 full-rank residual=0），regression（PCR）
允许 ``k <= p``。PCR canonical 通过 ``_pca_svd(..., rank_policy="regression")``
进入 regression policy。
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: canonical 常量（与 panel_model 一致，§P0-061：单一 truth）。
PCA_MIN_HISTORY = 2
PCA_MIN_COVERAGE = 0.7
_EPS = 1e-12


def _fit(
    X: np.ndarray,
    n_components: int,
    *,
    min_coverage_ratio: float = PCA_MIN_COVERAGE,
    absolute_min_obs: int = PCA_MIN_HISTORY,
    rank_policy: str = "reconstruction",
) -> dict[str, Any] | None:
    """Standardised SVD PCA（与 canonical ``_pca_svd`` 逐位一致，active-space only）。

    ``rank_policy``（M-021）：reconstruction（默认）用 ``k <= n_active - 1``——
    避免 full-rank residual=0；regression（PCR）允许 ``k <= n_active``（全部
    主成分都可用于回归）。默认 reconstruction 与 legacy ``_pca_svd`` 行为逐位
    一致。非法 policy 值 fail-closed（回退 reconstruction）。
    """
    n, d = X.shape
    window = max(1, int(n))
    finite_count = np.sum(np.isfinite(X), axis=0)
    min_obs = max(int(absolute_min_obs), int(np.ceil(window * min_coverage_ratio)))
    active = finite_count >= min_obs
    n_active = int(active.sum())
    if n_active < 2:
        return None
    sub = X[:, active]
    mu_sub = np.nanmean(sub, axis=0)
    sd_sub = np.nanstd(sub, axis=0)
    sd_sub = np.where(sd_sub > _EPS, sd_sub, 1.0)
    Xc = np.where(np.isfinite(sub), sub, mu_sub)
    Xs = (Xc - mu_sub) / sd_sub if sd_sub != 0 else np.nan
    if rank_policy == "regression":
        k = int(min(n_components, n_active, Xs.shape[0] - 1))
    else:
        k = int(min(n_components, n_active - 1, Xs.shape[0] - 1))
    if k < 1:
        return None
    _U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    total_var = float(np.sum(s * s))
    explained = s[:k] ** 2 / max(total_var, _EPS)
    coverage = np.where(float(window) != 0, finite_count[active].astype(float) / float(window), np.nan)
    return {
        "active": active,
        "k": k,
        "mu": mu_sub,
        "sd": sd_sub,
        "loadings": Vt[:k],
        "explained": explained,
        "active_breadth": int(n_active),
        "median_coverage": float(np.median(coverage)),
        "min_coverage": float(coverage.min()),
    }


class PCAState:
    """PCA 唯一 authoritative state（fit + derived outputs 的共享单点数学）。"""

    def __init__(self, fit: dict[str, Any]) -> None:
        self._fit = fit

    @classmethod
    def from_window(
        cls,
        X: np.ndarray,
        n_components: int,
        *,
        min_coverage_ratio: float = PCA_MIN_COVERAGE,
        absolute_min_obs: int = PCA_MIN_HISTORY,
        rank_policy: str = "reconstruction",
    ) -> "PCAState | None":
        fit = _fit(
            X, n_components,
            min_coverage_ratio=min_coverage_ratio,
            absolute_min_obs=absolute_min_obs,
            rank_policy=rank_policy,
        )
        return cls(fit) if fit is not None else None

    # -- canonical fit surface（与 ``_pca_svd`` 返回一致）--
    @property
    def active(self) -> np.ndarray:
        return self._fit["active"]

    @property
    def k(self) -> int:
        return self._fit["k"]

    @property
    def loadings(self) -> np.ndarray:
        return self._fit["loadings"]

    @property
    def explained(self) -> np.ndarray:
        return self._fit["explained"]

    @property
    def mu(self) -> np.ndarray:
        return self._fit["mu"]

    @property
    def sd(self) -> np.ndarray:
        return self._fit["sd"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self._fit)

    # -- derived outputs（§P0-060：唯一 commonality 公式）--

    def resid(self, row: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """当前横截面残差（canonical ``_pca_resid`` 语义，active-space aware）。"""
        out = np.full(len(row), np.nan)
        active = self.active
        row_a = row[active]
        cur_valid = np.isfinite(row_a)
        z = np.where(cur_valid, (row_a - self.mu) / self.sd, 0.0)
        k = int(min(self.k, n_components if n_components is not None else self.k))
        if k < 1:
            return out
        score = self.loadings[:k] @ z
        recon = self.mu + self.sd * (self.loadings[:k].T @ score)
        out[active] = row_a - recon
        return out

    def transform(self, row: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """当前行投影到 active 分量（canonical ``_pca_transform`` 语义）。"""
        row_a = row[self.active]
        cur_valid = np.isfinite(row_a)
        z = np.where(cur_valid, (row_a - self.mu) / self.sd, 0.0)
        k = int(min(self.k, n_components if n_components is not None else self.k))
        if k < 1:
            return np.zeros(0)
        return self.loadings[:k] @ z

    def commonality(self, X: np.ndarray) -> np.ndarray:
        """per-stock commonality ``1 - Var(resid_i)/Var(ret_i)``（canonical 公式）。

        X = 训练窗口（与 fit 同一窗口）；inactive 股票输出 NaN。
        """
        return pca_commonality(X, self._fit)


def pca_commonality(X: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    """Per-stock commonality ``1 - Var(resid_i)/Var(ret_i)`` over the window（canonical）。"""
    n_rows, n_cols = X.shape
    Xa = X[:, pca["active"]]
    Xa_imp = np.where(np.isfinite(Xa), Xa, pca["mu"][None, :])
    z = np.where(pca["sd"] != 0, (Xa_imp - pca["mu"]) / pca["sd"], np.nan)
    score = pca["loadings"] @ z.T                           # (k, n_rows)
    recon = (pca["mu"][:, None] + pca["sd"][:, None] * (pca["loadings"].T @ score)).T
    resid = Xa - recon
    var_resid = np.nanvar(resid, axis=0)
    var_ret = np.nanvar(Xa, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_active = np.where(var_ret > _EPS, 1.0 - var_resid / var_ret, np.nan)
    out = np.full(n_cols, np.nan)
    out[pca["active"]] = ratio_active
    return out
