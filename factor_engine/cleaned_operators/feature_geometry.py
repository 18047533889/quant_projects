# -*- coding: utf-8 -*-
"""Multi-field covariance geometry (2026-08 market language, P1).

A single name with several state fields (price / volume / volatility / ...) can
be driven by one common latent mode.  These operators measure how the *feature
correlation structure* behaves in time:

* ``ts_feature_mode_share``     — share of the largest eigenvalue of the
  feature correlation matrix (fields moving in lockstep => close to 1).
* ``ts_feature_effective_rank`` — exp(entropy) of the normalised eigenvalues
  (1 = one mode, 3 = fully independent).
* ``ts_feature_subspace_rotation`` — angle between the dominant eigenvectors of
  the recent vs prior feature correlation matrices (regime rotation).
* ``ts_beta_break_score``       — normalised change of a rolling beta
  (relation-strength breaks, e.g. valuation-vs-profitability).

All three inputs are robust-standardised per window before the correlation
matrix is built; complete-case only; fail-closed to NaN on degenerate windows.
Prefix-causal and deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="feature_geometry",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "feature_geometry", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _tri_rolling(a: np.ndarray, b: np.ndarray, c: np.ndarray, w: int, fn) -> np.ndarray:
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            out[r, col] = fn(a[lo : r + 1, col], b[lo : r + 1, col], c[lo : r + 1, col])
    return out


def _feature_matrix(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, min_rows: int) -> np.ndarray | None:
    F = np.stack([c1, c2, c3], axis=1)
    complete = np.isfinite(F).all(axis=1)
    if int(complete.sum()) < min_rows:
        return None
    Z = F[complete]
    z = np.empty_like(Z)
    for j in range(3):
        med = float(np.median(Z[:, j]))
        mad = float(np.median(np.abs(Z[:, j] - med)))
        z[:, j] = (Z[:, j] - med) / (1.4826 * mad + _EPS)
    return z


def _corr_eigenvalues(z: np.ndarray) -> np.ndarray | None:
    c = np.corrcoef(z, rowvar=False)
    if not np.all(np.isfinite(c)):
        return None
    w = np.maximum(np.linalg.eigvalsh(c), 0.0)
    if w.sum() <= _EPS:
        return None
    return w  # ascending


def _mode_share_chunk(c1, c2, c3, min_rows: int) -> float:
    z = _feature_matrix(c1, c2, c3, min_rows)
    if z is None:
        return np.nan
    w = _corr_eigenvalues(z)
    if w is None:
        return np.nan
    return float(w[-1] / w.sum())


def _effective_rank_chunk(c1, c2, c3, min_rows: int) -> float:
    z = _feature_matrix(c1, c2, c3, min_rows)
    if z is None:
        return np.nan
    w = _corr_eigenvalues(z)
    if w is None:
        return np.nan
    p = w / w.sum()
    ent = float(-np.sum(p * np.log(p)))
    return float(np.clip(np.exp(ent), 1.0, 3.0))


def _dominant_direction(z: np.ndarray) -> np.ndarray | None:
    w = _corr_eigenvalues(z)
    if w is None:
        return None
    c = np.corrcoef(z, rowvar=False)
    _, v = np.linalg.eigh(c)
    v1 = v[:, -1]
    # deterministic orientation: make the largest-|loading| element positive.
    k = int(np.argmax(np.abs(v1)))
    if v1[k] < 0:
        v1 = -v1
    return v1


def _subspace_rotation_chunk(c1, c2, c3, recent: int, prior: int, min_rows: int) -> float:
    n = c1.shape[0]
    if n < recent + prior or recent < min_rows or prior < min_rows:
        return np.nan
    zr = _feature_matrix(c1[n - recent :], c2[n - recent :], c3[n - recent :], min_rows)
    zp = _feature_matrix(
        c1[n - recent - prior : n - recent],
        c2[n - recent - prior : n - recent],
        c3[n - recent - prior : n - recent],
        min_rows,
    )
    if zr is None or zp is None:
        return np.nan
    vr = _dominant_direction(zr)
    vp = _dominant_direction(zp)
    if vr is None or vp is None:
        return np.nan
    ang = float(np.arccos(np.clip(abs(float(np.dot(vr, vp))), 0.0, 1.0)))
    return float(ang / (0.5 * np.pi))  # [0, 1]: 0 = same regime, 1 = orthogonal


def _beta_break_chunk(yc, xc, recent: int, prior: int, min_pairs: int) -> float:
    n = yc.shape[0]
    if n < recent + prior:
        return np.nan
    yr = yc[n - recent :]
    xr = xc[n - recent :]
    yp = yc[n - recent - prior : n - recent]
    xp = xc[n - recent - prior : n - recent]

    def _beta(yy, xx) -> float | None:
        ok = np.isfinite(yy) & np.isfinite(xx)
        if int(ok.sum()) < min_pairs:
            return None
        yy = yy[ok]
        xx = xx[ok]
        vx = float(np.var(xx))
        if vx <= _EPS:
            return None
        return float(np.cov(yy, xx, ddof=0)[0, 1] / vx)

    br = _beta(yr, xr)
    bp = _beta(yp, xp)
    if br is None or bp is None:
        return np.nan
    return float((br - bp) / (1.0 + abs(bp)))


@register_operator(
    name="ts_feature_mode_share",
    category="feature_geometry",
    business_category="feature_geometry",
    canonical="ts_feature_mode_share",
    source="feature_geometry",
)
class TsFeatureModeShare(SeriesOperator):
    """多字段共同驱动程度（特征相关矩阵最大特征值占比 λ1/Σλ）。

    三个字段（如 price/volume/volatility）逐窗口 robust 标准化后取相关矩阵，
    输出最大特征值占比。高 = 多个字段最近基本被同一个共同状态驱动
    （risk-on/off mode）；低 = 字段各自独立。P1。
    """

    metadata = _metadata(
        "ts_feature_mode_share",
        "特征相关矩阵最大特征值占比 λ1/Σλ（共同 mode 强度）。",
        ["f1", "f2", "f3", "window"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 5:
            raise ValueError("ts_feature_mode_share requires window >= 5")
        return frame_like(
            f1,
            _tri_rolling(
                f1.to_numpy(dtype=float),
                f2.to_numpy(dtype=float),
                f3.to_numpy(dtype=float),
                w,
                lambda a, b, c: _mode_share_chunk(a, b, c, 5),
            ),
        )


@register_operator(
    name="ts_feature_effective_rank",
    category="feature_geometry",
    business_category="feature_geometry",
    canonical="ts_feature_effective_rank",
    source="feature_geometry",
)
class TsFeatureEffectiveRank(SeriesOperator):
    """多字段有效自由度 exp(-Σ p log p)（1=单 mode，3=完全独立）。

    同一特征相关矩阵的特征值归一化为 p，输出谱熵的指数。低 = 字段被一个
    latent mode 压制；高 = 字段自由。日频 liquidity/noise state 的天然代理。P1。
    """

    metadata = _metadata(
        "ts_feature_effective_rank",
        "特征谱有效秩 exp(-Σ p log p)（多字段有效自由度）。",
        ["f1", "f2", "f3", "window"],
        domain="price_volume",
        unit="rank",
        cost=5,
    )

    def _calculate_series(
        self, f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 5:
            raise ValueError("ts_feature_effective_rank requires window >= 5")
        return frame_like(
            f1,
            _tri_rolling(
                f1.to_numpy(dtype=float),
                f2.to_numpy(dtype=float),
                f3.to_numpy(dtype=float),
                w,
                lambda a, b, c: _effective_rank_chunk(a, b, c, 5),
            ),
        )


@register_operator(
    name="ts_feature_subspace_rotation",
    category="feature_geometry",
    business_category="feature_geometry",
    canonical="ts_feature_subspace_rotation",
    source="feature_geometry",
)
class TsFeatureSubspaceRotation(SeriesOperator):
    """特征主导方向旋转（recent vs prior 相关矩阵主特征向量夹角）。

    recent 与 prior 两个不重叠窗口各估计特征相关矩阵的主特征向量，输出夹角
    归一化到 [0,1]（1 = 正交，regime 彻底切换）。捕捉"字段间关系结构"的
    旋转，而非字段本身的水平。P1。
    """

    metadata = _metadata(
        "ts_feature_subspace_rotation",
        "recent vs prior 特征主导方向夹角（regime rotation）。",
        ["f1", "f2", "f3", "window", "recent_window", "prior_window"],
        domain="price_volume",
        unit="angle",
        cost=6,
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        window: int = 120,
        recent_window: int = 30,
        prior_window: int = 90,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        r = int(recent_window)
        p = int(prior_window)
        if r < 5 or p < 5:
            raise ValueError("ts_feature_subspace_rotation requires recent/prior >= 5")
        if w < r + p:
            raise ValueError("ts_feature_subspace_rotation requires window >= recent + prior")
        return frame_like(
            f1,
            _tri_rolling(
                f1.to_numpy(dtype=float),
                f2.to_numpy(dtype=float),
                f3.to_numpy(dtype=float),
                w,
                lambda a, b, c: _subspace_rotation_chunk(a, b, c, r, p, 5),
            ),
        )


@register_operator(
    name="ts_beta_break_score",
    category="feature_geometry",
    business_category="feature_geometry",
    canonical="ts_beta_break_score",
    source="feature_geometry",
)
class TsBetaBreakScore(SeriesOperator):
    """字段关系突变（recent vs prior 滚动 beta 的归一化变化）。

    ``(beta_recent - beta_prior) / (1 + |beta_prior|)``：y 对 x 的敏感度在
    recent 与 prior 窗口之间变化多大。y=valuation、x=profitability/growth 等
    组合可捕捉"估值-基本面关系断裂"。P1。
    """

    metadata = _metadata(
        "ts_beta_break_score",
        "recent vs prior 滚动 beta 变化 (Δbeta)/(1+|beta_prior|)。",
        ["y", "x", "window", "recent_window", "prior_window"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        window: int = 120,
        recent_window: int = 30,
        prior_window: int = 90,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        r = int(recent_window)
        p = int(prior_window)
        if r < 5 or p < 5:
            raise ValueError("ts_beta_break_score requires recent/prior >= 5")
        if w < r + p:
            raise ValueError("ts_beta_break_score requires window >= recent + prior")

        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                out[row, col] = _beta_break_chunk(yv[lo : row + 1, col], xv[lo : row + 1, col], r, p, 5)
        return frame_like(y, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_feature_mode_share",
            "ts_feature_effective_rank",
            "ts_feature_subspace_rotation",
            "ts_beta_break_score",
        }
    )
    for _canon in (
        "ts_feature_mode_share",
        "ts_feature_effective_rank",
        "ts_feature_subspace_rotation",
        "ts_beta_break_score",
    ):
        register_polars_udf(_canon)


_register_surface()
