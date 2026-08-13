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
matrix is built; the correlation is the **biweight midcorrelation** (a robust
correlation that differs from Pearson on outlying pairs — review #29), not the
Pearson correlation on z-scores (which is affine-invariant and would add no
expressiveness).  Complete-case only; fail-closed to NaN on degenerate windows.
Prefix-causal and deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
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
    param_specs: dict | None = None,
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
        param_specs=dict(param_specs) if param_specs else {},
    )


def _tri_rolling(a: np.ndarray, b: np.ndarray, c: np.ndarray, w: int, fn) -> np.ndarray:
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            out[r, col] = fn(a[lo : r + 1, col], b[lo : r + 1, col], c[lo : r + 1, col])
    return out


def _robust_scale(col: np.ndarray) -> float | None:
    """Median/MAD robust scale with a std fallback (R5 P1-42(a)).

    When MAD == 0 (at least half the observations equal the median) but the
    column is not truly constant, dividing by ``_EPS`` would turn a near-constant
    feature into a single extreme outlier and corrupt the correlation matrix;
    fall back to ``std`` in that case.  Only a truly constant column (MAD == 0
    and std == 0) has no scale at all — the caller must fail closed.
    """
    med = float(np.median(col))
    mad = float(np.median(np.abs(col - med)))
    if mad > 0.0:
        return 1.4826 * mad
    std = float(np.std(col))
    if std <= _EPS:
        return None
    return std


def _feature_matrix(c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, min_rows: int) -> np.ndarray | None:
    F = np.stack([c1, c2, c3], axis=1)
    complete = np.isfinite(F).all(axis=1)
    if int(complete.sum() < min_rows:
        return None
    Z = F[complete]
    z = np.empty_like(Z)
    for j in range(3):
        med = float(np.median(Z[:, j]))
        scale = _robust_scale(Z[:, j])
        if scale is None:
            # R5 P1-42(a): a constant feature carries no information — fail
            # closed to NaN rather than emitting a degenerate all-zero column.
            return None
        z[:, j] = (Z[:, j] - med) / scale
    return z


def _biweight_midcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Biweight midcorrelation (Wilcox 2005 / WGCNA) between two vectors.

    Pearson correlation is invariant to independent affine rescaling, so
    computing it on median/MAD z-scores adds NO expressiveness over the raw
    correlation (review finding #29).  The biweight midcorrelation weights each
    observation down *before* the covariance (``(1-u^2)^2`` for ``|u|<1`` with
    ``u=(x-med)/(9*MAD)``), so outliers are down-weighted and the result is a
    genuinely different, robust statistic.
    """
    if a.size < 3 or b.size < 3:
        return np.nan
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum() < 3:
        return np.nan
    a = a[ok]
    b = b[ok]
    ma = float(np.median(a))
    mb = float(np.median(b))
    mada = float(np.median(np.abs(a - ma)))
    madb = float(np.median(np.abs(b - mb)))
    if mada <= _EPS:
        mada = float(np.std(a))
    if madb <= _EPS:
        madb = float(np.std(b))
    if mada <= _EPS or madb <= _EPS:
        return np.nan  # degenerate column: no scale -> fail closed
    u = (a - ma) / (9.0 * mada)
    v = (b - mb) / (9.0 * madb)
    wu = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    wv = np.where(np.abs(v) < 1.0, (1.0 - v * v) ** 2, 0.0)
    if float(np.sum(wu * wv) <= _EPS:
        return np.nan
    da = (a - ma) * wu
    db = (b - mb) * wv
    num = float(np.sum(da * db))
    den = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if den <= _EPS:
        return np.nan
    return float(np.clip(num / den, -1.0, 1.0))


def _bicor_matrix(z: np.ndarray) -> np.ndarray | None:
    """Pairwise biweight midcorrelation matrix (robust, not Pearson)."""
    m = z.shape[1]
    out = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            c = _biweight_midcorr(z[:, i], z[:, j])
            if not np.isfinite(c):
                return None
            out[i, j] = c
            out[j, i] = c
    return out


def _nearest_psd_correlation(c: np.ndarray, *, max_iter: int = 20, tol: float = 1e-8) -> np.ndarray:
    """Nearest-PSD correlation projection (Higham-style alternating projections).

    A pairwise robust correlation matrix — each off-diagonal estimated on its own
    pair — is NOT guaranteed positive-semidefinite, yet the spectral geometry
    operators treat it as a covariance structure (eigendecomposition, dominant
    mode, spectral entropy).  This helper projects the matrix onto the
    intersection of the PSD cone and the unit-diagonal correlation set by
    alternating:

    1. eigen-decompose, clamp eigenvalues to a small floor ``max(λ, _EPS)`` and
       reconstruct — this removes negative eigenvalue mass instead of silently
       dropping it (the pre-R11 ``np.maximum(w, 0.0)`` clipping changed the
       trace / mode-share / effective-rank);
    2. rescale rows/cols by ``sqrt(diag)`` so the diagonal returns to 1.

    The rescale is a congruence ``D^{-1/2} C D^{-1/2}`` (with invertible ``D``),
    which preserves PSD, so the fixed point is a genuine correlation matrix whose
    eigenvalues are all >= 0 up to floating-point noise.  Iterate until the
    Frobenius change is small.  Self-contained, numpy only.
    """
    x = np.asarray(c, dtype=float).copy()
    for _ in range(max_iter):
        prev = x
        w, v = np.linalg.eigh(x)
        w = np.maximum(w, _EPS)
        x = (v * w) @ v.T
        # rescale to unit diagonal (congruence preserves PSD)
        dg = np.sqrt(np.diag(x))
        dg[dg < _EPS] = 1.0
        x = x / np.outer(dg, dg)
        np.fill_diagonal(x, 1.0)
        if np.linalg.norm(x - prev, ord="fro") <= tol * max(1.0, np.linalg.norm(prev, ord="fro")):
            break
    return x


def _canonical_corr(z: np.ndarray) -> np.ndarray | None:
    """Canonical feature correlation matrix: pairwise bicor, PSD-projected.

    EVERY spectral consumer (eigenvalues, dominant direction, mode share,
    effective rank, subspace rotation) must read this ONE matrix so the
    direction, the mode-share and the entropy all come from the same PSD
    correlation structure.  Returns None when the pairwise bicor fails closed.
    """
    c = _bicor_matrix(z)
    if c is None:
        return None
    return _nearest_psd_correlation(c)


def _corr_eigenvalues(z: np.ndarray) -> np.ndarray | None:
    c = _canonical_corr(z)
    if c is None:
        return None
    w = np.linalg.eigvalsh(c)
    # The PSD projection is the primary mechanism (eigenvalues are naturally
    # >= 0 up to numerical noise); only clamp a tiny floating-point-noise tail
    # instead of wholesale clipping as a substitute for projection.
    w = np.maximum(w, 0.0)
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
    # Zero eigenvalue -> p=0: define 0*log(0) = 0 (spectral entropy convention).
    plog = np.zeros_like(p)
    nz = p > 0.0
    plog[nz] = p[nz] * np.log(p[nz])
    ent = float(-np.sum(plog))
    return float(np.clip(np.exp(ent), 1.0, 3.0))


_MIN_EIGENGAP = 0.02  # normalized ``(λ1-λ2)/sum(λ)`` below which the top direction is not identifiable

# M-8xx: the dominant-direction identifiability gap is an explicit, versioned,
# non-searchable ESTIMATOR knob (default 0.02).  It decides when the top two
# eigenvalues are too degenerate for the dominant eigenvector to be stable; it
# is surfaced on ``ts_feature_subspace_rotation``'s metadata (param_specs,
# searchable=False) instead of being a hidden kernel constant.
_EIGENGAP_SPEC = ParamSpec(
    dtype=float,
    min=0.0,
    max=1.0,
    default=_MIN_EIGENGAP,
    searchable=False,
    param_role=ParamRole.ESTIMATOR_RESOLUTION,
)


def _dominant_direction(z: np.ndarray, eigen_gap: float = _MIN_EIGENGAP) -> np.ndarray | None:
    # R11 P1: the dominant direction must come from the SAME projected matrix
    # whose eigenvalues feed mode_share / effective_rank.  A pairwise bicor
    # matrix is not guaranteed PSD, so the raw matrix's eigenvector could
    # disagree with the eigenvalue gap computed on a (clipped) version of it.
    c = _canonical_corr(z)
    if c is None:
        return None
    w = np.linalg.eigvalsh(c)
    w = np.maximum(w, 0.0)
    if w.sum() <= _EPS:
        return None
    # Round-7 P0 (review §33): when the top two eigenvalues are nearly tied
    # (``(λ1-λ2)/sum(λ) < eigen_gap``, default ``_MIN_EIGENGAP``) the dominant
    # eigenvector is not identifiable — small noise can flip the "rotation" to
    # ~90° even though the covariance structure did not change.  Fail closed
    # (None -> NaN) instead of emitting a spurious regime-rotation angle.
    # M-8xx: ``eigen_gap`` is the explicit, versioned ESTIMATOR knob — a caller
    # can relax it (e.g. 0.0) to admit near-degenerate directions, which changes
    # the output and is reflected in the operator's param_specs / semantic
    # identity.
    gap = float(w[-1] - (w[-2] if w.size >= 2 else 0.0))
    if gap / float(w.sum() < eigen_gap:
        return None
    _, v = np.linalg.eigh(c)
    v1 = v[:, -1]
    # deterministic orientation: make the largest-|loading| element positive.
    k = int(np.argmax(np.abs(v1)))
    if v1[k] < 0:
        v1 = -v1
    return v1


def geometry_audit_bundle(
    panel: pd.DataFrame, *, window: int = 30, min_rows: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """R13 NEW-P0-10: internal TEST bundle exposing the shared-PSD-matrix property.

    NOT part of the operator surface.  Returns ``(raw_corr, projected_corr,
    eigvals, eigvecs)`` so the machine semantic auditor can verify that the
    nearest-PSD projection, the eigenvalues and the eigenvectors all come from
    ONE matrix — the mode-share / effective-rank / dominant-direction kernels
    must never read a raw (possibly non-PSD) correlation while another reads the
    projected one.  ``None`` when the panel cannot form a valid correlation.
    """
    z = panel.to_numpy(dtype=float)
    if z.ndim != 2 or z.shape[1] < 3 or z.shape[0] < min_rows:
        return None
    seg = z[:window]
    raw = _bicor_matrix(seg)
    if raw is None:
        return None
    projected = _nearest_psd_correlation(raw)
    eigvals, eigvecs = np.linalg.eigh(projected)
    return raw, projected, eigvals, eigvecs


def _subspace_rotation_chunk(c1, c2, c3, recent: int, prior: int, min_rows: int, eigen_gap: float = _MIN_EIGENGAP) -> float:
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
    vr = _dominant_direction(zr, eigen_gap=eigen_gap)
    vp = _dominant_direction(zp, eigen_gap=eigen_gap)
    if vr is None or vp is None:
        return np.nan
    ang = float(np.arccos(np.clip(abs(float(np.dot(vr, vp))), 0.0, 1.0)))
    return float(ang / (0.5 * np.pi))  # [0, 1]: 0 = same regime, 1 = orthogonal


def _true_beta(yy: np.ndarray, xx: np.ndarray, min_pairs: int) -> float | None:
    """OLS slope ``Cov(y, x)/Var(x)`` — the TRUE beta, not a correlation.

    Review finding #30: standardising BOTH series and taking the slope
    degenerates beta to the correlation (a correlation break, not a beta
    break).  The operator is named ``beta_break``, so it must report a change
    in the genuine regression coefficient ``Cov(y,x)/Var(x)`` (which carries
    ``unit(y)/unit(x)``).  The break score is then normalised by
    ``|beta_prior| + rolling_median(|beta_recent|)`` — a scale in the same
    beta unit — so the relative change stays scale-aware.
    """
    ok = np.isfinite(yy) & np.isfinite(xx)
    if int(ok.sum() < min_pairs:
        return None
    yy = yy[ok]
    xx = xx[ok]
    vx = float(np.var(xx))
    if vx <= _EPS:
        return None
    return float(np.cov(yy, xx, ddof=0)[0, 1] / vx)


def _beta_break_chunk(yc, xc, recent: int, prior: int, min_pairs: int) -> float:
    n = yc.shape[0]
    if n < recent + prior:
        return np.nan
    yr = yc[n - recent :]
    xr = xc[n - recent :]
    yp = yc[n - recent - prior : n - recent]
    xp = xc[n - recent - prior : n - recent]

    br = _true_beta(yr, xr, min_pairs)
    bp = _true_beta(yp, xp, min_pairs)
    if br is None or bp is None:
        return np.nan
    # P0-51: the old ``1 + |beta_prior|`` denominator mixed a dimensionless 1
    # with a unitful beta (unit(y)/unit(x)) — dimensionally invalid.  Normalise
    # by a robust scale in the SAME beta unit: |beta_prior| plus the rolling
    # median of |beta| over sub-windows of the recent window.
    sub = min(20, int(yr.shape[0]))
    abs_betas: list[float] = []
    if sub >= min_pairs:
        for s in range(yr.shape[0] - sub + 1):
            b = _true_beta(yr[s : s + sub], xr[s : s + sub], min_pairs)
            if b is not None:
                abs_betas.append(abs(b))
    if abs_betas:
        scale = abs(bp) + float(np.median(abs_betas))
    else:
        scale = abs(bp) + abs(br)
    if not np.isfinite(scale) or scale <= _EPS:
        return np.nan
    return float((br - bp) / scale)


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
        "特征谱有效秩 exp(-Σ p log p)（多字段有效自由度；effective dimension，"
        "dimensionless，非序数 rank）。",
        ["f1", "f2", "f3", "window"],
        domain="price_volume",
        unit="dimensionless",
        cost=5,
        extra_tags=("semantic_kind:effective_dimension",),
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

    **M-8xx：主导方向可识别性门槛 ``eigen_gap``（默认 0.02）是显式、versioned、
    searchable=False 的 ESTIMATOR 常数**——当前两特征值相对谱隙低于该值时，
    主导特征向量不稳定，输出 NaN（fail-closed）。默认 0.02 已文档化；调低
    （如 0.0）可承认近简并方向，会改变输出并进入语义身份。
    """

    metadata = _metadata(
        "ts_feature_subspace_rotation",
        "recent vs prior 特征主导方向夹角（regime rotation；eigen_gap=0.02 可识别性门槛，"
        "ESTIMATOR_RESOLUTION searchable=False）。",
        ["f1", "f2", "f3", "recent_window", "prior_window", "eigen_gap"],
        domain="price_volume",
        unit="angle",
        cost=6,
        param_specs={
            "eigen_gap": _EIGENGAP_SPEC,
        },
        extra_tags=(f"eigen_gap:{_MIN_EIGENGAP}",),
    )

    def _calculate_series(
        self,
        f1: pd.DataFrame,
        f2: pd.DataFrame,
        f3: pd.DataFrame,
        recent_window: int = 30,
        prior_window: int = 90,
        eigen_gap: float = _MIN_EIGENGAP,
        **_: Any,
    ) -> pd.DataFrame:
        # P1-32: ``window`` was redundant — the kernel only ever uses the last
        # ``recent + prior`` rows, so any window >= recent+prior produced the
        # exact same output and only inflated the search surface.
        r = int(recent_window)
        p = int(prior_window)
        if r < 5 or p < 5:
            raise ValueError("ts_feature_subspace_rotation requires recent/prior >= 5")
        if not (0.0 <= float(eigen_gap) <= 1.0):
            raise ValueError("ts_feature_subspace_rotation requires 0 <= eigen_gap <= 1")
        w = r + p
        return frame_like(
            f1,
            _tri_rolling(
                f1.to_numpy(dtype=float),
                f2.to_numpy(dtype=float),
                f3.to_numpy(dtype=float),
                w,
                lambda a, b, c: _subspace_rotation_chunk(a, b, c, r, p, 5, float(eigen_gap)),
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
    """字段关系突变（recent vs prior 滚动真实 beta 的归一化变化）。

    ``(beta_recent - beta_prior) / (|beta_prior| + rolling_median(|beta_recent|))``，其中
    ``beta = Cov(y, x)/Var(x)`` 是真实回归斜率（携带 unit(y)/unit(x)）。
    Review #30：若把 x、y 各自标准化再取斜率，beta 就退化成相关系数——那是
    correlation break，不是 beta break。本算子坚持真实 beta；断点分数用与 beta
    同量纲的 robust scale（|β_prior| + median|β_recent|）归一化，使其在不同
    y/x 组合间仍是相对变化。P1。
    """

    metadata = _metadata(
        "ts_beta_break_score",
        "recent vs prior 真实 beta=Cov(y,x)/Var(x) 变化 (Δβ)/(|β_prior|+median|β_recent|)。",
        ["y", "x", "recent_window", "prior_window"],
        domain="price_volume",
        unit="unit(y)/unit(x)",
        cost=5,
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        x: pd.DataFrame,
        recent_window: int = 30,
        prior_window: int = 90,
        **_: Any,
    ) -> pd.DataFrame:
        # P1-33: ``window`` was redundant — the kernel only uses the last
        # ``recent + prior`` rows, so it is dropped from the search surface.
        r = int(recent_window)
        p = int(prior_window)
        if r < 5 or p < 5:
            raise ValueError("ts_beta_break_score requires recent/prior >= 5")
        w = r + p
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

    _surface.extend_extended_only({
            "ts_feature_mode_share",
            "ts_feature_effective_rank",
            "ts_feature_subspace_rotation",
            "ts_beta_break_score",
        })
    for _canon in (
        "ts_feature_mode_share",
        "ts_feature_effective_rank",
        "ts_feature_subspace_rotation",
        "ts_beta_break_score",
    ):
        register_polars_udf(_canon)


_register_surface()
