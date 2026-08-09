# -*- coding: utf-8 -*-
"""Hankel / SSA structure operators (2026-08 geometry/math expansion).

The shared kernel prepares the trailing window as a *strict trailing contiguous
finite run* (production default — data on either side of a gap is never
re-connected), then forms the trajectory ``Hankel`` matrix ``H`` of shape
``(N - embedding_dim + 1, embedding_dim)`` from lagged rows, and performs an
SVD → singular values ``σ``.  The family then reads the singular spectrum:

* R4-66: the contiguous run must cover at least ``min_contiguous_fraction``
  (default 0.8) of the window, otherwise the row emits NaN.  Without this gate a
  sparse day could build an 18-point Hankel matrix and a full day a 60-point one
  under the same operator name — a different factor from one row to the next.
* R4-67: linear interpolation across a gap (``missing_mode="interpolate"``) is
  research-only and requires explicit opt-in; the production default is
  ``strict_contiguous``.

* ``ts_hankel_effective_rank``       — exponential of the entropy of the
  normalized squared-singular-value distribution, normalized by ``min(H.shape)``.
* ``ts_hankel_singular_gap``         — relative gap between the two largest
  singular values (a proxy for mode separation / structure strength).
* ``ts_ssa_reconstruction_residual`` — normalized mean-squared residual of the
  window reconstructed from the top ``n_components`` SVD modes (anti-diagonal
  averaging).

All operators are trailing-window, prefix-causal and deterministic.  Windows
too short to form a Hankel matrix emit NaN.  Invalid parameters raise
``ValueError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12
_MIN_FINITE_FRAC = 0.5


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    # R4-95: the trailing window is consumed as a strict trailing contiguous run;
    # exact row count is NOT required — rows with too short a run emit NaN.
    return OperatorMetadata(
        name=name,
        category="hankel",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "hankel", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:ts_structure",
            f"unit:{unit}", f"cost:{cost}",
        ],
        window_semantics="trailing_contiguous",
    )


def _fill_window(
    chunk: np.ndarray,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray | None:
    """Prepare a window for Hankel/SSA.

    Audit P1-A: the production default is ``strict_contiguous`` — the longest
    trailing contiguous finite run, with a 50%-finite coverage gate.  Linear
    interpolation (which uses data *after* a gap to reconstruct observations
    before it) is research-only and requires ``missing_mode="interpolate"``.

    R4-66: the contiguous run must cover at least ``min_contiguous_fraction`` of
    the window (default 0.8), otherwise ``None`` is returned so the row emits
    NaN.  Without this gate the same operator would alternate between an
    18-point and a 60-point Hankel matrix from day to day.
    """
    if chunk.size == 0:
        return None
    finite = np.isfinite(chunk)
    n_fin = int(np.count_nonzero(finite))
    if n_fin / float(chunk.size) < _MIN_FINITE_FRAC:
        return None
    if n_fin == chunk.size:
        return chunk.astype(float)
    if missing_mode == "interpolate":
        n = chunk.size
        t = np.arange(n, dtype=float)
        filled = chunk.astype(float).copy()
        filled[~finite] = np.interp(t[~finite], t[finite], chunk[finite].astype(float))
        return filled
    # strict_contiguous (default): never re-connect data across a gap.
    end = chunk.size
    while end > 0 and not np.isfinite(chunk[end - 1]):
        end -= 1
    start = end
    while start > 0 and np.isfinite(chunk[start - 1]):
        start -= 1
    run_len = end - start
    if run_len / float(chunk.size) < float(min_contiguous_fraction):
        return None
    return chunk[start:end].astype(float)


def _hankel_singular_values(filled: np.ndarray, emb: int) -> np.ndarray | None:
    """SVD singular values (descending) of the trajectory Hankel matrix."""
    n = filled.size
    rows_h = n - emb + 1
    if rows_h < 1:
        return None
    h = np.empty((rows_h, emb), dtype=float)
    for j in range(emb):
        h[:, j] = filled[j : j + rows_h]
    s = np.linalg.svd(h, compute_uv=False)
    return s


def _hankel_effective_rank_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e = int(window), int(emb)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                continue
            s = _hankel_singular_values(filled, e)
            if s is None:
                continue
            energy = s ** 2
            total = float(energy.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            p = energy / total
            pos = p[p > 0]
            if pos.size == 0:
                continue
            ent = -float(np.sum(pos * np.log(pos)))
            n = filled.size
            denom = min(n - e + 1, e)
            if denom > 0:
                out[r, c] = float(np.exp(ent) / denom)
    return out


def _hankel_singular_gap_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e = int(window), int(emb)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                continue
            s = _hankel_singular_values(filled, e)
            if s is None:
                continue
            s1 = float(s[0])
            s2 = float(s[1]) if s.size > 1 else 0.0
            total = float(s.sum())
            out[r, c] = (s1 - s2) / (total + _EPS)
    return out


def _ssa_reconstruction_residual_series(
    x2d: np.ndarray,
    window: int,
    emb: int,
    n_components: int,
    missing_mode: str = "strict_contiguous",
    min_contiguous_fraction: float = 0.8,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, e, k = int(window), int(emb), int(n_components)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            filled = _fill_window(col[i0 : r + 1], missing_mode, min_contiguous_fraction)
            if filled is None:
                continue
            n = filled.size
            rows_h = n - e + 1
            if rows_h < 1:
                continue
            # data-dependent degeneracy: fewer singular values than requested
            if k >= min(rows_h, e):
                continue
            h = np.empty((rows_h, e), dtype=float)
            for j in range(e):
                h[:, j] = filled[j : j + rows_h]
            u, s, vt = np.linalg.svd(h, full_matrices=False)
            hk = (u[:, :k] * s[:k]) @ vt[:k, :]
            recon = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for i in range(rows_h):
                for j in range(e):
                    recon[i + j] += hk[i, j]
                    counts[i + j] += 1.0
            recon /= counts
            resid = float(np.mean((filled - recon) ** 2))
            var_x = float(np.var(filled))
            out[r, c] = resid / (var_x + _EPS)
    return out


def _check_int(value: Any, name: str, minimum: int) -> int:
    """R4-68: strict integer contract — reject bools and non-integer floats so
    ``5.9`` never silently truncates to ``5`` and compiles to the same factor as
    ``5.0`` (a false search space)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    fv = float(value)
    if not np.isfinite(fv) or fv != float(int(fv)):
        raise ValueError(f"{name} must be an integer")
    iv = int(fv)
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return iv


def _check_hankel_params(
    window: int, emb: int, n_components: int | None = None, min_contiguous_fraction: float = 0.8
) -> tuple[int, int, int | None, float]:
    w = _check_int(window, "window", 2)
    e = _check_int(emb, "embedding_dim", 2)
    if w - e + 1 < 1:
        raise ValueError("window must be >= embedding_dim")
    mcf = float(min_contiguous_fraction)
    if not np.isfinite(mcf) or not 0.0 < mcf <= 1.0:
        raise ValueError("min_contiguous_fraction must be in (0, 1]")
    k = None
    if n_components is not None:
        k = _check_int(n_components, "n_components", 1)
        if k >= min(w - e + 1, e):
            raise ValueError(
                "n_components must be < min(window-embedding_dim+1, embedding_dim)"
            )
    return w, e, k, mcf


@register_operator(
    name="ts_hankel_effective_rank",
    category="hankel",
    business_category="hankel",
    canonical="ts_hankel_effective_rank",
    source="hankel",
)
class TsHankelEffectiveRank(SeriesOperator):
    """Hankel 有效秩：``p_i = σ_i²/Σσ²``，``ER = exp(-Σ p log p)/min(H.shape)``。

    反映奇异谱的铺开程度——时间序列的有效自由度。纯正弦 → 接近 1/秩（低）；
    白噪声 → 接近 1（高）。P2。
    """

    metadata = _metadata(
        "ts_hankel_effective_rank",
        "奇异值能量分布的熵指数 / min(H.shape)（有效自由度）。",
        ["x", "window", "embedding_dim", "min_contiguous_fraction"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 15, min_contiguous_fraction: float = 0.8, **_: Any
    ) -> pd.DataFrame:
        w, e, _, mcf = _check_hankel_params(window, embedding_dim, None, min_contiguous_fraction)
        return frame_like(x, _hankel_effective_rank_series(x.to_numpy(dtype=float), w, e, "strict_contiguous", mcf))


@register_operator(
    name="ts_hankel_singular_gap",
    category="hankel",
    business_category="hankel",
    canonical="ts_hankel_singular_gap",
    source="hankel",
)
class TsHankelSingularGap(SeriesOperator):
    """Hankel 奇异值间隙：``(σ_1 - σ_2)/(Σσ + eps)``。

    大 → 第一个模式远强于其余（强主导结构）；小 → 模式接近（噪声/多周期）。
    P2。
    """

    metadata = _metadata(
        "ts_hankel_singular_gap",
        "最大与次大奇异值的相对间隙（主导结构强度）。",
        ["x", "window", "embedding_dim", "min_contiguous_fraction"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, embedding_dim: int = 15, min_contiguous_fraction: float = 0.8, **_: Any
    ) -> pd.DataFrame:
        w, e, _, mcf = _check_hankel_params(window, embedding_dim, None, min_contiguous_fraction)
        return frame_like(x, _hankel_singular_gap_series(x.to_numpy(dtype=float), w, e, "strict_contiguous", mcf))


@register_operator(
    name="ts_ssa_reconstruction_residual",
    category="hankel",
    business_category="hankel",
    canonical="ts_ssa_reconstruction_residual",
    source="hankel",
)
class TsSsaReconstructionResidual(SeriesOperator):
    """SSA 重构残差：前 ``n_components`` 个 SVD 模式经反对角平均重构窗口，
    ``residual = mean((x - x̂)²)/(var(x) + eps)``。

    残差低 → 序列几乎被少数结构模式解释（强可预测结构）；高 → 残差大（噪声/
    非线性）。P2。
    """

    metadata = _metadata(
        "ts_ssa_reconstruction_residual",
        "前 n_components 个 SVD 模式重构窗口的归一化均方残差。",
        ["x", "window", "embedding_dim", "n_components", "min_contiguous_fraction"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 60,
        embedding_dim: int = 15,
        n_components: int = 3,
        min_contiguous_fraction: float = 0.8,
        **_: Any,
    ) -> pd.DataFrame:
        w, e, k, mcf = _check_hankel_params(window, embedding_dim, n_components, min_contiguous_fraction)
        return frame_like(x, _ssa_reconstruction_residual_series(x.to_numpy(dtype=float), w, e, k, "strict_contiguous", mcf))


_NEW_CANONICALS = (
    "ts_hankel_effective_rank",
    "ts_hankel_singular_gap",
    "ts_ssa_reconstruction_residual",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
