# -*- coding: utf-8 -*-
"""Intrinsic-dimension operator (2026-08 geometry/math expansion).

``ts_delay_intrinsic_dimension`` embeds the trailing window into a
``(embedding_dim, delay)`` Takens delay space, builds the pairwise distance
matrix, and for every embedded point takes its ``k`` nearest neighbours
(excluding the point itself) with distances ``T_1 <= ... <= T_k``.  The
Levina & Bickel (2004) local dimension estimate for that point is

    d_hat = [ (1/(k-1)) * sum_{j=1}^{k-1} log(T_k / T_j) ]^{-1},

and the operator outputs the *median* of ``d_hat`` over the points of the
window.  Low values indicate the path is concentrated on a low-dimensional
attractor (e.g. a near-deterministic cycle); high values indicate
high-dimensional noise.

The operator is trailing-window, prefix-causal and deterministic.  Points with
a zero nearest-neighbour distance (duplicate embedding vectors) are dropped
from the median; a window that cannot produce at least ``k+1`` distinct
embedding points emits NaN.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intrinsic_dimension",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intrinsic_dimension", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _delay_points(chunk: np.ndarray, dim: int, delay: int) -> np.ndarray | None:
    n = int(chunk.shape[0])
    lag = delay * (dim - 1)
    if n < lag + 1:
        return None
    pts = np.stack([chunk[s - lag : s + 1 : delay] for s in range(lag, n)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        return None
    return pts


def _delay_intrinsic_dim(chunk: np.ndarray, dim: int, k: int, delay: int) -> float:
    pts = _delay_points(chunk, dim, delay)
    if pts is None:
        return np.nan
    n_pts = int(pts.shape[0])
    if n_pts < k + 1:
        return np.nan
    d = np.sqrt(np.maximum(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, np.inf)
    d_sorted = np.sort(d, axis=1)[:, :k]          # T_1..T_k per point, ascending
    t_k = d_sorted[:, -1]                          # T_k
    t_j = d_sorted[:, :-1]                         # T_1..T_{k-1}
    valid = (
        (t_k > 0.0)
        & (t_j[:, 0] > 0.0)
        & np.isfinite(t_k)
        & np.isfinite(t_j).all(axis=1)
    )
    if not valid.any():
        return np.nan
    t_k_v = t_k[valid]
    t_j_v = t_j[valid]
    log_term = np.sum(np.log(t_k_v[:, None] / t_j_v), axis=1)
    dims = (k - 1) / log_term
    dims = dims[np.isfinite(dims) & (dims > 0.0)]
    if dims.size == 0:
        return np.nan
    return float(np.median(dims))


def _intrinsic_dim_series(x2d: np.ndarray, window: int, dim: int, k: int, delay: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _delay_intrinsic_dim(col[i0 : r + 1], dim, k, delay)
    return out


@register_operator(
    name="ts_delay_intrinsic_dimension",
    category="intrinsic_dimension",
    business_category="intrinsic_dimension",
    canonical="ts_delay_intrinsic_dimension",
    source="intrinsic_dimension",
)
class TsDelayIntrinsicDimension(SeriesOperator):
    """延迟嵌入点云的 Levina-Bickel 局部维度中位数。

    低 -> 价格路径集中在低维吸引子上（周期/趋势主导）；高 -> 高维噪声。
    输出中位数对异常点稳健。P2。
    """

    metadata = _metadata(
        "ts_delay_intrinsic_dimension",
        "Takens 延迟嵌入的 Levina-Bickel 局部维度中位数。",
        ["x", "window", "embedding_dim", "k", "delay"],
        unit="dim",
        cost=8,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, embedding_dim: int = 3, k: int = 5, delay: int = 1, **_: Any
    ) -> pd.DataFrame:
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        dim = int(embedding_dim)
        if dim < 2:
            raise ValueError("embedding_dim must be >= 2")
        kk = int(k)
        if kk < 2:
            raise ValueError("k must be >= 2")
        dl = int(delay)
        if dl < 1:
            raise ValueError("delay must be >= 1")
        return frame_like(x, _intrinsic_dim_series(x.to_numpy(dtype=float), w, dim, kk, dl))


_NEW_CANONICALS = (
    "ts_delay_intrinsic_dimension",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
