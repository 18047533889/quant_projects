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

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from factor_engine.cleaned_operators.closure.strict_scalar import strict_int
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    param_specs: dict | None = None,
    relational_specs: list | None = None,
) -> OperatorMetadata:
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
        param_specs=dict(param_specs) if param_specs else {},
        relational_specs=list(relational_specs) if relational_specs else [],
    )


# M-2xx: the Takens embedding resolution is an ESTIMATOR-resolution knob set —
# ``embedding_dim`` / ``k`` / ``delay`` / ``theiler_window`` are never freely
# searched economic alphas (coarse certified grid only, searchable=False).
_INTRINSIC_PARAM_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON, searchable=True),
    "embedding_dim": ParamSpec(
        dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=3
    ),
    "k": ParamSpec(
        dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=5
    ),
    "delay": ParamSpec(
        dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=1
    ),
    "theiler_window": ParamSpec(
        dtype=int, min=0,
        param_role=ParamRole.ESTIMATOR_RESOLUTION, searchable=False, default=None,
    ),
}
# M-2xx: relational feasibility — the window must be able to produce at least
# ``k + 1`` delay embeddings, i.e. ``window - (embedding_dim-1)*delay >= k+1``.
# Below that the kernel is guaranteed to emit all-NaN, so the combination is
# rejected at binding instead of running then failing.
_INTRINSIC_RELATIONAL_SPECS: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "window - (embedding_dim - 1) * delay >= k + 1",
        "ts_delay_intrinsic_dimension requires window-(embedding_dim-1)*delay "
        ">= k+1 embeddings (window={window}, embedding_dim={embedding_dim}, "
        "delay={delay}, k={k})",
    ),
]


def _delay_points(chunk: np.ndarray, dim: int, delay: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Delay-embed ``chunk`` and return ``(points, original_time_index)``.

    ``points`` are the Takens embedding vectors that survive NaN removal and
    ``original_time_index`` is the integer time position (index into ``chunk``)
    of each surviving vector's *latest* coordinate.  Keeping the original time
    positions is essential: the Theiler temporal-exclusion window must be
    evaluated on real time, not on the compressed post-NaN ordinal, otherwise a
    gap in the data makes two far-apart points look like temporal neighbours.
    """
    n = int(chunk.shape[0])
    lag = delay * (dim - 1)
    if n < lag + 1:
        return None
    pts = np.stack([chunk[s - lag : s + 1 : delay] for s in range(lag, n)], axis=0)
    orig_time = np.arange(lag, n, dtype=np.int64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    orig_time = orig_time[finite]
    if pts.shape[0] < 2:
        return None
    return pts, orig_time


def _distinct_count_scale_robust(pts: np.ndarray, rel_tol: float = 1e-8) -> int:
    """Number of genuinely distinct embedding points, robust to overall scale.

    The historic ``np.unique(pts.round(10), axis=0)`` merged points at a fixed
    absolute decimal precision, so ``x``, ``1000*x`` and ``1e-6*x`` reported
    different duplicate counts (which corrupts the ``n_unique >= k+1`` gate and
    the intrinsic-dimension estimate).  Here every embedding dimension is first
    normalised by its robust per-dimension scale (median absolute deviation,
    with a standard-deviation fallback for near-constant dimensions) and points
    are then considered duplicates when their MAD-normalised co-ordinates agree
    to a relative tolerance ``rel_tol``.  Because MAD (and the fallback)
    rescale linearly, the count is invariant to a global rescaling of the
    series.
    """
    if pts.shape[0] == 0:
        return 0
    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med), axis=0)
    scale = np.where(mad > 0.0, mad, pts.std(axis=0))
    scale = np.where(scale > 0.0, scale, 1.0)
    norm = (pts - med) / scale
    grid = np.rint(norm / rel_tol)
    return int(np.unique(grid, axis=0).shape[0])


def _delay_intrinsic_dim(chunk: np.ndarray, dim: int, k: int, delay: int, theiler_window: int) -> float:
    res = _delay_points(chunk, dim, delay)
    if res is None:
        return np.nan
    pts, orig_time = res
    n_pts = int(pts.shape[0])
    # Round-7 P0 (review §30): require *genuinely distinct* embedding points.
    # Duplicate vectors (identical values at different times) share a zero
    # nearest-neighbour distance; demanding ``n_unique >= k+1`` before the KNN
    # prevents a near-empty point cloud from manufacturing a dimension.  The
    # duplicate definition is scale-robust (per-dimension MAD normalisation), so
    # f(x), f(1000x) and f(1e-6x) all count the same distinct points.
    if _distinct_count_scale_robust(pts) < k + 1:
        return np.nan
    if n_pts < k + 1:
        return np.nan
    d = np.sqrt(np.maximum(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, np.inf)
    # Round-7 P0 (review §29): Theiler window.  Takens-embedded points that are
    # close in *time* share most coordinates and are artificially-near nearest
    # neighbours; they must not count as state-space neighbours or the local
    # dimension is biased downward.  Exclude ``|i - j| <= theiler_window`` from
    # each point's neighbour set.  The exclusion uses the *original* time
    # coordinates of the surviving vectors (not the compressed post-NaN
    # ordinal), so real-time distance is the sole criterion.  Default
    # (``embedding_dim * delay``) is the embedding span recommended by the
    # review.
    if theiler_window > 0:
        temporal = np.abs(orig_time[:, None] - orig_time[None, :]) <= theiler_window
        np.fill_diagonal(temporal, False)
        d[temporal] = np.inf
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


def _intrinsic_dim_series(x2d: np.ndarray, window: int, dim: int, k: int, delay: int, theiler_window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            out[r, c] = _delay_intrinsic_dim(col[i0 : r + 1], dim, k, delay, theiler_window)
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
        "Takens 延迟嵌入的 Levina-Bickel 局部维度中位数。"
        "参数全部 ParamSpec（embedding_dim/k/delay/theiler_window=ESTIMATOR_RESOLUTION"
        "，searchable=False）；window-(embedding_dim-1)*delay >= k+1 的关系可行"
        "性在绑定期强制（不满足 raise，不做 int() 截断）。",
        ["x", "window", "embedding_dim", "k", "delay", "theiler_window"],
        unit="dim",
        cost=8,
        param_specs=_INTRINSIC_PARAM_SPECS,
        relational_specs=_INTRINSIC_RELATIONAL_SPECS,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, embedding_dim: int = 3, k: int = 5, delay: int = 1, theiler_window: Any = None, **_: Any
    ) -> pd.DataFrame:
        # M-2xx: strict operator-boundary validation — a fractional / NaN /
        # bool / negative value is rejected outright, never ``int()``-truncated
        # into a fake window.  (The ParamSpec / relational gate already runs at
        # binding via validate_operator_call; these keep a direct
        # ``_calculate_series`` call fail-closed too.)
        w = strict_int(window, "window", lower=2)
        dim = strict_int(embedding_dim, "embedding_dim", lower=2)
        kk = strict_int(k, "k", lower=2)
        dl = strict_int(delay, "delay", lower=1)
        if w - (dim - 1) * dl < kk + 1:
            raise ValueError(
                "ts_delay_intrinsic_dimension requires window-(embedding_dim-1)*delay "
                f">= k+1 embeddings (window={w}, embedding_dim={dim}, delay={dl}, k={kk})"
            )
        # Theiler window (round-7 P0): default = embedding span ``dim * delay``;
        # a point's temporal neighbours within that span are excluded from its
        # state-space neighbour set so time-adjacency is not read as proximity.
        tw = strict_int(theiler_window, "theiler_window", lower=0) if theiler_window is not None else dim * dl
        return frame_like(x, _intrinsic_dim_series(x.to_numpy(dtype=float), w, dim, kk, dl, tw))


_NEW_CANONICALS = (
    "ts_delay_intrinsic_dimension",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
