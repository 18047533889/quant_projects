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
# B06: endpoint feasibility must include the Theiler exclusion. Requiring the
# endpoint to have ``k`` candidates is the necessary boundary; demanding that
# every interior anchor have ``k`` candidates would incorrectly reject windows
# that still contain usable endpoint anchors.
_INTRINSIC_RELATIONAL_SPECS: list[RelationalParamSpec] = [
    RelationalParamSpec(
        "window - (embedding_dim - 1) * delay >= k + "
        "(embedding_dim * delay if theiler_window is None else theiler_window) + 1",
        "ts_delay_intrinsic_dimension requires N_embed >= k + effective_theiler + 1 "
        "(window={window}, embedding_dim={embedding_dim}, delay={delay}, k={k}, "
        "theiler_window={theiler_window})",
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
    the intrinsic-dimension estimate).  Here the cloud is centered with a
    finite midrange and divided by one bounded common scale before quantizing
    at ``rel_tol``.  This avoids overflow in variance-based fallbacks, preserves
    the Euclidean geometry used by the estimator, and makes the count invariant
    to a finite nonzero global rescaling of the series.
    """
    if pts.shape[0] == 0:
        return 0
    norm = _common_scale_points(pts)
    if norm is None:
        return 1
    grid = np.rint(norm / rel_tol)
    return int(np.unique(grid, axis=0).shape[0])


def _finite_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Finite binary64 midpoint without same-sign overflow."""
    direct = (
        (np.signbit(a) != np.signbit(b))
        | ((np.abs(a) <= np.finfo(float).max / 2.0)
           & (np.abs(b) <= np.finfo(float).max / 2.0))
    )
    out = np.empty_like(a, dtype=float)
    out[direct] = (a[direct] + b[direct]) / 2.0
    out[~direct] = a[~direct] / 2.0 + b[~direct] / 2.0
    return out


def _common_scale_points(pts: np.ndarray) -> np.ndarray | None:
    """Center with a stable midrange and apply one scalar Euclidean scale."""
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    center = _finite_midpoint(lo, hi)
    shifted = pts - center
    scale = float(np.max(np.abs(shifted)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    normalized = shifted / scale
    return normalized if np.isfinite(normalized).all() else None


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
    # One common scalar preserves the Euclidean metric. Stable midrange
    # centering also preserves it under translations and avoids letting a large
    # level offset erase representable differences during normalization.
    metric_pts = _common_scale_points(pts)
    if metric_pts is None:
        return np.nan
    # Round-7 P0 (review §29): Theiler window.  Takens-embedded points that are
    # close in *time* share most coordinates and are artificially-near nearest
    # neighbours; they must not count as state-space neighbours or the local
    # dimension is biased downward.  Exclude ``|i - j| <= theiler_window`` from
    # each point's neighbour set.  The exclusion uses the *original* time
    # coordinates of the surviving vectors (not the compressed post-NaN
    # ordinal), so real-time distance is the sole criterion.  Default
    # (``embedding_dim * delay``) is the embedding span recommended by the
    # review.
    # Keep one anchor's distances plus its top-k selection in memory. This
    # avoids the old unbudgeted N×N×dim broadcast while preserving original-
    # time Theiler exclusion and the exact common-scale Euclidean metric.
    local_dims: list[float] = []
    for anchor in range(n_pts):
        candidates = np.flatnonzero(
            np.abs(orig_time - orig_time[anchor]) > theiler_window
        )
        if candidates.size < k:
            continue
        delta = metric_pts[candidates] - metric_pts[anchor]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        distances = distances[np.isfinite(distances) & (distances > 0.0)]
        if distances.size < k:
            continue
        nearest = np.partition(distances, k - 1)[:k]
        nearest.sort()
        log_term = float(
            np.sum(np.log(nearest[-1]) - np.log(nearest[:-1]))
        )
        if np.isfinite(log_term) and log_term > 0.0:
            estimate = (k - 1) / log_term
            if np.isfinite(estimate) and estimate > 0.0:
                local_dims.append(float(estimate))
    dims = np.asarray(local_dims, dtype=float)
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
        "，searchable=False）；N_embed >= k + effective_theiler + 1 的关系可行"
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
        n_embed = w - (dim - 1) * dl
        if n_embed < kk + tw + 1:
            raise ValueError(
                "ts_delay_intrinsic_dimension requires N_embed >= "
                "k + effective_theiler + 1 "
                f"(N_embed={n_embed}, k={kk}, effective_theiler={tw})"
            )
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
