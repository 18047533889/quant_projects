# -*- coding: utf-8 -*-
"""Confirmed-extremum divergence operators (2026-08 geometry/math expansion).

Both operators are built on a single shared *confirmed-extremum* kernel.  A bar
``k`` is a **confirmed peak** of a series when ``x_k`` is a strict maximum over
the confirmation window ``[k-confirmation, k+confirmation]`` *and* the drop from
the most recent confirmed trough exceeds ``prominence * x_k`` (troughs are
symmetric: strict minimum over the window and drop from the most recent
confirmed peak exceeds ``prominence * x_k``).  A confirmed extremum only becomes
*usable* once its confirmation window has fully passed (``age >= confirmation``),
so the kernel is strictly prefix-causal: output row ``r`` never looks past row
``r`` and only sees extrema confirmed at or before ``r``.

The family answers *whether two price-like series move together or quietly
diverge at their turning points*:

* ``ts_extrema_divergence_strength``   — how different the most recent two
  same-side extrema move in ``x`` vs the matching same-side extrema in ``y``.
* ``ts_extrema_confirmation_rate``     — what fraction of ``x``'s confirmed
  turning points are echoed by a same-side turning point in ``y``.

Both operators are trailing-window, per-column, deterministic and NaN-safe
(degenerate windows emit NaN; invalid parameters raise ``ValueError``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="extrema_divergence",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "extrema_divergence", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


# ---------------------------------------------------------------------------
# shared kernels
# ---------------------------------------------------------------------------
def _confirmed_extrema(x: np.ndarray, prominence: float, confirmation: int) -> np.ndarray:
    """Per-column confirmed-extremum detector (prefix-causal).

    Returns an int array of the same length: ``+1`` = confirmed peak,
    ``-1`` = confirmed trough, ``0`` = no confirmed extremum.  Bar ``j`` is
    decided at time ``t = j + confirmation`` (its confirmation window is
    complete), so the result at any row only uses rows ``<= t``.

    R4-45: peaks and troughs *strictly alternate*.  After a confirmed peak only a
    trough may be confirmed next (and vice-versa); a same-side candidate that is
    more extreme than the last confirmed extremum of that side *replaces* it
    instead of appending.  Two peaks are therefore never confirmed without an
    intervening trough and a monotone run is never double-counted as several
    turning points.  NaN bars are skipped: they neither break nor seed the chain.
    """
    n = len(x)
    ext = np.zeros(n, dtype=np.int8)
    last_peak = -1
    last_trough = -1
    last_side = 0  # 0 = none yet, +1 = last was a peak, -1 = last was a trough
    conf = int(confirmation)
    prom = float(prominence)
    for t in range(2 * conf, n):
        j = t - conf
        seg = x[j - conf : j + conf + 1]
        if not np.all(np.isfinite(seg)):
            continue
        xj = x[j]
        left = seg[:conf]
        right = seg[conf + 1 :]
        is_peak = bool(np.all(xj > left) and np.all(xj > right))
        is_trough = bool(np.all(xj < left) and np.all(xj < right))
        if is_peak:
            if last_side == -1:
                # previous confirmed extremum is a trough: the new peak must
                # clear a prominence rise from that trough.
                if last_trough >= 0 and xj - x[last_trough] > prom * xj:
                    ext[j] = 1
                    last_peak = j
                    last_side = 1
            elif last_side == 0:
                # seed the chain (first confirmed extremum needs no reference).
                ext[j] = 1
                last_peak = j
                last_side = 1
            elif last_peak >= 0 and xj > x[last_peak]:
                # same-side candidate more extreme: replace, never append.
                ext[last_peak] = 0
                ext[j] = 1
                last_peak = j
        if is_trough:
            if last_side == 1:
                if last_peak >= 0 and x[last_peak] - xj > prom * xj:
                    ext[j] = -1
                    last_trough = j
                    last_side = -1
            elif last_side == 0:
                ext[j] = -1
                last_trough = j
                last_side = -1
            elif last_trough >= 0 and xj < x[last_trough]:
                ext[last_trough] = 0
                ext[j] = -1
                last_trough = j
    return ext


def _local_confirmed_extrema(x: np.ndarray, prominence: float, confirmation: int) -> np.ndarray:
    """Per-column *candidate* extremum detector for the matching series ``y``.

    Like ``_confirmed_extrema`` each bar is a strict local max/min over
    ``±confirmation`` and must clear a prominence step from the most recent
    opposite extremum — but there is **no** strict alternation state machine, so
    several same-side candidates can coexist.

    R4-45: the strict alternating kernel is applied to the primary series ``x``,
    whose confirmed extrema define the turning-point chain.  For ``y`` we want
    every genuine turning point inside a small ``±match_lag`` neighbourhood of an
    x extremum, so a globally alternating chain would wrongly drop legitimate y
    matches and make the divergence sparse whenever x and y are only locally
    aligned (e.g. the operator run against an unrelated volume panel).
    """
    n = len(x)
    ext = np.zeros(n, dtype=np.int8)
    last_peak = -1
    last_trough = -1
    conf = int(confirmation)
    prom = float(prominence)
    for t in range(2 * conf, n):
        j = t - conf
        seg = x[j - conf : j + conf + 1]
        if not np.all(np.isfinite(seg)):
            continue
        xj = x[j]
        left = seg[:conf]
        right = seg[conf + 1 :]
        if np.all(xj > left) and np.all(xj > right):
            if last_trough < 0 or xj - x[last_trough] > prom * xj:
                ext[j] = 1
                last_peak = j
        elif np.all(xj < left) and np.all(xj < right):
            if last_peak < 0 or x[last_peak] - xj > prom * xj:
                ext[j] = -1
                last_trough = j
    return ext


def _side_indices(
    ext: np.ndarray, side: int, r: int, window: int, confirmation: int
) -> list[int]:
    """Indices of confirmed extrema of ``side`` known at row ``r`` and within
    the trailing window ``[r-window+1, r]`` (extremum must be confirmed, i.e.
    ``index + confirmation <= r``)."""
    lo = max(0, r - window + 1)
    hi = r - confirmation
    out: list[int] = []
    for j in range(lo, hi + 1):
        if ext[j] == side:
            out.append(j)
    return out


def _match_y(
    ext: np.ndarray, side: int, p: int, r: int, lag: int, confirmation: int
) -> int | None:
    """Nearest confirmed same-side extremum in ``y`` within ``±lag`` bars of
    ``p``, only counting y-extrema already confirmed by row ``r``.  Returns the
    index or ``None``."""
    q_lo = max(0, p - lag)
    q_hi = min(r - confirmation, p + lag)
    best = -1
    best_d: int | None = None
    for q in range(q_lo, q_hi + 1):
        if ext[q] == side:
            d = abs(q - p)
            if best_d is None or d < best_d:
                best_d = d
                best = q
    return best if best >= 0 else None


def _match_y_pair(
    ext_y: np.ndarray,
    sgn: int,
    p1: int,
    p2: int,
    r: int,
    lag: int,
    confirmation: int,
) -> tuple[int, int] | None:
    """One-to-one monotonic match of two x-extrema ``p1 < p2`` onto two distinct
    same-side y-extrema ``q1 < q2``, each within ``±lag`` of its counterpart and
    confirmed by row ``r``.

    R4-46: independent nearest-neighbour matching let both x-extrema land on the
    *same* y-extremum when ``lag`` is large (``q1 == q2`` → ``dy = 0``, a false
    divergence).  The pair minimises ``|q1-p1| + |q2-p2|`` subject to ``q1 < q2``.
    """
    n = len(ext_y)
    lo1 = max(0, p1 - lag)
    hi1 = min(n - 1, p1 + lag)
    lo2 = max(0, p2 - lag)
    hi2 = min(n - 1, p2 + lag)
    cand1 = [q for q in range(lo1, hi1 + 1) if ext_y[q] == sgn and q + confirmation <= r]
    cand2 = [q for q in range(lo2, hi2 + 1) if ext_y[q] == sgn and q + confirmation <= r]
    best: tuple[int, int] | None = None
    best_cost: int | None = None
    for q1 in cand1:
        for q2 in cand2:
            if q1 >= q2:
                continue
            cost = abs(q1 - p1) + abs(q2 - p2)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (q1, q2)
    return best


def _divergence_series(
    x2d: np.ndarray,
    y2d: np.ndarray,
    window: int,
    prominence: float,
    confirmation: int,
    match_lag: int,
    side: str,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    conf = int(confirmation)
    lag = int(match_lag)
    sgn = 1 if side == "peak" else -1
    for c in range(cols):
        x, y = x2d[:, c], y2d[:, c]
        ext_x = _confirmed_extrema(x, prominence, conf)
        # R4-45: y uses the candidate detector (local confirmed extrema), so the
        # matching set is not thinned by the x chain's strict alternation.
        ext_y = _local_confirmed_extrema(y, prominence, conf)
        for r in range(rows):
            idx = _side_indices(ext_x, sgn, r, w, conf)
            if len(idx) < 2:
                continue
            p1, p2 = idx[-2], idx[-1]
            i0 = max(0, r - w + 1)
            sx = float(np.nanstd(x[i0 : r + 1]))
            sy = float(np.nanstd(y[i0 : r + 1]))
            if not (np.isfinite(sx) and sx > 0 and np.isfinite(sy) and sy > 0):
                continue
            q_pair = _match_y_pair(ext_y, sgn, p1, p2, r, lag, conf)
            if q_pair is None:
                continue
            q1, q2 = q_pair
            dx = (x[p2] - x[p1]) / sx
            dy = (y[q2] - y[q1]) / sy
            out[r, c] = (dx - dy) / (abs(dx) + abs(dy) + _EPS)
    return out


def _confirmation_rate_series(
    x2d: np.ndarray,
    y2d: np.ndarray,
    window: int,
    prominence: float,
    confirmation: int,
    tolerance: int,
    side: str,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    conf = int(confirmation)
    tol = int(tolerance)
    sgn = 1 if side == "peak" else -1
    for c in range(cols):
        x, y = x2d[:, c], y2d[:, c]
        ext_x = _confirmed_extrema(x, prominence, conf)
        # R4-45: y uses the candidate detector for matching (see above).
        ext_y = _local_confirmed_extrema(y, prominence, conf)
        for r in range(rows):
            idx = _side_indices(ext_x, sgn, r, w, conf)
            if not idx:
                continue
            matched = sum(1 for p in idx if _match_y(ext_y, sgn, p, r, tol, conf) is not None)
            out[r, c] = matched / len(idx)
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_extrema_divergence_strength",
    category="extrema_divergence",
    business_category="extrema_divergence",
    canonical="ts_extrema_divergence_strength",
    source="extrema_divergence",
)
class TsExtremaDivergenceStrength(SeriesOperator):
    """最近两个已确认同侧极值, x/y 归一化变动幅度之间的分歧强度。

    对 x 取窗口内最近两个已确认峰(或谷), 在 y 上按 ±match_lag 匹配同侧已确认
    极值; Δ_x=(x_p2-x_p1)/std_x, Δ_y=(y_q2-y_q1)/std_y,
    Divergence=(Δ_x-Δ_y)/(|Δ_x|+|Δ_y|+eps)。 正 → x 比 y 走得更强;
    负 → y 领先/更强。 极值或匹配不足时输出 NaN。 P1。
    """

    metadata = _metadata(
        "ts_extrema_divergence_strength",
        "最近两个已确认同侧极值在 x/y 上的归一化变动分歧强度。",
        ["x", "y", "window", "prominence", "confirmation", "match_lag", "side"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int = 60,
        prominence: float = 0.02,
        confirmation: int = 3,
        match_lag: int = 4,
        side: str = "peak",
        **_: Any,
    ) -> pd.DataFrame:
        if side not in ("peak", "trough"):
            raise ValueError("side must be 'peak' or 'trough'")
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        prom = float(prominence)
        if not prom > 0:
            raise ValueError("prominence must be > 0")
        conf = int(confirmation)
        if conf < 1:
            raise ValueError("confirmation must be >= 1")
        lag = int(match_lag)
        if lag < 0:
            raise ValueError("match_lag must be >= 0")
        return frame_like(
            x,
            _divergence_series(
                x.to_numpy(dtype=float), y.to_numpy(dtype=float), w, prom, conf, lag, side
            ),
        )


@register_operator(
    name="ts_extrema_confirmation_rate",
    category="extrema_divergence",
    business_category="extrema_divergence",
    canonical="ts_extrema_confirmation_rate",
    source="extrema_divergence",
)
class TsExtremaConfirmationRate(SeriesOperator):
    """窗口内 x 的已确认同侧极值中, 在 y 上 ±tolerance 内存在同侧已确认极值的占比。

    高 → 两个序列在同一转折点上相互确认(共振); 低 → x 的转折点多在 y 中
    找不到对应, 结构分歧。 无已确认 x 极值时输出 NaN。 P2。
    """

    metadata = _metadata(
        "ts_extrema_confirmation_rate",
        "x 的已确认同侧极值在 y 上得到同侧确认的比例。",
        ["x", "y", "window", "prominence", "confirmation", "tolerance", "side"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int = 120,
        prominence: float = 0.02,
        confirmation: int = 3,
        tolerance: int = 3,
        side: str = "peak",
        **_: Any,
    ) -> pd.DataFrame:
        if side not in ("peak", "trough"):
            raise ValueError("side must be 'peak' or 'trough'")
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        prom = float(prominence)
        if not prom > 0:
            raise ValueError("prominence must be > 0")
        conf = int(confirmation)
        if conf < 1:
            raise ValueError("confirmation must be >= 1")
        tol = int(tolerance)
        if tol < 0:
            raise ValueError("tolerance must be >= 0")
        return frame_like(
            x,
            _confirmation_rate_series(
                x.to_numpy(dtype=float), y.to_numpy(dtype=float), w, prom, conf, tol, side
            ),
        )


_NEW_CANONICALS = (
    "ts_extrema_divergence_strength",
    "ts_extrema_confirmation_rate",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
