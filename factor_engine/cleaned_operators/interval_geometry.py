# -*- coding: utf-8 -*-
"""K-line interval geometry operators (2026-08 geometry/math expansion).

A candle is treated as a *price interval* ``I_t = [low_t, high_t]``, so a run of
candles is a moving, nesting, intersecting, breaking chain of intervals.  This
module characterises that chain *as a set of intervals*, which is orthogonal to
single-candle geometry (body/wick ratios) and to range stats (ATR):

* ``ts_interval_union_coverage``          — what fraction of the price envelope
  was actually "walked through" (uncovered price gaps show up as a low ratio).
* ``ts_interval_occupancy_entropy``       — whether trading concentrated in a
  few price bands or spread evenly across the envelope.
* ``ts_interval_occupancy_mode_distance`` — distance of a state ``x`` from the
  modal price band of the trailing interval mass.
* ``ts_interval_nesting_depth``           — consecutive inside/outside-bar depth
  (stateful recursion; CTA compression / expansion gauge).
* ``ts_interval_exploration_efficiency``  — net span explored per unit of true
  range travelled (interval version of path efficiency).
* ``ts_interval_overlap_component_ratio`` — fraction of bars belonging to the
  largest mutually-overlapping price structure.

All operators are trailing-window, prefix-causal and deterministic.  NaN inputs
are dropped from the window; a window with no valid pairs emits NaN.
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
        category="interval_geometry",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "interval_geometry", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_geometry",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _valid_pairs(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(lo) & np.isfinite(hi)
    # Inverted candles (low > high) are invalid interval data, not a price
    # state: a window made of them must emit NaN, and a mixed window must not
    # let an inverted pair corrupt the envelope (review P0: interval inputs).
    m &= lo <= hi
    return lo[m].astype(float), hi[m].astype(float)


def _union_length(l: np.ndarray, h: np.ndarray) -> float:
    """Measure of the union of sorted-by-start intervals ``[l_i, h_i]``."""
    if l.size == 0:
        return np.nan
    order = np.argsort(l, kind="stable")
    l, h = l[order], h[order]
    total, cur_l, cur_h = 0.0, l[0], h[0]
    for i in range(1, l.size):
        if l[i] <= cur_h:
            cur_h = max(cur_h, h[i])
        else:
            total += cur_h - cur_l
            cur_l, cur_h = l[i], h[i]
    total += cur_h - cur_l
    return float(total)


def _occupancy_profile(l: np.ndarray, h: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Mass-per-bin profile of the interval coverage over ``[min L, max H]``.

    Each bar contributes the length of its overlap with each bin, so a bar
    spanning several bins spreads its mass across them.  Returns (profile, edges)
    or None when degenerate.
    """
    lo0, hi0 = float(l.min()), float(h.max())
    if not (np.isfinite(lo0) and np.isfinite(hi0)) or hi0 - lo0 <= 0:
        return None
    edges = np.linspace(lo0, hi0, int(bins) + 1)
    mass = np.zeros(int(bins), dtype=float)
    for a, b in zip(l, h):
        lb = np.maximum(edges[:-1], a)
        ub = np.minimum(edges[1:], b)
        mass += np.maximum(0.0, ub - lb)
    total = mass.sum()
    if total <= 0 or not np.isfinite(total):
        return None
    return mass / total, edges


# ---------------------------------------------------------------------------
# kernels (one per operator; 2D panel = TradeDate x Symbol)
# ---------------------------------------------------------------------------
def _union_coverage_series(lo2d: np.ndarray, hi2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = lo2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        lo, hi = lo2d[:, c], hi2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            l, h = _valid_pairs(lo[i0 : r + 1], hi[i0 : r + 1])
            if l.size == 0:
                continue
            env = float(h.max() - l.min())
            if env <= 0:
                continue
            u = _union_length(l, h)
            out[r, c] = u / (env + _EPS)
    return out


def _occupancy_series(lo2d: np.ndarray, hi2d: np.ndarray, window: int, bins: int, mode: str) -> np.ndarray:
    rows, cols = lo2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(bins)
    for c in range(cols):
        lo, hi = lo2d[:, c], hi2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            l, h = _valid_pairs(lo[i0 : r + 1], hi[i0 : r + 1])
            prof = _occupancy_profile(l, h, b) if l.size else None
            if prof is None:
                continue
            p, edges = prof
            if mode == "entropy":
                ent = -float(np.sum(p * np.log(np.clip(p, _EPS, 1.0))))
                out[r, c] = ent / np.log(b) if b > 1 else np.nan
            else:  # mode_distance — modal band centre
                modal = int(np.argmax(p))
                out[r, c] = float((edges[modal] + edges[modal + 1]) / 2.0)
    return out


def _nesting_depth_series(lo2d: np.ndarray, hi2d: np.ndarray, mode: str) -> np.ndarray:
    """Forward per-column recursion: consecutive inside/outside bars stack up."""
    rows, cols = lo2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    depth = np.zeros(cols, dtype=float)
    for r in range(rows):
        for c in range(cols):
            lo, hi = lo2d[r, c], hi2d[r, c]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                depth[c] = 0.0
                out[r, c] = np.nan
                continue
            if r == 0:
                # No previous bar: the nesting state is unknown, not 0 (R4-07).
                depth[c] = 0.0
                continue
            plo, phi = lo2d[r - 1, c], hi2d[r - 1, c]
            if not (np.isfinite(plo) and np.isfinite(phi)):
                # Previous bar missing -> cannot tell whether nesting continues;
                # emit NaN (unknown) and reset the chain (R4-07).  A fresh
                # depth=0 at this row would silently claim a non-nesting break.
                depth[c] = 0.0
                continue
            if mode == "inside":
                cont = bool(hi <= phi + _EPS) and bool(lo >= plo - _EPS)
            else:  # outside
                cont = bool(hi >= phi - _EPS) and bool(lo <= plo + _EPS)
            depth[c] = depth[c] + 1.0 if cont else 0.0
            out[r, c] = depth[c]
    return out


def _exploration_efficiency_series(hi2d: np.ndarray, lo2d: np.ndarray, cl2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = hi2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        hi, lo, cl = hi2d[:, c], lo2d[:, c], cl2d[:, c]
        prev_cl = np.concatenate([[np.nan], cl[:-1]])
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
        # A missing previous close makes the |high-low|-extension to the prior
        # close genuinely unknown; ``np.maximum`` would collapse that row to
        # ``high-low`` (a lower bound), silently hiding the gap.  Mark the row
        # NaN so the gap breaks the travel path instead of undercounting it.
        tr = np.where(np.isfinite(prev_cl), tr, np.nan)
        for r in range(rows):
            i0 = max(0, r - w + 1)
            # ``_valid_pairs`` returns (low, high); a swapped assignment made
            # ``span = max(low) - min(high)`` instead of ``max(high) - min(low)``
            # (review P0: interval-exploration variable reversal).
            l, h = _valid_pairs(lo[i0 : r + 1], hi[i0 : r + 1])
            if l.size == 0:
                continue
            span = float(h.max() - l.min())
            # Gap handling: a missing previous close makes that row's TR
            # undefined.  Counting it as zero (``np.nansum``) would pretend an
            # unobserved price step has no path cost, overstating efficiency;
            # the gap breaks the path, so travel is only the *trailing
            # contiguous* run of known TRs (review P0: interval path cost).
            seg_tr = tr[i0 : r + 1]
            j = len(seg_tr) - 1
            while j >= 0 and np.isfinite(seg_tr[j]):
                j -= 1
            travel = float(np.sum(seg_tr[j + 1 :]))
            if not np.isfinite(travel) or travel <= 0:
                continue
            out[r, c] = span / (travel + _EPS)
    return out


def _overlap_component_ratio_series(lo2d: np.ndarray, hi2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = lo2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        lo, hi = lo2d[:, c], hi2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            l, h = _valid_pairs(lo[i0 : r + 1], hi[i0 : r + 1])
            n = l.size
            if n == 0:
                continue
            # union-find: overlap iff l_i <= h_j and l_j <= h_i
            parent = list(range(n))
            size = [1] * n
            def _find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for i in range(n):
                for j in range(i + 1, n):
                    if l[i] <= h[j] and l[j] <= h[i]:
                        ri, rj = _find(i), _find(j)
                        if ri != rj:
                            parent[ri] = rj
                            size[rj] += size[ri]
            out[r, c] = float(max(size)) / n
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="ts_interval_union_coverage",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_union_coverage",
    source="interval_geometry",
)
class TsIntervalUnionCoverage(SeriesOperator):
    """区间并集覆盖率：过去 window 根 K 线并集长度 / 价格包络。

    接近 1 → 价格空间被连续走过（结构连续）；很低 → 存在大量未覆盖区域 /
    跳跃断层。与 ATR、range、overlap ratio 均不同。P1。
    """

    metadata = _metadata(
        "ts_interval_union_coverage",
        "K 线区间并集长度 / 价格包络（断层与覆盖程度）。",
        ["low", "high", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, low: pd.DataFrame, high: pd.DataFrame, window: int = 20, **_: Any
    ) -> pd.DataFrame:
        return frame_like(low, _union_coverage_series(low.to_numpy(dtype=float), high.to_numpy(dtype=float), window))


@register_operator(
    name="ts_interval_occupancy_entropy",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_occupancy_entropy",
    source="interval_geometry",
)
class TsIntervalOccupancyEntropy(SeriesOperator):
    """价格区间占用熵：价格在包络内集中访问少数价格带还是广泛铺开。

    低 → 明确价格中枢；高 → 价格活动范围分散。P1。
    """

    metadata = _metadata(
        "ts_interval_occupancy_entropy",
        "区间覆盖各 bin 质量的归一化熵（集中度/铺开度）。",
        ["low", "high", "window", "bins"],
        unit="entropy",
        cost=3,
    )

    def _calculate_series(
        self, low: pd.DataFrame, high: pd.DataFrame, window: int = 20, bins: int = 8, **_: Any
    ) -> pd.DataFrame:
        return frame_like(low, _occupancy_series(low.to_numpy(dtype=float), high.to_numpy(dtype=float), window, bins, "entropy"))


@register_operator(
    name="ts_interval_occupancy_mode_distance",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_occupancy_mode_distance",
    source="interval_geometry",
)
class TsIntervalOccupancyModeDistance(SeriesOperator):
    """状态 x 与历史区间占用众数价格带的距离（按包络归一）。

    x 可为 close / VWAP / 估值等；不需要成交量，也不同于筹码成本。P1。
    """

    metadata = _metadata(
        "ts_interval_occupancy_mode_distance",
        "x 相对历史区间占用众数带中心的归一化距离。",
        ["x", "low", "high", "window", "bins"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, low: pd.DataFrame, high: pd.DataFrame, window: int = 20, bins: int = 8, **_: Any
    ) -> pd.DataFrame:
        x2, lo2, hi2 = (a.to_numpy(dtype=float) for a in (x, low, high))
        rows, cols = x2.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        w, b = int(window), int(bins)
        for c in range(cols):
            for r in range(rows):
                i0 = max(0, r - w + 1)
                l, h = _valid_pairs(lo2[i0 : r + 1, c], hi2[i0 : r + 1, c])
                prof = _occupancy_profile(l, h, b) if l.size else None
                if prof is None:
                    continue
                p, edges = prof
                modal = int(np.argmax(p))
                centre = (edges[modal] + edges[modal + 1]) / 2.0
                span = edges[-1] - edges[0]
                out[r, c] = (x2[r, c] - centre) / (span + _EPS)
        return frame_like(x, out)


@register_operator(
    name="ts_interval_nesting_depth",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_nesting_depth",
    source="interval_geometry",
)
class TsIntervalNestingDepth(SeriesOperator):
    """连续 inside/outside bar 深度（stateful 递归）。

    inside 模式: ``high<=prev_high and low>=prev_low`` 则深度 +1,否则归零;
    outside 模式反向。输出 0,1,2,3,... 直接表达压缩/扩张已经持续几层。P1。
    首根 bar 或上一 bar 缺失时输出 NaN(无法判断是否延续 nesting,R4-07)。
    """

    metadata = _metadata(
        "ts_interval_nesting_depth",
        "连续 inside/outside bar 的嵌套深度（stateful）。",
        ["low", "high", "mode"],
        unit="bars",
        cost=1,
    )

    def _calculate_series(
        self, low: pd.DataFrame, high: pd.DataFrame, mode: str = "inside", **_: Any
    ) -> pd.DataFrame:
        if mode not in ("inside", "outside"):
            raise ValueError("mode must be 'inside' or 'outside'")
        return frame_like(low, _nesting_depth_series(low.to_numpy(dtype=float), high.to_numpy(dtype=float), mode))


@register_operator(
    name="ts_interval_exploration_efficiency",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_exploration_efficiency",
    source="interval_geometry",
)
class TsIntervalExplorationEfficiency(SeriesOperator):
    """区间版路径效率：探索出的价格宽度 / 付出的真实路径。

    高 → 每单位波动都有效探索新价格区域（directional）；低 → chop。P1。
    """

    metadata = _metadata(
        "ts_interval_exploration_efficiency",
        "价格包络跨度 / 窗口真实波幅总和（区间版 path efficiency）。",
        ["high", "low", "close", "window"],
        unit="ratio",
        cost=2,
    )

    def _calculate_series(
        self, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 20, **_: Any
    ) -> pd.DataFrame:
        return frame_like(high, _exploration_efficiency_series(
            high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float), window))


@register_operator(
    name="ts_interval_overlap_component_ratio",
    category="interval_geometry",
    business_category="interval_geometry",
    canonical="ts_interval_overlap_component_ratio",
    source="interval_geometry",
)
class TsIntervalOverlapComponentRatio(SeriesOperator):
    """最大重叠区间连通分量占比。

    高 → 最近行情属于同一个连续价格结构；低 → 分裂成多个互不重叠的价格区域
    （跳空/regime shift/价格重心迁移敏感）。P2。
    """

    metadata = _metadata(
        "ts_interval_overlap_component_ratio",
        "最大重叠连通分量 / 窗口 K 线数（价格结构连续 vs 分裂）。",
        ["low", "high", "window"],
        unit="ratio",
        cost=4,
    )

    def _calculate_series(
        self, low: pd.DataFrame, high: pd.DataFrame, window: int = 20, **_: Any
    ) -> pd.DataFrame:
        return frame_like(low, _overlap_component_ratio_series(low.to_numpy(dtype=float), high.to_numpy(dtype=float), window))


_NEW_CANONICALS = (
    "ts_interval_union_coverage",
    "ts_interval_occupancy_entropy",
    "ts_interval_occupancy_mode_distance",
    "ts_interval_nesting_depth",
    "ts_interval_exploration_efficiency",
    "ts_interval_overlap_component_ratio",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | set(_NEW_CANONICALS)
    )
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()
