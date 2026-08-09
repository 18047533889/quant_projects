# -*- coding: utf-8 -*-
"""Cross-sectional rotation / style-stability primitives (2026-08 CTA pack).

* ``cs_rank_churn``            — per-day mean absolute change of cross-sectional
  percentile ranks over ``lag`` days (within an optional group), broadcast back
  to each stock.  High = the cross-section is reshuffling; low = stable
  leadership.  This is the **intersection-cohort** rank churn: it is averaged
  over stocks present (and in the same group) at BOTH dates.  Rank changes are
  decomposed (R11 #153) into:
    * ``cs_rank_churn``                 — same-stock rank moves inside the
      intersection cohort;
    * ``cs_rank_composition_churn``     — pool membership turnover
      ``|A_t Δ A_{t-lag}| / |A_t ∪ A_{t-lag}|``;
    * ``cs_rank_combined_churn``        — the sum of the two (total reshuffling).
* ``cs_tail_retention``        — overlap of the top/bottom tail membership today
  vs ``lag`` days ago (within group), broadcast back.  The tail is an **exact
  top-k** tail (``k = ceil(q·N)``) with fractional mass on boundary ties, and
  the cohort is explicit (``current`` / ``historical`` / ``intersection``)
  (R11 #154/#155/#156).  ``q`` must be a real tail: ``0 < q <= 0.5``.
* ``cs_tail_breadth``          — the effective tail breadth (``Σ`` fractional
  membership ≈ ``ceil(q·N)``) that the retention ratio normalises by (R11 #156).

Group labels follow the engine convention: a panel of object dtype whose
non-null entries are group labels (one fixed ``IndustrySource`` per use).
Output is always a panel with the input index/columns.

Role (R11 #157): every operator here produces a value that is IDENTICAL for all
members of the same group/date (broadcast back).  They are ``group_state``
regime/rotation descriptors, NOT per-stock cross-sectional alpha — the search
grammar must not emit them as a terminal per-stock factor.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like
from cleaned_operators.stateful._common import metadata

_EPS = 1e-12


def _valid_label(v: Any) -> bool:
    if v is None:
        return False
    try:
        if isinstance(v, float) and np.isnan(v):
            return False
    except TypeError:
        pass
    return not (isinstance(v, str) and not v)


def _group_pct_ranks(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank of each finite value within its group."""
    n = len(values)
    out = np.full(n, np.nan)
    labels = {v for v in group if _valid_label(v)}
    for lab in labels:
        mask = np.array([g == lab for g in group], dtype=bool)
        idx = np.flatnonzero(mask & np.isfinite(values))
        if idx.size == 0:
            continue
        if idx.size == 1:
            out[idx[0]] = 0.5
            continue
        vals = values[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(idx.size, dtype=float)
        start = 0
        prev = vals[order[0]]
        for i in range(1, idx.size + 1):
            if i == idx.size or vals[order[i]] != prev:
                avg = (start + i - 1) / 2.0
                for j in range(start, i):
                    ranks[order[j]] = avg
                start = i
                prev = vals[order[i]] if i < idx.size else prev
        out[idx] = ranks / (idx.size - 1)
    return out


def _tail_quantile(q: float, name: str) -> float:
    """Validate a tail quantile: ``0 < q <= 0.5`` (R11 #155)."""
    q = float(q)
    if not (0.0 < q <= 0.5):
        raise ValueError(
            f"{name}: quantile must be a real tail in (0, 0.5], got {q} "
            "(R11 #155 — q>0.5 is not a tail)"
        )
    return q


def _exact_tail_weights(values: np.ndarray, q: float, top: bool) -> np.ndarray:
    """Exact top/bottom-k fractional tail membership for one row (R11 #156).

    ``k = ceil(q·N)`` over the finite observations; boundary ties receive a
    fractional mass so the effective tail breadth is exactly ``k``.  A stock
    with a missing value never enters the tail (unknown is not a tail member).
    """
    n = len(values)
    out = np.zeros(n, dtype=float)
    finite = np.isfinite(values)
    m = int(finite.sum())
    if m == 0:
        return out
    k = int(math.ceil(q * m))
    if k <= 0:
        return out
    if k >= m:
        out[finite] = 1.0
        return out
    vals = values[finite]
    order = np.argsort(vals, kind="mergesort")
    pos = np.flatnonzero(finite)
    if top:
        boundary_idx = m - k
        bval = vals[order[boundary_idx]]
        above = int(np.sum(vals > bval))
        at = int(np.sum(vals == bval))
        frac = (k - above) / at if at else 0.0
    else:
        boundary_idx = k - 1
        bval = vals[order[boundary_idx]]
        below = int(np.sum(vals < bval))
        at = int(np.sum(vals == bval))
        frac = (k - below) / at if at else 0.0
    for j, vi in zip(pos, vals):
        if top:
            if vi > bval:
                out[j] = 1.0
            elif vi == bval:
                out[j] = frac
        else:
            if vi < bval:
                out[j] = 1.0
            elif vi == bval:
                out[j] = frac
    return out


def _cohort_denominator(w_prev, w_cur, observable_now, cohort):
    """Denominator for tail retention under the declared cohort (R11 #154).

    * ``intersection``: previous-tail members that are observable today (a stock
      missing today is "cannot judge", not "exited the tail" — P1-22);
    * ``historical``: the full previous tail (data gaps read as rotation — only
      when explicitly requested);
    * ``current``: today's tail.
    """
    if cohort == "intersection":
        return float(np.sum(np.where(observable_now, w_prev, 0.0)))
    if cohort == "historical":
        return float(np.sum(w_prev))
    return float(np.sum(w_cur))


@register_operator(
    name="cs_rank_churn",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_rank_churn",
    source="stateful.rotation",
)
class CsRankChurn(SeriesOperator):
    """Per-day mean absolute cross-sectional rank change over ``lag`` days.

    Rank is the within-group percentile rank at each date; the churn for a
    group on date ``t`` is the mean ``|rank_t - rank_{t-lag}|`` over stocks
    present at both dates (rank change is already in [0,1]).  Broadcast to the
    group's stocks on ``t``.  NaN where ``lag`` has no history.

    R11 #153: this is the **intersection-cohort** (same-stock rank move)
    component of rotation; pool membership turnover is
    ``cs_rank_composition_churn`` and the total is ``cs_rank_combined_churn``.
    """

    metadata = metadata(
        "cs_rank_churn",
        "横截面排序洗牌度(交集同股秩变化): 日均|秩变化|, 广播回股票。",
        ["x", "lag", "group"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )
    metadata.role = "group_state"
    metadata.tags = list(metadata.tags) + ["group_state"]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 5,
        group: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        lk = max(1, int(lag))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        rank_panel = np.full_like(xv, np.nan, dtype=float)
        if group is not None and isinstance(group, pd.DataFrame):
            gv = group.to_numpy(dtype=object)
        else:
            gv = None
        for r in range(rows):
            g_row = gv[r] if gv is not None else None
            if g_row is None:
                vals = xv[r]
                valid = np.isfinite(vals)
                if valid.sum() < 2:
                    continue
                order = np.argsort(vals[valid], kind="mergesort")
                ranks = np.empty(valid.sum(), dtype=float)
                arr = vals[valid]
                start = 0
                prev = arr[order[0]]
                for i in range(1, arr.size + 1):
                    if i == arr.size or arr[order[i]] != prev:
                        avg = (start + i - 1) / 2.0
                        for j in range(start, i):
                            ranks[order[j]] = avg
                        start = i
                        prev = arr[order[i]] if i < arr.size else prev
                pct = ranks / (arr.size - 1)
                pos = np.flatnonzero(valid)
                rank_panel[r, pos] = pct
            else:
                rank_panel[r] = _group_pct_ranks(xv[r], g_row)

        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            if r < lk:
                continue
            g_row = gv[r] if gv is not None else None
            if g_row is None:
                cur = rank_panel[r]
                prev_r = rank_panel[r - lk]
                matched = np.isfinite(cur) & np.isfinite(prev_r)
                if not np.any(matched):
                    continue
                churn = float(np.mean(np.abs(cur[matched] - prev_r[matched])))
                out[r] = churn
            else:
                labels = {v for v in g_row if _valid_label(v)}
                for lab in labels:
                    # P0-011: cohort-consistent matched sample — a stock enters
                    # only if it belongs to the SAME group at both dates.  The
                    # lagged rank was computed inside the lagged group, so it is
                    # only comparable today if the label has not rotated.
                    mask = (g_row == lab) & (gv[r - lk] == lab)
                    cur = rank_panel[r][mask]
                    prev_r = rank_panel[r - lk][mask]
                    matched = np.isfinite(cur) & np.isfinite(prev_r)
                    if not np.any(matched):
                        continue
                    churn = float(np.mean(np.abs(cur[matched] - prev_r[matched])))
                    out[r][mask] = churn
        return frame_like(x, out)


@register_operator(
    name="cs_rank_composition_churn",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_rank_composition_churn",
    source="stateful.rotation",
)
class CsRankCompositionChurn(SeriesOperator):
    """Pool membership turnover between ``t`` and ``t-lag`` (R11 #153).

    ``|A_t Δ A_{t-lag}| / |A_t ∪ A_{t-lag}|`` where ``A_t`` is the set of
    finite (observable) names inside the group on date ``t``.  0 = identical
    pool; 1 = fully disjoint.  Broadcast to the group's stocks on ``t``.
    """

    metadata = metadata(
        "cs_rank_composition_churn",
        "横截面池成员变动: |A_t Δ A_{t-lag}| / |A_t ∪ A_{t-lag}|, 广播回股票。",
        ["x", "lag", "group"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )
    metadata.role = "group_state"
    metadata.tags = list(metadata.tags) + ["group_state"]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 5,
        group: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        lk = max(1, int(lag))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        if group is not None and isinstance(group, pd.DataFrame):
            gv = group.to_numpy(dtype=object)
        else:
            gv = None
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            if r < lk:
                continue
            g_row = gv[r] if gv is not None else None
            if g_row is None:
                cur = np.isfinite(xv[r])
                prev = np.isfinite(xv[r - lk])
                union = float(np.sum(cur | prev))
                if union <= _EPS:
                    continue
                inter = float(np.sum(cur & prev))
                out[r] = (union - inter) / union
            else:
                labels = {v for v in g_row if _valid_label(v)}
                for lab in labels:
                    mask = (g_row == lab) & (gv[r - lk] == lab)
                    if not np.any(mask):
                        continue
                    cur = np.isfinite(xv[r][mask])
                    prev = np.isfinite(xv[r - lk][mask])
                    union = float(np.sum(cur | prev))
                    if union <= _EPS:
                        continue
                    inter = float(np.sum(cur & prev))
                    out[r][mask] = (union - inter) / union
        return frame_like(x, out)


@register_operator(
    name="cs_rank_combined_churn",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_rank_combined_churn",
    source="stateful.rotation",
)
class CsRankCombinedChurn(SeriesOperator):
    """Combined rotation: intersection-cohort rank churn + composition churn.

    R11 #153: total reshuffling decomposes into same-stock rank moves
    (``cs_rank_churn``, intersection cohort) plus pool membership turnover
    (``cs_rank_composition_churn``).  ``combined = rank_churn + composition``,
    both in [0,1] for a single-date change, so the sum lies in [0,2].
    """

    metadata = metadata(
        "cs_rank_combined_churn",
        "总洗牌: 交集同股秩变化 + 池成员变动 (二者之和), 广播回股票。",
        ["x", "lag", "group"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )
    metadata.role = "group_state"
    metadata.tags = list(metadata.tags) + ["group_state"]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 5,
        group: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        rank_churn = CsRankChurn()._calculate_series(x, lag=lag, group=group)
        composition = CsRankCompositionChurn()._calculate_series(x, lag=lag, group=group)
        return rank_churn + composition


@register_operator(
    name="cs_tail_retention",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_tail_retention",
    source="stateful.rotation",
)
class CsTailRetention(SeriesOperator):
    """Overlap of the exact top/bottom tail membership today vs ``lag`` ago.

    The tail is an **exact top-k** tail (``k = ceil(q·N)``) with fractional mass
    on boundary ties (R11 #156).  ``q`` must satisfy ``0 < q <= 0.5`` — a real
    tail (R11 #155).  ``cohort`` names the retention denominator explicitly
    (R11 #154): ``intersection`` (previous tail ∩ observable today — the
    default, a data gap is "cannot judge", not "exited"), ``historical``
    (full previous tail), or ``current`` (today's tail).  Broadcast to the
    group's stocks on ``t``.
    """

    metadata = metadata(
        "cs_tail_retention",
        "横截面极端尾部成员留存率(精确top-K+分数并列质量; cohort 显式)。",
        ["x", "lag", "quantile", "side", "group", "cohort"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )
    metadata.role = "group_state"
    metadata.tags = list(metadata.tags) + ["group_state"]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 5,
        quantile: float = 0.1,
        side: str = "top",
        group: Any = None,
        cohort: str = "intersection",
        **_: Any,
    ) -> pd.DataFrame:
        lk = max(1, int(lag))
        q = _tail_quantile(quantile, "cs_tail_retention")
        side_s = str(side).lower()
        if side_s not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        cohort_s = str(cohort).lower()
        if cohort_s not in ("current", "historical", "intersection"):
            raise ValueError("cohort must be 'current', 'historical' or 'intersection'")
        top = side_s == "top"
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        if group is not None and isinstance(group, pd.DataFrame):
            gv = group.to_numpy(dtype=object)
        else:
            gv = None
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            if r < lk:
                continue
            g_row = gv[r] if gv is not None else None
            if g_row is None:
                w_cur = _exact_tail_weights(xv[r], q, top)
                w_prev = _exact_tail_weights(xv[r - lk], q, top)
                num = float(np.sum(np.minimum(w_prev, w_cur)))
                den = _cohort_denominator(w_prev, w_cur, np.isfinite(xv[r]), cohort_s)
                if den > _EPS:
                    out[r] = num / den
            else:
                labels = {v for v in g_row if _valid_label(v)}
                for lab in labels:
                    # P0-012: cohort-consistent tail sets — a stock in today's
                    # group B whose lagged tail was computed inside group A must
                    # not enter B's historical tail retention.
                    mask = (g_row == lab) & (gv[r - lk] == lab)
                    if not np.any(mask):
                        continue
                    cur_vals = xv[r][mask]
                    prev_vals = xv[r - lk][mask]
                    w_cur = _exact_tail_weights(cur_vals, q, top)
                    w_prev = _exact_tail_weights(prev_vals, q, top)
                    num = float(np.sum(np.minimum(w_prev, w_cur)))
                    den = _cohort_denominator(w_prev, w_cur, np.isfinite(cur_vals), cohort_s)
                    if den > _EPS:
                        out[r][mask] = num / den
        return frame_like(x, out)


@register_operator(
    name="cs_tail_breadth",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_tail_breadth",
    source="stateful.rotation",
)
class CsTailBreadth(SeriesOperator):
    """Effective tail breadth: ``Σ`` fractional tail membership (≈ ``ceil(q·N)``).

    Records the effective number of names the tail spans after fractional
    boundary-tie mass (R11 #156).  Broadcast to the group's stocks on ``t``.
    """

    metadata = metadata(
        "cs_tail_breadth",
        "精确尾部有效宽度: Σ 分数并列质量(≈ ceil(q·N)), 广播回股票。",
        ["x", "quantile", "side", "group"],
        domain="price_volume",
        unit="count",
        category="cross_sectional",
    )
    metadata.role = "group_state"
    metadata.tags = list(metadata.tags) + ["group_state"]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        quantile: float = 0.1,
        side: str = "top",
        group: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        q = _tail_quantile(quantile, "cs_tail_breadth")
        side_s = str(side).lower()
        if side_s not in ("top", "bottom"):
            raise ValueError("side must be 'top' or 'bottom'")
        top = side_s == "top"
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        if group is not None and isinstance(group, pd.DataFrame):
            gv = group.to_numpy(dtype=object)
        else:
            gv = None
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            g_row = gv[r] if gv is not None else None
            if g_row is None:
                w = _exact_tail_weights(xv[r], q, top)
                if float(np.sum(w)) > _EPS:
                    out[r] = float(np.sum(w))
            else:
                labels = {v for v in g_row if _valid_label(v)}
                for lab in labels:
                    mask = g_row == lab
                    w = _exact_tail_weights(xv[r][mask], q, top)
                    if float(np.sum(w)) > _EPS:
                        out[r][mask] = float(np.sum(w))
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(
        [
            "cs_rank_churn", "cs_tail_retention",
            "cs_rank_composition_churn", "cs_rank_combined_churn",
            "cs_tail_breadth",
        ]
    )


_register_surface()
