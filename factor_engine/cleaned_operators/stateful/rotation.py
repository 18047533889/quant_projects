# -*- coding: utf-8 -*-
"""Cross-sectional rotation / style-stability primitives (2026-08 CTA pack).

* ``cs_rank_churn``     — per-day mean absolute change of cross-sectional
  percentile ranks over ``lag`` days (within an optional group), broadcast back
  to each stock.  High = the cross-section is reshuffling; low = stable
  leadership.
* ``cs_tail_retention`` — overlap of the top/bottom ``quantile`` membership
  today vs ``lag`` days ago (within group), broadcast back.  High = the extreme
  leaders/trailers persist.

Group labels follow the engine convention: a panel of object dtype whose
non-null entries are group labels (one fixed ``IndustrySource`` per use).
Output is always a panel with the input index/columns.
"""
from __future__ import annotations

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
    """

    metadata = metadata(
        "cs_rank_churn",
        "横截面排序洗牌度: 日均|秩变化|, 广播回股票。",
        ["x", "lag", "group"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )

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
                    mask = np.array([g == lab for g in g_row], dtype=bool)
                    cur = rank_panel[r][mask]
                    prev_r = rank_panel[r - lk][mask]
                    matched = np.isfinite(cur) & np.isfinite(prev_r)
                    if not np.any(matched):
                        continue
                    churn = float(np.mean(np.abs(cur[matched] - prev_r[matched])))
                    out[r][mask] = churn
        return frame_like(x, out)


@register_operator(
    name="cs_tail_retention",
    category="cross_sectional",
    business_category="cross_sectional_rotation",
    canonical="cs_tail_retention",
    source="stateful.rotation",
)
class CsTailRetention(SeriesOperator):
    """Overlap of the top/bottom ``quantile`` membership today vs ``lag`` ago.

    For side ``top``, ``A_t`` is the set of stocks whose within-group rank is in
    the top ``quantile`` fraction today; retention = ``|A_t ∩ A_{t-lag}| /
    |A_{t-lag}|``.  Broadcast to the group's stocks on ``t``.  NaN when the
    reference tail is empty or history is short.
    """

    metadata = metadata(
        "cs_tail_retention",
        "横截面极端尾部成员留存率(龙头/垫底是否持续)。",
        ["x", "lag", "quantile", "side", "group"],
        domain="price_volume",
        unit="ratio",
        category="cross_sectional",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        lag: int = 5,
        quantile: float = 0.1,
        side: str = "top",
        group: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        lk = max(1, int(lag))
        q = float(quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        top = str(side).lower() != "bottom"
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
                valid_prev = np.isfinite(prev_r)
                if not np.any(valid_prev):
                    continue
                if top:
                    in_prev = valid_prev & (prev_r >= 1.0 - q)
                else:
                    in_prev = valid_prev & (prev_r <= q)
                if not np.any(in_prev):
                    continue
                in_cur = np.isfinite(cur) & (
                    (cur >= 1.0 - q) if top else (cur <= q)
                )
                overlap = float(np.sum(in_prev & in_cur))
                out[r] = overlap / float(np.sum(in_prev))
            else:
                labels = {v for v in g_row if _valid_label(v)}
                for lab in labels:
                    mask = np.array([g == lab for g in g_row], dtype=bool)
                    cur = rank_panel[r][mask]
                    prev_r = rank_panel[r - lk][mask]
                    valid_prev = np.isfinite(prev_r)
                    if not np.any(valid_prev):
                        continue
                    if top:
                        in_prev = valid_prev & (prev_r >= 1.0 - q)
                    else:
                        in_prev = valid_prev & (prev_r <= q)
                    if not np.any(in_prev):
                        continue
                    in_cur = np.isfinite(cur) & (
                        (cur >= 1.0 - q) if top else (cur <= q)
                    )
                    overlap = float(np.sum(in_prev & in_cur))
                    out[r][mask] = overlap / float(np.sum(in_prev))
        return frame_like(x, out)


def _register_surface() -> None:
    from cleaned_operators.stateful._common import register_stateful_surface

    register_stateful_surface(["cs_rank_churn", "cs_tail_retention"])


_register_surface()
