# -*- coding: utf-8 -*-
"""Index-weight deviation, reconstitution churn and listing-age operators (P0).

``member`` is a 0/1 membership panel (1 = constituent, 0 = non-member,
NaN = unknown).  Only already-effective membership / weights are consumed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "level") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="index",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "index", "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:index",
            f"unit:{unit}", "cost:2",
        ],
    )


def _safe_div(num, den):
    out = np.where(den.replace(0, np.nan) != 0, num / den.replace(0, np.nan), np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "level"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"IndexListing_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="index",
        business_category="index",
        canonical=name,
        source="index_listing.ops_v2",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})
    return cls


def _index_weight_gap(index_weight, free_float_weight):
    gap = index_weight - free_float_weight
    return np.where(free_float_weight.replace(0, np.nan).abs() != 0, gap / free_float_weight.replace(0, np.nan).abs(), np.nan)


_mk(
    "index_weight_gap_to_free_float",
    "(指数权重 - 自由流通权重) / |自由流通权重|。",
    ["index_weight", "free_float_weight"],
    _index_weight_gap,
)


def _reconstitution_churn(member, window=60):
    """窗口内指数纳入+剔除次数。

    Unknown membership (NaN) must NOT be treated as a non-member: doing so
    fabricates entry/exit events at the boundary of a data gap.  Only count a
    transition when both the current and the previous status are known.
    """
    m = member.copy()
    known = m.notna()
    # Previous-row known mask: build explicitly to avoid shift+fillna
    # downcast deprecation.
    prev_known = pd.DataFrame(False, index=m.index, columns=m.columns)
    if len(m) > 1:
        prev_known.iloc[1:] = known.iloc[:-1].to_numpy()
    prev_known = prev_known.astype(bool)
    m_cur = m.where(known)
    m_prev = m.shift(1).where(prev_known)
    both_known = known & prev_known
    # Count a transition only when both sides are known; all other positions are
    # 0 (not NaN) so the rolling sum still aggregates countable transitions.
    entries = ((m_cur == 1) & (m_prev != 1)).astype(float).where(both_known, 0.0)
    exits = ((m_cur != 1) & (m_prev == 1)).astype(float).where(both_known, 0.0)
    return entries.rolling(int(window)).sum() + exits.rolling(int(window)).sum()


_mk(
    "index_reconstitution_churn",
    "窗口内指数纳入与剔除次数合计。",
    ["member", "window"],
    _reconstitution_churn,
    unit="count",
)


def _multi_index_entry_intensity(entry_a, entry_b, entry_c, window=20):
    # P1-141: partial-unknown flags (A=1,B=NaN,C=NaN) must NOT collapse to
    # "1 entered" via 1+0+0 — B/C could also be entered but are undisclosed.
    # Require ALL THREE index flags known per day; unknown is never treated as
    # "not entered".  The rolling sum then fails closed (min_periods=window)
    # whenever any day inside the window carries an unknown flag.
    all_known = entry_a.notna() & entry_b.notna() & entry_c.notna()
    total = (entry_a + entry_b + entry_c).where(all_known)
    return total.rolling(int(window)).sum()


_mk(
    "multi_index_entry_intensity",
    "短期内同时进入多个指数的数量。",
    ["entry_index_a", "entry_index_b", "entry_index_c", "window"],
    _multi_index_entry_intensity,
    unit="count",
)


def _listing_age(listing_date, index_dates):
    # P1-142: a ListingDate that falls on a non-trading day (weekend/holiday)
    # must map to the first trading day ON OR AFTER it (searchsorted "left"),
    # instead of being dropped for not appearing verbatim in the index.
    index_array = np.asarray(index_dates, dtype="datetime64[ns]")
    out = pd.DataFrame(np.nan, index=listing_date.index, columns=listing_date.columns, dtype=float)
    for col in listing_date.columns:
        ld = listing_date[col].dropna()
        if len(ld) == 0:
            continue
        first_date = pd.Timestamp(ld.iloc[0]).normalize().to_datetime64()
        base = int(np.searchsorted(index_array, first_date, side="left"))
        if base >= len(index_array):
            continue  # listing date after the whole panel → unknown age
        ages = np.arange(len(out), dtype=float) - base
        # Pre-listing rows are UNKNOWN (the stock does not trade yet), not an
        # age of zero.  Returning 0 conflates "just listed" with "not listed".
        ages[ages < 0] = np.nan
        out[col] = ages
    return out


_mk(
    "listing_age",
    "上市日距当前交易日数（非上市股 NaN）。",
    ["listing_date"],
    lambda ld: _listing_age(ld, ld.index),
    unit="count",
)


def _suspension_frequency(is_suspend, window=60):
    """停牌频率：已知状态日中停牌日占比。

    Unknown status (NaN) must not be counted as a normal trading day — doing so
    turns data gaps into a spurious "rarely suspended" factor.  The denominator
    is the number of days with a known status; the numerator is the number of
    those days flagged as suspended.  ``!= 0``（已知且非零）视为停牌，与 Polars
    后端保持一致（0/1 指标下与 ``== 1`` 等价）。
    """
    m = is_suspend.copy()
    known = m.notna().astype(float)
    num = ((m != 0) & m.notna()).astype(float).rolling(int(window)).sum()
    den = known.rolling(int(window)).sum()
    return (num / den.replace(0, np.nan)).where(den > 0)


def _suspension_status_coverage(is_suspend, window=60):
    """已知状态覆盖率：窗口内非 NaN 状态日占比。"""
    known = is_suspend.notna().astype(float)
    return known.rolling(int(window)).mean()


_mk(
    "suspension_frequency",
    "停牌频率：已知状态日中停牌日占比。",
    ["is_suspend", "window"],
    _suspension_frequency,
)


_mk(
    "suspension_status_coverage",
    "停牌状态覆盖率：窗口内已知状态日占比。",
    ["is_suspend", "window"],
    _suspension_status_coverage,
)


def _index_event_decay(entry_event, window=20, decay=0.9, missing_policy="break"):
    """纳入/剔除事件后的衰减近因信号（observed-state, review §7.3）。

    缺失事件（NaN）绝不当作「无事件=0」：
      - 首个已知事件之前 → NaN；
      - 窗口内出现数据缺口：
        ``missing_policy="break"``（默认）输出 NaN 并重置状态——须重新积累一个
        完整已知窗口后才恢复输出（缺口不归零、不制造虚假「无事件」）；
        ``"carry"`` 保留衰减权重状态，把最近一个有效输出向后携带（信号在缺口期
        不中断，也不谎报为 0 事件）。
    事件序列为 0/±1（纳入=+1、剔除=-1），权重按时间衰减。
    """
    w = int(window)
    d = float(decay)
    if not np.isfinite(d) or d < 0.0 or d > 1.0:
        raise ValueError("index_event_decay decay must satisfy 0 <= decay <= 1")
    policy = str(missing_policy or "break").lower()
    if policy not in ("break", "carry"):
        raise ValueError("index_event_decay missing_policy must be 'break' or 'carry'")
    weight = np.array([d ** k for k in range(w)], dtype=float)
    xv = entry_event.to_numpy(dtype=float)
    out = np.full(xv.shape, np.nan, dtype=float)
    for col in range(xv.shape[1]):
        seen = False
        last_valid = np.nan
        for row in range(xv.shape[0]):
            value = xv[row, col]
            if np.isfinite(value):
                seen = True
            if not seen:
                continue
            seg = xv[max(0, row - w + 1): row + 1, col]
            if len(seg) < w:
                continue  # full-window warmup contract, same as rolling(window)
            if np.isnan(seg).any():
                # Data gap inside the window: never treat as zero events.
                if policy == "carry" and np.isfinite(last_valid):
                    out[row, col] = last_valid
                continue
            out[row, col] = float(np.dot(seg[::-1], weight))
            last_valid = out[row, col]
    result = entry_event.copy()
    result.iloc[:, :] = out
    return result


_mk(
    "index_event_decay",
    "指数事件（纳入=1/剔除=-1）的衰减近因信号（observed-state）。",
    ["entry_event", "window", "decay", "missing_policy"],
    _index_event_decay,
)
