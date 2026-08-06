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
    out = num / den.replace(0, np.nan)
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

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {name}
    )
    return cls


def _index_weight_gap(index_weight, free_float_weight):
    gap = index_weight - free_float_weight
    return gap / free_float_weight.replace(0, np.nan).abs()


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
    # Unknown state (NaN) must not be silently treated as "not entered": a missing
    # index-membership datum could mask a real entry.  Sum entries only over the
    # index flags that are known, and emit NaN on days where none of the three
    # index statuses are known (S9 unknown-state contract).
    known = entry_a.notna() | entry_b.notna() | entry_c.notna()
    total = (
        entry_a.where(entry_a.notna(), 0.0)
        + entry_b.where(entry_b.notna(), 0.0)
        + entry_c.where(entry_c.notna(), 0.0)
    ).where(known)
    return total.rolling(int(window)).sum()


_mk(
    "multi_index_entry_intensity",
    "短期内同时进入多个指数的数量。",
    ["entry_index_a", "entry_index_b", "entry_index_c", "window"],
    _multi_index_entry_intensity,
    unit="count",
)


def _listing_age(listing_date, index_dates):
    pos = {ts: i for i, ts in enumerate(index_dates)}
    out = pd.DataFrame(np.nan, index=listing_date.index, columns=listing_date.columns, dtype=float)
    for col in listing_date.columns:
        ld = listing_date[col].dropna()
        if len(ld) == 0:
            continue
        first_date = pd.Timestamp(ld.iloc[0]).normalize()
        base = pos.get(first_date)
        if base is None:
            continue
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


def _index_event_decay(entry_event, window=20, decay=0.9):
    """纳入/剔除事件后的衰减信号：事件近因加权。"""
    weight = np.array([float(decay) ** k for k in range(int(window))], dtype=float)
    return entry_event.fillna(0).rolling(int(window)).apply(lambda s: float(np.dot(s[::-1], weight)), raw=True)


_mk(
    "index_event_decay",
    "指数事件（纳入=1/剔除=-1）的衰减近因信号。",
    ["entry_event", "window", "decay"],
    _index_event_decay,
)
