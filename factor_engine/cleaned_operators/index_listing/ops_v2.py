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
    m = member.copy()
    m = m.fillna(0)
    entries = ((m == 1) & (m.shift(1).fillna(0) != 1)).astype(float)
    exits = ((m != 1) & (m.shift(1).fillna(0) == 1)).astype(float)
    return entries.rolling(int(window)).sum() + exits.rolling(int(window)).sum()


_mk(
    "index_reconstitution_churn",
    "窗口内指数纳入与剔除次数合计。",
    ["member", "window"],
    _reconstitution_churn,
    unit="count",
)


def _multi_index_entry_intensity(entry_a, entry_b, entry_c, window=20):
    total = entry_a.fillna(0) + entry_b.fillna(0) + entry_c.fillna(0)
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
        out[col] = np.arange(len(out)) - base
    return out.clip(lower=0)


_mk(
    "listing_age",
    "上市日距当前交易日数（非上市股 NaN）。",
    ["listing_date"],
    lambda ld: _listing_age(ld, ld.index),
    unit="count",
)


def _suspension_frequency(is_suspend, window=60):
    return is_suspend.fillna(0).rolling(int(window)).mean()


_mk(
    "suspension_frequency",
    "停牌频率：窗口内 IsSuspend=1 占比。",
    ["is_suspend", "window"],
    _suspension_frequency,
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
