# -*- coding: utf-8 -*-
"""Strict PIT-safe fiscal-period operators."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    aligned_pd,
    finite_pd,
    frame_pd,
    nonnegative_int,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
    register_specs,
)


def period_key(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return value.item()
    except AttributeError:
        return value


def pd_period_lag(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    periods: int = 1,
    revision_policy: str = "latest_available",
    **_,
) -> pd.DataFrame:
    x, period_id = aligned_pd(x, period_id)
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    xv, pv = x.to_numpy(), period_id.to_numpy()
    out = np.full(x.shape, np.nan)
    for col in range(x.shape[1]):
        order, positions, values = [], {}, {}
        for row in range(x.shape[0]):
            key = period_key(pv[row, col])
            if key is None:
                continue
            if key not in positions:
                positions[key] = len(order)
                order.append(key)
            value = xv[row, col]
            if np.isfinite(value) and (policy == "latest_available" or key not in values):
                values[key] = float(value)
            target = positions[key] - lag
            if target >= 0:
                out[row, col] = values.get(order[target], np.nan)
    return frame_pd(x, out)


def pl_period_lag(
    x,
    period_id,
    periods=1,
    revision_policy="latest_available",
    **_,
):
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    replacements = {}
    for col in [c for c in pl_cols(x) if c in period_id.columns]:
        vals = x[col].cast(pl.Float64, strict=False).to_numpy()
        ids = period_id[col].to_list()
        out = np.full(len(vals), np.nan)
        order, positions, values = [], {}, {}
        for row, raw in enumerate(ids):
            key = period_key(raw)
            if key is None:
                continue
            if key not in positions:
                positions[key] = len(order)
                order.append(key)
            value = vals[row]
            if np.isfinite(value) and (policy == "latest_available" or key not in values):
                values[key] = float(value)
            target = positions[key] - lag
            if target >= 0:
                out[row] = values.get(order[target], np.nan)
        replacements[col] = pl.Series(col, out)
    return pl_base_with(x, replacements)


def pd_quarter_from_cumulative(x, period_id, fiscal_quarter, **_):
    x, period_id, fiscal_quarter = aligned_pd(x, period_id, fiscal_quarter)
    previous = pd_period_lag(x, period_id, 1)
    previous_q = pd_period_lag(fiscal_quarter, period_id, 1)
    q = fiscal_quarter.astype(float)
    consecutive = ((previous_q == 4) & (q == 1)) | (q == previous_q + 1)
    result = x.where(q == 1, x - previous.where(consecutive))
    return result.where(finite_pd(x) & finite_pd(q))


def pd_ttm_from_quarterly(
    x,
    period_id,
    periods=4,
    require_consecutive=True,
    **_,
):
    count = positive_int(periods, "periods")
    pieces = [x] + [pd_period_lag(x, period_id, lag) for lag in range(1, count)]
    total = pieces[0].copy()
    valid = pieces[0].notna()
    for piece in pieces[1:]:
        total = total + piece
        valid &= piece.notna()
    if require_consecutive:
        return total.where(valid)
    return total.where(pieces[0].notna())


def pd_ttm_from_cumulative(x, period_id, fiscal_quarter, **_):
    quarterly = pd_quarter_from_cumulative(x, period_id, fiscal_quarter)
    return pd_ttm_from_quarterly(quarterly, period_id, periods=4, require_consecutive=True)


def pd_yoy_by_period(
    x,
    period_id,
    periods=4,
    denominator="signed",
    **_,
):
    previous = pd_period_lag(x, period_id, positive_int(periods, "periods"))
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be 'signed' or 'absolute'")
    valid = finite_pd(x) & finite_pd(denom) & denom.abs().gt(EPS)
    return ((x - previous) / denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan)


def register() -> None:
    register_specs({
        "period_lag": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "revision_policy"],
            "按真实可见的不同财务报告期滞后",
            pd_period_lag,
            pl_period_lag,
        ),
        "quarter_from_cumulative": Spec(
            "fundamental_period",
            ["x", "period_id", "fiscal_quarter"],
            "累计流量值按报告期转单季度",
            pd_quarter_from_cumulative,
        ),
        "ttm_from_quarterly": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "require_consecutive"],
            "按不同报告期累计 TTM，避免日频 forward-fill 重复计数",
            pd_ttm_from_quarterly,
        ),
        "ttm_from_cumulative": Spec(
            "fundamental_period",
            ["x", "period_id", "fiscal_quarter"],
            "累计值转单季后按报告期计算 TTM",
            pd_ttm_from_cumulative,
        ),
        "yoy_by_period": Spec(
            "fundamental_period",
            ["x", "period_id", "periods", "denominator"],
            "按不同报告期计算同比变化",
            pd_yoy_by_period,
        ),
    })
