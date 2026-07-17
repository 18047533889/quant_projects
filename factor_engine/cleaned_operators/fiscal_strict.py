# -*- coding: utf-8 -*-
"""Final strict PIT fiscal-period overrides.

Fiscal lags use exact quarter ordinals rather than first-observed row order.
Late filings and amendments therefore cannot redefine the previous period.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS, Spec, aligned_pd, finite_pd, frame_pd, nonnegative_int,
    pl, pl_base_with, pl_cols, positive_int, register_specs,
)


def period_ordinal(value: Any) -> int | None:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        year, quarter = divmod(number, 10)
        return year * 4 + quarter - 1 if year >= 1000 and 1 <= quarter <= 4 else number
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return period_ordinal(int(value))
    text = str(value).strip().upper()
    match = re.search(r"(\d{4})\D*Q?([1-4])$", text)
    if match:
        return int(match.group(1)) * 4 + int(match.group(2)) - 1
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    return None if pd.isna(ts) else int(ts.year * 4 + (ts.month - 1) // 3)


def ordinal_frame(period_id: pd.DataFrame) -> pd.DataFrame:
    return period_id.apply(lambda col: col.map(period_ordinal)).astype("Float64")


def _lag_array(values: np.ndarray, period_values: np.ndarray, lag: int, policy: str) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for col in range(values.shape[1]):
        visible: dict[int, float] = {}
        for row in range(values.shape[0]):
            ordinal = period_ordinal(period_values[row, col])
            if ordinal is None:
                continue
            value = values[row, col]
            if np.isfinite(value) and (policy == "latest_available" or ordinal not in visible):
                visible[int(ordinal)] = float(value)
            out[row, col] = visible.get(int(ordinal) - lag, np.nan)
    return out


def pd_period_lag(x, period_id, periods=1, revision_policy="latest_available", **_):
    x, period_id = aligned_pd(x, period_id)
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be latest_available or first_available")
    return frame_pd(x, _lag_array(x.to_numpy(dtype=float), period_id.to_numpy(dtype=object), lag, policy))


def pl_period_lag(x, period_id, periods=1, revision_policy="latest_available", **_):
    lag = nonnegative_int(periods, "periods")
    policy = str(revision_policy).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be latest_available or first_available")
    cols = [c for c in pl_cols(x) if c in period_id.columns]
    values = np.column_stack([x[c].cast(pl.Float64, strict=False).to_numpy() for c in cols])
    periods_array = np.column_stack([np.asarray(period_id[c].to_list(), dtype=object) for c in cols])
    result = _lag_array(values, periods_array, lag, policy)
    return pl_base_with(x, {c: pl.Series(c, result[:, i]) for i, c in enumerate(cols)})


def _ordinal_lag(period_id, lag, policy):
    return pd_period_lag(ordinal_frame(period_id).astype(float), period_id, lag, policy)


def pd_quarter_from_cumulative(x, period_id, fiscal_quarter=None, revision_policy="latest_available", **_):
    x, period_id = aligned_pd(x, period_id)
    ordinal = ordinal_frame(period_id).astype(float)
    previous = pd_period_lag(x, period_id, 1, revision_policy)
    previous_ordinal = _ordinal_lag(period_id, 1, revision_policy)
    if fiscal_quarter is None:
        quarter = ordinal.mod(4).add(1)
    else:
        _, fiscal_quarter = aligned_pd(x, fiscal_quarter)
        quarter = fiscal_quarter.astype(float)
    consecutive = ordinal.sub(previous_ordinal).eq(1.0)
    result = x.where(quarter.eq(1.0), x - previous.where(consecutive))
    valid = finite_pd(x) & finite_pd(quarter) & (quarter.eq(1.0) | consecutive)
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def _pieces(x, period_id, count, policy):
    ordinal = ordinal_frame(period_id).astype(float)
    pieces = [x]
    strict = finite_pd(x)
    available = finite_pd(x)
    for lag in range(1, count):
        piece = pd_period_lag(x, period_id, lag, policy)
        prior_ordinal = _ordinal_lag(period_id, lag, policy)
        pieces.append(piece)
        available &= finite_pd(piece)
        strict &= finite_pd(piece) & ordinal.sub(prior_ordinal).eq(float(lag))
    return pieces, strict, available


def pd_ttm_from_quarterly(x, period_id, periods=4, require_consecutive=True, revision_policy="latest_available", **_):
    x, period_id = aligned_pd(x, period_id)
    count = positive_int(periods, "periods")
    policy = str(revision_policy).lower()
    pieces, strict, available = _pieces(x, period_id, count, policy)
    total = pieces[0].copy()
    for piece in pieces[1:]:
        total = total + piece
    return total.where(strict if bool(require_consecutive) else available).replace([np.inf, -np.inf], np.nan)


def pd_ttm_from_cumulative(x, period_id, fiscal_quarter=None, revision_policy="latest_available", **_):
    quarterly = pd_quarter_from_cumulative(x, period_id, fiscal_quarter, revision_policy=revision_policy)
    return pd_ttm_from_quarterly(quarterly, period_id, 4, True, revision_policy)


def pd_yoy_by_period(x, period_id, periods=4, denominator="signed", require_consecutive=True, revision_policy="latest_available", **_):
    x, period_id = aligned_pd(x, period_id)
    lag = positive_int(periods, "periods")
    previous = pd_period_lag(x, period_id, lag, revision_policy)
    mode = str(denominator).lower()
    if mode == "signed":
        denom = previous
    elif mode == "absolute":
        denom = previous.abs()
    else:
        raise ValueError("denominator must be signed or absolute")
    valid = finite_pd(x) & finite_pd(denom) & denom.abs().gt(EPS)
    if bool(require_consecutive):
        valid &= ordinal_frame(period_id).astype(float).sub(_ordinal_lag(period_id, lag, revision_policy)).eq(float(lag))
    return ((x - previous) / denom.where(valid)).where(valid).replace([np.inf, -np.inf], np.nan)


def register() -> None:
    register_specs({
        "period_lag": Spec("fundamental_period", ["x", "period_id", "periods", "revision_policy"], "exact fiscal ordinal lag with visible revision policy", pd_period_lag, pl_period_lag),
        "quarter_from_cumulative": Spec("fundamental_period", ["x", "period_id", "fiscal_quarter", "revision_policy"], "strict cumulative-to-quarter conversion", pd_quarter_from_cumulative),
        "ttm_from_quarterly": Spec("fundamental_period", ["x", "period_id", "periods", "require_consecutive", "revision_policy"], "strict consecutive-period TTM", pd_ttm_from_quarterly),
        "ttm_from_cumulative": Spec("fundamental_period", ["x", "period_id", "fiscal_quarter", "revision_policy"], "strict cumulative-to-TTM", pd_ttm_from_cumulative),
        "yoy_by_period": Spec("fundamental_period", ["x", "period_id", "periods", "denominator", "require_consecutive", "revision_policy"], "strict fiscal-period growth", pd_yoy_by_period),
    })


register()
