# -*- coding: utf-8 -*-
"""Strict production primitives added by the field/operator layer governance."""
from __future__ import annotations

import re
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry

PANDAS_SOURCE = "layer_governance_primitives"
POLARS_SOURCE = "layer_governance_native_polars"
EPS = 1e-12


def _register(name: str, category: str, params: list[str], description: str, pandas_fn, polars_fn=None) -> None:
    OperatorRegistry.register(
        PandasFunctionOperator(name, category, params, description, pandas_fn),
        canonical=name,
        backend="pandas_numpy",
        source=PANDAS_SOURCE,
        status="production",
        backend_explicit=True,
    )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(name, category, params, description, polars_fn),
            canonical=name,
            backend="polars",
            source=POLARS_SOURCE,
            status="production",
            backend_explicit=True,
        )


def _period_ordinal_one(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        number = int(value)
        year, quarter = divmod(number, 10)
        if year >= 1000 and 1 <= quarter <= 4:
            return float(year * 4 + quarter - 1)
        return float(number)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return float(ts.year * 4 + ((ts.month - 1) // 3))
    text = str(value).strip().upper()
    match = re.search(r"(\d{4})\D*Q?([1-4])$", text)
    if match:
        return float(int(match.group(1)) * 4 + int(match.group(2)) - 1)
    try:
        ts = pd.Timestamp(text)
        return float(ts.year * 4 + ((ts.month - 1) // 3))
    except Exception:
        return np.nan


def _period_ordinal_frame(period_id: pd.DataFrame) -> pd.DataFrame:
    return period_id.apply(lambda col: col.map(_period_ordinal_one)).astype(float)


def _period_lag(x: pd.DataFrame, period_id: pd.DataFrame, periods: int) -> pd.DataFrame:
    op = OperatorRegistry.get("period_lag", backend="pandas_numpy")
    if op is None:
        raise RuntimeError("period_lag primitive is required")
    return op.calculate(x, period_id, periods)


def _period_inputs(x: pd.DataFrame, period_id: pd.DataFrame, periods: int):
    x, period_id = aligned_pd(x, period_id)
    lag_n = positive_int(periods, "periods")
    ordinal = _period_ordinal_frame(period_id)
    previous = _period_lag(x, period_id, lag_n)
    previous_ordinal = _period_lag(ordinal, period_id, lag_n)
    consecutive = ordinal.sub(previous_ordinal).eq(float(lag_n))
    return x, period_id, ordinal, previous, consecutive


def pd_period_change(x, period_id, periods=1, mode="absolute", require_consecutive=True, **_):
    x, _, _, previous, consecutive = _period_inputs(x, period_id, periods)
    valid = x.notna() & previous.notna()
    if bool(require_consecutive):
        valid &= consecutive
    mode = str(mode).lower()
    if mode == "absolute":
        result = x - previous
    elif mode == "ratio":
        denominator = previous.where(previous.abs() > EPS)
        result = x / denominator - 1.0
    elif mode == "log":
        ratio = x / previous.where(previous.abs() > EPS)
        result = np.log(ratio.where(ratio > 0))
    else:
        raise ValueError("mode must be 'absolute', 'ratio', or 'log'")
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_period_average(x, period_id, periods=2, require_consecutive=True, **_):
    x, period_id = aligned_pd(x, period_id)
    count = positive_int(periods, "periods")
    ordinal = _period_ordinal_frame(period_id)
    pieces = [x]
    valid = x.notna()
    for lag in range(1, count):
        piece = _period_lag(x, period_id, lag)
        lag_ordinal = _period_lag(ordinal, period_id, lag)
        pieces.append(piece)
        valid &= piece.notna()
        if bool(require_consecutive):
            valid &= ordinal.sub(lag_ordinal).eq(float(lag))
    result = sum(pieces[1:], pieces[0].copy()) / float(count)
    return result.where(valid)


def pd_period_cagr(
    x,
    period_id,
    periods=12,
    periods_per_year=4,
    sign_policy="strict",
    require_consecutive=True,
    **_,
):
    x, _, _, previous, consecutive = _period_inputs(x, period_id, periods)
    ppy = positive_int(periods_per_year, "periods_per_year")
    lag_n = positive_int(periods, "periods")
    policy = str(sign_policy).lower()
    valid = x.notna() & previous.notna()
    if bool(require_consecutive):
        valid &= consecutive
    if policy == "strict":
        valid &= (x > 0) & (previous > 0)
        ratio = x / previous
    elif policy == "absolute":
        valid &= previous.abs().gt(EPS)
        ratio = np.where(previous.abs() != 0, x.abs() / previous.abs(), np.nan)
    else:
        raise ValueError("sign_policy must be 'strict' or 'absolute'")
    result = ratio.pow(float(ppy) / float(lag_n)) - 1.0
    return result.where(valid).replace([np.inf, -np.inf], np.nan)


def pd_ttm_quarterly_strict(x, period_id, periods=4, require_consecutive=True, **_):
    x, period_id = aligned_pd(x, period_id)
    count = positive_int(periods, "periods")
    ordinal = _period_ordinal_frame(period_id)
    pieces = [x]
    valid = x.notna()
    for lag in range(1, count):
        piece = _period_lag(x, period_id, lag)
        lag_ordinal = _period_lag(ordinal, period_id, lag)
        pieces.append(piece)
        valid &= piece.notna()
        if bool(require_consecutive):
            valid &= ordinal.sub(lag_ordinal).eq(float(lag))
    return sum(pieces[1:], pieces[0].copy()).where(valid)


def pd_yoy_period_strict(x, period_id, periods=4, denominator="signed", require_consecutive=True, **_):
    mode = str(denominator).lower()
    if mode == "signed":
        return pd_period_change(x, period_id, periods, "ratio", require_consecutive)
    if mode == "absolute":
        x, _, _, previous, consecutive = _period_inputs(x, period_id, periods)
        valid = x.notna() & previous.notna() & previous.abs().gt(EPS)
        if bool(require_consecutive):
            valid &= consecutive
        return ((x - previous) / previous.abs()).where(valid)
    raise ValueError("denominator must be 'signed' or 'absolute'")


def pd_true_range(high, low, close, **_):
    high, low, close = aligned_pd(high, low, close)
    previous = close.shift(1)
    values = np.maximum.reduce([
        (high - low).to_numpy(dtype=float),
        (high - previous).abs().to_numpy(dtype=float),
        (low - previous).abs().to_numpy(dtype=float),
    ])
    return frame_pd(close, values)


def pl_true_range(high, low, close, **_):
    cols = [c for c in pl_cols(close) if c in high.columns and c in low.columns]
    return close.with_columns([
        pl.max_horizontal(
            high[c] - low[c],
            (high[c] - close[c].shift(1)).abs(),
            (low[c] - close[c].shift(1)).abs(),
        ).alias(c)
        for c in cols
    ])


def pd_ffill_limit(x, max_periods, **_):
    limit = positive_int(max_periods, "max_periods")
    return x.ffill(limit=limit)


def pl_ffill_limit(x, max_periods, **_):
    limit = positive_int(max_periods, "max_periods")
    return x.with_columns([pl.col(c).forward_fill(limit=limit).alias(c) for c in pl_cols(x)])


def pd_cs_fill_mean(x, **_):
    mean = x.mean(axis=1, skipna=True)
    return x.T.fillna(mean).T


def pl_cs_fill_mean(x, **_):
    cols = pl_cols(x)
    mean = pl.mean_horizontal([pl.col(c) for c in cols])
    return x.with_columns([pl.when(pl.col(c).is_null()).then(mean).otherwise(pl.col(c)).alias(c) for c in cols])


def pd_cs_fill_median(x, **_):
    median = x.median(axis=1, skipna=True)
    return x.T.fillna(median).T


def pl_cs_fill_median(x, **_):
    cols = pl_cols(x)
    median = pl.concat_list([pl.col(c) for c in cols]).list.drop_nulls().list.median()
    return x.with_columns([pl.when(pl.col(c).is_null()).then(median).otherwise(pl.col(c)).alias(c) for c in cols])


def _rank_gaussian_array(values: np.ndarray, method: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    normal = NormalDist()
    for row in range(values.shape[0]):
        current = values[row]
        valid = np.isfinite(current)
        n = int(valid.sum())
        if n == 0:
            continue
        series = pd.Series(current[valid])
        ranks = series.rank(method="average").to_numpy(dtype=float)
        if method == "blom":
            probability = (ranks - 0.375) / (n + 0.25)
        elif method == "van_der_waerden":
            probability = ranks / (n + 1.0)
        else:
            raise ValueError("method must be 'blom' or 'van_der_waerden'")
        # 显式 clamp 到 (0,1) 内点，防止逆正态 CDF 因浮点边界产生 ±Inf。
        eps = 1e-12
        probability = np.clip(probability, eps, 1.0 - eps)
        out[row, valid] = np.array([normal.inv_cdf(float(p)) for p in probability])
    return out


def pd_cs_rank_gaussian(x, method="blom", **_):
    return frame_pd(x, _rank_gaussian_array(x.to_numpy(dtype=float), str(method).lower()))


def pl_cs_rank_gaussian(x, method="blom", **_):
    cols = pl_cols(x)
    values = x.select(cols).to_numpy()
    out = _rank_gaussian_array(values.astype(float), str(method).lower())
    return pl_base_with(x, {col: pl.Series(col, out[:, idx]) for idx, col in enumerate(cols)})


_register("true_range", "price_volume", ["high", "low", "close"], "真实波幅基础算子", pd_true_range, pl_true_range)
_register("ffill_limit", "data_cleaning", ["x", "max_periods"], "有最大连续期数的前向填充", pd_ffill_limit, pl_ffill_limit)
_register("cs_fill_mean", "cross_sectional", ["x"], "按当日横截面均值填充缺失", pd_cs_fill_mean, pl_cs_fill_mean)
_register("cs_fill_median", "cross_sectional", ["x"], "按当日横截面中位数填充缺失", pd_cs_fill_median, pl_cs_fill_median)
_register("cs_rank_gaussian", "cross_sectional", ["x", "method"], "横截面排名高斯化", pd_cs_rank_gaussian, pl_cs_rank_gaussian)
_register("period_change", "fundamental_period", ["x", "period_id", "periods", "mode", "require_consecutive"], "按连续报告期计算变化", pd_period_change)
_register("period_average", "fundamental_period", ["x", "period_id", "periods", "require_consecutive"], "按连续报告期计算平均值", pd_period_average)
_register("period_cagr", "fundamental_period", ["x", "period_id", "periods", "periods_per_year", "sign_policy", "require_consecutive"], "按连续报告期计算复合增长率", pd_period_cagr)
_register("ttm_from_quarterly", "fundamental_period", ["x", "period_id", "periods", "require_consecutive"], "严格连续报告期TTM", pd_ttm_quarterly_strict)
_register("yoy_by_period", "fundamental_period", ["x", "period_id", "periods", "denominator", "require_consecutive"], "严格连续报告期同比", pd_yoy_period_strict)
