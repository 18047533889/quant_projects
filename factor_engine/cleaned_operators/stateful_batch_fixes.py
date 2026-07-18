# -*- coding: utf-8 -*-
"""Batch/checkpoint convergence fixes for Wilder stateful indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    frame_pd,
    pl,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry
from stateful_runtime import _adx_segment, _atr_segment, _rsi_segment


PANDAS_SOURCE = "composite_fastpath_primitives"
POLARS_SOURCE = "composite_fastpath_native_polars"


def _register(canonical, pandas_fn, polars_fn=None):
    catalog = OperatorRegistry._catalog.get(canonical, {})
    params = list(catalog.get("param_names") or [])
    description = str(catalog.get("description") or canonical)
    category = str(catalog.get("business_category") or "technical_signal")
    OperatorRegistry.register(
        PandasFunctionOperator(canonical, category, params, description, pandas_fn),
        canonical=canonical,
        backend="pandas_numpy",
        source=PANDAS_SOURCE,
        status=str(catalog.get("status") or "production"),
        backend_explicit=True,
    )
    if pl is not None and polars_fn is not None:
        OperatorRegistry.register(
            PolarsFunctionOperator(canonical, category, params, description, polars_fn),
            canonical=canonical,
            backend="polars",
            source=POLARS_SOURCE,
            status=str(catalog.get("status") or "production"),
            backend_explicit=True,
        )


def pd_rsi_wilder(x, window=14, **_):
    w = positive_int(window, "window")
    values = x.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        out[:, col], _ = _rsi_segment(values[:, col], {}, w)
    return frame_pd(x, out)


def pd_atr_wilder(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    w = positive_int(window, "window")
    out = np.full(close.shape, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col], _ = _atr_segment(
            high.iloc[:, col].to_numpy(dtype=float),
            low.iloc[:, col].to_numpy(dtype=float),
            close.iloc[:, col].to_numpy(dtype=float),
            {},
            w,
        )
    return frame_pd(close, out)


def pd_adx(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    w = positive_int(window, "window")
    out = np.full(close.shape, np.nan, dtype=float)
    for col in range(close.shape[1]):
        out[:, col], _ = _adx_segment(
            high.iloc[:, col].to_numpy(dtype=float),
            low.iloc[:, col].to_numpy(dtype=float),
            close.iloc[:, col].to_numpy(dtype=float),
            {},
            w,
        )
    return frame_pd(close, out)


if pl is not None:

    def _finite(expr):
        return expr.is_not_null() & expr.is_finite()

    def _clean(expr):
        return pl.when(_finite(expr)).then(expr.cast(pl.Float64)).otherwise(None)

    def _wilder_seeded(expr, window):
        """Native expression matching the checkpoint SMA-seeded recurrence."""
        w = positive_int(window, "window")
        clean = _clean(expr)
        valid = clean.is_not_null()
        count = valid.cast(pl.Int64).cum_sum()
        running_sum = (
            pl.when(valid).then(clean).otherwise(0.0).cum_sum()
        )
        seeded_input = (
            pl.when(count < w)
            .then(None)
            .when(valid & (count == w))
            .then(running_sum / float(w))
            .when(valid & (count > w))
            .then(clean)
            .otherwise(None)
        )
        smoothed = seeded_input.ewm_mean(
            alpha=1.0 / float(w),
            adjust=False,
            min_samples=1,
            ignore_nulls=True,
        )
        return pl.when(valid & (count >= w)).then(smoothed).otherwise(None)

    def pl_rsi_wilder(x, window=14, **_):
        w = positive_int(window, "window")
        exprs = []
        for col in pl_cols(x):
            close = _clean(pl.col(col))
            previous = close.forward_fill().shift(1)
            delta = pl.when(close.is_not_null()).then(close - previous).otherwise(None)
            gain = pl.when(delta.is_null()).then(None).when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta.is_null()).then(None).when(delta < 0).then(-delta).otherwise(0.0)
            avg_gain = _wilder_seeded(gain, w)
            avg_loss = _wilder_seeded(loss, w)
            ratio = avg_gain / pl.when(avg_loss.abs() > EPS).then(avg_loss).otherwise(None)
            raw = 100.0 - 100.0 / (1.0 + ratio)
            value = (
                pl.when((avg_loss == 0) & (avg_gain > 0))
                .then(100.0)
                .when((avg_gain == 0) & (avg_loss > 0))
                .then(0.0)
                .when((avg_gain == 0) & (avg_loss == 0))
                .then(50.0)
                .otherwise(raw)
            )
            exprs.append(pl.when(close.is_not_null()).then(value).otherwise(None).alias(col))
        return x.with_columns(exprs)

    def _true_range(high, low, close):
        h, l, c = _clean(high), _clean(low), _clean(close)
        previous_close = c.forward_fill().shift(1)
        return (
            pl.when(h.is_null() | l.is_null() | c.is_null())
            .then(None)
            .when(previous_close.is_null())
            .then(h - l)
            .otherwise(
                pl.max_horizontal(
                    h - l,
                    (h - previous_close).abs(),
                    (l - previous_close).abs(),
                )
            )
        )

    def pl_atr_wilder(high, low, close, window=14, **_):
        w = positive_int(window, "window")
        cols = [
            col for col in pl_cols(close)
            if col in high.columns and col in low.columns
        ]
        return close.with_columns(
            [
                _wilder_seeded(
                    _true_range(high[col], low[col], close[col]),
                    w,
                ).alias(col)
                for col in cols
            ]
        )

    def pl_adx(high, low, close, window=14, **_):
        w = positive_int(window, "window")
        exprs = []
        for col in [
            c for c in pl_cols(close)
            if c in high.columns and c in low.columns
        ]:
            h = _clean(high[col])
            l = _clean(low[col])
            c = _clean(close[col])
            previous_h = h.forward_fill().shift(1)
            previous_l = l.forward_fill().shift(1)
            up = h - previous_h
            down = previous_l - l
            plus = (
                pl.when(h.is_null() | l.is_null() | c.is_null())
                .then(None)
                .when(previous_h.is_null() | previous_l.is_null())
                .then(0.0)
                .when((up > down) & (up > 0))
                .then(up)
                .otherwise(0.0)
            )
            minus = (
                pl.when(h.is_null() | l.is_null() | c.is_null())
                .then(None)
                .when(previous_h.is_null() | previous_l.is_null())
                .then(0.0)
                .when((down > up) & (down > 0))
                .then(down)
                .otherwise(0.0)
            )
            atr = _wilder_seeded(_true_range(h, l, c), w)
            plus_smoothed = _wilder_seeded(plus, w)
            minus_smoothed = _wilder_seeded(minus, w)
            plus_di = pl.when(atr.abs() > EPS).then(100.0 * plus_smoothed / atr).otherwise(0.0)
            minus_di = pl.when(atr.abs() > EPS).then(100.0 * minus_smoothed / atr).otherwise(0.0)
            denominator = plus_di + minus_di
            dx = pl.when(denominator > EPS).then(
                100.0 * (plus_di - minus_di).abs() / denominator
            ).otherwise(0.0)
            exprs.append(_wilder_seeded(dx, w).alias(col))
        return close.with_columns(exprs)


_register(
    "RSI_WILDER",
    pd_rsi_wilder,
    pl_rsi_wilder if pl is not None else None,
)
_register(
    "ATR_WILDER",
    pd_atr_wilder,
    pl_atr_wilder if pl is not None else None,
)
_register(
    "ADX",
    pd_adx,
    pl_adx if pl is not None else None,
)
