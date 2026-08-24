# -*- coding: utf-8 -*-
"""Small parity corrections for the composite fast path layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.composite_fastpath import (
    PANDAS_SOURCE,
    POLARS_SOURCE,
    _pl_align,
    _pl_wilder_expr,
)
from cleaned_operators.overhaul.base import (
    EPS,
    PandasFunctionOperator,
    PolarsFunctionOperator,
    aligned_pd,
    pl,
    pl_cols,
    positive_int,
)
from cleaned_operators.registry import OperatorRegistry


def _true_range_expr_strict(high: "pl.Expr", low: "pl.Expr", close: "pl.Expr") -> "pl.Expr":
    previous_close = close.shift(1)
    value = pl.max_horizontal(
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    )
    # pandas/NumPy maximum propagates the missing previous close on the first
    # observation. Keep that seed identical so Wilder min_samples aligns.
    return pl.when(previous_close.is_not_null()).then(value).otherwise(None)


def pl_atr_wilder(high, low, close, window=14, **_):
    w = positive_int(window, "window")
    cols = _pl_align(high, low, close)
    return close.with_columns(
        [
            _pl_wilder_expr(
                _true_range_expr_strict(high[c], low[c], close[c]),
                w,
                w,
            ).alias(c)
            for c in cols
        ]
    )


def pd_volatility(x: pd.DataFrame, window=20, min_periods=None, **_):
    w = positive_int(window, "window")
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 2)
    return x.rolling(window=w, min_periods=mp).std(ddof=1) * np.sqrt(252.0)


def pl_volatility(x, window=20, min_periods=None, **_):
    w = positive_int(window, "window")
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 2)
    scale = float(np.sqrt(252.0))
    return x.with_columns(
        [
            (
                pl.col(c).rolling_std(window_size=w, min_samples=mp, ddof=1)
                * scale
            ).alias(c)
            for c in pl_cols(x)
        ]
    )


def _register() -> None:
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "ATR_WILDER",
            "technical_signal",
            ["high", "low", "close", "window"],
            "共享真实波幅内核的 Wilder ATR",
            pl_atr_wilder,
        ),
        canonical="ATR_WILDER",
        backend="polars",
        source=POLARS_SOURCE,
        status="production",
        backend_explicit=True,
    )
    OperatorRegistry.register(
        PandasFunctionOperator(
            "volatility",
            "price_volume",
            ["x", "window", "min_periods"],
            "严格 min_periods 的年化波动率",
            pd_volatility,
        ),
        canonical="volatility",
        backend="pandas_numpy",
        source=PANDAS_SOURCE,
        status="production",
        backend_explicit=True,
    )
    OperatorRegistry.register(
        PolarsFunctionOperator(
            "volatility",
            "price_volume",
            ["x", "window", "min_periods"],
            "严格 min_periods 的年化波动率",
            pl_volatility,
        ),
        canonical="volatility",
        backend="polars",
        source=POLARS_SOURCE,
        status="production",
        backend_explicit=True,
    )


if pl is not None:
    _register()
else:
    OperatorRegistry.register(
        PandasFunctionOperator(
            "volatility",
            "price_volume",
            ["x", "window", "min_periods"],
            "严格 min_periods 的年化波动率",
            pd_volatility,
        ),
        canonical="volatility",
        backend="pandas_numpy",
        source=PANDAS_SOURCE,
        status="production",
        backend_explicit=True,
    )
