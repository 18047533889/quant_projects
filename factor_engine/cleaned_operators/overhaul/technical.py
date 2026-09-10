# -*- coding: utf-8 -*-
"""Corrected technical indicators and composite-operator reuse."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

from factor_engine.cleaned_operators.overhaul.base import (
    EPS,
    Spec,
    aligned_pd,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    pl_unary_rolling_map,
    positive_int,
    register_specs,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


def call_registered(name, backend, *args, **kwargs):
    operator = OperatorRegistry.get(name, backend=backend)
    if operator is None:
        raise RuntimeError(f"required primitive missing: {name}/{backend}")
    return operator.calculate(*args, **kwargs)


def ema_registered(x, window, backend):
    operator = OperatorRegistry.get("ts_ema", backend=backend)
    if operator is None:
        raise RuntimeError(f"required primitive missing: ts_ema/{backend}")
    try:
        return operator.calculate(x, int(window))
    except TypeError:
        return operator.calculate(x, window=int(window))


def pd_macd_line(x, fast=12, slow=26, signal=None, **_):
    # ``signal`` is accepted for backward compatibility with historical
    # ``MACD(x, fast, slow, signal)`` calls.  The line itself does not use it.
    fast, slow = positive_int(fast, "fast"), positive_int(slow, "slow")
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    return call_registered(
        "subtract",
        "pandas_numpy",
        ema_registered(x, fast, "pandas_numpy"),
        ema_registered(x, slow, "pandas_numpy"),
    )


def pd_macd_signal(x, fast=12, slow=26, signal=9, **_):
    return ema_registered(pd_macd_line(x, fast, slow), positive_int(signal, "signal"), "pandas_numpy")


def pd_macd_hist(x, fast=12, slow=26, signal=9, **_):
    line = pd_macd_line(x, fast, slow)
    return call_registered(
        "subtract",
        "pandas_numpy",
        line,
        ema_registered(line, positive_int(signal, "signal"), "pandas_numpy"),
    )


def pl_macd_line(x, fast=12, slow=26, signal=None, **_):
    # Keep the same compatibility contract as the pandas implementation.
    fast, slow = positive_int(fast, "fast"), positive_int(slow, "slow")
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    return call_registered(
        "subtract",
        "polars",
        ema_registered(x, fast, "polars"),
        ema_registered(x, slow, "polars"),
    )


def pl_macd_signal(x, fast=12, slow=26, signal=9, **_):
    return ema_registered(pl_macd_line(x, fast, slow), positive_int(signal, "signal"), "polars")


def pl_macd_hist(x, fast=12, slow=26, signal=9, **_):
    line = pl_macd_line(x, fast, slow)
    return call_registered(
        "subtract",
        "polars",
        line,
        ema_registered(line, positive_int(signal, "signal"), "polars"),
    )


def pd_aroon_component(close, window, pick):
    window = positive_int(window, "window")
    values, out = close.to_numpy(dtype=float), np.full(close.shape, np.nan)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            segment = values[max(0, row - window) : row + 1, col]
            if segment.size < window + 1 or not np.isfinite(segment).all():
                continue
            target = segment.max() if pick == "max" else segment.min()
            out[row, col] = 100.0 * np.flatnonzero(segment == target)[-1] / window
    return frame_pd(close, out)


def pd_aroon_up(close, window=25, **_):
    return pd_aroon_component(close, window, "max")


def pd_aroon_down(close, window=25, **_):
    return pd_aroon_component(close, window, "min")


def pd_aroon(close, window=25, **_):
    return call_registered(
        "subtract",
        "pandas_numpy",
        pd_aroon_up(close, window),
        pd_aroon_down(close, window),
    )


def pl_aroon_component(close, window, pick):
    window = positive_int(window, "window")

    def fn(values):
        values = np.asarray(values, dtype=float)
        if values.size < window + 1 or not np.isfinite(values).all():
            return np.nan
        target = values.max() if pick == "max" else values.min()
        return float(100.0 * np.flatnonzero(values == target)[-1] / window)

    return pl_unary_rolling_map(close, window + 1, window + 1, fn)


def pl_aroon_up(close, window=25, **_):
    return pl_aroon_component(close, window, "max")


def pl_aroon_down(close, window=25, **_):
    return pl_aroon_component(close, window, "min")


def pl_aroon(close, window=25, **_):
    return call_registered(
        "subtract",
        "polars",
        pl_aroon_up(close, window),
        pl_aroon_down(close, window),
    )


def pd_adx(high, low, close, window=14, **_):
    high, low, close = aligned_pd(high, low, close)
    window = positive_int(window, "window")
    previous_close = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(dtype=float),
                (high - previous_close).abs().to_numpy(dtype=float),
                (low - previous_close).abs().to_numpy(dtype=float),
            ]
        ),
        index=high.index,
        columns=high.columns,
    )
    raw_plus = high - high.shift(1)
    raw_minus = low.shift(1) - low
    plus = raw_plus.where((raw_plus > raw_minus) & (raw_plus > 0), 0.0)
    minus = raw_minus.where((raw_minus > raw_plus) & (raw_minus > 0), 0.0)
    atr = true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def pl_adx(high, low, close, window=14, **_):
    window = positive_int(window, "window")
    replacements = {}
    for col in [c for c in pl_cols(high) if c in low.columns and c in close.columns]:
        temp = pl.DataFrame({"h": high[col], "l": low[col], "c": close[col]})
        raw_plus = pl.col("h") - pl.col("h").shift(1)
        raw_minus = pl.col("l").shift(1) - pl.col("l")
        true_range = pl.max_horizontal(
            pl.col("h") - pl.col("l"),
            (pl.col("h") - pl.col("c").shift(1)).abs(),
            (pl.col("l") - pl.col("c").shift(1)).abs(),
        )
        plus = pl.when((raw_plus > raw_minus) & (raw_plus > 0)).then(raw_plus).otherwise(0.0)
        minus = pl.when((raw_minus > raw_plus) & (raw_minus > 0)).then(raw_minus).otherwise(0.0)
        staged = temp.with_columns(
            true_range.ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).alias("atr"),
            plus.ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).alias("pdm"),
            minus.ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).alias("mdm"),
        ).with_columns(
            (100.0 * pl.col("pdm") / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)).alias("pdi"),
            (100.0 * pl.col("mdm") / pl.when(pl.col("atr").abs() > EPS).then(pl.col("atr")).otherwise(None)).alias("mdi"),
        ).with_columns(
            (100.0 * (pl.col("pdi") - pl.col("mdi")).abs() /
             pl.when((pl.col("pdi") + pl.col("mdi")).abs() > EPS)
             .then(pl.col("pdi") + pl.col("mdi")).otherwise(None)).alias("dx")
        )
        replacements[col] = staged.select(
            pl.col("dx").ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).alias("v")
        )["v"]
    return pl_base_with(high, replacements)


def register() -> None:
    register_specs({
        # NEW-040: ``signal`` is a declared non-searchable compatibility knob
        # (the historical MACD(x,fast,slow,signal) call aliases to MACD_line).
        "MACD_line": Spec("technical_signal", ["x", "fast", "slow", "signal"], "快慢 EMA 之差；复用 ts_ema 与 subtract（signal 为兼容 no-op）", pd_macd_line, pl_macd_line),
        "MACD_signal": Spec("technical_signal", ["x", "fast", "slow", "signal"], "MACD 信号线；复用 ts_ema", pd_macd_signal, pl_macd_signal),
        "MACD_hist": Spec("technical_signal", ["x", "fast", "slow", "signal"], "MACD 柱；复用已有基础算子", pd_macd_hist, pl_macd_hist),
        "AROON": Spec("technical_signal", ["close", "window"], "Aroon Up-Down，并列极值取最近一次", pd_aroon, pl_aroon),
        "AROON_up": Spec("technical_signal", ["close", "window"], "Aroon Up，并列极值取最近一次", pd_aroon_up, pl_aroon_up),
        "AROON_down": Spec("technical_signal", ["close", "window"], "Aroon Down，并列极值取最近一次", pd_aroon_down, pl_aroon_down),
        "ADX": Spec(
            "technical_signal", ["high", "low", "close", "window"],
            "Wilder ADX，方向运动独立判断", pd_adx, pl_adx,
            param_specs={
                "window": ParamSpec(
                    dtype=int, min=1, default=14,
                    param_role=ParamRole.HORIZON, searchable=True,
                )
            },
        ),
    })
