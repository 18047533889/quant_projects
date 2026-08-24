# -*- coding: utf-8 -*-
"""Phase 3 Module 11: Additional candlestick pattern operators (Polars native).

Most candlestick patterns are already in price_volume/polars_candle.py.
This module provides the generic candlestick_pattern operator and ensures
all pattern operators are accessible with consistent registration.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "polars_candle_patterns_phase3"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _frame(
    open_: pl.DataFrame | None,
    high: pl.DataFrame | None,
    low: pl.DataFrame | None,
    close: pl.DataFrame | None,
    column: str,
) -> pl.DataFrame:
    data: dict[str, pl.Series] = {}
    if open_ is not None:
        data["open"] = open_[column].fill_nan(None)
    if high is not None:
        data["high"] = high[column].fill_nan(None)
    if low is not None:
        data["low"] = low[column].fill_nan(None)
    if close is not None:
        data["close"] = close[column].fill_nan(None)
    return pl.DataFrame(data)


@register_operator(
    name="candlestick_pattern",
    category="candle_pattern",
    business_category="technical_extension",
    canonical="candlestick_pattern",
    source=_SRC,
    backend="polars",
)
class CandlestickPatternNative(SeriesOperator):
    """Generic candlestick pattern recognizer with configurable thresholds.

    Detects multiple pattern types based on pattern_type parameter:
    - "doji": Small body relative to range
    - "hammer": Long lower shadow, small upper shadow
    - "inverted_hammer": Long upper shadow, small lower shadow
    - "engulfing": Current body engulfs previous body
    - "inside": Current range inside previous range
    - "outside": Current range outside previous range
    """

    metadata = OperatorMetadata(
        name="candlestick_pattern",
        category="candle_pattern",
        description="Generic candlestick pattern detector",
        param_names=["open", "high", "low", "close", "pattern_type", "body_threshold", "shadow_threshold"],
        return_type="series",
        tags=["pit_safe", "causal", "candle", "polars", "native"],
        param_specs={
            "pattern_type": ParamSpec(dtype=str, default="doji", searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "body_threshold": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.10, searchable=True, param_role=ParamRole.THRESHOLD),
            "shadow_threshold": ParamSpec(dtype=float, min=0.0, default=2.0, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(
        self,
        open_: pl.DataFrame,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        pattern_type: str = "doji",
        body_threshold: float = 0.10,
        shadow_threshold: float = 2.0,
        **kwargs,
    ) -> pl.DataFrame:
        values = {}
        for c in _cols(open_, high, low, close):
            frame = _frame(open_, high, low, close, c)
            o, h, l, cl = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")

            body = cl - o
            abs_body = body.abs()
            rng = h - l
            max_oc = pl.max_horizontal(o, cl)
            min_oc = pl.min_horizontal(o, cl)
            upper = h - max_oc
            lower = min_oc - l

            if pattern_type == "doji":
                flag = abs_body <= body_threshold * rng
            elif pattern_type == "hammer":
                flag = (abs_body <= 0.35 * rng) & (lower >= shadow_threshold * abs_body) & (upper <= 0.35 * abs_body.clip(lower_bound=1e-12))
            elif pattern_type == "inverted_hammer":
                flag = (abs_body <= 0.35 * rng) & (upper >= shadow_threshold * abs_body) & (lower <= 0.35 * abs_body.clip(lower_bound=1e-12))
            elif pattern_type == "engulfing":
                prev_o, prev_c = o.shift(1), cl.shift(1)
                bull = (cl > o) & (prev_c < prev_o) & (o <= prev_c) & (cl >= prev_o)
                bear = (cl < o) & (prev_c > prev_o) & (o >= prev_c) & (cl <= prev_o)
                flag = bull | bear
            elif pattern_type == "inside":
                flag = (h < h.shift(1)) & (l > l.shift(1))
            elif pattern_type == "outside":
                flag = (h > h.shift(1)) & (l < l.shift(1))
            else:
                flag = pl.lit(False)

            result = pl.when(flag.fill_null(False)).then(1.0).otherwise(0.0)
            values[c] = _one(frame, c, result)

        return _result(close, values)
