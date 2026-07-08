# -*- coding: utf-8
"""技术指标 Polars 实现（EMA / WMA / RSI / MACD 簇）。"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _ewm_mean(x: pl.DataFrame, span: int) -> pl.DataFrame:
    cols = _numeric_cols(x)
    alpha = 2.0 / (float(span) + 1.0)
    return x.with_columns([
        pl.col(c).ewm_mean(alpha=alpha, adjust=False).alias(c) for c in cols
    ])


def _wma(x: pl.DataFrame, window: int) -> pl.DataFrame:
    cols = _numeric_cols(x)
    w = np.arange(1, int(window) + 1, dtype=float)
    w /= w.sum()

    def _apply(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        weights = w[-len(arr):]
        return float(np.dot(arr, weights))

    return x.with_columns([
        pl.col(c).rolling_map(_apply, window_size=int(window), min_periods=1).alias(c)
        for c in cols
    ])


@register_operator(name="EMA", category="time_series", business_category="time_series", canonical="ts_ema", source="factor_dsl_polars")
class EMAPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="EMA", category="time_series", description="指数移动平均",
        param_names=["x", "span"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 12, **kwargs) -> pl.DataFrame:
        window = int(kwargs.get("window", kwargs.get("d", span)))
        return _ewm_mean(x, window)


@register_operator(name="WMA", category="time_series", business_category="time_series", canonical="WMA", source="factor_dsl_polars")
class WMAPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="WMA", category="time_series", description="加权移动平均",
        param_names=["x", "window"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        return _wma(x, w)


@register_operator(name="RSI", category="financial", business_category="technical_signal", canonical="RSI", source="factor_dsl_polars")
class RSIPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="RSI", category="financial", description="相对强弱指数",
        param_names=["x", "window"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 14, **kwargs) -> pl.DataFrame:
        w = int(kwargs.get("d", window))
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            delta = pl.col(c).diff()
            gain = pl.when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
            avg_gain = gain.rolling_mean(window_size=w, min_periods=1)
            avg_loss = loss.rolling_mean(window_size=w, min_periods=1)
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            exprs.append(rsi.alias(c))
        return x.with_columns(exprs)


@register_operator(name="MACD", category="financial", business_category="technical_signal", canonical="MACD", source="factor_dsl_polars")
class MACDPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="MACD", category="financial", description="MACD 线",
        param_names=["x", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        fast_e = _ewm_mean(x, int(fast))
        slow_e = _ewm_mean(x, int(slow))
        cols = _numeric_cols(x)
        return fast_e.with_columns([
            (fast_e[c] - slow_e[c]).alias(c) for c in cols
        ])


@register_operator(name="MACD_line", category="financial", business_category="technical_signal", canonical="MACD_line", source="factor_dsl_polars")
class MACDLinePolars(MACDPolars):
    metadata = OperatorMetadata(
        name="MACD_line", category="financial", description="MACD 线",
        param_names=["price", "fast", "slow"], return_type="series", tags=["financial", "polars"],
    )


@register_operator(name="MACD_signal", category="financial", business_category="technical_signal", canonical="MACD_signal", source="factor_dsl_polars")
class MACDSignalPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="MACD_signal", category="financial", description="MACD 信号线",
        param_names=["price", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        line = MACDPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        cols = _numeric_cols(line)
        return _ewm_mean(line, int(signal))


@register_operator(name="MACD_hist", category="financial", business_category="technical_signal", canonical="MACD_hist", source="factor_dsl_polars")
class MACDHistPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="MACD_hist", category="financial", description="MACD 柱",
        param_names=["price", "fast", "slow", "signal"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
    ) -> pl.DataFrame:
        line = MACDPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        sig = MACDSignalPolars()._calculate_series(price, fast=fast, slow=slow, signal=signal)
        cols = _numeric_cols(line)
        return line.with_columns([(line[c] - sig[c]).alias(c) for c in cols])
