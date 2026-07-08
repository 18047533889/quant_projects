# -*- coding: utf-8 -*-
"""高频元素级算子的 Polars 实现（auto 路径优先选用）。"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import (
    OperatorMetadata,
    SeriesOperator,
    register_operator,
)

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _unary(expr_fn):
    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])

    return calc


@register_operator(name="abs", category="math", business_category="elementwise_math", canonical="abs", source="factor_dsl_polars")
class AbsPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="abs", category="math", description="绝对值",
        examples=["abs(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.abs())


@register_operator(name="neg", category="math", business_category="elementwise_math", canonical="neg", source="factor_dsl_polars")
class NegPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="neg", category="math", description="取负",
        examples=["neg(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: -c)


@register_operator(name="log", category="math", business_category="elementwise_math", canonical="log", source="factor_dsl_polars")
class LogPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="log", category="math", description="自然对数",
        examples=["log(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.log())


@register_operator(name="exp", category="math", business_category="elementwise_math", canonical="exp", source="factor_dsl_polars")
class ExpPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="exp", category="math", description="指数",
        examples=["exp(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.exp())


@register_operator(name="sqrt", category="math", business_category="elementwise_math", canonical="sqrt", source="factor_dsl_polars")
class SqrtPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="sqrt", category="math", description="平方根",
        examples=["sqrt(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sqrt())


@register_operator(name="sin", category="math", business_category="elementwise_math", canonical="sin", source="factor_dsl_polars")
class SinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="sin", category="math", description="正弦",
        examples=["sin(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sin())


@register_operator(name="cos", category="math", business_category="elementwise_math", canonical="cos", source="factor_dsl_polars")
class CosPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="cos", category="math", description="余弦",
        examples=["cos(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.cos())


@register_operator(name="sign", category="math", business_category="elementwise_math", canonical="sign", source="factor_dsl_polars")
class SignPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="sign", category="math", description="符号函数",
        examples=["sign(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.sign())


@register_operator(name="square", category="math", business_category="elementwise_math", canonical="square", source="factor_dsl_polars")
class SquarePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="square", category="math", description="平方",
        examples=["square(x)"], param_names=["x"], return_type="series", tags=["math", "polars"],
    )
    _calculate_series = _unary(lambda c: c.pow(2))


@register_operator(name="clip", category="math", business_category="elementwise_math", canonical="clip", source="factor_dsl_polars")
class ClipPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="clip", category="math", description="裁剪到 [lo, hi]",
        examples=["clip(x, -3, 3)"], param_names=["x", "lo", "hi"], return_type="series", tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, lo: float = -3.0, hi: float = 3.0, **kwargs) -> pl.DataFrame:
        lo_v = float(kwargs.get("min", lo))
        hi_v = float(kwargs.get("max", hi))
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).clip(lo_v, hi_v).alias(c) for c in cols])


@register_operator(name="where", category="signal", business_category="technical_signal", canonical="where", source="factor_dsl_polars")
class WherePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="where", category="signal", description="条件选择",
        examples=["where(cond, a, b)"], param_names=["cond", "a", "b"], return_type="series",
        tags=["signal", "polars"],
    )

    def _calculate_series(
        self,
        cond: pl.DataFrame,
        a: pl.DataFrame,
        b: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = [c for c in _numeric_cols(cond) if c in a.columns and c in b.columns]
        return cond.select([
            pl.when(pl.col(c).cast(pl.Boolean, strict=False))
            .then(a[c])
            .otherwise(b[c])
            .alias(c)
            for c in cols
        ])


@register_operator(name="ts_pct", category="time_series", business_category="time_series", canonical="ts_pct", source="factor_dsl_polars")
class TSPctPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_pct", category="time_series", description="d 期变化率",
        examples=["ts_pct(close, 1)"], param_names=["x", "d"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        periods = int(kwargs.get("periods", d))
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) / pl.col(c).shift(periods) - 1.0).alias(c) for c in cols
        ])


@register_operator(name="log_returns", category="financial", business_category="price_volume", canonical="log_returns", source="factor_dsl_polars")
class LogReturnsPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="log_returns", category="financial", description="对数收益率",
        examples=["log_returns(close)"], param_names=["x"], return_type="series", tags=["financial", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) / pl.col(c).shift(1)).log().alias(c) for c in cols
        ])


@register_operator(name="ts_argmax", category="time_series", business_category="time_series", canonical="ts_argmax", source="factor_dsl_polars")
class TSArgmaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_argmax", category="time_series", description="窗口内最大值偏移",
        examples=["ts_argmax(close, 20)"], param_names=["x", "d"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        window = int(kwargs.get("window", d))

        def argmax_offset(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            idx = int(np.nanargmax(arr))
            return float(len(arr) - 1 - idx)

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(argmax_offset, window_size=window, min_periods=1).alias(c)
            for c in cols
        ])


@register_operator(name="ts_argmin", category="time_series", business_category="time_series", canonical="ts_argmin", source="factor_dsl_polars")
class TSArgminPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_argmin", category="time_series", description="窗口内最小值偏移",
        examples=["ts_argmin(close, 20)"], param_names=["x", "d"], return_type="series", tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        window = int(kwargs.get("window", d))

        def argmin_offset(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            idx = int(np.nanargmin(arr))
            return float(len(arr) - 1 - idx)

        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_map(argmin_offset, window_size=window, min_periods=1).alias(c)
            for c in cols
        ])


@register_operator(name="ts_quantile", category="time_series", business_category="time_series", canonical="ts_quantile", source="factor_dsl_polars")
class TSQuantilePolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_quantile", category="time_series", description="滚动分位数",
        examples=["ts_quantile(returns, 20, 0.75)"], param_names=["x", "d", "q"], return_type="series",
        tags=["time_series", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, q: float = 0.5, **kwargs) -> pl.DataFrame:
        window = int(kwargs.get("window", d))
        quantile = float(kwargs.get("p", q))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).rolling_quantile(quantile=quantile, window_size=window, min_samples=1).alias(c)
            for c in cols
        ])
