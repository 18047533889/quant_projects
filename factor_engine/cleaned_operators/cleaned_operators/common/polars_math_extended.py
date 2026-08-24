# -*- coding: utf-8 -*-
"""元素级 / 基础数学算子 Polars 扩展（补齐 pandas-only 高频缺口）。"""
from __future__ import annotations

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common._polars_bridge import (
    align_cols,
    bridge_pandas,
    bridge_registry,
    colwise_numpy_kernel,
    numeric_cols,
    to_pandas_panel,
    from_pandas_panel,
)


def _unary(expr_fn):
    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = numeric_cols(x)
        return x.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])

    return calc


def _binary(combine):
    def calc(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = align_cols(x, y)
        return x.with_columns([combine(pl.col(c), y[c]).alias(c) for c in cols])

    return calc


def _domain_arc(fn):
    """R19-095: asin/acos 定义域 [-1,1]，越界/±Inf/NaN → NaN；NULL 保持 NULL。

    与 pandas 侧（elementwise.Asin/Acos）及 duckdb 侧一致 —— 任何 backend 不得
    对越界输入抛错或返回复数。
    """

    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = numeric_cols(x)
        return x.with_columns([
            pl.when(pl.col(c).is_null()).then(pl.col(c))
            .when((pl.col(c) >= -1.0) & (pl.col(c) <= 1.0)).then(fn(pl.col(c)))
            .otherwise(float("nan"))
            .alias(c)
            for c in cols
        ])

    return calc


@register_operator(name="asin", category="math", business_category="elementwise_math", canonical="asin", source="factor_dsl_polars")
class AsinPolars(SeriesOperator):
    """Polars 反正弦（定义域 [-1,1]，越界 → NaN）"""
    metadata = OperatorMetadata(name="asin", category="math", description="反正弦（定义域 [-1,1]，越界 → NaN）", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _domain_arc(lambda c: c.arcsin())


@register_operator(name="acos", category="math", business_category="elementwise_math", canonical="acos", source="factor_dsl_polars")
class AcosPolars(SeriesOperator):
    """Polars 反余弦（定义域 [-1,1]，越界 → NaN）"""
    metadata = OperatorMetadata(name="acos", category="math", description="反余弦（定义域 [-1,1]，越界 → NaN）", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _domain_arc(lambda c: c.arccos())


@register_operator(name="atan", category="math", business_category="elementwise_math", canonical="atan", source="factor_dsl_polars")
class AtanPolars(SeriesOperator):
    """Polars 反正切"""
    metadata = OperatorMetadata(name="atan", category="math", description="反正切", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: c.arctan())


@register_operator(name="tan", category="math", business_category="elementwise_math", canonical="tan", source="factor_dsl_polars")
class TanPolars(SeriesOperator):
    """Polars 正切"""
    metadata = OperatorMetadata(name="tan", category="math", description="正切", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: c.tan())


@register_operator(name="cbrt", category="math", business_category="elementwise_math", canonical="cbrt", source="factor_dsl_polars")
class CbrtPolars(SeriesOperator):
    """Polars 立方根"""
    metadata = OperatorMetadata(name="cbrt", category="math", description="立方根", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: c.sign() * c.abs().pow(1.0 / 3.0))


@register_operator(name="ceil", category="math", business_category="elementwise_math", canonical="ceil", source="factor_dsl_polars")
class CeilPolars(SeriesOperator):
    """Polars 向上取整"""
    metadata = OperatorMetadata(name="ceil", category="math", description="向上取整", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: c.ceil())


@register_operator(name="floor", category="math", business_category="elementwise_math", canonical="floor", source="factor_dsl_polars")
class FloorPolars(SeriesOperator):
    """Polars 向下取整"""
    metadata = OperatorMetadata(name="floor", category="math", description="向下取整", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: c.floor())


@register_operator(name="round", category="math", business_category="elementwise_math", canonical="round", source="factor_dsl_polars")
class RoundPolars(SeriesOperator):
    """Polars 四舍五入"""
    metadata = OperatorMetadata(
        name="round", category="math", description="四舍五入", param_names=["x", "decimals"], tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, decimals: int = 0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        d = strict_integer(decimals, "decimals", minimum=-18, maximum=18)
        cols = numeric_cols(x)
        return x.with_columns([pl.col(c).round(d).alias(c) for c in cols])


@register_operator(name="truncate", category="math", business_category="elementwise_math", canonical="truncate", source="factor_dsl_polars")
class TruncatePolars(SeriesOperator):
    """Polars 向零截断到指定小数位。"""
    metadata = OperatorMetadata(
        name="truncate", category="math", description="向零截断",
        param_names=["x", "decimals"], tags=["math", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, decimals: int = 0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        value = strict_integer(decimals, "decimals", minimum=-18, maximum=18)
        scale = 10.0 ** value
        cols = numeric_cols(x)
        return x.with_columns([((pl.col(c) * scale).truncate() / scale).alias(c) for c in cols])


@register_operator(name="inv", category="math", business_category="elementwise_math", canonical="inv", source="factor_dsl_polars")
class InvPolars(SeriesOperator):
    """Polars 倒数"""
    metadata = OperatorMetadata(name="inv", category="math", description="倒数", param_names=["x"], tags=["math", "polars"])
    _calculate_series = _unary(lambda c: 1.0 / c)


@register_operator(name="inverse", category="elementwise_math", business_category="elementwise_math", canonical="inverse", source="factor_dsl_polars")
class InversePolars(SeriesOperator):
    """Polars 倒数"""
    metadata = OperatorMetadata(name="inverse", category="elementwise_math", description="倒数", param_names=["x"], tags=["polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = numeric_cols(x)
        return x.with_columns([
            # Pandas/NumPy's reciprocal uses NaN for zero and missing inputs;
            # returning Polars null changes the backend's missing-value contract.
            pl.when(pl.col(c).is_null() | (pl.col(c) == 0))
            .then(float("nan"))
            .otherwise(1.0 / pl.col(c))
            .alias(c)
            for c in cols
        ])


@register_operator(name="reverse", category="elementwise_math", business_category="elementwise_math", canonical="reverse", source="factor_dsl_polars")
class ReversePolars(SeriesOperator):
    """Polars 取负"""
    metadata = OperatorMetadata(name="reverse", category="elementwise_math", description="取负", param_names=["x"], tags=["polars"])
    _calculate_series = _unary(lambda c: -c)


@register_operator(name="real", category="math", business_category="elementwise_math", canonical="real", source="factor_dsl_polars")
class RealPolars(SeriesOperator):
    """Polars 实部"""
    metadata = OperatorMetadata(name="real", category="math", description="实部", param_names=["x"], tags=["math", "polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        import pandas as pd

        return bridge_pandas(
            x,
            lambda pdf: pd.DataFrame(np.real(pdf.values), index=pdf.index, columns=pdf.columns),
        )


@register_operator(name="imag", category="math", business_category="elementwise_math", canonical="imag", source="factor_dsl_polars")
class ImagPolars(SeriesOperator):
    """Polars 虚部"""
    metadata = OperatorMetadata(name="imag", category="math", description="虚部", param_names=["x"], tags=["math", "polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        import pandas as pd

        return bridge_pandas(
            x,
            lambda pdf: pd.DataFrame(np.imag(pdf.values), index=pdf.index, columns=pdf.columns).replace(
                [np.inf, -np.inf], np.nan
            ),
        )


@register_operator(name="arg", category="math", business_category="elementwise_math", canonical="arg", source="factor_dsl_polars")
class ArgPolars(SeriesOperator):
    """Polars 相位"""
    metadata = OperatorMetadata(name="arg", category="math", description="相位", param_names=["x"], tags=["math", "polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        import pandas as pd

        return bridge_pandas(
            x,
            lambda pdf: pd.DataFrame(np.angle(pdf.values), index=pdf.index, columns=pdf.columns).replace(
                [np.inf, -np.inf], np.nan
            ),
        )


@register_operator(name="atan2", category="math", business_category="elementwise_math", canonical="atan2", source="factor_dsl_polars")
class Atan2Polars(SeriesOperator):
    """Polars atan2"""
    metadata = OperatorMetadata(name="atan2", category="math", description="atan2", param_names=["y", "x"], tags=["math", "polars"])

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return bridge_registry("atan2", y, x)


@register_operator(name="lerp", category="math", business_category="elementwise_math", canonical="lerp", source="factor_dsl_polars")
class LerpPolars(SeriesOperator):
    """Polars 线性插值"""
    metadata = OperatorMetadata(name="lerp", category="math", description="线性插值", param_names=["a", "b", "fraction"], tags=["math", "polars"])

    def _calculate_series(self, a: pl.DataFrame, b: pl.DataFrame, fraction: float = 0.5, **kwargs) -> pl.DataFrame:
        frac = float(fraction)
        cols = align_cols(a, b)
        return a.with_columns([(pl.col(c) + frac * (b[c] - pl.col(c))).alias(c) for c in cols])


@register_operator(name="blom_transform", category="math", business_category="elementwise_math", canonical="blom_transform", source="factor_dsl_polars")
class BlomTransformPolars(SeriesOperator):
    """Polars Blom 变换"""
    metadata = OperatorMetadata(name="blom_transform", category="math", description="Blom 变换", param_names=["x"], tags=["math", "polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return bridge_registry("blom_transform", x)


@register_operator(name="at_imax", category="statistics", business_category="statistics_regression", canonical="at_imax", source="factor_dsl_polars")
class AtImaxPolars(SeriesOperator):
    """Polars 扩展 argmax"""
    metadata = OperatorMetadata(name="at_imax", category="statistics", description="扩展 argmax", param_names=["x"], tags=["polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return bridge_registry("at_imax", x)


@register_operator(name="at_imin", category="statistics", business_category="statistics_regression", canonical="at_imin", source="factor_dsl_polars")
class AtIminPolars(SeriesOperator):
    """Polars 扩展 argmin"""
    metadata = OperatorMetadata(name="at_imin", category="statistics", description="扩展 argmin", param_names=["x"], tags=["polars"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return bridge_registry("at_imin", x)


@register_operator(name="digital_count", category="time_series", business_category="time_series", canonical="digital_count", source="factor_dsl_polars")
class DigitalCountPolars(SeriesOperator):
    """Polars 连续小波动计数"""
    metadata = OperatorMetadata(
        name="digital_count", category="time_series", description="连续小波动计数",
        param_names=["x", "d", "threshold", "run"], tags=["polars"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, d: int = 20, threshold: float = 0.01, run: int = 3, **kwargs
    ) -> pl.DataFrame:
        from factor_engine.cleaned_operators._numpy_kernels import digital_count_

        return colwise_numpy_kernel(
            x,
            digital_count_,
            d=int(kwargs.get("d", d)),
            threshold=float(threshold),
            run=int(run),
        )
