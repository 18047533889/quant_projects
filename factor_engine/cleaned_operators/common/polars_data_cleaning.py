# -*- coding: utf-8 -*-
"""数据清洗 / EWM / 扩展窗口算子的 Polars 实现。

语义要点
--------
- **EWM**：``adjust=False`` 与 pandas 默认一致；``span`` 可经 kwargs ``d`` 传入。
- **ewm_corr / ewm_cov**：Polars 暂无与 pandas 完全等价的二元 EWM API，
  故转 pandas 计算后再写回，避免 rolling 近似带来的偏差。
- **fillna_interpolate**：必须走 ``causal_interpolate_panel``，仅使用历史点插值，禁止引用未来。
- **expanding_rank**：当前值在 **前缀窗口** 内的百分位排名（与 ``cum_rank`` 别名同义）。
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


def _ewm_alpha(span: int) -> float:
    """pandas 风格 span → EWM alpha：alpha = 2 / (span + 1)。"""
    s = max(int(span), 1)
    return 2.0 / (float(s) + 1.0)


@register_operator(name="ewm_mean", category="data_handling", business_category="data_cleaning", canonical="ewm_mean", source="factor_dsl_polars")
class EWMMeanPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ewm_mean", category="data_handling", description="EMA",
        param_names=["x", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        alpha = _ewm_alpha(int(kwargs.get("d", span)))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).ewm_mean(alpha=alpha, adjust=False).alias(c) for c in cols
        ])


@register_operator(name="ewm", category="data_handling", business_category="data_cleaning", canonical="ewm", source="factor_dsl_polars")
class EWMPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ewm", category="data_handling", description="EWM（alpha）",
        param_names=["x", "alpha"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, alpha: float = 0.1, **kwargs) -> pl.DataFrame:
        a = float(kwargs.get("span", alpha))
        if a > 1.0:
            a = _ewm_alpha(int(a))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).ewm_mean(alpha=a, adjust=False).alias(c) for c in cols
        ])


@register_operator(name="ewm_std", category="data_handling", business_category="data_cleaning", canonical="ewm_std", source="factor_dsl_polars")
class EWMStdPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ewm_std", category="data_handling", description="EWM 标准差",
        param_names=["x", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        alpha = _ewm_alpha(int(kwargs.get("d", span)))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).ewm_std(alpha=alpha, adjust=False).alias(c) for c in cols
        ])


@register_operator(name="ewm_var", category="data_handling", business_category="data_cleaning", canonical="ewm_var", source="factor_dsl_polars")
class EWMVarPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="ewm_var", category="data_handling", description="EWM 方差",
        param_names=["x", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        alpha = _ewm_alpha(int(kwargs.get("d", span)))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.col(c).ewm_var(alpha=alpha, adjust=False).alias(c) for c in cols
        ])


@register_operator(name="ewm_corr", category="data_handling", business_category="data_cleaning", canonical="ewm_corr", source="factor_dsl_polars")
class EWMCorrPolars(SeriesOperator):
    """指数加权相关系数；经 pandas 桥接以与 ``EWMCorr`` (pandas) 数值对齐。"""

    metadata = OperatorMetadata(
        name="ewm_corr", category="data_handling", description="EWM 相关",
        param_names=["x", "y", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", span)), 2)
        cols = _align_cols(x, y)
        # Polars 无 ewm().corr()；pandas 为权威实现
        pdf_x = x.select(cols).to_pandas()
        pdf_y = y.select(cols).to_pandas()
        corr = pdf_x.ewm(span=w, adjust=False).corr(pdf_y)
        return x.with_columns([pl.Series(name=c, values=corr[c].to_numpy()) for c in cols])


@register_operator(name="ewm_cov", category="data_handling", business_category="data_cleaning", canonical="ewm_cov", source="factor_dsl_polars")
class EWMCovPolars(SeriesOperator):
    """指数加权协方差；经 pandas 桥接以与 ``EWMCov`` (pandas) 数值对齐。"""

    metadata = OperatorMetadata(
        name="ewm_cov", category="data_handling", description="EWM 协方差",
        param_names=["x", "y", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        w = max(int(kwargs.get("d", span)), 2)
        cols = _align_cols(x, y)
        pdf_x = x.select(cols).to_pandas()
        pdf_y = y.select(cols).to_pandas()
        cov = pdf_x.ewm(span=w, adjust=False).cov(pdf_y)
        return x.with_columns([pl.Series(name=c, values=cov[c].to_numpy()) for c in cols])


@register_operator(name="expanding_max", category="data_handling", business_category="data_cleaning", canonical="expanding_max", source="factor_dsl_polars")
class ExpandingMaxPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_max", category="data_handling", description="扩展最大值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_max().alias(c) for c in cols])


@register_operator(name="expanding_min", category="data_handling", business_category="data_cleaning", canonical="expanding_min", source="factor_dsl_polars")
class ExpandingMinPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_min", category="data_handling", description="扩展最小值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_min().alias(c) for c in cols])


@register_operator(name="expanding_sum", category="data_handling", business_category="data_cleaning", canonical="expanding_sum", source="factor_dsl_polars")
class ExpandingSumPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_sum", category="data_handling", description="扩展求和",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_sum().alias(c) for c in cols])


@register_operator(name="fillna_const", category="data_handling", business_category="data_cleaning", canonical="fillna_const", source="factor_dsl_polars")
class FillNAConstPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="fillna_const", category="data_handling", description="常量填充",
        param_names=["x", "value"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, value: float = 0, **kwargs) -> pl.DataFrame:
        v = float(kwargs.get("fill_value", value))
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_null(v).alias(c) for c in cols])


@register_operator(name="fillna_interpolate", category="data_handling", business_category="data_cleaning", canonical="fillna_interpolate", source="factor_dsl_polars")
class FillnaInterpolatePolars(SeriesOperator):
    """因果插值填充：仅依据 t 之前已知点，method 支持 linear / quadratic / cubic。"""

    metadata = OperatorMetadata(
        name="fillna_interpolate", category="data_handling", description="因果插值填充",
        param_names=["x", "method"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, method: str = "linear", **kwargs) -> pl.DataFrame:
        from cleaned_operators._causal import causal_interpolate_panel

        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        # 因果插值逻辑在 _causal.py，禁止前视
        filled = causal_interpolate_panel(pdf, method=str(kwargs.get("interp", method)))
        return x.with_columns([pl.Series(name=c, values=filled[c].to_numpy()) for c in cols])


@register_operator(name="expanding_rank", category="data_handling", business_category="data_cleaning", canonical="expanding_rank", source="factor_dsl_polars")
class ExpandingRankPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="expanding_rank", category="data_handling", description="扩展窗口百分位排名",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        ranked = pdf.expanding(min_periods=1).rank(pct=True)
        return x.with_columns([pl.Series(name=c, values=ranked[c].to_numpy()) for c in cols])


@register_operator(name="is_nan", category="data_handling", business_category="data_cleaning", canonical="is_nan", source="factor_dsl_polars")
class IsNaNPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="is_nan", category="data_handling", description="是否为 NaN/null",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            # Polars 区分 null 与 NaN；与 pandas isna() 对齐需二者皆判
            (pl.col(c).is_null() | pl.col(c).is_nan()).cast(pl.Float64).alias(c) for c in cols
        ])


@register_operator(name="is_inf", category="data_handling", business_category="data_cleaning", canonical="is_inf", source="factor_dsl_polars")
class IsInfPolars(SeriesOperator):
    metadata = OperatorMetadata(
        name="is_inf", category="data_handling", description="是否为无穷",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).is_infinite().cast(pl.Float64).alias(c) for c in cols])
