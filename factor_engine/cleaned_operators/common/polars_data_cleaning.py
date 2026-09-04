# -*- coding: utf-8 -*-
"""数据清洗 / EWM / 扩展窗口算子的 Polars 实现。

语义要点
--------
- **EWM**：``adjust=False`` 与 pandas 默认一致；``span`` 可经 kwargs ``d`` 传入。
- **ewm_corr / ewm_cov**：Polars 暂无与 pandas 完全等价的二元 EWM API，
  故转 pandas 计算后再写回，避免 rolling 近似带来的偏差。
- **causal_linear_extrapolate**：仅使用最近两个历史点外推，禁止引用未来。
- **expanding_rank**：当前值在 **前缀窗口** 内的百分位排名（与 ``cum_rank`` 别名同义）。
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

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
    """Polars EMA"""
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
    """Polars EWM（alpha）"""
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
    """Polars EWM 标准差"""
    # 100k GO P0#1: native polars EWM operators carry an explicit
    # PhysicalImplementationSpec (POLARS_NATIVE_EXPR) so production-capability
    # classification is authoritative rather than source-inspection.
    _physical_spec = PhysicalImplementationSpec(
        canonical="ewm_std",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=False,
        stateful=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=False,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.polars_data_cleaning:EWMStdPolars:v1",
        emitter_identity="polars.Expr.ewm_std:v1",
        parameter_domain_hash="ewm_std.span:int:min=1",
        semantic_contract_hash="ewm_std:min_samples=2:axis=time:v1",
    )
    metadata = OperatorMetadata(
        name="ewm_std", category="data_handling", description="EWM 标准差",
        param_names=["x", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        span_i = int(kwargs.get("d", span))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.when(pl.col(c).is_not_null().cum_sum() < 2)
            .then(None)
            .otherwise(
                pl.col(c).ewm_std(span=span_i, adjust=False, bias=False, min_samples=1)
                .fill_null(strategy="forward")
            ).alias(c)
            for c in cols
        ])


@register_operator(name="ewm_var", category="data_handling", business_category="data_cleaning", canonical="ewm_var", source="factor_dsl_polars")
class EWMVarPolars(SeriesOperator):
    """Polars EWM 方差"""
    # 100k GO P0#1: explicit native-polars PhysicalImplementationSpec (see
    # EWMStdPolars above for the rationale).
    _physical_spec = PhysicalImplementationSpec(
        canonical="ewm_var",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=False,
        stateful=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=False,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.polars_data_cleaning:EWMVarPolars:v1",
        emitter_identity="polars.Expr.ewm_var:v1",
        parameter_domain_hash="ewm_var.span:int:min=1",
        semantic_contract_hash="ewm_var:min_samples=2:axis=time:v1",
    )
    metadata = OperatorMetadata(
        name="ewm_var", category="data_handling", description="EWM 方差",
        param_names=["x", "span"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, span: int = 20, **kwargs) -> pl.DataFrame:
        span_i = int(kwargs.get("d", span))
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.when(pl.col(c).is_not_null().cum_sum() < 2)
            .then(None)
            .otherwise(
                pl.col(c).ewm_var(span=span_i, adjust=False, bias=False, min_samples=1)
                .fill_null(strategy="forward")
            ).alias(c)
            for c in cols
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
    """Polars 扩展最大值"""
    metadata = OperatorMetadata(
        name="expanding_max", category="data_handling", description="扩展最大值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_max().alias(c) for c in cols])


@register_operator(name="expanding_min", category="data_handling", business_category="data_cleaning", canonical="expanding_min", source="factor_dsl_polars")
class ExpandingMinPolars(SeriesOperator):
    """Polars 扩展最小值"""
    metadata = OperatorMetadata(
        name="expanding_min", category="data_handling", description="扩展最小值",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_min().alias(c) for c in cols])


@register_operator(name="expanding_sum", category="data_handling", business_category="data_cleaning", canonical="expanding_sum", source="factor_dsl_polars")
class ExpandingSumPolars(SeriesOperator):
    """Polars 扩展求和"""
    metadata = OperatorMetadata(
        name="expanding_sum", category="data_handling", description="扩展求和",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).cum_sum().alias(c) for c in cols])


@register_operator(name="fillna_const", category="data_handling", business_category="data_cleaning", canonical="fillna_const", source="factor_dsl_polars")
class FillNAConstPolars(SeriesOperator):
    """Polars 常量填充"""
    metadata = OperatorMetadata(
        name="fillna_const", category="data_handling", description="常量填充",
        param_names=["x", "value"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, value: float = 0, **kwargs) -> pl.DataFrame:
        v = float(kwargs.get("fill_value", value))
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_null(v).alias(c) for c in cols])


@register_operator(name="causal_linear_extrapolate", category="data_handling", business_category="data_cleaning", canonical="causal_linear_extrapolate", source="factor_dsl_polars", status="research")
class CausalLinearExtrapolatePolars(SeriesOperator):
    """仅使用最近两个历史有效点进行线性外推。"""

    metadata = OperatorMetadata(
        name="causal_linear_extrapolate", category="data_handling", description="因果线性外推",
        param_names=["x"], return_type="series", tags=["data_handling", "polars", "causal"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators._causal import causal_linear_extrapolate_panel

        cols = _numeric_cols(x)
        pdf = x.select(cols).to_pandas()
        filled = causal_linear_extrapolate_panel(pdf)
        return x.with_columns([pl.Series(name=c, values=filled[c].to_numpy()) for c in cols])


@register_operator(name="expanding_rank", category="data_handling", business_category="data_cleaning", canonical="expanding_rank", source="factor_dsl_polars")
class ExpandingRankPolars(SeriesOperator):
    """Polars 扩展窗口百分位排名"""
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
    """Polars 是否为 IEEE NaN（NULL → 0）。"""
    metadata = OperatorMetadata(
        name="is_nan", category="data_handling", description="是否为 IEEE NaN",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.logical_semantics import is_nan_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_nan_polars_expr(c).alias(c) for c in cols])


@register_operator(name="is_null", category="data_handling", business_category="data_cleaning", canonical="is_null", source="factor_dsl_polars")
class IsNullPolars(SeriesOperator):
    """Polars 是否为缺失（NULL 或 IEEE NaN，与 Pandas ``isna`` 对齐）。"""
    metadata = OperatorMetadata(
        name="is_null", category="data_handling", description="是否为 NULL",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.logical_semantics import is_null_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_null_polars_expr(c).alias(c) for c in cols])


@register_operator(name="is_not_null", category="data_handling", business_category="data_cleaning", canonical="is_not_null", source="factor_dsl_polars")
class IsNotNullPolars(SeriesOperator):
    """Polars 是否为非缺失（与 Pandas ``notna`` 对齐）。"""
    metadata = OperatorMetadata(
        name="is_not_null", category="data_handling", description="是否为非 NULL",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.logical_semantics import is_not_null_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_not_null_polars_expr(c).alias(c) for c in cols])


@register_operator(name="is_infinite", category="data_handling", business_category="data_cleaning", canonical="is_infinite", source="factor_dsl_polars")
class IsInfinitePolars(SeriesOperator):
    """Polars 是否为 ±Inf（NULL/NaN/有限 → 0）。"""
    metadata = OperatorMetadata(
        name="is_infinite", category="data_handling", description="是否为 ±Inf",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.elementwise_semantics import is_infinite_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_infinite_polars_expr(c).alias(c) for c in cols])


@register_operator(name="is_inf", category="data_handling", business_category="data_cleaning", canonical="is_inf", source="factor_dsl_polars")
class IsInfPolars(SeriesOperator):
    """Polars 是否为无穷"""
    metadata = OperatorMetadata(
        name="is_inf", category="data_handling", description="是否为无穷",
        param_names=["x"], return_type="series", tags=["data_handling", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        from factor_engine.backend.elementwise_semantics import is_infinite_polars_expr

        cols = _numeric_cols(x)
        return x.with_columns([is_infinite_polars_expr(c).alias(c) for c in cols])


@register_operator(
    name="protected_sqrt",
    category="data_cleaning",
    business_category="data_cleaning",
    canonical="protected_sqrt",
    source="factor_dsl_polars",
    backend="polars",
)
class ProtectedSqrtPolars(SeriesOperator):
    """安全平方根：``sqrt(max(x, 0))``，与 pandas ``ProtectedSqrtOp`` 对齐。"""

    metadata = OperatorMetadata(
        name="protected_sqrt",
        category="data_cleaning",
        description="安全平方根：sqrt(max(x, 0))",
        param_names=["x"],
        return_type="series",
        tags=["data_cleaning", "pit_safe", "polars"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.when(pl.col(c) > 0).then(pl.col(c).sqrt()).otherwise(0.0).alias(c) for c in cols
        ])
