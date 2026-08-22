# -*- coding: utf-8 -*-
"""Basic math and elementwise operators - Polars native implementations.

All operators use pure Polars expressions.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_EPS = 1e-12
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


# ---------------------------------------------------------------------------
# Basic math operators
# ---------------------------------------------------------------------------


@register_operator(
    name="sqrt_abs",
    category="math",
    business_category="elementwise_math",
    canonical="sqrt_abs",
    source=_SRC,
    backend="polars",
)
class SqrtAbsNative(SeriesOperator):
    """sqrt(|x|)."""

    metadata = OperatorMetadata(
        name="sqrt_abs",
        category="math",
        description="绝对值开方",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).abs().sqrt().alias(c) for c in cols])


@register_operator(
    name="log_positive_or_nan",
    category="math",
    business_category="elementwise_math",
    canonical="log_positive_or_nan",
    source=_SRC,
    backend="polars",
)
class LogPositiveOrNanNative(SeriesOperator):
    """log(x) if x > 0, else null."""

    metadata = OperatorMetadata(
        name="log_positive_or_nan",
        category="math",
        description="正值对数",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([
            pl.when(pl.col(c) > 0).then(pl.col(c).log()).otherwise(None).alias(c)
            for c in cols
        ])


@register_operator(
    name="acos_bounded",
    category="math",
    business_category="elementwise_math",
    canonical="acos_bounded",
    source=_SRC,
    backend="polars",
)
class AcosBoundedNative(SeriesOperator):
    """acos(clip(x, -1, 1))."""

    metadata = OperatorMetadata(
        name="acos_bounded",
        category="math",
        description="有界反余弦",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).clip(-1.0, 1.0).arccos().alias(c) for c in cols])


@register_operator(
    name="asin_bounded",
    category="math",
    business_category="elementwise_math",
    canonical="asin_bounded",
    source=_SRC,
    backend="polars",
)
class AsinBoundedNative(SeriesOperator):
    """asin(clip(x, -1, 1))."""

    metadata = OperatorMetadata(
        name="asin_bounded",
        category="math",
        description="有界反正弦",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).clip(-1.0, 1.0).arcsin().alias(c) for c in cols])


@register_operator(
    name="atan2",
    category="math",
    business_category="elementwise_math",
    canonical="atan2",
    source=_SRC,
    backend="polars",
)
class Atan2Native(SeriesOperator):
    """atan2(y, x)."""

    metadata = OperatorMetadata(
        name="atan2",
        category="math",
        description="两参数反正切",
        param_names=["y", "x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(y)
        exprs = []
        for c in cols:
            if c not in x.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(pl.col(c).arctan2(x[c]).alias(c))
        return y.with_columns(exprs)


@register_operator(
    name="safe_div_null",
    category="math",
    business_category="elementwise_math",
    canonical="safe_div_null",
    source=_SRC,
    backend="polars",
)
class SafeDivNullNative(SeriesOperator):
    """x / y, returning null if y == 0."""

    metadata = OperatorMetadata(
        name="safe_div_null",
        category="math",
        description="安全除法",
        param_names=["x", "y", "epsilon"],
        return_type="series",
        tags=["math", "polars", "native"],
        param_specs={
            "epsilon": ParamSpec(dtype=float, min=0.0, default=_EPS, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, epsilon: float = _EPS, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar

        eps = strict_finite_scalar(epsilon, "epsilon", minimum=0.0)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in y.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                numerator = pl.col(c).cast(pl.Float64, strict=False)
                denominator = y[c].cast(pl.Float64, strict=False)
                exprs.append(
                    pl.when(
                        numerator.is_finite()
                        & denominator.is_finite()
                        & (denominator.abs() > eps)
                    ).then(numerator / denominator).otherwise(None).alias(c)
                )
        return x.with_columns(exprs)


@register_operator(
    name="unitize",
    category="math",
    business_category="elementwise_math",
    canonical="unitize",
    source=_SRC,
    backend="polars",
)
class UnitizeNative(SeriesOperator):
    """sign(x): -1, 0, or 1."""

    metadata = OperatorMetadata(
        name="unitize",
        category="math",
        description="符号函数",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).sign().alias(c) for c in cols])


@register_operator(
    name="where",
    category="math",
    business_category="elementwise_math",
    canonical="where",
    source=_SRC,
    backend="polars",
)
class WhereNative(SeriesOperator):
    """Ternary where: cond ? x : y."""

    metadata = OperatorMetadata(
        name="where",
        category="math",
        description="条件选择",
        param_names=["cond", "x", "y"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, cond: pl.DataFrame, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(cond)
        exprs = []
        for c in cols:
            if c not in x.columns or c not in y.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(pl.when(cond[c]).then(x[c]).otherwise(y[c]).alias(c))
        return cond.with_columns(exprs)


@register_operator(
    name="cos_phase",
    category="math",
    business_category="elementwise_math",
    canonical="cos_phase",
    source=_SRC,
    backend="polars",
)
class CosPhaseNative(SeriesOperator):
    """cos(2π * x / period)."""

    metadata = OperatorMetadata(
        name="cos_phase",
        category="math",
        description="余弦相位",
        param_names=["x", "period"],
        return_type="series",
        tags=["math", "polars", "native"],
        param_specs={
            "period": ParamSpec(dtype=float, min=1.0, default=252.0, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, period: float = 252.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        import math

        p = strict_finite_scalar(period, "period", minimum=1.0)
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) * (2 * math.pi / p)).cos().alias(c) for c in cols
        ])


@register_operator(
    name="sin_phase",
    category="math",
    business_category="elementwise_math",
    canonical="sin_phase",
    source=_SRC,
    backend="polars",
)
class SinPhaseNative(SeriesOperator):
    """sin(2π * x / period)."""

    metadata = OperatorMetadata(
        name="sin_phase",
        category="math",
        description="正弦相位",
        param_names=["x", "period"],
        return_type="series",
        tags=["math", "polars", "native"],
        param_specs={
            "period": ParamSpec(dtype=float, min=1.0, default=252.0, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, period: float = 252.0, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        import math

        p = strict_finite_scalar(period, "period", minimum=1.0)
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) * (2 * math.pi / p)).sin().alias(c) for c in cols
        ])


@register_operator(
    name="row_sum_skipna",
    category="math",
    business_category="elementwise_math",
    canonical="row_sum_skipna",
    source=_SRC,
    backend="polars",
)
class RowSumSkipnaNative(SeriesOperator):
    """Row-wise sum across all numeric columns."""

    metadata = OperatorMetadata(
        name="row_sum_skipna",
        category="math",
        description="行求和(跳过NA)",
        param_names=["x"],
        return_type="series",
        tags=["math", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        if not cols:
            return x
        # Sum across columns for each row
        sum_expr = pl.sum_horizontal([pl.col(c) for c in cols])
        # Return single column result
        result = pl.DataFrame({"_sum": sum_expr})
        if "date" in x.columns:
            result = result.with_columns(x["date"])
        # Expand to all original columns with same value
        for c in cols:
            result = result.with_columns(result["_sum"].alias(c))
        return result.lazy().select(x.columns).collect()
