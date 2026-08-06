# -*- coding: utf-8 -*-
"""批量注册简单元素级算子的 Polars 实现（跳过已有 polars backend 的 canonical）。"""
from __future__ import annotations

from typing import Callable

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


def _pl():
    if pl is None:
        raise RuntimeError(
            "polars is required for this operator; install factor-engine[polars]"
        )
    return pl


from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.registry import OperatorRegistry

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _has_polars(canonical: str) -> bool:
    return "polars" in OperatorRegistry.backends_for(canonical)


def _unary_calc(expr_fn: Callable):
    def calc(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])

    return calc


def _register_unary(
    canonical: str,
    *,
    name: str | None = None,
    description: str = "",
    expr_fn: Callable,
    category: str = "math",
    business_category: str = "elementwise_math",
) -> None:
    if _has_polars(canonical):
        return
    op_name = name or canonical
    desc = description or f"Polars {canonical}"

    class _UnaryPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=op_name,
            category=category,
            description=desc,
            examples=[f"{op_name}(x)"],
            param_names=(["x", "decimals"] if canonical == "round" else ["x"]),
            return_type="series",
            tags=[category, "polars", "auto"],
        )
        _calculate_series = _unary_calc(expr_fn)

    _UnaryPolars.__doc__ = desc

    _UnaryPolars.__name__ = f"{canonical.title().replace('_', '')}PolarsAuto"
    register_operator(
        name=op_name,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_UnaryPolars)


def _register_binary_horizontal(
    canonical: str,
    *,
    name: str | None = None,
    description: str = "",
    combine,
    category: str = "math",
    business_category: str = "elementwise_math",
) -> None:
    if _has_polars(canonical):
        return
    op_name = name or canonical

    class _BinaryPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=op_name,
            category=category,
            description=description or f"Polars {canonical}",
            examples=[f"{op_name}(x, y)"],
            param_names=["x", "y"],
            return_type="series",
            tags=[category, "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([combine(pl.col(c), y[c]).alias(c) for c in cols])

    _BinaryPolars.__doc__ = description or f"Polars {canonical}"

    _BinaryPolars.__name__ = f"{canonical.title().replace('_', '')}PolarsAuto"
    register_operator(
        name=op_name,
        category=category,
        business_category=business_category,
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_BinaryPolars)


# --- 一元映射（canonical → pl.Expr 变换）---
_UNARY: list[tuple[str, str, Callable]] = [
    ("acos", "acos", lambda c: c.arccos()),
    ("asin", "asin", lambda c: c.arcsin()),
    ("atan", "atan", lambda c: c.arctan()),
    ("cbrt", "cbrt", lambda c: c.pow(1.0 / 3.0)),
    ("ceil", "ceil", lambda c: c.ceil()),
    ("floor", "floor", lambda c: c.floor()),
    ("round", "round", lambda c: c.round()),
    ("fix", "fix", lambda c: c.truncate()),
    ("truncate", "truncate", lambda c: c.truncate()),
    ("cosh", "cosh", lambda c: c.cosh()),
    ("sinh", "sinh", lambda c: c.sinh()),
    ("tan", "tan", lambda c: c.tan()),
    ("log2", "log2", lambda c: c.log(2.0)),
    ("log_abs", "log_abs", lambda c: c.abs().log()),
    ("exp_neg", "exp_neg", lambda c: (-c).exp()),
    ("reciprocal", "reciprocal", lambda c: 1.0 / c),
    ("cube", "cube", lambda c: c.pow(3)),
    ("identity", "identity", lambda c: c),
    ("negate", "negate", lambda c: -c),
    ("sec", "sec", lambda c: 1.0 / c.cos()),
    ("csc", "csc", lambda c: 1.0 / c.sin()),
    ("cot", "cot", lambda c: 1.0 / c.tan()),
    ("cumulative_max", "cumulative_max", lambda c: c.cum_max()),
    ("cumulative_min", "cumulative_min", lambda c: c.cum_min()),
    ("running_sum", "running_sum", lambda c: c.cum_sum()),
]

for _canon, _name, _fn in _UNARY:
    _register_unary(_canon, name=_name, expr_fn=_fn)


def _expanding_mean_col(c: pl.Expr) -> pl.Expr:
    cnt = c.is_not_null().cast(pl.Float64).cum_sum()
    return c.cum_sum() / cnt


def _expanding_std_col(c: pl.Expr) -> pl.Expr:
    cnt = c.is_not_null().cast(pl.Float64).cum_sum()
    mu = c.cum_sum() / cnt
    mean_sq = c.pow(2).cum_sum() / cnt
    return (mean_sq - mu.pow(2)).sqrt()


for _canon, _name in (
    ("cumulative_mean", "cumulative_mean"),
    ("running_mean", "running_mean"),
):
    if not _has_polars(_canon):

        class _ExpandingMeanPolars(SeriesOperator):
            metadata = OperatorMetadata(
                name=_name,
                category="math",
                description=f"Polars {_canon}",
                param_names=["x"],
                return_type="series",
                tags=["math", "polars", "auto"],
            )

            def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
                cols = _numeric_cols(x)
                return x.with_columns([_expanding_mean_col(pl.col(c)).alias(c) for c in cols])

        _ExpandingMeanPolars.__doc__ = f"Polars {_canon}"

        _ExpandingMeanPolars.__name__ = f"{_canon.title()}PolarsAuto"
        register_operator(
            name=_name,
            category="math",
            business_category="elementwise_math",
            canonical=_canon,
            source="factor_dsl_polars_auto",
        )(_ExpandingMeanPolars)

if not _has_polars("running_std"):

    class RunningStdPolarsAuto(SeriesOperator):
        """Polars running_std"""
        metadata = OperatorMetadata(
            name="running_std",
            category="math",
            description="Polars running_std",
            param_names=["x"],
            return_type="series",
            tags=["math", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = _numeric_cols(x)
            return x.with_columns([_expanding_std_col(pl.col(c)).alias(c) for c in cols])

    register_operator(
        name="running_std",
        category="math",
        business_category="elementwise_math",
        canonical="running_std",
        source="factor_dsl_polars_auto",
    )(RunningStdPolarsAuto)

_register_binary_horizontal(
    "fmax",
    name="fmax",
    description="逐元素 fmax",
    combine=lambda *args, **kwargs: _pl().max_horizontal(*args, **kwargs),
)
_register_binary_horizontal(
    "fmin",
    name="fmin",
    description="逐元素 fmin",
    combine=lambda *args, **kwargs: _pl().min_horizontal(*args, **kwargs),
)


def _register_cleaning(canonical: str, name: str, calc_fn) -> None:
    if _has_polars(canonical):
        return

    class _CleaningPolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=name,
            category="data_handling",
            description=f"Polars {canonical}",
            param_names=["x"],
            return_type="series",
            tags=["data_handling", "polars", "auto"],
        )
        _calculate_series = calc_fn

    _CleaningPolars.__doc__ = f"Polars {canonical}"

    _CleaningPolars.__name__ = f"{canonical.title()}PolarsAuto"
    register_operator(
        name=name,
        category="data_handling",
        business_category="data_cleaning",
        canonical=canonical,
        source="factor_dsl_polars_auto",
    )(_CleaningPolars)


def _ffill(self, x, **kwargs) -> pl.DataFrame:
    cols = _numeric_cols(x)
    return x.with_columns([pl.col(c).forward_fill().alias(c) for c in cols])


def _dropna(self, x, **kwargs) -> pl.DataFrame:
    cols = _numeric_cols(x)
    return x.with_columns([pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in cols])


_register_cleaning("ffill", "ffill", _ffill)
_register_cleaning("dropna", "dropna", _dropna)

if not _has_polars("coalesce"):

    class CoalescePolarsAuto(SeriesOperator):
        """Polars coalesce"""
        metadata = OperatorMetadata(
            name="coalesce",
            category="elementwise_math",
            description="Polars coalesce",
            param_names=["x", "y"],
            return_type="series",
            tags=["elementwise", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([
                pl.coalesce([pl.col(c), y[c]]).alias(c) for c in cols
            ])

    register_operator(
        name="coalesce",
        category="elementwise_math",
        business_category="elementwise_math",
        canonical="coalesce",
        source="factor_dsl_polars_auto",
    )(CoalescePolarsAuto)

_register_unary("sigmoid", name="sigmoid", description="Sigmoid", expr_fn=lambda c: 1.0 / (1.0 + (-c).exp()))

if not _has_polars("and_"):

    class AndPolarsAuto(SeriesOperator):
        """逻辑与"""
        metadata = OperatorMetadata(
            name="and_", category="elementwise_math", description="逻辑与",
            param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([
                (pl.col(c).cast(pl.Boolean) & y[c].cast(pl.Boolean)).cast(pl.Float64).alias(c)
                for c in cols
            ])

    register_operator(
        name="and_", category="elementwise_math", business_category="elementwise_math",
        canonical="and_", source="factor_dsl_polars_auto",
    )(AndPolarsAuto)

if not _has_polars("or_"):

    class OrPolarsAuto(SeriesOperator):
        """逻辑或"""
        metadata = OperatorMetadata(
            name="or_", category="elementwise_math", description="逻辑或",
            param_names=["x", "y"], return_type="series", tags=["elementwise", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([
                (pl.col(c).cast(pl.Boolean) | y[c].cast(pl.Boolean)).cast(pl.Float64).alias(c)
                for c in cols
            ])

    register_operator(
        name="or_", category="elementwise_math", business_category="elementwise_math",
        canonical="or_", source="factor_dsl_polars_auto",
    )(OrPolarsAuto)

_register_unary("not_", name="not_", description="逻辑非", expr_fn=lambda c: (~c.cast(pl.Boolean)).cast(pl.Float64))

if not _has_polars("fillna"):

    class FillNaPolarsAuto(SeriesOperator):
        """NaN 填充"""
        metadata = OperatorMetadata(
            name="fillna", category="data_handling", description="NaN 填充",
            param_names=["x", "method"], return_type="series", tags=["data_handling", "polars"],
        )

        def _calculate_series(self, x: pl.DataFrame, value: float = 0.0, **kwargs) -> pl.DataFrame:
            fill = float(kwargs.get("v", value))
            cols = _numeric_cols(x)
            return x.with_columns([
                pl.col(c).fill_nan(fill).fill_null(fill).alias(c) for c in cols
            ])

    register_operator(
        name="fillna", category="data_handling", business_category="data_cleaning",
        canonical="fillna", source="factor_dsl_polars_auto",
    )(FillNaPolarsAuto)

def _register_compare(canon: str, name: str, op_fn) -> None:
    """注册比较算子；始终覆盖以修复旧版 for-loop 闭包 late-binding。"""

    class _ComparePolars(SeriesOperator):
        metadata = OperatorMetadata(
            name=name,
            category="elementwise_math",
            description=f"Polars {canon}",
            param_names=["x", "y"],
            return_type="series",
            tags=["elementwise", "polars", "auto"],
        )

        def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(x) if c in y.columns]
            return x.with_columns([op_fn(pl.col(c), y[c]).alias(c) for c in cols])

    _ComparePolars.__doc__ = f"Polars {canon}"

    _ComparePolars.__name__ = f"{canon.upper()}PolarsAuto"
    register_operator(
        name=name,
        category="elementwise_math",
        business_category="elementwise_math",
        canonical=canon,
        source="factor_dsl_polars_auto",
    )(_ComparePolars)


for _canon, _name, _op in (
    ("eq", "eq", lambda a, b: (a == b).cast(pl.Float64)),
    ("ne", "ne", lambda a, b: (a != b).cast(pl.Float64)),
    ("gt", "gt", lambda a, b: (a > b).cast(pl.Float64)),
    ("ge", "ge", lambda a, b: (a >= b).cast(pl.Float64)),
    ("lt", "lt", lambda a, b: (a < b).cast(pl.Float64)),
    ("le", "le", lambda a, b: (a <= b).cast(pl.Float64)),
):
    _register_compare(_canon, _name, _op)

if not _has_polars("atan2"):

    class Atan2PolarsAuto(SeriesOperator):
        """atan2"""
        metadata = OperatorMetadata(
            name="atan2", category="math", description="atan2",
            param_names=["y", "x"], return_type="series", tags=["math", "polars"],
        )

        def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
            cols = [c for c in _numeric_cols(y) if c in x.columns]
            return y.with_columns([pl.arctan2(y[c], x[c]).alias(c) for c in cols])

    register_operator(
        name="atan2", category="math", business_category="elementwise_math",
        canonical="atan2", source="factor_dsl_polars_auto",
    )(Atan2PolarsAuto)
