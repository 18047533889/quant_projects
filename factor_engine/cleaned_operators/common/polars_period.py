# -*- coding: utf-8 -*-
"""Period operators - Polars native implementations.

Period operators analyze data at the reporting period level (quarterly, annual, etc.).
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


@register_operator(
    name="period_average",
    category="period",
    business_category="period",
    canonical="period_average",
    source=_SRC,
    backend="polars")
class PeriodAverageNative(SeriesOperator):
    """Average value over a period window."""

    metadata = OperatorMetadata(
        name="period_average",
        category="period",
        description="期间平均值",
        param_names=["x", "window"],
        return_type="series",
        tags=["period", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=4, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 4, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)

        # TODO: Implement period average
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="period_lag",
    category="period",
    business_category="period",
    canonical="period_lag",
    source=_SRC,
    backend="polars")
class PeriodLagNative(SeriesOperator):
    """Lag by n periods."""

    metadata = OperatorMetadata(
        name="period_lag",
        category="period",
        description="期间滞后",
        param_names=["x", "n"],
        return_type="series",
        tags=["period", "polars", "native"],
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        lag = strict_integer(n, "n", minimum=1)

        # TODO: Implement period lag
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="period_change",
    category="period",
    business_category="period",
    canonical="period_change",
    source=_SRC,
    backend="polars")
class PeriodChangeNative(SeriesOperator):
    """Period-over-period change."""

    metadata = OperatorMetadata(
        name="period_change",
        category="period",
        description="期间变化",
        param_names=["x", "n"],
        return_type="series",
        tags=["period", "polars", "native"],
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        lag = strict_integer(n, "n", minimum=1)

        # TODO: Implement period change
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="period_cagr",
    category="period",
    business_category="period",
    canonical="period_cagr",
    source=_SRC,
    backend="polars")
class PeriodCAGRNative(SeriesOperator):
    """Compound annual growth rate over periods."""

    metadata = OperatorMetadata(
        name="period_cagr",
        category="period",
        description="期间复合增长率",
        param_names=["x", "n", "periods_per_year"],
        return_type="series",
        tags=["period", "polars", "native"],
        param_specs={
            "n": ParamSpec(dtype=int, min=1, default=4, searchable=True, param_role=ParamRole.HORIZON),
            "periods_per_year": ParamSpec(dtype=int, min=1, default=4, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n: int = 4, periods_per_year: int = 4, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        lag = strict_integer(n, "n", minimum=1)
        ppy = strict_integer(periods_per_year, "periods_per_year", minimum=1)

        # TODO: Implement period CAGR
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="period_stability",
    category="period",
    business_category="period",
    canonical="period_stability",
    source=_SRC,
    backend="polars")
class PeriodStabilityNative(SeriesOperator):
    """Stability measure: 1 - (std / mean) over periods."""

    metadata = OperatorMetadata(
        name="period_stability",
        category="period",
        description="期间稳定性",
        param_names=["x", "window"],
        return_type="series",
        tags=["period", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement period stability
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
