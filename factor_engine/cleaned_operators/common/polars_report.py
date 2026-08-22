# -*- coding: utf-8 -*-
"""Report-level operators - Polars native implementations.

Report operators analyze characteristics of financial reporting and filing behavior.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


@register_operator(
    name="report_rolling_mean",
    category="report",
    business_category="report",
    canonical="report_rolling_mean",
    source=_SRC,
    backend="polars")
class ReportRollingMeanNative(SeriesOperator):
    """Rolling mean across report periods."""

    metadata = OperatorMetadata(
        name="report_rolling_mean",
        category="report",
        description="报告期滚动均值",
        param_names=["x", "window"],
        return_type="series",
        tags=["report", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=4, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 4, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement report rolling mean
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_yoy_lag",
    category="report",
    business_category="report",
    canonical="report_yoy_lag",
    source=_SRC,
    backend="polars")
class ReportYoYLagNative(SeriesOperator):
    """Year-over-year value (lag 4 quarters for quarterly data)."""

    metadata = OperatorMetadata(
        name="report_yoy_lag",
        category="report",
        description="同比值",
        param_names=["x", "periods_per_year"],
        return_type="series",
        tags=["report", "polars", "native"],
        param_specs={
            "periods_per_year": ParamSpec(dtype=int, min=1, default=4, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, periods_per_year: int = 4, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        ppy = strict_integer(periods_per_year, "periods_per_year", minimum=1)

        # TODO: Implement YoY lag
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_change_breadth",
    category="report",
    business_category="report",
    canonical="report_change_breadth",
    source=_SRC,
    backend="polars")
class ReportChangeBreadthNative(SeriesOperator):
    """Fraction of line items that changed significantly."""

    metadata = OperatorMetadata(
        name="report_change_breadth",
        category="report",
        description="变化广度",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["report", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.05, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.05, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)

        # TODO: Implement change breadth
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_change_coherence",
    category="report",
    business_category="report",
    canonical="report_change_coherence",
    source=_SRC,
    backend="polars")
class ReportChangeCoherenceNative(SeriesOperator):
    """Coherence of changes across related line items."""

    metadata = OperatorMetadata(
        name="report_change_coherence",
        category="report",
        description="变化一致性",
        param_names=["x", "y"],
        return_type="series",
        tags=["report", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("report_change_coherence requires y")

        # TODO: Implement change coherence
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_revision_magnitude",
    category="report",
    business_category="report",
    canonical="report_revision_magnitude",
    source=_SRC,
    backend="polars")
class ReportRevisionMagnitudeNative(SeriesOperator):
    """Magnitude of revision from previous report."""

    metadata = OperatorMetadata(
        name="report_revision_magnitude",
        category="report",
        description="修正幅度",
        param_names=["x_current", "x_previous"],
        return_type="series",
        tags=["report", "polars", "native"],
    )

    def _calculate_series(self, x_current: pl.DataFrame, x_previous: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if x_previous is None:
            raise ValueError("report_revision_magnitude requires x_previous")

        # TODO: Implement revision magnitude
        cols = _numeric_cols(x_current)
        return x_current.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_filing_delay_surprise",
    category="report",
    business_category="report",
    canonical="report_filing_delay_surprise",
    source=_SRC,
    backend="polars")
class ReportFilingDelaySurpriseNative(SeriesOperator):
    """Unexpected delay in filing (days beyond typical)."""

    metadata = OperatorMetadata(
        name="report_filing_delay_surprise",
        category="report",
        description="申报延迟意外",
        param_names=["filing_lag", "window"],
        return_type="series",
        tags=["report", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=4, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, filing_lag: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=4)

        # TODO: Implement filing delay surprise
        cols = _numeric_cols(filing_lag)
        return filing_lag.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="report_benford_js_divergence",
    category="report",
    business_category="report",
    canonical="report_benford_js_divergence",
    source=_SRC,
    backend="polars")
class ReportBenfordJSDivergenceNative(SeriesOperator):
    """Jensen-Shannon divergence from Benford's Law distribution."""

    metadata = OperatorMetadata(
        name="report_benford_js_divergence",
        category="report",
        description="Benford分布偏离度",
        param_names=["x"],
        return_type="series",
        tags=["report", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # TODO: Implement Benford JS divergence
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
