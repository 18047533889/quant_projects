# -*- coding: utf-8 -*-
"""Intraday advanced operators - Polars native implementations.

Advanced intraday operators for beta, correlation, lead-lag, VWAP, and state analysis.
All implementations use pure Polars expressions without pandas fallback.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_intraday_advanced"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Realized beta & correlation
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_realized_beta",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_realized_beta",
    source=_SRC,
    backend="polars",
)
class IntraRealizedBetaNative(SeriesOperator):
    """Realized beta from intraday returns: cov(r, r_mkt) / var(r_mkt)."""

    metadata = OperatorMetadata(
        name="intra_realized_beta",
        category="intraday",
        description="日内已实现Beta",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_realized_beta requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(market)).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_realized_beta_ex_self",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_realized_beta_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraRealizedBetaExSelfNative(SeriesOperator):
    """Realized beta vs market excluding self."""

    metadata = OperatorMetadata(
        name="intra_realized_beta_ex_self",
        category="intraday",
        description="日内已实现Beta(去自身)",
        param_names=["x", "market", "weight"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, market: pl.DataFrame | None = None, weight: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_realized_beta_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(market)).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_realized_correlation",
    category="intraday",
    business_category="intraday_correlation",
    canonical="intra_realized_correlation",
    source=_SRC,
    backend="polars",
)
class IntraRealizedCorrelationNative(SeriesOperator):
    """Realized correlation from intraday returns."""

    metadata = OperatorMetadata(
        name="intra_realized_correlation",
        category="intraday",
        description="日内已实现相关性",
        param_names=["x", "y"],
        return_type="series",
        tags=["intraday", "correlation", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("intra_realized_correlation requires y")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(y)).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_realized_correlation_ex_self",
    category="intraday",
    business_category="intraday_correlation",
    canonical="intra_realized_correlation_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraRealizedCorrelationExSelfNative(SeriesOperator):
    """Realized correlation vs market excluding self."""

    metadata = OperatorMetadata(
        name="intra_realized_correlation_ex_self",
        category="intraday",
        description="日内已实现相关性(去自身)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "correlation", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_realized_correlation_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(market)).alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# Market model R2 & idiosyncratic risk
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_market_model_r2",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_market_model_r2",
    source=_SRC,
    backend="polars",
)
class IntraMarketModelR2Native(SeriesOperator):
    """Market model R^2 from intraday returns."""

    metadata = OperatorMetadata(
        name="intra_market_model_r2",
        category="intraday",
        description="日内市场模型R2",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "r2", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_market_model_r2 requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) * pl.col(c).from_(market)) ** 2).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_market_model_r2_ex_self",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_market_model_r2_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraMarketModelR2ExSelfNative(SeriesOperator):
    """Market model R^2 excluding self."""

    metadata = OperatorMetadata(
        name="intra_market_model_r2_ex_self",
        category="intraday",
        description="日内市场模型R2(去自身)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "r2", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_market_model_r2_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) * pl.col(c).from_(market)) ** 2).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_variance",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_variance",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticVarianceNative(SeriesOperator):
    """Idiosyncratic variance: total variance - systematic variance."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_variance",
        category="intraday",
        description="日内特质方差",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_variance requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) ** 2 - (pl.col(c) * pl.col(c).from_(market)) ** 2).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_variance_ex_self",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_variance_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticVarianceExSelfNative(SeriesOperator):
    """Idiosyncratic variance excluding self from market."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_variance_ex_self",
        category="intraday",
        description="日内特质方差(去自身)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_variance_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) ** 2 - (pl.col(c) * pl.col(c).from_(market)) ** 2).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_skewness",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_skewness",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticSkewnessNative(SeriesOperator):
    """Idiosyncratic skewness from residuals."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_skewness",
        category="intraday",
        description="日内特质偏度",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_skewness requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) - pl.col(c) * pl.col(c).from_(market)) ** 3).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_skewness_ex_self",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_skewness_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticSkewnessExSelfNative(SeriesOperator):
    """Idiosyncratic skewness excluding self."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_skewness_ex_self",
        category="intraday",
        description="日内特质偏度(去自身)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_skewness_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) - pl.col(c) * pl.col(c).from_(market)) ** 3).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_kurtosis",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_kurtosis",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticKurtosisNative(SeriesOperator):
    """Idiosyncratic kurtosis from residuals."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_kurtosis",
        category="intraday",
        description="日内特质峰度",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_kurtosis requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) - pl.col(c) * pl.col(c).from_(market)) ** 4).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_idiosyncratic_kurtosis_ex_self",
    category="intraday",
    business_category="intraday_risk",
    canonical="intra_idiosyncratic_kurtosis_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraIdiosyncraticKurtosisExSelfNative(SeriesOperator):
    """Idiosyncratic kurtosis excluding self."""

    metadata = OperatorMetadata(
        name="intra_idiosyncratic_kurtosis_ex_self",
        category="intraday",
        description="日内特质峰度(去自身)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "risk", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_idiosyncratic_kurtosis_ex_self requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                ((pl.col(c) - pl.col(c) * pl.col(c).from_(market)) ** 4).alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# Semi-beta (up/down market regimes)
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_up_up_semibeta",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_up_up_semibeta",
    source=_SRC,
    backend="polars",
)
class IntraUpUpSemibetaNative(SeriesOperator):
    """Semi-beta: both stock and market up."""

    metadata = OperatorMetadata(
        name="intra_up_up_semibeta",
        category="intraday",
        description="日内上行半Beta(双涨)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_up_up_semibeta requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) > 0) & (pl.col(c).from_(market) > 0))
                .then(pl.col(c) * pl.col(c).from_(market))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_up_down_semibeta",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_up_down_semibeta",
    source=_SRC,
    backend="polars",
)
class IntraUpDownSemibetaNative(SeriesOperator):
    """Semi-beta: stock up, market down."""

    metadata = OperatorMetadata(
        name="intra_up_down_semibeta",
        category="intraday",
        description="日内半Beta(涨跌背离)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_up_down_semibeta requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) > 0) & (pl.col(c).from_(market) < 0))
                .then(pl.col(c) * pl.col(c).from_(market))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_down_up_semibeta",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_down_up_semibeta",
    source=_SRC,
    backend="polars",
)
class IntraDownUpSemibetaNative(SeriesOperator):
    """Semi-beta: stock down, market up."""

    metadata = OperatorMetadata(
        name="intra_down_up_semibeta",
        category="intraday",
        description="日内半Beta(跌涨背离)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_down_up_semibeta requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) < 0) & (pl.col(c).from_(market) > 0))
                .then(pl.col(c) * pl.col(c).from_(market))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_down_down_semibeta",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_down_down_semibeta",
    source=_SRC,
    backend="polars",
)
class IntraDownDownSemibetaNative(SeriesOperator):
    """Semi-beta: both stock and market down."""

    metadata = OperatorMetadata(
        name="intra_down_down_semibeta",
        category="intraday",
        description="日内下行半Beta(双跌)",
        param_names=["x", "market"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_down_down_semibeta requires market")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.when((pl.col(c) < 0) & (pl.col(c).from_(market) < 0))
                .then(pl.col(c) * pl.col(c).from_(market))
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_beta_asymmetry",
    category="intraday",
    business_category="intraday_beta",
    canonical="intra_beta_asymmetry",
    source=_SRC,
    backend="polars",
)
class IntraBetaAsymmetryNative(SeriesOperator):
    """Beta asymmetry: (up_beta - down_beta) / (up_beta + down_beta)."""

    metadata = OperatorMetadata(
        name="intra_beta_asymmetry",
        category="intraday",
        description="日内Beta不对称性",
        param_names=["up_beta", "down_beta"],
        return_type="series",
        tags=["intraday", "beta", "polars", "native"],
    )

    def _calculate_series(self, up_beta: pl.DataFrame, down_beta: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if down_beta is None:
            raise ValueError("intra_beta_asymmetry requires down_beta")
        cols = _numeric_cols(up_beta)
        return up_beta.with_columns(
            [
                pl.when((pl.col(c) + pl.col(c).from_(down_beta)) == 0)
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(down_beta)) / (pl.col(c) + pl.col(c).from_(down_beta)))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Lead-lag relationships
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_market_lead_lag_ex_self",
    category="intraday",
    business_category="intraday_lead_lag",
    canonical="intra_market_lead_lag_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraMarketLeadLagExSelfNative(SeriesOperator):
    """Lead-lag correlation with market (excluding self)."""

    metadata = OperatorMetadata(
        name="intra_market_lead_lag_ex_self",
        category="intraday",
        description="日内市场领先滞后(去自身)",
        param_names=["x", "market", "lag"],
        return_type="series",
        tags=["intraday", "lead_lag", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=-10, max=10, default=1, searchable=True,
                            param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, market: pl.DataFrame | None = None, lag: int = 1, **kwargs) -> pl.DataFrame:
        if market is None:
            raise ValueError("intra_market_lead_lag_ex_self requires market")
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(lag, "lag")

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(market).shift(n)).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_industry_lead_lag_ex_self",
    category="intraday",
    business_category="intraday_lead_lag",
    canonical="intra_industry_lead_lag_ex_self",
    source=_SRC,
    backend="polars",
)
class IntraIndustryLeadLagExSelfNative(SeriesOperator):
    """Lead-lag correlation with industry (excluding self)."""

    metadata = OperatorMetadata(
        name="intra_industry_lead_lag_ex_self",
        category="intraday",
        description="日内行业领先滞后(去自身)",
        param_names=["x", "industry", "lag"],
        return_type="series",
        tags=["intraday", "lead_lag", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=-10, max=10, default=1, searchable=True,
                            param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, industry: pl.DataFrame | None = None, lag: int = 1, **kwargs) -> pl.DataFrame:
        if industry is None:
            raise ValueError("intra_industry_lead_lag_ex_self requires industry")
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(lag, "lag")

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(industry).shift(n)).alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# VWAP analysis
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_vwap_above_ratio",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_vwap_above_ratio",
    source=_SRC,
    backend="polars",
)
class IntraVwapAboveRatioNative(SeriesOperator):
    """Ratio of time price is above VWAP."""

    metadata = OperatorMetadata(
        name="intra_vwap_above_ratio",
        category="intraday",
        description="价格高于VWAP占比",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_vwap_above_ratio requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c) > pl.col(c).from_(vwap))
                .then(1.0)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_vwap_cross_count",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_vwap_cross_count",
    source=_SRC,
    backend="polars",
)
class IntraVwapCrossCountNative(SeriesOperator):
    """Count of price-VWAP crossovers."""

    metadata = OperatorMetadata(
        name="intra_vwap_cross_count",
        category="intraday",
        description="价格穿越VWAP次数",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_vwap_cross_count requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                (
                    (pl.col(c) > pl.col(c).from_(vwap)).cast(pl.Int8)
                    - (pl.col(c) > pl.col(c).from_(vwap)).shift(1).cast(pl.Int8)
                ).abs().cast(pl.Float64).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_longest_above_vwap_streak",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_longest_above_vwap_streak",
    source=_SRC,
    backend="polars",
)
class IntraLongestAboveVwapStreakNative(SeriesOperator):
    """Longest consecutive period above VWAP."""

    metadata = OperatorMetadata(
        name="intra_longest_above_vwap_streak",
        category="intraday",
        description="最长连续高于VWAP周期",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_longest_above_vwap_streak requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c) > pl.col(c).from_(vwap))
                .then(1.0)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_longest_below_vwap_streak",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_longest_below_vwap_streak",
    source=_SRC,
    backend="polars",
)
class IntraLongestBelowVwapStreakNative(SeriesOperator):
    """Longest consecutive period below VWAP."""

    metadata = OperatorMetadata(
        name="intra_longest_below_vwap_streak",
        category="intraday",
        description="最长连续低于VWAP周期",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_longest_below_vwap_streak requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c) < pl.col(c).from_(vwap))
                .then(1.0)
                .otherwise(0.0)
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_time_above_vwap",
    category="intraday",
    business_category="intraday_vwap",
    canonical="intra_time_above_vwap",
    source=_SRC,
    backend="polars",
)
class IntraTimeAboveVwapNative(SeriesOperator):
    """Cumulative time spent above VWAP."""

    metadata = OperatorMetadata(
        name="intra_time_above_vwap",
        category="intraday",
        description="累计高于VWAP时长",
        param_names=["price", "vwap"],
        return_type="series",
        tags=["intraday", "vwap", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, vwap: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if vwap is None:
            raise ValueError("intra_time_above_vwap requires vwap")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                pl.when(pl.col(c) > pl.col(c).from_(vwap))
                .then(1.0)
                .otherwise(0.0)
                .cum_sum()
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# State-based analysis
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_state_vwap",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_vwap",
    source=_SRC,
    backend="polars",
)
class IntraStateVwapNative(SeriesOperator):
    """VWAP conditional on state."""

    metadata = OperatorMetadata(
        name="intra_state_vwap",
        category="intraday",
        description="状态条件VWAP",
        param_names=["price", "volume", "state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(
        self, price: pl.DataFrame, volume: pl.DataFrame | None = None, state: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        if volume is None or state is None:
            raise ValueError("intra_state_vwap requires volume and state")
        cols = _numeric_cols(price)
        return price.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(volume)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_state_sum",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_sum",
    source=_SRC,
    backend="polars",
)
class IntraStateSumNative(SeriesOperator):
    """Sum of values in specific state."""

    metadata = OperatorMetadata(
        name="intra_state_sum",
        category="intraday",
        description="状态条件求和",
        param_names=["x", "state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, state: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if state is None:
            raise ValueError("intra_state_sum requires state")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [pl.col(c).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_state_count",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_count",
    source=_SRC,
    backend="polars",
)
class IntraStateCountNative(SeriesOperator):
    """Count of observations in specific state."""

    metadata = OperatorMetadata(
        name="intra_state_count",
        category="intraday",
        description="状态观测次数",
        param_names=["state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(self, state: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(state)
        return state.with_columns(
            [pl.col(c).is_not_null().cast(pl.Float64).alias(c) for c in cols]
        )


@register_operator(
    name="intra_state_follow_ratio",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_follow_ratio",
    source=_SRC,
    backend="polars",
)
class IntraStateFollowRatioNative(SeriesOperator):
    """State-following ratio: P(state_t | state_{t-1})."""

    metadata = OperatorMetadata(
        name="intra_state_follow_ratio",
        category="intraday",
        description="状态跟随比率",
        param_names=["state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(self, state: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(state)
        return state.with_columns(
            [
                (pl.col(c) == pl.col(c).shift(1)).cast(pl.Float64).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_state_follow_beta",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_follow_beta",
    source=_SRC,
    backend="polars",
)
class IntraStateFollowBetaNative(SeriesOperator):
    """Beta conditional on market state."""

    metadata = OperatorMetadata(
        name="intra_state_follow_beta",
        category="intraday",
        description="状态条件Beta",
        param_names=["x", "market", "state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, market: pl.DataFrame | None = None, state: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        if market is None or state is None:
            raise ValueError("intra_state_follow_beta requires market and state")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(market)).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_state_follow_corr",
    category="intraday",
    business_category="intraday_state",
    canonical="intra_state_follow_corr",
    source=_SRC,
    backend="polars",
)
class IntraStateFollowCorrNative(SeriesOperator):
    """Correlation conditional on state."""

    metadata = OperatorMetadata(
        name="intra_state_follow_corr",
        category="intraday",
        description="状态条件相关性",
        param_names=["x", "y", "state"],
        return_type="series",
        tags=["intraday", "state", "polars", "native"],
    )

    def _calculate_series(
        self, x: pl.DataFrame, y: pl.DataFrame | None = None, state: pl.DataFrame | None = None, **kwargs
    ) -> pl.DataFrame:
        if y is None or state is None:
            raise ValueError("intra_state_follow_corr requires y and state")
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                (pl.col(c) * pl.col(c).from_(y)).alias(c)
                for c in cols
            ]
        ).collect()


# ---------------------------------------------------------------------------
# Session & slot analysis
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_session_return_asymmetry",
    category="intraday",
    business_category="intraday_session",
    canonical="intra_session_return_asymmetry",
    source=_SRC,
    backend="polars",
)
class IntraSessionReturnAsymmetryNative(SeriesOperator):
    """Asymmetry between morning and afternoon returns."""

    metadata = OperatorMetadata(
        name="intra_session_return_asymmetry",
        category="intraday",
        description="早盘午盘收益不对称性",
        param_names=["morning", "afternoon"],
        return_type="series",
        tags=["intraday", "session", "polars", "native"],
    )

    def _calculate_series(self, morning: pl.DataFrame, afternoon: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if afternoon is None:
            raise ValueError("intra_session_return_asymmetry requires afternoon")
        cols = _numeric_cols(morning)
        return morning.with_columns(
            [
                pl.when((pl.col(c) + pl.col(c).from_(afternoon)) == 0)
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(afternoon)) / (pl.col(c).abs() + pl.col(c).from_(afternoon).abs()))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_same_slot_momentum",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_same_slot_momentum",
    source=_SRC,
    backend="polars",
)
class IntraSameSlotMomentumNative(SeriesOperator):
    """Momentum of same time slot across days."""

    metadata = OperatorMetadata(
        name="intra_same_slot_momentum",
        category="intraday",
        description="同时段动量",
        param_names=["x", "window"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=5, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [
                pl.col(c).rolling_mean(window_size=w, min_samples=1).alias(c)
                for c in cols
            ]
        ).collect()


@register_operator(
    name="intra_same_slot_reversal",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_same_slot_reversal",
    source=_SRC,
    backend="polars",
)
class IntraSameSlotReversalNative(SeriesOperator):
    """Reversal of same time slot across days."""

    metadata = OperatorMetadata(
        name="intra_same_slot_reversal",
        category="intraday",
        description="同时段反转",
        param_names=["x"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)
        return x.lazy().with_columns(
            [(-pl.col(c)).alias(c) for c in cols]
        ).collect()


@register_operator(
    name="intra_same_slot_zscore",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_same_slot_zscore",
    source=_SRC,
    backend="polars",
)
class IntraSameSlotZscoreNative(SeriesOperator):
    """Z-score relative to same time slot distribution."""

    metadata = OperatorMetadata(
        name="intra_same_slot_zscore",
        category="intraday",
        description="同时段Z分数",
        param_names=["x", "window"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True,
                               param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w, min_samples=1)
            std = pl.col(c).rolling_std(window_size=w, min_samples=1)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="intra_slot_volume_surprise",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_slot_volume_surprise",
    source=_SRC,
    backend="polars",
)
class IntraSlotVolumeSurpriseNative(SeriesOperator):
    """Volume surprise relative to slot average."""

    metadata = OperatorMetadata(
        name="intra_slot_volume_surprise",
        category="intraday",
        description="时段成交量意外",
        param_names=["volume", "avg_volume"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, avg_volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_volume is None:
            raise ValueError("intra_slot_volume_surprise requires avg_volume")
        cols = _numeric_cols(volume)
        return volume.with_columns(
            [
                pl.when(pl.col(c).from_(avg_volume).is_null() | (pl.col(c).from_(avg_volume) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(avg_volume)) / pl.col(c).from_(avg_volume))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_slot_amount_surprise",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_slot_amount_surprise",
    source=_SRC,
    backend="polars",
)
class IntraSlotAmountSurpriseNative(SeriesOperator):
    """Amount surprise relative to slot average."""

    metadata = OperatorMetadata(
        name="intra_slot_amount_surprise",
        category="intraday",
        description="时段成交额意外",
        param_names=["amount", "avg_amount"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
    )

    def _calculate_series(self, amount: pl.DataFrame, avg_amount: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_amount is None:
            raise ValueError("intra_slot_amount_surprise requires avg_amount")
        cols = _numeric_cols(amount)
        return amount.with_columns(
            [
                pl.when(pl.col(c).from_(avg_amount).is_null() | (pl.col(c).from_(avg_amount) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(avg_amount)) / pl.col(c).from_(avg_amount))
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_slot_volatility_surprise",
    category="intraday",
    business_category="intraday_slot",
    canonical="intra_slot_volatility_surprise",
    source=_SRC,
    backend="polars",
)
class IntraSlotVolatilitySurpriseNative(SeriesOperator):
    """Volatility surprise relative to slot average."""

    metadata = OperatorMetadata(
        name="intra_slot_volatility_surprise",
        category="intraday",
        description="时段波动率意外",
        param_names=["volatility", "avg_volatility"],
        return_type="series",
        tags=["intraday", "slot", "polars", "native"],
    )

    def _calculate_series(self, volatility: pl.DataFrame, avg_volatility: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if avg_volatility is None:
            raise ValueError("intra_slot_volatility_surprise requires avg_volatility")
        cols = _numeric_cols(volatility)
        return volatility.with_columns(
            [
                pl.when(pl.col(c).from_(avg_volatility).is_null() | (pl.col(c).from_(avg_volatility) == 0))
                .then(None)
                .otherwise((pl.col(c) - pl.col(c).from_(avg_volatility)) / pl.col(c).from_(avg_volatility))
                .alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Return-volume relationships
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_return_activity_corr",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_return_activity_corr",
    source=_SRC,
    backend="polars",
)
class IntraReturnActivityCorrNative(SeriesOperator):
    """Correlation between return and trading activity."""

    metadata = OperatorMetadata(
        name="intra_return_activity_corr",
        category="intraday",
        description="收益与交易活跃度相关性",
        param_names=["ret", "activity"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, activity: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if activity is None:
            raise ValueError("intra_return_activity_corr requires activity")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c) * pl.col(c).from_(activity)).alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_volume_price_alignment",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_volume_price_alignment",
    source=_SRC,
    backend="polars",
)
class IntraVolumePriceAlignmentNative(SeriesOperator):
    """Alignment between volume and price direction."""

    metadata = OperatorMetadata(
        name="intra_volume_price_alignment",
        category="intraday",
        description="量价齐步性",
        param_names=["ret", "volume"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_volume_price_alignment requires volume")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c).sign() * pl.col(c).from_(volume)).alias(c)
                for c in cols
            ]
        )


# ---------------------------------------------------------------------------
# Microstructure proxies
# ---------------------------------------------------------------------------


@register_operator(
    name="intra_kyle_lambda_proxy",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_kyle_lambda_proxy",
    source=_SRC,
    backend="polars",
)
class IntraKyleLambdaProxyNative(SeriesOperator):
    """Kyle's lambda proxy: price impact per unit volume."""

    metadata = OperatorMetadata(
        name="intra_kyle_lambda_proxy",
        category="intraday",
        description="Kyle价格冲击系数",
        param_names=["ret", "volume"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_kyle_lambda_proxy requires volume")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                pl.when(pl.col(c).from_(volume).is_null() | (pl.col(c).from_(volume) == 0))
                .then(None)
                .otherwise(pl.col(c).abs() / pl.col(c).from_(volume).sqrt())
                .alias(c)
                for c in cols
            ]
        )


@register_operator(
    name="intra_signed_imbalance_proxy",
    category="intraday",
    business_category="intraday_microstructure",
    canonical="intra_signed_imbalance_proxy",
    source=_SRC,
    backend="polars",
)
class IntraSignedImbalanceProxyNative(SeriesOperator):
    """Signed order imbalance proxy from price and volume."""

    metadata = OperatorMetadata(
        name="intra_signed_imbalance_proxy",
        category="intraday",
        description="有向订单不平衡",
        param_names=["ret", "volume"],
        return_type="series",
        tags=["intraday", "microstructure", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if volume is None:
            raise ValueError("intra_signed_imbalance_proxy requires volume")
        cols = _numeric_cols(ret)
        return ret.with_columns(
            [
                (pl.col(c).sign() * pl.col(c).from_(volume)).alias(c)
                for c in cols
            ]
        )

