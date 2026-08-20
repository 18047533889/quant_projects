# -*- coding: utf-8 -*-
"""Liquidity and spread operators - Polars native implementations.

Illiquidity measures (Amihud), spread proxies (Roll, Corwin-Schultz, high-low),
bounded volume indicators (NVI, PVI), and float/turnover metrics.
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


# ---------------------------------------------------------------------------
# Illiquidity measures
# ---------------------------------------------------------------------------

@register_operator(
    name="amihud_illiquidity",
    category="liquidity",
    business_category="liquidity",
    canonical="amihud_illiquidity",
    source=_SRC,
    backend="polars",
)
class AmihudIlliquidityNative(SeriesOperator):
    """Amihud illiquidity: |return| / volume; zero volume → null."""

    metadata = OperatorMetadata(
        name="amihud_illiquidity",
        category="liquidity",
        description="Amihud非流动性指标",
        param_names=["returns", "volume"],
        return_type="series",
        tags=["liquidity", "polars", "native"],
    )

    def _calculate_series(
        self, returns: pl.DataFrame, volume: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            vol = volume[c] if c in volume.columns else pl.lit(None)
            exprs.append(
                pl.when((vol.is_null()) | (vol == 0))
                .then(None)
                .otherwise(ret.abs() / vol)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, returns)


@register_operator(
    name="price_impact",
    category="liquidity",
    business_category="liquidity",
    canonical="price_impact",
    source=_SRC,
    backend="polars",
)
class PriceImpactNative(SeriesOperator):
    """Price impact: |return| / sqrt(volume); zero volume → null."""

    metadata = OperatorMetadata(
        name="price_impact",
        category="liquidity",
        description="价格冲击",
        param_names=["returns", "volume"],
        return_type="series",
        tags=["liquidity", "polars", "native"],
    )

    def _calculate_series(
        self, returns: pl.DataFrame, volume: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            vol = volume[c] if c in volume.columns else pl.lit(None)
            exprs.append(
                pl.when((vol.is_null()) | (vol <= 0))
                .then(None)
                .otherwise(ret.abs() / vol.sqrt())
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, returns)


# ---------------------------------------------------------------------------
# Spread proxies
# ---------------------------------------------------------------------------

@register_operator(
    name="roll_spread_proxy",
    category="liquidity",
    business_category="spread",
    canonical="roll_spread_proxy",
    source=_SRC,
    backend="polars",
)
class RollSpreadProxyNative(SeriesOperator):
    """Roll spread: 2 * sqrt(-cov(r_t, r_{t-1})); negative cov → null."""

    metadata = OperatorMetadata(
        name="roll_spread_proxy",
        category="liquidity",
        description="Roll买卖价差代理",
        param_names=["returns", "window"],
        return_type="series",
        tags=["liquidity", "spread", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, returns: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            lag1 = ret.shift(1)
            # Rolling covariance: cov = E[XY] - E[X]E[Y]
            mean_ret = ret.rolling_mean(window_size=w)
            mean_lag = lag1.rolling_mean(window_size=w)
            mean_prod = (ret * lag1).rolling_mean(window_size=w)
            cov = mean_prod - mean_ret * mean_lag
            exprs.append(
                pl.when(cov >= 0)
                .then(None)
                .otherwise(pl.lit(2.0) * (-cov).sqrt())
                .alias(c)
            )
        result = returns.with_columns(exprs)
        return result


@register_operator(
    name="corwin_schultz_spread",
    category="liquidity",
    business_category="spread",
    canonical="corwin_schultz_spread",
    source=_SRC,
    backend="polars",
)
class CorwinSchultzSpreadNative(SeriesOperator):
    """Corwin-Schultz spread from high-low ratio variance."""

    metadata = OperatorMetadata(
        name="corwin_schultz_spread",
        category="liquidity",
        description="Corwin-Schultz买卖价差",
        param_names=["high_low_ratio"],
        return_type="series",
        tags=["liquidity", "spread", "polars", "native"],
    )

    def _calculate_series(
        self, high_low_ratio: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(high_low_ratio)
        exprs = []
        for c in cols:
            hl = high_low_ratio[c]
            # Beta = E[ln(H/L)^2]
            beta = (hl.log() ** 2).rolling_mean(window_size=2, min_samples=2)
            # Gamma = [ln(H2/L2)]^2 where H2=max(H_t, H_{t-1}), L2=min(L_t, L_{t-1})
            # For simplicity, approximate with 2-day range
            spread_est = ((pl.lit(2.0) * (pl.lit(2.0).exp() - pl.lit(1.0)).sqrt() - pl.lit(1.0)) / (
                pl.lit(3.0) - pl.lit(2.0) * pl.lit(2.0).sqrt()
            )) * beta.sqrt()
            exprs.append(
                pl.when(beta <= 0).then(None).otherwise(spread_est).alias(c)
            )
        result = high_low_ratio.with_columns(exprs)
        return result


@register_operator(
    name="ohlc_corwin_schultz_spread",
    category="liquidity",
    business_category="spread",
    canonical="ohlc_corwin_schultz_spread",
    source=_SRC,
    backend="polars",
)
class OHLCCorwinSchultzSpreadNative(SeriesOperator):
    """Corwin-Schultz from OHLC directly."""

    metadata = OperatorMetadata(
        name="ohlc_corwin_schultz_spread",
        category="liquidity",
        description="OHLC Corwin-Schultz价差",
        param_names=["high", "low"],
        return_type="series",
        tags=["liquidity", "spread", "polars", "native"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            h = high[c]
            lo = low[c] if c in low.columns else pl.lit(None)
            hl_ratio = pl.when(lo != 0).then(h / lo).otherwise(None)
            beta = (hl_ratio.log() ** 2).rolling_mean(window_size=2, min_samples=2)
            alpha = (
                (pl.lit(2.0).sqrt() - pl.lit(1.0))
                / (pl.lit(3.0) - pl.lit(2.0) * pl.lit(2.0).sqrt())
            )
            spread = alpha * beta.sqrt()
            exprs.append(
                pl.when((beta.is_null()) | (beta <= 0)).then(None).otherwise(spread).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, high)


@register_operator(
    name="high_low_spread_proxy",
    category="liquidity",
    business_category="spread",
    canonical="high_low_spread_proxy",
    source=_SRC,
    backend="polars",
)
class HighLowSpreadProxyNative(SeriesOperator):
    """Simple high-low spread: (high - low) / ((high + low) / 2)."""

    metadata = OperatorMetadata(
        name="high_low_spread_proxy",
        category="liquidity",
        description="高低价差代理",
        param_names=["high", "low"],
        return_type="series",
        tags=["liquidity", "spread", "polars", "native"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            h = high[c]
            lo = low[c] if c in low.columns else pl.lit(None)
            mid = (h + lo) / pl.lit(2.0)
            exprs.append(
                pl.when((mid.is_null()) | (mid == 0))
                .then(None)
                .otherwise((h - lo) / mid)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, high)


# ---------------------------------------------------------------------------
# Time-series spread estimators
# ---------------------------------------------------------------------------

@register_operator(
    name="ts_abdi_ranaldo_spread",
    category="liquidity",
    business_category="spread",
    canonical="ts_abdi_ranaldo_spread",
    source=_SRC,
    backend="polars",
)
class TSAbdiRanaldoSpreadNative(SeriesOperator):
    """Abdi-Ranaldo spread: rolling avg of 2 * sqrt(|r_t * r_{t-1}|)."""

    metadata = OperatorMetadata(
        name="ts_abdi_ranaldo_spread",
        category="liquidity",
        description="Abdi-Ranaldo时序价差",
        param_names=["returns", "window"],
        return_type="series",
        tags=["liquidity", "spread", "time_series", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, returns: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            lag1 = ret.shift(1)
            prod = ret * lag1
            spread = pl.lit(2.0) * prod.abs().sqrt()
            exprs.append(spread.rolling_mean(window_size=w).alias(c))
        result = returns.with_columns(exprs)
        return result


@register_operator(
    name="ts_roll_effective_spread",
    category="liquidity",
    business_category="spread",
    canonical="ts_roll_effective_spread",
    source=_SRC,
    backend="polars",
)
class TSRollEffectiveSpreadNative(SeriesOperator):
    """Rolling effective spread: rolling average of 2 * |r_t|."""

    metadata = OperatorMetadata(
        name="ts_roll_effective_spread",
        category="liquidity",
        description="Roll有效价差时序",
        param_names=["returns", "window"],
        return_type="series",
        tags=["liquidity", "spread", "time_series", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, returns: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            exprs.append(
                (pl.lit(2.0) * ret.abs()).rolling_mean(window_size=w).alias(c)
            )
        result = returns.with_columns(exprs)
        return result


@register_operator(
    name="ts_edge_effective_spread",
    category="liquidity",
    business_category="spread",
    canonical="ts_edge_effective_spread",
    source=_SRC,
    backend="polars",
)
class TSEdgeEffectiveSpreadNative(SeriesOperator):
    """Edge effective spread: rolling quantile(|return|, 0.75)."""

    metadata = OperatorMetadata(
        name="ts_edge_effective_spread",
        category="liquidity",
        description="边缘有效价差",
        param_names=["returns", "window"],
        return_type="series",
        tags=["liquidity", "spread", "time_series", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, returns: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            exprs.append(
                ret.abs().rolling_quantile(quantile=0.75, window_size=w, interpolation="linear").alias(c)
            )
        result = returns.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Bounded volume indicators
# ---------------------------------------------------------------------------

@register_operator(
    name="bounded_nvi",
    category="liquidity",
    business_category="volume_indicator",
    canonical="bounded_nvi",
    source=_SRC,
    backend="polars",
)
class BoundedNVINative(SeriesOperator):
    """Negative Volume Index: cumulative return on down-volume days, clipped."""

    metadata = OperatorMetadata(
        name="bounded_nvi",
        category="liquidity",
        description="有界负成交量指标",
        param_names=["returns", "volume"],
        return_type="series",
        tags=["liquidity", "volume", "polars", "native"],
    )

    def _calculate_series(
        self, returns: pl.DataFrame, volume: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            vol = volume[c] if c in volume.columns else pl.lit(None)
            vol_change = vol - vol.shift(1)
            # Accumulate returns when volume decreases
            contrib = pl.when(vol_change < 0).then(ret).otherwise(0)
            nvi = contrib.cum_sum().clip(-1.0, 1.0)
            exprs.append(nvi.alias(c))
        result = returns.with_columns(exprs)
        return result


@register_operator(
    name="bounded_pvi",
    category="liquidity",
    business_category="volume_indicator",
    canonical="bounded_pvi",
    source=_SRC,
    backend="polars",
)
class BoundedPVINative(SeriesOperator):
    """Positive Volume Index: cumulative return on up-volume days, clipped."""

    metadata = OperatorMetadata(
        name="bounded_pvi",
        category="liquidity",
        description="有界正成交量指标",
        param_names=["returns", "volume"],
        return_type="series",
        tags=["liquidity", "volume", "polars", "native"],
    )

    def _calculate_series(
        self, returns: pl.DataFrame, volume: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(returns)
        exprs = []
        for c in cols:
            ret = returns[c]
            vol = volume[c] if c in volume.columns else pl.lit(None)
            vol_change = vol - vol.shift(1)
            # Accumulate returns when volume increases
            contrib = pl.when(vol_change > 0).then(ret).otherwise(0)
            pvi = contrib.cum_sum().clip(-1.0, 1.0)
            exprs.append(pvi.alias(c))
        result = returns.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Float and turnover metrics
# ---------------------------------------------------------------------------

@register_operator(
    name="free_float_ratio",
    category="liquidity",
    business_category="float",
    canonical="free_float_ratio",
    source=_SRC,
    backend="polars",
)
class FreeFloatRatioNative(SeriesOperator):
    """Free float / total shares; zero total → null."""

    metadata = OperatorMetadata(
        name="free_float_ratio",
        category="liquidity",
        description="自由流通比例",
        param_names=["free_float", "total_shares"],
        return_type="series",
        tags=["liquidity", "float", "polars", "native"],
    )

    def _calculate_series(
        self, free_float: pl.DataFrame, total_shares: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(free_float)
        exprs = []
        for c in cols:
            ff = free_float[c]
            ts = total_shares[c] if c in total_shares.columns else pl.lit(None)
            exprs.append(
                pl.when((ts.is_null()) | (ts == 0))
                .then(None)
                .otherwise(ff / ts)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, free_float)


@register_operator(
    name="free_float_share_ratio",
    category="liquidity",
    business_category="float",
    canonical="free_float_share_ratio",
    source=_SRC,
    backend="polars",
)
class FreeFloatShareRatioNative(SeriesOperator):
    """Alias of free_float_ratio."""

    metadata = OperatorMetadata(
        name="free_float_share_ratio",
        category="liquidity",
        description="自由流通股比例",
        param_names=["free_float", "total_shares"],
        return_type="series",
        tags=["liquidity", "float", "polars", "native"],
    )

    def _calculate_series(
        self, free_float: pl.DataFrame, total_shares: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(free_float)
        exprs = []
        for c in cols:
            ff = free_float[c]
            ts = total_shares[c] if c in total_shares.columns else pl.lit(None)
            exprs.append(
                pl.when((ts.is_null()) | (ts == 0))
                .then(None)
                .otherwise(ff / ts)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, free_float)


@register_operator(
    name="free_to_circulating_ratio",
    category="liquidity",
    business_category="float",
    canonical="free_to_circulating_ratio",
    source=_SRC,
    backend="polars",
)
class FreeToCirculatingRatioNative(SeriesOperator):
    """Free float / circulating shares."""

    metadata = OperatorMetadata(
        name="free_to_circulating_ratio",
        category="liquidity",
        description="自由流通占流通股比例",
        param_names=["free_float", "circulating_shares"],
        return_type="series",
        tags=["liquidity", "float", "polars", "native"],
    )

    def _calculate_series(
        self, free_float: pl.DataFrame, circulating_shares: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(free_float)
        exprs = []
        for c in cols:
            ff = free_float[c]
            cs = circulating_shares[c] if c in circulating_shares.columns else pl.lit(None)
            exprs.append(
                pl.when((cs.is_null()) | (cs == 0))
                .then(None)
                .otherwise(ff / cs)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, free_float)


@register_operator(
    name="free_float_turnover",
    category="liquidity",
    business_category="float",
    canonical="free_float_turnover",
    source=_SRC,
    backend="polars",
)
class FreeFloatTurnoverNative(SeriesOperator):
    """Volume / free_float; zero free_float → null."""

    metadata = OperatorMetadata(
        name="free_float_turnover",
        category="liquidity",
        description="自由流通换手率",
        param_names=["volume", "free_float"],
        return_type="series",
        tags=["liquidity", "turnover", "polars", "native"],
    )

    def _calculate_series(
        self, volume: pl.DataFrame, free_float: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            vol = volume[c]
            ff = free_float[c] if c in free_float.columns else pl.lit(None)
            exprs.append(
                pl.when((ff.is_null()) | (ff == 0))
                .then(None)
                .otherwise(vol / ff)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, volume)


@register_operator(
    name="float_share_ratio",
    category="liquidity",
    business_category="float",
    canonical="float_share_ratio",
    source=_SRC,
    backend="polars",
)
class FloatShareRatioNative(SeriesOperator):
    """Circulating shares / total shares."""

    metadata = OperatorMetadata(
        name="float_share_ratio",
        category="liquidity",
        description="流通股比例",
        param_names=["circulating_shares", "total_shares"],
        return_type="series",
        tags=["liquidity", "float", "polars", "native"],
    )

    def _calculate_series(
        self, circulating_shares: pl.DataFrame, total_shares: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(circulating_shares)
        exprs = []
        for c in cols:
            cs = circulating_shares[c]
            ts = total_shares[c] if c in total_shares.columns else pl.lit(None)
            exprs.append(
                pl.when((ts.is_null()) | (ts == 0))
                .then(None)
                .otherwise(cs / ts)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, circulating_shares)


@register_operator(
    name="fundamental_staleness",
    category="liquidity",
    business_category="staleness",
    canonical="fundamental_staleness",
    source=_SRC,
    backend="polars",
)
class FundamentalStalenessNative(SeriesOperator):
    """Days since last non-null fundamental value."""

    metadata = OperatorMetadata(
        name="fundamental_staleness",
        category="liquidity",
        description="基本面数据陈旧度",
        param_names=["fundamental"],
        return_type="series",
        tags=["liquidity", "staleness", "polars", "native"],
    )

    def _calculate_series(
        self, fundamental: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(fundamental)
        exprs = []
        for c in cols:
            val = fundamental[c]
            # Create counter: increment on null, reset to 0 on non-null
            is_null = val.is_null().cast(pl.Int32)
            # Use cumsum to track blocks, then within each block count nulls
            block = (~val.is_null()).cum_sum()
            staleness = is_null.cum_sum() - is_null.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(staleness.alias(c))
        result = fundamental.with_columns(exprs)
        return result
