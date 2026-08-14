# -*- coding: utf-8 -*-
"""OHLC and price-based operators - Polars native implementations.

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
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


# ---------------------------------------------------------------------------
# Basic OHLC returns
# ---------------------------------------------------------------------------


@register_operator(
    name="open_close_return",
    category="ohlc",
    business_category="ohlc_price",
    canonical="open_close_return",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class OpenCloseReturnNative(SeriesOperator):
    """Intraday return: (close - open) / open."""

    metadata = OperatorMetadata(
        name="open_close_return",
        category="ohlc",
        description="日内收益",
        param_names=["open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            if c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(
                    pl.when(pl.col(c).is_null() | (pl.col(c) == 0))
                    .then(None)
                    .otherwise((close[c] - pl.col(c)) / pl.col(c))
                    .alias(c)
                )
        return open.with_columns(exprs)


@register_operator(
    name="overnight_return",
    category="ohlc",
    business_category="ohlc_price",
    canonical="overnight_return",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class OvernightReturnNative(SeriesOperator):
    """Overnight return: (open_t - close_{t-1}) / close_{t-1}."""

    metadata = OperatorMetadata(
        name="overnight_return",
        category="ohlc",
        description="隔夜收益",
        param_names=["open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            if c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                prev_close = close[c].shift(1)
                exprs.append(
                    pl.when(prev_close.is_null() | (prev_close == 0))
                    .then(None)
                    .otherwise((pl.col(c) - prev_close) / prev_close)
                    .alias(c)
                )
        return open.with_columns(exprs)


@register_operator(
    name="overnight_volatility",
    category="ohlc",
    business_category="ohlc_price",
    canonical="overnight_volatility",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class OvernightVolatilityNative(SeriesOperator):
    """Rolling std of overnight returns."""

    metadata = OperatorMetadata(
        name="overnight_volatility",
        category="ohlc",
        description="隔夜波动率",
        param_names=["open", "close", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            if c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                prev_close = close[c].shift(1)
                overnight_ret = pl.when(prev_close.is_null() | (prev_close == 0)).then(None).otherwise((pl.col(c) - prev_close) / prev_close)
                exprs.append(overnight_ret.rolling_std(window_size=w).alias(c))
        return open.with_columns(exprs)


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------


@register_operator(
    name="true_range",
    category="ohlc",
    business_category="ohlc_price",
    canonical="true_range",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class TrueRangeNative(SeriesOperator):
    """True range: max(high-low, |high-prev_close|, |low-prev_close|)."""

    metadata = OperatorMetadata(
        name="true_range",
        category="ohlc",
        description="真实波幅",
        param_names=["high", "low", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                prev_close = close[c].shift(1)
                tr1 = pl.col(c) - low[c]
                tr2 = (pl.col(c) - prev_close).abs()
                tr3 = (low[c] - prev_close).abs()
                exprs.append(pl.max_horizontal([tr1, tr2, tr3]).alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="garman_klass_vol",
    category="ohlc",
    business_category="ohlc_price",
    canonical="garman_klass_vol",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class GarmanKlassVolNative(SeriesOperator):
    """Garman-Klass volatility estimator."""

    metadata = OperatorMetadata(
        name="garman_klass_vol",
        category="ohlc",
        description="Garman-Klass波动率",
        param_names=["high", "low", "open", "close", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                hl = (pl.col(c) / low[c]).log() ** 2
                co = (close[c] / open[c]).log() ** 2
                gk = (0.5 * hl - (2 * 2.0 ** 0.5 - 1) * co).rolling_mean(window_size=w).sqrt()
                exprs.append(gk.alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="parkinson_vol",
    category="ohlc",
    business_category="ohlc_price",
    canonical="parkinson_vol",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class ParkinsonVolNative(SeriesOperator):
    """Parkinson volatility: sqrt(mean(log(H/L)^2 / (4*ln(2))))."""

    metadata = OperatorMetadata(
        name="parkinson_vol",
        category="ohlc",
        description="Parkinson波动率",
        param_names=["high", "low", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        import math

        w = strict_integer(d, "d", minimum=2)
        factor = 1.0 / (4 * math.log(2))
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                log_hl = (pl.col(c) / low[c]).log()
                pv = (log_hl ** 2 * factor).rolling_mean(window_size=w).sqrt()
                exprs.append(pv.alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="rogers_satchell_vol",
    category="ohlc",
    business_category="ohlc_price",
    canonical="rogers_satchell_vol",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class RogersSatchellVolNative(SeriesOperator):
    """Rogers-Satchell volatility estimator."""

    metadata = OperatorMetadata(
        name="rogers_satchell_vol",
        category="ohlc",
        description="Rogers-Satchell波动率",
        param_names=["high", "low", "open", "close", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                hc = (pl.col(c) / close[c]).log()
                ho = (pl.col(c) / open[c]).log()
                lc = (low[c] / close[c]).log()
                lo = (low[c] / open[c]).log()
                rs = (hc * ho + lc * lo).rolling_mean(window_size=w).sqrt()
                exprs.append(rs.alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="yang_zhang_vol",
    category="ohlc",
    business_category="ohlc_price",
    canonical="yang_zhang_vol",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class YangZhangVolNative(SeriesOperator):
    """Yang-Zhang volatility (simplified)."""

    metadata = OperatorMetadata(
        name="yang_zhang_vol",
        category="ohlc",
        description="Yang-Zhang波动率",
        param_names=["high", "low", "open", "close", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                # Simplified: use combination of overnight and RS
                prev_close = close[c].shift(1)
                overnight = (open[c] / prev_close).log() ** 2
                hc = (pl.col(c) / close[c]).log()
                ho = (pl.col(c) / open[c]).log()
                lc = (low[c] / close[c]).log()
                lo = (low[c] / open[c]).log()
                rs_sq = hc * ho + lc * lo
                yz = (overnight + rs_sq).rolling_mean(window_size=w).sqrt()
                exprs.append(yz.alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="range_volatility",
    category="ohlc",
    business_category="ohlc_price",
    canonical="range_volatility",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class RangeVolatilityNative(SeriesOperator):
    """Rolling std of (high - low) / close."""

    metadata = OperatorMetadata(
        name="range_volatility",
        category="ohlc",
        description="价格区间波动率",
        param_names=["high", "low", "close", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                range_pct = (pl.col(c) - low[c]) / close[c]
                exprs.append(range_pct.rolling_std(window_size=w).alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="intraday_volatility",
    category="ohlc",
    business_category="ohlc_price",
    canonical="intraday_volatility",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class IntradayVolatilityNative(SeriesOperator):
    """Rolling std of (high - low)."""

    metadata = OperatorMetadata(
        name="intraday_volatility",
        category="ohlc",
        description="日内波动率",
        param_names=["high", "low", "d"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                range_val = pl.col(c) - low[c]
                exprs.append(range_val.rolling_std(window_size=w).alias(c))
        return high.with_columns(exprs)


# ---------------------------------------------------------------------------
# Candle patterns
# ---------------------------------------------------------------------------


@register_operator(
    name="candle_body",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_body",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleBodyNative(SeriesOperator):
    """Candle body: close - open."""

    metadata = OperatorMetadata(
        name="candle_body",
        category="ohlc",
        description="K线实体",
        param_names=["close", "open"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, close: pl.DataFrame, open: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            if c not in open.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) - open[c]).alias(c))
        return close.with_columns(exprs)


@register_operator(
    name="candle_abs_body",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_abs_body",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleAbsBodyNative(SeriesOperator):
    """Absolute candle body: |close - open|."""

    metadata = OperatorMetadata(
        name="candle_abs_body",
        category="ohlc",
        description="K线实体绝对值",
        param_names=["close", "open"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, close: pl.DataFrame, open: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            if c not in open.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) - open[c]).abs().alias(c))
        return close.with_columns(exprs)


@register_operator(
    name="candle_range",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_range",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleRangeNative(SeriesOperator):
    """Candle range: high - low."""

    metadata = OperatorMetadata(
        name="candle_range",
        category="ohlc",
        description="K线区间",
        param_names=["high", "low"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) - low[c]).alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="candle_upper_shadow",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_upper_shadow",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleUpperShadowNative(SeriesOperator):
    """Upper shadow: high - max(open, close)."""

    metadata = OperatorMetadata(
        name="candle_upper_shadow",
        category="ohlc",
        description="上影线",
        param_names=["high", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body_top = pl.max_horizontal([open[c], close[c]])
                exprs.append((pl.col(c) - body_top).alias(c))
        return high.with_columns(exprs)


@register_operator(
    name="candle_lower_shadow",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_lower_shadow",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleLowerShadowNative(SeriesOperator):
    """Lower shadow: min(open, close) - low."""

    metadata = OperatorMetadata(
        name="candle_lower_shadow",
        category="ohlc",
        description="下影线",
        param_names=["low", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(low)
        exprs = []
        for c in cols:
            if c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body_bottom = pl.min_horizontal([open[c], close[c]])
                exprs.append((body_bottom - pl.col(c)).alias(c))
        return low.with_columns(exprs)


@register_operator(
    name="candle_body_ratio",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_body_ratio",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleBodyRatioNative(SeriesOperator):
    """Body ratio: |close - open| / (high - low)."""

    metadata = OperatorMetadata(
        name="candle_body_ratio",
        category="ohlc",
        description="K线实体比例",
        param_names=["high", "low", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body = (close[c] - open[c]).abs()
                range_val = pl.col(c) - low[c]
                exprs.append(
                    pl.when(range_val.is_null() | (range_val == 0))
                    .then(None)
                    .otherwise(body / range_val)
                    .alias(c)
                )
        return high.with_columns(exprs)


@register_operator(
    name="candle_upper_shadow_ratio",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_upper_shadow_ratio",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleUpperShadowRatioNative(SeriesOperator):
    """Upper shadow ratio: upper_shadow / range."""

    metadata = OperatorMetadata(
        name="candle_upper_shadow_ratio",
        category="ohlc",
        description="上影线比例",
        param_names=["high", "low", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body_top = pl.max_horizontal([open[c], close[c]])
                upper_shadow = pl.col(c) - body_top
                range_val = pl.col(c) - low[c]
                exprs.append(
                    pl.when(range_val.is_null() | (range_val == 0))
                    .then(None)
                    .otherwise(upper_shadow / range_val)
                    .alias(c)
                )
        return high.with_columns(exprs)


@register_operator(
    name="candle_lower_shadow_ratio",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_lower_shadow_ratio",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleLowerShadowRatioNative(SeriesOperator):
    """Lower shadow ratio: lower_shadow / range."""

    metadata = OperatorMetadata(
        name="candle_lower_shadow_ratio",
        category="ohlc",
        description="下影线比例",
        param_names=["high", "low", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body_bottom = pl.min_horizontal([open[c], close[c]])
                lower_shadow = body_bottom - low[c]
                range_val = pl.col(c) - low[c]
                exprs.append(
                    pl.when(range_val.is_null() | (range_val == 0))
                    .then(None)
                    .otherwise(lower_shadow / range_val)
                    .alias(c)
                )
        return high.with_columns(exprs)


@register_operator(
    name="candle_direction",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_direction",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleDirectionNative(SeriesOperator):
    """Candle direction: sign(close - open)."""

    metadata = OperatorMetadata(
        name="candle_direction",
        category="ohlc",
        description="K线方向",
        param_names=["close", "open"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, close: pl.DataFrame, open: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            if c not in open.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) - open[c]).sign().alias(c))
        return close.with_columns(exprs)


@register_operator(
    name="candle_gap",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_gap",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleGapNative(SeriesOperator):
    """Gap: open_t - close_{t-1}."""

    metadata = OperatorMetadata(
        name="candle_gap",
        category="ohlc",
        description="跳空缺口",
        param_names=["open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            if c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                prev_close = close[c].shift(1)
                exprs.append((pl.col(c) - prev_close).alias(c))
        return open.with_columns(exprs)


@register_operator(
    name="candle_gap_pct",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_gap_pct",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleGapPctNative(SeriesOperator):
    """Gap percent: (open_t - close_{t-1}) / close_{t-1}."""

    metadata = OperatorMetadata(
        name="candle_gap_pct",
        category="ohlc",
        description="跳空缺口百分比",
        param_names=["open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            if c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                prev_close = close[c].shift(1)
                exprs.append(
                    pl.when(prev_close.is_null() | (prev_close == 0))
                    .then(None)
                    .otherwise((pl.col(c) - prev_close) / prev_close)
                    .alias(c)
                )
        return open.with_columns(exprs)


@register_operator(
    name="candle_close_location",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_close_location",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleCloseLocationNative(SeriesOperator):
    """Close location within range: (close - low) / (high - low)."""

    metadata = OperatorMetadata(
        name="candle_close_location",
        category="ohlc",
        description="收盘位置",
        param_names=["high", "low", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                range_val = pl.col(c) - low[c]
                exprs.append(
                    pl.when(range_val.is_null() | (range_val == 0))
                    .then(None)
                    .otherwise((close[c] - low[c]) / range_val)
                    .alias(c)
                )
        return high.with_columns(exprs)


@register_operator(
    name="candle_body_position",
    category="ohlc",
    business_category="ohlc_price",
    canonical="candle_body_position",
    source=_SRC,
    backend="polars",
    research_only=True,
)
class CandleBodyPositionNative(SeriesOperator):
    """Body position: (open + close) / 2 relative to range."""

    metadata = OperatorMetadata(
        name="candle_body_position",
        category="ohlc",
        description="实体位置",
        param_names=["high", "low", "open", "close"],
        return_type="series",
        tags=["ohlc", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            if c not in low.columns or c not in open.columns or c not in close.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                body_mid = (open[c] + close[c]) / 2
                range_val = pl.col(c) - low[c]
                exprs.append(
                    pl.when(range_val.is_null() | (range_val == 0))
                    .then(None)
                    .otherwise((body_mid - low[c]) / range_val)
                    .alias(c)
                )
        return high.with_columns(exprs)

