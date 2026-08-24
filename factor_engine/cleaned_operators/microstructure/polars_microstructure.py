# -*- coding: utf-8 -*-
"""微观结构算子 Polars 实现。

``real_turnover_rate`` = volume / float_shares（真实流通盘换手率）。
宽表逐列相除，列名需与 volume、float_shares 对齐。
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _align_cols(*dfs: pl.DataFrame) -> list[str]:
    cols = _numeric_cols(dfs[0])
    for df in dfs[1:]:
        cols = [c for c in cols if c in df.columns]
    return cols


@register_operator(
    name="real_turnover_rate",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="real_turnover_rate",
    source="factor_dsl_polars",
)
class RealTurnoverRatePolars(SeriesOperator):
    """Polars 真实换手率 volume/float"""
    metadata = OperatorMetadata(
        name="real_turnover_rate",
        category="intraday_microstructure",
        description="真实换手率 volume/float",
        param_names=["volume", "float_shares"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, volume: pl.DataFrame, float_shares: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _align_cols(volume, float_shares)
        return volume.with_columns([
            (volume[c] / float_shares[c]).alias(c) for c in cols
        ])


@register_operator(
    name="micro_realized_vol",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_realized_vol",
    source="factor_dsl_polars",
)
class MicroRealizedVolPolars(SeriesOperator):
    """Polars 已实现波动率"""
    metadata = OperatorMetadata(
        name="micro_realized_vol",
        category="intraday_microstructure",
        description="已实现波动率",
        param_names=["close", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = max(1, int(window))
        cols = _numeric_cols(close)
        out_exprs = []
        for c in cols:
            ret = close[c] / close[c].shift(1) - 1.0
            out_exprs.append(ret.pow(2).rolling_sum(w).sqrt().alias(c))
        return close.with_columns(out_exprs)


@register_operator(
    name="micro_spread",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_spread",
    source="factor_dsl_polars",
)
class MicroSpreadPolars(SeriesOperator):
    """Polars 相对价差"""
    metadata = OperatorMetadata(
        name="micro_spread",
        category="intraday_microstructure",
        description="相对价差",
        param_names=["high", "low", "close"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _align_cols(high, low, close)
        return high.with_columns([
            ((high[c] - low[c]) / close[c].replace(0.0, None)).alias(c) for c in cols
        ])


@register_operator(
    name="micro_amihud_hf",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_amihud_hf",
    source="factor_dsl_polars",
)
class MicroAmihudHfPolars(SeriesOperator):
    """Polars Amihud illiquidity"""
    metadata = OperatorMetadata(
        name="micro_amihud_hf",
        category="intraday_microstructure",
        description="Amihud illiquidity",
        param_names=["close", "volume"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _align_cols(close, volume)
        return close.with_columns([
            ((close[c] / close[c].shift(1) - 1.0).abs() / (close[c] * volume[c]).replace(0.0, None)).alias(c)
            for c in cols
        ])


@register_operator(
    name="micro_mid_return",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_mid_return",
    source="factor_dsl_polars",
)
class MicroMidReturnPolars(SeriesOperator):
    """Polars Mid-price return"""
    metadata = OperatorMetadata(
        name="micro_mid_return",
        category="intraday_microstructure",
        description="Mid-price return",
        param_names=["high", "low"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _align_cols(high, low)
        return high.with_columns([
            (((high[c] + low[c]) / 2.0) / ((high[c] + low[c]) / 2.0).shift(1) - 1.0).alias(c)
            for c in cols
        ])


@register_operator(
    name="micro_bipower_var",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_bipower_var",
    source="factor_dsl_polars",
)
class MicroBipowerVarPolars(SeriesOperator):
    """Polars Bipower variation"""
    metadata = OperatorMetadata(
        name="micro_bipower_var",
        category="intraday_microstructure",
        description="Bipower variation",
        param_names=["close", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        import math

        w = max(1, int(window))
        scale = math.pi / 2.0
        cols = _numeric_cols(close)
        return close.with_columns([
            (
                (
                    (close[c] / close[c].shift(1) - 1.0).abs()
                    * (close[c] / close[c].shift(1) - 1.0).abs().shift(1)
                )
                .rolling_mean(w)
                * scale
            ).alias(c)
            for c in cols
        ])


@register_operator(
    name="micro_jump_indicator",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_jump_indicator",
    source="factor_dsl_polars",
)
class MicroJumpIndicatorPolars(SeriesOperator):
    """Polars Jump indicator"""
    metadata = OperatorMetadata(
        name="micro_jump_indicator",
        category="intraday_microstructure",
        description="Jump indicator",
        param_names=["close", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        import math

        w = max(1, int(window))
        scale = math.pi / 2.0
        cols = _numeric_cols(close)
        out_exprs = []
        for c in cols:
            ret = close[c] / close[c].shift(1) - 1.0
            rv = ret.pow(2).rolling_sum(w)
            bv = (ret.abs() * ret.abs().shift(1)).rolling_mean(w) * scale
            out_exprs.append((rv - bv).clip(lower_bound=0.0).alias(c))
        return close.with_columns(out_exprs)


@register_operator(
    name="micro_trade_imbalance",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_trade_imbalance",
    source="factor_dsl_polars",
)
class MicroTradeImbalancePolars(SeriesOperator):
    """Polars Trade imbalance proxy"""
    metadata = OperatorMetadata(
        name="micro_trade_imbalance",
        category="intraday_microstructure",
        description="Trade imbalance proxy",
        param_names=["close", "volume", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, volume: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        w = max(1, int(window))
        vol_cols = _numeric_cols(volume)
        out_exprs = []
        for c in _numeric_cols(close):
            vcol = c if c in vol_cols else vol_cols[0]
            ret = close[c] / close[c].shift(1) - 1.0
            signed = ret.sign() * volume[vcol]
            out_exprs.append(
                (signed.rolling_sum(w) / volume[vcol].rolling_sum(w)).alias(c)
            )
        return close.with_columns(out_exprs)


@register_operator(
    name="micro_vpin",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_vpin",
    source="factor_dsl_polars",
)
class MicroVpinPolars(SeriesOperator):
    """Polars VPIN proxy"""
    metadata = OperatorMetadata(
        name="micro_vpin",
        category="intraday_microstructure",
        description="VPIN proxy",
        param_names=["close", "volume", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, volume: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        w = max(1, int(window))
        vol_cols = _numeric_cols(volume)
        out_exprs = []
        for c in _numeric_cols(close):
            vcol = c if c in vol_cols else vol_cols[0]
            ret = (close[c] / close[c].shift(1) - 1.0).abs()
            out_exprs.append(
                ((ret * volume[vcol]).rolling_sum(w) / volume[vcol].rolling_sum(w)).alias(c)
            )
        return close.with_columns(out_exprs)


@register_operator(
    name="micro_kyle_lambda",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="micro_kyle_lambda",
    source="factor_dsl_polars",
)
class MicroKyleLambdaPolars(SeriesOperator):
    """Polars Kyle lambda proxy"""
    metadata = OperatorMetadata(
        name="micro_kyle_lambda",
        category="intraday_microstructure",
        description="Kyle lambda proxy",
        param_names=["close", "volume", "window"],
        return_type="series",
        tags=["microstructure", "polars"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, volume: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        w = max(2, int(window))
        vol_cols = _numeric_cols(volume)
        out_exprs = []
        for c in _numeric_cols(close):
            vcol = c if c in vol_cols else vol_cols[0]
            ret = (close[c] / close[c].shift(1) - 1.0).abs()
            cov = ret.rolling_cov(volume[vcol], window_size=w)
            var = volume[vcol].rolling_var(w)
            out_exprs.append((cov / var).alias(c))
        return close.with_columns(out_exprs)
