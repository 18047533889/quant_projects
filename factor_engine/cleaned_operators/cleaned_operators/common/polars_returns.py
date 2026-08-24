# -*- coding: utf-8 -*-
"""Return calculation operators - Polars native implementations.

All operators use pure Polars expressions.
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


# ---------------------------------------------------------------------------
# Benchmark-relative returns
# ---------------------------------------------------------------------------


@register_operator(
    name="benchmark_excess_return",
    category="returns",
    business_category="returns",
    canonical="benchmark_excess_return",
    source=_SRC,
    backend="polars",
)
class BenchmarkExcessReturnNative(SeriesOperator):
    """Excess return vs benchmark: ret - benchmark_ret."""

    metadata = OperatorMetadata(
        name="benchmark_excess_return",
        category="returns",
        description="超额收益",
        param_names=["ret", "benchmark_ret"],
        return_type="series",
        tags=["returns", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, benchmark_ret: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in benchmark_ret.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) - benchmark_ret[c]).alias(c))
        return ret.with_columns(exprs)


@register_operator(
    name="benchmark_relative_price",
    category="returns",
    business_category="returns",
    canonical="benchmark_relative_price",
    source=_SRC,
    backend="polars",
)
class BenchmarkRelativePriceNative(SeriesOperator):
    """Relative price: price / benchmark_price."""

    metadata = OperatorMetadata(
        name="benchmark_relative_price",
        category="returns",
        description="相对价格",
        param_names=["price", "benchmark_price"],
        return_type="series",
        tags=["returns", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, benchmark_price: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            if c not in benchmark_price.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(
                    pl.when(benchmark_price[c].is_null() | (benchmark_price[c] == 0))
                    .then(None)
                    .otherwise(pl.col(c) / benchmark_price[c])
                    .alias(c)
                )
        return price.with_columns(exprs)


# ---------------------------------------------------------------------------
# Return-volume relationships
# ---------------------------------------------------------------------------


@register_operator(
    name="return_per_turnover",
    category="returns",
    business_category="returns",
    canonical="return_per_turnover",
    source=_SRC,
    backend="polars",
)
class ReturnPerTurnoverNative(SeriesOperator):
    """Return per unit turnover: ret / turnover."""

    metadata = OperatorMetadata(
        name="return_per_turnover",
        category="returns",
        description="单位换手收益",
        param_names=["ret", "turnover"],
        return_type="series",
        tags=["returns", "polars", "native"],
    )

    def _calculate_series(self, ret: pl.DataFrame, turnover: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in turnover.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(
                    pl.when(turnover[c].is_null() | (turnover[c] == 0))
                    .then(None)
                    .otherwise(pl.col(c) / turnover[c])
                    .alias(c)
                )
        return ret.with_columns(exprs)


@register_operator(
    name="return_volume_beta",
    category="returns",
    business_category="returns",
    canonical="return_volume_beta",
    source=_SRC,
    backend="polars",
)
class ReturnVolumeBetaNative(SeriesOperator):
    """Rolling beta of returns vs volume changes."""

    metadata = OperatorMetadata(
        name="return_volume_beta",
        category="returns",
        description="收益-成交量Beta",
        param_names=["ret", "volume", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=20, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame, d: int = 60, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=20)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            vol_change = pl.when(volume[c].shift(1) > 1e-10).then((volume[c] - volume[c].shift(1)) / volume[c].shift(1)).otherwise(None)
            mean_ret = pl.col(c).rolling_mean(window_size=w)
            mean_vol = vol_change.rolling_mean(window_size=w)
            cov = ((pl.col(c) - mean_ret) * (vol_change - mean_vol)).rolling_mean(window_size=w)
            var_vol = ((vol_change - mean_vol) ** 2).rolling_mean(window_size=w)
            exprs.append(
                pl.when(var_vol.is_null() | (var_vol == 0))
                .then(None)
                .otherwise(cov / var_vol)
                .alias(c)
            )
        return ret.with_columns(exprs)


@register_operator(
    name="return_volume_corr",
    category="returns",
    business_category="returns",
    canonical="return_volume_corr",
    source=_SRC,
    backend="polars",
)
class ReturnVolumeCorrNative(SeriesOperator):
    """Rolling correlation between returns and volume."""

    metadata = OperatorMetadata(
        name="return_volume_corr",
        category="returns",
        description="收益-成交量相关性",
        param_names=["ret", "volume", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=20, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame, d: int = 60, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=20)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            mean_ret = pl.col(c).rolling_mean(window_size=w)
            mean_vol = volume[c].rolling_mean(window_size=w)
            cov = ((pl.col(c) - mean_ret) * (volume[c] - mean_vol)).rolling_mean(window_size=w)
            std_ret = pl.col(c).rolling_std(window_size=w)
            std_vol = volume[c].rolling_std(window_size=w)
            exprs.append(
                pl.when((std_ret.is_null()) | (std_vol.is_null()) | (std_ret == 0) | (std_vol == 0))
                .then(None)
                .otherwise(cov / (std_ret * std_vol))
                .alias(c)
            )
        return ret.with_columns(exprs)


@register_operator(
    name="return_turnover_beta",
    category="returns",
    business_category="returns",
    canonical="return_turnover_beta",
    source=_SRC,
    backend="polars",
)
class ReturnTurnoverBetaNative(SeriesOperator):
    """Rolling beta of returns vs turnover changes."""

    metadata = OperatorMetadata(
        name="return_turnover_beta",
        category="returns",
        description="收益-换手率Beta",
        param_names=["ret", "turnover", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=20, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, turnover: pl.DataFrame, d: int = 60, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=20)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in turnover.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            to_change = pl.when(turnover[c].shift(1) != 0).then((turnover[c] - turnover[c].shift(1)) / turnover[c].shift(1)).otherwise(None)
            mean_ret = pl.col(c).rolling_mean(window_size=w)
            mean_to = to_change.rolling_mean(window_size=w)
            cov = ((pl.col(c) - mean_ret) * (to_change - mean_to)).rolling_mean(window_size=w)
            var_to = ((to_change - mean_to) ** 2).rolling_mean(window_size=w)
            exprs.append(
                pl.when(var_to.is_null() | (var_to == 0))
                .then(None)
                .otherwise(cov / var_to)
                .alias(c)
            )
        return ret.with_columns(exprs)


@register_operator(
    name="abs_return_volume_corr",
    category="returns",
    business_category="returns",
    canonical="abs_return_volume_corr",
    source=_SRC,
    backend="polars",
)
class AbsReturnVolumeCorrNative(SeriesOperator):
    """Rolling correlation between |returns| and volume."""

    metadata = OperatorMetadata(
        name="abs_return_volume_corr",
        category="returns",
        description="绝对收益-成交量相关性",
        param_names=["ret", "volume", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=20, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame, d: int = 60, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=20)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            abs_ret = pl.col(c).abs()
            mean_abs_ret = abs_ret.rolling_mean(window_size=w)
            mean_vol = volume[c].rolling_mean(window_size=w)
            cov = ((abs_ret - mean_abs_ret) * (volume[c] - mean_vol)).rolling_mean(window_size=w)
            std_abs_ret = abs_ret.rolling_std(window_size=w)
            std_vol = volume[c].rolling_std(window_size=w)
            exprs.append(
                pl.when((std_abs_ret.is_null()) | (std_vol.is_null()) | (std_abs_ret == 0) | (std_vol == 0))
                .then(None)
                .otherwise(cov / (std_abs_ret * std_vol))
                .alias(c)
            )
        return ret.with_columns(exprs)


@register_operator(
    name="price_volume_divergence",
    category="returns",
    business_category="returns",
    canonical="price_volume_divergence",
    source=_SRC,
    backend="polars",
)
class PriceVolumeDivergenceNative(SeriesOperator):
    """Divergence between price and volume trends."""

    metadata = OperatorMetadata(
        name="price_volume_divergence",
        category="returns",
        description="价量背离",
        param_names=["price", "volume", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=5)
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            # Price momentum
            price_ma = pl.col(c).rolling_mean(window_size=w)
            price_trend = pl.when(price_ma != 0).then((pl.col(c) - price_ma) / price_ma).otherwise(None)
            # Volume momentum
            vol_ma = volume[c].rolling_mean(window_size=w)
            vol_trend = pl.when(vol_ma > 1e-10).then((volume[c] - vol_ma) / vol_ma).otherwise(None)
            # Divergence = price up but volume down, or vice versa
            exprs.append((price_trend - vol_trend).alias(c))
        return price.with_columns(exprs)


@register_operator(
    name="price_turnover_divergence",
    category="returns",
    business_category="returns",
    canonical="price_turnover_divergence",
    source=_SRC,
    backend="polars",
)
class PriceTurnoverDivergenceNative(SeriesOperator):
    """Divergence between price and turnover trends."""

    metadata = OperatorMetadata(
        name="price_turnover_divergence",
        category="returns",
        description="价格-换手率背离",
        param_names=["price", "turnover", "d"],
        return_type="series",
        tags=["returns", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, price: pl.DataFrame, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=5)
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            if c not in turnover.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            # Price momentum
            price_ma = pl.col(c).rolling_mean(window_size=w)
            price_trend = pl.when(price_ma != 0).then((pl.col(c) - price_ma) / price_ma).otherwise(None)
            # Turnover momentum
            to_ma = turnover[c].rolling_mean(window_size=w)
            to_trend = pl.when(to_ma != 0).then((turnover[c] - to_ma) / to_ma).otherwise(None)
            # Divergence
            exprs.append((price_trend - to_trend).alias(c))
        return price.with_columns(exprs)
