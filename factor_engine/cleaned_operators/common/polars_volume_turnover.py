# -*- coding: utf-8 -*-
"""Volume and turnover operators - Polars native implementations.

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
# Basic volume/turnover metrics
# ---------------------------------------------------------------------------


@register_operator(
    name="adv",
    category="volume",
    business_category="volume_turnover",
    canonical="adv",
    source=_SRC,
    backend="polars",
)
class AdvNative(SeriesOperator):
    """Average daily volume."""

    metadata = OperatorMetadata(
        name="adv",
        category="volume",
        description="平均日成交量",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        return volume.with_columns([pl.col(c).rolling_mean(window_size=w).alias(c) for c in cols])


@register_operator(
    name="average_turnover",
    category="volume",
    business_category="volume_turnover",
    canonical="average_turnover",
    source=_SRC,
    backend="polars",
)
class AverageTurnoverNative(SeriesOperator):
    """Average turnover rate."""

    metadata = OperatorMetadata(
        name="average_turnover",
        category="volume",
        description="平均换手率",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(turnover)
        return turnover.with_columns([pl.col(c).rolling_mean(window_size=w).alias(c) for c in cols])


@register_operator(
    name="dollar_volume",
    category="volume",
    business_category="volume_turnover",
    canonical="dollar_volume",
    source=_SRC,
    backend="polars",
)
class DollarVolumeNative(SeriesOperator):
    """Dollar volume: price * volume."""

    metadata = OperatorMetadata(
        name="dollar_volume",
        category="volume",
        description="成交额",
        param_names=["price", "volume"],
        return_type="series",
        tags=["volume", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((pl.col(c) * volume[c]).alias(c))
        return price.with_columns(exprs)


@register_operator(
    name="signed_dollar_volume",
    category="volume",
    business_category="volume_turnover",
    canonical="signed_dollar_volume",
    source=_SRC,
    backend="polars",
)
class SignedDollarVolumeNative(SeriesOperator):
    """Signed dollar volume: sign(ret) * price * volume."""

    metadata = OperatorMetadata(
        name="signed_dollar_volume",
        category="volume",
        description="带符号成交额",
        param_names=["price", "volume", "ret"],
        return_type="series",
        tags=["volume", "polars", "native"],
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame, ret: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(price)
        exprs = []
        for c in cols:
            if c not in volume.columns or c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((ret[c].sign() * pl.col(c) * volume[c]).alias(c))
        return price.with_columns(exprs)


@register_operator(
    name="signed_volume",
    category="volume",
    business_category="volume_turnover",
    canonical="signed_volume",
    source=_SRC,
    backend="polars",
)
class SignedVolumeNative(SeriesOperator):
    """Signed volume: sign(ret) * volume."""

    metadata = OperatorMetadata(
        name="signed_volume",
        category="volume",
        description="带符号成交量",
        param_names=["volume", "ret"],
        return_type="series",
        tags=["volume", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, ret: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append((ret[c].sign() * pl.col(c)).alias(c))
        return volume.with_columns(exprs)


# ---------------------------------------------------------------------------
# Volume dynamics
# ---------------------------------------------------------------------------


@register_operator(
    name="volume_momentum",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_momentum",
    source=_SRC,
    backend="polars",
)
class VolumeMomentumNative(SeriesOperator):
    """Volume change rate: (vol_t - vol_{t-d}) / vol_{t-d}."""

    metadata = OperatorMetadata(
        name="volume_momentum",
        category="volume",
        description="成交量动量",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            prev = pl.col(c).shift(lag)
            exprs.append(
                pl.when(prev.is_null() | (prev == 0))
                .then(None)
                .otherwise((pl.col(c) - prev) / prev)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="volume_shock",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_shock",
    source=_SRC,
    backend="polars",
)
class VolumeShockNative(SeriesOperator):
    """Volume shock: (vol - mean) / std."""

    metadata = OperatorMetadata(
        name="volume_shock",
        category="volume",
        description="成交量冲击",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="volume_volatility",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_volatility",
    source=_SRC,
    backend="polars",
)
class VolumeVolatilityNative(SeriesOperator):
    """Volume volatility: std(volume)."""

    metadata = OperatorMetadata(
        name="volume_volatility",
        category="volume",
        description="成交量波动率",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(volume)
        return volume.with_columns([pl.col(c).rolling_std(window_size=w).alias(c) for c in cols])


@register_operator(
    name="volume_zscore",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_zscore",
    source=_SRC,
    backend="polars",
)
class VolumeZscoreNative(SeriesOperator):
    """Volume z-score."""

    metadata = OperatorMetadata(
        name="volume_zscore",
        category="volume",
        description="成交量标准化",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="volume_acceleration",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_acceleration",
    source=_SRC,
    backend="polars",
)
class VolumeAccelerationNative(SeriesOperator):
    """Volume acceleration: delta(volume_momentum)."""

    metadata = OperatorMetadata(
        name="volume_acceleration",
        category="volume",
        description="成交量加速度",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        return volume.with_columns([(pl.col(c) - pl.col(c).shift(lag)).alias(c) for c in cols])


@register_operator(
    name="volume_autocorr",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_autocorr",
    source=_SRC,
    backend="polars",
)
class VolumeAutocorrNative(SeriesOperator):
    """Volume autocorrelation."""

    metadata = OperatorMetadata(
        name="volume_autocorr",
        category="volume",
        description="成交量自相关",
        param_names=["volume", "d", "lag"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, lag: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        l = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            x = pl.col(c)
            y = pl.col(c).shift(l)
            mean_x = x.rolling_mean(window_size=w)
            mean_y = y.rolling_mean(window_size=w)
            cov = ((x - mean_x) * (y - mean_y)).rolling_mean(window_size=w)
            std_x = x.rolling_std(window_size=w)
            std_y = y.rolling_std(window_size=w)
            exprs.append(
                pl.when((std_x.is_null()) | (std_y.is_null()) | (std_x == 0) | (std_y == 0))
                .then(None)
                .otherwise(cov / (std_x * std_y))
                .alias(c)
            )
        return volume.with_columns(exprs)


# ---------------------------------------------------------------------------
# Turnover dynamics
# ---------------------------------------------------------------------------


@register_operator(
    name="turnover_momentum",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_momentum",
    source=_SRC,
    backend="polars",
)
class TurnoverMomentumNative(SeriesOperator):
    """Turnover momentum."""

    metadata = OperatorMetadata(
        name="turnover_momentum",
        category="volume",
        description="换手率动量",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(turnover)
        exprs = []
        for c in cols:
            prev = pl.col(c).shift(lag)
            exprs.append(
                pl.when(prev.is_null() | (prev == 0))
                .then(None)
                .otherwise((pl.col(c) - prev) / prev)
                .alias(c)
            )
        return turnover.with_columns(exprs)


@register_operator(
    name="turnover_shock",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_shock",
    source=_SRC,
    backend="polars",
)
class TurnoverShockNative(SeriesOperator):
    """Turnover shock."""

    metadata = OperatorMetadata(
        name="turnover_shock",
        category="volume",
        description="换手率冲击",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(turnover)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return turnover.with_columns(exprs)


@register_operator(
    name="turnover_volatility",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_volatility",
    source=_SRC,
    backend="polars",
)
class TurnoverVolatilityNative(SeriesOperator):
    """Turnover volatility."""

    metadata = OperatorMetadata(
        name="turnover_volatility",
        category="volume",
        description="换手率波动率",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(turnover)
        return turnover.with_columns([pl.col(c).rolling_std(window_size=w).alias(c) for c in cols])


@register_operator(
    name="turnover_zscore",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_zscore",
    source=_SRC,
    backend="polars",
)
class TurnoverZscoreNative(SeriesOperator):
    """Turnover z-score."""

    metadata = OperatorMetadata(
        name="turnover_zscore",
        category="volume",
        description="换手率标准化",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(turnover)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return turnover.with_columns(exprs)


@register_operator(
    name="turnover_acceleration",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_acceleration",
    source=_SRC,
    backend="polars",
)
class TurnoverAccelerationNative(SeriesOperator):
    """Turnover acceleration."""

    metadata = OperatorMetadata(
        name="turnover_acceleration",
        category="volume",
        description="换手率加速度",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(turnover)
        return turnover.with_columns([(pl.col(c) - pl.col(c).shift(lag)).alias(c) for c in cols])


@register_operator(
    name="turnover_autocorr",
    category="volume",
    business_category="volume_turnover",
    canonical="turnover_autocorr",
    source=_SRC,
    backend="polars",
)
class TurnoverAutocorrNative(SeriesOperator):
    """Turnover autocorrelation."""

    metadata = OperatorMetadata(
        name="turnover_autocorr",
        category="volume",
        description="换手率自相关",
        param_names=["turnover", "d", "lag"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, lag: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        l = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(turnover)
        exprs = []
        for c in cols:
            x = pl.col(c)
            y = pl.col(c).shift(l)
            mean_x = x.rolling_mean(window_size=w)
            mean_y = y.rolling_mean(window_size=w)
            cov = ((x - mean_x) * (y - mean_y)).rolling_mean(window_size=w)
            std_x = x.rolling_std(window_size=w)
            std_y = y.rolling_std(window_size=w)
            exprs.append(
                pl.when((std_x.is_null()) | (std_y.is_null()) | (std_x == 0) | (std_y == 0))
                .then(None)
                .otherwise(cov / (std_x * std_y))
                .alias(c)
            )
        return turnover.with_columns(exprs)


# ---------------------------------------------------------------------------
# Relative and abnormal volume/turnover
# ---------------------------------------------------------------------------


@register_operator(
    name="relative_volume",
    category="volume",
    business_category="volume_turnover",
    canonical="relative_volume",
    source=_SRC,
    backend="polars",
)
class RelativeVolumeNative(SeriesOperator):
    """Volume / average_volume."""

    metadata = OperatorMetadata(
        name="relative_volume",
        category="volume",
        description="相对成交量",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            avg = pl.col(c).rolling_mean(window_size=w)
            exprs.append(
                pl.when(avg.is_null() | (avg == 0))
                .then(None)
                .otherwise(pl.col(c) / avg)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="abnormal_volume",
    category="volume",
    business_category="volume_turnover",
    canonical="abnormal_volume",
    source=_SRC,
    backend="polars",
)
class AbnormalVolumeNative(SeriesOperator):
    """Abnormal volume: (vol - avg) / std."""

    metadata = OperatorMetadata(
        name="abnormal_volume",
        category="volume",
        description="异常成交量",
        param_names=["volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="abnormal_turnover",
    category="volume",
    business_category="volume_turnover",
    canonical="abnormal_turnover",
    source=_SRC,
    backend="polars",
)
class AbnormalTurnoverNative(SeriesOperator):
    """Abnormal turnover."""

    metadata = OperatorMetadata(
        name="abnormal_turnover",
        category="volume",
        description="异常换手率",
        param_names=["turnover", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, turnover: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        cols = _numeric_cols(turnover)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise((pl.col(c) - mean) / std)
                .alias(c)
            )
        return turnover.with_columns(exprs)


@register_operator(
    name="true_turnover_rate",
    category="volume",
    business_category="volume_turnover",
    canonical="true_turnover_rate",
    source=_SRC,
    backend="polars",
)
class TrueTurnoverRateNative(SeriesOperator):
    """True turnover rate: volume / float_shares."""

    metadata = OperatorMetadata(
        name="true_turnover_rate",
        category="volume",
        description="真实换手率",
        param_names=["volume", "float_shares"],
        return_type="series",
        tags=["volume", "polars", "native"],
    )

    def _calculate_series(self, volume: pl.DataFrame, float_shares: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in float_shares.columns:
                exprs.append(pl.lit(None).alias(c))
            else:
                exprs.append(
                    pl.when(float_shares[c].is_null() | (float_shares[c] == 0))
                    .then(None)
                    .otherwise(pl.col(c) / float_shares[c])
                    .alias(c)
                )
        return volume.with_columns(exprs)


# ---------------------------------------------------------------------------
# Up/down volume ratios
# ---------------------------------------------------------------------------


@register_operator(
    name="up_volume_ratio",
    category="volume",
    business_category="volume_turnover",
    canonical="up_volume_ratio",
    source=_SRC,
    backend="polars",
)
class UpVolumeRatioNative(SeriesOperator):
    """Ratio of up-day volume to total volume."""

    metadata = OperatorMetadata(
        name="up_volume_ratio",
        category="volume",
        description="上涨日成交量占比",
        param_names=["volume", "ret", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, ret: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            up_vol = pl.when(ret[c] > 0).then(pl.col(c)).otherwise(0.0)
            total_vol = pl.col(c).rolling_sum(window_size=w)
            up_vol_sum = up_vol.rolling_sum(window_size=w)
            exprs.append(
                pl.when(total_vol.is_null() | (total_vol == 0))
                .then(None)
                .otherwise(up_vol_sum / total_vol)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="down_volume_ratio",
    category="volume",
    business_category="volume_turnover",
    canonical="down_volume_ratio",
    source=_SRC,
    backend="polars",
)
class DownVolumeRatioNative(SeriesOperator):
    """Ratio of down-day volume to total volume."""

    metadata = OperatorMetadata(
        name="down_volume_ratio",
        category="volume",
        description="下跌日成交量占比",
        param_names=["volume", "ret", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, ret: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            down_vol = pl.when(ret[c] < 0).then(pl.col(c)).otherwise(0.0)
            total_vol = pl.col(c).rolling_sum(window_size=w)
            down_vol_sum = down_vol.rolling_sum(window_size=w)
            exprs.append(
                pl.when(total_vol.is_null() | (total_vol == 0))
                .then(None)
                .otherwise(down_vol_sum / total_vol)
                .alias(c)
            )
        return volume.with_columns(exprs)


@register_operator(
    name="up_down_volume_ratio",
    category="volume",
    business_category="volume_turnover",
    canonical="up_down_volume_ratio",
    source=_SRC,
    backend="polars",
)
class UpDownVolumeRatioNative(SeriesOperator):
    """Ratio of up-day volume to down-day volume."""

    metadata = OperatorMetadata(
        name="up_down_volume_ratio",
        category="volume",
        description="上涨下跌成交量比",
        param_names=["volume", "ret", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, ret: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            up_vol = pl.when(ret[c] > 0).then(pl.col(c)).otherwise(0.0).rolling_sum(window_size=w)
            down_vol = pl.when(ret[c] < 0).then(pl.col(c)).otherwise(0.0).rolling_sum(window_size=w)
            exprs.append(
                pl.when(down_vol.is_null() | (down_vol == 0))
                .then(None)
                .otherwise(up_vol / down_vol)
                .alias(c)
            )
        return volume.with_columns(exprs)


# ---------------------------------------------------------------------------
# Volume-price interactions
# ---------------------------------------------------------------------------


@register_operator(
    name="signed_volume_imbalance",
    category="volume",
    business_category="volume_turnover",
    canonical="signed_volume_imbalance",
    source=_SRC,
    backend="polars",
)
class SignedVolumeImbalanceNative(SeriesOperator):
    """Sum of signed volume."""

    metadata = OperatorMetadata(
        name="signed_volume_imbalance",
        category="volume",
        description="带符号成交量不平衡",
        param_names=["volume", "ret", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, volume: pl.DataFrame, ret: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            if c not in ret.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            signed = ret[c].sign() * pl.col(c)
            exprs.append(signed.rolling_sum(window_size=w).alias(c))
        return volume.with_columns(exprs)


@register_operator(
    name="volume_weighted_return",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_weighted_return",
    source=_SRC,
    backend="polars",
)
class VolumeWeightedReturnNative(SeriesOperator):
    """Volume-weighted average return."""

    metadata = OperatorMetadata(
        name="volume_weighted_return",
        category="volume",
        description="成交量加权收益",
        param_names=["ret", "volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            weighted_sum = (pl.col(c) * volume[c]).rolling_sum(window_size=w)
            vol_sum = volume[c].rolling_sum(window_size=w)
            exprs.append(
                pl.when(vol_sum.is_null() | (vol_sum == 0))
                .then(None)
                .otherwise(weighted_sum / vol_sum)
                .alias(c)
            )
        return ret.with_columns(exprs)


@register_operator(
    name="volume_weighted_momentum",
    category="volume",
    business_category="volume_turnover",
    canonical="volume_weighted_momentum",
    source=_SRC,
    backend="polars",
)
class VolumeWeightedMomentumNative(SeriesOperator):
    """Volume-weighted momentum."""

    metadata = OperatorMetadata(
        name="volume_weighted_momentum",
        category="volume",
        description="成交量加权动量",
        param_names=["ret", "volume", "d"],
        return_type="series",
        tags=["volume", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, volume: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(ret)
        exprs = []
        for c in cols:
            if c not in volume.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            weighted_sum = (pl.col(c) * volume[c]).rolling_sum(window_size=w)
            vol_sum = volume[c].rolling_sum(window_size=w)
            exprs.append(
                pl.when(vol_sum.is_null() | (vol_sum == 0))
                .then(None)
                .otherwise(weighted_sum / vol_sum)
                .alias(c)
            )
        return ret.with_columns(exprs)

