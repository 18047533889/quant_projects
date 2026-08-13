# -*- coding: utf-8 -*-
"""A-share limit operators - Polars native implementations.

A-share specific limit-up/down touch, streak, volume, failed limit, open limit,
distance, asymmetry, and suspension metrics.
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
# Limit touch indicators
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_up_touch",
    category="ashare_limit",
    business_category="limit_touch",
    canonical="ashare_limit_up_touch",
    source=_SRC,
    backend="polars",
)
class AShareLimitUpTouchNative(SeriesOperator):
    """Binary indicator: 1 if touched limit up, 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_limit_up_touch",
        category="ashare_limit",
        description="涨停触及",
        param_names=["close", "limit_up"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, limit_up: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            cl = close[c]
            lim = limit_up[c] if c in limit_up.columns else pl.lit(None)
            exprs.append(
                pl.when((cl.is_not_null()) & (lim.is_not_null()) & (cl >= lim))
                .then(1)
                .otherwise(0)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, close)


@register_operator(
    name="ashare_limit_down_touch",
    category="ashare_limit",
    business_category="limit_touch",
    canonical="ashare_limit_down_touch",
    source=_SRC,
    backend="polars",
)
class AShareLimitDownTouchNative(SeriesOperator):
    """Binary indicator: 1 if touched limit down, 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_limit_down_touch",
        category="ashare_limit",
        description="跌停触及",
        param_names=["close", "limit_down"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self, close: pl.DataFrame, limit_down: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            cl = close[c]
            lim = limit_down[c] if c in limit_down.columns else pl.lit(None)
            exprs.append(
                pl.when((cl.is_not_null()) & (lim.is_not_null()) & (cl <= lim))
                .then(1)
                .otherwise(0)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, close)


@register_operator(
    name="ashare_limit_touch_count",
    category="ashare_limit",
    business_category="limit_touch",
    canonical="ashare_limit_touch_count",
    source=_SRC,
    backend="polars",
)
class AShareLimitTouchCountNative(SeriesOperator):
    """Rolling count of limit touches (up or down) over window."""

    metadata = OperatorMetadata(
        name="ashare_limit_touch_count",
        category="ashare_limit",
        description="涨跌停触及次数",
        param_names=["limit_touch", "window"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, limit_touch: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(limit_touch)
        exprs = []
        for c in cols:
            touch = limit_touch[c]
            count = touch.rolling_sum(window_size=w)
            exprs.append(count.alias(c))
        result = limit_touch.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Limit streaks
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_up_streak",
    category="ashare_limit",
    business_category="limit_streak",
    canonical="ashare_limit_up_streak",
    source=_SRC,
    backend="polars",
)
class AShareLimitUpStreakNative(SeriesOperator):
    """Consecutive limit-up days (resets on non-limit day)."""

    metadata = OperatorMetadata(
        name="ashare_limit_up_streak",
        category="ashare_limit",
        description="连续涨停天数",
        param_names=["limit_up_touch"],
        return_type="series",
        tags=["ashare", "limit", "streak", "polars", "native"],
    )

    def _calculate_series(
        self, limit_up_touch: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(limit_up_touch)
        exprs = []
        for c in cols:
            touch = limit_up_touch[c]
            # Create block ID that increments on streak breaks
            block = (~touch.cast(pl.Boolean)).cum_sum()
            streak = touch.cum_sum() - touch.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(streak.alias(c))
        result = limit_up_touch.with_columns(exprs)
        return result


@register_operator(
    name="ashare_limit_down_streak",
    category="ashare_limit",
    business_category="limit_streak",
    canonical="ashare_limit_down_streak",
    source=_SRC,
    backend="polars",
)
class AShareLimitDownStreakNative(SeriesOperator):
    """Consecutive limit-down days (resets on non-limit day)."""

    metadata = OperatorMetadata(
        name="ashare_limit_down_streak",
        category="ashare_limit",
        description="连续跌停天数",
        param_names=["limit_down_touch"],
        return_type="series",
        tags=["ashare", "limit", "streak", "polars", "native"],
    )

    def _calculate_series(
        self, limit_down_touch: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(limit_down_touch)
        exprs = []
        for c in cols:
            touch = limit_down_touch[c]
            block = (~touch.cast(pl.Boolean)).cum_sum()
            streak = touch.cum_sum() - touch.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(streak.alias(c))
        result = limit_down_touch.with_columns(exprs)
        return result


@register_operator(
    name="ashare_limit_one_price_limit_streak",
    category="ashare_limit",
    business_category="limit_streak",
    canonical="ashare_limit_one_price_limit_streak",
    source=_SRC,
    backend="polars",
)
class AShareLimitOnePriceLimitStreakNative(SeriesOperator):
    """Consecutive days at one-price limit (open=high=low=close at limit)."""

    metadata = OperatorMetadata(
        name="ashare_limit_one_price_limit_streak",
        category="ashare_limit",
        description="连续一字涨跌停天数",
        param_names=["one_price_indicator"],
        return_type="series",
        tags=["ashare", "limit", "streak", "polars", "native"],
    )

    def _calculate_series(
        self, one_price_indicator: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(one_price_indicator)
        exprs = []
        for c in cols:
            ind = one_price_indicator[c]
            block = (~ind.cast(pl.Boolean)).cum_sum()
            streak = ind.cum_sum() - ind.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(streak.alias(c))
        result = one_price_indicator.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Days since limit
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_days_since_limit_up",
    category="ashare_limit",
    business_category="limit_recency",
    canonical="ashare_days_since_limit_up",
    source=_SRC,
    backend="polars",
)
class AShareDaysSinceLimitUpNative(SeriesOperator):
    """Days since last limit-up touch."""

    metadata = OperatorMetadata(
        name="ashare_days_since_limit_up",
        category="ashare_limit",
        description="距上次涨停天数",
        param_names=["limit_up_touch"],
        return_type="series",
        tags=["ashare", "limit", "recency", "polars", "native"],
    )

    def _calculate_series(
        self, limit_up_touch: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(limit_up_touch)
        exprs = []
        for c in cols:
            touch = limit_up_touch[c]
            # Counter: increment each row, reset to 0 when touch==1
            is_touch = touch.cast(pl.Boolean)
            block = is_touch.cum_sum()
            counter = pl.int_range(0, pl.len()).over(block) - pl.int_range(0, pl.len()).shift(1).fill_null(0).over(block)
            days_since = pl.when(is_touch).then(0).otherwise(counter)
            exprs.append(days_since.alias(c))
        result = limit_up_touch.with_columns(exprs)
        return result


@register_operator(
    name="ashare_days_since_limit_down",
    category="ashare_limit",
    business_category="limit_recency",
    canonical="ashare_days_since_limit_down",
    source=_SRC,
    backend="polars",
)
class AShareDaysSinceLimitDownNative(SeriesOperator):
    """Days since last limit-down touch."""

    metadata = OperatorMetadata(
        name="ashare_days_since_limit_down",
        category="ashare_limit",
        description="距上次跌停天数",
        param_names=["limit_down_touch"],
        return_type="series",
        tags=["ashare", "limit", "recency", "polars", "native"],
    )

    def _calculate_series(
        self, limit_down_touch: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(limit_down_touch)
        exprs = []
        for c in cols:
            touch = limit_down_touch[c]
            is_touch = touch.cast(pl.Boolean)
            block = is_touch.cum_sum()
            counter = pl.int_range(0, pl.len()).over(block) - pl.int_range(0, pl.len()).shift(1).fill_null(0).over(block)
            days_since = pl.when(is_touch).then(0).otherwise(counter)
            exprs.append(days_since.alias(c))
        result = limit_down_touch.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Limit volume ratios
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_up_volume_ratio",
    category="ashare_limit",
    business_category="limit_volume",
    canonical="ashare_limit_up_volume_ratio",
    source=_SRC,
    backend="polars",
)
class AShareLimitUpVolumeRatioNative(SeriesOperator):
    """Volume on limit-up day / avg volume over window."""

    metadata = OperatorMetadata(
        name="ashare_limit_up_volume_ratio",
        category="ashare_limit",
        description="涨停成交量比率",
        param_names=["volume", "limit_up_touch", "window"],
        return_type="series",
        tags=["ashare", "limit", "volume", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, volume: pl.DataFrame, limit_up_touch: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            vol = volume[c]
            touch = limit_up_touch[c] if c in limit_up_touch.columns else pl.lit(0)
            avg_vol = vol.rolling_mean(window_size=w)
            ratio = pl.when((touch > 0) & (avg_vol > 0)).then(vol / avg_vol).otherwise(None)
            exprs.append(ratio.alias(c))
        result = volume.with_columns(exprs)
        return result


@register_operator(
    name="ashare_limit_down_volume_ratio",
    category="ashare_limit",
    business_category="limit_volume",
    canonical="ashare_limit_down_volume_ratio",
    source=_SRC,
    backend="polars",
)
class AShareLimitDownVolumeRatioNative(SeriesOperator):
    """Volume on limit-down day / avg volume over window."""

    metadata = OperatorMetadata(
        name="ashare_limit_down_volume_ratio",
        category="ashare_limit",
        description="跌停成交量比率",
        param_names=["volume", "limit_down_touch", "window"],
        return_type="series",
        tags=["ashare", "limit", "volume", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, volume: pl.DataFrame, limit_down_touch: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(volume)
        exprs = []
        for c in cols:
            vol = volume[c]
            touch = limit_down_touch[c] if c in limit_down_touch.columns else pl.lit(0)
            avg_vol = vol.rolling_mean(window_size=w)
            ratio = pl.when((touch > 0) & (avg_vol > 0)).then(vol / avg_vol).otherwise(None)
            exprs.append(ratio.alias(c))
        result = volume.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Failed limit indicators
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_failed",
    category="ashare_limit",
    business_category="failed_limit",
    canonical="ashare_limit_failed",
    source=_SRC,
    backend="polars",
)
class AShareLimitFailedNative(SeriesOperator):
    """Binary: 1 if reached limit during day but closed off limit, 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_limit_failed",
        category="ashare_limit",
        description="涨跌停失败",
        param_names=["high", "low", "close", "limit_up", "limit_down"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        limit_up: pl.DataFrame,
        limit_down: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(high)
        exprs = []
        for c in cols:
            h = high[c]
            lo = low[c] if c in low.columns else pl.lit(None)
            cl = close[c] if c in close.columns else pl.lit(None)
            lim_up = limit_up[c] if c in limit_up.columns else pl.lit(None)
            lim_down = limit_down[c] if c in limit_down.columns else pl.lit(None)

            # Touched limit up but closed below
            failed_up = (h >= lim_up) & (cl < lim_up)
            # Touched limit down but closed above
            failed_down = (lo <= lim_down) & (cl > lim_down)

            exprs.append(
                pl.when(failed_up | failed_down).then(1).otherwise(0).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, high)


@register_operator(
    name="ashare_failed_limit_count",
    category="ashare_limit",
    business_category="failed_limit",
    canonical="ashare_failed_limit_count",
    source=_SRC,
    backend="polars",
)
class AShareFailedLimitCountNative(SeriesOperator):
    """Rolling count of failed limit events over window."""

    metadata = OperatorMetadata(
        name="ashare_failed_limit_count",
        category="ashare_limit",
        description="涨跌停失败次数",
        param_names=["limit_failed", "window"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, limit_failed: pl.DataFrame, window: int = 20, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(limit_failed)
        exprs = []
        for c in cols:
            failed = limit_failed[c]
            count = failed.rolling_sum(window_size=w)
            exprs.append(count.alias(c))
        result = limit_failed.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Open limit indicators
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_open_up_streak",
    category="ashare_limit",
    business_category="open_limit",
    canonical="ashare_limit_open_up_streak",
    source=_SRC,
    backend="polars",
)
class AShareLimitOpenUpStreakNative(SeriesOperator):
    """Consecutive days opening at limit-up."""

    metadata = OperatorMetadata(
        name="ashare_limit_open_up_streak",
        category="ashare_limit",
        description="连续开盘涨停天数",
        param_names=["open_at_limit_up"],
        return_type="series",
        tags=["ashare", "limit", "open", "polars", "native"],
    )

    def _calculate_series(
        self, open_at_limit_up: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(open_at_limit_up)
        exprs = []
        for c in cols:
            ind = open_at_limit_up[c]
            block = (~ind.cast(pl.Boolean)).cum_sum()
            streak = ind.cum_sum() - ind.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(streak.alias(c))
        result = open_at_limit_up.with_columns(exprs)
        return result


@register_operator(
    name="ashare_limit_open_down_streak",
    category="ashare_limit",
    business_category="open_limit",
    canonical="ashare_limit_open_down_streak",
    source=_SRC,
    backend="polars",
)
class AShareLimitOpenDownStreakNative(SeriesOperator):
    """Consecutive days opening at limit-down."""

    metadata = OperatorMetadata(
        name="ashare_limit_open_down_streak",
        category="ashare_limit",
        description="连续开盘跌停天数",
        param_names=["open_at_limit_down"],
        return_type="series",
        tags=["ashare", "limit", "open", "polars", "native"],
    )

    def _calculate_series(
        self, open_at_limit_down: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(open_at_limit_down)
        exprs = []
        for c in cols:
            ind = open_at_limit_down[c]
            block = (~ind.cast(pl.Boolean)).cum_sum()
            streak = ind.cum_sum() - ind.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(streak.alias(c))
        result = open_at_limit_down.with_columns(exprs)
        return result


@register_operator(
    name="ashare_limit_open_failed",
    category="ashare_limit",
    business_category="open_limit",
    canonical="ashare_limit_open_failed",
    source=_SRC,
    backend="polars",
)
class AShareLimitOpenFailedNative(SeriesOperator):
    """Binary: 1 if opened at limit but closed off limit, 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_limit_open_failed",
        category="ashare_limit",
        description="开盘涨跌停失败",
        param_names=["open", "close", "limit_up", "limit_down"],
        return_type="series",
        tags=["ashare", "limit", "open", "polars", "native"],
    )

    def _calculate_series(
        self,
        open: pl.DataFrame,
        close: pl.DataFrame,
        limit_up: pl.DataFrame,
        limit_down: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            op = open[c]
            cl = close[c] if c in close.columns else pl.lit(None)
            lim_up = limit_up[c] if c in limit_up.columns else pl.lit(None)
            lim_down = limit_down[c] if c in limit_down.columns else pl.lit(None)

            failed_up = (op >= lim_up) & (cl < lim_up)
            failed_down = (op <= lim_down) & (cl > lim_down)

            exprs.append(
                pl.when(failed_up | failed_down).then(1).otherwise(0).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, open)


@register_operator(
    name="ashare_open_at_upper_limit",
    category="ashare_limit",
    business_category="open_limit",
    canonical="ashare_open_at_upper_limit",
    source=_SRC,
    backend="polars",
)
class AShareOpenAtUpperLimitNative(SeriesOperator):
    """Binary: 1 if open == limit_up, 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_open_at_upper_limit",
        category="ashare_limit",
        description="开盘涨停",
        param_names=["open", "limit_up"],
        return_type="series",
        tags=["ashare", "limit", "open", "polars", "native"],
    )

    def _calculate_series(
        self, open: pl.DataFrame, limit_up: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            op = open[c]
            lim = limit_up[c] if c in limit_up.columns else pl.lit(None)
            exprs.append(
                pl.when((op.is_not_null()) & (lim.is_not_null()) & (op >= lim))
                .then(1)
                .otherwise(0)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, open)


@register_operator(
    name="ashare_limit_one_price",
    category="ashare_limit",
    business_category="one_price_limit",
    canonical="ashare_limit_one_price",
    source=_SRC,
    backend="polars",
)
class AShareLimitOnePriceNative(SeriesOperator):
    """Binary: 1 if open=high=low=close (one-price day), 0 otherwise."""

    metadata = OperatorMetadata(
        name="ashare_limit_one_price",
        category="ashare_limit",
        description="一字涨跌停",
        param_names=["open", "high", "low", "close"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self,
        open: pl.DataFrame,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(open)
        exprs = []
        for c in cols:
            op = open[c]
            h = high[c] if c in high.columns else pl.lit(None)
            lo = low[c] if c in low.columns else pl.lit(None)
            cl = close[c] if c in close.columns else pl.lit(None)

            one_price = (op == h) & (h == lo) & (lo == cl)
            exprs.append(
                pl.when(one_price).then(1).otherwise(0).alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, open)


# ---------------------------------------------------------------------------
# Limit distance and asymmetry
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_limit_distance",
    category="ashare_limit",
    business_category="limit_distance",
    canonical="ashare_limit_distance",
    source=_SRC,
    backend="polars",
)
class AShareLimitDistanceNative(SeriesOperator):
    """Min distance to either limit: min(|close - limit_up|, |close - limit_down|) / close."""

    metadata = OperatorMetadata(
        name="ashare_limit_distance",
        category="ashare_limit",
        description="涨跌停距离",
        param_names=["close", "limit_up", "limit_down"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        limit_up: pl.DataFrame,
        limit_down: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            cl = close[c]
            lim_up = limit_up[c] if c in limit_up.columns else pl.lit(None)
            lim_down = limit_down[c] if c in limit_down.columns else pl.lit(None)

            dist_up = (lim_up - cl).abs()
            dist_down = (cl - lim_down).abs()
            min_dist = pl.min_horizontal(dist_up, dist_down)

            exprs.append(
                pl.when((cl.is_null()) | (cl == 0))
                .then(None)
                .otherwise(min_dist / cl)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, close)


@register_operator(
    name="ashare_limit_asymmetry",
    category="ashare_limit",
    business_category="limit_asymmetry",
    canonical="ashare_limit_asymmetry",
    source=_SRC,
    backend="polars",
)
class AShareLimitAsymmetryNative(SeriesOperator):
    """(dist_to_up - dist_to_down) / (dist_to_up + dist_to_down); >0 closer to down."""

    metadata = OperatorMetadata(
        name="ashare_limit_asymmetry",
        category="ashare_limit",
        description="涨跌停不对称度",
        param_names=["close", "limit_up", "limit_down"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
    )

    def _calculate_series(
        self,
        close: pl.DataFrame,
        limit_up: pl.DataFrame,
        limit_down: pl.DataFrame,
        **kwargs,
    ) -> pl.DataFrame:
        cols = _numeric_cols(close)
        exprs = []
        for c in cols:
            cl = close[c]
            lim_up = limit_up[c] if c in limit_up.columns else pl.lit(None)
            lim_down = limit_down[c] if c in limit_down.columns else pl.lit(None)

            dist_up = lim_up - cl
            dist_down = cl - lim_down
            total = dist_up + dist_down

            exprs.append(
                pl.when((total.is_null()) | (total == 0))
                .then(None)
                .otherwise((dist_up - dist_down) / total)
                .alias(c)
            )
        result = pl.DataFrame(exprs)
        return _with_meta(result, close)


@register_operator(
    name="ashare_limit_event_density",
    category="ashare_limit",
    business_category="limit_density",
    canonical="ashare_limit_event_density",
    source=_SRC,
    backend="polars",
)
class AShareLimitEventDensityNative(SeriesOperator):
    """Rolling limit event count / window (normalized frequency)."""

    metadata = OperatorMetadata(
        name="ashare_limit_event_density",
        category="ashare_limit",
        description="涨跌停事件密度",
        param_names=["limit_touch", "window"],
        return_type="series",
        tags=["ashare", "limit", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(
        self, limit_touch: pl.DataFrame, window: int = 60, **kwargs
    ) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(limit_touch)
        exprs = []
        for c in cols:
            touch = limit_touch[c]
            density = pl.when(pl.lit(float(w) != 0).then(touch.rolling_sum(window_size=w) / pl.lit(float(w))).otherwise(None)
            exprs.append(density.alias(c))
        result = limit_touch.with_columns(exprs)
        return result


# ---------------------------------------------------------------------------
# Suspension metrics
# ---------------------------------------------------------------------------

@register_operator(
    name="ashare_suspension_episode_length",
    category="ashare_limit",
    business_category="suspension",
    canonical="ashare_suspension_episode_length",
    source=_SRC,
    backend="polars",
)
class AShareSuspensionEpisodeLengthNative(SeriesOperator):
    """Current consecutive suspension days (resets on trading day)."""

    metadata = OperatorMetadata(
        name="ashare_suspension_episode_length",
        category="ashare_limit",
        description="停牌连续天数",
        param_names=["suspension_indicator"],
        return_type="series",
        tags=["ashare", "suspension", "polars", "native"],
    )

    def _calculate_series(
        self, suspension_indicator: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        cols = _numeric_cols(suspension_indicator)
        exprs = []
        for c in cols:
            sus = suspension_indicator[c]
            # Suspension indicator: 1 if suspended, 0 otherwise
            block = (~sus.cast(pl.Boolean)).cum_sum()
            length = sus.cum_sum() - sus.cum_sum().shift(1).fill_null(0).over(block)
            exprs.append(length.alias(c))
        result = suspension_indicator.with_columns(exprs)
        return result
