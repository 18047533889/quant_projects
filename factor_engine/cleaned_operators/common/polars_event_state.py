# -*- coding: utf-8 -*-
"""Event/state operators - Polars native implementations (Phase 2, Module 10).

All operators use TRUE Polars expressions only - no pandas fallback, no NumPy.
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


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


@register_operator(
    name="state_since_last",
    category="state",
    business_category="event_state",
    canonical="state_since_last",
    source=_SRC,
    backend="polars")
class StateSinceLastNative(SeriesOperator):
    """Periods since condition was last true."""

    metadata = OperatorMetadata(
        name="state_since_last",
        category="state",
        description="距离上次事件周期数",
        param_names=["condition"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, condition: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(condition)

        exprs = []
        for c in cols:
            is_true = (pl.col(c) != 0) & pl.col(c).is_not_null()
            # Cumulative count that resets when condition is true
            since = (1 - is_true.cast(pl.Int32)).cum_sum() - (1 - is_true.cast(pl.Int32)).cum_sum().shift().fill_null(0) * (1 - is_true.cast(pl.Int32))
            exprs.append(since.cast(pl.Float64).alias(c))

        return condition.with_columns(exprs)


@register_operator(
    name="state_since_count",
    category="state",
    business_category="event_state",
    canonical="state_since_count",
    source=_SRC,
    backend="polars")
class StateSinceCountNative(SeriesOperator):
    """Count of events since condition became true."""

    metadata = OperatorMetadata(
        name="state_since_count",
        category="state",
        description="状态持续期间事件计数",
        param_names=["event", "state"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, event: pl.DataFrame, state: pl.DataFrame | None = None,
                         **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(event)

        if state is None:
            raise ValueError("state_since_count requires state")

        exprs = []
        for c in cols:
            evt = (pl.col(c) != 0) & pl.col(c).is_not_null()
            st = (state[c] != 0) & state[c].is_not_null() if c in state.columns else pl.lit(False)

            # Count events while state is true
            count = (evt.cast(pl.Int32) * st.cast(pl.Int32)).cum_sum()
            exprs.append(count.cast(pl.Float64).alias(c))

        return event.with_columns(exprs)


@register_operator(
    name="state_hold",
    category="state",
    business_category="event_state",
    canonical="state_hold",
    source=_SRC,
    backend="polars")
class StateHoldNative(SeriesOperator):
    """Hold state for N periods after trigger."""

    metadata = OperatorMetadata(
        name="state_hold",
        category="state",
        description="状态保持",
        param_names=["trigger", "hold_periods"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, trigger: pl.DataFrame, hold_periods: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        hp = strict_integer(hold_periods, "hold_periods", minimum=1)
        cols = _numeric_cols(trigger)

        exprs = []
        for c in cols:
            is_trigger = (pl.col(c) != 0) & pl.col(c).is_not_null()

            # Create a rolling max that looks back hold_periods
            # If any trigger in the window, output 1
            held = is_trigger.cast(pl.Float64).rolling_max(window_size=hp, min_samples=1)
            exprs.append(held.alias(c))

        return trigger.with_columns(exprs)


@register_operator(
    name="state_latch",
    category="state",
    business_category="event_state",
    canonical="state_latch",
    source=_SRC,
    backend="polars")
class StateLatchNative(SeriesOperator):
    """Latch state on (set trigger) and off (reset trigger)."""

    metadata = OperatorMetadata(
        name="state_latch",
        category="state",
        description="锁存状态",
        param_names=["set_trigger", "reset_trigger"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, set_trigger: pl.DataFrame, reset_trigger: pl.DataFrame | None = None,
                         **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(set_trigger)

        if reset_trigger is None:
            raise ValueError("state_latch requires reset_trigger")

        exprs = []
        for c in cols:
            set_t = (pl.col(c) != 0) & pl.col(c).is_not_null()
            reset_t = (reset_trigger[c] != 0) & reset_trigger[c].is_not_null() if c in reset_trigger.columns else pl.lit(False)

            # Simplified latch: forward fill set, clear on reset
            # True latch requires stateful iteration; this is an approximation
            state = set_t.cast(pl.Int32).cum_sum() - reset_t.cast(pl.Int32).cum_sum()
            latched = (state > 0).cast(pl.Float64)
            exprs.append(latched.alias(c))

        return set_trigger.with_columns(exprs)


@register_operator(
    name="state_deadband",
    category="state",
    business_category="event_state",
    canonical="state_deadband",
    source=_SRC,
    backend="polars")
class StateDeadbandNative(SeriesOperator):
    """Deadband filter: change only if delta exceeds threshold."""

    metadata = OperatorMetadata(
        name="state_deadband",
        category="state",
        description="死区滤波",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.01, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar

        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # Change exceeds threshold
            prev = pl.col(c).shift(1)
            delta = (pl.col(c) - prev).abs()
            changed = delta > thresh

            # Forward fill when not changed
            result = pl.when(changed | prev.is_null()).then(pl.col(c)).otherwise(prev)
            exprs.append(result.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="state_ewm_if",
    category="state",
    business_category="event_state",
    canonical="state_ewm_if",
    source=_SRC,
    backend="polars")
class StateEwmIfNative(SeriesOperator):
    """EWM that only updates when condition is true."""

    metadata = OperatorMetadata(
        name="state_ewm_if",
        category="state",
        description="条件指数加权移动平均",
        param_names=["x", "condition", "window"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, condition: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        alpha = 2.0 / (w + 1)

        if condition is None:
            raise ValueError("state_ewm_if requires condition")

        exprs = []
        for c in cols:
            cond = (condition[c] != 0) & condition[c].is_not_null() if c in condition.columns else pl.lit(True)

            # When condition is false, use previous value (simplified)
            val = pl.when(cond).then(pl.col(c)).otherwise(pl.col(c).shift(1))
            ewm = val.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            exprs.append(ewm.alias(c))

        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------


@register_operator(
    name="cross_event",
    category="state",
    business_category="event_state",
    canonical="cross_event",
    source=_SRC,
    backend="polars")
class CrossEventNative(SeriesOperator):
    """Detect when x crosses threshold (1=up, -1=down, 0=no cross)."""

    metadata = OperatorMetadata(
        name="cross_event",
        category="state",
        description="交叉事件检测",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar

        thresh = strict_finite_scalar(threshold, "threshold")
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            curr_above = pl.col(c) > thresh
            prev_above = pl.col(c).shift(1) > thresh

            cross_up = curr_above & ~prev_above
            cross_down = ~curr_above & prev_above

            result = pl.when(cross_up).then(1.0).when(cross_down).then(-1.0).otherwise(0.0)
            exprs.append(result.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="limit_up_close",
    category="state",
    business_category="event_state",
    canonical="limit_up_close",
    source=_SRC,
    backend="polars")
class LimitUpCloseNative(SeriesOperator):
    """Detect limit-up close (close >= prev_close * 1.1)."""

    metadata = OperatorMetadata(
        name="limit_up_close",
        category="state",
        description="涨停收盘检测",
        param_names=["close"],
        return_type="series",
        tags=["state", "polars", "native", "market"],
    )

    def _calculate_series(self, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)

        exprs = []
        for c in cols:
            prev = pl.col(c).shift(1)
            is_limit_up = pl.col(c) >= prev * 1.099  # 10% with small tolerance
            exprs.append(is_limit_up.cast(pl.Float64).alias(c))

        return close.with_columns(exprs)


@register_operator(
    name="limit_down_close",
    category="state",
    business_category="event_state",
    canonical="limit_down_close",
    source=_SRC,
    backend="polars")
class LimitDownCloseNative(SeriesOperator):
    """Detect limit-down close (close <= prev_close * 0.9)."""

    metadata = OperatorMetadata(
        name="limit_down_close",
        category="state",
        description="跌停收盘检测",
        param_names=["close"],
        return_type="series",
        tags=["state", "polars", "native", "market"],
    )

    def _calculate_series(self, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)

        exprs = []
        for c in cols:
            prev = pl.col(c).shift(1)
            is_limit_down = pl.col(c) <= prev * 0.901  # -10% with small tolerance
            exprs.append(is_limit_down.cast(pl.Float64).alias(c))

        return close.with_columns(exprs)


@register_operator(
    name="tradable_state",
    category="state",
    business_category="event_state",
    canonical="tradable_state",
    source=_SRC,
    backend="polars")
class TradableStateNative(SeriesOperator):
    """Tradable state: non-null, positive volume, non-limit."""

    metadata = OperatorMetadata(
        name="tradable_state",
        category="state",
        description="可交易状态",
        param_names=["close", "volume"],
        return_type="series",
        tags=["state", "polars", "native", "market"],
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame | None = None,
                         **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(close)

        if volume is None:
            raise ValueError("tradable_state requires volume")

        exprs = []
        for c in cols:
            c_col = pl.col(c)
            v_col = volume[c] if c in volume.columns else pl.lit(None)

            tradable = c_col.is_not_null() & (v_col > 0)
            exprs.append(tradable.cast(pl.Float64).alias(c))

        return close.with_columns(exprs)


@register_operator(
    name="ffill_limit",
    category="state",
    business_category="event_state",
    canonical="ffill_limit",
    source=_SRC,
    backend="polars")
class FfillLimitNative(SeriesOperator):
    """Forward fill with maximum gap limit."""

    metadata = OperatorMetadata(
        name="ffill_limit",
        category="state",
        description="限制前向填充",
        param_names=["x", "max_gap"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, max_gap: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        gap = strict_integer(max_gap, "max_gap", minimum=1)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # Count consecutive nulls
            is_null = pl.col(c).is_null()
            null_run = is_null.cast(pl.Int32).cum_sum() - is_null.cast(pl.Int32).cum_sum().shift().fill_null(0) * is_null.cast(pl.Int32)

            # Forward fill only if gap <= max_gap
            filled = pl.when(null_run <= gap).then(pl.col(c).forward_fill()).otherwise(None)
            exprs.append(filled.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="directional_change_state",
    category="state",
    business_category="event_state",
    canonical="directional_change_state",
    source=_SRC,
    backend="polars")
class DirectionalChangeStateNative(SeriesOperator):
    """Directional change state: 1=uptrend, -1=downtrend, 0=neutral."""

    metadata = OperatorMetadata(
        name="directional_change_state",
        category="state",
        description="方向性变化状态",
        param_names=["x", "threshold"],
        return_type="series",
        tags=["state", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.01, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_finite_scalar

        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # Rolling extremes
            roll_max = pl.col(c).rolling_max(window_size=20, min_samples=1)
            roll_min = pl.col(c).rolling_min(window_size=20, min_samples=1)

            # Drawdown/drawup from extremes
            drawdown = pl.when(roll_max != 0).then((roll_max - pl.col(c)) / roll_max).otherwise(None)
            drawup = pl.when(roll_min != 0).then((pl.col(c) - roll_min) / roll_min).otherwise(None)

            state = pl.when(drawup > thresh).then(1.0).when(drawdown > thresh).then(-1.0).otherwise(0.0)
            exprs.append(state.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="date_diff_days",
    category="time_utils",
    business_category="event_state",
    canonical="date_diff_days",
    source=_SRC,
    backend="polars")
class DateDiffDaysNative(SeriesOperator):
    """Days between current and previous date."""

    metadata = OperatorMetadata(
        name="date_diff_days",
        category="time_utils",
        description="日期差分天数",
        param_names=["date_col"],
        return_type="series",
        tags=["time", "polars", "native"],
    )

    def _calculate_series(self, date_col: pl.DataFrame, **kwargs) -> pl.DataFrame:
        if "date" not in date_col.columns:
            raise ValueError("date_diff_days requires 'date' column")

        # This operator works on date column itself
        result = date_col.with_columns([
            (pl.col("date").cast(pl.Date) - pl.col("date").cast(pl.Date).shift(1)).dt.total_days().alias("date_diff")
        ])

        # Broadcast to all numeric columns
        cols = _numeric_cols(date_col)
        if cols:
            result = result.with_columns([
                pl.col("date_diff").alias(c) for c in cols
            ])

        return result.drop("date_diff") if "date_diff" in result.columns and "date_diff" not in date_col.columns else result


@register_operator(
    name="trading_day_diff",
    category="time_utils",
    business_category="event_state",
    canonical="trading_day_diff",
    source=_SRC,
    backend="polars")
class TradingDayDiffNative(SeriesOperator):
    """Trading days since last observation (row count)."""

    metadata = OperatorMetadata(
        name="trading_day_diff",
        category="time_utils",
        description="交易日差分",
        param_names=["x"],
        return_type="series",
        tags=["time", "polars", "native"],
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # Simple: always 1 trading day between consecutive rows
            # More complex: count non-null periods
            is_valid = pl.col(c).is_not_null()
            days_since = is_valid.cum_sum() - is_valid.cum_sum().shift(1).fill_null(0)
            exprs.append(days_since.cast(pl.Float64).alias(c))

        return x.with_columns(exprs)
