# -*- coding: utf-8 -*-
"""Event analysis operators - Polars native implementations.

Event operators analyze discrete events and their temporal patterns.
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
    name="event_frequency",
    category="event",
    business_category="event",
    canonical="event_frequency",
    source=_SRC,
    backend="polars")
class EventFrequencyNative(SeriesOperator):
    """Count of events in rolling window."""

    metadata = OperatorMetadata(
        name="event_frequency",
        category="event",
        description="事件频率",
        param_names=["x", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)

        # TODO: Implement event frequency
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_active_count",
    category="event",
    business_category="event",
    canonical="event_active_count",
    source=_SRC,
    backend="polars")
class EventActiveCountNative(SeriesOperator):
    """Count of active (non-zero) events in window."""

    metadata = OperatorMetadata(
        name="event_active_count",
        category="event",
        description="活跃事件计数",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement active event count with event_effective_lag
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_cumulative_return_past",
    category="event",
    business_category="event",
    canonical="event_cumulative_return_past",
    source=_SRC,
    backend="polars")
class EventCumulativeReturnPastNative(SeriesOperator):
    """Cumulative return since last event."""

    metadata = OperatorMetadata(
        name="event_cumulative_return_past",
        category="event",
        description="事件后累计收益",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement cumulative return past
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_abnormal_return_past",
    category="event",
    business_category="event",
    canonical="event_abnormal_return_past",
    source=_SRC,
    backend="polars")
class EventAbnormalReturnPastNative(SeriesOperator):
    """Abnormal return since last event (vs benchmark)."""

    metadata = OperatorMetadata(
        name="event_abnormal_return_past",
        category="event",
        description="事件后超额收益",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement abnormal return past
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_arithmetic_return_sum",
    category="event",
    business_category="event",
    canonical="event_arithmetic_return_sum",
    source=_SRC,
    backend="polars")
class EventArithmeticReturnSumNative(SeriesOperator):
    """Sum of arithmetic returns over event window."""

    metadata = OperatorMetadata(
        name="event_arithmetic_return_sum",
        category="event",
        description="事件窗口算术收益和",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement arithmetic return sum
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_log_return_sum",
    category="event",
    business_category="event",
    canonical="event_log_return_sum",
    source=_SRC,
    backend="polars")
class EventLogReturnSumNative(SeriesOperator):
    """Sum of log returns over event window."""

    metadata = OperatorMetadata(
        name="event_log_return_sum",
        category="event",
        description="事件窗口对数收益和",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement log return sum
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_return_since_last",
    category="event",
    business_category="event",
    canonical="event_return_since_last",
    source=_SRC,
    backend="polars")
class EventReturnSinceLastNative(SeriesOperator):
    """Return accumulated since last event occurrence."""

    metadata = OperatorMetadata(
        name="event_return_since_last",
        category="event",
        description="自上次事件以来收益",
        param_names=["ret", "event", "window", "event_effective_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1),
            "event_effective_lag": ParamSpec(dtype=int, min=0),
        },
    )

    def _calculate_series(self, ret: pl.DataFrame, event: pl.DataFrame, window: int = 20, event_effective_lag: int = 1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        lag = strict_integer(event_effective_lag, "event_effective_lag", minimum=0)

        # TODO: Implement return since last
        cols = _numeric_cols(ret)
        return ret.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_decay_asof",
    category="event",
    business_category="event",
    canonical="event_decay_asof",
    source=_SRC,
    backend="polars")
class EventDecayAsofNative(SeriesOperator):
    """Exponentially decayed value since last event."""

    metadata = OperatorMetadata(
        name="event_decay_asof",
        category="event",
        description="事件指数衰减值",
        param_names=["x", "event_indicator", "half_life"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "half_life": ParamSpec(dtype=float, min=0.1, default=5.0, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         half_life: float = 5.0, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_decay_asof requires event_indicator")
        from cleaned_operators.parameter_validation import strict_finite_scalar
        hl = strict_finite_scalar(half_life, "half_life", minimum=0.1)

        # TODO: Implement event decay asof
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_cluster_count",
    category="event",
    business_category="event",
    canonical="event_cluster_count",
    source=_SRC,
    backend="polars")
class EventClusterCountNative(SeriesOperator):
    """Count of event clusters in window."""

    metadata = OperatorMetadata(
        name="event_cluster_count",
        category="event",
        description="事件聚类数",
        param_names=["event_indicator", "window", "max_gap"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "max_gap": ParamSpec(dtype=int, min=1, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, max_gap: int = 3, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        gap = strict_integer(max_gap, "max_gap", minimum=1)

        # TODO: Implement cluster count
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_cluster_mean_size",
    category="event",
    business_category="event",
    canonical="event_cluster_mean_size",
    source=_SRC,
    backend="polars")
class EventClusterMeanSizeNative(SeriesOperator):
    """Average size of event clusters."""

    metadata = OperatorMetadata(
        name="event_cluster_mean_size",
        category="event",
        description="事件聚类平均大小",
        param_names=["event_indicator", "window", "max_gap"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "max_gap": ParamSpec(dtype=int, min=1, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, max_gap: int = 3, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=1)
        gap = strict_integer(max_gap, "max_gap", minimum=1)

        # TODO: Implement cluster mean size
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_fano_factor",
    category="event",
    business_category="event",
    canonical="event_fano_factor",
    source=_SRC,
    backend="polars")
class EventFanoFactorNative(SeriesOperator):
    """Fano factor: variance / mean of event counts."""

    metadata = OperatorMetadata(
        name="event_fano_factor",
        category="event",
        description="Fano因子",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement Fano factor
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_fano_excess",
    category="event",
    business_category="event",
    canonical="event_fano_excess",
    source=_SRC,
    backend="polars")
class EventFanoExcessNative(SeriesOperator):
    """Excess Fano factor: (variance - mean) / mean."""

    metadata = OperatorMetadata(
        name="event_fano_excess",
        category="event",
        description="超额Fano因子",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement Fano excess
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_allan_factor",
    category="event",
    business_category="event",
    canonical="event_allan_factor",
    source=_SRC,
    backend="polars")
class EventAllanFactorNative(SeriesOperator):
    """Allan factor for event timing stability."""

    metadata = OperatorMetadata(
        name="event_allan_factor",
        category="event",
        description="Allan因子",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement Allan factor
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_allan_log_mean",
    category="event",
    business_category="event",
    canonical="event_allan_log_mean",
    source=_SRC,
    backend="polars")
class EventAllanLogMeanNative(SeriesOperator):
    """Log of Allan factor mean."""

    metadata = OperatorMetadata(
        name="event_allan_log_mean",
        category="event",
        description="Allan对数均值",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=2)

        # TODO: Implement Allan log mean
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_allan_scaling_slope",
    category="event",
    business_category="event",
    canonical="event_allan_scaling_slope",
    source=_SRC,
    backend="polars")
class EventAllanScalingSlopeNative(SeriesOperator):
    """Scaling exponent of Allan factor vs window size."""

    metadata = OperatorMetadata(
        name="event_allan_scaling_slope",
        category="event",
        description="Allan标度斜率",
        param_names=["event_indicator", "min_window", "max_window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "min_window": ParamSpec(dtype=int, min=2, default=5, searchable=True, param_role=ParamRole.HORIZON),
            "max_window": ParamSpec(dtype=int, min=3, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, min_window: int = 5, max_window: int = 20, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        min_w = strict_integer(min_window, "min_window", minimum=2)
        max_w = strict_integer(max_window, "max_window", minimum=3)

        # TODO: Implement Allan scaling slope
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_hawkes_branching_ratio_proxy",
    category="event",
    business_category="event",
    canonical="event_hawkes_branching_ratio_proxy",
    source=_SRC,
    backend="polars")
class EventHawkesBranchingRatioProxyNative(SeriesOperator):
    """Proxy for Hawkes process branching ratio."""

    metadata = OperatorMetadata(
        name="event_hawkes_branching_ratio_proxy",
        category="event",
        description="Hawkes分支比率代理",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement Hawkes branching ratio proxy
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_interval_memory",
    category="event",
    business_category="event",
    canonical="event_interval_memory",
    source=_SRC,
    backend="polars")
class EventIntervalMemoryNative(SeriesOperator):
    """Correlation between successive inter-event intervals."""

    metadata = OperatorMetadata(
        name="event_interval_memory",
        category="event",
        description="事件间隔记忆",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=50, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 50, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement interval memory
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_interval_mark_coupling",
    category="event",
    business_category="event",
    canonical="event_interval_mark_coupling",
    source=_SRC,
    backend="polars")
class EventIntervalMarkCouplingNative(SeriesOperator):
    """Correlation between inter-event interval and event mark."""

    metadata = OperatorMetadata(
        name="event_interval_mark_coupling",
        category="event",
        description="事件间隔-标记耦合",
        param_names=["event_mark", "event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=50, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_mark: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         window: int = 50, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_interval_mark_coupling requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement interval mark coupling
        cols = _numeric_cols(event_mark)
        return event_mark.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_mark_autocorr",
    category="event",
    business_category="event",
    canonical="event_mark_autocorr",
    source=_SRC,
    backend="polars")
class EventMarkAutocorrNative(SeriesOperator):
    """Autocorrelation of event marks."""

    metadata = OperatorMetadata(
        name="event_mark_autocorr",
        category="event",
        description="事件标记自相关",
        param_names=["event_mark", "lag", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=10, default=50, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_mark: pl.DataFrame, lag: int = 1, window: int = 50, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        lag_val = strict_integer(lag, "lag", minimum=1)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement mark autocorr
        cols = _numeric_cols(event_mark)
        return event_mark.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_refractory",
    category="event",
    business_category="event",
    canonical="event_refractory",
    source=_SRC,
    backend="polars")
class EventRefractoryNative(SeriesOperator):
    """Refractory period indicator after event."""

    metadata = OperatorMetadata(
        name="event_refractory",
        category="event",
        description="事件不应期",
        param_names=["event_indicator", "refractory_period"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "refractory_period": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, refractory_period: int = 5, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        rp = strict_integer(refractory_period, "refractory_period", minimum=1)

        # TODO: Implement refractory period
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_local_variation",
    category="event",
    business_category="event",
    canonical="event_local_variation",
    source=_SRC,
    backend="polars")
class EventLocalVariationNative(SeriesOperator):
    """Local variation coefficient of inter-event intervals."""

    metadata = OperatorMetadata(
        name="event_local_variation",
        category="event",
        description="事件局部变异",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=50, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 50, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement local variation
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_level_survival_share",
    category="event",
    business_category="event",
    canonical="event_level_survival_share",
    source=_SRC,
    backend="polars")
class EventLevelSurvivalShareNative(SeriesOperator):
    """Share of events that persist for given duration."""

    metadata = OperatorMetadata(
        name="event_level_survival_share",
        category="event",
        description="事件水平存活比例",
        param_names=["event_indicator", "duration", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "duration": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, duration: int = 5, window: int = 60, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        dur = strict_integer(duration, "duration", minimum=1)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement level survival share
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_historical_response_mean",
    category="event",
    business_category="event",
    canonical="event_historical_response_mean",
    source=_SRC,
    backend="polars")
class EventHistoricalResponseMeanNative(SeriesOperator):
    """Mean response magnitude following similar events."""

    metadata = OperatorMetadata(
        name="event_historical_response_mean",
        category="event",
        description="历史事件响应均值",
        param_names=["response", "event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         window: int = 60, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_historical_response_mean requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement historical response mean
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_historical_response_sign_balance",
    category="event",
    business_category="event",
    canonical="event_historical_response_sign_balance",
    source=_SRC,
    backend="polars")
class EventHistoricalResponseSignBalanceNative(SeriesOperator):
    """Balance of positive vs negative historical responses."""

    metadata = OperatorMetadata(
        name="event_historical_response_sign_balance",
        category="event",
        description="历史响应符号平衡",
        param_names=["response", "event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         window: int = 60, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_historical_response_sign_balance requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement response sign balance
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_peak_lag",
    category="event",
    business_category="event",
    canonical="event_response_peak_lag",
    source=_SRC,
    backend="polars")
class EventResponsePeakLagNative(SeriesOperator):
    """Lag to peak response after event."""

    metadata = OperatorMetadata(
        name="event_response_peak_lag",
        category="event",
        description="事件响应峰值滞后",
        param_names=["response", "event_indicator", "max_lag"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "max_lag": ParamSpec(dtype=int, min=1, default=10, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         max_lag: int = 10, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_response_peak_lag requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        ml = strict_integer(max_lag, "max_lag", minimum=1)

        # TODO: Implement response peak lag
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_decay_rate",
    category="event",
    business_category="event",
    canonical="event_response_decay_rate",
    source=_SRC,
    backend="polars")
class EventResponseDecayRateNative(SeriesOperator):
    """Decay rate of response after event."""

    metadata = OperatorMetadata(
        name="event_response_decay_rate",
        category="event",
        description="事件响应衰减率",
        param_names=["response", "event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=10, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         window: int = 10, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_response_decay_rate requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=3)

        # TODO: Implement response decay rate
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_dispersion",
    category="event",
    business_category="event",
    canonical="event_response_dispersion",
    source=_SRC,
    backend="polars")
class EventResponseDispersionNative(SeriesOperator):
    """Dispersion of responses across similar events."""

    metadata = OperatorMetadata(
        name="event_response_dispersion",
        category="event",
        description="事件响应分散度",
        param_names=["response", "event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         window: int = 60, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_response_dispersion requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement response dispersion
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_effective_events",
    category="event",
    business_category="event",
    canonical="event_response_effective_events",
    source=_SRC,
    backend="polars")
class EventResponseEffectiveEventsNative(SeriesOperator):
    """Effective number of independent events contributing to response."""

    metadata = OperatorMetadata(
        name="event_response_effective_events",
        category="event",
        description="有效事件数",
        param_names=["event_indicator", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, window: int = 60, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement effective events
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_overlap_ratio",
    category="event",
    business_category="event",
    canonical="event_response_overlap_ratio",
    source=_SRC,
    backend="polars")
class EventResponseOverlapRatioNative(SeriesOperator):
    """Ratio of overlapping event response windows."""

    metadata = OperatorMetadata(
        name="event_response_overlap_ratio",
        category="event",
        description="事件响应重叠比例",
        param_names=["event_indicator", "response_window", "window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "response_window": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, event_indicator: pl.DataFrame, response_window: int = 5, window: int = 60, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        rw = strict_integer(response_window, "response_window", minimum=1)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement response overlap ratio
        cols = _numeric_cols(event_indicator)
        return event_indicator.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="event_response_reversal_strength",
    category="event",
    business_category="event",
    canonical="event_response_reversal_strength",
    source=_SRC,
    backend="polars")
class EventResponseReversalStrengthNative(SeriesOperator):
    """Strength of reversal after initial response."""

    metadata = OperatorMetadata(
        name="event_response_reversal_strength",
        category="event",
        description="事件响应反转强度",
        param_names=["response", "event_indicator", "initial_window", "reversal_window"],
        return_type="series",
        tags=["event", "polars", "native"],
        param_specs={
            "initial_window": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
            "reversal_window": ParamSpec(dtype=int, min=1, default=10, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, response: pl.DataFrame, event_indicator: pl.DataFrame | None = None,
                         initial_window: int = 5, reversal_window: int = 10, **kwargs) -> pl.DataFrame:
        if event_indicator is None:
            raise ValueError("event_response_reversal_strength requires event_indicator")
        from cleaned_operators.parameter_validation import strict_integer
        iw = strict_integer(initial_window, "initial_window", minimum=1)
        rw = strict_integer(reversal_window, "reversal_window", minimum=1)

        # TODO: Implement response reversal strength
        cols = _numeric_cols(response)
        return response.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])
