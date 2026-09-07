# -*- coding: utf-8 -*-
"""Basic time-series operators - Polars native implementations.

All operators use pure Polars expressions (no pandas/numpy fallback).
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"
_MOMENT_CANONICALS = frozenset({"ts_kurt", "ts_moment"})


def _register_basic_operator(**kwargs):
    """Keep quarantined moment classes importable without production registration."""
    if kwargs.get("canonical") in _MOMENT_CANONICALS:
        return lambda cls: cls
    return register_operator(**kwargs)


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


# ---------------------------------------------------------------------------
# Basic rolling statistics
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_mean",
    category="time_series",
    business_category="time_series",
    canonical="ts_mean",
    source=_SRC,
    backend="polars",
)
class TSMeanNative(SeriesOperator):
    """Rolling mean."""

    # 100k GO P0#1: parity with the canonical ``TSMeanPolars`` in
    # ``common.time_series`` — carry an explicit PhysicalImplementationSpec so a
    # pytest late-surface staging of this module (conftest stages it before the
    # canonical time_series registration) does NOT leave the ts_mean/polars slot
    # spec-less.  Mirrors the time_series spec's binding fields.
    _physical_spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=False,
        stateful=False,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash="cleaned_operators.common.polars_ts_basic:TSMeanNative:v1",
        emitter_identity="polars.Expr.rolling_mean:v1",
        parameter_domain_hash="ts_mean.window:int:min=1",
        semantic_contract_hash="ts_mean:min_samples=1:axis=time:v1",
    )

    metadata = OperatorMetadata(
        name="ts_mean",
        category="time_series",
        description="滚动均值",
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "min_periods": ParamSpec(dtype=int, min=1, default=1, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        mp = strict_integer(min_periods, "min_periods", minimum=1) if min_periods is not None else 1
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples matches pandas rolling(window, min_periods).
        return x.lazy().with_columns([pl.col(c).rolling_mean(window_size=w, min_samples=mp).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_std",
    category="time_series",
    business_category="time_series",
    canonical="ts_std",
    source=_SRC,
    backend="polars",
)
class TSStdNative(SeriesOperator):
    """Rolling standard deviation with configurable ddof (delta degrees of freedom).

    ddof=1 (default): sample standard deviation (denominator: n-1)
    ddof=0: population standard deviation (denominator: n)
    """

    metadata = OperatorMetadata(
        name="ts_std",
        category="time_series",
        description="滚动标准差",
        param_names=["x", "d", "ddof"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "ddof": ParamSpec(dtype=int, min=0, max=1, default=1, searchable=False, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, ddof: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        delta_dof = strict_integer(ddof, "ddof", minimum=0, maximum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples=1 reproduces the pandas reference
        # ``rolling(window, min_periods=1)`` warmup — NaN only while fewer than
        # one observation has been seen, so a full window is not required to
        # start emitting (pandas semantics).  The polars default (full window)
        # emitted NaN for every window not yet full, diverging from pandas on
        # plain/NaN-gap/short inputs.
        return x.lazy().with_columns([pl.col(c).rolling_std(window_size=w, ddof=delta_dof, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_sum",
    category="time_series",
    business_category="time_series",
    canonical="ts_sum",
    source=_SRC,
    backend="polars",
)
class TSSumNative(SeriesOperator):
    """Rolling sum."""

    metadata = OperatorMetadata(
        name="ts_sum",
        category="time_series",
        description="滚动求和",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples=1 == pandas rolling(window, min_periods=1)
        # warmup (emit from the first observation; NaN only before any data).
        return x.lazy().with_columns([pl.col(c).rolling_sum(window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_max",
    category="time_series",
    business_category="time_series",
    canonical="ts_max",
    source=_SRC,
    backend="polars",
)
class TSMaxNative(SeriesOperator):
    """Rolling maximum."""

    metadata = OperatorMetadata(
        name="ts_max",
        category="time_series",
        description="滚动最大值",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples=1 == pandas rolling(window, min_periods=1).
        return x.lazy().with_columns([pl.col(c).rolling_max(window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_min",
    category="time_series",
    business_category="time_series",
    canonical="ts_min",
    source=_SRC,
    backend="polars",
)
class TSMinNative(SeriesOperator):
    """Rolling minimum."""

    metadata = OperatorMetadata(
        name="ts_min",
        category="time_series",
        description="滚动最小值",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples=1 == pandas rolling(window, min_periods=1).
        return x.lazy().with_columns([pl.col(c).rolling_min(window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_median",
    category="time_series",
    business_category="time_series",
    canonical="ts_median",
    source=_SRC,
    backend="polars",
)
class TSMedianNative(SeriesOperator):
    """Rolling median."""

    metadata = OperatorMetadata(
        name="ts_median",
        category="time_series",
        description="滚动中位数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: min_samples=1 == pandas rolling(window, min_periods=1).
        return x.lazy().with_columns([pl.col(c).rolling_median(window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_quantile",
    category="time_series",
    business_category="time_series",
    canonical="ts_quantile",
    source=_SRC,
    backend="polars",
)
class TSQuantileNative(SeriesOperator):
    """Rolling quantile."""

    metadata = OperatorMetadata(
        name="ts_quantile",
        category="time_series",
        description="滚动分位数",
        param_names=["x", "d", "q"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "q": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.STATE_THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, q: float = 0.5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

        w = strict_integer(d, "d", minimum=1)
        quantile = strict_finite_scalar(q, "q", minimum=0.0, maximum=1.0)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: pandas rolling.quantile uses min_periods=1 -> min_samples=1.
        return x.lazy().with_columns([pl.col(c).rolling_quantile(quantile=quantile, window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@_register_basic_operator(
    name="ts_kurt",
    category="time_series",
    business_category="time_series",
    canonical="ts_kurt",
    source=_SRC,
    backend="polars",
    status="deprecated",
)
class TSKurtNative(SeriesOperator):
    """Rolling kurtosis (excess kurtosis)."""

    metadata = OperatorMetadata(
        name="ts_kurt",
        category="time_series",
        description="滚动峰度",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=4, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=4)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            std = pl.col(c).rolling_std(window_size=w)
            m4 = ((pl.col(c) - mean) ** 4).rolling_mean(window_size=w)
            exprs.append(
                pl.when(std.is_null() | (std == 0))
                .then(None)
                .otherwise(m4 / (std ** 4) - 3.0)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@_register_basic_operator(
    name="ts_moment",
    category="time_series",
    business_category="time_series",
    canonical="ts_moment",
    source=_SRC,
    backend="polars",
    status="deprecated",
)
class TSMomentNative(SeriesOperator):
    """Rolling n-th central moment."""

    metadata = OperatorMetadata(
        name="ts_moment",
        category="time_series",
        description="滚动n阶中心矩",
        param_names=["x", "d", "n"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "n": ParamSpec(dtype=int, min=1, default=3, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, n: int = 3, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=2)
        order = strict_integer(n, "n", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w)
            exprs.append(((pl.col(c) - mean) ** order).rolling_mean(window_size=w).alias(c))
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Change & lag operators
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_delta",
    category="time_series",
    business_category="time_series",
    canonical="ts_delta",
    source=_SRC,
    backend="polars",
)
class TSDeltaNative(SeriesOperator):
    """x_t - x_{t-d}."""

    metadata = OperatorMetadata(
        name="ts_delta",
        category="time_series",
        description="d期差分",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([(pl.col(c) - pl.col(c).shift(lag)).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_returns",
    category="time_series",
    business_category="time_series",
    canonical="ts_returns",
    source=_SRC,
    backend="polars",
)
class TSReturnsNative(SeriesOperator):
    """Simple return: (x_t - x_{t-d}) / x_{t-d}."""

    metadata = OperatorMetadata(
        name="ts_returns",
        category="time_series",
        description="d期简单收益",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            prev = pl.col(c).shift(lag)
            exprs.append(
                pl.when(prev.is_null() | (prev == 0))
                .then(None)
                .otherwise((pl.col(c) - prev) / prev)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_delay",
    category="time_series",
    business_category="time_series",
    canonical="ts_delay",
    source=_SRC,
    backend="polars",
)
class TSDelayNative(SeriesOperator):
    """Lag by d periods (alias for shift)."""

    metadata = OperatorMetadata(
        name="ts_delay",
        category="time_series",
        description="滞后d期",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        lag = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([pl.col(c).shift(lag).alias(c) for c in cols]).collect()


# ---------------------------------------------------------------------------
# Argmax/argmin and related
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_argmax",
    category="time_series",
    business_category="time_series",
    canonical="ts_argmax",
    source=_SRC,
    backend="polars",
)
class TSArgmaxNative(SeriesOperator):
    """Index of maximum value in rolling window (0-based from window start)."""

    metadata = OperatorMetadata(
        name="ts_argmax",
        category="time_series",
        description="滚动窗口最大值索引",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: canonical ts_argmax = AGE of the max (0 = current bar,
        # tie = newest), matching pandas rolling_days_since_extreme.  The old
        # ``window-1 - arg_max`` produced the opposite offset and, combined with
        # rolling_map's full-window default, NaN for every non-full window.
        def _age_max(s: pl.Series) -> float:
            import numpy as _np

            v = _np.asarray(s.to_numpy(), dtype=float)
            finite = v[_np.isfinite(v)]
            if finite.size == 0:
                return _np.nan
            value = float(_np.max(finite))
            hits = _np.flatnonzero(_np.isfinite(v) & (v == value))
            return float(v.size - 1 - int(hits[-1]))

        return x.lazy().with_columns([pl.col(c).rolling_map(_age_max, window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_argmin",
    category="time_series",
    business_category="time_series",
    canonical="ts_argmin",
    source=_SRC,
    backend="polars",
)
class TSArgminNative(SeriesOperator):
    """Index of minimum value in rolling window."""

    metadata = OperatorMetadata(
        name="ts_argmin",
        category="time_series",
        description="滚动窗口最小值索引",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        # ROLLING-EDGE: canonical ts_argmin = AGE of the min (0 = current bar,
        # tie = newest), matching pandas rolling_days_since_extreme.
        def _age_min(s: pl.Series) -> float:
            import numpy as _np

            v = _np.asarray(s.to_numpy(), dtype=float)
            finite = v[_np.isfinite(v)]
            if finite.size == 0:
                return _np.nan
            value = float(_np.min(finite))
            hits = _np.flatnonzero(_np.isfinite(v) & (v == value))
            return float(v.size - 1 - int(hits[-1]))

        return x.lazy().with_columns([pl.col(c).rolling_map(_age_min, window_size=w, min_samples=1).alias(c) for c in cols]).collect()


@register_operator(
    name="ts_argmax_age",
    category="time_series",
    business_category="time_series",
    canonical="ts_argmax_age",
    source=_SRC,
    backend="polars",
)
class TSArgmaxAgeNative(SeriesOperator):
    """Days since maximum in window (window_size - 1 - argmax)."""

    metadata = OperatorMetadata(
        name="ts_argmax_age",
        category="time_series",
        description="距离最大值天数",
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        mp = strict_integer(min_periods, "min_periods", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            def age(s):
                import numpy as np
                v = np.asarray(s.to_numpy(), dtype=float)
                hits = np.flatnonzero(np.isfinite(v) & (v == np.nanmax(v)))
                return float(v.size - 1 - hits[-1]) if hits.size else np.nan
            exprs.append(pl.col(c).rolling_map(age, window_size=w, min_samples=mp).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_argmin_age",
    category="time_series",
    business_category="time_series",
    canonical="ts_argmin_age",
    source=_SRC,
    backend="polars",
)
class TSArgminAgeNative(SeriesOperator):
    """Days since minimum in window."""

    metadata = OperatorMetadata(
        name="ts_argmin_age",
        category="time_series",
        description="距离最小值天数",
        param_names=["x", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        mp = strict_integer(min_periods, "min_periods", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            def age(s):
                import numpy as np
                v = np.asarray(s.to_numpy(), dtype=float)
                hits = np.flatnonzero(np.isfinite(v) & (v == np.nanmin(v)))
                return float(v.size - 1 - hits[-1]) if hits.size else np.nan
            exprs.append(pl.col(c).rolling_map(age, window_size=w, min_samples=mp).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_new_high",
    category="time_series",
    business_category="time_series",
    canonical="ts_new_high",
    source=_SRC,
    backend="polars",
)
class TSNewHighNative(SeriesOperator):
    """1 if current value equals rolling max, else 0."""

    metadata = OperatorMetadata(
        name="ts_new_high",
        category="time_series",
        description="是否创新高",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            max_val = pl.col(c).rolling_max(window_size=w)
            exprs.append(
                pl.when(pl.col(c).is_null() | max_val.is_null())
                .then(None)
                .when(pl.col(c) == max_val)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_new_low",
    category="time_series",
    business_category="time_series",
    canonical="ts_new_low",
    source=_SRC,
    backend="polars",
)
class TSNewLowNative(SeriesOperator):
    """1 if current value equals rolling min, else 0."""

    metadata = OperatorMetadata(
        name="ts_new_low",
        category="time_series",
        description="是否创新低",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            min_val = pl.col(c).rolling_min(window_size=w)
            exprs.append(
                pl.when(pl.col(c).is_null() | min_val.is_null())
                .then(None)
                .when(pl.col(c) == min_val)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Conditional statistics
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_rank_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_rank_if",
    source=_SRC,
    backend="polars",
)
class TSRankIfNative(SeriesOperator):
    """Rolling rank considering only values where condition is true."""

    metadata = OperatorMetadata(
        name="ts_rank_if",
        category="time_series",
        description="条件滚动排名",
        param_names=["x", "condition", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"cond": "condition", "d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, condition: pl.DataFrame, window: int = 20,
                          min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cond = condition
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in cond.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(cond[c]).then(pl.col(c)).otherwise(None)
            exprs.append(
                (
                    masked.fill_nan(None).rolling_rank(window_size=w, method="average")
                    / masked.fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w)
                ).alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_count_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_count_if",
    source=_SRC,
    backend="polars",
)
class TSCountIfNative(SeriesOperator):
    """Count of true conditions in rolling window."""

    metadata = OperatorMetadata(
        name="ts_count_if",
        category="time_series",
        description="条件计数",
        # P0-B1: the canonical ts_count_if contract is ``["condition",
        # "window", "min_periods"]`` (daily_panel / overhaul).  This legacy
        # native kernel keeps the exact canonical names so the R6-157 registry
        # invariant (keys(param_specs) ⊆ param_names) holds when the canonical
        # logical contract is inherited by the overhaul layer.  ``d`` is the old
        # alpha-language alias of ``window`` and is exposed via param_aliases,
        # never as a declared parameter / ParamSpec key.
        param_names=["condition", "window", "min_periods"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window", "cond": "condition"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, condition: pl.DataFrame, window: int = 20, min_periods: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        mp = strict_integer(min_periods, "min_periods", minimum=1)
        return condition.select(
            [pl.col(c).cast(pl.Float64).rolling_sum(window_size=w, min_samples=mp).alias(c) for c in condition.columns]
        )


@register_operator(
    name="ts_max_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_max_if",
    source=_SRC,
    backend="polars",
)
class TSMaxIfNative(SeriesOperator):
    """Rolling max of x where condition is true."""

    metadata = OperatorMetadata(
        name="ts_max_if",
        category="time_series",
        description="条件最大值",
        param_names=["x", "cond", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, cond: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in cond.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(cond[c]).then(pl.col(c)).otherwise(None)
            exprs.append(masked.rolling_max(window_size=w).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_min_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_min_if",
    source=_SRC,
    backend="polars",
)
class TSMinIfNative(SeriesOperator):
    """Rolling min of x where condition is true."""

    metadata = OperatorMetadata(
        name="ts_min_if",
        category="time_series",
        description="条件最小值",
        param_names=["x", "cond", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, cond: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in cond.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(cond[c]).then(pl.col(c)).otherwise(None)
            exprs.append(masked.rolling_min(window_size=w).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_mean_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_mean_if",
    source=_SRC,
    backend="polars",
)
class TSMeanIfNative(SeriesOperator):
    """Rolling mean of x where condition is true."""

    metadata = OperatorMetadata(
        name="ts_mean_if",
        category="time_series",
        description="条件均值",
        param_names=["x", "condition", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"cond": "condition", "d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, condition: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in condition.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(condition[c]).then(pl.col(c)).otherwise(None)
            exprs.append(masked.rolling_mean(window_size=w).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_std_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_std_if",
    source=_SRC,
    backend="polars",
)
class TSStdIfNative(SeriesOperator):
    """Rolling std of x where condition is true."""

    metadata = OperatorMetadata(
        name="ts_std_if",
        category="time_series",
        description="条件标准差",
        param_names=["x", "condition", "window", "ddof"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"cond": "condition", "d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, condition: pl.DataFrame, window: int = 20, ddof: int = 1, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in condition.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(condition[c]).then(pl.col(c)).otherwise(None)
            exprs.append(masked.rolling_std(window_size=w, ddof=ddof).alias(c))
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_sum_if",
    category="time_series",
    business_category="time_series",
    canonical="ts_sum_if",
    source=_SRC,
    backend="polars",
)
class TSSumIfNative(SeriesOperator):
    """Rolling sum of x where condition is true."""

    metadata = OperatorMetadata(
        name="ts_sum_if",
        category="time_series",
        description="条件求和",
        param_names=["x", "condition", "window"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window", "cond": "condition"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, condition: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            if c not in condition.columns:
                exprs.append(pl.lit(None).alias(c))
                continue
            masked = pl.when(condition[c]).then(pl.col(c)).otherwise(None)
            exprs.append(masked.rolling_sum(window_size=w).alias(c))
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Coverage and validity metrics
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_valid_count",
    category="time_series",
    business_category="time_series",
    canonical="ts_valid_count",
    source=_SRC,
    backend="polars",
)
class TSValidCountNative(SeriesOperator):
    """Count of non-null values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_valid_count",
        category="time_series",
        description="滚动有效计数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_coverage_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_coverage_ratio",
    source=_SRC,
    backend="polars",
)
class TSCoverageRatioNative(SeriesOperator):
    """Ratio of non-null values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_coverage_ratio",
        category="time_series",
        description="滚动覆盖率",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            (pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w) / w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_negative_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_negative_ratio",
    source=_SRC,
    backend="polars",
)
class TSNegativeRatioNative(SeriesOperator):
    """Ratio of negative values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_negative_ratio",
        category="time_series",
        description="滚动负值占比",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            neg_count = (pl.col(c) < 0).cast(pl.Float64).rolling_sum(window_size=w)
            valid_count = pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w)
            exprs.append(
                pl.when(valid_count == 0)
                .then(None)
                .otherwise(neg_count / valid_count)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_positive_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_positive_ratio",
    source=_SRC,
    backend="polars",
)
class TSPositiveRatioNative(SeriesOperator):
    """Ratio of positive values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_positive_ratio",
        category="time_series",
        description="滚动正值占比",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            pos_count = (pl.col(c) > 0).cast(pl.Float64).rolling_sum(window_size=w)
            valid_count = pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w)
            exprs.append(
                pl.when(valid_count == 0)
                .then(None)
                .otherwise(pos_count / valid_count)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_zero_ratio",
    category="time_series",
    business_category="time_series",
    canonical="ts_zero_ratio",
    source=_SRC,
    backend="polars",
)
class TSZeroRatioNative(SeriesOperator):
    """Ratio of zero values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_zero_ratio",
        category="time_series",
        description="滚动零值占比",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            zero_count = (pl.col(c) == 0).cast(pl.Float64).rolling_sum(window_size=w)
            valid_count = pl.col(c).fill_nan(None).is_not_null().cast(pl.Float64).rolling_sum(window_size=w)
            exprs.append(
                pl.when(valid_count == 0)
                .then(None)
                .otherwise(zero_count / valid_count)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Days since events
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_days_since",
    category="time_series",
    business_category="time_series",
    canonical="ts_days_since",
    source=_SRC,
    backend="polars", status="unsupported",
)
class TSDaysSinceNative(SeriesOperator):
    """Days since condition was last true."""

    metadata = OperatorMetadata(
        name="ts_days_since",
        category="time_series",
        description="距上次条件为真天数",
        param_names=["condition", "max_lookback"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"cond": "condition"},
    )

    def _calculate_series(self, condition: pl.DataFrame, max_lookback: int = 252, **kwargs) -> pl.DataFrame:
        raise NotImplementedError("legacy ts_days_since candidate disabled: max_lookback/null contract is not certified")
        cols = _numeric_cols(condition)
        exprs = []
        for c in cols:
            # Use cumsum to track position, then diff to find days since
            event_idx = pl.when(condition[c]).then(pl.lit(1)).otherwise(pl.lit(0)).cum_sum()
            exprs.append(
                (pl.int_range(0, pl.len()).over(event_idx) - 1)
                .cast(pl.Float64)
                .alias(c)
            )
        return condition.with_columns(exprs)


@register_operator(
    name="ts_days_since_high",
    category="time_series",
    business_category="time_series",
    canonical="ts_days_since_high",
    source=_SRC,
    backend="polars", status="unsupported",
)
class TSDaysSinceHighNative(SeriesOperator):
    """Days since d-period high."""

    metadata = OperatorMetadata(
        name="ts_days_since_high",
        category="time_series",
        description="距新高天数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 252, **kwargs) -> pl.DataFrame:
        raise NotImplementedError("legacy ts_days_since_high candidate disabled: event-age contract is not certified")
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            max_val = pl.col(c).rolling_max(window_size=w)
            is_high = pl.col(c) == max_val
            event_idx = pl.when(is_high).then(pl.lit(1)).otherwise(pl.lit(0)).cum_sum()
            exprs.append(
                (pl.int_range(0, pl.len()).over(event_idx) - 1)
                .cast(pl.Float64)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


@register_operator(
    name="ts_days_since_low",
    category="time_series",
    business_category="time_series",
    canonical="ts_days_since_low",
    source=_SRC,
    backend="polars", status="unsupported",
)
class TSDaysSinceLowNative(SeriesOperator):
    """Days since d-period low."""

    metadata = OperatorMetadata(
        name="ts_days_since_low",
        category="time_series",
        description="距新低天数",
        param_names=["x", "d"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 252, **kwargs) -> pl.DataFrame:
        raise NotImplementedError("legacy ts_days_since_low candidate disabled: event-age contract is not certified")
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(d, "d", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            min_val = pl.col(c).rolling_min(window_size=w)
            is_low = pl.col(c) == min_val
            event_idx = pl.when(is_low).then(pl.lit(1)).otherwise(pl.lit(0)).cum_sum()
            exprs.append(
                (pl.int_range(0, pl.len()).over(event_idx) - 1)
                .cast(pl.Float64)
                .alias(c)
            )
        return x.lazy().with_columns(exprs).collect()


# ---------------------------------------------------------------------------
# Top-k and bottom-k statistics
# ---------------------------------------------------------------------------


@register_operator(
    name="ts_topk_mean",
    category="time_series",
    business_category="time_series",
    canonical="ts_topk_mean",
    source=_SRC,
    backend="polars",
)
class TSTopkMeanNative(SeriesOperator):
    """Mean of top k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_topk_mean",
        category="time_series",
        description="滚动topk均值",
        param_names=["x", "window", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, k: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        topk = strict_integer(k, "k", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.top_k(topk).mean(), window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_topk_std",
    category="time_series",
    business_category="time_series",
    canonical="ts_topk_std",
    source=_SRC,
    backend="polars",
)
class TSTopkStdNative(SeriesOperator):
    """Std of top k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_topk_std",
        category="time_series",
        description="滚动topk标准差",
        param_names=["x", "window", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, k: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        topk = strict_integer(k, "k", minimum=2)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.top_k(topk).std(), window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_topk_sum",
    category="time_series",
    business_category="time_series",
    canonical="ts_topk_sum",
    source=_SRC,
    backend="polars",
)
class TSTopkSumNative(SeriesOperator):
    """Sum of top k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_topk_sum",
        category="time_series",
        description="滚动topk求和",
        param_names=["x", "d", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        # R19-050: canonical aliases window→d, n→k (P0-23 single logical
        # authority — the pandas reference in common/time_series.py owns the
        # contract; the legacy native surface must declare the same spelling).
        param_aliases={"window": "d", "n": "k"},
        param_specs={
            "d": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=None, searchable=True, param_role=ParamRole.ECONOMIC),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, k: int | None = None, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(kwargs.get("window", d), "d", minimum=1)
        topk = strict_integer(k if k is not None else kwargs.get("n", d), "k", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.top_k(topk).sum(), window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_bottomk_mean",
    category="time_series",
    business_category="time_series",
    canonical="ts_bottomk_mean",
    source=_SRC,
    backend="polars",
)
class TSBottomkMeanNative(SeriesOperator):
    """Mean of bottom k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_bottomk_mean",
        category="time_series",
        description="滚动bottomk均值",
        param_names=["x", "window", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, k: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        bottomk = strict_integer(k, "k", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.bottom_k(bottomk).mean(), window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_bottomk_std",
    category="time_series",
    business_category="time_series",
    canonical="ts_bottomk_std",
    source=_SRC,
    backend="polars",
)
class TSBottomkStdNative(SeriesOperator):
    """Std of bottom k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_bottomk_std",
        category="time_series",
        description="滚动bottomk标准差",
        param_names=["x", "window", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=2, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, k: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        bottomk = strict_integer(k, "k", minimum=2)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.bottom_k(bottomk).std(), window_size=w).alias(c)
            for c in cols
        ]).collect()


@register_operator(
    name="ts_bottomk_sum",
    category="time_series",
    business_category="time_series",
    canonical="ts_bottomk_sum",
    source=_SRC,
    backend="polars",
)
class TSBottomkSumNative(SeriesOperator):
    """Sum of bottom k values in rolling window."""

    metadata = OperatorMetadata(
        name="ts_bottomk_sum",
        category="time_series",
        description="滚动bottomk求和",
        param_names=["x", "window", "k"],
        return_type="series",
        tags=["time_series", "polars", "native"],
        param_aliases={"d": "window"},
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=20, searchable=True, param_role=ParamRole.HORIZON),
            "k": ParamSpec(dtype=int, min=1, default=5, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, k: int = 5, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        bottomk = strict_integer(k, "k", minimum=1)
        cols = _numeric_cols(x)
        return x.lazy().with_columns([
            pl.col(c).rolling_map(lambda s: s.bottom_k(bottomk).sum(), window_size=w).alias(c)
            for c in cols
        ]).collect()
