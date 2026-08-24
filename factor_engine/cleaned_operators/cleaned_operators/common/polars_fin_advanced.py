# -*- coding: utf-8 -*-
"""Phase 3 Module 13: Advanced financial operators (Polars native).

Implements financial surprise, expectation revision, earnings persistence,
and seasonal analysis operators using pure Polars expressions.
All operators are causal and vectorized across columns.
"""
from __future__ import annotations

try:
    import polars as pl
    import numpy as np
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    np = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec
from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

_SKIP = frozenset({"date", "stock_code"})
_SRC = "polars_fin_advanced_phase3"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Financial Surprise Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_surprise",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_surprise",
    source=_SRC,
    backend="polars")
class FinSurpriseNative(SeriesOperator):
    """Financial surprise: actual - expected."""

    metadata = OperatorMetadata(
        name="fin_surprise",
        category="financial",
        description="财务意外：实际值 - 预期值",
        param_names=["actual", "expected"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(actual)
        exprs = [
            (pl.col(c) - expected[c]).alias(c) for c in cols if c in expected.columns
        ]
        return actual.with_columns(exprs)


@register_operator(
    name="fin_surprise_zscore",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_surprise_zscore",
    source=_SRC,
    backend="polars")
class FinSurpriseZscoreNative(SeriesOperator):
    """Financial surprise z-score: (actual - expected) / std(historical surprises)."""

    metadata = OperatorMetadata(
        name="fin_surprise_zscore",
        category="financial",
        description="财务意外 Z 分数",
        param_names=["actual", "expected", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(actual)
        exprs = []
        for c in cols:
            if c not in expected.columns:
                continue
            surprise = pl.col(c) - expected[c]
            mean = surprise.rolling_mean(window_size=w, min_samples=w)
            std = surprise.rolling_std(window_size=w, min_samples=w)
            z = pl.when((std.is_not_null()) & (std > 1e-12)).then((surprise - mean) / std).otherwise(None)
            exprs.append(z.alias(c))
        return actual.with_columns(exprs) if exprs else actual


@register_operator(
    name="fin_surprise_event_zscore",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_surprise_event_zscore",
    source=_SRC,
    backend="polars")
class FinSurpriseEventZscoreNative(SeriesOperator):
    """Financial surprise event z-score using only announcement dates."""

    metadata = OperatorMetadata(
        name="fin_surprise_event_zscore",
        category="financial",
        description="财务意外事件 Z 分数",
        param_names=["actual", "expected", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(actual)

        def _event_zscore(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < 2:
                return np.nan
            if len(arr) < w:
                return np.nan
            current = arr[-1]
            past = arr[-w:-1] if len(arr) > w else arr[:-1]
            mean = np.mean(past)
            std = np.std(past, ddof=1)
            if std < 1e-12:
                return np.nan
            return float(current - mean) / std

        exprs = []
        for c in cols:
            if c not in expected.columns:
                continue
            surprise = actual[c] - expected[c]
            # Filter to only events (non-null surprises)
            z = surprise.fill_nan(None).rolling_map(_event_zscore, window_size=w+1, min_samples=2)
            exprs.append(z.alias(c))

        return actual.with_columns(exprs) if exprs else actual


@register_operator(
    name="fin_surprise_event_percentile",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_surprise_event_percentile",
    source=_SRC,
    backend="polars")
class FinSurpriseEventPercentileNative(SeriesOperator):
    """Financial surprise percentile rank among historical events."""

    metadata = OperatorMetadata(
        name="fin_surprise_event_percentile",
        category="financial",
        description="财务意外事件百分位",
        param_names=["actual", "expected", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(actual)

        def _event_pctile(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 2:
                return np.nan
            arr = arr[valid]
            if len(arr) < 2:
                return np.nan
            current = arr[-1]
            past = arr[:-1]
            rank = np.sum(past < current) + 0.5 * np.sum(past == current)
            return float(rank) / len(past)

        exprs = []
        for c in cols:
            if c not in expected.columns:
                continue
            surprise = actual[c] - expected[c]
            pct = surprise.fill_nan(None).rolling_map(_event_pctile, window_size=w+1, min_samples=2)
            exprs.append(pct.alias(c))

        return actual.with_columns(exprs) if exprs else actual


# ---------------------------------------------------------------------------
# Expectation Revision Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_expectation_revision",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_revision",
    source=_SRC,
    backend="polars")
class FinExpectationRevisionNative(SeriesOperator):
    """Expectation revision: current_expectation - prior_expectation."""

    metadata = OperatorMetadata(
        name="fin_expectation_revision",
        category="financial",
        description="预期修正",
        param_names=["expectation", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(expectation)
        exprs = [
            (pl.col(c) - pl.col(c).shift(d)).alias(c) for c in cols
        ]
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_expectation_revision_pct",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_revision_pct",
    source=_SRC,
    backend="polars")
class FinExpectationRevisionPctNative(SeriesOperator):
    """Expectation revision percentage: (current - prior) / |prior|."""

    metadata = OperatorMetadata(
        name="fin_expectation_revision_pct",
        category="financial",
        description="预期修正百分比",
        param_names=["expectation", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(expectation)
        exprs = []
        for c in cols:
            prior = pl.col(c).shift(d)
            rev_pct = pl.when((prior.is_not_null()) & (prior.abs() > 1e-12)).then(
                (pl.col(c) - prior) / prior.abs()
            ).otherwise(None)
            exprs.append(rev_pct.alias(c))
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_expectation_revision_magnitude",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_revision_magnitude",
    source=_SRC,
    backend="polars")
class FinExpectationRevisionMagnitudeNative(SeriesOperator):
    """Expectation revision magnitude: |current - prior|."""

    metadata = OperatorMetadata(
        name="fin_expectation_revision_magnitude",
        category="financial",
        description="预期修正幅度",
        param_names=["expectation", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(expectation)
        exprs = [
            (pl.col(c) - pl.col(c).shift(d)).abs().alias(c) for c in cols
        ]
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_expectation_revision_count",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_revision_count",
    source=_SRC,
    backend="polars")
class FinExpectationRevisionCountNative(SeriesOperator):
    """Count of expectation revisions in window."""

    metadata = OperatorMetadata(
        name="fin_expectation_revision_count",
        category="financial",
        description="预期修正次数",
        param_names=["expectation", "window", "threshold"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, window: int = 8, threshold: float = 0.01, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        th = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        cols = _numeric_cols(expectation)
        exprs = []
        for c in cols:
            change = (pl.col(c) - pl.col(c).shift(1)).abs()
            count = (change > th).cast(pl.Float64).rolling_sum(window_size=w, min_samples=1)
            exprs.append(count.alias(c))
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_expectation_revision_speed",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_revision_speed",
    source=_SRC,
    backend="polars")
class FinExpectationRevisionSpeedNative(SeriesOperator):
    """Average magnitude of expectation revisions in window."""

    metadata = OperatorMetadata(
        name="fin_expectation_revision_speed",
        category="financial",
        description="预期修正速度",
        param_names=["expectation", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(expectation)
        exprs = []
        for c in cols:
            change = (pl.col(c) - pl.col(c).shift(1)).abs()
            speed = change.rolling_mean(window_size=w, min_samples=2)
            exprs.append(speed.alias(c))
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_expectation_dispersion",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_expectation_dispersion",
    source=_SRC,
    backend="polars")
class FinExpectationDispersionNative(SeriesOperator):
    """Dispersion of expectations over window (std)."""

    metadata = OperatorMetadata(
        name="fin_expectation_dispersion",
        category="financial",
        description="预期离散度",
        param_names=["expectation", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(expectation)
        exprs = [
            pl.col(c).rolling_std(window_size=w, min_samples=2).alias(c) for c in cols
        ]
        return expectation.with_columns(exprs)


@register_operator(
    name="fin_days_since_expectation_revision",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_days_since_expectation_revision",
    source=_SRC,
    backend="polars")
class FinDaysSinceExpectationRevisionNative(SeriesOperator):
    """Days since last meaningful expectation revision."""

    metadata = OperatorMetadata(
        name="fin_days_since_expectation_revision",
        category="financial",
        description="上次预期修正天数",
        param_names=["expectation", "threshold"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, expectation: pl.DataFrame, threshold: float = 0.01, **kwargs) -> pl.DataFrame:
        th = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        cols = _numeric_cols(expectation)

        def _days_since(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 2:
                return np.nan
            changes = np.abs(np.diff(arr))
            changes = np.concatenate([[np.nan], changes])
            # Find last significant change
            for i in range(len(changes) - 1, -1, -1):
                if not np.isnan(changes[i]) and changes[i] > th:
                    return float(len(changes) - 1 - i)
            return float(len(changes) - 1)

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_days_since, window_size=30, min_samples=2).alias(c)
            for c in cols
        ]
        return expectation.with_columns(exprs)


# ---------------------------------------------------------------------------
# Revision Delta Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_revision_delta",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_revision_delta",
    source=_SRC,
    backend="polars")
class FinRevisionDeltaNative(SeriesOperator):
    """Revision delta: current - lag."""

    metadata = OperatorMetadata(
        name="fin_revision_delta",
        category="financial",
        description="修正变化",
        param_names=["x", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(x)
        exprs = [(pl.col(c) - pl.col(c).shift(d)).alias(c) for c in cols]
        return x.with_columns(exprs)


@register_operator(
    name="fin_revision_magnitude",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_revision_magnitude",
    source=_SRC,
    backend="polars")
class FinRevisionMagnitudeNative(SeriesOperator):
    """Revision magnitude: |current - lag|."""

    metadata = OperatorMetadata(
        name="fin_revision_magnitude",
        category="financial",
        description="修正幅度",
        param_names=["x", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(x)
        exprs = [(pl.col(c) - pl.col(c).shift(d)).abs().alias(c) for c in cols]
        return x.with_columns(exprs)


@register_operator(
    name="fin_revision_pct",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_revision_pct",
    source=_SRC,
    backend="polars")
class FinRevisionPctNative(SeriesOperator):
    """Revision percentage: (current - lag) / |lag|."""

    metadata = OperatorMetadata(
        name="fin_revision_pct",
        category="financial",
        description="修正百分比",
        param_names=["x", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            prior = pl.col(c).shift(d)
            rev_pct = pl.when((prior.is_not_null()) & (prior.abs() > 1e-12)).then(
                (pl.col(c) - prior) / prior.abs()
            ).otherwise(None)
            exprs.append(rev_pct.alias(c))
        return x.with_columns(exprs)


@register_operator(
    name="fin_revision_direction",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_revision_direction",
    source=_SRC,
    backend="polars")
class FinRevisionDirectionNative(SeriesOperator):
    """Revision direction: sign(current - lag)."""

    metadata = OperatorMetadata(
        name="fin_revision_direction",
        category="financial",
        description="修正方向",
        param_names=["x", "lag"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, **kwargs) -> pl.DataFrame:
        d = strict_integer(lag, "lag", minimum=1)
        cols = _numeric_cols(x)
        exprs = [(pl.col(c) - pl.col(c).shift(d)).sign().alias(c) for c in cols]
        return x.with_columns(exprs)


@register_operator(
    name="fin_revision_count",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_revision_count",
    source=_SRC,
    backend="polars")
class FinRevisionCountNative(SeriesOperator):
    """Count of revisions in window."""

    metadata = OperatorMetadata(
        name="fin_revision_count",
        category="financial",
        description="修正次数",
        param_names=["x", "window", "threshold"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.0, searchable=True, param_role=ParamRole.THRESHOLD),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, threshold: float = 0.0, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        th = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            change = (pl.col(c) - pl.col(c).shift(1)).abs()
            count = (change > th).cast(pl.Float64).rolling_sum(window_size=w, min_samples=1)
            exprs.append(count.alias(c))
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Beat/Miss Streak Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_beat_streak",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_beat_streak",
    source=_SRC,
    backend="polars")
class FinBeatStreakNative(SeriesOperator):
    """Consecutive quarters beating expectations."""

    metadata = OperatorMetadata(
        name="fin_beat_streak",
        category="financial",
        description="连续超预期季度数",
        param_names=["actual", "expected"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(actual)

        def _beat_streak(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0 or np.isnan(arr[-1]):
                return np.nan
            streak = 0
            for i in range(len(arr) - 1, -1, -1):
                if np.isnan(arr[i]) or arr[i] <= 0:
                    break
                streak += 1
            return float(streak)

        exprs = []
        for c in cols:
            if c not in expected.columns:
                continue
            beat = (actual[c] > expected[c]).cast(pl.Float64)
            streak = beat.fill_nan(None).rolling_map(_beat_streak, window_size=20, min_samples=1)
            exprs.append(streak.alias(c))

        return actual.with_columns(exprs) if exprs else actual


@register_operator(
    name="fin_miss_streak",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_miss_streak",
    source=_SRC,
    backend="polars")
class FinMissStreakNative(SeriesOperator):
    """Consecutive quarters missing expectations."""

    metadata = OperatorMetadata(
        name="fin_miss_streak",
        category="financial",
        description="连续低于预期季度数",
        param_names=["actual", "expected"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
    )

    def _calculate_series(self, actual: pl.DataFrame, expected: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(actual)

        def _miss_streak(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0 or np.isnan(arr[-1]):
                return np.nan
            streak = 0
            for i in range(len(arr) - 1, -1, -1):
                if np.isnan(arr[i]) or arr[i] <= 0:
                    break
                streak += 1
            return float(streak)

        exprs = []
        for c in cols:
            if c not in expected.columns:
                continue
            miss = (actual[c] < expected[c]).cast(pl.Float64)
            streak = miss.fill_nan(None).rolling_map(_miss_streak, window_size=20, min_samples=1)
            exprs.append(streak.alias(c))

        return actual.with_columns(exprs) if exprs else actual


# ---------------------------------------------------------------------------
# Persistence Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_earnings_persistence",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_earnings_persistence",
    source=_SRC,
    backend="polars")
class FinEarningsPersistenceNative(SeriesOperator):
    """Earnings persistence: autocorrelation of earnings changes."""

    metadata = OperatorMetadata(
        name="fin_earnings_persistence",
        category="financial",
        description="盈利持续性",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _persistence(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            arr = arr[valid]
            if len(arr) < 3:
                return np.nan
            changes = np.diff(arr)
            if len(changes) < 2:
                return np.nan
            if np.std(changes) < 1e-12:
                return np.nan
            corr = np.corrcoef(changes[:-1], changes[1:])[0, 1]
            return float(corr) if not np.isnan(corr) else np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_persistence, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_cashflow_persistence",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_cashflow_persistence",
    source=_SRC,
    backend="polars")
class FinCashflowPersistenceNative(SeriesOperator):
    """Cashflow persistence: autocorrelation of cashflow changes."""

    metadata = OperatorMetadata(
        name="fin_cashflow_persistence",
        category="financial",
        description="现金流持续性",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _persistence(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            arr = arr[valid]
            if len(arr) < 3:
                return np.nan
            changes = np.diff(arr)
            if len(changes) < 2:
                return np.nan
            if np.std(changes) < 1e-12:
                return np.nan
            corr = np.corrcoef(changes[:-1], changes[1:])[0, 1]
            return float(corr) if not np.isnan(corr) else np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_persistence, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_margin_persistence",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_margin_persistence",
    source=_SRC,
    backend="polars")
class FinMarginPersistenceNative(SeriesOperator):
    """Margin persistence: autocorrelation of margin changes."""

    metadata = OperatorMetadata(
        name="fin_margin_persistence",
        category="financial",
        description="利润率持续性",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _persistence(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            arr = arr[valid]
            if len(arr) < 3:
                return np.nan
            changes = np.diff(arr)
            if len(changes) < 2:
                return np.nan
            if np.std(changes) < 1e-12:
                return np.nan
            corr = np.corrcoef(changes[:-1], changes[1:])[0, 1]
            return float(corr) if not np.isnan(corr) else np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_persistence, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_growth_persistence",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_growth_persistence",
    source=_SRC,
    backend="polars")
class FinGrowthPersistenceNative(SeriesOperator):
    """Growth persistence: autocorrelation of growth rates."""

    metadata = OperatorMetadata(
        name="fin_growth_persistence",
        category="financial",
        description="增长持续性",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=3, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=3)
        cols = _numeric_cols(x)

        def _persistence(vals):
            arr = np.asarray(vals, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 3:
                return np.nan
            arr = arr[valid]
            if len(arr) < 3:
                return np.nan
            # Growth rates
            growth = []
            for i in range(1, len(arr)):
                if arr[i-1] != 0:
                    growth.append((arr[i] - arr[i-1]) / abs(arr[i-1]))
            if len(growth) < 2:
                return np.nan
            growth = np.array(growth)
            if np.std(growth) < 1e-12:
                return np.nan
            corr = np.corrcoef(growth[:-1], growth[1:])[0, 1]
            return float(corr) if not np.isnan(corr) else np.nan

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_persistence, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_earnings_smoothness",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_earnings_smoothness",
    source=_SRC,
    backend="polars")
class FinEarningsSmoothnessNative(SeriesOperator):
    """Earnings smoothness: std(earnings) / mean(|earnings|)."""

    metadata = OperatorMetadata(
        name="fin_earnings_smoothness",
        category="financial",
        description="盈利平滑度",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=8, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 8, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            std = pl.col(c).rolling_std(window_size=w, min_samples=2)
            mean_abs = pl.col(c).abs().rolling_mean(window_size=w, min_samples=2)
            smoothness = pl.when((mean_abs.is_not_null()) & (mean_abs > 1e-12)).then(
                std / mean_abs
            ).otherwise(None)
            exprs.append(smoothness.alias(c))
        return x.with_columns(exprs)


@register_operator(
    name="fin_accrual_ratio",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_accrual_ratio",
    source=_SRC,
    backend="polars")
class FinAccrualRatioNative(SeriesOperator):
    """Accrual ratio: (earnings - cashflow) / |earnings|."""

    metadata = OperatorMetadata(
        name="fin_accrual_ratio",
        category="financial",
        description="应计项目比率",
        param_names=["earnings", "cashflow"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
    )

    def _calculate_series(self, earnings: pl.DataFrame, cashflow: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = _numeric_cols(earnings)
        exprs = []
        for c in cols:
            if c not in cashflow.columns:
                continue
            accrual = earnings[c] - cashflow[c]
            ratio = pl.when((earnings[c].abs() > 1e-12)).then(
                accrual / earnings[c].abs()
            ).otherwise(None)
            exprs.append(ratio.alias(c))
        return earnings.with_columns(exprs) if exprs else earnings


# ---------------------------------------------------------------------------
# Seasonal Analysis Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_seasonal_zscore",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_seasonal_zscore",
    source=_SRC,
    backend="polars")
class FinSeasonalZscoreNative(SeriesOperator):
    """Seasonal z-score: (current - seasonal_mean) / seasonal_std."""

    metadata = OperatorMetadata(
        name="fin_seasonal_zscore",
        category="financial",
        description="季节性 Z 分数",
        param_names=["x", "period", "n_periods"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "period": ParamSpec(dtype=int, min=2, default=4, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "n_periods": ParamSpec(dtype=int, min=2, default=4, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 4, n_periods: int = 4, **kwargs) -> pl.DataFrame:
        p = strict_integer(period, "period", minimum=2)
        n = strict_integer(n_periods, "n_periods", minimum=2)
        cols = _numeric_cols(x)

        def _seasonal_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < p * n:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            # Get same season from previous periods
            seasonal = []
            for i in range(1, n + 1):
                idx = len(arr) - 1 - i * p
                if idx >= 0 and not np.isnan(arr[idx]):
                    seasonal.append(arr[idx])
            if len(seasonal) < 2:
                return np.nan
            mean = np.mean(seasonal)
            std = np.std(seasonal, ddof=1)
            if std < 1e-12:
                return np.nan
            return float(current - mean) / std

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_seasonal_z, window_size=p * n + 1, min_samples=p * 2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_seasonal_percentile",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_seasonal_percentile",
    source=_SRC,
    backend="polars")
class FinSeasonalPercentileNative(SeriesOperator):
    """Seasonal percentile: current vs same season historical values."""

    metadata = OperatorMetadata(
        name="fin_seasonal_percentile",
        category="financial",
        description="季节性百分位",
        param_names=["x", "period", "n_periods"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "period": ParamSpec(dtype=int, min=2, default=4, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "n_periods": ParamSpec(dtype=int, min=2, default=4, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 4, n_periods: int = 4, **kwargs) -> pl.DataFrame:
        p = strict_integer(period, "period", minimum=2)
        n = strict_integer(n_periods, "n_periods", minimum=2)
        cols = _numeric_cols(x)

        def _seasonal_pct(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < p * n:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            seasonal = []
            for i in range(1, n + 1):
                idx = len(arr) - 1 - i * p
                if idx >= 0 and not np.isnan(arr[idx]):
                    seasonal.append(arr[idx])
            if len(seasonal) < 1:
                return np.nan
            rank = np.sum(np.array(seasonal) < current) + 0.5 * np.sum(np.array(seasonal) == current)
            return float(rank) / len(seasonal)

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_seasonal_pct, window_size=p * n + 1, min_samples=p + 1).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Historical Z-score and Percentile Operators
# ---------------------------------------------------------------------------


@register_operator(
    name="fin_zscore_history",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_zscore_history",
    source=_SRC,
    backend="polars")
class FinZscoreHistoryNative(SeriesOperator):
    """Z-score vs full history."""

    metadata = OperatorMetadata(
        name="fin_zscore_history",
        category="financial",
        description="历史 Z 分数",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = []
        for c in cols:
            mean = pl.col(c).rolling_mean(window_size=w, min_samples=2)
            std = pl.col(c).rolling_std(window_size=w, min_samples=2)
            z = pl.when((std.is_not_null()) & (std > 1e-12)).then(
                (pl.col(c) - mean) / std
            ).otherwise(None)
            exprs.append(z.alias(c))
        return x.with_columns(exprs)


@register_operator(
    name="fin_zscore_vs_prior_history",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_zscore_vs_prior_history",
    source=_SRC,
    backend="polars")
class FinZscoreVsPriorHistoryNative(SeriesOperator):
    """Z-score vs prior history (excludes current)."""

    metadata = OperatorMetadata(
        name="fin_zscore_vs_prior_history",
        category="financial",
        description="历史 Z 分数（排除当前）",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        def _prior_z(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 3:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 2:
                return np.nan
            past = past[valid]
            mean = np.mean(past)
            std = np.std(past, ddof=1)
            if std < 1e-12:
                return np.nan
            return float(current - mean) / std

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_prior_z, window_size=w, min_samples=3).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_percentile_history",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_percentile_history",
    source=_SRC,
    backend="polars")
class FinPercentileHistoryNative(SeriesOperator):
    """Percentile rank vs full history."""

    metadata = OperatorMetadata(
        name="fin_percentile_history",
        category="financial",
        description="历史百分位",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        exprs = [
            (
                pl.col(c)
                .fill_nan(None)
                .rolling_rank(window_size=w, method="average", min_samples=2)
                / pl.col(c)
                .fill_nan(None)
                .is_not_null()
                .cast(pl.Float64)
                .rolling_sum(window_size=w, min_samples=2)
            ).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)


@register_operator(
    name="fin_percentile_vs_prior_history",
    category="financial",
    business_category="fundamental_analysis",
    canonical="fin_percentile_vs_prior_history",
    source=_SRC,
    backend="polars")
class FinPercentileVsPriorHistoryNative(SeriesOperator):
    """Percentile rank vs prior history (excludes current)."""

    metadata = OperatorMetadata(
        name="fin_percentile_vs_prior_history",
        category="financial",
        description="历史百分位（排除当前）",
        param_names=["x", "window"],
        return_type="series",
        tags=["financial", "polars", "native", "causal"],
        param_specs={
            "window": ParamSpec(dtype=int, min=2, default=20, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        def _prior_pct(vals):
            arr = np.asarray(vals, dtype=float)
            if len(arr) < 2:
                return np.nan
            current = arr[-1]
            if np.isnan(current):
                return np.nan
            past = arr[:-1]
            valid = ~np.isnan(past)
            if np.sum(valid) < 1:
                return np.nan
            past = past[valid]
            rank = np.sum(past < current) + 0.5 * np.sum(past == current)
            return float(rank) / len(past)

        exprs = [
            pl.col(c).fill_nan(None).rolling_map(_prior_pct, window_size=w, min_samples=2).alias(c)
            for c in cols
        ]
        return x.with_columns(exprs)

