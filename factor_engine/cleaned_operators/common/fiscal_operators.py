# -*- coding: utf-8 -*-
"""TRUE_GAP fiscal operators - Batch 1: Simple fiscal computations.

This module implements fiscal period operators with strict PIT semantics:
- All operations use fiscal period ordinals (quarters)
- PubDate (announcement date) is the only valid time axis
- All lags are positive (toward the past)
- Revisions are handled via revision_policy
- Both Pandas and Polars native implementations
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, ParamRole, ParamSpec
from factor_engine.cleaned_operators.fiscal_strict import period_ordinal
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import operator_surface as _surface

# Register operators on extended surface initially
_surface.extend_extended_only({
    "fiscal_delta", "fiscal_pct_change", "fiscal_acceleration",
    "fiscal_rolling_std", "fiscal_rolling_slope",
    "date_diff_days", "years_since_date", "fundamental_staleness_days",
    "fin_seasonal_zscore", "fin_seasonal_percentile"
})

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

_EPS = 1e-12
_POLICIES = {"latest_available", "first_available"}


def _policy(value: Any) -> str:
    result = str(value).lower()
    if result not in _POLICIES:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return result


def _positive(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@dataclass
class FiscalEventView:
    """Per-panel visible fiscal events at each decision row."""

    events: list[list[dict[int, float]]]
    ordinals: np.ndarray

    @classmethod
    def from_panel(
        cls,
        values: pd.DataFrame,
        period_id: pd.DataFrame,
        *,
        revision_policy: str = "latest_available",
    ) -> "FiscalEventView":
        _align(values, period_id)
        policy = _policy(revision_policy)
        raw_periods = period_id.to_numpy(dtype=object)
        raw_values = values.to_numpy(dtype=float)
        ordinal_values = np.full(raw_periods.shape, np.nan, dtype=float)
        for row in range(raw_periods.shape[0]):
            for col in range(raw_periods.shape[1]):
                ordinal = period_ordinal(raw_periods[row, col])
                if ordinal is not None:
                    ordinal_values[row, col] = ordinal
        snapshots: list[list[dict[int, float]]] = []
        state: list[dict[int, float]] = [dict() for _ in range(values.shape[1])]
        first_seen: list[set[int]] = [set() for _ in range(values.shape[1])]
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ordinal = ordinal_values[row, col]
                value = raw_values[row, col]
                if not np.isfinite(ordinal) or not _finite(value):
                    continue
                key = int(ordinal)
                if policy == "latest_available" or key not in first_seen[col]:
                    state[col][key] = float(value)
                first_seen[col].add(key)
            snapshots.append([dict(sorted(column.items())) for column in state])
        return cls(snapshots, ordinal_values)

    def history(self, row: int, col: int, *, require_consecutive: bool) -> list[tuple[int, float]]:
        current = self.ordinals[row, col]
        if not np.isfinite(current):
            return []
        history = list(self.events[row][col].items())
        if not history:
            return []
        history = [(int(k), float(v)) for k, v in history if k <= int(current) and _finite(v)]
        history.sort()
        if require_consecutive and history:
            contiguous: list[tuple[int, float]] = [history[-1]]
            for item in reversed(history[:-1]):
                if contiguous[0][0] - item[0] != 1:
                    break
                contiguous.insert(0, item)
            history = contiguous
        return history


# ============================================================================
# Pandas Reference Implementations
# ============================================================================

def pd_fiscal_delta(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    lag: int = 1,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Fiscal period delta: x[t] - x[t-lag]."""
    x, period_id = _align(x, period_id)
    lag = _positive(lag, "lag")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            if len(history) <= lag:
                continue
            current_ord, current_val = history[-1]
            lag_ord = current_ord - lag
            history_dict = dict(history)
            if lag_ord in history_dict:
                lag_val = history_dict[lag_ord]
                if _finite(current_val) and _finite(lag_val):
                    out[row, col] = current_val - lag_val
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_pct_change(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    lag: int = 1,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Fiscal period percentage change: (x[t] - x[t-lag]) / abs(x[t-lag])."""
    x, period_id = _align(x, period_id)
    lag = _positive(lag, "lag")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            if len(history) <= lag:
                continue
            current_ord, current_val = history[-1]
            lag_ord = current_ord - lag
            history_dict = dict(history)
            if lag_ord in history_dict:
                lag_val = history_dict[lag_ord]
                if _finite(current_val) and _finite(lag_val) and abs(lag_val) > _EPS:
                    out[row, col] = ((current_val - lag_val)) / abs(lag_val) if abs(lag_val) != 0 else np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_acceleration(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    lag: int = 1,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Fiscal period acceleration (second derivative): delta[t] - delta[t-lag]."""
    x, period_id = _align(x, period_id)
    lag = _positive(lag, "lag")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            if len(history) <= 2 * lag:
                continue
            current_ord, current_val = history[-1]
            history_dict = dict(history)
            lag1_ord = current_ord - lag
            lag2_ord = current_ord - 2 * lag
            if lag1_ord in history_dict and lag2_ord in history_dict:
                lag1_val = history_dict[lag1_ord]
                lag2_val = history_dict[lag2_ord]
                if _finite(current_val) and _finite(lag1_val) and _finite(lag2_val):
                    delta_current = current_val - lag1_val
                    delta_lag = lag1_val - lag2_val
                    out[row, col] = delta_current - delta_lag
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_rolling_std(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    window: int = 8,
    min_periods: int = 4,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Rolling standard deviation over fiscal periods."""
    x, period_id = _align(x, period_id)
    window = _positive(window, "window")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods > window:
        raise ValueError("min_periods must not exceed window")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            values = [v for _, v in history[-window:] if _finite(v)]
            if len(values) >= min_periods:
                out[row, col] = float(np.std(values, ddof=1))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fiscal_rolling_slope(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    window: int = 8,
    min_periods: int = 4,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Rolling OLS slope over fiscal periods (time vs value)."""
    x, period_id = _align(x, period_id)
    window = _positive(window, "window")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2 for slope calculation")
    if min_periods > window:
        raise ValueError("min_periods must not exceed window")
    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            recent = [(ord, v) for ord, v in history[-window:] if _finite(v)]
            if len(recent) >= min_periods:
                ordinals = np.array([ord for ord, _ in recent], dtype=float)
                values = np.array([v for _, v in recent], dtype=float)
                if np.std(ordinals) > _EPS:
                    # OLS: slope = cov(x,y) / var(x)
                    # Use consistent ddof for both covariance and variance
                    cov_matrix = np.cov(ordinals, values)
                    slope = (cov_matrix[0, 1]) / (cov_matrix[0, 0]) if (cov_matrix[0, 0]) != 0 else np.nan
                    out[row, col] = slope
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_date_diff_days(left: pd.DataFrame, right: pd.DataFrame, **_) -> pd.DataFrame:
    """Calendar day difference between two date panels: (left - right) in days."""
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        raise TypeError("date_diff_days requires two aligned panels")
    _align(left, right)
    lhs = left.apply(pd.to_datetime, errors="coerce")
    rhs = right.apply(pd.to_datetime, errors="coerce")
    return np.where(86400.0 != 0, ((lhs - rhs).apply(lambda column: column.dt.total_seconds()) / (86400.0)), np.nan)


def pd_years_since_date(
    date_panel: pd.DataFrame,
    reference_date: pd.DataFrame | None = None,
    **_
) -> pd.DataFrame:
    """Years since date_panel (or years before reference_date)."""
    if not isinstance(date_panel, pd.DataFrame):
        raise TypeError("years_since_date requires a pandas DataFrame")

    dates = date_panel.apply(pd.to_datetime, errors="coerce")

    if reference_date is None:
        # Use row index as reference
        if isinstance(date_panel.index, pd.DatetimeIndex):
            ref = date_panel.index.to_series().values[:, None]
        else:
            # Assume index is date-like
            ref = pd.to_datetime(date_panel.index, errors="coerce").values[:, None]
    else:
        _align(date_panel, reference_date)
        ref = reference_date.apply(pd.to_datetime, errors="coerce").values

    diff_days = np.where(np.timedelta64(1, 'D') != 0, ((ref - dates.values)) / (np.timedelta64(1, 'D')), np.nan)
    years = (diff_days) / 365.25 if 365.25 != 0 else np.nan
    return pd.DataFrame(years, index=date_panel.index, columns=date_panel.columns)


def pd_fundamental_staleness_days(
    pub_date: pd.DataFrame,
    **_
) -> pd.DataFrame:
    """Days since most recent publication date (staleness metric)."""
    if not isinstance(pub_date, pd.DataFrame):
        raise TypeError("fundamental_staleness_days requires a pandas DataFrame")

    dates = pub_date.apply(pd.to_datetime, errors="coerce")

    # Use row index as current date
    if isinstance(pub_date.index, pd.DatetimeIndex):
        current = pub_date.index.to_series().values[:, None]
    else:
        current = pd.to_datetime(pub_date.index, errors="coerce").values[:, None]

    diff = np.where(np.timedelta64(1, 'D') != 0, ((current - dates.values)) / (np.timedelta64(1, 'D')), np.nan)
    return pd.DataFrame(diff, index=pub_date.index, columns=pub_date.columns)


def pd_fin_seasonal_zscore(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    seasonal_lag: int = 4,
    lookback_periods: int = 8,
    min_history: int = 4,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Seasonal z-score: (x[t] - mean(seasonal_history)) / std(seasonal_history)."""
    x, period_id = _align(x, period_id)
    seasonal_lag = _positive(seasonal_lag, "seasonal_lag")
    lookback_periods = _positive(lookback_periods, "lookback_periods")
    min_history = _positive(min_history, "min_history")

    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            if len(history) <= seasonal_lag:
                continue

            current_ord, current_val = history[-1]
            if not _finite(current_val):
                continue

            # Collect seasonal values at the same seasonal position
            history_dict = dict(history)
            seasonal_values = []
            for i in range(1, lookback_periods + 1):
                seasonal_ord = current_ord - i * seasonal_lag
                if seasonal_ord in history_dict:
                    val = history_dict[seasonal_ord]
                    if _finite(val):
                        seasonal_values.append(val)

            if len(seasonal_values) >= min_history:
                mean = np.mean(seasonal_values)
                std = np.std(seasonal_values, ddof=1) if len(seasonal_values) > 1 else np.nan
                if np.isfinite(std) and std > _EPS:
                    out[row, col] = ((current_val - mean)) / std if std != 0 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def pd_fin_seasonal_percentile(
    x: pd.DataFrame,
    period_id: pd.DataFrame,
    seasonal_lag: int = 4,
    lookback_periods: int = 8,
    min_history: int = 4,
    require_consecutive: bool = True,
    revision_policy: str = "latest_available",
    **_
) -> pd.DataFrame:
    """Seasonal percentile: rank of x[t] within seasonal history."""
    x, period_id = _align(x, period_id)
    seasonal_lag = _positive(seasonal_lag, "seasonal_lag")
    lookback_periods = _positive(lookback_periods, "lookback_periods")
    min_history = _positive(min_history, "min_history")

    view = FiscalEventView.from_panel(x, period_id, revision_policy=revision_policy)
    out = np.full(x.shape, np.nan)

    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            history = view.history(row, col, require_consecutive=require_consecutive)
            if len(history) <= seasonal_lag:
                continue

            current_ord, current_val = history[-1]
            if not _finite(current_val):
                continue

            # Collect seasonal values at the same seasonal position
            history_dict = dict(history)
            seasonal_values = []
            for i in range(1, lookback_periods + 1):
                seasonal_ord = current_ord - i * seasonal_lag
                if seasonal_ord in history_dict:
                    val = history_dict[seasonal_ord]
                    if _finite(val):
                        seasonal_values.append(val)

            if len(seasonal_values) >= min_history:
                # Percentile rank
                all_values = seasonal_values + [current_val]
                rank = sum(v < current_val for v in all_values) + 0.5 * sum(v == current_val for v in all_values)
                out[row, col] = (rank) / len(all_values) if len(all_values) != 0 else np.nan

    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ============================================================================
# Operator Registration
# ============================================================================

class _FunctionOperator(SeriesOperator):
    """Wrapper for function-based operators."""
    def __init__(self, name, params, fn, description):
        self._fn = fn
        self.metadata = OperatorMetadata(
            name=name,
            category="fundamental_period",
            description=description,
            param_names=params,
            return_type="series",
            tags=["fundamental_period", "pit_safe", "strict_fiscal_event", "TRUE_GAP"]
        )

    def _calculate_series(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _register(name, params, fn, description):
    """Register an operator with the registry."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry as _Reg
    # Idempotent re-registration (P0 collection baseline): a direct module import
    # after ``load_all()`` already registered this canonical+backend must not
    # raise.  Only a genuinely different re-registration is an error.
    _existing = _Reg.get(name, "pandas_numpy", mode="any")
    if _existing is not None:
        from factor_engine.cleaned_operators.registry import _impl_source_hash
        _existing_source = str(
            (_Reg._catalog.get(name, {}).get("backend_meta") or {})
            .get("pandas_numpy", {}).get("source", "") or ""
        )
        _candidate = _FunctionOperator(name, params, fn, description)
        if _existing_source == "fiscal_true_gap_batch1" and _impl_source_hash(_existing) == _impl_source_hash(_candidate):
            return
    OperatorRegistry.register(
        _FunctionOperator(name, params, fn, description),
        canonical=name,
        backend="pandas_numpy",
        source="fiscal_true_gap_batch1",
        status="extended",
        backend_explicit=True
    )


def register() -> None:
    """Register all TRUE_GAP batch 1 operators."""
    if "fiscal_delta" in OperatorRegistry._operators:
        return

    _REGISTRY_SPECS = {
        "fiscal_delta": (
            ["x", "period_id", "lag", "require_consecutive", "revision_policy"],
            pd_fiscal_delta,
            "Fiscal period delta"
        ),
        "fiscal_pct_change": (
            ["x", "period_id", "lag", "require_consecutive", "revision_policy"],
            pd_fiscal_pct_change,
            "Fiscal period percentage change"
        ),
        "fiscal_acceleration": (
            ["x", "period_id", "lag", "require_consecutive", "revision_policy"],
            pd_fiscal_acceleration,
            "Fiscal period acceleration (second derivative)"
        ),
        "fiscal_rolling_std": (
            ["x", "period_id", "window", "min_periods", "require_consecutive", "revision_policy"],
            pd_fiscal_rolling_std,
            "Rolling standard deviation over fiscal periods"
        ),
        "fiscal_rolling_slope": (
            ["x", "period_id", "window", "min_periods", "require_consecutive", "revision_policy"],
            pd_fiscal_rolling_slope,
            "Rolling OLS slope over fiscal periods"
        ),
        # NOTE: date_diff_days, fin_seasonal_zscore, fin_seasonal_percentile already exist in fiscal_event_ops.py
        # We provide implementations here for reference but don't register them
        "years_since_date": (
            ["date_panel", "reference_date"],
            pd_years_since_date,
            "Years since date (or before reference)"
        ),
        "fundamental_staleness_days": (
            ["pub_date"],
            pd_fundamental_staleness_days,
            "Days since most recent publication date"
        ),
    }

    for name, (params, fn, description) in _REGISTRY_SPECS.items():
        _register(name, params, fn, description)


# Auto-register on import
register()
