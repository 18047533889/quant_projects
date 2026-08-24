# -*- coding: utf-8 -*-
"""Fiscal TRUE_GAP operators: batch 1 (acceleration, pct_change, rolling_std, accrual_quality, direction_consistency).

These operators work on distinct fiscal events using FiscalEventView kernel,
implementing trailing-only causal computations over fiscal ordinals.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.fiscal_event_ops import FiscalEventView, _align, _finite, _positive, _policy

_EPS = 1e-12


def pd_fiscal_acceleration(
    value,
    period_id,
    lag=1,
    periods=8,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal acceleration: second difference of fiscal values.

    Measures the rate of change of the rate of change:
        acceleration_t = (value_t - value_{t-lag}) - (value_{t-lag} - value_{t-2*lag})
                       = value_t - 2*value_{t-lag} + value_{t-2*lag}

    Args:
        value: Fiscal metric panel (e.g., earnings, revenue)
        period_id: Fiscal period identifier (e.g., "2024Q3")
        lag: Number of fiscal periods for differencing (default 1)
        periods: Number of recent fiscal events to consider (default 8)
        min_periods: Minimum fiscal events required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal acceleration values

    Notes:
        - Requires at least 2*lag + 1 fiscal events
        - TRUE_GAP: works on distinct fiscal events, not daily forward-fill
        - Extended surface only
    """
    value, period_id = _align(value, period_id)
    lag = _positive(lag, "lag")
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < 2 * lag + 1:
        raise ValueError(f"min_periods must be at least {2 * lag + 1} for lag={lag}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) < min_periods:
                continue

            by_ordinal = dict(history[-periods:])
            if not history:
                continue

            current_ord, current_val = history[-1]
            lag1_val = by_ordinal.get(current_ord - lag)
            lag2_val = by_ordinal.get(current_ord - 2 * lag)

            if lag1_val is None or lag2_val is None:
                continue
            if not _finite(current_val) or not _finite(lag1_val) or not _finite(lag2_val):
                continue

            # acceleration = value_t - 2*value_{t-lag} + value_{t-2*lag}
            out[row, col] = float(current_val - 2 * lag1_val + lag2_val)

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_pct_change(
    value,
    period_id,
    lag=1,
    periods=8,
    min_periods=2,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal percentage change over specified lag.

    Computes: (value_t - value_{t-lag}) / |value_{t-lag}|

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        lag: Number of fiscal periods for differencing (default 1)
        periods: Number of recent fiscal events to consider (default 8)
        min_periods: Minimum fiscal events required (default 2)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal percentage changes

    Notes:
        - Denominator uses absolute value to handle sign changes
        - Returns NaN when denominator is near zero
        - TRUE_GAP: works on distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    lag = _positive(lag, "lag")
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < lag + 1:
        raise ValueError(f"min_periods must be at least {lag + 1} for lag={lag}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) < min_periods:
                continue

            by_ordinal = dict(history[-periods:])
            if not history:
                continue

            current_ord, current_val = history[-1]
            lag_val = by_ordinal.get(current_ord - lag)

            if lag_val is None:
                continue
            if not _finite(current_val) or not _finite(lag_val):
                continue

            denom = abs(lag_val)
            if denom <= _EPS:
                continue

            out[row, col] = float((current_val - lag_val) / denom)

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_rolling_std(
    value,
    period_id,
    periods=8,
    min_periods=3,
    ddof=1,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal rolling standard deviation over recent fiscal events.

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 3)
        ddof: Delta degrees of freedom (default 1)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of fiscal rolling standard deviations

    Notes:
        - Uses sample standard deviation (ddof=1) by default
        - TRUE_GAP: computed over distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if min_periods <= ddof:
        raise ValueError(f"min_periods must be greater than ddof={ddof}")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            values = [v for _, v in window if _finite(v)]
            if len(values) < min_periods:
                continue

            if len(values) <= ddof:
                continue

            std_val = float(np.std(values, ddof=ddof))
            if _finite(std_val):
                out[row, col] = std_val

    return pd.DataFrame(out, index=value.index, columns=value.columns)


def pd_fiscal_accrual_quality(
    accruals,
    period_id,
    periods=8,
    min_periods=4,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal accrual quality: negative standard deviation of accruals.

    Measures accrual volatility as a proxy for earnings quality. Lower volatility
    (higher score, closer to 0) indicates more stable accruals.

    Returns: -std(accruals) over recent fiscal events

    Args:
        accruals: Accruals panel (e.g., net income - operating cash flow)
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 4)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of accrual quality scores (negative std; closer to 0 is better)

    Notes:
        - Based on Dechow-Dichev (2002) accrual quality framework
        - Inputs should be preprocessed (typically scaled by assets)
        - TRUE_GAP: computed over distinct fiscal events
        - Extended surface only
    """
    accruals, period_id = _align(accruals, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(accruals, period_id, revision_policy=policy)
    out = np.full(accruals.shape, np.nan, dtype=float)

    for row in range(accruals.shape[0]):
        for col in range(accruals.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            values = [v for _, v in window if _finite(v)]
            if len(values) < min_periods:
                continue

            if len(values) <= 1:
                continue

            std_val = float(np.std(values, ddof=1))
            if _finite(std_val):
                out[row, col] = -std_val  # Negative: lower volatility is better

    return pd.DataFrame(out, index=accruals.index, columns=accruals.columns)


def pd_fiscal_direction_consistency(
    value,
    period_id,
    periods=8,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Fiscal direction consistency: fraction of fiscal changes in same direction.

    Measures the consistency of fiscal metric growth/decline direction over time.
    Returns the fraction of consecutive changes that agree with the most recent change.

    Args:
        value: Fiscal metric panel
        period_id: Fiscal period identifier
        periods: Number of recent fiscal events to include (default 8)
        min_periods: Minimum fiscal events required (default 3)
        require_consecutive: Whether to require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of direction consistency ratios [0, 1]

    Notes:
        - 1.0 indicates all changes in same direction as current
        - 0.0 indicates all changes in opposite direction
        - Only considers non-zero changes
        - TRUE_GAP: computed over distinct fiscal events
    """
    value, period_id = _align(value, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")

    policy = _policy(revision_policy)
    view = FiscalEventView.from_panel(value, period_id, revision_policy=policy)
    out = np.full(value.shape, np.nan, dtype=float)

    for row in range(value.shape[0]):
        for col in range(value.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            window = history[-periods:]

            if len(window) < min_periods:
                continue

            # Compute consecutive changes
            changes = []
            for i in range(1, len(window)):
                prev_val = window[i - 1][1]
                curr_val = window[i][1]
                if _finite(prev_val) and _finite(curr_val):
                    change = curr_val - prev_val
                    if abs(change) > _EPS:  # Non-zero change
                        changes.append(change)

            if len(changes) < min_periods - 1:
                continue

            # Get current direction and compute agreement
            current_direction = np.sign(changes[-1])
            agreement = sum(np.sign(c) == current_direction for c in changes)
            out[row, col] = float(agreement / len(changes))

    return pd.DataFrame(out, index=value.index, columns=value.columns)


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False


if HAS_POLARS:
    def pl_fiscal_acceleration(
        value,
        period_id,
        lag=1,
        periods=8,
        min_periods=3,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_acceleration."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        # Convert to pandas, compute, convert back
        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_acceleration(
            value_pd, period_id_pd, lag=lag, periods=periods,
            min_periods=min_periods, require_consecutive=require_consecutive,
            revision_policy=revision_policy
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_pct_change(
        value,
        period_id,
        lag=1,
        periods=8,
        min_periods=2,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_pct_change."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_pct_change(
            value_pd, period_id_pd, lag=lag, periods=periods,
            min_periods=min_periods, require_consecutive=require_consecutive,
            revision_policy=revision_policy
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_rolling_std(
        value,
        period_id,
        periods=8,
        min_periods=3,
        ddof=1,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_rolling_std."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_rolling_std(
            value_pd, period_id_pd, periods=periods, min_periods=min_periods,
            ddof=ddof, require_consecutive=require_consecutive,
            revision_policy=revision_policy
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_accrual_quality(
        accruals,
        period_id,
        periods=8,
        min_periods=4,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_accrual_quality."""
        if not isinstance(accruals, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        accruals_pd = accruals.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_accrual_quality(
            accruals_pd, period_id_pd, periods=periods, min_periods=min_periods,
            require_consecutive=require_consecutive, revision_policy=revision_policy
        )

        return pl.from_pandas(result_pd)

    def pl_fiscal_direction_consistency(
        value,
        period_id,
        periods=8,
        min_periods=3,
        require_consecutive=True,
        revision_policy="latest_available",
        **_,
    ) -> "pl.DataFrame":
        """Polars backend for fiscal_direction_consistency."""
        if not isinstance(value, pl.DataFrame) or not isinstance(period_id, pl.DataFrame):
            raise TypeError("Polars backend requires Polars DataFrames")

        value_pd = value.to_pandas()
        period_id_pd = period_id.to_pandas()

        result_pd = pd_fiscal_direction_consistency(
            value_pd, period_id_pd, periods=periods, min_periods=min_periods,
            require_consecutive=require_consecutive, revision_policy=revision_policy
        )

        return pl.from_pandas(result_pd)


# Operator class wrappers
class _FiscalAcceleration(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_acceleration",
        category="fundamental_period",
        description="Second difference of fiscal values: value_t - 2*value_{t-lag} + value_{t-2*lag}. "
        "Measures acceleration of fiscal metric changes over distinct fiscal events.",
        param_names=[
            "value",
            "period_id",
            "lag",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_acceleration(*args, **kwargs)


class _FiscalPctChange(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_pct_change",
        category="fundamental_period",
        description="Fiscal percentage change over specified lag: (value_t - value_{t-lag}) / |value_{t-lag}|. "
        "Works on distinct fiscal events.",
        param_names=[
            "value",
            "period_id",
            "lag",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_pct_change(*args, **kwargs)


class _FiscalRollingStd(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_rolling_std",
        category="fundamental_period",
        description="Rolling standard deviation over recent fiscal events. "
        "Measures fiscal metric volatility over distinct fiscal periods.",
        param_names=[
            "value",
            "period_id",
            "periods",
            "min_periods",
            "ddof",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_rolling_std(*args, **kwargs)


class _FiscalAccrualQuality(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_accrual_quality",
        category="fundamental_period",
        description="Negative standard deviation of accruals over recent fiscal events. "
        "Higher (closer to 0) indicates more stable accruals. Based on Dechow-Dichev 2002.",
        param_names=[
            "accruals",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
            "research",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_accrual_quality(*args, **kwargs)


class _FiscalDirectionConsistency(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_direction_consistency",
        category="fundamental_period",
        description="Fraction of fiscal changes in same direction as most recent change. "
        "Returns ratio [0, 1] measuring growth/decline consistency over fiscal events.",
        param_names=[
            "value",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "strict_fiscal_event",
            "true_gap",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_direction_consistency(*args, **kwargs)


def register() -> None:
    """Register fiscal batch1 operators."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    operators = [
        ("fiscal_acceleration", _FiscalAcceleration),
        ("fiscal_pct_change", _FiscalPctChange),
        ("fiscal_rolling_std", _FiscalRollingStd),
        ("fiscal_accrual_quality", _FiscalAccrualQuality),
        ("fiscal_direction_consistency", _FiscalDirectionConsistency),
    ]

    for name, cls in operators:
        if name in OperatorRegistry._operators:
            continue

        # Register pandas_numpy backend
        register_operator(
            name=name,
            category="fundamental_period",
            business_category="fundamental",
            canonical=name,
            source="fundamental.fiscal_batch1",
            backend="pandas_numpy",
            status="production_ready",
        )(cls)

        # Register polars backend if available
        if HAS_POLARS:
            polars_fn = globals().get(f"pl_{name}")
            if polars_fn:
                register_operator(
                    name=name,
                    category="fundamental_period",
                    business_category="fundamental",
                    canonical=name,
                    source="fundamental.fiscal_batch1",
                    backend="polars",
                    status="production_ready",
                )(type(f"_{name}_polars", (SeriesOperator,), {
                    "metadata": cls.metadata,
                    "_calculate_series": lambda self, *args, fn=polars_fn, **kwargs: fn(*args, **kwargs)
                }))

    # Add to extended surface
    import factor_engine.cleaned_operators.operator_surface as _surface
    _surface.extend_extended_only({
        "fiscal_acceleration",
        "fiscal_pct_change",
        "fiscal_rolling_std",
        "fiscal_accrual_quality",
        "fiscal_direction_consistency",
    })


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "fiscal_acceleration": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_pct_change": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 2,
    },
    "fiscal_rolling_std": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "fiscal_accrual_quality": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 4,
    },
    "fiscal_direction_consistency": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
}


register()
