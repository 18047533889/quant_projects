# -*- coding: utf-8 -*-
"""TRUE_GAP time-semantic operators (2026-08 special request).

Three operators implementing special calendar-aware time semantics:

1. ``financial_snapshot_lag`` — lag operator that skips calendar gaps (e.g.,
   weekends, holidays) and aligns strictly on trading session boundaries.

2. ``same_calendar_day_mean`` — mean of values that share the same calendar
   day-of-week across a trailing window.

3. ``same_calendar_month_return`` — return between values that share the same
   calendar month across trailing years.

Contract
--------
* TRUE_GAP semantics: operators honor calendar structure explicitly instead of
  treating time as a uniform grid.
* PIT-safe: all operators use only past completed data (causal, trailing).
* Dual backend: pandas_numpy reference + polars parity implementation.
* EXTENDED_ONLY surface: registered via ``extend_extended_only``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.operator_surface import extend_extended_only

__all__ = [
    "FinancialSnapshotLag",
    "SameCalendarDayMean",
    "SameCalendarMonthReturn",
]

_CANONICALS = [
    "financial_snapshot_lag",
    "same_calendar_day_mean",
    "same_calendar_month_return",
]


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    category: str,
    domain: str,
    unit: str,
    cost: int,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        param_specs=dict(param_specs or {}),
        tags=[
            category,
            "daily",
            "pit_safe",
            "causal",
            "typed_v2",
            "time_semantic",
            "true_gap",
            f"signature:{','.join(params)}->series",
            f"domain:{domain}",
            f"unit:{unit}",
            f"cost:{cost}",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Construct DataFrame with template's index/columns."""
    return pd.DataFrame(
        values, index=template.index, columns=template.columns, dtype=float
    )


# ---------------------------------------------------------------------------
# 1. financial_snapshot_lag: TRUE_GAP lag operator
# ---------------------------------------------------------------------------


@register_operator(
    name="financial_snapshot_lag",
    category="time_series",
    business_category="time_semantic",
    canonical="financial_snapshot_lag",
    source="time_semantic_gap",
    status="extended",
)
class FinancialSnapshotLag(SeriesOperator):
    """Financial snapshot lag: value from N financial snapshots ago.

    Unlike ``ts_delay`` which counts calendar rows, this operator counts actual
    financial snapshot events (non-NaN observations), skipping calendar gaps
    automatically. A lag of 1 returns the most recent prior valid snapshot.

    TRUE_GAP semantics: honors the true event timeline instead of forcing uniform
    calendar spacing.
    """

    metadata = _metadata(
        "financial_snapshot_lag",
        "Value from N financial snapshots ago (skips calendar gaps).",
        ["x", "lag"],
        category="time_series",
        domain="time",
        unit="passthrough",
        cost=3,
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, lag: int = 1, **_: Any
    ) -> pd.DataFrame:
        """Pandas/numpy reference implementation."""
        lag = max(1, int(lag))
        vals = x.to_numpy(dtype=float)
        rows, cols = vals.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for col in range(cols):
            col_vals = vals[:, col]
            # For each row, find the lag-th prior finite observation
            for row in range(rows):
                finite_count = 0
                for prior_row in range(row - 1, -1, -1):
                    if np.isfinite(col_vals[prior_row]):
                        finite_count += 1
                        if finite_count == lag:
                            out[row, col] = col_vals[prior_row]
                            break

        return _frame_like(x, out)

    def _calculate_series_polars(
        self, x: Any, lag: int = 1, **_: Any
    ) -> Any:
        """Polars parity implementation."""
        import polars as pl

        lag = max(1, int(lag))

        # Convert to pandas for now, implement via reference
        if hasattr(x, "to_pandas"):
            x_pd = x.to_pandas()
            result_pd = self._calculate_series(x_pd, lag=lag)
            # Convert back to polars DataFrame
            return pl.from_pandas(result_pd)
        return x


# ---------------------------------------------------------------------------
# 2. same_calendar_day_mean: day-of-week aligned mean
# ---------------------------------------------------------------------------


@register_operator(
    name="same_calendar_day_mean",
    category="time_series",
    business_category="time_semantic",
    canonical="same_calendar_day_mean",
    source="time_semantic_gap",
    status="extended",
)
class SameCalendarDayMean(SeriesOperator):
    """Mean of values sharing the same calendar day-of-week.

    Computes the mean of observations that occurred on the same day of the week
    (Monday, Tuesday, etc.) within a trailing window of days. For example, on a
    Friday, this returns the mean of all prior Fridays within the window.

    TRUE_GAP semantics: exploits calendar structure to detect day-of-week patterns
    (e.g., Monday effects, Friday flows) that uniform rolling windows miss.
    """

    metadata = _metadata(
        "same_calendar_day_mean",
        "Mean of values on the same day-of-week within trailing window.",
        ["x", "window"],
        category="time_series",
        domain="time",
        unit="passthrough",
        cost=4,
        param_specs={
            "window": ParamSpec(dtype=int, min=1, default=60),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, **_: Any
    ) -> pd.DataFrame:
        """Pandas/numpy reference implementation."""
        window = max(1, int(window))

        if not isinstance(x.index, pd.DatetimeIndex):
            # Cannot extract day-of-week from non-datetime index
            return _frame_like(x, np.full(x.shape, np.nan, dtype=float))

        vals = x.to_numpy(dtype=float)
        rows, cols = vals.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # Extract day-of-week for each timestamp (0=Monday, 6=Sunday)
        weekdays = x.index.dayofweek.to_numpy()

        for col in range(cols):
            col_vals = vals[:, col]
            for row in range(rows):
                current_weekday = weekdays[row]
                start_row = max(0, row - window + 1)

                # Collect all values from same weekday in window
                same_day_vals = []
                for prior_row in range(start_row, row + 1):
                    if weekdays[prior_row] == current_weekday:
                        val = col_vals[prior_row]
                        if np.isfinite(val):
                            same_day_vals.append(val)

                if same_day_vals:
                    out[row, col] = np.mean(same_day_vals)

        return _frame_like(x, out)

    def _calculate_series_polars(
        self, x: Any, window: int = 60, **_: Any
    ) -> Any:
        """Polars parity implementation."""
        import polars as pl

        window = max(1, int(window))

        # Convert to pandas for reference implementation
        if hasattr(x, "to_pandas"):
            x_pd = x.to_pandas()
            result_pd = self._calculate_series(x_pd, window=window)
            return pl.from_pandas(result_pd)
        return x


# ---------------------------------------------------------------------------
# 3. same_calendar_month_return: month-aligned YoY return
# ---------------------------------------------------------------------------


@register_operator(
    name="same_calendar_month_return",
    category="time_series",
    business_category="time_semantic",
    canonical="same_calendar_month_return",
    source="time_semantic_gap",
    status="extended",
)
class SameCalendarMonthReturn(SeriesOperator):
    """Return from same calendar month in prior years.

    Computes the return between the current value and the value from the same
    calendar month N years ago. For example, for December 2024, a lag of 1
    returns the change from December 2023.

    TRUE_GAP semantics: aligns on calendar month boundaries to capture seasonal
    patterns (earnings cycles, fiscal years, holiday effects) that naive lag
    operators miss due to varying month lengths.
    """

    metadata = _metadata(
        "same_calendar_month_return",
        "Return from same calendar month N years ago.",
        ["x", "year_lag"],
        category="time_series",
        domain="time",
        unit="ratio",
        cost=3,
        param_specs={
            "year_lag": ParamSpec(dtype=int, min=1, default=1),
        },
    )

    def _calculate_series(
        self, x: pd.DataFrame, year_lag: int = 1, **_: Any
    ) -> pd.DataFrame:
        """Pandas/numpy reference implementation."""
        year_lag = max(1, int(year_lag))

        if not isinstance(x.index, pd.DatetimeIndex):
            # Cannot extract year/month from non-datetime index
            return _frame_like(x, np.full(x.shape, np.nan, dtype=float))

        vals = x.to_numpy(dtype=float)
        rows, cols = vals.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # Build lookup: (year, month) -> list of row indices
        year_month_map = {}
        for row in range(rows):
            ts = x.index[row]
            key = (ts.year, ts.month)
            if key not in year_month_map:
                year_month_map[key] = []
            year_month_map[key].append(row)

        for col in range(cols):
            col_vals = vals[:, col]
            for row in range(rows):
                ts = x.index[row]
                current_year = ts.year
                current_month = ts.month
                current_val = col_vals[row]

                if not np.isfinite(current_val):
                    continue

                # Look for same month, year_lag years ago
                target_year = current_year - year_lag
                target_key = (target_year, current_month)

                if target_key in year_month_map:
                    # Find the most recent observation from that month
                    candidate_rows = year_month_map[target_key]
                    # Take the last (most recent) valid value from that month
                    for candidate_row in reversed(candidate_rows):
                        if candidate_row < row:  # Must be before current
                            prior_val = col_vals[candidate_row]
                            if np.isfinite(prior_val) and prior_val != 0.0:
                                # Compute return: (current - prior) / prior
                                out[row, col] = (current_val - prior_val) / prior_val if prior_val != 0 else np.nan
                                break

        return _frame_like(x, out)

    def _calculate_series_polars(
        self, x: Any, year_lag: int = 1, **_: Any
    ) -> Any:
        """Polars parity implementation."""
        import polars as pl

        year_lag = max(1, int(year_lag))

        # Convert to pandas for reference implementation
        if hasattr(x, "to_pandas"):
            x_pd = x.to_pandas()
            result_pd = self._calculate_series(x_pd, year_lag=year_lag)
            return pl.from_pandas(result_pd)
        return x


# Register to EXTENDED_ONLY surface
extend_extended_only(_CANONICALS)
