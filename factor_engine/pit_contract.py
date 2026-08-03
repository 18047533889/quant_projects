# -*- coding: utf-8 -*-
"""Point-in-time contracts for fundamental event data before factor calculation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

VisiblePeriodSelector = Literal["latest_visible_period", "annual_only", "quarterly_only"]


@dataclass(frozen=True)
class PITColumns:
    instrument: str = "instrument"
    period_end: str = "period_end"
    available_at: str = "available_at"
    revision_id: str = "revision_id"


def validate_fundamental_events(events: pd.DataFrame, columns: PITColumns = PITColumns()) -> None:
    required = {columns.instrument, columns.period_end, columns.available_at}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"fundamental events missing PIT columns: {missing}")
    available = pd.to_datetime(events[columns.available_at], errors="coerce", utc=True)
    period_end = pd.to_datetime(events[columns.period_end], errors="coerce", utc=True)
    if available.isna().any():
        raise ValueError("available_at contains null or invalid timestamps")
    if period_end.isna().any():
        raise ValueError("period_end contains null or invalid timestamps")
    # A report cannot become available before its accounting period ends.
    if (available < period_end).any():
        raise ValueError("available_at precedes period_end; PIT leakage risk")
    keys = [columns.instrument, columns.period_end, columns.available_at]
    if columns.revision_id in events.columns:
        keys.append(columns.revision_id)
    if events.duplicated(keys).any():
        raise ValueError(f"duplicate fundamental event vintages on keys={keys}")


def pit_asof_join(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
    max_age_days: int | None = 180,
) -> pd.DataFrame:
    """Backward as-of join enforcing ``available_at <= decision_timestamp``.

    Historical revision vintages must be retained in ``events``.  This function
    never deduplicates a report period to its final revised value before joining.
    """
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns:
        raise ValueError(f"decisions missing {decision_time!r}")
    if columns.instrument not in decisions.columns:
        raise ValueError(f"decisions missing {columns.instrument!r}")

    left = decisions.copy()
    right = events.copy()
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    right[columns.available_at] = pd.to_datetime(right[columns.available_at], errors="raise", utc=True)
    left = left.sort_values([columns.instrument, decision_time])
    right = right.sort_values([columns.instrument, columns.available_at])

    joined = pd.merge_asof(
        left,
        right,
        left_on=decision_time,
        right_on=columns.available_at,
        by=columns.instrument,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_fundamental"),
    )
    if max_age_days is not None:
        if int(max_age_days) < 0:
            raise ValueError("max_age_days must be non-negative or None")
        age = joined[decision_time] - joined[columns.available_at]
        stale = age > pd.Timedelta(days=int(max_age_days))
        event_columns = [c for c in events.columns if c != columns.instrument]
        joined.loc[stale, event_columns] = pd.NA
    return joined


def _period_mask(period_end: pd.Series, selector: VisiblePeriodSelector) -> pd.Series:
    dates = pd.to_datetime(period_end, errors="coerce")
    if selector == "latest_visible_period":
        return pd.Series(True, index=period_end.index)
    if selector == "annual_only":
        return (dates.dt.month == 12) & (dates.dt.day == 31)
    if selector == "quarterly_only":
        quarter_ends = dates.dt.is_quarter_end
        annual = (dates.dt.month == 12) & (dates.dt.day == 31)
        return quarter_ends & ~annual
    raise ValueError(f"unknown visible period selector {selector!r}")


def select_visible_row_bundles(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    selector: VisiblePeriodSelector = "latest_visible_period",
    decision_time: str = "decision_timestamp",
    columns: PITColumns = PITColumns(),
) -> pd.DataFrame:
    """Select one whole visible report row for every decision.

    Rows are selected by availability first, then by the latest eligible report
    period and revision.  All value columns therefore come from the same row.
    """
    validate_fundamental_events(events, columns)
    if decision_time not in decisions.columns or columns.instrument not in decisions.columns:
        raise ValueError("decisions missing PIT decision columns")
    filtered = events.loc[_period_mask(events[columns.period_end], selector)].copy()
    left = decisions.copy()
    left["__pit_row_order__"] = range(len(left))
    left[decision_time] = pd.to_datetime(left[decision_time], errors="raise", utc=True)
    filtered[columns.available_at] = pd.to_datetime(
        filtered[columns.available_at], errors="raise", utc=True
    )
    filtered[columns.period_end] = pd.to_datetime(
        filtered[columns.period_end], errors="raise", utc=True
    )

    selected_rows: list[dict[str, object]] = []
    event_columns = [c for c in filtered.columns if c != columns.instrument]
    for decision in left.to_dict("records"):
        visible = filtered.loc[
            (filtered[columns.instrument] == decision[columns.instrument])
            & (filtered[columns.available_at] <= decision[decision_time])
        ]
        if visible.empty:
            selected_rows.append({**decision, **{column: pd.NA for column in event_columns}})
            continue
        latest_period = visible[columns.period_end].max()
        candidates = visible.loc[visible[columns.period_end] == latest_period]
        sort_columns = [columns.available_at]
        if columns.revision_id in candidates.columns:
            sort_columns.append(columns.revision_id)
        chosen = candidates.sort_values(sort_columns, kind="stable").iloc[-1].to_dict()
        chosen.pop(columns.instrument, None)
        selected_rows.append({**decision, **chosen})
    return (
        pd.DataFrame(selected_rows)
        .sort_values("__pit_row_order__")
        .drop(columns="__pit_row_order__")
        .reset_index(drop=True)
    )
