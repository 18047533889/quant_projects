"""Select decision dates with exact real-session label endpoints."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchSessionWindow:
    """A bounded decision window and its exact +1/+2 session endpoints."""

    decision_dates: pd.DatetimeIndex
    execution_dates: pd.DatetimeIndex
    label_end_dates: pd.DatetimeIndex
    decision_positions: tuple[int, ...]
    execution_positions: tuple[int, ...]
    label_end_positions: tuple[int, ...]


def _session_dates(values, name):
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain valid date values") from exc
    if dates.hasnans:
        raise ValueError(f"{name} must not contain NaT")
    if dates.tz is not None:
        raise ValueError(f"{name} must use timezone-naive session dates")
    dates = dates.normalize()
    if dates.has_duplicates:
        raise ValueError(f"{name} must be unique after session-date normalization")
    if not dates.is_monotonic_increasing:
        raise ValueError(f"{name} must be strictly increasing")
    return dates


def select_research_session_window(panel_indexes, trading_dates, requested_rows):
    """Return the latest exact number of shared panel dates with +1/+2 sessions."""
    if type(requested_rows) is not int or not 1 <= requested_rows <= 1260:
        raise ValueError("requested_rows must be an integer in 1..1260")
    panels = tuple(panel_indexes)
    if not panels:
        raise ValueError("at least one panel index is required")
    calendar = _session_dates(trading_dates, "calendar")
    normalized_panels = tuple(_session_dates(index, "panel index") for index in panels)
    common = normalized_panels[0]
    for panel_dates in normalized_panels[1:]:
        common = common.intersection(panel_dates, sort=False)
    positions = calendar.get_indexer(common)
    eligible = (positions >= 0) & (positions + 2 < len(calendar))
    common = common[eligible]
    positions = positions[eligible]
    if len(common) < requested_rows:
        raise ValueError(
            f"requested {requested_rows} decision dates with two endpoints; available {len(common)}"
        )
    common = common[-requested_rows:]
    positions = positions[-requested_rows:]
    execution_positions = positions + 1
    label_end_positions = positions + 2
    return ResearchSessionWindow(
        decision_dates=common,
        execution_dates=calendar.take(execution_positions),
        label_end_dates=calendar.take(label_end_positions),
        decision_positions=tuple(int(position) for position in positions),
        execution_positions=tuple(int(position) for position in execution_positions),
        label_end_positions=tuple(int(position) for position in label_end_positions),
    )
