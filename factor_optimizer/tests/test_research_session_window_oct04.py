"""Contract tests for pure real-session research window selection."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_optimizer.research_session_window import select_research_session_window


def test_keeps_last_panel_date_when_two_later_sessions_exist():
    calendar = pd.date_range("2026-01-01", periods=20, freq="D")
    panel_dates = calendar[:17]

    window = select_research_session_window([panel_dates], calendar, 17)

    assert window.decision_dates.equals(panel_dates)
    assert window.execution_dates.equals(calendar[1:18])
    assert window.label_end_dates.equals(calendar[2:19])
    assert window.decision_positions == tuple(range(17))
    assert window.execution_positions == tuple(range(1, 18))
    assert window.label_end_positions == tuple(range(2, 19))


def test_maps_endpoints_across_nonconsecutive_trading_sessions():
    calendar = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    panel_dates = calendar[:2]

    window = select_research_session_window([panel_dates], calendar, 2)

    assert window.decision_dates.equals(calendar[:2])
    assert window.execution_dates.equals(calendar[1:3])
    assert window.label_end_dates.equals(calendar[2:4])


def test_uses_latest_common_panel_dates_only():
    calendar = pd.date_range("2026-02-01", periods=12, freq="D")
    first_panel = calendar[:9]
    second_panel = calendar[[0, 1, 3, 4, 5, 6, 7, 8]]

    window = select_research_session_window([first_panel, second_panel], calendar, 5)

    expected = calendar[[4, 5, 6, 7, 8]]
    assert window.decision_dates.equals(expected)
    assert window.execution_dates.equals(calendar[[5, 6, 7, 8, 9]])
    assert window.label_end_dates.equals(calendar[[6, 7, 8, 9, 10]])


@pytest.mark.parametrize("requested_rows", [0, -1, 1261, True, 1.5])
def test_rejects_invalid_requested_row_bounds(requested_rows):
    calendar = pd.date_range("2026-01-01", periods=5, freq="D")

    with pytest.raises(ValueError, match="requested_rows"):
        select_research_session_window([calendar[:3]], calendar, requested_rows)


@pytest.mark.parametrize(
    "bad_calendar",
    [
        pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-03"]),
        pd.DatetimeIndex(["2026-01-02", "2026-01-01", "2026-01-03"]),
        pd.DatetimeIndex(["2026-01-01", pd.NaT, "2026-01-03"]),
    ],
)
def test_rejects_duplicate_unsorted_or_nat_calendar_dates(bad_calendar):
    with pytest.raises(ValueError, match="calendar"):
        select_research_session_window([bad_calendar[:1]], bad_calendar, 1)


def test_rejects_panel_dates_that_collapse_to_duplicate_session_dates():
    panel_dates = pd.DatetimeIndex(["2026-01-01 09:30", "2026-01-01 15:00"])
    calendar = pd.date_range("2026-01-01", periods=4, freq="D")

    with pytest.raises(ValueError, match="panel"):
        select_research_session_window([panel_dates], calendar, 1)


def test_rejects_when_fewer_than_requested_dates_have_two_endpoints():
    calendar = pd.date_range("2026-01-01", periods=5, freq="D")
    panel_dates = calendar[:4]

    with pytest.raises(ValueError, match="requested 4.*available 3"):
        select_research_session_window([panel_dates], calendar, 4)
