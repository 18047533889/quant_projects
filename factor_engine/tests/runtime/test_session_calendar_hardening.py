from __future__ import annotations

import pytest

from cleaned_operators.operator_policy import bars_per_day
from runtime.session_calendar import SessionCalendar


def test_unknown_intraday_frequency_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported bar frequency"):
        bars_per_day("2m", market="CN")


def test_hk_calendar_uses_lunch_aware_session() -> None:
    calendar = SessionCalendar(
        bar_freq="1m", market="HK", timestamp_convention="bar_start"
    )
    assert calendar.segments == (("09:30", "12:00"), ("13:00", "16:00"))
    assert calendar.session_minutes == 330
    assert calendar.session_bars == 330
