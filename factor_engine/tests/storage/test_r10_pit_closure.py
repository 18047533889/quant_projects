# -*- coding: utf-8 -*-
"""R10 #8/#45/#46 regression tests: production PIT must fail closed.

* #8  — ``next_trading_day`` in production requires a real exchange calendar;
        the decision-grid fallback is rejected (it conflates market_visible_at
        with decision_at).
* #45 — same-instrument/same-available_at candidates need a DETERMINISTIC
        tie-break (newest period_end, then newest revision), never original
        row order.
* #46 — ``same_day`` requires TIMESTAMP-precision availability; date-only
        (midnight) PubDate is rejected in production, warned in research.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pit_contract import (
    PITColumns,
    pit_asof_join,
    select_visible_row_bundles,
)


def _decisions(*dates: str, instrument: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(dates, utc=True),
            "instrument": [instrument] * len(dates),
        }
    )


def _events(*rows: tuple[str, str, float], instrument: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": [instrument] * len(rows),
            "period_end": pd.to_datetime([r[0] for r in rows], utc=True),
            "available_at": pd.to_datetime([r[1] for r in rows], utc=True),
            "value": [r[2] for r in rows],
        }
    )


def _calendar(*dates: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(dates, utc=True))


# --- R10 #8: production next_trading_day requires a market calendar ---------


def test_production_next_trading_day_without_calendar_rejected():
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    with pytest.raises(ValueError, match="requires a real exchange calendar"):
        pit_asof_join(
            _decisions("2024-04-30", "2024-05-02"),
            events,
            available_policy="next_trading_day",
            production=True,
        )
    with pytest.raises(ValueError, match="requires a real exchange calendar"):
        select_visible_row_bundles(
            _decisions("2024-04-30", "2024-05-02"),
            events,
            available_policy="next_trading_day",
            production=True,
        )


def test_production_next_trading_day_with_calendar_ok():
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        available_policy="next_trading_day",
        market_calendar=_calendar("2024-04-30", "2024-05-01", "2024-05-02"),
        market_timezone="UTC",
        production=True,
    )
    assert pd.isna(joined["value"].iloc[0])  # bar dated PubDate does not see it
    assert joined["value"].iloc[1] == 100.0


def test_research_next_trading_day_without_calendar_keeps_fallback():
    # research keeps the legacy decision-grid fallback (documented), no raise.
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        available_policy="next_trading_day",
        production=False,
    )
    assert joined["value"].iloc[1] == 100.0


# --- R10 #46: same_day requires TIMESTAMP precision --------------------------


def test_production_same_day_date_only_rejected():
    events = _events(("2024-03-31", "2024-04-30", 100.0))  # midnight PubDate
    with pytest.raises(ValueError, match="TIMESTAMP-precision"):
        pit_asof_join(
            _decisions("2024-04-30"),
            events,
            available_policy="same_day",
            production=True,
        )


def test_production_same_day_real_timestamp_ok():
    events = _events(("2024-03-31", "2024-04-30 09:30:00", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30 15:00:00"),
        events,
        available_policy="same_day",
        production=True,
    )
    assert joined["value"].iloc[0] == 100.0


def test_research_same_day_date_only_keeps_legacy():
    # legacy research opt-in stays (regression guard for existing suite).
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        available_policy="same_day",
        production=False,
    )
    assert joined["value"].iloc[0] == 100.0


# --- R10 #45: deterministic tie-break on same available_at -------------------


def test_tie_break_prefers_newest_period_end():
    # Two events for instrument A announced on the SAME available_at; the
    # decision on/after that date must deterministically see the NEWER report.
    events = pd.DataFrame(
        {
            "instrument": ["A", "A"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-06-30"], utc=True),
            "available_at": pd.to_datetime(["2024-07-15", "2024-07-15"], utc=True),
            "value": [100.0, 200.0],
        }
    )
    joined = pit_asof_join(
        _decisions("2024-07-20"),
        events,
        max_age_days=None,
        available_policy="same_day",
        production=False,
    )
    assert joined["value"].iloc[0] == 200.0
    assert joined["period_end"].iloc[0].normalize() == pd.Timestamp("2024-06-30", tz="UTC")


def test_tie_break_prefers_newer_revision_within_period():
    events = pd.DataFrame(
        {
            "instrument": ["A", "A"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"], utc=True),
            "available_at": pd.to_datetime(["2024-04-30", "2024-04-30"], utc=True),
            "revision_id": [1, 2],
            "value": [100.0, 300.0],
        }
    )
    joined = pit_asof_join(
        _decisions("2024-05-02"),
        events,
        max_age_days=None,
        available_policy="same_day",
        production=False,
    )
    assert joined["value"].iloc[0] == 300.0
