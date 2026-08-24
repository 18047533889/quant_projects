# -*- coding: utf-8 -*-
"""Acceptance tests for the §2.9 PubDate availability policy.

A-share disclosures (earnings, top-ten holder filings) carry a date, not a
time-of-day, and land after close.  With ``available_policy="next_trading_day"``
an announcement published on ``PubDate`` is only visible to the first decision
strictly after it — a bar dated ``PubDate`` must not act on the same-day filing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.pit_contract import (
    PITColumns,
    pit_asof_join,
    select_visible_row_bundles,
    validate_fundamental_events,
)


def _decisions(*dates: str, instrument: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(dates, utc=True),
            "instrument": [instrument] * len(dates),
        }
    )


def _events(*rows: tuple[str, str, float], instrument: str = "A") -> pd.DataFrame:
    # rows are (period_end, available_at, value)
    return pd.DataFrame(
        {
            "instrument": [instrument] * len(rows),
            "period_end": pd.to_datetime([r[0] for r in rows], utc=True),
            "available_at": pd.to_datetime([r[1] for r in rows], utc=True),
            "value": [r[2] for r in rows],
        }
    )


def test_same_day_opt_in_keeps_historical_behaviour() -> None:
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        columns=PITColumns(),
        max_age_days=None,
        available_policy="same_day",
    )
    # same_day: a decision dated PubDate sees the event announced that day.
    assert joined["value"].iloc[0] == 100.0
    assert joined["value"].iloc[1] == 100.0


def test_default_policy_is_conservative_next_trading_day() -> None:
    """Round-6 P0-03: a date-level knowledge time must NOT default to same-day
    visibility — an after-close filing on PubDate would otherwise drive that
    day's close (look-ahead).  The default is next_trading_day."""
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        columns=PITColumns(),
        max_age_days=None,
    )
    assert pd.isna(joined["value"].iloc[0])
    assert joined["value"].iloc[1] == 100.0


def test_next_trading_day_hides_same_day_announcement() -> None:
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-02"),
        events,
        columns=PITColumns(),
        max_age_days=None,
        available_policy="next_trading_day",
    )
    # A bar dated PubDate must NOT see an announcement made on PubDate.
    assert pd.isna(joined["value"].iloc[0])
    # The first decision strictly after PubDate sees the value.
    assert joined["value"].iloc[1] == 100.0


def test_next_trading_day_uses_actual_decision_grid() -> None:
    # Sparse (monthly) factors see the value at their next decision bar.
    events = _events(("2024-03-31", "2024-04-30", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30", "2024-05-31", "2024-06-28"),
        events,
        columns=PITColumns(),
        max_age_days=None,
        available_policy="next_trading_day",
    )
    assert pd.isna(joined["value"].iloc[0])
    assert joined["value"].iloc[1] == 100.0
    assert joined["value"].iloc[2] == 100.0


def test_next_trading_day_event_after_last_decision_is_never_visible() -> None:
    events = _events(("2024-03-31", "2024-05-02", 100.0))
    joined = pit_asof_join(
        _decisions("2024-04-30"),
        events,
        columns=PITColumns(),
        max_age_days=None,
        available_policy="next_trading_day",
    )
    assert pd.isna(joined["value"].iloc[0])


def test_next_trading_day_select_visible_row_bundles() -> None:
    events = _events(("2023-12-31", "2024-03-01", 100.0), ("2024-03-31", "2024-04-30", 30.0))
    # Decision dated 2024-03-01: the annual report published that day is not yet
    # usable; the previous report remains visible.
    selected = select_visible_row_bundles(
        _decisions("2024-03-01", "2024-05-02"),
        events,
        selector="latest_visible_period",
        columns=PITColumns(),
        available_policy="next_trading_day",
    )
    # The Q1 report is hidden from a bar dated PubDate (2024-03-01) but the older
    # annual report (available 2024-03-01 -> shifted to next decision) is the
    # first decision strictly after 2024-03-01... which is 2024-05-02 here.  For
    # the 2024-03-01 decision no event is visible yet -> NA period_end.
    assert pd.isna(selected["period_end"].iloc[0])
    assert selected["period_end"].iloc[1].normalize() == pd.Timestamp("2024-03-31", tz="UTC")
    assert selected["value"].iloc[1] == 30.0


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown available policy"):
        pit_asof_join(
            _decisions("2024-04-30"),
            _events(("2024-03-31", "2024-04-30", 100.0)),
            available_policy="eod_exact",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown available policy"):
        select_visible_row_bundles(
            _decisions("2024-04-30"),
            _events(("2024-03-31", "2024-04-30", 100.0)),
            available_policy="eod_exact",  # type: ignore[arg-type]
        )


def test_shifted_events_are_still_pit_validated() -> None:
    # available_at before period_end must fail regardless of policy.
    events = _events(("2024-03-31", "2024-03-01", 100.0))
    with pytest.raises(ValueError, match="available_at precedes period_end"):
        pit_asof_join(
            _decisions("2024-04-30"),
            events,
            columns=PITColumns(),
            max_age_days=None,
            available_policy="next_trading_day",
        )
    with pytest.raises(ValueError, match="available_at precedes period_end"):
        validate_fundamental_events(events, PITColumns())


def test_shift_output_dtype_is_tz_aware_for_merge() -> None:
    # Regression guard: merge_asof requires matching key dtypes; the shifted
    # availability must stay tz-aware UTC so the join never raises MergeError.
    events = _events(
        ("2023-12-31", "2024-03-01", 100.0),
        ("2024-03-31", "2024-04-30", 30.0),
    )
    joined = pit_asof_join(
        _decisions("2024-05-02"),
        events,
        columns=PITColumns(),
        max_age_days=None,
        available_policy="next_trading_day",
    )
    assert joined["value"].iloc[0] == 30.0
    assert joined[PITColumns().available_at].isna().sum() == 0
