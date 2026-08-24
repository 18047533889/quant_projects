# -*- coding: utf-8 -*-
"""R24-039..042: the selector APIs are semantically distinct — event/
knowledge-time vs fiscal-period-state vs specific-period-vintage vs revision
stream — and an old-period restatement never moves the latest financial state
backwards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.pit_contract import (
    PITColumns,
    pit_asof_join,
    select_latest_event_by_knowledge_time,
    select_latest_fiscal_period_state,
    select_revision_event_stream,
    select_specific_period_vintage,
    select_visible_row_bundles,
)


def _decisions(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"decision_timestamp": pd.to_datetime(dates), "instrument": ["A"] * len(dates)})


def test_explicit_apis_exist_and_are_distinct() -> None:
    # R24-040: four explicit selectors, each with its own semantic contract.
    assert callable(select_latest_fiscal_period_state)
    assert callable(select_latest_event_by_knowledge_time)
    assert callable(select_specific_period_vintage)
    assert callable(select_revision_event_stream)
    # The period-state selector delegates to the row-bundle implementation.
    assert select_latest_fiscal_period_state.__name__ == "select_latest_fiscal_period_state"
    assert select_latest_event_by_knowledge_time.__name__ == "select_latest_event_by_knowledge_time"


def test_old_period_restatement_does_not_move_latest_period_back() -> None:
    # R24-042 golden: Q3 is visible; today the vendor restates Q1.  The latest
    # financial state must STAY Q3, not regress to the restated Q1.
    events = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-09-30"]),
        "available_at": pd.to_datetime(["2024-05-01", "2024-11-20", "2024-10-30"]),
        "revision_id": [1, 2, 1],
        "value": [100.0, 110.0, 300.0],
    })
    decisions = _decisions("2024-11-25")
    sel = select_latest_fiscal_period_state(decisions, events, columns=PITColumns())
    assert sel.loc[0, "period_end"].tz_localize(None) == pd.Timestamp("2024-09-30")
    assert sel.loc[0, "value"] == 300.0


def test_specific_period_vintage_isolates_one_period() -> None:
    events = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-09-30"]),
        "available_at": pd.to_datetime(["2024-05-01", "2024-11-20", "2024-10-30"]),
        "revision_id": [1, 2, 1],
        "value": [100.0, 110.0, 300.0],
    })
    decisions = _decisions("2024-11-25")
    q1 = select_specific_period_vintage(
        decisions, events, period_end="2024-03-31", columns=PITColumns()
    )
    assert q1.loc[0, "value"] == 110.0  # the LATEST visible revision of Q1
    assert q1.loc[0, "period_end"].tz_localize(None) == pd.Timestamp("2024-03-31")


def test_revision_stream_keeps_all_vintages() -> None:
    events = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-09-30"]),
        "available_at": pd.to_datetime(["2024-05-01", "2024-11-20", "2024-10-30"]),
        "revision_id": [1, 2, 1],
        "value": [100.0, 110.0, 300.0],
    })
    stream = select_revision_event_stream(events, decision_time="2024-12-01", columns=PITColumns())
    assert len(stream) == 3  # every revision retained, nothing collapsed


def test_event_selector_and_period_selector_differ_on_restatement() -> None:
    # R24-039/041: pit_asof_join (latest EVENT by knowledge time) would pick the
    # restated Q1 (available 11-20); the period-state selector must pick Q3.
    events = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-09-30"]),
        "available_at": pd.to_datetime(["2024-05-01", "2024-11-20", "2024-10-30"]),
        "revision_id": [1, 2, 1],
        "value": [100.0, 110.0, 300.0],
    })
    decisions = _decisions("2024-11-25")
    ev = select_latest_event_by_knowledge_time(
        decisions, events, columns=PITColumns(), max_age_days=None,
        available_policy="same_day",  # raw knowledge time drives the event pick
    )
    state = select_latest_fiscal_period_state(decisions, events, columns=PITColumns())
    assert ev.loc[0, "period_end"].tz_localize(None) == pd.Timestamp("2024-03-31")  # latest event = restated Q1
    assert state.loc[0, "period_end"].tz_localize(None) == pd.Timestamp("2024-09-30")  # latest period stays Q3
