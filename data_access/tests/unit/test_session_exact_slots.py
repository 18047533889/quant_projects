# -*- coding: utf-8 -*-
"""R61-P0 #60: unit tests for the exact-slot validator on MarketSession.

Validates ``MarketSession.expected_slots`` / ``expected_minutes`` and the
per-(TradeDate, Symbol) ``validate_session_bars`` report — missing slots,
duplicate bars, off-session timestamps (12:30 lunch, 11:31/15:01 outside
bar_end slots, 09:30/13:00 markers), unexpected timestamps, suspension
classification and quarantine.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data_access.read.session_calendar import build_ashare_session


@pytest.fixture(scope="module")
def session():
    return build_ashare_session()


def _full_day_minutes(day: str = "2024-01-02") -> list[pd.Timestamp]:
    """Exact 240 one-minute A-share labels for one trading day."""
    d = pd.Timestamp(day).normalize()
    return [
        d + pd.Timedelta(hours=h, minutes=m)
        for h, m in (
            [(9, i) for i in range(31, 60)]
            + [(10, i) for i in range(60)]
            + [(11, i) for i in range(31)]
            + [(13, i) for i in range(1, 60)]
            + [(14, i) for i in range(60)]
            + [(15, 0)]
        )
    ]


# --------------------------------------------------------------------------- #
# expected_slots / expected_minutes
# --------------------------------------------------------------------------- #

def test_expected_slots_exactly_240_bar_end(session) -> None:
    slots = session.expected_slots("2024-01-02")
    assert len(slots) == 240
    # 09:30 auction marker and 13:00 lunch-boundary marker are NOT slots.
    assert slots[0] == pd.Timestamp("2024-01-02 09:31")
    assert slots[-1] == pd.Timestamp("2024-01-02 15:00")
    minutes = {s.hour * 60 + s.minute for s in slots}
    assert 9 * 60 + 31 in minutes
    assert 9 * 60 + 30 not in minutes
    assert 11 * 60 + 30 in minutes
    assert 12 * 60 + 30 not in minutes
    assert 13 * 60 not in minutes
    assert 13 * 60 + 1 in minutes


def test_expected_slots_respects_cutoff_and_bar_freq(session) -> None:
    # cutoff 14:00 -> 09:31-11:30 (120) + 13:01-14:00 (60) = 180.
    assert len(session.expected_slots("2024-01-02", end_cutoff="14:00")) == 180
    # 5-minute grid -> 48 slots.
    five = session.expected_slots("2024-01-02", bar_freq="5min")
    assert len(five) == 48
    assert five[0] == pd.Timestamp("2024-01-02 09:31")
    assert five[-1] == pd.Timestamp("2024-01-02 14:56")


def test_expected_slots_skips_lunch_gap(session) -> None:
    slots = session.expected_slots("2024-01-02", bar_freq="5min")
    minutes = {s.hour * 60 + s.minute for s in slots}
    # No slot in 11:31..13:00.
    assert not any(11 * 60 + 31 <= m < 13 * 60 for m in minutes)


# --------------------------------------------------------------------------- #
# validate_session_bars — per-(TradeDate, Symbol) report
# --------------------------------------------------------------------------- #

def test_clean_full_day_not_quarantined(session) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    rep = session.validate_session_bars(
        bars * 2, ["A"] * 240 + ["B"] * 240, [day] * 480
    )
    assert len(rep) == 2
    for key in [(day, "A"), (day, "B")]:
        r = rep[key]
        assert r["expected_count"] == 240
        assert r["observed_count"] == 240
        assert r["missing"] == []
        assert r["duplicates"] == []
        assert r["off_session"] == []
        assert r["unexpected"] == []
        assert r["quarantine"] is False
        assert r["expected_suspension"] is False


def test_missing_slot_reported_per_symbol(session) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    a = bars  # clean
    b = [t for t in bars if t.minute != 15 or t.hour != 10]  # drop 10:15
    ts = a + b
    syms = ["A"] * len(a) + ["B"] * len(b)
    rep = session.validate_session_bars(ts, syms, [day] * len(ts))
    assert not rep[(day, "A")]["quarantine"]
    r_b = rep[(day, "B")]
    assert r_b["quarantine"] is True
    assert 615 in r_b["missing"]
    assert r_b["missing"] == [615]
    assert r_b["duplicates"] == []


def test_duplicate_bar_reported(session) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    ts = bars + [pd.Timestamp("2024-01-02 10:15")]
    rep = session.validate_session_bars(ts, ["A"] * len(ts), [day] * len(ts))
    r = rep[(day, "A")]
    assert r["quarantine"] is True
    assert r["observed_count"] == 241
    assert any(d["minute"] == 615 and d["count"] == 2 for d in r["duplicates"])
    assert r["missing"] == []


@pytest.mark.parametrize(
    ("bad", "kind"),
    [
        ("12:30", "off_session"),  # lunch recess
        ("11:31", "off_session"),  # after morning close / before afternoon
        ("15:01", "off_session"),  # after close
        ("09:30", "off_session"),  # auction marker, not a bar_end slot
        ("13:00", "off_session"),  # lunch-boundary marker, not a slot
        ("2024-01-02 10:15", "duplicate"),  # in-session -> duplicate
    ],
)
def test_off_session_and_duplicate_kinds(session, bad, kind) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    extra = pd.Timestamp("2024-01-02 " + bad) if " " not in bad else pd.Timestamp(bad)
    ts = bars + [extra]
    rep = session.validate_session_bars(ts, ["A"] * len(ts), [day] * len(ts))
    r = rep[(day, "A")]
    assert r["quarantine"] is True
    if kind == "duplicate":
        assert any(d["minute"] == 615 for d in r["duplicates"])
        assert r["off_session"] == []
    else:
        assert r["off_session"] == [str(extra)]
        assert r["duplicates"] == []
    # The exact expected slots are all present: no missing.
    assert r["missing"] == []


def test_unexpected_timestamp_date_mismatch(session) -> None:
    # A bar whose OWN date disagrees with the (TradeDate, Symbol) group key is
    # reported as unexpected (the row belongs to a different trading day).
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    rogue = pd.Timestamp("2024-01-03 10:00")
    ts = bars + [rogue]
    # Trade-date column for the rogue row still says 2024-01-02 (mislabeled row).
    rep = session.validate_session_bars(
        ts,
        ["A"] * len(ts),
        [day] * len(ts),
    )
    r = rep[(day, "A")]
    assert r["quarantine"] is True
    assert str(rogue) in r["unexpected"], r
    assert r["off_session"] == []
    # The clean 240 slots are all still present (rogue went to unexpected).
    assert r["missing"] == []


def test_suspended_group_not_quarantined(session) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    susp = {("A", day.date()), ("B", day.date())}

    def _fn(d, sym):
        return (str(sym), pd.Timestamp(d).date()) in susp

    # A is partial (only morning) and suspended -> expected_suspension True, NOT quarantine.
    partial = bars[:120]
    ts = partial + bars  # A partial morning only; B full
    syms = ["A"] * len(partial) + ["B"] * len(bars)
    rep = session.validate_session_bars(
        ts, syms, [day] * len(ts), expected_suspension_fn=_fn
    )
    assert rep[(day, "A")]["expected_suspension"] is True
    assert rep[(day, "A")]["quarantine"] is False
    assert rep[(day, "B")]["expected_suspension"] is True
    assert rep[(day, "B")]["quarantine"] is False
    # B is complete even if suspended: quarantine stays False.


def test_non_suspended_partial_is_quarantined(session) -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _full_day_minutes()
    partial = bars[:120]  # morning only, not suspended
    rep = session.validate_session_bars(
        partial, ["A"] * len(partial), [day] * len(partial)
    )
    r = rep[(day, "A")]
    assert r["expected_suspension"] is False
    assert r["quarantine"] is True
    assert len(r["missing"]) == 120


def test_broken_length_inputs_rejected(session) -> None:
    with pytest.raises(Exception, match="长度不一致"):
        session.validate_session_bars(
            [pd.Timestamp("2024-01-02 09:31")],
            ["A", "B"],
            [pd.Timestamp("2024-01-02")],
        )


def test_empty_input_ok(session) -> None:
    assert session.validate_session_bars([], [], []) == {}


def test_assert_session_complete_backward_compatible(session) -> None:
    # Old count-only gate still exists and passes for a full clean day.
    bars = _full_day_minutes()
    session.assert_session_complete(bars)  # no raise
    with pytest.raises(Exception, match="不完整"):
        session.assert_session_complete(bars[:100], require_full=True)
