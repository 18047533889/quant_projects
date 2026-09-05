# -*- coding: utf-8 -*-
"""R61-P0 #60: per-(TradeDate, Symbol) exact-slot session gate for the minute panel.

Regression guard for the old symbol-agnostic daily-count bug in
``assert_session_complete`` (5000 symbols × 240 ≫ 240 always passed; single-stock
missing/duplicate/off-session bars invisible).  The gate now lives in
``_grouped_bars`` and validates every (TradeDate, Symbol) group against the exact
240 one-minute slots (09:31–11:30 / 13:01–15:00, bar_end), reports missing /
duplicates / off_session / unexpected SEPARATELY, turns EXPECTED_SUSPENSION
groups into NaN (no abort) and quarantines non-suspended broken groups
(fail-closed when ``session_require_full=True``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_access.read.session_calendar import build_ashare_session

from factor_engine.storage.sources import intraday_feature_runtime_v2 as _v2_mod

_original_default_suspension_fn = _v2_mod._default_suspension_fn
_original_build_ashare_session = build_ashare_session


def _session_minutes(day: str = "2024-01-02") -> list[pd.Timestamp]:
    """Exact 240 one-minute A-share session labels for one day (09:31-11:30 + 13:01-15:00)."""
    d = pd.Timestamp(day).normalize()
    out: list[pd.Timestamp] = []
    for h, m in (
        [(9, i) for i in range(31, 60)]
        + [(10, i) for i in range(60)]
        + [(11, i) for i in range(31)]
        + [(13, i) for i in range(1, 60)]
        + [(14, i) for i in range(60)]
        + [(15, 0)]
    ):
        out.append(d + pd.Timedelta(hours=h, minutes=m))
    return out


class _Inner:
    start_date = "2024-01-02"
    end_date = "2024-01-02"
    instrument_filter = None
    data_snapshot_id = "snap-sg"
    params = {}
    dataset = "ashare_stock_minute"
    run_mode = None
    production = False
    strict_unknown_fields = False
    enforce_mining_gate = False
    snapshot_now_only = False
    mining_coverage_threshold = None
    pit_enforce = False


class _Source:
    """Minimal LQTPLogicalDataSource-like source whose minute wide-frame is
    CONTROLLED by the test (avoids any real DataAccess read in this unit file)."""

    inner = _Inner()

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        # Written by the runtime's _store_session_gate_report.
        self._feature_runtime_session_gate_report = None

    @property
    def session_gate_report(self):
        return self._feature_runtime_session_gate_report

    def _anchor_index(self) -> pd.MultiIndex:
        idx = pd.MultiIndex.from_tuples(
            sorted(
                {
                    (d, i)
                    for d, i in zip(self._frame["date"], self._frame["instrument"])
                }
            ),
            names=["timestamp", "instrument"],
        )
        return idx

    def _record_dependency(self, *args, **kwargs):
        self._dep = (args, kwargs)

    @staticmethod
    def _align_exact_by_instrument(anchor, series):
        s = series.copy()
        s.index = s.index.set_names(["timestamp", "instrument"])
        idx = anchor.set_names(["timestamp", "instrument"])
        out = s.reindex(idx)
        out.index = anchor
        return out

    def _hhmm(self, *a, **k):
        return None


def _frame_for(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows = (symbol, HH:MM) -> wide minute frame with OHLCV."""
    recs = []
    for sym, hhmm in rows:
        ts = pd.Timestamp("2024-01-02 " + hhmm)
        recs.append(
            {
                "timestamp": ts,
                "date": pd.Timestamp("2024-01-02"),
                "instrument": sym,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1.0,
                "amount": 10.0,
            }
        )
    return pd.DataFrame(recs)


def _invoke_grouped_bars(source, **kwargs):
    """Run the real _grouped_bars over the controlled source (monkeypatched frame cache)."""
    from factor_engine.storage.sources import intraday_feature_runtime_v2 as v2

    import factor_engine.storage.sources.intraday_feature_extension as base

    # The controlled source has no real DataAccess read: _wide_frame serves the
    # controlled frame, and _hhmm_filter is bypassed so the test frame reaches the
    # gate as-is (the unit test controls which timestamps are present).
    frame = source._frame.copy()
    frame = frame.sort_values(["timestamp", "instrument"]).reset_index(drop=True)

    orig_wide = base._wide_frame
    base._wide_frame = lambda _src: frame
    orig_hhmm = base._hhmm_filter
    base._hhmm_filter = lambda f, _o, _c: f
    orig_child = v2.base._child
    v2.base._child = lambda _src, _ds, _hd: _Source(frame)
    orig_susp = v2._default_suspension_fn
    v2._default_suspension_fn = lambda _src, _ds: None  # no real suspension read
    # The runtime only gates the real "ashare" market session; provide one via
    # the DA singleton factory (same static segments) so the per-symbol gate runs.
    import data_access.read.session_calendar as da_sc

    orig_build = da_sc.build_ashare_session
    da_sc.build_ashare_session = build_ashare_session
    # Clear the grouped cache so the gate runs (do not reuse a prior entry).
    bars_cache = v2._cache(source, "bars")
    for k in list(bars_cache):
        bars_cache.pop(k, None)
    try:
        return v2._grouped_bars(
            source,
            "ashare_stock_minute",
            "09:30",
            "15:00",
            "15:00",
            5,
            0.8,
            0,
            "bar_end",
            2,
            **kwargs,
        )
    finally:
        base._wide_frame = orig_wide
        base._hhmm_filter = orig_hhmm
        v2.base._child = orig_child
        v2._default_suspension_fn = orig_susp
        da_sc.build_ashare_session = orig_build


def _clean_sym_rows(sym: str, minutes: list[pd.Timestamp]) -> list[tuple[str, str]]:
    return [(sym, t.strftime("%H:%M")) for t in minutes]


# --------------------------------------------------------------------------- #
# 1. The old bug regression: 5000 symbols × 240 exact bars pass cleanly.
# --------------------------------------------------------------------------- #

def test_many_symbols_full_session_passes_cleanly() -> None:
    minutes = _session_minutes()
    rows: list[tuple[str, str]] = []
    for i in range(60):  # 60 symbols × 240 = full clean panel
        rows += _clean_sym_rows(f"S{i:04d}", minutes)
    frame = _frame_for(rows)
    source = _Source(frame)
    _, grouped, expected = _invoke_grouped_bars(source)
    assert len(grouped) == 60
    assert expected == 48  # 5-min bars from 240 one-minute slots
    report = source.session_gate_report
    assert report == {} or report is None or all(
        not rep.get("quarantine") for rep in report.values()
    ), report


# --------------------------------------------------------------------------- #
# 2. A single symbol missing ONE bar is detected for exactly that symbol.
# --------------------------------------------------------------------------- #

def test_single_symbol_missing_bar_detected_for_that_symbol_only() -> None:
    minutes = _session_minutes()
    # S0 clean; S1 missing 10:15 (minute 615).
    s0 = _clean_sym_rows("S0", minutes)
    s1_min = [t for t in minutes if t.strftime("%H:%M") != "10:15"]
    s1 = _clean_sym_rows("S1", s1_min)
    frame = _frame_for(s0 + s1)
    source = _Source(frame)

    from data_access.core.exceptions import ValidationError as DAValidationError

    # Quarantine raises under require_full (default) -> ValidationError.
    with pytest.raises(Exception, match="quarantine"):
        _invoke_grouped_bars(source)
    with pytest.raises(Exception, match="quarantine"):
        _invoke_grouped_bars(source, session_require_full=True)
    # Research mode (require_full=False): no raise; report carries per-symbol facts.
    _, grouped_research, _ = _invoke_grouped_bars(source, session_require_full=False)
    report = source.session_gate_report
    assert report is not None
    key_s1 = (pd.Timestamp("2024-01-02"), "S1")
    assert report[key_s1]["quarantine"] is True
    assert len(report[key_s1]["missing"]) == 1
    assert 615 in report[key_s1]["missing"]
    key_s0 = (pd.Timestamp("2024-01-02"), "S0")
    assert not report[key_s0]["quarantine"]
    assert report[key_s0]["missing"] == []


# --------------------------------------------------------------------------- #
# 3. Duplicate timestamp in one symbol's day is detected.
# --------------------------------------------------------------------------- #

def test_duplicate_timestamp_detected() -> None:
    minutes = _session_minutes()
    s0 = _clean_sym_rows("S0", minutes)
    s0_dup = s0 + [("S0", "10:15")]  # duplicate the 10:15 bar
    frame = _frame_for(s0_dup)
    source = _Source(frame)
    with pytest.raises(Exception, match="quarantine"):
        _invoke_grouped_bars(source)
    _, _, _ = _invoke_grouped_bars(source, session_require_full=False)
    report = source.session_gate_report
    rep = report[(pd.Timestamp("2024-01-02"), "S0")]
    assert rep["quarantine"] is True
    assert any(d["minute"] == 615 and d["count"] == 2 for d in rep["duplicates"])


# --------------------------------------------------------------------------- #
# 4. Off-session bars (12:30 lunch, 11:31-afternoon-open overlap, 15:01) detected.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "off_hhmm",
    [
        "12:30",  # lunch recess -> off_session
        "11:31",  # after morning close (11:30), before afternoon open -> off_session
        "15:01",  # after close -> off_session
        "09:30",  # auction marker (09:30 is NOT a session slot under bar_end) -> off_session
        "13:00",  # lunch/afternoon-open marker (13:00 NOT a session slot) -> off_session
        "10:15",  # real in-session slot duplicated on top of full session -> duplicate
    ],
)
def test_off_session_bars_detected(off_hhmm: str) -> None:
    minutes = _session_minutes()
    # Full clean session for S0 PLUS one extra bar at off_hhmm.
    s0 = _clean_sym_rows("S0", minutes)
    frame = _frame_for(s0 + [("S0", off_hhmm)])
    source = _Source(frame)
    with pytest.raises(Exception, match="quarantine"):
        _invoke_grouped_bars(source)
    _, _, _ = _invoke_grouped_bars(source, session_require_full=False)
    report = source.session_gate_report
    rep = report[(pd.Timestamp("2024-01-02"), "S0")]
    assert rep["quarantine"] is True
    if off_hhmm == "10:15":
        # 10:15 is inside the expected slot grid -> shows up as duplicate.
        assert any(d["minute"] == 615 for d in rep["duplicates"]), rep
    else:
        assert any(off_hhmm in str(x) for x in rep["off_session"]), rep


# --------------------------------------------------------------------------- #
# 5. Suspended (TradeDate, Symbol) -> NaN path, no abort, expected_suspension.
# --------------------------------------------------------------------------- #

def test_suspended_symbol_goes_nan_no_abort() -> None:
    from data_access.read.session_calendar import ValidationError as SCValidationError

    minutes = _session_minutes()
    s0 = _clean_sym_rows("S0", minutes)
    # S1 is "suspended": supply complete 240 bars but flag it on the suspension
    # calendar; must NOT quarantine/abort; its bars should be dropped -> NaN output.
    s1 = _clean_sym_rows("S1", minutes)
    frame = _frame_for(s0 + s1)
    source = _Source(frame)
    susp = {("S1", pd.Timestamp("2024-01-02").date())}

    def _susp_fn(day, sym):
        return (str(sym), pd.Timestamp(day).date()) in susp

    _, grouped, _ = _invoke_grouped_bars(
        source, expected_suspension_fn=_susp_fn, session_require_full=True
    )
    report = source.session_gate_report
    key_s1 = (pd.Timestamp("2024-01-02"), "S1")
    assert report[key_s1]["expected_suspension"] is True
    assert report[key_s1]["quarantine"] is False
    # S1 dropped from grouped -> NaN downstream.
    assert all(str(inst) != "S1" for _, inst, _ in grouped)
    assert any(str(inst) == "S0" for _, inst, _ in grouped)


# --------------------------------------------------------------------------- #
# 6. IsSuspend==False && partial -> quarantine report + fail-closed raise.
# --------------------------------------------------------------------------- #

def test_partial_non_suspended_fails_closed_under_require_full() -> None:
    from data_access.read.session_calendar import ValidationError as SCValidationError

    minutes = _session_minutes()
    s0 = _clean_sym_rows("S0", minutes)
    # S1 partial (missing many bars) + NOT on the suspension calendar.
    s1 = _clean_sym_rows("S1", minutes[:120])  # only morning
    frame = _frame_for(s0 + s1)
    source = _Source(frame)

    def _susp_fn(day, sym):
        return False  # nothing suspended

    with pytest.raises(Exception, match="quarantine"):
        _invoke_grouped_bars(source, expected_suspension_fn=_susp_fn)
    report = source.session_gate_report
    key_s1 = (pd.Timestamp("2024-01-02"), "S1")
    assert report[key_s1]["quarantine"] is True
    assert report[key_s1]["expected_suspension"] is False
    assert len(report[key_s1]["missing"]) == 120
