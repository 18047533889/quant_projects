# -*- coding: utf-8 -*-
"""R10 #5 regression tests: PanelIdentity must time-verify a MultiIndex
``(timestamp, instrument)`` long panel — two panels with different date axes but
the same instrument columns are NOT the same execution identity."""
from __future__ import annotations

import pandas as pd

from factor_engine.cleaned_operators.common._polars_bridge import (
    PanelIdentity,
    verify_frames_share_identity,
)


def _panel(dates, instruments=("A", "B")):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), list(instruments)],
        names=["timestamp", "instrument"],
    )
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_multiindex_time_axis_is_hashed():
    p = _panel(["2024-01-01", "2024-01-02"])
    ident = PanelIdentity.from_frame(p)
    assert ident.time_index_hash is not None  # no longer blanked to None
    assert ident.time_index_hash == PanelIdentity.from_frame(p).time_index_hash


def test_multiindex_date_shift_changes_identity():
    a = PanelIdentity.from_frame(_panel(["2024-01-01", "2024-01-02"]))
    b = PanelIdentity.from_frame(_panel(["2024-01-02", "2024-01-03"]))
    assert a != b  # date axis shifted one day must loud-fail


def test_multiindex_same_dates_same_identity():
    a = PanelIdentity.from_frame(_panel(["2024-01-01", "2024-01-02"]))
    b = PanelIdentity.from_frame(_panel(["2024-01-01", "2024-01-02"]))
    assert a == b


def test_multiindex_instrument_change_changes_identity():
    a = PanelIdentity.from_frame(_panel(["2024-01-01"], ("A", "B")))
    b = PanelIdentity.from_frame(_panel(["2024-01-01"], ("A", "C")))
    assert a != b


def test_verify_frames_share_identity_detects_long_panel_shift():
    a = _panel(["2024-01-01", "2024-01-02"])
    b = _panel(["2024-01-02", "2024-01-03"])
    try:
        verify_frames_share_identity("multiindex shift", a, b)
    except ValueError as exc:
        assert "different PanelIdentity" in str(exc)
    else:
        raise AssertionError("expected a shift-detection ValueError")
