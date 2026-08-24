# -*- coding: utf-8 -*-
"""Real A-share COS data → intraday operator integration (opt-in).

These tests read actual A-share minute data directly from COS through
dataaccess (no mirror download) and run the production intraday aggregation
operators on it.  They require ``DATA_ACCESS_COS_READ_MODE=remote`` (and COS
credentials/CLI configuration) and are skipped otherwise, so local/CI runs
without COS access stay green.
"""
from __future__ import annotations

import datetime
import os

import pytest

from factor_engine.cleaned_operators import load_all

pytestmark = pytest.mark.skipif(
    os.environ.get("DATA_ACCESS_COS_READ_MODE", "").lower() != "remote",
    reason="set DATA_ACCESS_COS_READ_MODE=remote to run real-COS integration",
)


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()
    return True


def _minute_panel(symbol: str = "688370.SH", day: str = "2026-06-30"):
    from data_access import get_store

    store = get_store()
    r = store.read_result(
        "ashare_stock_minute",
        columns=["QuoteTime", "Symbol", "Close", "Amount", "Volume"],
        time_range=(datetime.date.fromisoformat(day), datetime.date.fromisoformat(day)),
        instrument_filter=[symbol],
    )
    df = r.table.to_pandas().sort_values(["Symbol", "QuoteTime"])
    assert len(df) > 0, "COS minute read returned no rows"
    panel = df.set_index("QuoteTime")["Close"].to_frame(symbol)
    amount = df.set_index("QuoteTime")["Amount"].to_frame(symbol)
    volume = df.set_index("QuoteTime")["Volume"].astype(float).to_frame(symbol)
    return panel, amount, volume


def test_real_cos_minute_produces_sensible_intraday_values():
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        IntraAmihud,
        IntraExtremeBarReturn,
        IntraPathEfficiency,
        IntraRealizedVariance,
    )

    panel, amount, _ = _minute_panel()
    assert len(panel) >= 200  # full session ≈ 240 bars
    rv = IntraRealizedVariance()._calculate_series(panel.copy()).iloc[0, 0]
    eff = IntraPathEfficiency()._calculate_series(panel.copy()).iloc[0, 0]
    ext = IntraExtremeBarReturn()._calculate_series(panel.copy()).iloc[0, 0]
    amih = IntraAmihud()._calculate_series(panel.copy(), amount.copy()).iloc[0, 0]
    assert rv > 0 and rv < 0.1      # one-day realized variance is small
    assert 0.0 <= eff <= 1.0        # path efficiency is a ratio
    assert abs(ext) < 0.21          # a single minute move is bounded
    assert amih > 0


def test_real_cos_minute_segment_and_lunch():
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        IntraLunchGapReturn,
        IntraSegmentReturn,
        IntraSegmentVolumeShare,
    )

    panel, _, volume = _minute_panel()
    morning = IntraSegmentReturn()._calculate_series(panel.copy(), segment="morning").iloc[0, 0]
    afternoon = IntraSegmentReturn()._calculate_series(panel.copy(), segment="afternoon").iloc[0, 0]
    lunch = IntraLunchGapReturn()._calculate_series(panel.copy(), panel.copy()).iloc[0, 0]
    vol_share = IntraSegmentVolumeShare()._calculate_series(volume.copy(), segment="morning").iloc[0, 0]
    assert abs(morning) < 0.21
    assert abs(afternoon) < 0.21
    assert abs(lunch) < 0.21
    assert 0.0 < vol_share < 1.0
