# -*- coding: utf-8 -*-
"""R24-182/183: ``gap_bars`` is a BAR count (bar clock), never a calendar
Timedelta — a daily label with gap_bars=2 means 2 sessions, not 2 days."""
from __future__ import annotations

import pytest

from api.label_pit import LabelIR, LabelOp


def test_gap_bars_is_bar_count() -> None:
    ir = LabelIR(op=LabelOp.FORWARD_RETURN, horizon_bars=5, gap_bars=2)
    assert ir.gap_bars == 2  # a bar count, not a pd.Timedelta


def test_gap_bars_non_negative_int() -> None:
    with pytest.raises(ValueError, match="BAR count"):
        LabelIR(op=LabelOp.FORWARD_RETURN, horizon_bars=5, gap_bars=-1)
    with pytest.raises(ValueError, match="BAR count"):
        LabelIR(op=LabelOp.FORWARD_RETURN, horizon_bars=5, gap_bars=1.5)


def test_label_window_gap_does_not_use_calendar_timedelta() -> None:
    # The LabelWindowSpec keeps gap_bars as an integer bar count.
    from api.label_pit import LabelWindowSpec

    spec = LabelWindowSpec(horizon_bars=5, feature_lookback_bars=20, gap_bars=1)
    assert spec.min_separation_bars == 1
    assert isinstance(spec.min_separation_bars, int)
