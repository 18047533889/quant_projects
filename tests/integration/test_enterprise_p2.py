# -*- coding: utf-8
"""session_calendar / warmup_service / lineage_service 测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.runtime.session_calendar import SessionBarCalendar
from factor_engine.runtime.warmup_service import _build_intraday_run_window


def test_session_bar_calendar_warmup_load_start():
    cal = SessionBarCalendar("5m")
    start = cal.warmup_load_start("2024-01-02", lookback_bars=80)
    assert start.normalize() < pd.Timestamp("2024-01-02").normalize()


def test_build_intraday_run_window_expands_load_start():
    win = _build_intraday_run_window(
        requested_start="2024-01-02",
        requested_end="2024-01-31",
        lookback_bars=80,
        bar_freq="5m",
        trim_output=True,
    )
    assert win.warmup_bars == 80
    assert win.actual_load_start is not None
    assert pd.Timestamp(win.actual_load_start).normalize() < pd.Timestamp("2024-01-02").normalize()


def test_lineage_service_expression_prefers_source_expr():
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.runtime.lineage_service import resolve_lineage_expression

    factor = parse_factor("rank(close)", name="t")
    assert resolve_lineage_expression(factor, None) == "rank(close)"
    assert resolve_lineage_expression(factor, "override") == "override"
