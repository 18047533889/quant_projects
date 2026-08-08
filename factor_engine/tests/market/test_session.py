# -*- coding: utf-8 -*-
"""SessionSpec tests — A-share 240-bar continuous / US DST·early-close-aware."""
from __future__ import annotations

from datetime import datetime

import pytest

from market import ASHARE_SESSION, US_SESSION, session_for


def test_ashare_240_bar_session() -> None:
    spec = session_for("ashare")
    assert spec.session_id == "ASHARE_CONTINUOUS"
    assert spec.timezone == "Asia/Shanghai"
    assert spec.slot_count == 240
    # 09:31 -> slot 1, 11:30 -> slot 120, 13:01 -> slot 121, 15:00 -> slot 240
    assert spec.slot_for_local(datetime(2024, 1, 2, 9, 31)) == 1
    assert spec.slot_for_local(datetime(2024, 1, 2, 11, 30)) == 120
    assert spec.slot_for_local(datetime(2024, 1, 2, 13, 1)) == 121
    assert spec.slot_for_local(datetime(2024, 1, 2, 15, 0)) == 240
    # lunch break has no bar
    assert spec.slot_for_local(datetime(2024, 1, 2, 12, 0)) is None


def test_us_regular_session() -> None:
    spec = session_for("us")
    assert spec.session_id == "US_REGULAR"
    assert spec.timezone == "America/New_York"
    assert spec.early_close_policy == "down_weight"
    assert spec.slot_for_local(datetime(2024, 1, 2, 9, 30)) == 1
    assert spec.slot_for_local(datetime(2024, 1, 2, 16, 0)) == 391


def test_session_serialization() -> None:
    d = ASHARE_SESSION.to_dict()
    assert len(d["segments"]) == 2
    assert d["slot_count"] == 240
    assert d["timezone"] == "Asia/Shanghai"


def test_session_contracts_align_with_context() -> None:
    from market import ASHARE_CONTEXT, US_CONTEXT

    assert ASHARE_CONTEXT.session_id == ASHARE_SESSION.session_id
    assert US_CONTEXT.session_id == US_SESSION.session_id
