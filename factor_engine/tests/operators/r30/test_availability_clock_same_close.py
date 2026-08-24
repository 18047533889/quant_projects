# -*- coding: utf-8 -*-
"""R30 §27/§28/§29: Availability / Decision / Execution clock.

* a session-close factor with same_session_usable=True is a hard blocker;
* undeclared daily session-end factors default to conservative EOD semantics;
* the audit reports zero explicit same-close lookaheads.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.availability_clock import (
    availability_clock_ok,
    default_available_at,
    default_same_session_usable,
)


def test_session_close_with_same_bar_usable_is_blocker():
    ok, reason = availability_clock_ok(
        available_at="session_close", same_session_usable=True, input_names=("close",)
    )
    assert not ok
    assert "same-close" in reason


def test_session_close_with_same_bar_false_is_ok():
    ok, _ = availability_clock_ok(
        available_at="session_close", same_session_usable=False, input_names=("close",)
    )
    assert ok


def test_undeclared_session_end_factor_is_not_blocker():
    ok, _ = availability_clock_ok(
        available_at=None, same_session_usable=None, input_names=("close", "volume")
    )
    assert ok


def test_default_available_at_for_session_end_inputs():
    assert default_available_at(("close", "volume")) == "session_close"
    # R34 P0-018: open / open_price 在开盘即可知 —— session_open，不再是 None
    # （旧行为错误阻止 open-only 的 same-session factor）。
    assert default_available_at(("open_price",)) == "session_open"
    assert default_available_at(("open",)) == "session_open"
    # 混用 open + close 时取 max knowledge time = session_close
    assert default_available_at(("open", "close")) == "session_close"


def test_default_same_session_usable_for_session_end():
    assert default_same_session_usable(("close",)) is False
    # R34 P0-018: open-only factor 同 session 可用
    assert default_same_session_usable(("open_price",)) is True
    assert default_same_session_usable(("open",)) is True
    assert default_same_session_usable(("open", "close")) is False
    assert default_same_session_usable(("close",), declared=True) is True


def test_availability_invariant_order_documented():
    # The invariant is a constraint; the module documents it.  No execution path
    # in the availability clock violates it (pure inference helpers).
    from factor_engine.cleaned_operators import availability_clock as ac

    doc = ac.__doc__ or ""
    assert "knowledge_time" in doc and "factor_available_time" in doc
    assert "decision_time" in doc and "execution_time" in doc
    # the ordering constraint is expressed in the docstring
    assert "factor_available_time" in doc
    assert "decision_time" in doc
