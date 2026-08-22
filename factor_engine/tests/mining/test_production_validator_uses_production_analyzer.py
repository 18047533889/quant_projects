# -*- coding: utf-8 -*-
"""R24-158..160: the production DSL validator runs a REAL production+market
Analyzer and resolves bare fields through the per-market registry — never the
research Analyzer or the legacy A-only FIELD_REGISTRY."""
from __future__ import annotations

import pytest

from api.mining_integration import validate_production_dsl


def test_validator_accepts_simple_production_formula() -> None:
    ok, msg = validate_production_dsl("close", market="ashare")
    assert ok, msg


def test_validator_rejects_forward_label_op_in_feature() -> None:
    # A forward return is a LABEL op — never a production feature formula.
    ok, msg = validate_production_dsl("forward_return(close, 5)", market="ashare")
    assert not ok
    assert "forward" in msg.lower() or "label" in msg.lower() or "unexpected" in msg.lower()


def test_validator_market_param_plumbed() -> None:
    # The market must change the validator's behavior (US path does not hit the
    # A-only registry).  ``close`` resolves cleanly under ashare but the US
    # production Analyzer reports a catalog-hash issue — the DIFFERENT outcomes
    # prove the market context is actually threaded through.
    ok_a, msg_a = validate_production_dsl("close", market="ashare")
    ok_u, msg_u = validate_production_dsl("close", market="us")
    assert ok_a is True
    assert (ok_a, msg_a) != (ok_u, msg_u)
