# -*- coding: utf-8
"""Production gate：ffill 无限填充与 evidence artifact 要求。"""
from __future__ import annotations

import pytest

from backend.operator_call_capability import CapabilityLevel, check_operator_call_capability
from backend.production_signature import operational_production_allowed, verify_production_signature
from planner.logical_plan import PlanNode


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _ffill_plan(*, limit=None):
    attrs = {}
    inputs = [PlanNode(op="column", attrs={"name": "close"}, inputs=[])]
    if limit is not None:
        inputs.append(PlanNode(op="literal", attrs={"value": limit}, inputs=[]))
        attrs["limit"] = limit
    return PlanNode(op="ffill", inputs=inputs, attrs=attrs)


def test_ffill_unlimited_not_operational_production(_loaded):
    assert not operational_production_allowed("ffill")
    ok, reason = verify_production_signature("ffill", _ffill_plan(), production=True)
    assert not ok
    assert "unlimited" in reason


def test_ffill_with_limit_signature_ok(_loaded):
    node = _ffill_plan(limit=5)
    ok, _ = verify_production_signature("ffill", node, production=True)
    assert ok


def test_ffill_unlimited_blocked_in_production_gate(_loaded):
    r = check_operator_call_capability("ffill", node=_ffill_plan(), production=True)
    assert r.level == CapabilityLevel.RESEARCH
    assert "unlimited" in r.reason or "limit" in r.reason
