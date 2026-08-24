# -*- coding: utf-8
"""Production gate：ffill 无限填充与 evidence artifact 要求。"""
from __future__ import annotations

import pytest

from factor_engine.backend.operator_call_capability import CapabilityLevel, check_operator_call_capability
from factor_engine.backend.production_signature import (
    PRODUCTION_SIGNATURES,
    operational_production_allowed,
    verify_production_signature,
)
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture(scope="module")
def _loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

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
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert "ffill" not in OperatorRegistry._operators


def test_ffill_with_limit_signature_ok(_loaded):
    node = _ffill_plan(limit=5)
    ok, _ = verify_production_signature("ffill", node, production=True)
    assert ok


def test_ffill_unlimited_blocked_in_production_gate(_loaded):
    r = check_operator_call_capability("ffill", node=_ffill_plan(), production=True)
    assert r.level == CapabilityLevel.RESEARCH
    assert "unlimited" in r.reason or "limit" in r.reason


def _plan(op: str, *values, **attrs) -> PlanNode:
    inputs = [PlanNode(op="column", attrs={"name": "x"})]
    inputs.extend(PlanNode(op="literal", attrs={"value": value}) for value in values)
    return PlanNode(op=op, inputs=inputs, attrs=attrs)


def test_every_daily_canonical_has_exact_production_signature(_loaded):
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    assert set(PRODUCTION_SIGNATURES) == set(DAILY_CANONICALS)


@pytest.mark.parametrize("invalid", [True, 1.7, float("nan"), float("inf"), -1])
def test_integer_constraints_reject_non_exact_or_negative_values(_loaded, invalid):
    ok, _ = verify_production_signature(
        "ts_delay", _plan("ts_delay", invalid), production=True
    )
    assert not ok


def test_cross_parameter_constraints_are_executable(_loaded):
    ok, reason = verify_production_signature(
        "ts_mean", _plan("ts_mean", 5, min_periods=6), production=True
    )
    assert not ok and "min_periods" in reason

    ok, reason = verify_production_signature(
        "ts_autocorr", _plan("ts_autocorr", 5, 5), production=True
    )
    assert not ok and "lag" in reason

    ok, reason = verify_production_signature(
        "ts_sharpe", _plan("ts_sharpe", 1), production=True
    )
    assert not ok and "window" in reason

    ok, reason = verify_production_signature(
        "winsorize", _plan("winsorize", 0.9, 0.1), production=True
    )
    assert not ok and "lower" in reason


def test_mode_enums_match_runtime_contracts(_loaded):
    assert verify_production_signature(
        "period_change", _plan("period_change", "pid", 1, "log", True), production=True
    )[0]
    assert verify_production_signature(
        "period_cagr", _plan("period_cagr", "pid", 4, 4, "absolute", True), production=True
    )[0]


def test_production_rejects_missing_signature_and_undeclared_parameter(_loaded):
    ok, reason = verify_production_signature("ts_ema", _plan("ts_ema", 5), production=True)
    assert not ok and "missing production signature" in reason
    ok, reason = verify_production_signature(
        "abs", _plan("abs", surprise=True), production=True
    )
    assert not ok and "undeclared production parameters" in reason
