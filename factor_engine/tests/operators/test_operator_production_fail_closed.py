# -*- coding: utf-8
"""P0 production fail-closed：默认 research + 仅 PRODUCTION_CORE 可投递。"""

from __future__ import annotations

import pytest

from api.mining_integration import validate_production_dsl
from backend.cleaned_bridge import build_production_dsl_allowlist, ensure_cleaned_loaded
from cleaned_operators.operator_policy import RESEARCH_CORE_CANONICALS
from cleaned_operators.operator_spec import (
    PRODUCTION_CORE_CANONICALS,
    build_operator_spec,
    production_allowed_canonicals,
)


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_default_status_is_research_not_production(_loaded):
    spec = build_operator_spec("ts_mean")
    assert spec is not None
    assert spec.status == "research"
    assert spec.allow_in_production is True


def test_non_core_technical_denied_from_production(_loaded):
    for name in ("MACD", "ADX", "RSI"):
        spec = build_operator_spec(name)
        if spec is None:
            continue
        assert spec.allow_in_production is False, name
    assert build_operator_spec("SMA").allow_in_production is True


def test_research_core_micro_not_production(_loaded):
    for name in sorted(RESEARCH_CORE_CANONICALS):
        if name.startswith("micro_"):
            spec = build_operator_spec(name)
            assert spec is not None
            assert spec.allow_in_production is False, name


def test_production_allowed_subset_of_core(_loaded):
    allowed = production_allowed_canonicals()
    assert allowed <= PRODUCTION_CORE_CANONICALS
    assert "ts_mean" in allowed
    assert "rank" in allowed
    assert "MACD" not in allowed
    assert len(allowed) <= len(PRODUCTION_CORE_CANONICALS)


def test_production_dsl_allowlist_is_core_only(_loaded):
    from cleaned_operators.registry import OperatorRegistry

    prod = build_production_dsl_allowlist()
    assert "ts_mean" in prod
    assert "rank" in prod
    assert "MACD" not in prod
    assert "shuffle" not in prod
    canons = {OperatorRegistry._aliases.get(n, n) for n in prod}
    assert canons <= PRODUCTION_CORE_CANONICALS
    assert len(canons) == len(PRODUCTION_CORE_CANONICALS)


def test_validate_production_dsl_accepts_core(_loaded):
    ok, msg = validate_production_dsl("rank(ts_mean(col('close'), 20))")
    assert ok, msg


def test_validate_production_dsl_rejects_macd(_loaded):
    ok, msg = validate_production_dsl("MACD(col('close'))")
    assert not ok
    assert "MACD" in msg or "不允许" in msg or "production" in msg.lower()


def test_validate_production_dsl_rejects_shuffle(_loaded):
    ok, msg = validate_production_dsl("shuffle(col('close'), 1)")
    assert not ok
