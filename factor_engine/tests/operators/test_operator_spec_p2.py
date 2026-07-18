# -*- coding: utf-8
"""OperatorSpec / SessionCalendar / market bars_per_day P2 测试。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="legacy fiscal-quarter parameter contract superseded by strict period_id semantics")

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_policy import bars_per_day
from cleaned_operators.operator_spec import (
    PRODUCTION_CORE_CANONICALS,
    PRODUCTION_PIT_REQUIRED,
    OperatorSpec,
    build_operator_spec,
    check_fundamental_param_contracts,
    check_production_pit_declarations,
    iter_operator_specs,
)
from runtime.session_calendar import SessionBarCalendar, SessionCalendar


@pytest.fixture(scope="module")
def _loaded():
    ensure_cleaned_loaded()
    yield


def test_bars_per_day_market_aware():
    assert bars_per_day("5m") == 78
    assert bars_per_day("5m", market="US") == 78
    assert bars_per_day("5m", market="CN") == 48
    assert bars_per_day("5m", market="HK") == 66
    assert bars_per_day("1m", market="CN") == 240


def test_session_calendar_uses_market():
    cal_us = SessionCalendar(bar_freq="5m", market="US")
    cal_cn = SessionCalendar(bar_freq="5m", market="CN")
    assert cal_us.session_bars == 78
    assert cal_cn.session_bars == 48


def test_session_calendar_alias():
    assert SessionCalendar is SessionBarCalendar


def test_build_operator_spec_tier1(_loaded):
    spec = build_operator_spec("ts_mean")
    assert spec is not None
    assert isinstance(spec, OperatorSpec)
    assert spec.pit_safe is True
    assert spec.allow_in_production is True
    assert "pandas_numpy" in spec.backends


def test_production_pit_declarations_complete(_loaded):
    errors = check_production_pit_declarations()
    assert not errors, errors[:5]


def test_iter_operator_specs_nonempty(_loaded):
    specs = iter_operator_specs()
    assert len(specs) > 100
    canon_names = {s.canonical for s in specs}
    assert "ts_mean" in canon_names


def test_production_pit_required_subset_of_runtime(_loaded):
    from cleaned_operators.registry import OperatorRegistry

    missing_runtime = [
        c for c in PRODUCTION_PIT_REQUIRED if OperatorRegistry.get(c) is None
    ]
    assert not missing_runtime, missing_runtime


def test_production_core_equals_pit_required(_loaded):
    assert PRODUCTION_PIT_REQUIRED == PRODUCTION_CORE_CANONICALS


def test_fundamental_param_contracts(_loaded):
    errors = check_fundamental_param_contracts()
    assert not errors, errors
