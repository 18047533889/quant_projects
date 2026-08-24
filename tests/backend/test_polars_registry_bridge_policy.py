from __future__ import annotations

import pytest

from factor_engine.backend.polars_registry_bridge import _op_scope, plan_registry_long_capable, registry_op_long_capable
from factor_engine.planner.logical_plan import PlanNode


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    from factor_engine.cleaned_operators import load_all

    load_all()


def test_registry_long_capability_is_non_throwing_for_unknown_discovery_name():
    assert not registry_op_long_capable("not_a_registered_operator")
    assert not plan_registry_long_capable(PlanNode(op="not_a_registered_operator"))


def test_registry_long_scope_requires_explicit_policy(monkeypatch):
    from factor_engine.cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    monkeypatch.delitem(_EXPLICIT_POLICIES, "ts_mean")
    with pytest.raises(RuntimeError, match="missing explicit"):
        _op_scope("ts_mean")
