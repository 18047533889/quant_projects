# -*- coding: utf-8
"""Production fastpath strict gate + runtime audit。"""
from __future__ import annotations

import pytest

from backend.production_fastpath_gate import (
    audit_runtime_fastpath_violations,
    check_production_fastpath_plan_ops,
    fastpath_gate_strict,
)
from backend.sql_pushdown.plan_fixtures import minimal_plan


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_strict_gate_blocks_map_groups(_loaded, monkeypatch):
    plan = minimal_plan("ts_mean")
    monkeypatch.setattr(
        "backend.polars_long_policy.infer_polars_long_tier",
        lambda _c: "map_groups",
    )
    monkeypatch.setattr(
        "backend.production_fastpath_gate._duckdb_fastpath_ok",
        lambda _c: True,
    )
    loose = check_production_fastpath_plan_ops(plan, strict=False)
    strict = check_production_fastpath_plan_ops(plan, strict=True)
    assert loose.ok
    assert not strict.ok
    assert any("strict" in v for v in strict.violations)


def test_runtime_audit_catches_fallback(_loaded):
    v = audit_runtime_fastpath_violations(
        {"polars_long_fallback_reason": "plan not capable", "used_polars_long_map_groups": True}
    )
    assert any("fallback" in x for x in v)
    assert any("map_groups" in x for x in v)


def test_fastpath_gate_strict_env_default_off(_loaded):
    assert fastpath_gate_strict(strict=False) is False
