# -*- coding: utf-8
"""production fast path policy 集成。"""
from __future__ import annotations

import os

import pytest

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.sql_pushdown.plan_fixtures import minimal_plan
from runtime.engine import FactorEngine
from runtime.production_policy import (
    ProductionPolicyViolation,
    assert_no_unapproved_map_groups_in_production,
    assert_production_fastpath_plan,
)


@pytest.fixture(scope="module")
def loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_map_groups_blocked_in_production(loaded):
    plan = minimal_plan("ewm_corr")
    with pytest.raises(ProductionPolicyViolation):
        assert_no_unapproved_map_groups_in_production(plan, mode="production")


def test_fastpath_gate_opt_in(loaded, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    plan = minimal_plan("ts_mean")
    assert_production_fastpath_plan(plan, mode="production")  # should pass


def test_fastpath_gate_rejects_deferred(loaded, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    plan = minimal_plan("ewm_corr")
    with pytest.raises(ProductionPolicyViolation):
        assert_production_fastpath_plan(plan, mode="production")


def test_run_records_backend_path_summary(loaded):
    from tests.helpers import InMemorySeriesSource
    import pandas as pd

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "A"), (pd.Timestamp("2024-01-02"), "A")],
        names=["timestamp", "instrument"],
    )
    src = InMemorySeriesSource(data={"close": pd.Series([1.0, 2.0], index=idx)})
    from backend.factory import build_backend

    eng = FactorEngine(backend=build_backend("polars_long"), data_source=src)
    out = eng.run(Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2)))
    assert "backend_path_summary" in out
    assert "backend_path" in out
    assert out["backend_path_summary"]["primary_route"]
    assert out["backend_path"]["primary_route"] == out["backend_path_summary"]["primary_route"]
