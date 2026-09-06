# -*- coding: utf-8
"""production fast path policy 集成。"""
from __future__ import annotations

import os

import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.sql_pushdown.plan_fixtures import minimal_plan
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.production_policy import (
    ProductionPolicyViolation,
    assert_no_unapproved_map_groups_in_production,
    assert_production_fastpath_plan,
)


@pytest.fixture(scope="module")
def loaded():
    from factor_engine.cleaned_operators import load_all
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_map_groups_blocked_in_production(loaded, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    plan = minimal_plan("ts_kurt")
    with pytest.raises(ProductionPolicyViolation):
        assert_no_unapproved_map_groups_in_production(plan, mode="production")


def test_fastpath_gate_opt_in(loaded, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    plan = minimal_plan("ts_mean")
    with pytest.raises(ProductionPolicyViolation, match="dual-backend|双后端"):
        assert_production_fastpath_plan(plan, mode="production")


def test_fastpath_gate_rejects_deferred(loaded, monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "1")
    plan = minimal_plan("ts_ewm_corr")
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
    from factor_engine.backend.factory import build_backend

    eng = FactorEngine(backend=build_backend("polars_long"), data_source=src)
    out = eng.run(Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2)))
    assert "backend_path_summary" in out
    assert "backend_path" in out
    assert out["backend_path_summary"]["primary_route"]
    assert out["backend_path"]["primary_route"] == out["backend_path_summary"]["primary_route"]
