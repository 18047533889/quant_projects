# -*- coding: utf-8 -*-
"""runtime/execution_resources.py —— 全局 CPU/memory budget 协调（#34）。"""
from __future__ import annotations

import pytest

from runtime.execution_resources import (
    physical_cores,
    reset_resource_cache,
    resource_plan,
    set_duckdb_max_threads,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_resource_cache()
    monkeypatch.setenv("FACTOR_ENGINE_CPU_BUDGET", "16")
    yield
    reset_resource_cache()


def test_resource_plan_respects_cores_budget():
    plan = resource_plan(n_jobs=8)
    # 约束：n_jobs × duckdb_threads <= 16
    assert plan.n_jobs == 8
    assert plan.duckdb_threads <= 16 // 8
    assert plan.total_runnable <= 16
    assert plan.io_concurrency >= 1


def test_resource_plan_clamps_n_jobs():
    plan = resource_plan(n_jobs=999)
    assert plan.n_jobs <= 16
    assert plan.duckdb_threads >= 1


def test_resource_plan_default_uses_cores():
    plan = resource_plan()
    assert plan.n_jobs == physical_cores()


def test_set_duckdb_max_threads_writes_env():
    plan = resource_plan(n_jobs=4)
    value = set_duckdb_max_threads(plan)
    import os

    assert os.environ.get("DUCKDB_MAX_THREADS") == str(value)
