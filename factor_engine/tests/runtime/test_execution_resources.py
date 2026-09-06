# -*- coding: utf-8 -*-
"""runtime/execution_resources.py —— 全局 CPU/memory budget 协调（#34）。"""
from __future__ import annotations

import pytest

from factor_engine.runtime.execution_resources import (
    physical_cores,
    reset_resource_cache,
    resource_plan,
    set_duckdb_max_threads,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_resource_cache()
    monkeypatch.setenv("FACTOR_ENGINE_CPU_BUDGET", "16")
    # R27-021/167：显式给出足够小的 per-worker 峰值，让内存不再是 CPU 约束的
    # 主要限制（否则 host RAM 会通过 memory bound 把 8 核 cap 成 7，测试不确定）。
    monkeypatch.setenv("FACTOR_ENGINE_PER_WORKER_PEAK_BYTES", str(1024**3))
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


def test_memory_genuinely_limits_workers(monkeypatch):
    # R27-021/167：每 worker 峰值内存必须**真正**约束 worker 数——32 核 16GiB
    # 机器、每 worker 3GiB → 最多 floor(16GiB*0.75 / 3GiB) = 4 个 worker，
    # 而不是开满 32 核。
    reset_resource_cache()
    monkeypatch.setenv("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(16 * 1024**3))
    monkeypatch.setenv("FACTOR_ENGINE_CPU_BUDGET", "32")
    monkeypatch.setenv("FACTOR_ENGINE_PER_WORKER_PEAK_BYTES", str(3 * 1024**3))
    plan = resource_plan(n_jobs=32)
    # process_budget = 16GiB * 0.75 = 12GiB；12GiB // 3GiB = 4。
    assert plan.n_jobs == 4, f"memory bound must cap workers at 4, got {plan.n_jobs}"
    assert plan.n_jobs < 32
    # 每 worker 峰值放小后，内存不再约束 → 由 CPU 硬上限主导。
    # 128 MiB × 32 = 4 GiB < 12 GiB；原来的 1 GiB 仍只允许 12 个 worker。
    monkeypatch.setenv("FACTOR_ENGINE_PER_WORKER_PEAK_BYTES", str(128 * 1024**2))
    reset_resource_cache()
    plan2 = resource_plan(n_jobs=32)
    from factor_engine.runtime.resource_governor import effective_cpu_slots

    assert plan2.n_jobs == effective_cpu_slots(), (
        f"CPU hard limit must dominate when memory is not binding, got {plan2.n_jobs}"
    )
    assert plan2.n_jobs <= 32


def test_set_duckdb_max_threads_writes_env(monkeypatch):
    # monkeypatch.delenv 让测试结束后恢复原值，避免 DUCKDB_MAX_THREADS 泄漏
    monkeypatch.delenv("DUCKDB_MAX_THREADS", raising=False)
    plan = resource_plan(n_jobs=4)
    value = set_duckdb_max_threads(plan)
    import os

    assert os.environ.get("DUCKDB_MAX_THREADS") == str(value)
