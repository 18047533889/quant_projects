# -*- coding: utf-8 -*-
"""100k GO §110 item 9 / P0#9：多 worker 资源治理。

覆盖：
- solo profile → 每进程预算 31 核；
- COEXIST=2 → 每进程预算 15，且 workers × threads <= 31；
- 误配（COEXIST=4 但默认 worker 大）→ 告警 + clamp；
- 环境缓存失效对 4 个 env key 仍生效（不破坏 perf_config 测试）。

运行：``python3 -m pytest tests/test_multiworker_governance.py -q``
"""
from __future__ import annotations

import logging
import os

import pytest

from factor_engine.runtime.multiworker_governance import (
    DEFAULT_TOTAL_CORES,
    coexistence_process_count,
    log_governance_decision,
    per_process_cpu_budget,
    reset_governance_log,
    resolve_worker_budget,
)
from factor_engine.runtime.perf_config import PerfConfig

_ENV_KEYS = (
    "FACTOR_ENGINE_RESOURCE_PROFILE",
    "FACTOR_ENGINE_COEXIST",
    "FACTOR_ENGINE_NATIVE_FUSION",
    "FACTOR_ENGINE_SCHEDULER",
)


@pytest.fixture(autouse=True)
def _fresh_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    PerfConfig.invalidate_env_cache()
    reset_governance_log()
    yield
    PerfConfig.invalidate_env_cache()
    reset_governance_log()


def test_solo_profile_budget_31(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RESOURCE_PROFILE", "solo")
    perf = PerfConfig.from_env()
    assert coexistence_process_count(perf) == 1
    assert per_process_cpu_budget(perf) == DEFAULT_TOTAL_CORES  # 31


def test_coexist_2_budget_15(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_COEXIST", "2")
    perf = PerfConfig.from_env()
    assert coexistence_process_count(perf) == 2
    assert per_process_cpu_budget(perf) == 15  # floor(31/2)


def test_shared4_profile_budget_7(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_RESOURCE_PROFILE", "shared4")
    perf = PerfConfig.from_env()
    assert coexistence_process_count(perf) == 4
    assert per_process_cpu_budget(perf) == 7  # floor(31/4)


def test_workers_times_threads_capped_by_budget(monkeypatch):
    # COEXIST=2 → budget 15。请求 16 workers × 1 thread → clamp 到 15。
    monkeypatch.setenv("FACTOR_ENGINE_COEXIST", "2")
    perf = PerfConfig.from_env()
    budget = resolve_worker_budget(16, per_worker_threads=1, perf=perf)
    assert budget.clamped is True
    assert budget.workers <= 15
    assert budget.workers * budget.per_worker_threads <= 15
    assert budget.warning is not None


def test_misconfig_warns_and_clamps(monkeypatch, caplog):
    # COEXIST=4 → budget 7。请求 16 workers × 2 threads = 32 > 7 → clamp。
    monkeypatch.setenv("FACTOR_ENGINE_COEXIST", "4")
    perf = PerfConfig.from_env()
    budget = resolve_worker_budget(16, per_worker_threads=2, perf=perf)
    assert budget.clamped is True
    assert budget.workers * budget.per_worker_threads <= 7
    assert budget.warning is not None
    with caplog.at_level(logging.WARNING, logger="factor_engine.multiworker"):
        log_governance_decision(budget)
    assert any("oversubscribes" in r.message for r in caplog.records)


def test_no_clamp_when_within_budget(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_COEXIST", "2")  # budget 15
    perf = PerfConfig.from_env()
    budget = resolve_worker_budget(5, per_worker_threads=2, perf=perf)  # 10 <= 15
    assert budget.clamped is False
    assert budget.workers == 5
    assert budget.per_worker_threads == 2
    assert budget.warning is None


def test_default_single_process_no_hidden_spawn(monkeypatch):
    # 未设置任何共存旋钮 → 默认 solo（单主进程内部并发）。
    perf = PerfConfig.from_env()
    assert coexistence_process_count(perf) == 1
    assert per_process_cpu_budget(perf) == DEFAULT_TOTAL_CORES


def test_env_cache_invalidation_still_works(monkeypatch):
    """P0#9 不破坏 perf_config 的 4-key 缓存失效（回归护栏）。"""
    cfg_a = PerfConfig.from_env()
    assert cfg_a.resource_profile == "balanced"
    assert cfg_a.coexist is True

    monkeypatch.setenv("FACTOR_ENGINE_RESOURCE_PROFILE", "solo")
    monkeypatch.setenv("FACTOR_ENGINE_COEXIST", "false")
    cfg_b = PerfConfig.from_env()
    assert cfg_b.resource_profile == "solo"
    assert cfg_b.coexist is False
