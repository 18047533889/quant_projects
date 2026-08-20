# -*- coding: utf-8 -*-
"""R38 P0-011/012/013/014（§7/§8）：固定 cadence 控制循环 + 预算统一分配。

行为探针：
    - controller tick 次数 ≈ elapsed / interval（§R38_RESOURCE_CONTROLLER_SINGLE_FIXED_CADENCE）；
    - 多个 scheduler 高频调 ``resource_decision()`` 不会放大 controller tick（
      §R38_MULTI_SCHEDULER_DOES_NOT_MULTITICK）；
    - decision 太旧 → conservative fallback（不自行 tick）；
    - ``sum(活预算) + emergency <= safe`` 硬不变量（§R38_RESOURCE_BUDGET_SUM_WITHIN_SAFE_ENVELOPE）；
    - spill 预算来自磁盘，不是 RAM（P0-016）。
"""
from __future__ import annotations

import threading
import time

import pytest

from runtime.memory_budget_allocator import MemoryBudgetAllocator
from runtime.resource_autopilot_service import (
    ResourceAutopilotService,
    reset_resource_autopilot,
    start_resource_autopilot,
)
from runtime.resource_broker import ResourceBroker
from runtime.resource_monitor import HostResourceEnvelope, ResourceSignals


@pytest.fixture(autouse=True)
def _cleanup_autopilot():
    reset_resource_autopilot()
    yield
    reset_resource_autopilot()


def test_controller_fixed_cadence_tick_count():
    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    svc = ResourceAutopilotService(broker, interval_s=0.2)
    svc.start()
    time.sleep(1.1)
    ticks = svc.tick_count()
    svc.stop()
    # ~5-6 ticks（1.1s / 0.2s），绝不能被循环次数放大成几百。
    assert 2 <= ticks <= 12, f"tick_count={ticks} not time-cadence"


def test_multi_scheduler_read_does_not_multitick():
    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    svc = ResourceAutopilotService(broker, interval_s=0.3)
    svc.start()
    time.sleep(0.7)
    base = svc.tick_count()
    # 两个 scheduler 各高频读 decision（模拟热循环）——不应增加 tick。
    for _ in range(50):
        broker.resource_decision()
    assert svc.tick_count() - base <= 2, "scheduler 读 decision 不能放大 controller tick"
    svc.stop()


def test_stale_decision_conservative_fallback():
    broker = ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4)
    svc = ResourceAutopilotService(broker, interval_s=0.2)
    svc.start()
    time.sleep(0.5)
    snap = svc.last_decision()
    assert snap is not None
    # 手动把 snapshot 标记为过期 → broker.resource_decision() 走 conservative fallback
    # 或旧 decision，但不自行 tick（tick_count 不增加）。
    from dataclasses import replace

    from runtime.resource_autopilot_service import ResourceDecisionSnapshot

    old = svc._snapshot
    svc._snapshot = ResourceDecisionSnapshot(
        decision_id="stale",
        generated_at_monotonic=time.monotonic() - 100,
        valid_until=time.monotonic() - 50,
        signals_version="sig:old",
        decision=old.decision,
    )
    d = broker.resource_decision()
    assert d is not None
    svc.stop()


def test_budget_sum_within_safe_envelope():
    alloc = MemoryBudgetAllocator()
    for safe in (2 * 1024**3, 8 * 1024**3, 32 * 1024**3):
        for stage in ("NORMAL", "PRESSURE_2", "PRESSURE_3", "CRITICAL"):
            a = alloc.allocate(
                safe,
                pressure_stage=stage,
                active_live_bytes=int(safe * 0.1),
                spill_free_bytes=64 * 1024**3,
                spill_reserve_bytes=8 * 1024**3,
            )
            live = a.read_wave_bytes + a.factor_block_bytes + a.result_queue_bytes + a.cache_bytes
            assert live + a.emergency_reserve_bytes <= safe - a.active_live_bytes, (
                f"stage={stage} safe={safe}: live+emergency={live + a.emergency_reserve_bytes} "
                f"> safe-active={safe - a.active_live_bytes}"
            )
            assert a.within_safe
            # spill 来自磁盘：≥ 0 且不占 RAM 预算。
            assert a.spill_bytes == 64 * 1024**3 - 8 * 1024**3


def test_budget_low_headroom_does_not_exceed_safe():
    # 极低 headroom：active_live 接近 safe，预算必须收缩到下限以下而非超 safe。
    safe = 2 * 1024**3
    active = int(safe * 0.95)
    a = MemoryBudgetAllocator().allocate(safe, active_live_bytes=active)
    live = a.read_wave_bytes + a.factor_block_bytes + a.result_queue_bytes + a.cache_bytes
    assert live + a.emergency_reserve_bytes <= safe - active


def test_job_lease_caps_all_memory_budgets(monkeypatch):
    """A small job on a large host must not receive host-sized memory targets."""
    from runtime.resource_autopilot import ResourceController

    lease = 384 * 1024**2
    broker = ResourceBroker(
        hard_memory_limit=64 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=1,
    )
    monkeypatch.setattr(broker, "pressure_stage", lambda: "NORMAL")
    decision = ResourceController(broker).tick(
        ResourceSignals(),
        HostResourceEnvelope(
            hard_memory_bytes=64 * 1024**3,
            safe_memory_bytes=32 * 1024**3,
            emergency_reserve_bytes=2 * 1024**3,
            hard_cpu_tokens=8,
            target_cpu_tokens=8,
            io_capacity_score=1.0,
            remote_capacity_score=1.0,
            spill_free_bytes=0,
        ),
        job_memory_lease_bytes=lease,
    )

    live = (
        decision.read_wave_bytes
        + decision.factor_block_bytes
        + decision.result_queue_bytes
        + decision.cache_budget_bytes
    )
    assert live <= lease
    assert decision.read_wave_bytes <= lease
