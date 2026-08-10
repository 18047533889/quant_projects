# -*- coding: utf-8 -*-
"""R27-022..044/236..238/241..243: ResourceBroker live headroom + token admission。"""
from __future__ import annotations

from runtime.resource_broker import (
    STAGE_CRITICAL,
    STAGE_NORMAL,
    STAGE_PRESSURE_3,
    ResourceBroker,
)
from runtime.resource_governor import ExecutionResourcePlan, live_memory_headroom_bytes
from runtime.task_resource_contract import TaskResourceContract, panel_bytes


def _task(peak: int, *, cpu: int = 1, io: int = 0, spill: int = 0,
          backend: str = "pandas_numpy") -> TaskResourceContract:
    return TaskResourceContract(
        peak_memory_bytes=peak,
        cpu_tokens=cpu,
        io_tokens=io,
        spill_bytes=spill,
        backend=backend,
    )


def test_live_headroom_is_numeric():
    # R27-014/236：live headroom 是数值，不是 low/medium/high。
    h = live_memory_headroom_bytes()
    assert isinstance(h, int) and h >= 0
    plan = ExecutionResourcePlan.auto()
    assert plan.per_worker_peak_bytes == 3 * 1024**3  # R27-020/167 单位修正


def test_panel_bytes_formula():
    # R27-014：panel_bytes = rows * instruments * dtype_size。
    assert panel_bytes(10, 1000) == 80_000  # 8B
    assert panel_bytes(10, 1000, dtype_size=4.0) == 40_000


def test_memory_token_admission_caps_concurrency():
    # R27-037/038/241：live headroom 有限时，不是一口气开满 worker。
    broker = ResourceBroker(
        hard_memory_limit=16 * 1024**3,
        min_host_reserve_gb=1,
        min_host_reserve_fraction=0.1,
    )
    # 每个 task 8GB、uncertainty 1.3 → admissible 8GB*1.3=10.4GB。
    big = _task(8 * 1024**3)
    ok1 = broker.reserve(big, task_id="a")
    ok2 = broker.reserve(big, task_id="b")  # headroom 不足
    assert ok1 is True
    assert ok2 is False
    broker.release(big, task_id="a")
    assert broker.reserve(big, task_id="b") is True


def test_cpu_token_admission():
    # R27-043/044/242：DuckDB threads=4 的 task 申请 4 token，总量受限。
    broker = ResourceBroker(
        hard_memory_limit=64 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=1,
        min_host_reserve_fraction=0.05,
    )
    t5 = _task(1024**3, cpu=5)
    assert broker.reserve(t5, task_id="a")
    assert broker.reserve(t5, task_id="b") is False  # 5+5 > 8
    broker.release(t5, task_id="a")
    assert broker.reserve(t5, task_id="b")


def test_pressure_stage_external_awareness():
    # R27-041/131/237：内存 token 耗尽 → 停止新 admission（确定性）。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=4,
        min_host_reserve_gb=1,
        min_host_reserve_fraction=0.05,
    )
    # 单个 8GB task 的 admissible peak = 8GB*1.3 = 10.4GB > 可用 → 直接拒绝。
    assert broker.can_admit(_task(8 * 1024**3)) is False
    # 小 task 能过。
    assert broker.can_admit(_task(256 * 1024**2)) is True


def test_reserve_capped_on_small_machines():
    # R27-029：默认不要把 MemAvailable 吃到接近 0 —— reserve 封顶 35% RAM，
    # 小机器不被 min_host_reserve_gb=8 占满。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        min_host_reserve_gb=32,
        min_host_reserve_fraction=0.9,
    )
    assert broker._reserve_bytes() <= int(8 * 1024**3 * 0.35) + 1


def test_recommended_concurrency_adapts():
    broker = ResourceBroker(
        hard_memory_limit=128 * 1024**3,
        cpu_slots=32,
        min_host_reserve_gb=8,
        min_host_reserve_fraction=0.15,
    )
    assert 1 <= broker.recommended_concurrency() <= 32


def test_process_family_rss_accounted():
    # R27-036/238：进程族 RSS 计入（sum parent + children）。
    from runtime.resource_governor import process_family_rss_bytes

    rss = process_family_rss_bytes()
    assert isinstance(rss, int) and rss > 0


def test_spill_token_requires_free_disk():
    # R27-121/122/244：spill 不是无限，可用 spill = free_disk - reserve。
    broker = ResourceBroker(
        hard_memory_limit=64 * 1024**3,
        min_host_reserve_gb=1,
        min_host_reserve_fraction=0.05,
        spill_min_free_gb=1e6,  # 要求保留 1e6 GB → 无可用 spill
    )
    task = _task(1024**3, spill=1024**3)
    assert broker.can_admit(task) is False  # spill 不可用
