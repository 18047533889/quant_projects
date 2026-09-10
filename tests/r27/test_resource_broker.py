# -*- coding: utf-8 -*-
"""R27-022..044/236..238/241..243: ResourceBroker live headroom + token admission。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from factor_engine.runtime.resource_broker import (
    STAGE_CRITICAL,
    STAGE_NORMAL,
    STAGE_PRESSURE_3,
    ResourceBroker,
)
from factor_engine.runtime.resource_governor import ExecutionResourcePlan, live_memory_headroom_bytes
from factor_engine.runtime.task_resource_contract import TaskResourceContract, panel_bytes


_COHERENT_CAPACITY_TESTS = {
    "test_memory_token_admission_caps_concurrency",
    "test_cpu_token_admission",
    "test_pressure_stage_external_awareness",
    "test_duplicate_task_id_rejected",
    "test_reserve_backward_compat_duplicate_idempotent",
    "test_idempotent_release_via_lease",
    "test_idempotent_release_via_api",
    "test_token_accounting_exact_reserve_release_cycles",
    "test_concurrent_reserve_release_same_id",
}


@pytest.fixture(autouse=True)
def _coherent_capacity_for_accounting_tests(monkeypatch, request):
    """Keep positive-capacity fixtures independent of live shared-host use."""
    if request.node.name not in _COHERENT_CAPACITY_TESTS:
        return
    refresh = ResourceBroker._refresh

    def coherent_snapshot(broker, *, force=False):
        snapshot = refresh(broker, force=force)
        hard = broker.hard_memory_limit
        return replace(
            snapshot,
            hard_memory_limit=hard,
            cgroup_memory_current=0,
            host_mem_available=hard,
            process_rss=0,
            worker_rss=0,
            process_family_rss=0,
            process_family_pss=0,
            host_mem_available_known=True,
        )

    monkeypatch.setattr(ResourceBroker, "_refresh", coherent_snapshot)


def _task(peak: int, *, cpu: int = 1, io: int = 0, spill: int = 0,
          backend: str = "pandas_numpy") -> TaskResourceContract:
    return TaskResourceContract(
        peak_memory_bytes=peak,
        cpu_tokens=cpu,
        io_tokens=io,
        spill_bytes=spill,
        backend=backend,
    )


def test_try_reserve_rejects_duplicate_task_id_without_double_charge():
    broker = ResourceBroker(
        hard_memory_limit=64 * 1024**3,
        cpu_slots=4,
        min_host_reserve_gb=1,
        min_host_reserve_fraction=0.05,
    )
    task = _task(1024**3, cpu=3, io=1)

    first = broker.try_reserve(task, task_id="same-task")
    second = broker.try_reserve(task, task_id="same-task")

    assert first is not None
    assert second is None
    assert broker.summary()["cpu"]["in_use"] == 3
    assert broker.summary()["io"]["in_use"] == 1
    first.release()
    assert broker.summary()["cpu"]["in_use"] == 0
    assert broker.summary()["io"]["in_use"] == 0


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
    # 用小内存配置避免测试机 host memory 不足触发 PRESSURE_4。
    broker = ResourceBroker(
        hard_memory_limit=4 * 1024**3,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    # 每个 task 2GB、uncertainty 1.3 → admissible 2.6GB。
    big = _task(2 * 1024**3)
    ok1 = broker.reserve(big, task_id="a")
    ok2 = broker.reserve(big, task_id="b")  # headroom 不足
    assert ok1 is True
    assert ok2 is False
    broker.release(big, task_id="a")
    assert broker.reserve(big, task_id="b") is True


def test_cpu_token_admission():
    # R27-043/044/242：DuckDB threads=4 的 task 申请 4 token，总量受限。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    t5 = _task(256 * 1024**2, cpu=5)
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
    from factor_engine.runtime.resource_governor import process_family_rss_bytes

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


def test_duplicate_task_id_rejected():
    # R44-AUDIT-01: 同一 task_id 多次 try_reserve 必须拒绝第二次，防止 token 泄漏。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=2)
    lease1 = broker.try_reserve(task, task_id="job123")
    assert lease1 is not None
    assert broker._cpu.in_use == 2
    # 重复 task_id → 拒绝（不再获取 token）。
    lease2 = broker.try_reserve(task, task_id="job123")
    assert lease2 is None
    assert broker._cpu.in_use == 2  # 没有泄漏
    # 释放后可以再次预留同一 tid。
    lease1.release()
    assert broker._cpu.in_use == 0
    lease3 = broker.try_reserve(task, task_id="job123")
    assert lease3 is not None
    assert broker._cpu.in_use == 2


def test_reserve_backward_compat_duplicate_idempotent():
    # R44-AUDIT-03: reserve() 重复调用同一 task_id 应幂等返回 True，不泄漏 token。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=3)
    ok1 = broker.reserve(task, task_id="abc")
    assert ok1 is True
    assert broker._cpu.in_use == 3
    # 重复调用：幂等返回 True，不重复获取 token。
    ok2 = broker.reserve(task, task_id="abc")
    assert ok2 is True
    assert broker._cpu.in_use == 3  # 仍然 3，没有变成 6
    broker.release(task, task_id="abc")
    assert broker._cpu.in_use == 0


def test_idempotent_release_via_lease():
    # R44-AUDIT-04: ReservationLease.release() 重复调用不重复释放 token。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=4, io=2)
    lease = broker.try_reserve(task, task_id="xyz")
    assert lease is not None
    assert broker._cpu.in_use == 4
    assert broker._io.in_use == 2
    lease.release()
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0
    # 重复 release：幂等，不会使计数变负。
    lease.release()
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0


def test_idempotent_release_via_api():
    # R44-AUDIT-04: release() API 重复调用不重复释放 token。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=2, io=1)
    broker.reserve(task, task_id="def")
    assert broker._cpu.in_use == 2
    assert broker._io.in_use == 1
    broker.release(task, task_id="def")
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0
    # 重复 release：幂等，不使计数变负。
    broker.release(task, task_id="def")
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0


def test_release_unknown_task_id_no_op():
    # R44-AUDIT-02: 释放未曾预留的 task_id 应为 no-op，不影响计数。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=8,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=2)
    # 直接释放不存在的 tid：不应使计数变负。
    broker.release(task, task_id="ghost")
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0


def test_token_accounting_exact_reserve_release_cycles():
    # R44-AUDIT: CPU/IO token 计数在多个 reserve/release 循环后精确回零。
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=16,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    tasks = [
        _task(128 * 1024**2, cpu=2, io=1, backend="polars"),
        _task(128 * 1024**2, cpu=4, io=2, backend="duckdb"),
        _task(128 * 1024**2, cpu=1, io=0, backend="pandas_numpy"),
    ]
    leases = []
    for i, t in enumerate(tasks):
        lease = broker.try_reserve(t, task_id=f"task_{i}")
        assert lease is not None
        leases.append(lease)
    # 验证累计。
    assert broker._cpu.in_use == 2 + 4 + 1
    assert broker._io.in_use == 1 + 2 + 0
    # 释放全部。
    for lease in leases:
        lease.release()
    # 精确归零。
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0
    # 再次循环。
    for i, t in enumerate(tasks):
        ok = broker.reserve(t, task_id=f"task2_{i}")
        assert ok is True
    assert broker._cpu.in_use == 7
    assert broker._io.in_use == 3
    for i, t in enumerate(tasks):
        broker.release(t, task_id=f"task2_{i}")
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0


def test_concurrent_reserve_release_same_id():
    # R44-AUDIT-03: 并发场景下同一 task_id 只能被 reserve 一次。
    import threading
    import time
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=16,
        min_host_reserve_gb=0.1,
        min_host_reserve_fraction=0.02,
    )
    task = _task(256 * 1024**2, cpu=4, io=2)
    success_count = [0]
    failure_count = [0]
    lock = threading.Lock()
    # 用 barrier 确保所有线程同时尝试 reserve（真正的并发竞争）。
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()  # 等待全部线程就位
        lease = broker.try_reserve(task, task_id="concurrent_job")
        if lease is not None:
            with lock:
                success_count[0] += 1
            time.sleep(0.01)  # 持有一小段时间，确保其他线程看到冲突
            lease.release()
        else:
            with lock:
                failure_count[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # 只有一个线程成功，其余被拒绝（duplicate_rejected）。
    assert success_count[0] == 1
    assert failure_count[0] == 9
    # Token 精确归零。
    assert broker._cpu.in_use == 0
    assert broker._io.in_use == 0
