# -*- coding: utf-8 -*-
"""R38 P0-017/018/019/020 + P1-021/022（§9/§34）：Host lease tree 重构。

行为探针：
    - parent + child **不 double count** host memory（§R38_HOST_LEASE_NO_PARENT_CHILD_DOUBLE_COUNT）；
    - CPU / IO / spill 真实约束（§R38_HOST_LEASE_CPU_LIMIT_ENFORCED / IO / SPILL）；
    - JobLease 每 job 独立（§R38_MULTI_JOB_LEASE_ISOLATION）；
    - release 递归释放整棵子树 + 幂等（§R38_HOST_LEASE_RECURSIVE_RELEASE）；
    - released lease 进 terminal ring，不无限增长（P1-021）；
    - ``summary()`` 纯读，不触发 DA governor 副作用（P0-022）。
"""
from __future__ import annotations

from factor_engine.runtime.host_resource_coordinator import (
    KIND_COMPUTE,
    KIND_DA_SCAN,
    KIND_JOB,
    HostResourceCoordinator,
    JobLease,
    reset_host_coordinator,
)
from factor_engine.runtime.resource_broker import ResourceBroker


def _coordinator(hard_memory=8 * 1024**3, cpu_slots=4) -> HostResourceCoordinator:
    broker = ResourceBroker(
        hard_memory_limit=hard_memory,
        cpu_slots=cpu_slots,
        min_host_reserve_gb=0.0,
        min_host_reserve_fraction=0.0,
        spill_min_free_gb=0.0,
        spill_min_free_fraction=0.0,
    )
    return HostResourceCoordinator(broker=broker)


def test_parent_child_no_double_count():
    c = _coordinator()
    job = c.request_lease(owner="jobA", kind=KIND_JOB, memory_bytes=1024**3, cpu_tokens=2)
    assert job is not None
    child = c.request_lease(
        owner="jobA:compute", kind=KIND_COMPUTE, memory_bytes=512 * 1024**2,
        cpu_tokens=1, parent_lease_id=job.lease_id,
    )
    assert child is not None
    rec = c.reconcile()
    # host 只计 root（1GB），不计 child（0.5GB）——不 double count。
    assert rec["active_memory_bytes"] == 1024**3
    # child 不能超过 parent 剩余额度。
    too_big = c.request_lease(
        owner="jobA:compute2", memory_bytes=1024**3, parent_lease_id=job.lease_id,
    )
    assert too_big is None


def test_cpu_limit_enforced():
    c = _coordinator(hard_memory=8 * 1024**3, cpu_slots=2)
    r1 = c.request_lease(owner="job1", kind=KIND_JOB, memory_bytes=1024**3, cpu_tokens=2)
    assert r1 is not None
    # 第二个 root 需要 cpu=1 → 超过 target_cpu_tokens=2 → 拒绝。
    r2 = c.request_lease(owner="job2", kind=KIND_JOB, memory_bytes=1024**3, cpu_tokens=1)
    assert r2 is None
    # child 在 root 内分配不受 host cpu 限制（但受 parent 剩余约束）。
    child = c.request_lease(
        owner="job1:compute", memory_bytes=1, cpu_tokens=1, parent_lease_id=r1.lease_id,
    )
    assert child is not None


def test_spill_accounted_and_limited():
    c = _coordinator()
    r = c.request_lease(owner="jobS", kind=KIND_JOB, memory_bytes=1024**3, cpu_tokens=1, spill_bytes=12345)
    assert r is not None
    assert c._root_reserved("spill") == 12345
    # 超大 spill（> usable spill disk）→ 拒绝。
    huge = c.request_lease(owner="jobH", kind=KIND_JOB, memory_bytes=1, cpu_tokens=0, spill_bytes=10**15)
    assert huge is None


def test_job_lease_isolation_multi_job():
    c = _coordinator(hard_memory=16 * 1024**3, cpu_slots=8)
    job_a = c.request_job_lease(owner="svc:A", memory_bytes=4 * 1024**3, cpu_tokens=4)
    job_b = c.request_job_lease(owner="svc:B", memory_bytes=4 * 1024**3, cpu_tokens=4)
    assert isinstance(job_a, JobLease) and isinstance(job_b, JobLease)
    assert job_a.lease_id != job_b.lease_id
    # 每个 job 独立派生 child（不互相覆盖内存 ceiling）。
    child_a = job_a.request_child(owner="svc:A:duckdb", kind=KIND_DA_SCAN, memory_bytes=2 * 1024**3, cpu_tokens=2)
    child_b = job_b.request_child(owner="svc:B:duckdb", kind=KIND_DA_SCAN, memory_bytes=2 * 1024**3, cpu_tokens=2)
    assert child_a is not None and child_b is not None
    rec = c.reconcile()
    assert rec["active_memory_bytes"] == 8 * 1024**3  # 两个 root，不 double count child
    job_a.release()
    assert job_a.released


def test_recursive_release_idempotent():
    c = _coordinator()
    job = c.request_job_lease(owner="jobR", memory_bytes=4 * 1024**3, cpu_tokens=4)
    child1 = job.request_child(owner="jobR:c1", memory_bytes=2 * 1024**3, cpu_tokens=2)
    child2 = job.request_child(owner="jobR:c2", memory_bytes=1 * 1024**3, cpu_tokens=1)
    gchild = c.request_lease(
        owner="jobR:c1:numba", kind=KIND_COMPUTE, memory_bytes=512 * 1024**2,
        cpu_tokens=1, parent_lease_id=child1.lease_id,
    )
    assert gchild is not None
    assert c.release_lease(job.lease_id) is True
    assert job.released
    # 整棵子树全部 released。
    for lid in (child1.lease_id, child2.lease_id, gchild.lease_id):
        assert c._leases.get(lid) is None or c._leases[lid].released
    # 幂等：重复 release 无害。
    assert c.release_lease(job.lease_id) is True


def test_terminal_ring_bounds_growth():
    c = _coordinator()
    ids = []
    for i in range(200):
        r = c.request_lease(owner=f"j{i}", kind=KIND_JOB, memory_bytes=1, cpu_tokens=0)
        assert r is not None
        ids.append(r.lease_id)
        c.release_lease(r.lease_id)
    # 主 dict 不再保留 released lease；terminal ring 有界。
    assert len(c._leases) == 0
    assert len(c._terminal) <= 64


def test_summary_is_pure_no_da_side_effect():
    import factor_engine.runtime.host_resource_coordinator as hrc

    c = _coordinator()
    calls = []

    class _Gov:
        def set_max_total_reserved_memory(self, v):
            calls.append(("mem", v))

        def set_max_total_scan_bytes_inflight(self, v):
            calls.append(("scan", v))

    c._da_governor = lambda: _Gov()
    c.summary()
    assert calls == [], "summary() 必须纯读，不能触发 DA governor 副作用（P0-022）"
    c.sync_da_limits()
    assert len(calls) == 2, "sync_da_limits() 是显式动作"
