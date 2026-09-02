# -*- coding: utf-8 -*-
"""R32 P0-001..012：ResolutionLease / ExecutionLease / GlobalResourceGovernor 并发/正确性/死锁修复测试。

每个测试都有 timeout 装饰器——死锁/竞态会 FAIL 而非挂起。
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from data_access.core.exceptions import DeadlineExceeded, ResourceAdmissionError
from data_access.r30.execution_lease import ExecutionLease
from data_access.r30.resolution_lease import ResolutionLease
from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation


# ---- P0-001：ResolutionLease partial-failure rollback ----


class _PartialFailGovernor:
    """第 N 次 acquire 失败的 mock governor。"""

    def __init__(self, fail_at: int = 3) -> None:
        self.calls = 0
        self.held = 0
        self.fail_at = fail_at

    def acquire_remote_discovery_slot(self) -> bool:
        self.calls += 1
        if self.calls == self.fail_at:
            return False
        self.held += 1
        return True

    def release_remote_discovery_slot(self) -> None:
        if self.held > 0:
            self.held -= 1


@pytest.mark.timeout(5)
def test_r32_lease_001_partial_rollback_only_acquired_now():
    """T-R32-LEASE-001：部分失败时只回滚本次获取的 slot，不碰已有 slot。

    Setup: lease 已持有 1 slot，本次请求 3 slot；governor 在第 2 次 acquire 失败。
    Expected: 原有 1 slot 保留，本次获取的 1 slot 被回滚 → 最终 1 slot。
    """
    gov = _PartialFailGovernor(fail_at=2)  # 第 2 次 acquire 失败
    lease = ResolutionLease("test", max_resolution_slots=10, governor=gov)

    # 先成功获取 1 slot（已有 slot）
    assert lease.acquire_resolution(slots=1) is True
    assert lease.to_dict()["resolution_inflight"] == 1
    assert gov.held == 1

    # 请求 3 slot，第 2 次 acquire 会失败
    assert lease.acquire_resolution(slots=3) is False

    # 验证：原有 1 slot 仍在，本次获取的 1 slot 已回滚
    assert lease.to_dict()["resolution_inflight"] == 1
    assert gov.held == 1


# ---- P0-002：ResolutionLease lock-order / callback deadlock ----


class _ReentrantGovernor:
    """回调中重入 lease 的 governor（死锁探测器）。"""

    def __init__(self, target_lease: ResolutionLease | None = None) -> None:
        self.target_lease = target_lease
        self.lock_held_during_callback = False

    def acquire_remote_discovery_slot(self) -> bool:
        # 尝试重入 lease：如果 lease 的锁已持有，会死锁
        if self.target_lease is not None:
            try:
                # 简单探针：访问需要锁的状态
                _ = self.target_lease.to_dict()
            except Exception:
                pass
        return True

    def release_remote_discovery_slot(self) -> None:
        if self.target_lease is not None:
            try:
                _ = self.target_lease.to_dict()
            except Exception:
                pass


@pytest.mark.timeout(5)
def test_r32_lease_002_no_external_call_under_lock():
    """T-R32-LEASE-002：governor 回调不在 lease 内部锁下调用（防死锁）。

    使用重入 governor：回调中访问 lease 状态。如果 lease 持锁调用回调 → 死锁。
    timeout 确保死锁会 FAIL 而非挂起。
    """
    lease = ResolutionLease("reentrant", max_resolution_slots=4)
    gov = _ReentrantGovernor(target_lease=lease)
    lease.governor = gov

    # 如果实现正确，回调在锁外执行 → 不会死锁
    assert lease.acquire_resolution(slots=2) is True
    lease.release_resolution(slots=2)
    lease.release_all_resolution()


# ---- P0-003：absolute_deadline 真正执行 ----


@pytest.mark.timeout(5)
def test_r32_lease_004_expired_deadline_rejects_resolution():
    """T-R32-LEASE-004：已过期 deadline 拒绝 resolution acquisition。"""
    expired = time.monotonic() - 1.0
    lease = ResolutionLease("expired", absolute_deadline=expired, max_resolution_slots=4)

    with pytest.raises(DeadlineExceeded):
        lease.acquire_resolution()


@pytest.mark.timeout(5)
def test_r32_lease_005_transition_rejects_expired_deadline():
    """执行相转换必须拒绝已过期的 deadline。"""
    lease = ResolutionLease("exp2", absolute_deadline=time.monotonic() - 0.5, max_resolution_slots=2)
    # 强制进入 RESOLUTION_ACTIVE（跳过 deadline 检查，测试 transition 检查）
    lease.phase = lease.phase.__class__.RESOLUTION_ACTIVE
    lease._resolution_inflight = 0

    with pytest.raises(DeadlineExceeded):
        lease.transition_to_execution()


# ---- P0-004：transition_to_execution 租约校验 ----


@pytest.mark.timeout(5)
def test_r32_lease_006_transition_validates_execution_lease():
    """T-R32-LEASE-006：transition 校验外部传入的 execution_lease。"""
    lease = ResolutionLease("valid", max_resolution_slots=2)
    lease.acquire_resolution()
    lease.release_all_resolution()

    # 伪造未 acquire 的 lease
    fake = ExecutionLease("fake", 100, 0, 0, 0, 0, 0)
    with pytest.raises((RuntimeError, ValueError)):
        lease.transition_to_execution(execution_lease=fake)


# ---- P0-005：ExecutionLease child release 归还 parent ----


@pytest.mark.timeout(5)
def test_r32_lease_002_child_release_returns_parent_budget():
    """T-R32-LEASE-002：child.release() 必须归还 parent 预算。"""
    parent = ExecutionLease("parent", 100, 0, 0, 0, 0, 0).acquire()
    child = parent.request_child({"memory": 60})
    assert parent.to_dict()["remaining"]["memory"] == 40

    child.release()
    assert parent.to_dict()["remaining"]["memory"] == 100

    # 幂等：第二次 release 不重复归还
    child.release()
    assert parent.to_dict()["remaining"]["memory"] == 100


# ---- P0-006：ExecutionLease 线程安全 ----


@pytest.mark.timeout(10)
def test_r32_lease_003_concurrent_child_never_exceeds_budget():
    """T-R32-LEASE-003：并发 request_child/release 不超预算，无 double-release。"""
    parent = ExecutionLease("parent", 100, 0, 0, 0, 0, 0).acquire()
    errors = []
    children_lock = threading.Lock()
    children = []

    def allocate():
        for _ in range(50):
            try:
                child = parent.request_child({"memory": 10})
                with children_lock:
                    children.append(child)
                # 检查不超预算
                remaining = parent.to_dict()["remaining"]["memory"]
                if remaining < 0:
                    errors.append(f"negative: {remaining}")
                time.sleep(0.0001)
            except RuntimeError:
                # 预算不足是正常的
                pass

    def release():
        time.sleep(0.001)  # 让 allocate 先跑一会儿
        for _ in range(50):
            with children_lock:
                if children:
                    child = children.pop()
                else:
                    break
            try:
                child.release()
            except Exception as e:
                errors.append(f"release: {e}")

    threads = [
        threading.Thread(target=allocate),
        threading.Thread(target=allocate),
        threading.Thread(target=release),
        threading.Thread(target=release),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 释放剩余 children
    with children_lock:
        for c in children:
            c.release()

    assert not errors, errors
    # 最终全部归还
    assert parent.to_dict()["remaining"]["memory"] == 100


@pytest.mark.timeout(5)
def test_r32_lease_007_concurrent_release_vs_request_child():
    """parent.release() vs request_child() 竞态不崩溃。"""
    parent = ExecutionLease("racy", 100, 0, 0, 0, 0, 0).acquire()
    errors = []

    def requester():
        for _ in range(100):
            try:
                parent.request_child({"memory": 5})
            except (RuntimeError, AttributeError):
                # 父已释放 / 状态不一致是预期的
                pass

    def releaser():
        time.sleep(0.005)
        try:
            parent.release()
        except Exception as e:
            errors.append(str(e))

    t1 = threading.Thread(target=requester)
    t2 = threading.Thread(target=releaser)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)

    # 不应有未预期的异常
    assert not errors


# ---- P0-007：GlobalResourceGovernor host-backed 不绕过本地门禁 ----


@pytest.mark.timeout(5)
def test_r32_gov_001_host_backed_duplicate_query_id_rejected():
    """T-R32-GOV-001：host-backed 模式下，duplicate query_id 仍被拒绝。"""
    gov = GlobalResourceGovernor(max_total_reserved_memory=10000)
    gov.set_host_lease_request(lambda *_: MagicMock(release=lambda: None))

    r1 = ResourceReservation("dup", "principal1", estimated_memory=100)
    gov.admit(r1)

    r2 = ResourceReservation("dup", "principal1", estimated_memory=100)
    with pytest.raises(ResourceAdmissionError, match="已存在|duplicate"):
        gov.admit(r2)


@pytest.mark.timeout(5)
def test_r32_gov_003_host_lease_granted_but_local_fail_rolls_back():
    """T-R32-GOV-003：host lease 成功但本地记账失败 → host lease 回滚。

    同 P0-008：回滚语义在 strict/production 下生效——显式置 production 权威。

    R32-P0-007 语义：本地不变量检查**先于** host lease 请求，commit（写
    ``_active``）在 host grant 之后。触发回滚必须让本地预检通过、commit 失败
    ——用注入 commit 失败的方式（``_active.__setitem__`` 抛错）真实走到
    「host granted → 本地记账失败 → 锁外回滚 host lease」分支。旧写法
    （max_active_queries 预检拒绝）根本不会接触 host，测不到回滚。
    """
    from data_access.runtime.mode_identity import (
        reset_runtime_mode_identity,
        set_runtime_mode_identity,
    )

    token = set_runtime_mode_identity("production", source="test")
    try:
        gov = GlobalResourceGovernor(max_active_queries=8, max_total_reserved_memory=10000)
        released = []

        def fake_host_lease(*_):
            lease = MagicMock()
            lease.release = lambda: released.append(1)
            return lease

        gov.set_host_lease_request(fake_host_lease)

        r1 = ResourceReservation("q1", "p1", estimated_memory=100)
        gov.admit(r1)

        # 注入本地 commit 失败（q2）：本地预检通过、host 给 lease、
        # bookkeeping 抛错 → 必须回滚 host lease 并拒绝 admission。
        class _BoomDict(dict):
            def __setitem__(self, key, value):
                if key == "q2":
                    raise RuntimeError("simulated bookkeeping failure")
                super().__setitem__(key, value)

        gov._active = _BoomDict(gov._active)

        r2 = ResourceReservation("q2", "p1", estimated_memory=100)
        with pytest.raises(ResourceAdmissionError, match="bookkeeping"):
            gov.admit(r2)

        # 验证 host lease 被回滚
        assert len(released) == 1
        # 本地状态未被污染（q2 不在 active 里）
        assert gov.active_count() == 1
    finally:
        reset_runtime_mode_identity(token)


# ---- P0-008：配置的 HostCoordinator broken 不回退 standalone ----


@pytest.mark.timeout(5)
def test_r32_gov_002_broken_host_coordinator_fails_no_fallback():
    """T-R32-GOV-002：配置了 host coordinator 但抛异常 → fail，不回退 standalone。

    R32-P0-008 的 fail-closed 语义只在 strict/production 下生效（research 允许
    fallback 本地 governor）——测试显式置 production 权威（R39 #51 唯一权威）。
    """
    from data_access.runtime.mode_identity import (
        reset_runtime_mode_identity,
        set_runtime_mode_identity,
    )

    token = set_runtime_mode_identity("production", source="test")
    try:
        gov = GlobalResourceGovernor(max_total_reserved_memory=10000)

        def broken(*_):
            raise RuntimeError("coordinator unreachable")

        gov.set_host_lease_request(broken)

        r = ResourceReservation("fail", "principal", estimated_memory=100)
        with pytest.raises(ResourceAdmissionError):
            gov.admit(r)

        assert gov.active_count() == 0
    finally:
        reset_runtime_mode_identity(token)


# ---- P0-009：release 不持锁调用外部 host lease ----


@pytest.mark.timeout(5)
def test_r32_gov_004_release_no_lock_during_host_callback():
    """release() 释放 host lease 时不持内部锁（防死锁）。"""
    gov = GlobalResourceGovernor(max_total_reserved_memory=10000)
    reentry_ok = []

    def fake_host(*_):
        lease = MagicMock()

        def fake_release():
            # 回调中尝试访问 governor 状态（需要锁）
            try:
                _ = gov.active_count()
                reentry_ok.append(True)
            except Exception:
                reentry_ok.append(False)

        lease.release = fake_release
        return lease

    gov.set_host_lease_request(fake_host)
    r = ResourceReservation("test", "p", estimated_memory=100)
    gov.admit(r)
    gov.release("test")

    # 如果实现正确，回调在锁外 → reentry 成功
    assert reentry_ok == [True]


# ---- P0-010：DuckDB slot 支持 deadline / timeout ----


@pytest.mark.timeout(5)
def test_r32_gov_005_duckdb_slot_deadline_bounded():
    """DuckDB slot acquire 必须支持 deadline，过期立即失败。"""
    gov = GlobalResourceGovernor(max_duckdb_concurrency=1)
    # 先占住唯一 slot
    assert gov.acquire_duckdb_slot() is True

    # 第二个请求带已过期 deadline → 立即超时
    expired = time.monotonic() - 1.0
    with pytest.raises(DeadlineExceeded):
        gov.acquire_duckdb_slot(deadline=expired, timeout_ms=100)


@pytest.mark.timeout(5)
def test_r32_gov_006_remote_slot_deadline_bounded():
    """remote slot acquire 也必须支持 deadline。"""
    gov = GlobalResourceGovernor(max_remote_concurrency=1)
    assert gov.acquire_remote_slot() is True

    expired = time.monotonic() - 0.5
    with pytest.raises(DeadlineExceeded):
        gov.acquire_remote_slot(deadline=expired, timeout_ms=100)
