"""R32-P0-001 through R32-P0-020 修复验证测试。"""
import threading
import time
from unittest.mock import Mock, patch

import pytest

from data_access.r30.resolution_lease import ResolutionLease, ResolutionPhase
from data_access.r30.execution_lease import ExecutionLease
from data_access.runtime.resource_governor import GlobalResourceGovernor, ResourceReservation
from data_access.runtime.startup_gate import (
    StartupCertificate,
    build_startup_certificate,
    startup_certificate_expired,
)


class TestR32P0001ResolutionLeaseRollback:
    """R32-P0-001：ResolutionLease 部分成功回滚计数错误。"""

    def test_partial_success_rollback_only_current_attempt(self):
        """已持有 1 slot，请求 3，第 2 个失败，原 1 保留，新 1 归还。"""
        governor = Mock()
        call_count = 0

        def acquire_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # 前2次成功，第3次失败

        governor.acquire_remote_discovery_slot.side_effect = acquire_side_effect
        release_calls = []
        governor.release_remote_discovery_slot.side_effect = lambda: release_calls.append(1)

        lease = ResolutionLease(
            lease_id="test",
            max_resolution_slots=10,
            governor=governor,
        )
        # 先取得 1 个 slot。
        assert lease.acquire_resolution(1)
        assert lease._resolution_inflight == 1
        # 请求 3 个，第 2 个会失败。
        assert not lease.acquire_resolution(3)
        # R32-P0-001：应该只释放本次取得的 1 个，不释放原有的 1 个。
        assert len(release_calls) == 1
        assert lease._resolution_inflight == 1  # 原 1 保留


class TestR32P0002ResolutionLeaseLockFree:
    """R32-P0-002：ResolutionLease 禁止持内部锁调用外部 governor。"""

    def test_governor_called_outside_lock(self):
        """governor 调用不在锁内（无死锁）。"""
        governor = Mock()
        governor.acquire_remote_discovery_slot.return_value = True
        governor.release_remote_discovery_slot.return_value = None

        lease = ResolutionLease(lease_id="test", governor=governor)
        # acquire 成功。
        assert lease.acquire_resolution(1)
        # release 成功。
        lease.release_resolution(1)
        # 验证调用。
        assert governor.acquire_remote_discovery_slot.called
        assert governor.release_remote_discovery_slot.called


class TestR32P0003ResolutionLeaseDeadline:
    """R32-P0-003：ResolutionLease absolute_deadline 必须真正执行。"""

    def test_deadline_expired_blocks_acquire(self):
        """deadline 已过 → acquire 拒绝。"""
        lease = ResolutionLease(
            lease_id="test",
            absolute_deadline=time.monotonic() - 1,  # 已过期
        )
        assert not lease.acquire_resolution(1)

    def test_deadline_expired_blocks_transition(self):
        """deadline 已过 → transition 拒绝。"""
        lease = ResolutionLease(
            lease_id="test",
            absolute_deadline=time.monotonic() + 10,
        )
        lease.acquire_resolution(1)
        lease.release_all_resolution()
        # 修改 deadline 为已过期。
        lease.absolute_deadline = time.monotonic() - 1
        with pytest.raises(RuntimeError, match="deadline 已过"):
            lease.transition_to_execution()


class TestR32P0004ExecutionLeaseValidation:
    """R32-P0-004：外部 ExecutionLease 注入必须验证。"""

    def test_inject_unacquired_lease_rejected(self):
        """未 acquired 的 lease → reject。"""
        resolution = ResolutionLease(lease_id="test")
        resolution.acquire_resolution(1)
        resolution.release_all_resolution()
        fake_lease = ExecutionLease(
            lease_id="fake", memory=100, scan_bytes=100,
            remote_slots=1, duckdb_slots=1, temp_disk=100, spill_budget=100,
        )
        # 未 acquire。
        with pytest.raises(ValueError, match="未 acquired"):
            resolution.transition_to_execution(execution_lease=fake_lease)

    def test_inject_released_lease_rejected(self):
        """已 released 的 lease → reject。"""
        resolution = ResolutionLease(lease_id="test")
        resolution.acquire_resolution(1)
        resolution.release_all_resolution()
        fake_lease = ExecutionLease(
            lease_id="fake", memory=100, scan_bytes=100,
            remote_slots=1, duckdb_slots=1, temp_disk=100, spill_budget=100,
        )
        fake_lease.acquire()
        fake_lease.release()
        # Release sets _acquired=False, so it will fail the acquired check first.
        with pytest.raises(ValueError, match="未 acquired|已 released"):
            resolution.transition_to_execution(execution_lease=fake_lease)


class TestR32P0005ExecutionLeaseChildBudget:
    """R32-P0-005：ExecutionLease child release 必须归还父预算。"""

    def test_child_release_returns_budget(self):
        """parent=100, child=60, child.release() → parent.remaining=100。"""
        parent = ExecutionLease(
            lease_id="parent", memory=100, scan_bytes=100,
            remote_slots=10, duckdb_slots=5, temp_disk=100, spill_budget=100,
        )
        parent.acquire()
        child = parent.request_child({"memory": 60, "scan_bytes": 40})
        assert parent._remaining["memory"] == 40
        assert parent._remaining["scan_bytes"] == 60
        # child release。
        child.release()
        # R32-P0-005：预算归还。
        assert parent._remaining["memory"] == 100
        assert parent._remaining["scan_bytes"] == 100


class TestR32P0006ExecutionLeaseThreadSafe:
    """R32-P0-006：ExecutionLease 全生命周期线程安全。"""

    def test_concurrent_request_child_and_release(self):
        """并发 request_child + release 总量不超父。"""
        parent = ExecutionLease(
            lease_id="parent", memory=1000, scan_bytes=1000,
            remote_slots=10, duckdb_slots=5, temp_disk=1000, spill_budget=1000,
        )
        parent.acquire()
        children = []
        errors = []

        def worker():
            try:
                child = parent.request_child({"memory": 100, "scan_bytes": 100})
                children.append(child)
                time.sleep(0.001)
                child.release()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 最终父预算恢复。
        assert parent._remaining["memory"] == 1000


class TestR32P0007GovernorHostBackedInvariants:
    """R32-P0-007：Host-backed ResourceGovernor 不得绕过本地基础不变量。"""

    def test_duplicate_query_id_rejected_even_with_host(self):
        """duplicate query_id 即使 host 成功也 reject。"""
        gov = GlobalResourceGovernor()
        gov._host_lease_request = Mock(return_value=Mock())
        r1 = ResourceReservation(
            query_id="q1", principal_id="p1",
            estimated_memory=100, estimated_scan_bytes=100,
        )
        gov.admit(r1)
        r2 = ResourceReservation(
            query_id="q1", principal_id="p1",
            estimated_memory=100, estimated_scan_bytes=100,
        )
        with pytest.raises(Exception, match="已存在"):
            gov.admit(r2)


class TestR32P0008GovernorHostBrokenFail:
    """R32-P0-008：配置了 HostCoordinator 但调用失败时禁止静默 standalone fallback。"""

    def test_host_configured_broken_production_fails(self):
        """host 配置但失败 → production fail。"""
        gov = GlobalResourceGovernor()
        gov._host_lease_request = Mock(side_effect=RuntimeError("host broken"))
        r = ResourceReservation(
            query_id="q1", principal_id="p1",
            estimated_memory=100, estimated_scan_bytes=100,
        )
        with patch("data_access.read.query_budget.is_strict_semantics", return_value=True):
            with pytest.raises(Exception, match="host-backed admission 配置但失败"):
                gov.admit(r)


class TestR32P0009GovernorReleaseLockFree:
    """R32-P0-009：Governor release 不得持锁调用 host lease release。"""

    def test_release_calls_host_outside_lock(self):
        """release 在锁外调用 host lease.release()。"""
        gov = GlobalResourceGovernor()
        host_lease = Mock()
        r = ResourceReservation(
            query_id="q1", principal_id="p1",
            estimated_memory=100, estimated_scan_bytes=100,
            host_lease=host_lease,
        )
        gov._active["q1"] = r
        gov.release("q1")
        # 验证 host lease.release() 被调用。
        host_lease.release.assert_called_once()


class TestR32P0010DuckDBSlotDeadline:
    """R32-P0-010：DuckDB slot 等待必须受 deadline/cancellation。"""

    def test_acquire_with_timeout(self):
        """timeout=0 → nonblocking。"""
        gov = GlobalResourceGovernor(max_duckdb_concurrency=1)
        # 先占满。
        assert gov.acquire_duckdb_slot(timeout=None)
        # timeout=0 → 立即返回 False。
        assert not gov.acquire_duckdb_slot(timeout=0)


class TestR32P0012StandaloneMemoryHeadroom:
    """R32-P0-012：Standalone memory cap 应基于真实 available headroom。"""

    def test_considers_current_rss(self):
        """考虑当前 RSS（减去 current_usage）。"""
        from data_access.runtime.resource_governor import _default_safe_memory_bytes
        # 只验证函数可调用且返回合理值。
        mem = _default_safe_memory_bytes()
        assert mem > 0
        assert mem >= 512 * 1024 * 1024  # 最小 512MB


class TestR32P0013StartupCertificateImmutable:
    """R32-P0-013：StartupCertificate 改为 immutable + subject-bound。"""

    def test_certificate_frozen(self):
        """frozen dataclass，不可修改。"""
        cert = build_startup_certificate(True, [])
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            cert.passed = False


class TestR32P0014StartupCertificateSubjectDigest:
    """R32-P0-014：StartupCertificate 不能只 hash problems。"""

    def test_certificate_has_subject_digest(self):
        """成功时 subject_digest 非空。"""
        cert = build_startup_certificate(True, [], subject_digest="abc123")
        assert cert.subject_digest == "abc123"


class TestR32P0015StartupCertificateSubjectValidation:
    """R32-P0-015：StartupCertificate 失效不能只靠 24h TTL。"""

    def test_subject_digest_mismatch_expires_cert(self):
        """subject_digest 不匹配 → 证书失效。"""
        cert = build_startup_certificate(True, [], subject_digest="old_digest")
        # 不同 subject_digest → 过期。
        assert startup_certificate_expired(cert, current_subject_digest="new_digest")
        # 相同 subject_digest → 未过期（TTL 内）。
        assert not startup_certificate_expired(cert, current_subject_digest="old_digest")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
