# -*- coding: utf-8 -*-
"""Tests for MultiBackend P2 Governance (MB-P2-001 through MB-P2-016)."""
import pytest
import time
from factor_engine.backend.multibackend_governance import (
    # MB-P2-007: Cache
    CacheLayer,
    CachePolicy,
    CacheGovernor,
    # MB-P2-008: Failure/Retry
    FailureMode,
    RetryStrategy,
    FailureRetryGovernor,
    # MB-P2-009: Timeout
    TimeoutPolicy,
    RegionTimeoutGovernor,
    # MB-P2-010: Version Tracking
    BackendVersionRegistry,
    # MB-P2-011: Serialization
    serialize_physical_plan,
    deserialize_physical_plan,
    SerializedRegion,
    # MB-P2-012: Execution Log
    RegionExecutionLogger,
    # MB-P2-013: Cost Explainability
    explain_region_cost,
    # MB-P2-014: Health Monitoring
    BackendHealthMonitor,
    # MB-P2-015: Leak Detection
    ResourceLeakDetector,
    # MB-P2-016: Boundary Audit
    RegionBoundaryAuditor,
    BoundaryViolation,
    # Global coordinator
    MultiBackendGovernor,
    get_multibackend_governor,
)


# ============================================================================
# MB-P2-007: Cache Layered Governance
# ============================================================================

def test_cache_governor_l0_registration():
    """Test L0 cache registration and stats."""
    gov = CacheGovernor()

    gov.register_entry("key1", CacheLayer.L0_MEMORY, 1000)
    gov.register_entry("key2", CacheLayer.L0_MEMORY, 2000)

    stats = gov.get_stats()
    assert stats["l0_entries"] == 2
    assert stats["l0_bytes"] == 3000


def test_cache_governor_multi_layer():
    """Test multi-layer cache registration."""
    gov = CacheGovernor()

    gov.register_entry("k1", CacheLayer.L0_MEMORY, 1000)
    gov.register_entry("k2", CacheLayer.L1_LOCAL_DISK, 5000)
    gov.register_entry("k3", CacheLayer.L2_REMOTE, 10000)

    stats = gov.get_stats()
    assert stats["l0_entries"] == 1
    assert stats["l1_entries"] == 1
    assert stats["l2_entries"] == 1
    assert stats["l0_bytes"] == 1000
    assert stats["l1_bytes"] == 5000
    assert stats["l2_bytes"] == 10000


def test_cache_governor_eviction():
    """Test cache eviction."""
    gov = CacheGovernor()

    gov.register_entry("key1", CacheLayer.L0_MEMORY, 1000)
    stats_before = gov.get_stats()
    assert stats_before["l0_entries"] == 1

    gov.evict_entry("key1", CacheLayer.L0_MEMORY)
    stats_after = gov.get_stats()
    assert stats_after["l0_entries"] == 0
    assert stats_after["l0_bytes"] == 0


def test_cache_governor_access_tracking():
    """Test cache access tracking."""
    gov = CacheGovernor()

    gov.register_entry("key1", CacheLayer.L0_MEMORY, 1000)
    # Access count starts at 1 from register_entry
    assert gov._l0_entries["key1"].access_count == 1, "Access count should be 1 after registration"
    gov.access_entry("key1", CacheLayer.L0_MEMORY)
    # Access count should be 2 after one access
    assert gov._l0_entries["key1"].access_count == 2, "Access count should be 2 after one access"


# ============================================================================
# MB-P2-008: Failure/Retry Strategy Unified
# ============================================================================

def test_failure_classification():
    """Test failure mode classification."""
    gov = FailureRetryGovernor()

    network_error = Exception("connection failed")
    assert gov.classify_failure(network_error) == FailureMode.TRANSIENT_NETWORK

    oom_error = Exception("out of memory")
    assert gov.classify_failure(oom_error) == FailureMode.OOM

    timeout_error = Exception("timeout exceeded")
    assert gov.classify_failure(timeout_error) == FailureMode.TIMEOUT


def test_retry_decision():
    """Test retry decision logic."""
    gov = FailureRetryGovernor()

    # Transient network: should retry
    should_retry, backoff = gov.should_retry(FailureMode.TRANSIENT_NETWORK, 0)
    assert should_retry is True
    assert backoff == 100.0

    # Permanent data error: should not retry
    should_retry, backoff = gov.should_retry(FailureMode.PERMANENT_DATA_ERROR, 0)
    assert should_retry is False
    assert backoff == 0.0


def test_retry_backoff():
    """Test exponential backoff."""
    gov = FailureRetryGovernor()

    _, backoff0 = gov.should_retry(FailureMode.TRANSIENT_NETWORK, 0)
    _, backoff1 = gov.should_retry(FailureMode.TRANSIENT_NETWORK, 1)
    _, backoff2 = gov.should_retry(FailureMode.TRANSIENT_NETWORK, 2)

    assert backoff0 == 100.0
    assert backoff1 == 200.0
    assert backoff2 == 400.0


def test_fallback_decision():
    """Test fallback decision."""
    gov = FailureRetryGovernor()

    # Permanent unsupported: should fallback
    assert gov.should_fallback(FailureMode.PERMANENT_UNSUPPORTED) is True

    # Transient network: should not fallback
    assert gov.should_fallback(FailureMode.TRANSIENT_NETWORK) is False


def test_failure_recording():
    """Test failure event recording."""
    gov = FailureRetryGovernor()

    error = Exception("test error")
    event = gov.record_failure("region1", "polars", error, 0)

    assert event.region_id == "region1"
    assert event.backend == "polars"
    assert event.retry_count == 0

    stats = gov.get_failure_stats()
    assert stats["total_failures"] == 1


# ============================================================================
# MB-P2-009: Region Timeout Control
# ============================================================================

def test_timeout_allocation():
    """Test timeout allocation."""
    policy = TimeoutPolicy(
        default_timeout_ms=1000,
        per_node_budget_ms=50,
        max_timeout_ms=5000,
        enable_adaptive=True,
        timeout_multiplier_on_retry=1.5,
    )
    gov = RegionTimeoutGovernor(policy)

    context = gov.allocate_timeout("region1", "polars", node_count=10, retry_count=0)

    # default_timeout + 10 * per_node_budget = 1000 + 500 = 1500
    assert context.allocated_timeout_ms == 1500.0
    assert context.region_id == "region1"


def test_timeout_retry_multiplier():
    """Test timeout multiplier on retry."""
    policy = TimeoutPolicy(
        default_timeout_ms=1000,
        per_node_budget_ms=50,
        max_timeout_ms=10000,
        enable_adaptive=True,
        timeout_multiplier_on_retry=2.0,
    )
    gov = RegionTimeoutGovernor(policy)

    context0 = gov.allocate_timeout("r1", "polars", 10, retry_count=0)
    context1 = gov.allocate_timeout("r2", "polars", 10, retry_count=1)

    # Base: 1500ms, after 1 retry: 1500 * 2.0 = 3000ms
    assert context0.allocated_timeout_ms == 1500.0
    assert context1.allocated_timeout_ms == 3000.0


def test_timeout_check():
    """Test timeout checking."""
    policy = TimeoutPolicy(
        default_timeout_ms=100,
        per_node_budget_ms=10,
        max_timeout_ms=500,
        enable_adaptive=True,
        timeout_multiplier_on_retry=1.5,
    )
    gov = RegionTimeoutGovernor(policy)

    context = gov.allocate_timeout("region1", "polars", 5, retry_count=0)

    # Should not be timed out immediately
    is_timeout, remaining = gov.check_timeout("region1")
    assert is_timeout is False
    assert remaining > 0

    # After sleeping past deadline, should timeout
    time.sleep(0.2)  # 200ms
    is_timeout, remaining = gov.check_timeout("region1")
    assert is_timeout is True
    assert remaining <= 0


def test_timeout_release():
    """Test timeout context release."""
    policy = TimeoutPolicy(
        default_timeout_ms=1000,
        per_node_budget_ms=50,
        max_timeout_ms=5000,
        enable_adaptive=True,
        timeout_multiplier_on_retry=1.5,
    )
    gov = RegionTimeoutGovernor(policy)

    gov.allocate_timeout("region1", "polars", 10)
    gov.release_timeout("region1")

    # After release, check_timeout returns not timeout
    is_timeout, _ = gov.check_timeout("region1")
    assert is_timeout is False


# ============================================================================
# MB-P2-010: Backend Version Tracking
# ============================================================================

def test_version_registration():
    """Test backend version registration."""
    registry = BackendVersionRegistry()

    registry.register_version("polars", "0.20.0", "cap_hash_123")
    version = registry.get_version("polars")

    assert version is not None
    assert version.backend == "polars"
    assert version.version == "0.20.0"
    assert version.capabilities_hash == "cap_hash_123"


def test_composite_hash():
    """Test composite hash generation."""
    registry = BackendVersionRegistry()

    registry.register_version("polars", "0.20.0", "cap_hash_1")
    registry.register_version("duckdb", "0.10.0", "cap_hash_2")

    hash1 = registry.get_composite_hash()
    assert len(hash1) == 16

    # Changing a version changes the hash
    registry.register_version("polars", "0.21.0", "cap_hash_1")
    hash2 = registry.get_composite_hash()
    assert hash1 != hash2


# ============================================================================
# MB-P2-011: Physical Plan Serialization
# ============================================================================

def test_plan_serialization():
    """Test physical plan serialization."""
    class MockPlan:
        def __init__(self):
            self.plan_id = "plan123"
            self.plan_hash = "hash456"
            self.backend_version_hash = "ver789"
            self.regions = [MockRegion()]
            self.metadata = {"key": "value"}

    class MockRegion:
        def __init__(self):
            self.region_id = "r1"
            self.backend = "polars"
            self.representation = "POLARS_LAZY_LONG"
            self.execution_axis = "TIME_PER_INSTRUMENT"
            self.node_ids = ["n1", "n2"]
            self.estimated_cost_ms = 100.0
            self.estimated_memory_mb = 50.0

    plan = MockPlan()
    json_str = serialize_physical_plan(plan)

    assert "plan123" in json_str
    assert "polars" in json_str
    assert "POLARS_LAZY_LONG" in json_str


def test_plan_deserialization():
    """Test physical plan deserialization."""
    class MockPlan:
        def __init__(self):
            self.plan_id = "plan123"
            self.plan_hash = "hash456"
            self.backend_version_hash = "ver789"
            self.regions = [MockRegion()]
            self.metadata = {"test": "data"}

    class MockRegion:
        def __init__(self):
            self.region_id = "r1"
            self.backend = "polars"
            self.representation = "POLARS_LAZY_LONG"
            self.execution_axis = "TIME_PER_INSTRUMENT"
            self.node_ids = ["n1", "n2"]
            self.estimated_cost_ms = 100.0
            self.estimated_memory_mb = 50.0

    plan = MockPlan()
    json_str = serialize_physical_plan(plan)
    deserialized = deserialize_physical_plan(json_str)

    assert deserialized.plan_id == "plan123"
    assert deserialized.plan_hash == "hash456"
    assert len(deserialized.regions) == 1
    assert deserialized.regions[0].backend == "polars"


# ============================================================================
# MB-P2-012: Region Execution Log
# ============================================================================

def test_execution_logger_start():
    """Test region execution start logging."""
    logger = RegionExecutionLogger()

    logger.start_region("region1", "polars", 10)
    log = logger.get_log()

    assert len(log) == 1
    assert log[0].region_id == "region1"
    assert log[0].status == "running"


def test_execution_logger_complete():
    """Test region execution completion logging."""
    logger = RegionExecutionLogger()

    logger.start_region("region1", "polars", 10)
    time.sleep(0.01)
    logger.complete_region("region1", actual_rows=1000, actual_bytes=50000, peak_memory_bytes=100000)

    log = logger.get_log()
    assert len(log) == 1
    assert log[0].status == "completed"
    assert log[0].actual_rows == 1000
    assert log[0].duration_ms is not None
    assert log[0].duration_ms > 0


def test_execution_logger_failure():
    """Test region execution failure logging."""
    logger = RegionExecutionLogger()

    logger.start_region("region1", "polars", 10)
    logger.fail_region("region1", "test error")

    log = logger.get_log()
    assert len(log) == 1
    assert log[0].status == "failed"
    assert log[0].error_message == "test error"


def test_execution_logger_timeout():
    """Test region execution timeout logging."""
    logger = RegionExecutionLogger()

    logger.start_region("region1", "polars", 10)
    logger.timeout_region("region1")

    log = logger.get_log()
    assert len(log) == 1
    assert log[0].status == "timeout"


# ============================================================================
# MB-P2-013: Cost Model Explainability
# ============================================================================

def test_cost_explanation():
    """Test cost breakdown generation."""
    breakdown = explain_region_cost(
        region_id="r1",
        backend="polars",
        node_count=10,
        estimated_rows=100000,
        estimated_bytes=10_000_000,
        requires_sort=True,
        requires_repartition=False,
    )

    assert breakdown.region_id == "r1"
    assert breakdown.backend == "polars"
    assert breakdown.operator_compute_ms > 0
    assert breakdown.sort_ms > 0
    assert breakdown.repartition_ms == 0
    assert len(breakdown.optimization_hints) > 0


def test_cost_dominant_factor():
    """Test dominant factor detection."""
    # Large sort should dominate
    breakdown = explain_region_cost(
        region_id="r1",
        backend="polars",
        node_count=5,
        estimated_rows=1000,
        estimated_bytes=1_000_000_000,  # 1GB
        requires_sort=True,
        requires_repartition=False,
    )

    assert breakdown.dominant_factor == "sort"


# ============================================================================
# MB-P2-014: Backend Health Monitoring
# ============================================================================

def test_health_monitor_recording():
    """Test health metrics recording."""
    monitor = BackendHealthMonitor()

    monitor.record_execution("polars", 100.0, True)
    monitor.record_execution("polars", 150.0, True)
    monitor.record_execution("polars", 200.0, False)

    health = monitor.get_health("polars")
    assert health is not None
    assert health.total_executions == 3
    assert health.error_count == 1
    assert 0.6 < health.success_rate < 0.7


def test_health_monitor_availability():
    """Test backend availability detection."""
    monitor = BackendHealthMonitor()

    # High success rate: healthy
    for _ in range(10):
        monitor.record_execution("polars", 100.0, True)

    assert monitor.is_healthy("polars") is True

    # Low success rate: unhealthy
    for _ in range(10):
        monitor.record_execution("duckdb", 100.0, False)

    assert monitor.is_healthy("duckdb") is False


def test_health_monitor_latency():
    """Test latency metrics."""
    monitor = BackendHealthMonitor()

    monitor.record_execution("polars", 100.0, True)
    monitor.record_execution("polars", 200.0, True)
    monitor.record_execution("polars", 150.0, True)

    health = monitor.get_health("polars")
    assert health is not None
    assert 140 < health.avg_latency_ms < 160


# ============================================================================
# MB-P2-015: Resource Leak Detection
# ============================================================================

def test_leak_detector_registration():
    """Test resource handle registration."""
    detector = ResourceLeakDetector()

    detector.register_handle("h1", "connection", "polars", 1000)
    detector.register_handle("h2", "buffer", "duckdb", 5000)

    stats = detector.get_stats()
    assert stats["total_handles"] == 2
    assert stats["total_bytes"] == 6000


def test_leak_detector_release():
    """Test resource handle release."""
    detector = ResourceLeakDetector()

    detector.register_handle("h1", "connection", "polars", 1000)
    detector.release_handle("h1")

    stats = detector.get_stats()
    assert stats["total_handles"] == 0


def test_leak_detector_detection():
    """Test leak detection."""
    detector = ResourceLeakDetector()
    detector._leak_threshold_seconds = 0.05  # 50ms for testing

    detector.register_handle("h1", "connection", "polars", 1000)
    time.sleep(0.1)  # Wait for leak threshold

    leaks = detector.detect_leaks()
    assert len(leaks) == 1
    assert leaks[0].handle_id == "h1"


def test_leak_detector_stats_by_type():
    """Test statistics by resource type."""
    detector = ResourceLeakDetector()

    detector.register_handle("h1", "connection", "polars", 1000)
    detector.register_handle("h2", "connection", "duckdb", 2000)
    detector.register_handle("h3", "buffer", "polars", 5000)

    stats = detector.get_stats()
    assert stats["by_type"]["connection"] == 2
    assert stats["by_type"]["buffer"] == 1
    assert stats["by_backend"]["polars"] == 2
    assert stats["by_backend"]["duckdb"] == 1


# ============================================================================
# MB-P2-016: Region Boundary Audit
# ============================================================================

def test_boundary_auditor_schema_match():
    """Test boundary audit with matching schema."""
    auditor = RegionBoundaryAuditor()

    schema = {"col1": "int64", "col2": "float64"}
    violations = auditor.audit_boundary(
        "r1", "r2", schema, schema, 1000, 1000, False, False
    )

    assert len(violations) == 0


def test_boundary_auditor_schema_mismatch():
    """Test boundary audit with schema mismatch."""
    auditor = RegionBoundaryAuditor()

    source_schema = {"col1": "int64", "col2": "float64"}
    target_schema = {"col1": "int64", "col2": "int64"}

    violations = auditor.audit_boundary(
        "r1", "r2", source_schema, target_schema, 1000, 1000, False, False
    )

    assert len(violations) == 1
    assert violations[0].violation_type == "schema_mismatch"
    assert violations[0].severity == "error"


def test_boundary_auditor_row_count_mismatch():
    """Test boundary audit with row count mismatch."""
    auditor = RegionBoundaryAuditor()

    schema = {"col1": "int64"}
    violations = auditor.audit_boundary(
        "r1", "r2", schema, schema, 1000, 900, False, False
    )

    assert len(violations) == 1
    assert violations[0].violation_type == "missing_data"


def test_boundary_auditor_ordering_violation():
    """Test boundary audit with ordering violation."""
    auditor = RegionBoundaryAuditor()

    schema = {"col1": "int64"}
    violations = auditor.audit_boundary(
        "r1", "r2", schema, schema, 1000, 1000, requires_sort=True, is_sorted=False
    )

    assert len(violations) == 1
    assert violations[0].violation_type == "ordering_violated"


def test_boundary_auditor_critical_detection():
    """Test critical violation detection."""
    auditor = RegionBoundaryAuditor()

    schema = {"col1": "int64"}
    # Large row count difference: critical
    auditor.audit_boundary("r1", "r2", schema, schema, 1000, 500, False, False)

    assert auditor.has_critical_violations() is True


# ============================================================================
# Global Coordinator Tests
# ============================================================================

def test_multibackend_governor_singleton():
    """Test global governor singleton."""
    gov1 = get_multibackend_governor()
    gov2 = get_multibackend_governor()

    assert gov1 is gov2


def test_multibackend_governor_components():
    """Test governor has all components."""
    gov = MultiBackendGovernor()

    assert gov.cache_governor is not None
    assert gov.failure_retry_governor is not None
    assert gov.timeout_governor is not None
    assert gov.version_registry is not None
    assert gov.execution_logger is not None
    assert gov.health_monitor is not None
    assert gov.leak_detector is not None
    assert gov.boundary_auditor is not None


def test_multibackend_governor_report():
    """Test comprehensive governance report."""
    gov = MultiBackendGovernor()

    # Add some test data
    gov.cache_governor.register_entry("k1", CacheLayer.L0_MEMORY, 1000)
    gov.execution_logger.start_region("r1", "polars", 10)

    report = gov.get_full_report()

    assert "cache" in report
    assert "failures" in report
    assert "backend_versions" in report
    assert "execution_log_entries" in report
    assert "resource_leaks" in report
    assert "boundary_violations" in report


# ============================================================================
# Integration Tests
# ============================================================================

def test_integration_region_execution_flow():
    """Test complete region execution flow with governance."""
    gov = MultiBackendGovernor()

    # 1. Register backend version
    gov.version_registry.register_version("polars", "0.20.0", "cap_hash_123")

    # 2. Allocate timeout
    timeout_ctx = gov.timeout_governor.allocate_timeout("r1", "polars", 10)

    # 3. Start execution logging
    gov.execution_logger.start_region("r1", "polars", 10)

    # 4. Register resource
    gov.leak_detector.register_handle("h1", "connection", "polars", 1000)

    # 5. Complete execution
    gov.execution_logger.complete_region("r1", 1000, 50000, 100000)
    gov.health_monitor.record_execution("polars", 100.0, True)

    # 6. Release resources
    gov.leak_detector.release_handle("h1")
    gov.timeout_governor.release_timeout("r1")

    # 7. Verify no leaks
    leaks = gov.leak_detector.detect_leaks()
    assert len(leaks) == 0

    # 8. Check health
    assert gov.health_monitor.is_healthy("polars") is True


def test_integration_failure_retry_flow():
    """Test failure and retry flow with governance."""
    gov = MultiBackendGovernor()

    # 1. Start execution
    gov.execution_logger.start_region("r1", "polars", 10)

    # 2. Failure occurs
    error = Exception("connection timeout")
    event = gov.failure_retry_governor.record_failure("r1", "polars", error, 0)

    assert event.failure_mode == FailureMode.TIMEOUT
    assert event.will_fallback is True

    # 3. Log failure
    gov.execution_logger.fail_region("r1", str(error))
    gov.health_monitor.record_execution("polars", 0.0, False)

    # 4. Check failure stats
    stats = gov.failure_retry_governor.get_failure_stats()
    assert stats["total_failures"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
