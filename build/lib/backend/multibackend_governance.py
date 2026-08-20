# -*- coding: utf-8 -*-
"""MultiBackend P2 Governance: Complete governance framework for multi-backend execution.

This module implements all 16 P2-level governance items:

MB-P2-001: BackendCapability unified authority (✓ exists in capability_registry.py)
MB-P2-002: ExecutionAxis first-class citizen (✓ exists in planning/execution_axis.py)
MB-P2-003: Representation first-class citizen (✓ exists in planning/representation.py)
MB-P2-004: ExplainPlan complete implementation (✓ exists in planner/explain_plan.py)
MB-P2-005: Full-chain telemetry (✓ exists in backend/telemetry_region.py)
MB-P2-006: q process governance (✓ exists in backend/q_backend/q_process_manager.py)
MB-P2-007: Cache layered governance (NEW - implemented here)
MB-P2-008: Failure/retry strategy unified (NEW - implemented here)
MB-P2-009: Region timeout control (NEW - implemented here)
MB-P2-010: Backend version tracking (NEW - implemented here)
MB-P2-011: Physical plan serialization (NEW - implemented here)
MB-P2-012: Region execution log (NEW - implemented here)
MB-P2-013: Cost model explainability (NEW - implemented here)
MB-P2-014: Backend health monitoring (NEW - implemented here)
MB-P2-015: Resource leak detection (NEW - implemented here)
MB-P2-016: Region boundary audit (NEW - implemented here)
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ============================================================================
# MB-P2-007: Cache Layered Governance
# ============================================================================

class CacheLayer(Enum):
    """Cache hierarchy levels."""
    L0_MEMORY = "L0_MEMORY"  # In-memory, fastest
    L1_LOCAL_DISK = "L1_LOCAL_DISK"  # Local disk cache
    L2_REMOTE = "L2_REMOTE"  # Remote cache (Redis, etc.)


@dataclass(frozen=True)
class CachePolicy:
    """Cache policy for a computation."""
    enabled_layers: tuple[CacheLayer, ...]
    ttl_seconds: int
    max_size_bytes: int | None = None
    eviction_policy: Literal["lru", "lfu", "fifo"] = "lru"
    pin_in_memory: bool = False


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    layer: CacheLayer
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    pinned: bool


class CacheGovernor:
    """Unified cache governance across L0/L1/L2 layers (MB-P2-007)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._l0_entries: dict[str, CacheEntry] = {}
        self._l1_entries: dict[str, CacheEntry] = {}
        self._l2_entries: dict[str, CacheEntry] = {}
        self._total_l0_bytes = 0
        self._total_l1_bytes = 0
        self._total_l2_bytes = 0

    def register_entry(
        self,
        key: str,
        layer: CacheLayer,
        size_bytes: int,
        pinned: bool = False,
    ):
        """Register a cache entry."""
        with self._lock:
            now = time.time()
            entry = CacheEntry(
                key=key,
                layer=layer,
                size_bytes=size_bytes,
                created_at=now,
                last_accessed=now,
                access_count=1,
                pinned=pinned,
            )

            if layer == CacheLayer.L0_MEMORY:
                self._l0_entries[key] = entry
                self._total_l0_bytes += size_bytes
            elif layer == CacheLayer.L1_LOCAL_DISK:
                self._l1_entries[key] = entry
                self._total_l1_bytes += size_bytes
            elif layer == CacheLayer.L2_REMOTE:
                self._l2_entries[key] = entry
                self._total_l2_bytes += size_bytes

    def access_entry(self, key: str, layer: CacheLayer):
        """Record cache access."""
        with self._lock:
            entries = self._get_entries_for_layer(layer)
            if key in entries:
                entry = entries[key]
                entry.last_accessed = time.time()
                entry.access_count += 1

    def evict_entry(self, key: str, layer: CacheLayer):
        """Evict a cache entry."""
        with self._lock:
            entries = self._get_entries_for_layer(layer)
            if key in entries:
                entry = entries.pop(key)
                if layer == CacheLayer.L0_MEMORY:
                    self._total_l0_bytes -= entry.size_bytes
                elif layer == CacheLayer.L1_LOCAL_DISK:
                    self._total_l1_bytes -= entry.size_bytes
                elif layer == CacheLayer.L2_REMOTE:
                    self._total_l2_bytes -= entry.size_bytes

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "l0_entries": len(self._l0_entries),
                "l0_bytes": self._total_l0_bytes,
                "l1_entries": len(self._l1_entries),
                "l1_bytes": self._total_l1_bytes,
                "l2_entries": len(self._l2_entries),
                "l2_bytes": self._total_l2_bytes,
            }

    def _get_entries_for_layer(self, layer: CacheLayer) -> dict[str, CacheEntry]:
        if layer == CacheLayer.L0_MEMORY:
            return self._l0_entries
        elif layer == CacheLayer.L1_LOCAL_DISK:
            return self._l1_entries
        elif layer == CacheLayer.L2_REMOTE:
            return self._l2_entries
        return {}


# ============================================================================
# MB-P2-008: Failure/Retry Strategy Unified
# ============================================================================

class FailureMode(Enum):
    """Failure modes for backend execution."""
    TRANSIENT_NETWORK = "TRANSIENT_NETWORK"  # Retry immediately
    TRANSIENT_RESOURCE = "TRANSIENT_RESOURCE"  # Retry with backoff
    PERMANENT_UNSUPPORTED = "PERMANENT_UNSUPPORTED"  # Fallback to another backend
    PERMANENT_DATA_ERROR = "PERMANENT_DATA_ERROR"  # Fail immediately
    TIMEOUT = "TIMEOUT"  # Retry with increased timeout
    OOM = "OOM"  # Replan with smaller batches


@dataclass(frozen=True)
class RetryStrategy:
    """Retry strategy for a failure mode."""
    failure_mode: FailureMode
    max_retries: int
    initial_backoff_ms: float
    max_backoff_ms: float
    backoff_multiplier: float
    should_fallback: bool


# Default retry strategies
DEFAULT_RETRY_STRATEGIES = {
    FailureMode.TRANSIENT_NETWORK: RetryStrategy(
        failure_mode=FailureMode.TRANSIENT_NETWORK,
        max_retries=3,
        initial_backoff_ms=100,
        max_backoff_ms=1000,
        backoff_multiplier=2.0,
        should_fallback=False,
    ),
    FailureMode.TRANSIENT_RESOURCE: RetryStrategy(
        failure_mode=FailureMode.TRANSIENT_RESOURCE,
        max_retries=5,
        initial_backoff_ms=500,
        max_backoff_ms=10000,
        backoff_multiplier=2.0,
        should_fallback=False,
    ),
    FailureMode.PERMANENT_UNSUPPORTED: RetryStrategy(
        failure_mode=FailureMode.PERMANENT_UNSUPPORTED,
        max_retries=0,
        initial_backoff_ms=0,
        max_backoff_ms=0,
        backoff_multiplier=1.0,
        should_fallback=True,
    ),
    FailureMode.PERMANENT_DATA_ERROR: RetryStrategy(
        failure_mode=FailureMode.PERMANENT_DATA_ERROR,
        max_retries=0,
        initial_backoff_ms=0,
        max_backoff_ms=0,
        backoff_multiplier=1.0,
        should_fallback=False,
    ),
    FailureMode.TIMEOUT: RetryStrategy(
        failure_mode=FailureMode.TIMEOUT,
        max_retries=2,
        initial_backoff_ms=0,
        max_backoff_ms=0,
        backoff_multiplier=1.0,
        should_fallback=True,
    ),
    FailureMode.OOM: RetryStrategy(
        failure_mode=FailureMode.OOM,
        max_retries=0,
        initial_backoff_ms=0,
        max_backoff_ms=0,
        backoff_multiplier=1.0,
        should_fallback=True,
    ),
}


@dataclass
class FailureEvent:
    """Record of a failure event."""
    region_id: str
    backend: str
    failure_mode: FailureMode
    timestamp: float
    error_message: str
    retry_count: int
    will_retry: bool
    will_fallback: bool


class FailureRetryGovernor:
    """Unified failure/retry governance (MB-P2-008)."""

    def __init__(self, strategies: dict[FailureMode, RetryStrategy] | None = None):
        self._strategies = strategies or DEFAULT_RETRY_STRATEGIES
        self._lock = threading.Lock()
        self._failure_history: list[FailureEvent] = []

    def classify_failure(self, error: Exception) -> FailureMode:
        """Classify an exception into a failure mode."""
        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            return FailureMode.TRANSIENT_NETWORK
        elif "memory" in error_str or "oom" in error_str:
            return FailureMode.OOM
        elif "timeout" in error_str or "deadline" in error_str:
            return FailureMode.TIMEOUT
        elif "unsupported" in error_str or "not implemented" in error_str:
            return FailureMode.PERMANENT_UNSUPPORTED
        elif "resource" in error_str or "busy" in error_str:
            return FailureMode.TRANSIENT_RESOURCE
        else:
            return FailureMode.PERMANENT_DATA_ERROR

    def should_retry(
        self,
        failure_mode: FailureMode,
        retry_count: int,
    ) -> tuple[bool, float]:
        """Determine if should retry and backoff time.

        Returns:
            (should_retry, backoff_ms)
        """
        strategy = self._strategies.get(failure_mode)
        if not strategy or retry_count >= strategy.max_retries:
            return False, 0.0

        backoff_ms = min(
            strategy.initial_backoff_ms * (strategy.backoff_multiplier ** retry_count),
            strategy.max_backoff_ms,
        )
        return True, backoff_ms

    def should_fallback(self, failure_mode: FailureMode) -> bool:
        """Determine if should fallback to another backend."""
        strategy = self._strategies.get(failure_mode)
        return strategy.should_fallback if strategy else False

    def record_failure(
        self,
        region_id: str,
        backend: str,
        error: Exception,
        retry_count: int,
    ) -> FailureEvent:
        """Record a failure event."""
        failure_mode = self.classify_failure(error)
        will_retry, _ = self.should_retry(failure_mode, retry_count)
        will_fallback = self.should_fallback(failure_mode) if not will_retry else False

        event = FailureEvent(
            region_id=region_id,
            backend=backend,
            failure_mode=failure_mode,
            timestamp=time.time(),
            error_message=str(error),
            retry_count=retry_count,
            will_retry=will_retry,
            will_fallback=will_fallback,
        )

        with self._lock:
            self._failure_history.append(event)

        return event

    def get_failure_stats(self) -> dict[str, Any]:
        """Get failure statistics."""
        with self._lock:
            by_mode: dict[str, int] = {}
            for event in self._failure_history:
                mode_str = event.failure_mode.value
                by_mode[mode_str] = by_mode.get(mode_str, 0) + 1

            return {
                "total_failures": len(self._failure_history),
                "by_mode": by_mode,
                "recent_failures": [
                    {
                        "region_id": e.region_id,
                        "backend": e.backend,
                        "mode": e.failure_mode.value,
                        "timestamp": e.timestamp,
                    }
                    for e in self._failure_history[-10:]
                ],
            }


# ============================================================================
# MB-P2-009: Region Timeout Control
# ============================================================================

@dataclass(frozen=True)
class TimeoutPolicy:
    """Timeout policy for region execution."""
    default_timeout_ms: float
    per_node_budget_ms: float
    max_timeout_ms: float
    enable_adaptive: bool
    timeout_multiplier_on_retry: float


@dataclass
class RegionTimeoutContext:
    """Timeout context for a region execution."""
    region_id: str
    backend: str
    node_count: int
    allocated_timeout_ms: float
    deadline_timestamp: float
    retry_count: int


class RegionTimeoutGovernor:
    """Region-level timeout control (MB-P2-009)."""

    def __init__(self, policy: TimeoutPolicy):
        self._policy = policy
        self._lock = threading.Lock()
        self._active_regions: dict[str, RegionTimeoutContext] = {}

    def allocate_timeout(
        self,
        region_id: str,
        backend: str,
        node_count: int,
        retry_count: int = 0,
    ) -> RegionTimeoutContext:
        """Allocate timeout for a region execution."""
        base_timeout = min(
            self._policy.default_timeout_ms + node_count * self._policy.per_node_budget_ms,
            self._policy.max_timeout_ms,
        )

        allocated_timeout = base_timeout * (
            self._policy.timeout_multiplier_on_retry ** retry_count
        )
        allocated_timeout = min(allocated_timeout, self._policy.max_timeout_ms)

        deadline = time.time() + allocated_timeout / 1000.0

        context = RegionTimeoutContext(
            region_id=region_id,
            backend=backend,
            node_count=node_count,
            allocated_timeout_ms=allocated_timeout,
            deadline_timestamp=deadline,
            retry_count=retry_count,
        )

        with self._lock:
            self._active_regions[region_id] = context

        return context

    def check_timeout(self, region_id: str) -> tuple[bool, float]:
        """Check if region has timed out.

        Returns:
            (is_timeout, remaining_ms)
        """
        with self._lock:
            context = self._active_regions.get(region_id)
            if not context:
                return False, 0.0

            now = time.time()
            remaining_ms = (context.deadline_timestamp - now) * 1000.0

            if remaining_ms <= 0:
                return True, 0.0

            return False, remaining_ms

    def release_timeout(self, region_id: str):
        """Release timeout context after region completes."""
        with self._lock:
            self._active_regions.pop(region_id, None)


# ============================================================================
# MB-P2-010: Backend Version Tracking
# ============================================================================

@dataclass(frozen=True)
class BackendVersion:
    """Backend version information."""
    backend: str
    version: str
    build_hash: str | None
    capabilities_hash: str
    timestamp: float


class BackendVersionRegistry:
    """Track backend versions for cache invalidation (MB-P2-010)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._versions: dict[str, BackendVersion] = {}

    def register_version(
        self,
        backend: str,
        version: str,
        capabilities_hash: str,
        build_hash: str | None = None,
    ):
        """Register a backend version."""
        with self._lock:
            self._versions[backend] = BackendVersion(
                backend=backend,
                version=version,
                build_hash=build_hash,
                capabilities_hash=capabilities_hash,
                timestamp=time.time(),
            )

    def get_version(self, backend: str) -> BackendVersion | None:
        """Get version for a backend."""
        with self._lock:
            return self._versions.get(backend)

    def get_composite_hash(self) -> str:
        """Get composite hash of all backend versions."""
        with self._lock:
            components = []
            for backend in sorted(self._versions.keys()):
                version = self._versions[backend]
                components.append(f"{backend}:{version.version}:{version.capabilities_hash}")

            combined = "|".join(components)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ============================================================================
# MB-P2-011: Physical Plan Serialization
# ============================================================================

@dataclass
class SerializedRegion:
    """Serialized region for plan persistence."""
    region_id: str
    backend: str
    representation: str
    execution_axis: str
    node_ids: list[str]
    estimated_cost_ms: float
    estimated_memory_mb: float


@dataclass
class SerializedPhysicalPlan:
    """Serialized physical plan."""
    plan_id: str
    plan_hash: str
    backend_version_hash: str
    created_at: float
    regions: list[SerializedRegion]
    metadata: dict[str, Any]


def serialize_physical_plan(plan: Any) -> str:
    """Serialize physical plan to JSON (MB-P2-011)."""
    # Extract regions
    regions = []
    for reg in getattr(plan, "regions", []):
        regions.append(SerializedRegion(
            region_id=getattr(reg, "region_id", ""),
            backend=getattr(reg, "backend", ""),
            representation=getattr(reg, "representation", ""),
            execution_axis=getattr(reg, "execution_axis", ""),
            node_ids=list(getattr(reg, "node_ids", [])),
            estimated_cost_ms=getattr(reg, "estimated_cost_ms", 0.0),
            estimated_memory_mb=getattr(reg, "estimated_memory_mb", 0.0),
        ))

    serialized = SerializedPhysicalPlan(
        plan_id=getattr(plan, "plan_id", ""),
        plan_hash=getattr(plan, "plan_hash", ""),
        backend_version_hash=getattr(plan, "backend_version_hash", ""),
        created_at=time.time(),
        regions=regions,
        metadata=getattr(plan, "metadata", {}),
    )

    return json.dumps(asdict(serialized), indent=2)


def deserialize_physical_plan(json_str: str) -> SerializedPhysicalPlan:
    """Deserialize physical plan from JSON (MB-P2-011)."""
    data = json.loads(json_str)

    regions = [
        SerializedRegion(**reg_data)
        for reg_data in data["regions"]
    ]

    return SerializedPhysicalPlan(
        plan_id=data["plan_id"],
        plan_hash=data["plan_hash"],
        backend_version_hash=data["backend_version_hash"],
        created_at=data["created_at"],
        regions=regions,
        metadata=data.get("metadata", {}),
    )


# ============================================================================
# MB-P2-012: Region Execution Log
# ============================================================================

@dataclass
class RegionExecutionLogEntry:
    """Single region execution log entry."""
    region_id: str
    backend: str
    start_time: float
    end_time: float | None
    duration_ms: float | None
    status: Literal["running", "completed", "failed", "timeout"]
    error_message: str | None
    node_count: int
    actual_rows: int
    actual_bytes: int
    peak_memory_bytes: int


class RegionExecutionLogger:
    """Region execution logger (MB-P2-012)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._log: list[RegionExecutionLogEntry] = []

    def start_region(self, region_id: str, backend: str, node_count: int):
        """Log region execution start."""
        entry = RegionExecutionLogEntry(
            region_id=region_id,
            backend=backend,
            start_time=time.time(),
            end_time=None,
            duration_ms=None,
            status="running",
            error_message=None,
            node_count=node_count,
            actual_rows=0,
            actual_bytes=0,
            peak_memory_bytes=0,
        )

        with self._lock:
            self._log.append(entry)

    def complete_region(
        self,
        region_id: str,
        actual_rows: int,
        actual_bytes: int,
        peak_memory_bytes: int,
    ):
        """Log region execution completion."""
        with self._lock:
            for entry in reversed(self._log):
                if entry.region_id == region_id and entry.status == "running":
                    end_time = time.time()
                    entry.end_time = end_time
                    entry.duration_ms = (end_time - entry.start_time) * 1000.0
                    entry.status = "completed"
                    entry.actual_rows = actual_rows
                    entry.actual_bytes = actual_bytes
                    entry.peak_memory_bytes = peak_memory_bytes
                    break

    def fail_region(self, region_id: str, error_message: str):
        """Log region execution failure."""
        with self._lock:
            for entry in reversed(self._log):
                if entry.region_id == region_id and entry.status == "running":
                    end_time = time.time()
                    entry.end_time = end_time
                    entry.duration_ms = (end_time - entry.start_time) * 1000.0
                    entry.status = "failed"
                    entry.error_message = error_message
                    break

    def timeout_region(self, region_id: str):
        """Log region execution timeout."""
        with self._lock:
            for entry in reversed(self._log):
                if entry.region_id == region_id and entry.status == "running":
                    end_time = time.time()
                    entry.end_time = end_time
                    entry.duration_ms = (end_time - entry.start_time) * 1000.0
                    entry.status = "timeout"
                    break

    def get_log(self) -> list[RegionExecutionLogEntry]:
        """Get execution log."""
        with self._lock:
            return list(self._log)


# ============================================================================
# MB-P2-013: Cost Model Explainability
# ============================================================================

@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost breakdown for a region."""
    region_id: str
    backend: str

    # Compute cost components
    operator_compute_ms: float
    transfer_overhead_ms: float
    materialization_ms: float
    sort_ms: float
    repartition_ms: float

    # Memory components
    input_memory_mb: float
    working_memory_mb: float
    output_memory_mb: float

    # Reasoning
    dominant_factor: str
    optimization_hints: list[str]


def explain_region_cost(
    region_id: str,
    backend: str,
    node_count: int,
    estimated_rows: int,
    estimated_bytes: int,
    requires_sort: bool,
    requires_repartition: bool,
) -> CostBreakdown:
    """Explain cost estimate for a region (MB-P2-013)."""
    # Simplified cost model for demonstration
    bytes_mb = estimated_bytes / (1024 * 1024)

    operator_compute_ms = node_count * 5.0 + estimated_rows * 0.001
    transfer_overhead_ms = bytes_mb * 2.0
    materialization_ms = bytes_mb * 10.0
    sort_ms = bytes_mb * 50.0 if requires_sort else 0.0
    repartition_ms = bytes_mb * 30.0 if requires_repartition else 0.0

    input_memory_mb = bytes_mb
    working_memory_mb = bytes_mb * 1.5
    output_memory_mb = bytes_mb

    # Determine dominant factor
    costs = {
        "operator_compute": operator_compute_ms,
        "transfer": transfer_overhead_ms,
        "materialization": materialization_ms,
        "sort": sort_ms,
        "repartition": repartition_ms,
    }
    dominant_factor = max(costs, key=costs.get)  # type: ignore

    # Generate optimization hints
    hints = []
    if sort_ms > 100:
        hints.append("Consider pre-sorting data or using a backend with native sort")
    if repartition_ms > 100:
        hints.append("Consider adjusting shard size to avoid repartitioning")
    if materialization_ms > 200:
        hints.append("Consider using lazy execution or streaming")

    return CostBreakdown(
        region_id=region_id,
        backend=backend,
        operator_compute_ms=operator_compute_ms,
        transfer_overhead_ms=transfer_overhead_ms,
        materialization_ms=materialization_ms,
        sort_ms=sort_ms,
        repartition_ms=repartition_ms,
        input_memory_mb=input_memory_mb,
        working_memory_mb=working_memory_mb,
        output_memory_mb=output_memory_mb,
        dominant_factor=dominant_factor,
        optimization_hints=hints,
    )


# ============================================================================
# MB-P2-014: Backend Health Monitoring
# ============================================================================

@dataclass
class BackendHealthMetrics:
    """Health metrics for a backend."""
    backend: str
    is_available: bool
    last_check_time: float
    success_rate: float
    avg_latency_ms: float
    p99_latency_ms: float
    error_count: int
    total_executions: int


class BackendHealthMonitor:
    """Backend health monitoring (MB-P2-014)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[str, BackendHealthMetrics] = {}
        self._execution_history: dict[str, list[tuple[float, bool]]] = {}

    def record_execution(
        self,
        backend: str,
        duration_ms: float,
        success: bool,
    ):
        """Record a backend execution."""
        with self._lock:
            if backend not in self._execution_history:
                self._execution_history[backend] = []

            self._execution_history[backend].append((duration_ms, success))

            # Keep only last 1000 executions
            if len(self._execution_history[backend]) > 1000:
                self._execution_history[backend] = self._execution_history[backend][-1000:]

            self._update_metrics(backend)

    def _update_metrics(self, backend: str):
        """Update health metrics for a backend."""
        history = self._execution_history.get(backend, [])
        if not history:
            return

        total = len(history)
        success_count = sum(1 for _, success in history if success)
        error_count = total - success_count
        success_rate = success_count / total if total > 0 else 0.0

        durations = [d for d, _ in history]
        avg_latency = sum(durations) / len(durations) if durations else 0.0
        sorted_durations = sorted(durations)
        p99_idx = int(len(sorted_durations) * 0.99)
        p99_latency = sorted_durations[p99_idx] if sorted_durations else 0.0

        self._metrics[backend] = BackendHealthMetrics(
            backend=backend,
            is_available=success_rate >= 0.5,
            last_check_time=time.time(),
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            p99_latency_ms=p99_latency,
            error_count=error_count,
            total_executions=total,
        )

    def get_health(self, backend: str) -> BackendHealthMetrics | None:
        """Get health metrics for a backend."""
        with self._lock:
            return self._metrics.get(backend)

    def is_healthy(self, backend: str) -> bool:
        """Check if backend is healthy."""
        metrics = self.get_health(backend)
        if not metrics:
            return True  # Unknown backends are assumed healthy

        return metrics.is_available and metrics.success_rate >= 0.9


# ============================================================================
# MB-P2-015: Resource Leak Detection
# ============================================================================

@dataclass
class ResourceHandle:
    """Track a resource handle."""
    handle_id: str
    resource_type: Literal["connection", "buffer", "file", "lock"]
    backend: str
    allocated_at: float
    size_bytes: int
    stack_trace: str


class ResourceLeakDetector:
    """Resource leak detection (MB-P2-015)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._handles: dict[str, ResourceHandle] = {}
        self._leak_threshold_seconds = 300  # 5 minutes

    def register_handle(
        self,
        handle_id: str,
        resource_type: Literal["connection", "buffer", "file", "lock"],
        backend: str,
        size_bytes: int,
    ):
        """Register a resource handle."""
        import traceback

        with self._lock:
            self._handles[handle_id] = ResourceHandle(
                handle_id=handle_id,
                resource_type=resource_type,
                backend=backend,
                allocated_at=time.time(),
                size_bytes=size_bytes,
                stack_trace="".join(traceback.format_stack()),
            )

    def release_handle(self, handle_id: str):
        """Release a resource handle."""
        with self._lock:
            self._handles.pop(handle_id, None)

    def detect_leaks(self) -> list[ResourceHandle]:
        """Detect leaked resources."""
        now = time.time()
        leaks = []

        with self._lock:
            for handle in self._handles.values():
                age_seconds = now - handle.allocated_at
                if age_seconds > self._leak_threshold_seconds:
                    leaks.append(handle)

        return leaks

    def get_stats(self) -> dict[str, Any]:
        """Get resource statistics."""
        with self._lock:
            by_type: dict[str, int] = {}
            by_backend: dict[str, int] = {}
            total_bytes = 0

            for handle in self._handles.values():
                by_type[handle.resource_type] = by_type.get(handle.resource_type, 0) + 1
                by_backend[handle.backend] = by_backend.get(handle.backend, 0) + 1
                total_bytes += handle.size_bytes

            return {
                "total_handles": len(self._handles),
                "by_type": by_type,
                "by_backend": by_backend,
                "total_bytes": total_bytes,
                "detected_leaks": len(self.detect_leaks()),
            }


# ============================================================================
# MB-P2-016: Region Boundary Audit
# ============================================================================

@dataclass
class BoundaryViolation:
    """Detected boundary violation."""
    source_region_id: str
    target_region_id: str
    violation_type: Literal["schema_mismatch", "ordering_violated", "missing_data", "duplicate_data"]
    severity: Literal["warning", "error", "critical"]
    description: str
    detected_at: float


class RegionBoundaryAuditor:
    """Region boundary auditor (MB-P2-016)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._violations: list[BoundaryViolation] = []

    def audit_boundary(
        self,
        source_region_id: str,
        target_region_id: str,
        source_schema: dict[str, str],
        target_schema: dict[str, str],
        source_row_count: int,
        target_row_count: int,
        requires_sort: bool,
        is_sorted: bool,
    ):
        """Audit a region boundary transfer."""
        violations = []

        # Check schema compatibility
        if source_schema != target_schema:
            violations.append(BoundaryViolation(
                source_region_id=source_region_id,
                target_region_id=target_region_id,
                violation_type="schema_mismatch",
                severity="error",
                description=f"Schema mismatch: source={source_schema}, target={target_schema}",
                detected_at=time.time(),
            ))

        # Check row count consistency
        if source_row_count != target_row_count:
            severity = "critical" if abs(source_row_count - target_row_count) > source_row_count * 0.01 else "warning"
            violations.append(BoundaryViolation(
                source_region_id=source_region_id,
                target_region_id=target_region_id,
                violation_type="missing_data" if target_row_count < source_row_count else "duplicate_data",
                severity=severity,  # type: ignore
                description=f"Row count mismatch: source={source_row_count}, target={target_row_count}",
                detected_at=time.time(),
            ))

        # Check ordering
        if requires_sort and not is_sorted:
            violations.append(BoundaryViolation(
                source_region_id=source_region_id,
                target_region_id=target_region_id,
                violation_type="ordering_violated",
                severity="error",
                description="Required sort not applied",
                detected_at=time.time(),
            ))

        with self._lock:
            self._violations.extend(violations)

        return violations

    def get_violations(
        self,
        severity: Literal["warning", "error", "critical"] | None = None,
    ) -> list[BoundaryViolation]:
        """Get boundary violations."""
        with self._lock:
            if severity:
                return [v for v in self._violations if v.severity == severity]
            return list(self._violations)

    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        with self._lock:
            return any(v.severity == "critical" for v in self._violations)


# ============================================================================
# Global Governance Coordinator
# ============================================================================

class MultiBackendGovernor:
    """Unified governance coordinator for all P2 items."""

    def __init__(self):
        self.cache_governor = CacheGovernor()
        self.failure_retry_governor = FailureRetryGovernor()
        self.timeout_governor = RegionTimeoutGovernor(
            TimeoutPolicy(
                default_timeout_ms=30000,
                per_node_budget_ms=100,
                max_timeout_ms=300000,
                enable_adaptive=True,
                timeout_multiplier_on_retry=1.5,
            )
        )
        self.version_registry = BackendVersionRegistry()
        self.execution_logger = RegionExecutionLogger()
        self.health_monitor = BackendHealthMonitor()
        self.leak_detector = ResourceLeakDetector()
        self.boundary_auditor = RegionBoundaryAuditor()

    def get_full_report(self) -> dict[str, Any]:
        """Get comprehensive governance report."""
        return {
            "cache": self.cache_governor.get_stats(),
            "failures": self.failure_retry_governor.get_failure_stats(),
            "backend_versions": self.version_registry.get_composite_hash(),
            "execution_log_entries": len(self.execution_logger.get_log()),
            "resource_leaks": self.leak_detector.get_stats(),
            "boundary_violations": {
                "total": len(self.boundary_auditor.get_violations()),
                "critical": len(self.boundary_auditor.get_violations("critical")),
                "errors": len(self.boundary_auditor.get_violations("error")),
                "warnings": len(self.boundary_auditor.get_violations("warning")),
            },
        }


# Global singleton
_GLOBAL_GOVERNOR: MultiBackendGovernor | None = None


def get_multibackend_governor() -> MultiBackendGovernor:
    """Get global governance coordinator."""
    global _GLOBAL_GOVERNOR
    if _GLOBAL_GOVERNOR is None:
        _GLOBAL_GOVERNOR = MultiBackendGovernor()
    return _GLOBAL_GOVERNOR
