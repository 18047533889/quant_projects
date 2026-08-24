"""
R32-P0-091: Real CSE evidence from backend actual counters.

Prohibits planner-inferred `source_group_count == physical_scan_count`
masquerading as fact. Must record actual backend operations.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendCounters:
    """Actual backend operation counters - not planner estimates.

    R32-P0-091: Real 1000/10000 factors CSE evidence must come from
    backend actual counters, not planner inference.
    """

    # Actual scan operations
    scan_invocations: int = 0
    object_opens: int = 0
    bytes_read: int = 0
    remote_requests: int = 0

    # Source block operations
    source_block_producers: int = 0
    source_block_consumers: int = 0
    source_block_cache_hits: int = 0
    source_block_cache_misses: int = 0

    # CSE evidence
    unique_source_blocks: int = 0
    reused_source_blocks: int = 0

    # Physical operations
    physical_scans: int = 0
    logical_requests: int = 0

    def cse_ratio(self) -> float:
        """Actual CSE effectiveness: reuse / total."""
        total = self.source_block_producers + self.reused_source_blocks
        if total == 0:
            return 0.0
        return self.reused_source_blocks / total

    def deduplication_factor(self) -> float:
        """How many logical requests shared physical scans."""
        if self.physical_scans == 0:
            return 1.0
        return self.logical_requests / self.physical_scans

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_invocations": self.scan_invocations,
            "object_opens": self.object_opens,
            "bytes_read": self.bytes_read,
            "remote_requests": self.remote_requests,
            "source_block_producers": self.source_block_producers,
            "source_block_consumers": self.source_block_consumers,
            "source_block_cache_hits": self.source_block_cache_hits,
            "source_block_cache_misses": self.source_block_cache_misses,
            "unique_source_blocks": self.unique_source_blocks,
            "reused_source_blocks": self.reused_source_blocks,
            "physical_scans": self.physical_scans,
            "logical_requests": self.logical_requests,
            "cse_ratio": self.cse_ratio(),
            "deduplication_factor": self.deduplication_factor(),
        }


@dataclass
class CSEEvidence:
    """CSE evidence with actual backend proof.

    R32-P0-091: This is authoritative CSE evidence, not planner estimate.
    """

    experiment_id: str
    num_factors: int
    counters: BackendCounters
    proof_source: str = "backend_actual"  # Never "planner_inferred"

    def validate(self) -> None:
        """Ensure this is real evidence, not planner estimate."""
        if self.proof_source == "planner_inferred":
            raise ValueError(
                "R32-P0-091 violation: CSE evidence must come from "
                "backend actual counters, not planner inference"
            )

        if self.counters.physical_scans == 0 and self.num_factors > 0:
            raise ValueError(
                "Invalid CSE evidence: factors > 0 but physical_scans == 0. "
                "This suggests planner estimate, not real backend counters."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "num_factors": self.num_factors,
            "proof_source": self.proof_source,
            "counters": self.counters.to_dict(),
        }


class BackendCounterRegistry:
    """Thread-safe registry for backend operation counters.

    R32-P0-091: Central authority for actual backend counter collection.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, BackendCounters] = {}

    def get_or_create(self, scope_id: str) -> BackendCounters:
        """Get or create counter for given scope (query/batch/session)."""
        with self._lock:
            if scope_id not in self._counters:
                self._counters[scope_id] = BackendCounters()
            return self._counters[scope_id]

    def record_scan(self, scope_id: str) -> None:
        """Record actual scan invocation."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.scan_invocations += 1
            counter.physical_scans += 1

    def record_object_open(self, scope_id: str, num_bytes: int = 0) -> None:
        """Record actual object open."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.object_opens += 1
            counter.bytes_read += num_bytes

    def record_remote_request(self, scope_id: str) -> None:
        """Record actual remote request."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.remote_requests += 1

    def record_source_block_produce(self, scope_id: str) -> None:
        """Record source block creation (cache miss)."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.source_block_producers += 1
            counter.source_block_cache_misses += 1
            counter.unique_source_blocks += 1

    def record_source_block_consume(self, scope_id: str, *, from_cache: bool) -> None:
        """Record source block consumption."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.source_block_consumers += 1
            if from_cache:
                counter.source_block_cache_hits += 1
                counter.reused_source_blocks += 1

    def record_logical_request(self, scope_id: str) -> None:
        """Record logical data request (factor/dataset read)."""
        counter = self.get_or_create(scope_id)
        with self._lock:
            counter.logical_requests += 1

    def get_snapshot(self, scope_id: str) -> BackendCounters:
        """Get immutable snapshot of counters for scope."""
        with self._lock:
            if scope_id not in self._counters:
                return BackendCounters()
            c = self._counters[scope_id]
            return BackendCounters(
                scan_invocations=c.scan_invocations,
                object_opens=c.object_opens,
                bytes_read=c.bytes_read,
                remote_requests=c.remote_requests,
                source_block_producers=c.source_block_producers,
                source_block_consumers=c.source_block_consumers,
                source_block_cache_hits=c.source_block_cache_hits,
                source_block_cache_misses=c.source_block_cache_misses,
                unique_source_blocks=c.unique_source_blocks,
                reused_source_blocks=c.reused_source_blocks,
                physical_scans=c.physical_scans,
                logical_requests=c.logical_requests,
            )

    def clear(self, scope_id: str) -> None:
        """Clear counters for scope."""
        with self._lock:
            self._counters.pop(scope_id, None)


# Global registry singleton
_global_registry = BackendCounterRegistry()


def get_backend_counter_registry() -> BackendCounterRegistry:
    """Get global backend counter registry."""
    return _global_registry


def create_cse_evidence(
    experiment_id: str, scope_id: str, num_factors: int
) -> CSEEvidence:
    """Create CSE evidence from actual backend counters.

    R32-P0-091: This is the only valid way to create CSE evidence.
    Never accept planner-inferred estimates.
    """
    registry = get_backend_counter_registry()
    counters = registry.get_snapshot(scope_id)

    evidence = CSEEvidence(
        experiment_id=experiment_id,
        num_factors=num_factors,
        counters=counters,
        proof_source="backend_actual",
    )
    evidence.validate()
    return evidence


__all__ = [
    "BackendCounters",
    "CSEEvidence",
    "BackendCounterRegistry",
    "get_backend_counter_registry",
    "create_cse_evidence",
]
