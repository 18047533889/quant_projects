# -*- coding: utf-8 -*-
"""MB-P1-008: Concurrent region memory isolation.

Isolates memory usage between concurrent execution regions to prevent
interference and enable accurate memory accounting:
- Per-region memory pools
- Cross-region isolation enforcement
- Region-level memory limits
- Spill coordination across regions

Key principles:
- Strict isolation: regions cannot share memory
- Fair allocation: each region gets fair share
- Independent lifecycle: region cleanup is isolated
- Coordinated spilling: global spill policy across regions
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class RegionMemoryStats:
    """Memory statistics for execution region.

    Attributes:
        region_id: Region identifier
        allocated_bytes: Current allocation
        peak_bytes: Peak allocation observed
        limit_bytes: Region memory limit
        spilled_bytes: Bytes spilled to disk
        allocation_count: Number of allocations
        spill_count: Number of spill operations
    """
    region_id: str
    allocated_bytes: int = 0
    peak_bytes: int = 0
    limit_bytes: int = 0
    spilled_bytes: int = 0
    allocation_count: int = 0
    spill_count: int = 0


@dataclass
class RegionAllocation:
    """A memory allocation within a region.

    Attributes:
        allocation_id: Unique allocation identifier
        region_id: Owning region
        bytes_allocated: Size of allocation
        purpose: Allocation purpose (task_id, buffer, etc)
        is_spillable: Whether this allocation can be spilled
        allocated_at: Timestamp of allocation
    """
    allocation_id: str
    region_id: str
    bytes_allocated: int
    purpose: str
    is_spillable: bool
    allocated_at: float


class ConcurrentRegionMemoryPool:
    """Memory pool for single concurrent execution region.

    Provides isolated memory allocation and tracking for one region.
    """

    def __init__(
        self,
        region_id: str,
        limit_bytes: int,
        *,
        enable_spill: bool = True,
    ):
        """Initialize region memory pool.

        Args:
            region_id: Region identifier
            limit_bytes: Memory limit for this region
            enable_spill: Enable spilling when limit reached
        """
        self._region_id = region_id
        self._limit_bytes = limit_bytes
        self._enable_spill = enable_spill

        self._stats = RegionMemoryStats(
            region_id=region_id,
            limit_bytes=limit_bytes,
        )

        self._allocations: dict[str, RegionAllocation] = {}
        self._allocation_counter = 0
        self._lock = threading.RLock()

    def allocate(
        self,
        bytes_required: int,
        *,
        purpose: str = "unknown",
        is_spillable: bool = True,
    ) -> str | None:
        """Allocate memory in this region.

        Args:
            bytes_required: Bytes to allocate
            purpose: Purpose description
            is_spillable: Whether allocation can be spilled

        Returns:
            Allocation ID if successful, None if limit exceeded
        """
        with self._lock:
            # Check limit
            projected = self._stats.allocated_bytes + bytes_required
            if projected > self._limit_bytes:
                _logger.debug(
                    f"Region {self._region_id} allocation rejected: "
                    f"{projected // 1024 // 1024} MB > "
                    f"{self._limit_bytes // 1024 // 1024} MB limit"
                )
                return None

            # Create allocation
            self._allocation_counter += 1
            allocation_id = f"{self._region_id}_alloc_{self._allocation_counter}"

            import time
            allocation = RegionAllocation(
                allocation_id=allocation_id,
                region_id=self._region_id,
                bytes_allocated=bytes_required,
                purpose=purpose,
                is_spillable=is_spillable,
                allocated_at=time.time(),
            )

            self._allocations[allocation_id] = allocation
            self._stats.allocated_bytes += bytes_required
            self._stats.allocation_count += 1

            if self._stats.allocated_bytes > self._stats.peak_bytes:
                self._stats.peak_bytes = self._stats.allocated_bytes

            _logger.debug(
                f"Region {self._region_id} allocated {bytes_required // 1024} KB "
                f"for {purpose}, total={self._stats.allocated_bytes // 1024 // 1024} MB"
            )

            return allocation_id

    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate memory in this region.

        Args:
            allocation_id: Allocation to release

        Returns:
            True if deallocated, False if not found
        """
        with self._lock:
            allocation = self._allocations.pop(allocation_id, None)
            if allocation is None:
                return False

            self._stats.allocated_bytes -= allocation.bytes_allocated

            _logger.debug(
                f"Region {self._region_id} deallocated {allocation.bytes_allocated // 1024} KB "
                f"from {allocation.purpose}"
            )

            return True

    def get_spillable_allocations(
        self, min_bytes: int = 0
    ) -> list[RegionAllocation]:
        """Get spillable allocations sorted by size descending.

        Args:
            min_bytes: Minimum allocation size to consider

        Returns:
            List of spillable allocations
        """
        with self._lock:
            spillable = [
                alloc for alloc in self._allocations.values()
                if alloc.is_spillable and alloc.bytes_allocated >= min_bytes
            ]
            spillable.sort(key=lambda a: a.bytes_allocated, reverse=True)
            return spillable

    def mark_spilled(self, allocation_id: str) -> bool:
        """Mark allocation as spilled (frees memory but keeps tracking).

        Args:
            allocation_id: Allocation that was spilled

        Returns:
            True if marked, False if not found
        """
        with self._lock:
            allocation = self._allocations.get(allocation_id)
            if allocation is None:
                return False

            self._stats.allocated_bytes -= allocation.bytes_allocated
            self._stats.spilled_bytes += allocation.bytes_allocated
            self._stats.spill_count += 1

            # Keep allocation record but mark as spilled
            allocation.is_spillable = False

            _logger.info(
                f"Region {self._region_id} spilled {allocation.bytes_allocated // 1024} KB "
                f"from {allocation.purpose}"
            )

            return True

    def stats(self) -> RegionMemoryStats:
        """Get region memory statistics."""
        with self._lock:
            return RegionMemoryStats(
                region_id=self._stats.region_id,
                allocated_bytes=self._stats.allocated_bytes,
                peak_bytes=self._stats.peak_bytes,
                limit_bytes=self._stats.limit_bytes,
                spilled_bytes=self._stats.spilled_bytes,
                allocation_count=self._stats.allocation_count,
                spill_count=self._stats.spill_count,
            )

    def utilization(self) -> float:
        """Get memory utilization fraction [0.0, 1.0]."""
        with self._lock:
            if self._limit_bytes == 0:
                return 0.0
            return self._stats.allocated_bytes / self._limit_bytes

    def clear(self) -> None:
        """Clear all allocations (for cleanup)."""
        with self._lock:
            self._allocations.clear()
            self._stats.allocated_bytes = 0


class ConcurrentRegionIsolationManager:
    """Manager for memory isolation across concurrent execution regions.

    Creates and manages isolated memory pools for concurrent DAG regions,
    coordinating spill operations and enforcing fair allocation.
    """

    def __init__(
        self,
        total_memory_bytes: int,
        *,
        enable_spill: bool = True,
        spill_threshold: float = 0.8,
    ):
        """Initialize region isolation manager.

        Args:
            total_memory_bytes: Total memory available for all regions
            enable_spill: Enable spilling when regions exceed limits
            spill_threshold: Utilization threshold to trigger spilling
        """
        self._total_memory_bytes = total_memory_bytes
        self._enable_spill = enable_spill
        self._spill_threshold = spill_threshold

        self._regions: dict[str, ConcurrentRegionMemoryPool] = {}
        self._lock = threading.RLock()

        self._total_allocated_bytes = 0
        self._global_peak_bytes = 0

    def create_region(
        self,
        region_id: str,
        *,
        limit_bytes: int | None = None,
    ) -> ConcurrentRegionMemoryPool:
        """Create isolated memory region.

        Args:
            region_id: Unique region identifier
            limit_bytes: Optional specific limit (defaults to fair share)

        Returns:
            ConcurrentRegionMemoryPool for this region
        """
        with self._lock:
            if region_id in self._regions:
                _logger.warning(f"Region {region_id} already exists")
                return self._regions[region_id]

            # Calculate limit
            if limit_bytes is None:
                # Fair share: total / (current_regions + 1)
                region_count = len(self._regions) + 1
                fair_share = self._total_memory_bytes // region_count

                # Do not oversell: a default limit must fit within the
                # remaining budget after existing regions have claimed
                # their share. Explicit caller limits are left untouched.
                remaining_budget = self._total_memory_bytes - sum(
                    p._limit_bytes for p in self._regions.values()
                )
                limit_bytes = min(fair_share, max(remaining_budget, 0))

            pool = ConcurrentRegionMemoryPool(
                region_id=region_id,
                limit_bytes=limit_bytes,
                enable_spill=self._enable_spill,
            )

            self._regions[region_id] = pool

            _logger.info(
                f"Created region {region_id} with "
                f"{limit_bytes // 1024 // 1024} MB limit"
            )

            return pool

    def destroy_region(self, region_id: str) -> bool:
        """Destroy region and release resources.

        Args:
            region_id: Region to destroy

        Returns:
            True if destroyed, False if not found
        """
        with self._lock:
            pool = self._regions.pop(region_id, None)
            if pool is None:
                return False

            # Update total allocated
            stats = pool.stats()
            self._total_allocated_bytes -= stats.allocated_bytes

            # Clear pool
            pool.clear()

            _logger.info(f"Destroyed region {region_id}")

            return True

    def get_region(self, region_id: str) -> ConcurrentRegionMemoryPool | None:
        """Get memory pool for region.

        Args:
            region_id: Region identifier

        Returns:
            Memory pool or None if not found
        """
        with self._lock:
            return self._regions.get(region_id)

    def rebalance_limits(self) -> None:
        """Rebalance memory limits across active regions (fair share).

        Updates both the pool's private limit and the public
        ``RegionMemoryStats.limit_bytes`` so accounting stays consistent.
        """
        with self._lock:
            if not self._regions:
                return

            fair_share = self._total_memory_bytes // len(self._regions)

            for pool in self._regions.values():
                with pool._lock:
                    pool._limit_bytes = fair_share
                    pool._stats.limit_bytes = fair_share

            _logger.info(
                f"Rebalanced {len(self._regions)} regions to "
                f"{fair_share // 1024 // 1024} MB each"
            )

    def global_spill_if_needed(self) -> int:
        """Check all regions and trigger spilling if any exceed threshold.

        Returns:
            Number of allocations spilled
        """
        with self._lock:
            total_spilled = 0

            for pool in self._regions.values():
                utilization = pool.utilization()
                if utilization > self._spill_threshold:
                    # Region over threshold, spill largest allocations
                    spillable = pool.get_spillable_allocations(
                        min_bytes=10 * 1024 * 1024  # Min 10 MB
                    )

                    # Spill until below threshold
                    target_bytes = int(pool._limit_bytes * self._spill_threshold * 0.9)
                    current_bytes = pool.stats().allocated_bytes

                    for allocation in spillable:
                        if current_bytes <= target_bytes:
                            break

                        if pool.mark_spilled(allocation.allocation_id):
                            current_bytes -= allocation.bytes_allocated
                            total_spilled += 1

            if total_spilled > 0:
                _logger.info(f"Global spill reclaimed {total_spilled} allocations")

            return total_spilled

    def global_stats(self) -> dict[str, Any]:
        """Get global statistics across all regions."""
        with self._lock:
            region_stats = {
                region_id: pool.stats()
                for region_id, pool in self._regions.items()
            }

            total_allocated = sum(s.allocated_bytes for s in region_stats.values())
            total_peak = sum(s.peak_bytes for s in region_stats.values())
            total_spilled = sum(s.spilled_bytes for s in region_stats.values())

            return {
                "region_count": len(self._regions),
                "total_memory_bytes": self._total_memory_bytes,
                "total_allocated_bytes": total_allocated,
                "total_peak_bytes": total_peak,
                "total_spilled_bytes": total_spilled,
                "global_utilization": total_allocated / self._total_memory_bytes
                if self._total_memory_bytes > 0 else 0.0,
                "regions": {
                    rid: {
                        "allocated_bytes": s.allocated_bytes,
                        "peak_bytes": s.peak_bytes,
                        "limit_bytes": s.limit_bytes,
                        "utilization": s.allocated_bytes / s.limit_bytes
                        if s.limit_bytes > 0 else 0.0,
                        "spilled_bytes": s.spilled_bytes,
                    }
                    for rid, s in region_stats.items()
                },
            }


# Global singleton
_GLOBAL_REGION_MANAGER: ConcurrentRegionIsolationManager | None = None


def global_region_manager(
    total_memory_bytes: int | None = None,
) -> ConcurrentRegionIsolationManager:
    """Get global concurrent region isolation manager.

    Args:
        total_memory_bytes: Total memory (only on first call)

    Returns:
        Global ConcurrentRegionIsolationManager instance
    """
    global _GLOBAL_REGION_MANAGER
    if _GLOBAL_REGION_MANAGER is None:
        if total_memory_bytes is None:
            # Default: 80% of available memory
            try:
                import psutil
                total_memory_bytes = int(psutil.virtual_memory().available * 0.8)
            except Exception:
                total_memory_bytes = 4 * 1024**3  # 4 GB fallback

        _GLOBAL_REGION_MANAGER = ConcurrentRegionIsolationManager(total_memory_bytes)

    return _GLOBAL_REGION_MANAGER
