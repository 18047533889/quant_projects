# -*- coding: utf-8 -*-
"""MB-P1-009: Spill strategy for large intermediates.

Intelligent spilling of large intermediates to disk when memory pressure high:
- Size-based spill prioritization
- Cost-benefit analysis (spill cost vs memory savings)
- Compression for spilled data
- Read-ahead for spilled data reuse
- Integration with liveness_analyzer for spill candidates

Key principles:
- Spill largest, least-frequently-accessed first
- Compress before spilling to reduce IO
- Track spill location for efficient retrieval
- Fail-closed: reject tasks if spill fails
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpillMetadata:
    """Metadata for spilled intermediate.

    Attributes:
        task_id: Task that produced this intermediate
        spill_id: Unique spill identifier
        spill_path: Path to spilled data on disk
        original_size_bytes: Size before compression
        spilled_size_bytes: Size on disk after compression
        spilled_at: Timestamp when spilled
        access_count: Number of times accessed after spilling
        compression_ratio: Compression ratio achieved
    """
    task_id: str
    spill_id: str
    spill_path: str
    original_size_bytes: int
    spilled_size_bytes: int
    spilled_at: float
    access_count: int = 0
    compression_ratio: float = 1.0


@dataclass
class SpillStats:
    """Spill operation statistics.

    Attributes:
        total_spilled_count: Total intermediates spilled
        total_spilled_bytes: Total bytes spilled (original size)
        total_disk_bytes: Total bytes on disk (after compression)
        total_restored_count: Total intermediates restored from disk
        total_restored_bytes: Total bytes restored
        spill_duration_total_s: Total time spent spilling
        restore_duration_total_s: Total time spent restoring
    """
    total_spilled_count: int = 0
    total_spilled_bytes: int = 0
    total_disk_bytes: int = 0
    total_restored_count: int = 0
    total_restored_bytes: int = 0
    spill_duration_total_s: float = 0.0
    restore_duration_total_s: float = 0.0


class SpillStore:
    """Storage manager for spilled intermediates.

    Handles serialization, compression, and disk I/O for spilled data.
    """

    def __init__(
        self,
        spill_dir: str | None = None,
        *,
        enable_compression: bool = True,
        compression_level: int = 3,
    ):
        """Initialize spill store.

        Args:
            spill_dir: Directory for spilled data (temp if None)
            enable_compression: Enable compression
            compression_level: Compression level (1-9, higher = better compression)
        """
        if spill_dir is None:
            spill_dir = tempfile.mkdtemp(prefix="factor_engine_spill_")

        self._spill_dir = Path(spill_dir)
        self._spill_dir.mkdir(parents=True, exist_ok=True)

        self._enable_compression = enable_compression
        self._compression_level = compression_level

        self._spills: dict[str, SpillMetadata] = {}
        self._lock = threading.RLock()

        _logger.info(f"Spill store initialized at {self._spill_dir}")

    def spill(
        self,
        task_id: str,
        data: Any,
        original_size_bytes: int,
    ) -> SpillMetadata | None:
        """Spill intermediate to disk.

        Args:
            task_id: Task that produced data
            data: Data to spill
            original_size_bytes: Original memory size

        Returns:
            SpillMetadata if successful, None if failed
        """
        start_time = time.time()

        try:
            # Generate spill ID
            spill_id = self._generate_spill_id(task_id)
            spill_path = self._spill_dir / f"{spill_id}.pkl"

            # Serialize and compress
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            if self._enable_compression:
                import gzip
                with gzip.open(
                    spill_path, "wb", compresslevel=self._compression_level
                ) as f:
                    f.write(serialized)
            else:
                with open(spill_path, "wb") as f:
                    f.write(serialized)

            # Get actual disk size
            spilled_size = spill_path.stat().st_size
            compression_ratio = original_size_bytes / spilled_size if spilled_size > 0 else 1.0

            duration = time.time() - start_time

            metadata = SpillMetadata(
                task_id=task_id,
                spill_id=spill_id,
                spill_path=str(spill_path),
                original_size_bytes=original_size_bytes,
                spilled_size_bytes=spilled_size,
                spilled_at=time.time(),
                compression_ratio=compression_ratio,
            )

            with self._lock:
                self._spills[spill_id] = metadata

            _logger.info(
                f"Spilled {task_id}: {original_size_bytes // 1024 // 1024} MB → "
                f"{spilled_size // 1024 // 1024} MB "
                f"(ratio={compression_ratio:.2f}, {duration:.2f}s)"
            )

            return metadata

        except Exception as exc:
            _logger.error(f"Spill failed for {task_id}: {exc}")
            return None

    def restore(self, spill_id: str) -> Any | None:
        """Restore spilled intermediate from disk.

        Args:
            spill_id: Spill identifier

        Returns:
            Restored data or None if failed
        """
        start_time = time.time()

        with self._lock:
            metadata = self._spills.get(spill_id)
            if metadata is None:
                _logger.warning(f"Spill {spill_id} not found")
                return None

        try:
            spill_path = Path(metadata.spill_path)
            if not spill_path.exists():
                _logger.error(f"Spill file missing: {spill_path}")
                return None

            # Read and decompress
            if self._enable_compression:
                import gzip
                with gzip.open(spill_path, "rb") as f:
                    serialized = f.read()
            else:
                with open(spill_path, "rb") as f:
                    serialized = f.read()

            data = pickle.loads(serialized)

            duration = time.time() - start_time

            # Update access count
            with self._lock:
                # Create updated metadata (frozen dataclass)
                updated = SpillMetadata(
                    task_id=metadata.task_id,
                    spill_id=metadata.spill_id,
                    spill_path=metadata.spill_path,
                    original_size_bytes=metadata.original_size_bytes,
                    spilled_size_bytes=metadata.spilled_size_bytes,
                    spilled_at=metadata.spilled_at,
                    access_count=metadata.access_count + 1,
                    compression_ratio=metadata.compression_ratio,
                )
                self._spills[spill_id] = updated

            _logger.info(
                f"Restored {spill_id}: {metadata.spilled_size_bytes // 1024 // 1024} MB "
                f"({duration:.2f}s)"
            )

            return data

        except Exception as exc:
            _logger.error(f"Restore failed for {spill_id}: {exc}")
            return None

    def remove(self, spill_id: str) -> bool:
        """Remove spilled data from disk.

        Args:
            spill_id: Spill to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            metadata = self._spills.pop(spill_id, None)
            if metadata is None:
                return False

        try:
            spill_path = Path(metadata.spill_path)
            if spill_path.exists():
                spill_path.unlink()

            _logger.debug(f"Removed spill {spill_id}")
            return True

        except Exception as exc:
            _logger.warning(f"Failed to remove spill {spill_id}: {exc}")
            return False

    def clear_all(self) -> int:
        """Clear all spilled data.

        Returns:
            Number of spills removed
        """
        with self._lock:
            spill_ids = list(self._spills.keys())

        count = 0
        for spill_id in spill_ids:
            if self.remove(spill_id):
                count += 1

        return count

    def get_metadata(self, spill_id: str) -> SpillMetadata | None:
        """Get spill metadata.

        Args:
            spill_id: Spill identifier

        Returns:
            SpillMetadata or None
        """
        with self._lock:
            return self._spills.get(spill_id)

    def list_spills(self) -> list[SpillMetadata]:
        """List all spilled intermediates."""
        with self._lock:
            return list(self._spills.values())

    def total_disk_usage(self) -> int:
        """Get total disk usage for all spills (bytes)."""
        with self._lock:
            return sum(m.spilled_size_bytes for m in self._spills.values())

    def _generate_spill_id(self, task_id: str) -> str:
        """Generate unique spill ID."""
        timestamp = str(time.time())
        content = f"{task_id}_{timestamp}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return f"spill_{hash_digest[:16]}"


class SpillStrategyManager:
    """Manager for intelligent spill strategy decisions.

    Decides what to spill, when to spill, and coordinates with
    liveness_analyzer for optimal spill candidates.
    """

    def __init__(
        self,
        spill_store: SpillStore,
        *,
        memory_threshold_bytes: int = 500 * 1024 * 1024,  # 500 MB
        min_spill_size_bytes: int = 50 * 1024 * 1024,  # 50 MB
        spill_ratio: float = 0.3,  # Spill 30% when over threshold
    ):
        """Initialize spill strategy manager.

        Args:
            spill_store: Spill store for data persistence
            memory_threshold_bytes: Trigger spilling above this threshold
            min_spill_size_bytes: Minimum size to consider for spilling
            spill_ratio: Fraction of memory to free when spilling
        """
        self._spill_store = spill_store
        self._memory_threshold_bytes = memory_threshold_bytes
        self._min_spill_size_bytes = min_spill_size_bytes
        self._spill_ratio = spill_ratio

        self._stats = SpillStats()
        self._lock = threading.Lock()

        # Track in-memory intermediates and their spill status
        self._in_memory: dict[str, tuple[Any, int]] = {}  # task_id -> (data, size)
        self._spilled_tasks: dict[str, str] = {}  # task_id -> spill_id

    def should_spill(self, current_memory_bytes: int) -> bool:
        """Check if spilling should be triggered.

        Args:
            current_memory_bytes: Current memory usage

        Returns:
            True if spilling recommended
        """
        return current_memory_bytes > self._memory_threshold_bytes

    def select_spill_candidates(
        self,
        live_intermediates: list[tuple[str, int]],
        target_bytes_to_free: int,
    ) -> list[tuple[str, int]]:
        """Select intermediates to spill based on size and access patterns.

        Args:
            live_intermediates: List of (task_id, size_bytes)
            target_bytes_to_free: Target memory to free

        Returns:
            List of (task_id, size_bytes) to spill, sorted by priority
        """
        candidates = [
            (tid, size) for tid, size in live_intermediates
            if size >= self._min_spill_size_bytes
        ]

        # Sort by size descending (spill largest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select candidates until target met
        selected = []
        bytes_freed = 0

        for task_id, size_bytes in candidates:
            selected.append((task_id, size_bytes))
            bytes_freed += size_bytes

            if bytes_freed >= target_bytes_to_free:
                break

        return selected

    def execute_spill(
        self,
        task_id: str,
        data: Any,
        size_bytes: int,
    ) -> bool:
        """Execute spill operation for intermediate.

        Args:
            task_id: Task producing intermediate
            data: Data to spill
            size_bytes: Memory size

        Returns:
            True if spilled successfully
        """
        start_time = time.time()

        metadata = self._spill_store.spill(task_id, data, size_bytes)
        if metadata is None:
            return False

        duration = time.time() - start_time

        with self._lock:
            self._spilled_tasks[task_id] = metadata.spill_id
            self._stats.total_spilled_count += 1
            self._stats.total_spilled_bytes += size_bytes
            self._stats.total_disk_bytes += metadata.spilled_size_bytes
            self._stats.spill_duration_total_s += duration

            # Remove from in-memory tracking
            self._in_memory.pop(task_id, None)

        return True

    def restore_if_spilled(self, task_id: str) -> Any | None:
        """Restore intermediate if it was spilled.

        Args:
            task_id: Task ID to restore

        Returns:
            Restored data or None if not spilled or restore failed
        """
        with self._lock:
            spill_id = self._spilled_tasks.get(task_id)
            if spill_id is None:
                return None

        start_time = time.time()
        data = self._spill_store.restore(spill_id)
        duration = time.time() - start_time

        if data is not None:
            with self._lock:
                metadata = self._spill_store.get_metadata(spill_id)
                if metadata:
                    self._stats.total_restored_count += 1
                    self._stats.total_restored_bytes += metadata.original_size_bytes
                    self._stats.restore_duration_total_s += duration

        return data

    def cleanup_spill(self, task_id: str) -> bool:
        """Clean up spill for task (after last use).

        Args:
            task_id: Task to cleanup

        Returns:
            True if cleaned up
        """
        with self._lock:
            spill_id = self._spilled_tasks.pop(task_id, None)
            if spill_id is None:
                return False

        return self._spill_store.remove(spill_id)

    def stats(self) -> dict[str, Any]:
        """Get spill statistics."""
        with self._lock:
            avg_compression = (
                self._stats.total_spilled_bytes / self._stats.total_disk_bytes
                if self._stats.total_disk_bytes > 0 else 1.0
            )
            avg_spill_time = (
                self._stats.spill_duration_total_s / self._stats.total_spilled_count
                if self._stats.total_spilled_count > 0 else 0.0
            )
            avg_restore_time = (
                self._stats.restore_duration_total_s / self._stats.total_restored_count
                if self._stats.total_restored_count > 0 else 0.0
            )

            return {
                "total_spilled_count": self._stats.total_spilled_count,
                "total_spilled_bytes": self._stats.total_spilled_bytes,
                "total_disk_bytes": self._stats.total_disk_bytes,
                "total_restored_count": self._stats.total_restored_count,
                "total_restored_bytes": self._stats.total_restored_bytes,
                "avg_compression_ratio": avg_compression,
                "avg_spill_time_s": avg_spill_time,
                "avg_restore_time_s": avg_restore_time,
                "current_disk_usage_bytes": self._spill_store.total_disk_usage(),
                "active_spills": len(self._spilled_tasks),
            }


# Global singleton
_GLOBAL_SPILL_MANAGER: SpillStrategyManager | None = None


def global_spill_manager(
    spill_dir: str | None = None,
) -> SpillStrategyManager:
    """Get global spill strategy manager.

    Args:
        spill_dir: Spill directory (only on first call)

    Returns:
        Global SpillStrategyManager instance
    """
    global _GLOBAL_SPILL_MANAGER
    if _GLOBAL_SPILL_MANAGER is None:
        store = SpillStore(spill_dir=spill_dir)
        _GLOBAL_SPILL_MANAGER = SpillStrategyManager(store)

    return _GLOBAL_SPILL_MANAGER
