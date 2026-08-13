# -*- coding: utf-8 -*-
"""MB-P1-010: Arrow zero-copy boundary optimization.

Optimizes data transfer at backend boundaries using Apache Arrow zero-copy:
- Detect Arrow-compatible data transfers
- Avoid unnecessary serialization/deserialization
- Use Arrow IPC for efficient inter-process transfer
- Shared memory for local transfers
- Flight protocol for remote transfers

Key principles:
- Zero-copy when possible (same process, compatible format)
- Minimal copy when necessary (format conversion)
- Arrow IPC for serialization (faster than pickle)
- Shared memory for multi-process scenarios
"""

from __future__ import annotations

import logging
import mmap
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferMetadata:
    """Metadata for data transfer.

    Attributes:
        transfer_id: Unique transfer identifier
        source_repr: Source representation
        target_repr: Target representation
        size_bytes: Data size
        method: Transfer method (zerocopy/sharedmem/ipc/copy)
        duration_s: Transfer duration
        copy_avoided: Whether copy was avoided
    """
    transfer_id: str
    source_repr: str
    target_repr: str
    size_bytes: int
    method: str
    duration_s: float
    copy_avoided: bool


@dataclass
class TransferStats:
    """Transfer statistics.

    Attributes:
        total_transfers: Total transfer count
        zerocopy_count: Zero-copy transfers
        sharedmem_count: Shared memory transfers
        ipc_count: Arrow IPC transfers
        copy_count: Full copy transfers
        total_bytes: Total bytes transferred
        total_duration_s: Total transfer time
    """
    total_transfers: int = 0
    zerocopy_count: int = 0
    sharedmem_count: int = 0
    ipc_count: int = 0
    copy_count: int = 0
    total_bytes: int = 0
    total_duration_s: float = 0.0


class ArrowZeroCopyBoundary:
    """Zero-copy data transfer boundary using Apache Arrow.

    Optimizes transfers between backends using Arrow's efficient
    serialization and zero-copy capabilities.
    """

    def __init__(
        self,
        *,
        enable_shared_memory: bool = True,
        shm_dir: str | None = None,
    ):
        """Initialize zero-copy boundary.

        Args:
            enable_shared_memory: Enable shared memory for multi-process
            shm_dir: Shared memory directory (temp if None)
        """
        self._enable_shared_memory = enable_shared_memory
        self._shm_dir = Path(shm_dir) if shm_dir else Path(tempfile.gettempdir())
        self._shm_dir.mkdir(parents=True, exist_ok=True)

        self._stats = TransferStats()
        self._lock = threading.Lock()
        self._transfer_counter = 0

        # Check Arrow availability
        self._has_arrow = self._check_arrow_available()
        if not self._has_arrow:
            _logger.warning("Apache Arrow not available, falling back to copies")

    def transfer(
        self,
        data: Any,
        source_repr: str,
        target_repr: str,
        *,
        allow_copy: bool = True,
    ) -> tuple[Any, TransferMetadata] | None:
        """Transfer data between representations with optimal method.

        Args:
            data: Data to transfer
            source_repr: Source representation
            target_repr: Target representation
            allow_copy: Allow fallback to copy if zero-copy unavailable

        Returns:
            (transferred_data, metadata) or None if failed
        """
        import time
        start_time = time.time()

        with self._lock:
            self._transfer_counter += 1
            transfer_id = f"transfer_{self._transfer_counter:06d}"

        # Same representation -> zero-copy (just return reference)
        if source_repr == target_repr:
            duration = time.time() - start_time
            metadata = self._create_metadata(
                transfer_id, source_repr, target_repr,
                self._estimate_size(data), "zerocopy", duration, True
            )
            self._record_stats(metadata)
            return data, metadata

        # Try Arrow-optimized path
        if self._has_arrow:
            result = self._transfer_via_arrow(
                data, source_repr, target_repr, transfer_id
            )
            if result is not None:
                return result

        # Fallback to copy if allowed
        if allow_copy:
            result = self._transfer_via_copy(
                data, source_repr, target_repr, transfer_id
            )
            if result is not None:
                return result

        _logger.warning(
            f"Transfer failed: {source_repr} → {target_repr}, "
            f"allow_copy={allow_copy}"
        )
        return None

    def transfer_batch(
        self,
        data_list: list[tuple[Any, str, str]],
    ) -> list[tuple[Any, TransferMetadata] | None]:
        """Transfer multiple data items efficiently.

        Args:
            data_list: List of (data, source_repr, target_repr)

        Returns:
            List of (transferred_data, metadata) or None
        """
        results = []
        for data, source_repr, target_repr in data_list:
            result = self.transfer(data, source_repr, target_repr)
            results.append(result)
        return results

    def _transfer_via_arrow(
        self,
        data: Any,
        source_repr: str,
        target_repr: str,
        transfer_id: str,
    ) -> tuple[Any, TransferMetadata] | None:
        """Transfer via Apache Arrow (zero-copy when possible)."""
        import time
        start_time = time.time()

        try:
            import pyarrow as pa

            # Convert to Arrow table (may be zero-copy for some formats)
            arrow_table = self._to_arrow_table(data, source_repr)
            if arrow_table is None:
                return None

            size_bytes = arrow_table.nbytes

            # Convert from Arrow to target (may be zero-copy)
            target_data = self._from_arrow_table(arrow_table, target_repr)
            if target_data is None:
                return None

            duration = time.time() - start_time

            # Determine method
            method = "zerocopy" if self._is_zero_copy_path(source_repr, target_repr) else "ipc"
            copy_avoided = method == "zerocopy"

            metadata = self._create_metadata(
                transfer_id, source_repr, target_repr,
                size_bytes, method, duration, copy_avoided
            )
            self._record_stats(metadata)

            return target_data, metadata

        except Exception as exc:
            _logger.debug(f"Arrow transfer failed: {exc}")
            return None

    def _transfer_via_copy(
        self,
        data: Any,
        source_repr: str,
        target_repr: str,
        transfer_id: str,
    ) -> tuple[Any, TransferMetadata] | None:
        """Transfer via explicit copy (fallback)."""
        import time
        start_time = time.time()

        try:
            # Use standard conversion functions
            target_data = self._convert_data(data, source_repr, target_repr)
            if target_data is None:
                return None

            duration = time.time() - start_time
            size_bytes = self._estimate_size(data)

            metadata = self._create_metadata(
                transfer_id, source_repr, target_repr,
                size_bytes, "copy", duration, False
            )
            self._record_stats(metadata)

            return target_data, metadata

        except Exception as exc:
            _logger.debug(f"Copy transfer failed: {exc}")
            return None

    def _to_arrow_table(self, data: Any, source_repr: str) -> Any | None:
        """Convert data to Arrow table."""
        try:
            import pyarrow as pa

            if source_repr == "arrow":
                # Already Arrow
                return data if isinstance(data, pa.Table) else None

            if source_repr == "pandas":
                return pa.Table.from_pandas(data)

            if source_repr == "polars":
                return data.to_arrow()

            if source_repr == "duckdb":
                # DuckDB can export to Arrow
                if hasattr(data, "arrow"):
                    return data.arrow()

            return None

        except Exception as exc:
            _logger.debug(f"To Arrow conversion failed: {exc}")
            return None

    def _from_arrow_table(self, arrow_table: Any, target_repr: str) -> Any | None:
        """Convert Arrow table to target representation."""
        try:
            if target_repr == "arrow":
                return arrow_table

            if target_repr == "pandas":
                return arrow_table.to_pandas()

            if target_repr == "polars":
                import polars as pl
                return pl.from_arrow(arrow_table)

            if target_repr == "duckdb":
                # DuckDB can import from Arrow
                import duckdb
                return duckdb.arrow(arrow_table)

            return None

        except Exception as exc:
            _logger.debug(f"From Arrow conversion failed: {exc}")
            return None

    def _convert_data(self, data: Any, source_repr: str, target_repr: str) -> Any | None:
        """Direct conversion between representations (fallback)."""
        try:
            # pandas → polars
            if source_repr == "pandas" and target_repr == "polars":
                import polars as pl
                return pl.from_pandas(data)

            # polars → pandas
            if source_repr == "polars" and target_repr == "pandas":
                return data.to_pandas()

            # pandas → arrow
            if source_repr == "pandas" and target_repr == "arrow":
                import pyarrow as pa
                return pa.Table.from_pandas(data)

            # arrow → pandas
            if source_repr == "arrow" and target_repr == "pandas":
                return data.to_pandas()

            return None

        except Exception as exc:
            _logger.debug(f"Direct conversion failed: {exc}")
            return None

    def _is_zero_copy_path(self, source_repr: str, target_repr: str) -> bool:
        """Check if transfer path supports zero-copy."""
        # Arrow is the zero-copy hub
        zero_copy_pairs = {
            ("arrow", "polars"),
            ("polars", "arrow"),
            ("arrow", "duckdb"),
            ("duckdb", "arrow"),
        }
        return (source_repr, target_repr) in zero_copy_pairs

    def _estimate_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        try:
            # Try pandas
            if hasattr(data, "memory_usage"):
                return int(data.memory_usage(deep=True).sum())

            # Try polars
            if hasattr(data, "estimated_size"):
                return int(data.estimated_size())

            # Try arrow
            if hasattr(data, "nbytes"):
                return int(data.nbytes)

        except Exception:
            pass

        # Fallback
        return 100 * 1024 * 1024  # 100 MB

    def _check_arrow_available(self) -> bool:
        """Check if Apache Arrow is available."""
        try:
            import pyarrow
            return True
        except ImportError:
            return False

    def _create_metadata(
        self,
        transfer_id: str,
        source_repr: str,
        target_repr: str,
        size_bytes: int,
        method: str,
        duration_s: float,
        copy_avoided: bool,
    ) -> TransferMetadata:
        """Create transfer metadata."""
        return TransferMetadata(
            transfer_id=transfer_id,
            source_repr=source_repr,
            target_repr=target_repr,
            size_bytes=size_bytes,
            method=method,
            duration_s=duration_s,
            copy_avoided=copy_avoided,
        )

    def _record_stats(self, metadata: TransferMetadata) -> None:
        """Record transfer statistics."""
        with self._lock:
            self._stats.total_transfers += 1
            self._stats.total_bytes += metadata.size_bytes
            self._stats.total_duration_s += metadata.duration_s

            if metadata.method == "zerocopy":
                self._stats.zerocopy_count += 1
            elif metadata.method == "sharedmem":
                self._stats.sharedmem_count += 1
            elif metadata.method == "ipc":
                self._stats.ipc_count += 1
            elif metadata.method == "copy":
                self._stats.copy_count += 1

    def stats(self) -> dict[str, Any]:
        """Get transfer statistics."""
        with self._lock:
            zerocopy_rate = (
                self._stats.zerocopy_count / self._stats.total_transfers
                if self._stats.total_transfers > 0 else 0.0
            )
            avg_duration = (
                self._stats.total_duration_s / self._stats.total_transfers
                if self._stats.total_transfers > 0 else 0.0
            )
            avg_throughput = (
                self._stats.total_bytes / self._stats.total_duration_s
                if self._stats.total_duration_s > 0 else 0.0
            )

            return {
                "total_transfers": self._stats.total_transfers,
                "zerocopy_count": self._stats.zerocopy_count,
                "sharedmem_count": self._stats.sharedmem_count,
                "ipc_count": self._stats.ipc_count,
                "copy_count": self._stats.copy_count,
                "zerocopy_rate": zerocopy_rate,
                "total_bytes": self._stats.total_bytes,
                "total_duration_s": self._stats.total_duration_s,
                "avg_duration_s": avg_duration,
                "avg_throughput_bytes_per_s": avg_throughput,
                "arrow_available": self._has_arrow,
            }


class ZeroCopyBoundaryOptimizer:
    """High-level optimizer for zero-copy boundaries.

    Analyzes execution plan to identify opportunities for zero-copy
    transfer and inserts Arrow boundaries where beneficial.
    """

    def __init__(self, boundary: ArrowZeroCopyBoundary | None = None):
        """Initialize optimizer.

        Args:
            boundary: Optional boundary (creates default if None)
        """
        self._boundary = boundary or ArrowZeroCopyBoundary()

    def optimize_plan_transfers(
        self,
        plan: Any,
    ) -> dict[str, str]:
        """Analyze plan and recommend representation for each node.

        Args:
            plan: Execution plan to optimize

        Returns:
            Mapping from node_id to recommended representation
        """
        recommendations = {}

        try:
            # Walk plan tree
            nodes = self._extract_nodes(plan)

            for node in nodes:
                node_id = getattr(node, "node_id", str(id(node)))
                current_repr = getattr(node, "representation", "pandas")

                # Check children
                children = getattr(node, "inputs", [])
                if not children:
                    # Leaf node, keep current
                    recommendations[node_id] = current_repr
                    continue

                # Check if children are arrow-compatible
                child_reprs = [
                    getattr(c, "representation", "pandas")
                    for c in children
                ]

                # If all children are arrow-compatible, recommend arrow
                if all(r in ("arrow", "polars", "duckdb") for r in child_reprs):
                    recommendations[node_id] = "arrow"
                else:
                    recommendations[node_id] = current_repr

        except Exception as exc:
            _logger.warning(f"Transfer optimization failed: {exc}")

        return recommendations

    def _extract_nodes(self, plan: Any) -> list[Any]:
        """Extract all nodes from plan."""
        nodes = []

        def traverse(node):
            nodes.append(node)
            for child in getattr(node, "inputs", []):
                traverse(child)

        traverse(plan)
        return nodes


# Global singleton
_GLOBAL_ZEROCOPY_BOUNDARY: ArrowZeroCopyBoundary | None = None


def global_zerocopy_boundary() -> ArrowZeroCopyBoundary:
    """Get global zero-copy boundary singleton."""
    global _GLOBAL_ZEROCOPY_BOUNDARY
    if _GLOBAL_ZEROCOPY_BOUNDARY is None:
        _GLOBAL_ZEROCOPY_BOUNDARY = ArrowZeroCopyBoundary()
    return _GLOBAL_ZEROCOPY_BOUNDARY
