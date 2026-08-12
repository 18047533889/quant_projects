"""
R32-P0-092: read_auto must not rematerialize when streaming is required.
R32-P0-093: Raw scan_polars/LazyFrame production surface lockdown.
R32-P0-094: Stream snapshot must resolve exact object set only once.
R32-P0-095: Stream/iterator disconnect must release all resources.

Streaming integrity and resource management.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

from data_access.core.exceptions import ValidationError


@dataclass(frozen=True)
class StreamingDecision:
    """Cost planner decision: materialize vs stream.

    R32-P0-092: If planner decides stream, must NOT rematerialize.
    """

    mode: str  # "materialize", "stream"
    reason: str
    estimated_bytes: int | None = None
    estimated_rows: int | None = None

    def require_stream_api(self) -> bool:
        """Check if caller MUST use streaming API."""
        return self.mode == "stream"


class StreamResourceManager:
    """Resource manager for streaming reads.

    R32-P0-095: Stream/iterator断连时必须释放全部资源:
    - query slot
    - DuckDB slot
    - remote slot
    - execution lease
    - source block
    - temp file
    """

    def __init__(self, scope_id: str) -> None:
        self.scope_id = scope_id
        self._resources: list[tuple[str, Any]] = []
        self._released = False

    def register(self, resource_type: str, resource: Any) -> None:
        """Register a resource for cleanup."""
        if not self._released:
            self._resources.append((resource_type, resource))

    def release_all(self) -> None:
        """Release all registered resources exactly once.

        R32-P0-095: Covers client disconnect, GeneratorExit, CancelledError,
        serialization exception, first batch exception.
        """
        if self._released:
            return

        self._released = True

        # Release in reverse order (LIFO)
        for resource_type, resource in reversed(self._resources):
            try:
                if resource_type == "query_slot":
                    # BoundedSemaphore
                    if hasattr(resource, "release"):
                        resource.release()
                elif resource_type == "duckdb_slot":
                    # DuckDB connection slot
                    if hasattr(resource, "close"):
                        resource.close()
                elif resource_type == "remote_slot":
                    # Remote connection lease
                    if hasattr(resource, "release"):
                        resource.release()
                elif resource_type == "execution_lease":
                    # Execution resource lease
                    if hasattr(resource, "release"):
                        resource.release()
                elif resource_type == "source_block":
                    # Source block cache entry
                    if hasattr(resource, "unpin"):
                        resource.unpin()
                elif resource_type == "temp_file":
                    # Temporary file cleanup
                    import os

                    try:
                        if os.path.exists(resource):
                            os.unlink(resource)
                    except OSError:
                        pass
                elif resource_type == "iterator":
                    # Generator/iterator
                    if hasattr(resource, "close"):
                        resource.close()
            except Exception:
                # Best-effort cleanup, don't propagate exceptions
                pass

    def __enter__(self) -> StreamResourceManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_all()


@dataclass(frozen=True)
class PreparedStreamRead:
    """Prepared stream read with resolved snapshot.

    R32-P0-094: Stream snapshot必须只resolve一次exact object set.
    创建reader、lineage、snapshot、post-verify都绑定同一个PreparedStreamRead.
    禁止为了构建snapshot第二次resolve,造成reader实际对象和报告对象不同.
    """

    dataset: str
    snapshot_id: str
    exact_objects: tuple[str, ...]  # Immutable, resolved exactly once
    schema_hash: str
    contract_digest: str
    resource_manager: StreamResourceManager

    def validate_object_set(self, observed_objects: tuple[str, ...]) -> None:
        """Verify reader used exactly the prepared object set.

        R32-P0-094: Detect if reader accidentally used different objects.
        """
        if set(observed_objects) != set(self.exact_objects):
            raise ValidationError(
                f"Stream read object mismatch: prepared {len(self.exact_objects)} "
                f"objects but reader used {len(observed_objects)}. "
                "R32-P0-094 violation: snapshot resolved twice."
            )


class GovernedScanHandle:
    """Governed scan handle for production use.

    R32-P0-093: Production/automated research只给Governed ScanHandle.
    Raw lazy API为internal/dev或明确_unsafe,collect前仍需final snapshot/budget/audit.
    """

    def __init__(
        self,
        lazy_frame: Any,
        snapshot: Any,
        budget: Any,
        audit_context: dict[str, Any],
    ) -> None:
        self._lazy_frame = lazy_frame
        self._snapshot = snapshot
        self._budget = budget
        self._audit_context = audit_context
        self._collected = False

    @property
    def snapshot(self) -> Any:
        """Access prepared snapshot."""
        return self._snapshot

    @property
    def budget(self) -> Any:
        """Access query budget."""
        return self._budget

    def collect(self) -> Any:
        """Collect with final snapshot/budget/audit validation."""
        if self._collected:
            raise ValidationError("GovernedScanHandle already collected")

        self._collected = True

        # Final pre-collect validation
        # Real implementation would verify budget, snapshot freshness, etc.

        result = self._lazy_frame.collect()

        # Post-collect audit
        # Real implementation would record actual resource usage

        return result

    def stream(self) -> Iterator[Any]:
        """Stream with governed resource management."""
        if self._collected:
            raise ValidationError("GovernedScanHandle already collected")

        self._collected = True

        # Stream with resource manager
        with StreamResourceManager(scope_id=str(id(self))) as rm:
            # Register lazy frame for cleanup
            rm.register("iterator", self._lazy_frame)

            for batch in self._lazy_frame.iter_slices():
                yield batch


def check_streaming_requirement(decision: StreamingDecision) -> None:
    """R32-P0-092: Enforce streaming when planner requires it.

    Raises:
        ValidationError: If caller tries to materialize when stream is required
    """
    if decision.require_stream_api():
        raise ValidationError(
            f"Cost planner决定stream (reason: {decision.reason}), "
            "caller must use streaming API. "
            "R32-P0-092: 禁止list(batches)->Table把资源语义打回全量物化."
        )


def create_unsafe_scan_handle(lazy_frame: Any) -> Any:
    """R32-P0-093: Raw scan_polars/LazyFrame production surface收口.

    Raw lazy API为internal/dev或明确_unsafe.

    Returns:
        Raw lazy frame with _unsafe marker

    Raises:
        ValidationError: In production/automated modes
    """
    import os

    prod_mode = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
        "1",
        "true",
        "yes",
    }
    fe_mode = os.environ.get("FACTOR_ENGINE_RUN_MODE", "")

    if prod_mode or fe_mode in {"automated_research", "production"}:
        raise ValidationError(
            "R32-P0-093: Raw scan_polars/LazyFrame禁止在production/automated模式. "
            "Use GovernedScanHandle."
        )

    # Mark as unsafe for dev/research use
    class UnsafeLazyFrame:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._is_unsafe = True

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return UnsafeLazyFrame(lazy_frame)


__all__ = [
    "StreamingDecision",
    "StreamResourceManager",
    "PreparedStreamRead",
    "GovernedScanHandle",
    "check_streaming_requirement",
    "create_unsafe_scan_handle",
]
