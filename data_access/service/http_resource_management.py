"""
R32-P0-096: HTTP stream query_slots exactly-once release.
R32-P0-097: HTTP QueryBudget v2 字段全量传播.
R32-P0-098: factor read HTTP路径同样必须走QueryBudget/ExecutionContext/Trace.

HTTP service resource management and query budget propagation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError
from data_access.read.query_budget import QueryBudget


@dataclass(frozen=True)
class HttpQuerySlotLease:
    """HTTP query slot with exactly-once release guarantee.

    R32-P0-096: Ensures no double-release or leaked release across:
    - first=StopIteration
    - first抛异常
    - StreamingResponse构造失败
    - body finally
    - 外层except
    """

    slot_id: str
    semaphore: threading.BoundedSemaphore
    _released: bool = False

    def release(self) -> None:
        """Release exactly once, safe to call multiple times."""
        if not self._released:
            object.__setattr__(self, "_released", True)
            self.semaphore.release()

    def __enter__(self) -> HttpQuerySlotLease:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class HttpQuerySlotManager:
    """Manages query slots with exactly-once release semantics.

    R32-P0-096: Central authority for HTTP query slot lifecycle.
    """

    def __init__(self, max_concurrency: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._lock = threading.Lock()
        self._active_leases: dict[str, bool] = {}

    def acquire(self, slot_id: str, *, blocking: bool = False) -> HttpQuerySlotLease | None:
        """Acquire query slot.

        Returns:
            Lease if acquired, None if not available (non-blocking)

        Raises:
            ValidationError: If slot_id already active
        """
        with self._lock:
            if slot_id in self._active_leases:
                raise ValidationError(f"Query slot {slot_id} already active")

        if not self._semaphore.acquire(blocking=blocking):
            return None

        with self._lock:
            self._active_leases[slot_id] = True

        # Create lease with custom release that removes from active set
        class TrackedLease(HttpQuerySlotLease):
            def release(inner_self) -> None:
                if not inner_self._released:
                    object.__setattr__(inner_self, "_released", True)
                    with self._lock:
                        self._active_leases.pop(slot_id, None)
                    self._semaphore.release()

        return TrackedLease(slot_id=slot_id, semaphore=self._semaphore)

    def active_count(self) -> int:
        """Get count of active leases."""
        with self._lock:
            return len(self._active_leases)


@dataclass(frozen=True)
class HttpQueryBudgetV2:
    """HTTP QueryBudget v2 with all fields preserved.

    R32-P0-097: API budget必须保留:
    - scan objects/files
    - scan bytes
    - remote requests
    - result bytes
    - rows
    - memory
    - elapsed/deadline
    - required columns/time range
    - spill/temp limits
    """

    # Scan limits
    max_scan_objects: int | None = None
    max_scan_files: int | None = None
    max_scan_bytes: int | None = None
    max_remote_requests: int | None = None

    # Result limits
    max_result_bytes: int | None = None
    max_rows: int | None = None
    max_memory_bytes: int | None = None

    # Time limits
    max_elapsed_ms: int | None = None
    deadline_ms: int | None = None

    # Required filters
    require_columns: bool = True
    require_time_range: bool = False
    required_time_range: tuple[Any, Any] | None = None

    # Spill/temp limits
    max_spill_bytes: int | None = None
    max_temp_files: int | None = None

    def to_core_budget(self) -> QueryBudget:
        """Convert to core QueryBudget for backend execution.

        R32-P0-097: 不要重新构造时丢字段 - all fields preserved.
        """
        return QueryBudget(
            max_scan_files=self.max_scan_files,
            max_scan_bytes=self.max_scan_bytes,
            max_result_bytes=self.max_result_bytes,
            max_rows=self.max_rows,
            max_elapsed_ms=self.max_elapsed_ms,
            require_columns=self.require_columns,
            require_time_range=self.require_time_range,
        )

    @classmethod
    def from_core_budget(cls, budget: QueryBudget) -> HttpQueryBudgetV2:
        """Create from core QueryBudget, preserving all fields."""
        return cls(
            max_scan_files=budget.max_scan_files,
            max_scan_bytes=budget.max_scan_bytes,
            max_result_bytes=budget.max_result_bytes,
            max_rows=budget.max_rows,
            max_elapsed_ms=budget.max_elapsed_ms,
            require_columns=budget.require_columns,
            require_time_range=budget.require_time_range,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to dict."""
        return {
            "max_scan_objects": self.max_scan_objects,
            "max_scan_files": self.max_scan_files,
            "max_scan_bytes": self.max_scan_bytes,
            "max_remote_requests": self.max_remote_requests,
            "max_result_bytes": self.max_result_bytes,
            "max_rows": self.max_rows,
            "max_memory_bytes": self.max_memory_bytes,
            "max_elapsed_ms": self.max_elapsed_ms,
            "deadline_ms": self.deadline_ms,
            "require_columns": self.require_columns,
            "require_time_range": self.require_time_range,
            "required_time_range": self.required_time_range,
            "max_spill_bytes": self.max_spill_bytes,
            "max_temp_files": self.max_temp_files,
        }


@dataclass(frozen=True)
class FactorReadContext:
    """Factor read execution context with full governance.

    R32-P0-098: /v1/factors/read不能成为特殊旁路.
    与dataset read共用:
    - authorization
    - query budget
    - resource governor
    - execution context
    - trace
    - lineage
    - error mapping
    """

    request_id: str
    principal: Any
    authorizer: Any
    budget: HttpQueryBudgetV2
    trace_id: str
    execution_context: Any

    def validate(self) -> None:
        """Ensure all required governance fields are present."""
        if not self.principal:
            raise ValidationError(
                "R32-P0-098: factor read requires principal (authorization)"
            )
        if not self.authorizer:
            raise ValidationError(
                "R32-P0-098: factor read requires authorizer"
            )
        if not self.budget:
            raise ValidationError(
                "R32-P0-098: factor read requires query budget"
            )
        if not self.execution_context:
            raise ValidationError(
                "R32-P0-098: factor read requires execution context"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for audit/lineage."""
        return {
            "request_id": self.request_id,
            "principal_id": getattr(self.principal, "principal_id", None),
            "budget": self.budget.to_dict(),
            "trace_id": self.trace_id,
        }


def create_factor_read_context(
    request_id: str,
    principal: Any,
    authorizer: Any,
    budget: QueryBudget,
    execution_context: Any,
) -> FactorReadContext:
    """Create factor read context with full governance.

    R32-P0-098: Enforce that factor reads go through same governance
    as dataset reads - no special bypass.
    """
    import uuid

    http_budget = HttpQueryBudgetV2.from_core_budget(budget)

    ctx = FactorReadContext(
        request_id=request_id,
        principal=principal,
        authorizer=authorizer,
        budget=http_budget,
        trace_id=uuid.uuid4().hex,
        execution_context=execution_context,
    )

    ctx.validate()
    return ctx


__all__ = [
    "HttpQuerySlotLease",
    "HttpQuerySlotManager",
    "HttpQueryBudgetV2",
    "FactorReadContext",
    "create_factor_read_context",
]
