# -*- coding: utf-8 -*-
"""MB-P1-007: Memory budget token management.

Token-based memory budget allocation and enforcement:
- Pre-allocate memory tokens before task execution
- Dynamic token pool sized to available memory
- Token return on task completion
- Admission control based on token availability
- Integration with ResourceBroker for unified governance

Key principles:
- Fail-closed: reject tasks when tokens unavailable
- Dynamic sizing: token pool adjusts to memory pressure
- Fair allocation: prevent single task monopolizing tokens
- Predictable: token request based on static estimates
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryToken:
    """A memory budget allocation token.

    Attributes:
        token_id: Unique token identifier
        bytes_allocated: Memory bytes allocated
        task_id: Task holding this token
        issued_at: Timestamp when issued
        expires_at: Optional expiration timestamp
    """
    token_id: str
    bytes_allocated: int
    task_id: str
    issued_at: float
    expires_at: float | None = None


@dataclass
class BudgetAllocation:
    """Current budget allocation state.

    Attributes:
        total_budget_bytes: Total available budget
        allocated_bytes: Currently allocated bytes
        available_bytes: Remaining available bytes
        token_count: Number of active tokens
        pending_requests: Number of pending requests
        rejected_count: Total requests rejected
    """
    total_budget_bytes: int
    allocated_bytes: int = 0
    available_bytes: int = 0
    token_count: int = 0
    pending_requests: int = 0
    rejected_count: int = 0


class MemoryBudgetManager:
    """Token-based memory budget manager.

    Provides admission control for task execution based on memory budget.
    Integrates with adaptive_batch_scheduler for resource governance.
    """

    def __init__(
        self,
        total_budget_bytes: int,
        *,
        max_task_fraction: float = 0.4,
        reserve_bytes: int = 512 * 1024 * 1024,  # 512 MB reserve
        enable_dynamic_sizing: bool = True,
    ):
        """Initialize memory budget manager.

        Args:
            total_budget_bytes: Total memory budget (bytes)
            max_task_fraction: Maximum fraction single task can allocate
            reserve_bytes: Reserved memory (never allocated)
            enable_dynamic_sizing: Enable dynamic budget adjustment
        """
        self._total_budget_bytes = total_budget_bytes
        self._max_task_fraction = max_task_fraction
        self._reserve_bytes = reserve_bytes
        self._enable_dynamic_sizing = enable_dynamic_sizing

        self._allocation = BudgetAllocation(
            total_budget_bytes=total_budget_bytes - reserve_bytes,
            available_bytes=total_budget_bytes - reserve_bytes,
        )

        self._tokens: dict[str, MemoryToken] = {}
        self._token_counter = 0
        self._lock = threading.RLock()
        self._token_available = threading.Condition(self._lock)

    def request_token(
        self,
        task_id: str,
        bytes_required: int,
        *,
        timeout_s: float | None = None,
        allow_wait: bool = True,
    ) -> MemoryToken | None:
        """Request memory budget token for task.

        Args:
            task_id: Task requesting token
            bytes_required: Memory bytes required
            timeout_s: Optional timeout (seconds)
            allow_wait: If True, wait for tokens; if False, fail immediately

        Returns:
            MemoryToken if granted, None if rejected
        """
        with self._lock:
            # Check against maximum per-task allocation
            max_allowed = int(
                self._allocation.total_budget_bytes * self._max_task_fraction
            )
            if bytes_required > max_allowed:
                _logger.warning(
                    f"Task {task_id} requests {bytes_required // 1024 // 1024} MB "
                    f"exceeding per-task limit {max_allowed // 1024 // 1024} MB"
                )
                self._allocation.rejected_count += 1
                return None

            # Try immediate allocation
            if self._allocation.available_bytes >= bytes_required:
                return self._allocate_token(task_id, bytes_required)

            # Wait if allowed
            if allow_wait:
                self._allocation.pending_requests += 1
                try:
                    deadline = time.time() + timeout_s if timeout_s else None

                    while self._allocation.available_bytes < bytes_required:
                        wait_time = None
                        if deadline:
                            wait_time = max(0.0, deadline - time.time())
                            if wait_time <= 0:
                                break

                        self._token_available.wait(timeout=wait_time)

                        # Check if became available
                        if self._allocation.available_bytes >= bytes_required:
                            return self._allocate_token(task_id, bytes_required)

                        # Timeout check
                        if deadline and time.time() >= deadline:
                            break

                finally:
                    self._allocation.pending_requests -= 1

            # Failed to allocate
            _logger.debug(
                f"Token request rejected for {task_id}: "
                f"requested {bytes_required // 1024 // 1024} MB, "
                f"available {self._allocation.available_bytes // 1024 // 1024} MB"
            )
            self._allocation.rejected_count += 1
            return None

    def return_token(self, token: MemoryToken) -> None:
        """Return memory budget token after task completion.

        Args:
            token: Token to return
        """
        with self._lock:
            if token.token_id not in self._tokens:
                _logger.warning(f"Token {token.token_id} already returned")
                return

            del self._tokens[token.token_id]
            self._allocation.allocated_bytes -= token.bytes_allocated
            self._allocation.available_bytes += token.bytes_allocated
            self._allocation.token_count -= 1

            _logger.debug(
                f"Token returned for {token.task_id}: "
                f"{token.bytes_allocated // 1024 // 1024} MB, "
                f"available={self._allocation.available_bytes // 1024 // 1024} MB"
            )

            # Notify waiting threads
            self._token_available.notify_all()

    def adjust_budget(self, new_budget_bytes: int) -> None:
        """Dynamically adjust total budget.

        Args:
            new_budget_bytes: New total budget (before reserve)
        """
        with self._lock:
            old_budget = self._allocation.total_budget_bytes
            available_budget = new_budget_bytes - self._reserve_bytes

            delta = available_budget - old_budget

            self._allocation.total_budget_bytes = available_budget
            self._allocation.available_bytes += delta

            _logger.info(
                f"Budget adjusted: {old_budget // 1024 // 1024} MB → "
                f"{available_budget // 1024 // 1024} MB "
                f"({delta // 1024 // 1024:+d} MB)"
            )

            if delta > 0:
                # More budget available, wake waiters
                self._token_available.notify_all()

    def reclaim_expired_tokens(self) -> int:
        """Reclaim tokens that have expired.

        Returns:
            Number of tokens reclaimed
        """
        with self._lock:
            now = time.time()
            expired = [
                token_id
                for token_id, token in self._tokens.items()
                if token.expires_at and token.expires_at < now
            ]

            for token_id in expired:
                token = self._tokens[token_id]
                _logger.warning(
                    f"Reclaiming expired token for {token.task_id}: "
                    f"{token.bytes_allocated // 1024 // 1024} MB"
                )
                self.return_token(token)

            return len(expired)

    def get_allocation_state(self) -> BudgetAllocation:
        """Get current allocation state."""
        with self._lock:
            return BudgetAllocation(
                total_budget_bytes=self._allocation.total_budget_bytes,
                allocated_bytes=self._allocation.allocated_bytes,
                available_bytes=self._allocation.available_bytes,
                token_count=self._allocation.token_count,
                pending_requests=self._allocation.pending_requests,
                rejected_count=self._allocation.rejected_count,
            )

    def stats(self) -> dict[str, Any]:
        """Get budget manager statistics."""
        with self._lock:
            utilization = (
                self._allocation.allocated_bytes / self._allocation.total_budget_bytes
                if self._allocation.total_budget_bytes > 0 else 0.0
            )

            return {
                "total_budget_bytes": self._allocation.total_budget_bytes,
                "allocated_bytes": self._allocation.allocated_bytes,
                "available_bytes": self._allocation.available_bytes,
                "utilization": utilization,
                "token_count": self._allocation.token_count,
                "pending_requests": self._allocation.pending_requests,
                "rejected_count": self._allocation.rejected_count,
                "reserve_bytes": self._reserve_bytes,
            }

    def _allocate_token(self, task_id: str, bytes_required: int) -> MemoryToken:
        """Internal: allocate token (caller must hold lock)."""
        self._token_counter += 1
        token_id = f"token_{self._token_counter:06d}"

        token = MemoryToken(
            token_id=token_id,
            bytes_allocated=bytes_required,
            task_id=task_id,
            issued_at=time.time(),
            expires_at=None,  # No expiration by default
        )

        self._tokens[token_id] = token
        self._allocation.allocated_bytes += bytes_required
        self._allocation.available_bytes -= bytes_required
        self._allocation.token_count += 1

        _logger.debug(
            f"Token allocated for {task_id}: {bytes_required // 1024 // 1024} MB, "
            f"remaining={self._allocation.available_bytes // 1024 // 1024} MB"
        )

        return token


class TokenBasedAdmissionController:
    """Admission controller using memory budget tokens.

    Integrates with adaptive_batch_scheduler for task admission control.
    """

    def __init__(
        self,
        budget_manager: MemoryBudgetManager,
        *,
        size_estimator: Any | None = None,
    ):
        """Initialize admission controller.

        Args:
            budget_manager: Memory budget manager
            size_estimator: Optional task size estimator
        """
        self._budget_manager = budget_manager
        self._size_estimator = size_estimator
        self._active_tokens: dict[str, MemoryToken] = {}
        self._lock = threading.Lock()

    def admit_task(
        self,
        task: Any,
        *,
        timeout_s: float = 30.0,
    ) -> bool:
        """Admit task for execution (request token).

        Args:
            task: Task to admit
            timeout_s: Token request timeout

        Returns:
            True if admitted, False if rejected
        """
        task_id = getattr(task, "task_id", str(id(task)))

        # Estimate memory requirement
        bytes_required = self._estimate_task_memory(task)

        # Request token
        token = self._budget_manager.request_token(
            task_id, bytes_required, timeout_s=timeout_s
        )

        if token is None:
            return False

        # Track active token
        with self._lock:
            self._active_tokens[task_id] = token

        return True

    def release_task(self, task: Any) -> None:
        """Release task resources (return token).

        Args:
            task: Task to release
        """
        task_id = getattr(task, "task_id", str(id(task)))

        with self._lock:
            token = self._active_tokens.pop(task_id, None)

        if token is not None:
            self._budget_manager.return_token(token)

    def _estimate_task_memory(self, task: Any) -> int:
        """Estimate task memory requirement.

        Args:
            task: Task to estimate

        Returns:
            Estimated memory bytes
        """
        if self._size_estimator is not None:
            try:
                estimate = self._size_estimator.estimate_from_plan(
                    getattr(task, "node_ref", task)
                )
                return estimate.total_bytes
            except Exception as exc:
                _logger.debug(f"Size estimation failed: {exc}")

        # Conservative fallback
        return 100 * 1024 * 1024  # 100 MB


# Global singleton
_GLOBAL_BUDGET_MANAGER: MemoryBudgetManager | None = None


def global_budget_manager(
    total_budget_bytes: int | None = None,
) -> MemoryBudgetManager:
    """Get global memory budget manager singleton.

    Args:
        total_budget_bytes: Budget size (only on first call)

    Returns:
        Global MemoryBudgetManager instance
    """
    global _GLOBAL_BUDGET_MANAGER
    if _GLOBAL_BUDGET_MANAGER is None:
        if total_budget_bytes is None:
            # P3/P4: 从 ResourceBroker（单权威）的 ExecutionBudget 派生，不再固定
            # 4GiB。broker 无法给出真实预算时回退绝对上限。
            try:
                from factor_engine.runtime.resource_broker import ResourceBroker

                broker = ResourceBroker()
                total_budget_bytes = broker.execution_budget()
            except Exception:
                total_budget_bytes = None
            if not total_budget_bytes:
                try:
                    import psutil
                    total_budget_bytes = int(psutil.virtual_memory().available * 0.8)
                except Exception:
                    total_budget_bytes = 4 * 1024**3  # 4 GB 绝对上限回退

        _GLOBAL_BUDGET_MANAGER = MemoryBudgetManager(total_budget_bytes)

    return _GLOBAL_BUDGET_MANAGER
