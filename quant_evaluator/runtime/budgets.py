"""
Computation budget tracking for resource-aware evaluation.

Tracks memory, time, and operation counts to enforce budgets and
provide resource usage reports.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import psutil
import os


@dataclass
class ComputationBudget:
    """
    Resource budget for evaluation.

    Defines limits on memory, time, and operations.
    """
    max_memory_mb: Optional[float] = None
    max_time_seconds: Optional[float] = None
    max_operations: Optional[int] = None
    allow_overflow: bool = False
    warn_threshold: float = 0.8

    def has_limit(self) -> bool:
        """Check if any limit is set."""
        return (self.max_memory_mb is not None or
                self.max_time_seconds is not None or
                self.max_operations is not None)


@dataclass
class ResourceUsage:
    """
    Snapshot of resource usage.

    Captures memory, time, and operation counts at a point in time.
    """
    memory_mb: float = 0.0
    elapsed_seconds: float = 0.0
    operations_count: int = 0
    peak_memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "elapsed_seconds": self.elapsed_seconds,
            "operations_count": self.operations_count,
            "peak_memory_mb": self.peak_memory_mb,
            "timestamp": self.timestamp,
        }


class BudgetTracker:
    """
    Tracks resource usage against budget limits.

    Monitors memory, time, and operation counts and raises warnings
    or errors when limits are approached or exceeded.
    """

    def __init__(self, budget: ComputationBudget):
        """
        Initialize tracker.

        Args:
            budget: Computation budget to enforce
        """
        self.budget = budget
        self._start_time = time.time()
        self._operation_count = 0
        self._peak_memory_mb = 0.0
        self._process = psutil.Process(os.getpid())
        self._warned_memory = False
        self._warned_time = False
        self._warned_operations = False

    def record_operation(self, count: int = 1):
        """
        Record operations performed.

        Args:
            count: Number of operations to add
        """
        self._operation_count += count

    def get_current_usage(self) -> ResourceUsage:
        """
        Get current resource usage snapshot.

        Returns:
            ResourceUsage with current values
        """
        memory_mb = self._get_memory_mb()
        elapsed = time.time() - self._start_time

        # Update peak
        if memory_mb > self._peak_memory_mb:
            self._peak_memory_mb = memory_mb

        return ResourceUsage(
            memory_mb=memory_mb,
            elapsed_seconds=elapsed,
            operations_count=self._operation_count,
            peak_memory_mb=self._peak_memory_mb,
        )

    def check_budget(self, raise_on_exceed: bool = True) -> tuple[bool, Optional[str]]:
        """
        Check if current usage is within budget.

        Args:
            raise_on_exceed: Whether to raise exception on budget exceeded

        Returns:
            (within_budget, reason) tuple

        Raises:
            RuntimeError: If budget exceeded and raise_on_exceed is True
        """
        usage = self.get_current_usage()

        # Check memory
        if self.budget.max_memory_mb is not None:
            if usage.memory_mb > self.budget.max_memory_mb:
                msg = (f"Memory budget exceeded: {usage.memory_mb:.1f}MB > "
                       f"{self.budget.max_memory_mb:.1f}MB")
                if raise_on_exceed and not self.budget.allow_overflow:
                    raise RuntimeError(msg)
                return False, msg

            # Warn at threshold
            if (not self._warned_memory and
                usage.memory_mb > self.budget.max_memory_mb * self.budget.warn_threshold):
                self._warned_memory = True
                return True, (f"Memory approaching limit: {usage.memory_mb:.1f}MB "
                            f"({usage.memory_mb/self.budget.max_memory_mb:.0%} of budget)")

        # Check time
        if self.budget.max_time_seconds is not None:
            if usage.elapsed_seconds > self.budget.max_time_seconds:
                msg = (f"Time budget exceeded: {usage.elapsed_seconds:.1f}s > "
                       f"{self.budget.max_time_seconds:.1f}s")
                if raise_on_exceed and not self.budget.allow_overflow:
                    raise RuntimeError(msg)
                return False, msg

            # Warn at threshold
            if (not self._warned_time and
                usage.elapsed_seconds > self.budget.max_time_seconds * self.budget.warn_threshold):
                self._warned_time = True
                return True, (f"Time approaching limit: {usage.elapsed_seconds:.1f}s "
                            f"({usage.elapsed_seconds/self.budget.max_time_seconds:.0%} of budget)")

        # Check operations
        if self.budget.max_operations is not None:
            if usage.operations_count > self.budget.max_operations:
                msg = (f"Operation budget exceeded: {usage.operations_count} > "
                       f"{self.budget.max_operations}")
                if raise_on_exceed and not self.budget.allow_overflow:
                    raise RuntimeError(msg)
                return False, msg

            # Warn at threshold
            if (not self._warned_operations and
                usage.operations_count > self.budget.max_operations * self.budget.warn_threshold):
                self._warned_operations = True
                return True, (f"Operations approaching limit: {usage.operations_count} "
                            f"({usage.operations_count/self.budget.max_operations:.0%} of budget)")

        return True, None

    def get_remaining_budget(self) -> dict:
        """
        Get remaining budget amounts.

        Returns:
            Dictionary with remaining memory, time, and operations
        """
        usage = self.get_current_usage()
        remaining = {}

        if self.budget.max_memory_mb is not None:
            remaining["memory_mb"] = max(0, self.budget.max_memory_mb - usage.memory_mb)
            remaining["memory_pct"] = remaining["memory_mb"] / self.budget.max_memory_mb

        if self.budget.max_time_seconds is not None:
            remaining["time_seconds"] = max(0, self.budget.max_time_seconds - usage.elapsed_seconds)
            remaining["time_pct"] = remaining["time_seconds"] / self.budget.max_time_seconds

        if self.budget.max_operations is not None:
            remaining["operations"] = max(0, self.budget.max_operations - usage.operations_count)
            remaining["operations_pct"] = remaining["operations"] / self.budget.max_operations

        return remaining

    def reset(self):
        """Reset tracker to initial state."""
        self._start_time = time.time()
        self._operation_count = 0
        self._peak_memory_mb = 0.0
        self._warned_memory = False
        self._warned_time = False
        self._warned_operations = False

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except Exception:
            return 0.0

    def format_usage_report(self) -> str:
        """
        Format current usage as human-readable report.

        Returns:
            Formatted usage report string
        """
        usage = self.get_current_usage()
        lines = [
            "=== Resource Usage Report ===",
            f"Memory: {usage.memory_mb:.1f}MB (peak: {usage.peak_memory_mb:.1f}MB)",
            f"Time: {usage.elapsed_seconds:.2f}s",
            f"Operations: {usage.operations_count:,}",
        ]

        if self.budget.has_limit():
            lines.append("\n=== Budget Status ===")

            if self.budget.max_memory_mb is not None:
                pct = usage.memory_mb / self.budget.max_memory_mb * 100
                lines.append(f"Memory: {pct:.1f}% of {self.budget.max_memory_mb:.1f}MB limit")

            if self.budget.max_time_seconds is not None:
                pct = usage.elapsed_seconds / self.budget.max_time_seconds * 100
                lines.append(f"Time: {pct:.1f}% of {self.budget.max_time_seconds:.1f}s limit")

            if self.budget.max_operations is not None:
                pct = usage.operations_count / self.budget.max_operations * 100
                lines.append(f"Operations: {pct:.1f}% of {self.budget.max_operations:,} limit")

        return "\n".join(lines)
