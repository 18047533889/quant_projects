"""
Health check endpoints for factor_engine.

Provides component health monitoring with customizable checks.
All health checks are privacy-safe and opt-in only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HealthStatus(Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """Check if component is degraded."""
        return self.status == HealthStatus.DEGRADED

    def is_unhealthy(self) -> bool:
        """Check if component is unhealthy."""
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: float = field(default_factory=time.time)

    @property
    def all_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(c.is_healthy() for c in self.components.values())

    @property
    def any_unhealthy(self) -> bool:
        """Check if any component is unhealthy."""
        return any(c.is_unhealthy() for c in self.components.values())

    @property
    def any_degraded(self) -> bool:
        """Check if any component is degraded."""
        return any(c.is_degraded() for c in self.components.values())


class HealthCheck:
    """A health check function for a component."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        interval: float = 30.0,
        timeout: float = 5.0,
    ):
        self.name = name
        self.check_fn = check_fn
        self.interval = interval
        self.timeout = timeout
        self._last_result: Optional[ComponentHealth] = None
        self._last_check_time: float = 0.0
        self._lock = threading.Lock()

    def run(self, force: bool = False) -> ComponentHealth:
        """Run the health check."""
        with self._lock:
            now = time.time()

            # Use cached result if within interval
            if not force and self._last_result is not None:
                if now - self._last_check_time < self.interval:
                    return self._last_result

            # Run the check with timeout
            try:
                result = self._run_with_timeout()
                self._last_result = result
                self._last_check_time = now
                return result
            except Exception as e:
                result = ComponentHealth(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    last_check=now,
                )
                self._last_result = result
                self._last_check_time = now
                return result

    def _run_with_timeout(self) -> ComponentHealth:
        """Run check with timeout."""
        result_container: List[ComponentHealth] = []
        error_container: List[Exception] = []

        def target():
            try:
                result_container.append(self.check_fn())
            except Exception as e:
                error_container.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # Timeout
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
            )

        if error_container:
            raise error_container[0]

        if result_container:
            return result_container[0]

        return ComponentHealth(
            name=self.name,
            status=HealthStatus.UNKNOWN,
            message="Health check returned no result",
        )


class HealthChecker:
    """Central health checker for all components."""

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable health checking (opt-in)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable health checking."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if health checking is enabled."""
        return self._enabled

    def register(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        interval: float = 30.0,
        timeout: float = 5.0,
    ) -> None:
        """Register a health check."""
        with self._lock:
            self._checks[name] = HealthCheck(name, check_fn, interval, timeout)

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)

    def check(self, name: str, force: bool = False) -> Optional[ComponentHealth]:
        """Run a specific health check."""
        if not self._enabled:
            return None

        with self._lock:
            check = self._checks.get(name)

        if check is None:
            return None

        return check.run(force=force)

    def check_all(self, force: bool = False) -> SystemHealth:
        """Run all health checks."""
        if not self._enabled:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                components={},
            )

        with self._lock:
            checks = list(self._checks.values())

        components = {}
        for check in checks:
            components[check.name] = check.run(force=force)

        # Determine overall status
        if not components:
            overall_status = HealthStatus.UNKNOWN
        elif any(c.is_unhealthy() for c in components.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.is_degraded() for c in components.values()):
            overall_status = HealthStatus.DEGRADED
        elif all(c.is_healthy() for c in components.values()):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall_status,
            components=components,
        )

    def get_status_dict(self) -> Dict[str, Any]:
        """Get health status as a dictionary (for HTTP endpoints)."""
        health = self.check_all()

        return {
            "status": health.status.value,
            "timestamp": health.timestamp,
            "components": {
                name: {
                    "status": component.status.value,
                    "message": component.message,
                    "details": component.details,
                    "last_check": component.last_check,
                }
                for name, component in health.components.items()
            },
        }


# Global health checker
_global_checker: Optional[HealthChecker] = None
_checker_lock = threading.Lock()


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    global _global_checker
    if _global_checker is None:
        with _checker_lock:
            if _global_checker is None:
                _global_checker = HealthChecker(enabled=False)
    return _global_checker


def configure_health_checks(enabled: bool = False) -> HealthChecker:
    """Configure the global health checker."""
    checker = get_health_checker()
    if enabled:
        checker.enable()
    else:
        checker.disable()
    return checker


# Common health check builders
def create_simple_health_check(
    name: str,
    status: HealthStatus = HealthStatus.HEALTHY,
    message: str = "",
) -> Callable[[], ComponentHealth]:
    """Create a simple health check that always returns the same status."""
    def check() -> ComponentHealth:
        return ComponentHealth(name=name, status=status, message=message)
    return check


def create_callable_health_check(
    name: str,
    callable_obj: Any,
    healthy_message: str = "Component is responsive",
    unhealthy_message: str = "Component is not responsive",
) -> Callable[[], ComponentHealth]:
    """Create a health check that tests if a callable object is available."""
    def check() -> ComponentHealth:
        try:
            if callable(callable_obj):
                callable_obj()
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=healthy_message,
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Object is not callable",
                )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"{unhealthy_message}: {str(e)}",
            )
    return check


def create_value_health_check(
    name: str,
    value_fn: Callable[[], float],
    threshold_degraded: Optional[float] = None,
    threshold_unhealthy: Optional[float] = None,
    compare: str = "less_than",  # "less_than" or "greater_than"
) -> Callable[[], ComponentHealth]:
    """Create a health check based on a numeric value and thresholds."""
    def check() -> ComponentHealth:
        try:
            value = value_fn()

            details = {"value": value}

            if threshold_unhealthy is not None:
                if compare == "greater_than" and value > threshold_unhealthy:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Value {value} exceeds unhealthy threshold {threshold_unhealthy}",
                        details=details,
                    )
                elif compare == "less_than" and value < threshold_unhealthy:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Value {value} below unhealthy threshold {threshold_unhealthy}",
                        details=details,
                    )

            if threshold_degraded is not None:
                if compare == "greater_than" and value > threshold_degraded:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message=f"Value {value} exceeds degraded threshold {threshold_degraded}",
                        details=details,
                    )
                elif compare == "less_than" and value < threshold_degraded:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message=f"Value {value} below degraded threshold {threshold_degraded}",
                        details=details,
                    )

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"Value {value} is within acceptable range",
                details=details,
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to get value: {str(e)}",
            )

    return check
