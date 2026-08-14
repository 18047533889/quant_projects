"""
Health check system for production deployments.

Provides startup, liveness, and readiness probes.
"""

import time
import logging
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str, required: bool = True):
        """
        Initialize health check.

        Args:
            name: Check name
            required: Whether this check is required for overall health
        """
        self.name = name
        self.required = required

    def check(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            HealthCheckResult
        """
        raise NotImplementedError


class DiskSpaceCheck(HealthCheck):
    """Check available disk space."""

    def __init__(self, path: str = "/", threshold_percent: float = 90.0, required: bool = True):
        """
        Initialize disk space check.

        Args:
            path: Path to check
            threshold_percent: Alert threshold (percentage used)
            required: Whether this check is required
        """
        super().__init__("disk_space", required)
        self.path = path
        self.threshold_percent = threshold_percent

    def check(self) -> HealthCheckResult:
        """Check disk space."""
        try:
            usage = psutil.disk_usage(self.path)
            percent_used = usage.percent

            if percent_used >= self.threshold_percent:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk usage at {percent_used:.1f}% (threshold: {self.threshold_percent}%)",
                    details={
                        'path': self.path,
                        'total_gb': usage.total / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'free_gb': usage.free / (1024**3),
                        'percent_used': percent_used,
                    }
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=f"Disk space OK: {percent_used:.1f}% used",
                    details={
                        'path': self.path,
                        'percent_used': percent_used,
                        'free_gb': usage.free / (1024**3),
                    }
                )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk space: {e}",
                details={'error': str(e)}
            )


class MemoryCheck(HealthCheck):
    """Check available memory."""

    def __init__(self, threshold_percent: float = 95.0, required: bool = True):
        """
        Initialize memory check.

        Args:
            threshold_percent: Alert threshold (percentage used)
            required: Whether this check is required
        """
        super().__init__("memory", required)
        self.threshold_percent = threshold_percent

    def check(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            mem = psutil.virtual_memory()
            percent_used = mem.percent

            if percent_used >= self.threshold_percent:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory usage at {percent_used:.1f}% (threshold: {self.threshold_percent}%)",
                    details={
                        'total_gb': mem.total / (1024**3),
                        'available_gb': mem.available / (1024**3),
                        'used_gb': mem.used / (1024**3),
                        'percent_used': percent_used,
                    }
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message=f"Memory OK: {percent_used:.1f}% used",
                    details={
                        'percent_used': percent_used,
                        'available_gb': mem.available / (1024**3),
                    }
                )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory: {e}",
                details={'error': str(e)}
            )


class DatabaseCheck(HealthCheck):
    """Check database connectivity."""

    def __init__(self, check_func: Callable[[], bool], timeout_seconds: float = 3.0, required: bool = True):
        """
        Initialize database check.

        Args:
            check_func: Function that returns True if database is accessible
            timeout_seconds: Timeout for check
            required: Whether this check is required
        """
        super().__init__("database", required)
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds

    def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            start = time.time()
            result = self.check_func()
            duration = time.time() - start

            if result:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Database connection OK",
                    details={'response_time_ms': round(duration * 1000, 2)}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Database check returned False",
                    details={}
                )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {e}",
                details={'error': str(e)}
            )


class CacheCheck(HealthCheck):
    """Check cache connectivity."""

    def __init__(self, check_func: Callable[[], bool], timeout_seconds: float = 3.0, required: bool = False):
        """
        Initialize cache check.

        Args:
            check_func: Function that returns True if cache is accessible
            timeout_seconds: Timeout for check
            required: Whether this check is required
        """
        super().__init__("cache", required)
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds

    def check(self) -> HealthCheckResult:
        """Check cache connectivity."""
        try:
            start = time.time()
            result = self.check_func()
            duration = time.time() - start

            if result:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Cache connection OK",
                    details={'response_time_ms': round(duration * 1000, 2)}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Cache check returned False",
                    details={}
                )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED if not self.required else HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {e}",
                details={'error': str(e)}
            )


class CustomCheck(HealthCheck):
    """Custom health check with user-defined function."""

    def __init__(self, name: str, check_func: Callable[[], HealthCheckResult], required: bool = True):
        """
        Initialize custom check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
            required: Whether this check is required
        """
        super().__init__(name, required)
        self.check_func = check_func

    def check(self) -> HealthCheckResult:
        """Run custom check."""
        try:
            return self.check_func()
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check '{self.name}' failed: {e}",
                details={'error': str(e)}
            )


class HealthCheckManager:
    """
    Manages health checks for the application.

    Provides startup, liveness, and readiness probes.
    """

    def __init__(self, service_name: str = "quant-platform"):
        """
        Initialize health check manager.

        Args:
            service_name: Service name
        """
        self.service_name = service_name
        self.checks: List[HealthCheck] = []
        self.startup_complete = False
        self.logger = logging.getLogger(__name__)

    def add_check(self, check: HealthCheck):
        """
        Add a health check.

        Args:
            check: HealthCheck instance
        """
        self.checks.append(check)

    def remove_check(self, name: str):
        """
        Remove a health check by name.

        Args:
            name: Check name
        """
        self.checks = [c for c in self.checks if c.name != name]

    def startup_probe(self) -> Dict[str, Any]:
        """
        Startup probe - checks if application has started.

        Returns:
            Probe result dictionary
        """
        return {
            'status': 'ready' if self.startup_complete else 'starting',
            'service': self.service_name,
            'timestamp': time.time(),
        }

    def liveness_probe(self) -> Dict[str, Any]:
        """
        Liveness probe - checks if application is alive.

        Checks only critical components required for the app to be alive.

        Returns:
            Probe result dictionary
        """
        results = []
        overall_status = HealthStatus.HEALTHY

        for check in self.checks:
            if not check.required:
                continue

            result = check.check()
            results.append({
                'name': check.name,
                'status': result.status,
                'message': result.message,
            })

            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY

        return {
            'status': overall_status,
            'service': self.service_name,
            'timestamp': time.time(),
            'checks': results,
        }

    def readiness_probe(self) -> Dict[str, Any]:
        """
        Readiness probe - checks if application is ready to serve traffic.

        Checks all components including non-critical ones.

        Returns:
            Probe result dictionary
        """
        if not self.startup_complete:
            return {
                'status': HealthStatus.UNHEALTHY,
                'service': self.service_name,
                'message': 'Startup not complete',
                'timestamp': time.time(),
            }

        results = []
        overall_status = HealthStatus.HEALTHY
        has_degraded = False

        for check in self.checks:
            result = check.check()
            results.append({
                'name': check.name,
                'status': result.status,
                'message': result.message,
                'details': result.details,
            })

            if result.status == HealthStatus.UNHEALTHY and check.required:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED:
                has_degraded = True

        if overall_status == HealthStatus.HEALTHY and has_degraded:
            overall_status = HealthStatus.DEGRADED

        return {
            'status': overall_status,
            'service': self.service_name,
            'timestamp': time.time(),
            'checks': results,
        }

    def mark_startup_complete(self):
        """Mark startup as complete."""
        self.startup_complete = True
        self.logger.info(f"Service {self.service_name} startup complete")

    def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Comprehensive health report
        """
        return self.readiness_probe()


def configure_health_checks(config) -> HealthCheckManager:
    """
    Configure health checks from ProductionConfig.

    Args:
        config: ProductionConfig instance

    Returns:
        Configured HealthCheckManager
    """
    manager = HealthCheckManager(service_name=config.service_name)

    if not config.health_check.enabled:
        return manager

    # Add disk space check
    if config.health_check.check_disk_space:
        manager.add_check(DiskSpaceCheck(
            threshold_percent=config.health_check.disk_space_threshold_percent
        ))

    # Add memory check
    if config.health_check.check_memory:
        manager.add_check(MemoryCheck(
            threshold_percent=config.health_check.memory_threshold_percent
        ))

    return manager
