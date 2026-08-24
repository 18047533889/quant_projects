"""
Tests for telemetry.health module.
"""

import time
from factor_engine.telemetry.health import (
    HealthStatus,
    ComponentHealth,
    HealthChecker,
    get_health_checker,
    configure_health_checks,
    create_simple_health_check,
    create_value_health_check,
)


def test_component_health_creation():
    """Test creating component health."""
    health = ComponentHealth(
        name="test_component",
        status=HealthStatus.HEALTHY,
        message="All good",
    )

    assert health.name == "test_component"
    assert health.status == HealthStatus.HEALTHY
    assert health.message == "All good"
    assert health.is_healthy()
    assert not health.is_degraded()
    assert not health.is_unhealthy()


def test_component_health_statuses():
    """Test different health statuses."""
    healthy = ComponentHealth(name="test", status=HealthStatus.HEALTHY)
    degraded = ComponentHealth(name="test", status=HealthStatus.DEGRADED)
    unhealthy = ComponentHealth(name="test", status=HealthStatus.UNHEALTHY)
    unknown = ComponentHealth(name="test", status=HealthStatus.UNKNOWN)

    assert healthy.is_healthy()
    assert degraded.is_degraded()
    assert unhealthy.is_unhealthy()
    assert not unknown.is_healthy()


def test_health_checker_disabled_by_default():
    """Test that health checker is disabled by default."""
    checker = HealthChecker()
    assert not checker.is_enabled()

    def check_fn():
        return ComponentHealth(name="test", status=HealthStatus.HEALTHY)

    checker.register("test", check_fn)

    result = checker.check("test")
    assert result is None


def test_health_checker_enabled():
    """Test health checker when enabled."""
    checker = HealthChecker(enabled=True)
    assert checker.is_enabled()

    def check_fn():
        return ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="Working fine",
        )

    checker.register("test", check_fn)

    result = checker.check("test")
    assert result is not None
    assert result.name == "test"
    assert result.status == HealthStatus.HEALTHY
    assert result.message == "Working fine"


def test_health_checker_multiple_components():
    """Test health checker with multiple components."""
    checker = HealthChecker(enabled=True)

    def check_db():
        return ComponentHealth(name="database", status=HealthStatus.HEALTHY)

    def check_cache():
        return ComponentHealth(name="cache", status=HealthStatus.DEGRADED)

    def check_api():
        return ComponentHealth(name="api", status=HealthStatus.UNHEALTHY)

    checker.register("database", check_db)
    checker.register("cache", check_cache)
    checker.register("api", check_api)

    system_health = checker.check_all()

    assert len(system_health.components) == 3
    assert system_health.components["database"].is_healthy()
    assert system_health.components["cache"].is_degraded()
    assert system_health.components["api"].is_unhealthy()

    # Overall status should be unhealthy if any component is unhealthy
    assert system_health.status == HealthStatus.UNHEALTHY
    assert system_health.any_unhealthy


def test_health_checker_overall_status():
    """Test overall system status calculation."""
    checker = HealthChecker(enabled=True)

    # All healthy
    checker.register("comp1", lambda: ComponentHealth("comp1", HealthStatus.HEALTHY))
    system = checker.check_all()
    assert system.status == HealthStatus.HEALTHY
    assert system.all_healthy

    # One degraded
    checker.register("comp2", lambda: ComponentHealth("comp2", HealthStatus.DEGRADED))
    system = checker.check_all()
    assert system.status == HealthStatus.DEGRADED
    assert system.any_degraded

    # One unhealthy
    checker.register("comp3", lambda: ComponentHealth("comp3", HealthStatus.UNHEALTHY))
    system = checker.check_all()
    assert system.status == HealthStatus.UNHEALTHY
    assert system.any_unhealthy


def test_health_checker_unregister():
    """Test unregistering health checks."""
    checker = HealthChecker(enabled=True)

    checker.register("test", lambda: ComponentHealth("test", HealthStatus.HEALTHY))

    result = checker.check("test")
    assert result is not None

    checker.unregister("test")

    result = checker.check("test")
    assert result is None


def test_health_check_caching():
    """Test that health checks are cached."""
    checker = HealthChecker(enabled=True)

    call_count = [0]

    def check_fn():
        call_count[0] += 1
        return ComponentHealth("test", HealthStatus.HEALTHY)

    checker.register("test", check_fn, interval=1.0)

    # First call should invoke the check
    checker.check("test")
    assert call_count[0] == 1

    # Second call within interval should use cache
    checker.check("test")
    assert call_count[0] == 1

    # Force should bypass cache
    checker.check("test", force=True)
    assert call_count[0] == 2


def test_health_check_timeout():
    """Test health check timeout."""
    checker = HealthChecker(enabled=True)

    def slow_check():
        time.sleep(10.0)  # Longer than timeout
        return ComponentHealth("test", HealthStatus.HEALTHY)

    checker.register("test", slow_check, timeout=0.1)

    result = checker.check("test")
    assert result is not None
    assert result.status == HealthStatus.UNHEALTHY
    assert "timed out" in result.message


def test_health_check_exception():
    """Test health check that raises exception."""
    checker = HealthChecker(enabled=True)

    def failing_check():
        raise ValueError("Something went wrong")

    checker.register("test", failing_check)

    result = checker.check("test")
    assert result is not None
    assert result.status == HealthStatus.UNHEALTHY
    assert "Something went wrong" in result.message


def test_get_status_dict():
    """Test getting status as dictionary."""
    checker = HealthChecker(enabled=True)

    checker.register("test", lambda: ComponentHealth(
        "test",
        HealthStatus.HEALTHY,
        "All good",
        {"detail": "value"},
    ))

    status_dict = checker.get_status_dict()

    assert status_dict["status"] == "healthy"
    assert "timestamp" in status_dict
    assert "components" in status_dict
    assert "test" in status_dict["components"]
    assert status_dict["components"]["test"]["status"] == "healthy"
    assert status_dict["components"]["test"]["message"] == "All good"
    assert status_dict["components"]["test"]["details"]["detail"] == "value"


def test_global_health_checker():
    """Test global health checker singleton."""
    checker1 = get_health_checker()
    checker2 = get_health_checker()

    assert checker1 is checker2


def test_configure_health_checks():
    """Test configuring global health checks."""
    checker = configure_health_checks(enabled=True)
    assert checker.is_enabled()

    checker.register("test", lambda: ComponentHealth("test", HealthStatus.HEALTHY))

    result = checker.check("test")
    assert result is not None

    # Clean up
    checker.unregister("test")
    configure_health_checks(enabled=False)


def test_create_simple_health_check():
    """Test simple health check builder."""
    check_fn = create_simple_health_check(
        "test",
        HealthStatus.HEALTHY,
        "Everything is fine",
    )

    result = check_fn()
    assert result.name == "test"
    assert result.status == HealthStatus.HEALTHY
    assert result.message == "Everything is fine"


def test_create_value_health_check_greater_than():
    """Test value health check with greater_than comparison."""
    value = [50.0]

    def get_value():
        return value[0]

    check_fn = create_value_health_check(
        "cpu",
        get_value,
        threshold_degraded=70.0,
        threshold_unhealthy=90.0,
        compare="greater_than",
    )

    # Normal value
    result = check_fn()
    assert result.status == HealthStatus.HEALTHY

    # Degraded
    value[0] = 80.0
    result = check_fn()
    assert result.status == HealthStatus.DEGRADED

    # Unhealthy
    value[0] = 95.0
    result = check_fn()
    assert result.status == HealthStatus.UNHEALTHY


def test_create_value_health_check_less_than():
    """Test value health check with less_than comparison."""
    value = [100.0]

    def get_value():
        return value[0]

    check_fn = create_value_health_check(
        "memory",
        get_value,
        threshold_degraded=30.0,
        threshold_unhealthy=10.0,
        compare="less_than",
    )

    # Normal value
    result = check_fn()
    assert result.status == HealthStatus.HEALTHY

    # Degraded
    value[0] = 20.0
    result = check_fn()
    assert result.status == HealthStatus.DEGRADED

    # Unhealthy
    value[0] = 5.0
    result = check_fn()
    assert result.status == HealthStatus.UNHEALTHY


def test_create_value_health_check_exception():
    """Test value health check that raises exception."""
    def failing_value():
        raise RuntimeError("Cannot get value")

    check_fn = create_value_health_check(
        "test",
        failing_value,
        threshold_unhealthy=90.0,
    )

    result = check_fn()
    assert result.status == HealthStatus.UNHEALTHY
    assert "Cannot get value" in result.message


if __name__ == "__main__":
    import sys

    test_functions = [
        test_component_health_creation,
        test_component_health_statuses,
        test_health_checker_disabled_by_default,
        test_health_checker_enabled,
        test_health_checker_multiple_components,
        test_health_checker_overall_status,
        test_health_checker_unregister,
        test_health_check_caching,
        test_health_check_timeout,
        test_health_check_exception,
        test_get_status_dict,
        test_global_health_checker,
        test_configure_health_checks,
        test_create_simple_health_check,
        test_create_value_health_check_greater_than,
        test_create_value_health_check_less_than,
        test_create_value_health_check_exception,
    ]

    failed = []
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"✓ {test_fn.__name__}")
        except Exception as e:
            print(f"✗ {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test_fn.__name__)

    if failed:
        print(f"\n{len(failed)} tests failed:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"\nAll {len(test_functions)} tests passed!")
        sys.exit(0)
