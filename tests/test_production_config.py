"""
Test suite for production configuration, logging, and monitoring.
"""

import unittest
import tempfile
import time
from pathlib import Path

from config.production_config import (
    ProductionConfig,
    LoggingConfig,
    MetricsConfig,
    HealthCheckConfig,
    ResourceLimitsConfig,
    SecurityConfig,
)
from config.logging.sanitization import sanitize_message, sanitize_dict
from config.logging.structured_logger import setup_logging, get_logger
from config.monitoring.metrics import MetricsCollector
from config.monitoring.health import (
    HealthCheckManager,
    DiskSpaceCheck,
    MemoryCheck,
    HealthStatus,
)


class TestProductionConfig(unittest.TestCase):
    """Test production configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProductionConfig()
        self.assertEqual(config.environment, "production")
        self.assertEqual(config.service_name, "quant-platform")
        self.assertTrue(config.enable_graceful_shutdown)

    def test_config_from_dict(self):
        """Test loading from dictionary."""
        config_dict = {
            "environment": "staging",
            "service_name": "test-service",
            "version": "1.2.3",
            "logging": {
                "level": "DEBUG",
                "format": "text",
            },
            "metrics": {
                "enabled": False,
            }
        }
        config = ProductionConfig(config_dict)
        self.assertEqual(config.environment, "staging")
        self.assertEqual(config.service_name, "test-service")
        self.assertEqual(config.version, "1.2.3")
        self.assertEqual(config.logging.level, "DEBUG")
        self.assertEqual(config.logging.format, "text")
        self.assertFalse(config.metrics.enabled)

    def test_config_validation(self):
        """Test configuration validation."""
        config = ProductionConfig()
        # Should not raise
        config.validate_and_raise()
        self.assertTrue(config.is_validated())

    def test_invalid_config(self):
        """Test invalid configuration."""
        config_dict = {
            "logging": {
                "level": "INVALID",
            }
        }
        config = ProductionConfig(config_dict)
        with self.assertRaises(Exception):
            config.validate_and_raise()


class TestSanitization(unittest.TestCase):
    """Test sensitive data sanitization."""

    def test_sanitize_password(self):
        """Test password sanitization."""
        message = "Connecting with password=secret123"
        sanitized = sanitize_message(message)
        self.assertIn("***REDACTED***", sanitized)
        self.assertNotIn("secret123", sanitized)

    def test_sanitize_token(self):
        """Test token sanitization."""
        message = "Authorization: Bearer abc123xyz"
        sanitized = sanitize_message(message)
        self.assertIn("***REDACTED***", sanitized)
        self.assertNotIn("abc123xyz", sanitized)

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "user": "john",
            "password": "secret",
            "api_key": "key123",
            "data": {
                "token": "token456",
                "value": 100,
            }
        }
        sanitized = sanitize_dict(data)
        self.assertEqual(sanitized["user"], "john")
        self.assertEqual(sanitized["password"], "***REDACTED***")
        self.assertEqual(sanitized["api_key"], "***REDACTED***")
        self.assertEqual(sanitized["data"]["token"], "***REDACTED***")
        self.assertEqual(sanitized["data"]["value"], 100)

    def test_sanitize_jwt(self):
        """Test JWT token sanitization."""
        message = "JWT: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        sanitized = sanitize_message(message)
        self.assertIn("***REDACTED***", sanitized)


class TestStructuredLogging(unittest.TestCase):
    """Test structured logging."""

    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                level="INFO",
                log_dir=Path(tmpdir),
                log_format="json",
                service_name="test-service",
            )
            self.assertIsNotNone(logger)

            # Test logging
            logger.info("Test message", context={"key": "value"})

            # Check log file created
            log_file = Path(tmpdir) / "test-service.log"
            self.assertTrue(log_file.exists())

    def test_get_logger(self):
        """Test getting child logger."""
        logger = get_logger("test_module", parent="test_parent")
        self.assertIsNotNone(logger)

    def test_context_logging(self):
        """Test logging with context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                level="DEBUG",
                log_dir=Path(tmpdir),
                service_name="test",
            )
            logger.info("Test", context={"user": "john", "action": "login"})


class TestMetrics(unittest.TestCase):
    """Test metrics collection."""

    def test_metrics_collector(self):
        """Test metrics collector initialization."""
        try:
            metrics = MetricsCollector(namespace="test")
            self.assertIsNotNone(metrics)
        except ImportError:
            self.skipTest("prometheus_client not installed")

    def test_track_operation(self):
        """Test operation tracking."""
        try:
            metrics = MetricsCollector(namespace="test")

            with metrics.track_operation("test_op"):
                time.sleep(0.01)

            # Should not raise
        except ImportError:
            self.skipTest("prometheus_client not installed")

    def test_record_error(self):
        """Test error recording."""
        try:
            metrics = MetricsCollector(namespace="test")
            metrics.record_error("test_op", "ValueError")
            # Should not raise
        except ImportError:
            self.skipTest("prometheus_client not installed")

    def test_record_factor_evaluation(self):
        """Test factor evaluation recording."""
        try:
            metrics = MetricsCollector(namespace="test")
            metrics.record_factor_evaluation("momentum", 0.123)
            # Should not raise
        except ImportError:
            self.skipTest("prometheus_client not installed")


class TestHealthChecks(unittest.TestCase):
    """Test health checks."""

    def test_health_manager(self):
        """Test health check manager."""
        manager = HealthCheckManager(service_name="test")
        self.assertIsNotNone(manager)

    def test_startup_probe(self):
        """Test startup probe."""
        manager = HealthCheckManager()
        result = manager.startup_probe()
        self.assertEqual(result['status'], 'starting')

        manager.mark_startup_complete()
        result = manager.startup_probe()
        self.assertEqual(result['status'], 'ready')

    def test_disk_space_check(self):
        """Test disk space check."""
        check = DiskSpaceCheck(path="/", threshold_percent=99.0)
        result = check.check()
        self.assertIn(result.status, [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY])
        self.assertIsNotNone(result.message)

    def test_memory_check(self):
        """Test memory check."""
        check = MemoryCheck(threshold_percent=99.0)
        result = check.check()
        self.assertIn(result.status, [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY])
        self.assertIsNotNone(result.message)

    def test_liveness_probe(self):
        """Test liveness probe."""
        manager = HealthCheckManager()
        manager.add_check(DiskSpaceCheck())
        result = manager.liveness_probe()
        self.assertIn(result['status'], [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY])

    def test_readiness_probe(self):
        """Test readiness probe."""
        manager = HealthCheckManager()
        manager.add_check(MemoryCheck())
        manager.mark_startup_complete()
        result = manager.readiness_probe()
        self.assertIn(result['status'], [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY, HealthStatus.DEGRADED])


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        config.validate()
        self.assertEqual(config.level, "INFO")
        self.assertEqual(config.format, "json")

    def test_invalid_log_level(self):
        """Test invalid log level."""
        config = LoggingConfig(level="INVALID")
        with self.assertRaises(Exception):
            config.validate()

    def test_file_logging_requires_dir(self):
        """Test file logging requires output directory."""
        config = LoggingConfig(enable_file=True, output_dir=None)
        with self.assertRaises(Exception):
            config.validate()


class TestMetricsConfig(unittest.TestCase):
    """Test metrics configuration."""

    def test_default_metrics_config(self):
        """Test default metrics configuration."""
        config = MetricsConfig()
        config.validate()
        self.assertTrue(config.enabled)
        self.assertEqual(config.export_port, 9090)

    def test_invalid_port(self):
        """Test invalid port."""
        config = MetricsConfig(export_port=70000)
        with self.assertRaises(Exception):
            config.validate()


class TestHealthCheckConfig(unittest.TestCase):
    """Test health check configuration."""

    def test_default_health_config(self):
        """Test default health check configuration."""
        config = HealthCheckConfig()
        config.validate()
        self.assertTrue(config.enabled)
        self.assertEqual(config.port, 8080)

    def test_threshold_validation(self):
        """Test threshold validation."""
        config = HealthCheckConfig(disk_space_threshold_percent=150.0)
        with self.assertRaises(Exception):
            config.validate()


class TestResourceLimitsConfig(unittest.TestCase):
    """Test resource limits configuration."""

    def test_default_resource_limits(self):
        """Test default resource limits."""
        config = ResourceLimitsConfig()
        config.validate()
        self.assertEqual(config.max_concurrent_evaluations, 100)

    def test_invalid_concurrency(self):
        """Test invalid concurrency."""
        config = ResourceLimitsConfig(max_concurrent_evaluations=-1)
        with self.assertRaises(Exception):
            config.validate()


class TestSecurityConfig(unittest.TestCase):
    """Test security configuration."""

    def test_default_security_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        config.validate()
        self.assertFalse(config.require_authentication)
        self.assertFalse(config.enable_tls)

    def test_tls_requires_certs(self):
        """Test TLS requires certificates."""
        config = SecurityConfig(enable_tls=True)
        with self.assertRaises(Exception):
            config.validate()


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    unittest.main()
