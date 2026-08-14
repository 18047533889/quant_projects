"""
Production configuration with enterprise-grade settings.

Provides:
- Structured logging with rotation
- Monitoring and metrics
- Health checks
- Resource limits
- Security settings
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from .base_config import BaseConfig, ConfigValidationError


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enumeration."""
    JSON = "json"
    TEXT = "text"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    output_dir: Optional[Path] = None
    max_file_size_mb: int = 100
    backup_count: int = 10
    enable_console: bool = True
    enable_file: bool = False  # Default to False to avoid requiring output_dir
    enable_syslog: bool = False
    syslog_address: Optional[str] = None
    sanitize_sensitive: bool = True
    include_context: bool = True

    def validate(self) -> None:
        """Validate logging configuration."""
        if self.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigValidationError(f"Invalid log level: {self.level}")

        if self.format not in ["json", "text"]:
            raise ConfigValidationError(f"Invalid log format: {self.format}")

        if self.enable_file and not self.output_dir:
            raise ConfigValidationError("output_dir required when enable_file=True")

        if self.max_file_size_mb <= 0:
            raise ConfigValidationError("max_file_size_mb must be positive")

        if self.backup_count < 0:
            raise ConfigValidationError("backup_count must be non-negative")

        if self.enable_syslog and not self.syslog_address:
            raise ConfigValidationError("syslog_address required when enable_syslog=True")


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    enabled: bool = True
    export_port: int = 9090
    export_path: str = "/metrics"
    collect_interval_seconds: int = 60

    # Performance metrics
    enable_latency_metrics: bool = True
    enable_throughput_metrics: bool = True
    enable_error_metrics: bool = True

    # Resource metrics
    enable_memory_metrics: bool = True
    enable_cpu_metrics: bool = True
    enable_disk_metrics: bool = True

    # Business metrics
    enable_factor_metrics: bool = True
    enable_evaluation_metrics: bool = True

    # Histogram buckets for latency (in seconds)
    latency_buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    def validate(self) -> None:
        """Validate metrics configuration."""
        if self.export_port < 1024 or self.export_port > 65535:
            raise ConfigValidationError(f"Invalid export_port: {self.export_port}")

        if not self.export_path.startswith("/"):
            raise ConfigValidationError("export_path must start with /")

        if self.collect_interval_seconds <= 0:
            raise ConfigValidationError("collect_interval_seconds must be positive")

        if not self.latency_buckets or not all(b > 0 for b in self.latency_buckets):
            raise ConfigValidationError("latency_buckets must be non-empty positive values")


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    port: int = 8080
    path: str = "/health"

    # Check intervals
    startup_probe_initial_delay_seconds: int = 10
    startup_probe_period_seconds: int = 5
    startup_probe_timeout_seconds: int = 3
    startup_probe_failure_threshold: int = 30

    liveness_probe_period_seconds: int = 10
    liveness_probe_timeout_seconds: int = 3
    liveness_probe_failure_threshold: int = 3

    readiness_probe_period_seconds: int = 10
    readiness_probe_timeout_seconds: int = 3
    readiness_probe_failure_threshold: int = 3

    # Component checks
    check_database: bool = True
    check_cache: bool = True
    check_disk_space: bool = True
    disk_space_threshold_percent: float = 90.0
    check_memory: bool = True
    memory_threshold_percent: float = 95.0

    def validate(self) -> None:
        """Validate health check configuration."""
        if self.port < 1024 or self.port > 65535:
            raise ConfigValidationError(f"Invalid health check port: {self.port}")

        if not self.path.startswith("/"):
            raise ConfigValidationError("path must start with /")

        if self.disk_space_threshold_percent < 0 or self.disk_space_threshold_percent > 100:
            raise ConfigValidationError("disk_space_threshold_percent must be 0-100")

        if self.memory_threshold_percent < 0 or self.memory_threshold_percent > 100:
            raise ConfigValidationError("memory_threshold_percent must be 0-100")


@dataclass
class ResourceLimitsConfig:
    """Resource limits configuration."""
    # Memory limits (in MB)
    max_memory_mb: Optional[int] = None
    warning_memory_threshold_percent: float = 80.0

    # CPU limits
    max_cpu_percent: Optional[float] = None

    # Concurrency limits
    max_concurrent_evaluations: int = 100
    max_concurrent_io_operations: int = 50
    max_queue_size: int = 1000

    # Timeout limits (in seconds)
    default_operation_timeout_seconds: int = 300
    max_operation_timeout_seconds: int = 3600

    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 1000

    def validate(self) -> None:
        """Validate resource limits configuration."""
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ConfigValidationError("max_memory_mb must be positive")

        if self.warning_memory_threshold_percent < 0 or self.warning_memory_threshold_percent > 100:
            raise ConfigValidationError("warning_memory_threshold_percent must be 0-100")

        if self.max_cpu_percent is not None and (self.max_cpu_percent <= 0 or self.max_cpu_percent > 100):
            raise ConfigValidationError("max_cpu_percent must be 0-100")

        if self.max_concurrent_evaluations <= 0:
            raise ConfigValidationError("max_concurrent_evaluations must be positive")

        if self.max_concurrent_io_operations <= 0:
            raise ConfigValidationError("max_concurrent_io_operations must be positive")

        if self.max_queue_size <= 0:
            raise ConfigValidationError("max_queue_size must be positive")


@dataclass
class SecurityConfig:
    """Security configuration."""
    # Authentication
    require_authentication: bool = False
    api_key_header: str = "X-API-Key"

    # TLS/SSL
    enable_tls: bool = False
    tls_cert_file: Optional[Path] = None
    tls_key_file: Optional[Path] = None

    # Request validation
    validate_input: bool = True
    max_request_size_mb: int = 10

    # CORS
    enable_cors: bool = False
    allowed_origins: List[str] = field(default_factory=list)

    # Audit logging
    enable_audit_log: bool = False  # Default to False to avoid requiring audit_log_path
    audit_log_path: Optional[Path] = None

    def validate(self) -> None:
        """Validate security configuration."""
        if self.enable_tls:
            if not self.tls_cert_file or not self.tls_key_file:
                raise ConfigValidationError("TLS cert and key files required when TLS enabled")

        if self.max_request_size_mb <= 0:
            raise ConfigValidationError("max_request_size_mb must be positive")

        if self.enable_audit_log and not self.audit_log_path:
            raise ConfigValidationError("audit_log_path required when audit logging enabled")


class ProductionConfig(BaseConfig):
    """
    Production configuration with all enterprise features.

    Example:
        config = ProductionConfig.from_dict({
            "logging": {
                "level": "INFO",
                "output_dir": "/var/log/quant"
            },
            "metrics": {
                "enabled": True,
                "export_port": 9090
            }
        })
        config.validate_and_raise()
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.logging = LoggingConfig()
        self.metrics = MetricsConfig()
        self.health_check = HealthCheckConfig()
        self.resource_limits = ResourceLimitsConfig()
        self.security = SecurityConfig()

        # Environment
        self.environment: str = "production"
        self.service_name: str = "quant-platform"
        self.version: str = "unknown"

        # Graceful shutdown
        self.shutdown_timeout_seconds: int = 30
        self.enable_graceful_shutdown: bool = True

        super().__init__(config_dict)

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Environment settings
        self.environment = self.resolve_env_var(
            config_dict.get("environment", "production")
        )
        self.service_name = config_dict.get("service_name", "quant-platform")
        self.version = config_dict.get("version", "unknown")

        # Graceful shutdown
        self.shutdown_timeout_seconds = config_dict.get("shutdown_timeout_seconds", 30)
        self.enable_graceful_shutdown = config_dict.get("enable_graceful_shutdown", True)

        # Load sub-configurations
        if "logging" in config_dict:
            log_config = config_dict["logging"]
            self.logging = LoggingConfig(
                level=log_config.get("level", "INFO"),
                format=log_config.get("format", "json"),
                output_dir=self.resolve_path(log_config.get("output_dir")),
                max_file_size_mb=log_config.get("max_file_size_mb", 100),
                backup_count=log_config.get("backup_count", 10),
                enable_console=log_config.get("enable_console", True),
                enable_file=log_config.get("enable_file", True),
                enable_syslog=log_config.get("enable_syslog", False),
                syslog_address=log_config.get("syslog_address"),
                sanitize_sensitive=log_config.get("sanitize_sensitive", True),
                include_context=log_config.get("include_context", True),
            )

        if "metrics" in config_dict:
            metrics_config = config_dict["metrics"]
            self.metrics = MetricsConfig(
                enabled=metrics_config.get("enabled", True),
                export_port=metrics_config.get("export_port", 9090),
                export_path=metrics_config.get("export_path", "/metrics"),
                collect_interval_seconds=metrics_config.get("collect_interval_seconds", 60),
                enable_latency_metrics=metrics_config.get("enable_latency_metrics", True),
                enable_throughput_metrics=metrics_config.get("enable_throughput_metrics", True),
                enable_error_metrics=metrics_config.get("enable_error_metrics", True),
                enable_memory_metrics=metrics_config.get("enable_memory_metrics", True),
                enable_cpu_metrics=metrics_config.get("enable_cpu_metrics", True),
                enable_disk_metrics=metrics_config.get("enable_disk_metrics", True),
                enable_factor_metrics=metrics_config.get("enable_factor_metrics", True),
                enable_evaluation_metrics=metrics_config.get("enable_evaluation_metrics", True),
                latency_buckets=metrics_config.get("latency_buckets", [
                    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
                ]),
            )

        if "health_check" in config_dict:
            hc_config = config_dict["health_check"]
            self.health_check = HealthCheckConfig(
                enabled=hc_config.get("enabled", True),
                port=hc_config.get("port", 8080),
                path=hc_config.get("path", "/health"),
                check_database=hc_config.get("check_database", True),
                check_cache=hc_config.get("check_cache", True),
                check_disk_space=hc_config.get("check_disk_space", True),
                disk_space_threshold_percent=hc_config.get("disk_space_threshold_percent", 90.0),
                check_memory=hc_config.get("check_memory", True),
                memory_threshold_percent=hc_config.get("memory_threshold_percent", 95.0),
            )

        if "resource_limits" in config_dict:
            rl_config = config_dict["resource_limits"]
            self.resource_limits = ResourceLimitsConfig(
                max_memory_mb=rl_config.get("max_memory_mb"),
                warning_memory_threshold_percent=rl_config.get("warning_memory_threshold_percent", 80.0),
                max_cpu_percent=rl_config.get("max_cpu_percent"),
                max_concurrent_evaluations=rl_config.get("max_concurrent_evaluations", 100),
                max_concurrent_io_operations=rl_config.get("max_concurrent_io_operations", 50),
                max_queue_size=rl_config.get("max_queue_size", 1000),
                default_operation_timeout_seconds=rl_config.get("default_operation_timeout_seconds", 300),
                max_operation_timeout_seconds=rl_config.get("max_operation_timeout_seconds", 3600),
                enable_rate_limiting=rl_config.get("enable_rate_limiting", True),
                requests_per_minute=rl_config.get("requests_per_minute", 1000),
            )

        if "security" in config_dict:
            sec_config = config_dict["security"]
            self.security = SecurityConfig(
                require_authentication=sec_config.get("require_authentication", False),
                api_key_header=sec_config.get("api_key_header", "X-API-Key"),
                enable_tls=sec_config.get("enable_tls", False),
                tls_cert_file=self.resolve_path(sec_config.get("tls_cert_file")),
                tls_key_file=self.resolve_path(sec_config.get("tls_key_file")),
                validate_input=sec_config.get("validate_input", True),
                max_request_size_mb=sec_config.get("max_request_size_mb", 10),
                enable_cors=sec_config.get("enable_cors", False),
                allowed_origins=sec_config.get("allowed_origins", []),
                enable_audit_log=sec_config.get("enable_audit_log", True),
                audit_log_path=self.resolve_path(sec_config.get("audit_log_path")),
            )

    def validate(self) -> None:
        """Validate all configuration sections."""
        errors = []

        # Validate environment
        if self.environment not in ["production", "staging", "development"]:
            errors.append(f"Invalid environment: {self.environment}")

        if not self.service_name:
            errors.append("service_name is required")

        if self.shutdown_timeout_seconds <= 0:
            errors.append("shutdown_timeout_seconds must be positive")

        # Validate sub-configurations
        try:
            self.logging.validate()
        except ConfigValidationError as e:
            errors.append(f"Logging config: {e}")

        try:
            self.metrics.validate()
        except ConfigValidationError as e:
            errors.append(f"Metrics config: {e}")

        try:
            self.health_check.validate()
        except ConfigValidationError as e:
            errors.append(f"Health check config: {e}")

        try:
            self.resource_limits.validate()
        except ConfigValidationError as e:
            errors.append(f"Resource limits config: {e}")

        try:
            self.security.validate()
        except ConfigValidationError as e:
            errors.append(f"Security config: {e}")

        if errors:
            raise ConfigValidationError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Create configuration from environment variables."""
        import os

        config_dict = {
            "environment": os.getenv("QUANT_ENV", "production"),
            "service_name": os.getenv("SERVICE_NAME", "quant-platform"),
            "version": os.getenv("VERSION", "unknown"),
            "logging": {
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "format": os.getenv("LOG_FORMAT", "json"),
                "output_dir": os.getenv("LOG_DIR", "/var/log/quant"),
            },
            "metrics": {
                "enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true",
                "export_port": int(os.getenv("METRICS_PORT", "9090")),
            },
            "health_check": {
                "enabled": os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true",
                "port": int(os.getenv("HEALTH_CHECK_PORT", "8080")),
            },
        }

        return cls(config_dict)
