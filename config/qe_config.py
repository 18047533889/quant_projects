"""
Quant Evaluator (QE) configuration.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from config.base_config import BaseConfig, ConfigValidationError


class QEConfig(BaseConfig):
    """
    Configuration for Quant Evaluator package.

    Handles factor evaluation, backtesting, and performance analysis.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize QE configuration."""
        # Default values
        self.project_name = "quant_evaluator"
        self.data_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.temp_dir: Optional[Path] = None

        # Database configuration
        self.db_type = "duckdb"
        self.db_host: Optional[str] = None
        self.db_port: Optional[int] = None
        self.db_name = "evaluator.db"
        self.db_username: Optional[str] = None
        self.db_password: Optional[str] = None
        self.db_pool_size = 5
        self.db_timeout = 30

        # Cache configuration
        self.cache_backend = "memory"
        self.cache_max_size_mb = 1024
        self.cache_ttl_seconds = 3600
        self.redis_url: Optional[str] = None
        self.cache_disk_path: Optional[Path] = None

        # Logging configuration
        self.log_level = "INFO"
        self.log_file: Optional[Path] = None
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_max_bytes = 10485760
        self.log_backup_count = 5

        # Processing configuration
        self.parallel_workers = 4
        self.batch_size = 1000
        self.enable_profiling = False

        # Data sources
        self.data_sources: List[Dict[str, Any]] = []

        super().__init__(config_dict)

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Project settings
        self.project_name = config_dict.get("project_name", self.project_name)
        self.data_dir = self.resolve_path(config_dict.get("data_dir"))
        self.output_dir = self.resolve_path(config_dict.get("output_dir"))
        self.temp_dir = self.resolve_path(config_dict.get("temp_dir"))

        # Database settings
        db_config = config_dict.get("database", {})
        self.db_type = db_config.get("type", self.db_type)
        self.db_host = self.resolve_env_var(db_config.get("host"), self.db_host)
        self.db_port = db_config.get("port", self.db_port)
        self.db_name = db_config.get("database", self.db_name)
        self.db_username = self.resolve_env_var(db_config.get("username"), self.db_username)
        self.db_password = self.resolve_env_var(db_config.get("password"), self.db_password)
        self.db_pool_size = db_config.get("pool_size", self.db_pool_size)
        self.db_timeout = db_config.get("timeout", self.db_timeout)

        # Cache settings
        cache_config = config_dict.get("cache", {})
        self.cache_backend = cache_config.get("backend", self.cache_backend)
        self.cache_max_size_mb = cache_config.get("max_size_mb", self.cache_max_size_mb)
        self.cache_ttl_seconds = cache_config.get("ttl_seconds", self.cache_ttl_seconds)
        self.redis_url = self.resolve_env_var(cache_config.get("redis_url"), self.redis_url)
        self.cache_disk_path = self.resolve_path(cache_config.get("disk_path"))

        # Logging settings
        log_config = config_dict.get("logging", {})
        self.log_level = log_config.get("level", self.log_level)
        self.log_file = self.resolve_path(log_config.get("file"))
        self.log_format = log_config.get("format", self.log_format)
        self.log_max_bytes = log_config.get("max_bytes", self.log_max_bytes)
        self.log_backup_count = log_config.get("backup_count", self.log_backup_count)

        # Processing settings
        self.parallel_workers = config_dict.get("parallel_workers", self.parallel_workers)
        self.batch_size = config_dict.get("batch_size", self.batch_size)
        self.enable_profiling = config_dict.get("enable_profiling", self.enable_profiling)

        # Data sources
        self.data_sources = config_dict.get("data_sources", self.data_sources)

    def validate(self) -> None:
        """Validate configuration."""
        # Required paths
        if self.data_dir is None:
            raise ConfigValidationError("data_dir is required")
        if self.output_dir is None:
            raise ConfigValidationError("output_dir is required")

        # Database validation
        if self.db_type in ["postgresql", "mysql"]:
            if not self.db_host:
                raise ConfigValidationError(f"{self.db_type} requires db_host")
            if not self.db_username:
                raise ConfigValidationError(f"{self.db_type} requires db_username")

        if self.db_port is not None:
            if self.db_port < 1 or self.db_port > 65535:
                raise ConfigValidationError(f"Invalid db_port: {self.db_port}")

        self.validate_positive_number(self.db_pool_size, "db_pool_size")
        self.validate_positive_number(self.db_timeout, "db_timeout")

        # Cache validation
        if self.cache_backend == "redis" and not self.redis_url:
            raise ConfigValidationError("redis backend requires redis_url")
        if self.cache_backend == "disk" and not self.cache_disk_path:
            raise ConfigValidationError("disk backend requires cache_disk_path")

        self.validate_positive_number(self.cache_max_size_mb, "cache_max_size_mb")
        if self.cache_ttl_seconds < 0:
            raise ConfigValidationError("cache_ttl_seconds must be non-negative")

        # Processing validation
        self.validate_positive_number(self.parallel_workers, "parallel_workers")
        self.validate_positive_number(self.batch_size, "batch_size")

        if self.parallel_workers > 64:
            raise ConfigValidationError("parallel_workers must be <= 64")

        # Logging validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ConfigValidationError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}"
            )

    def get_db_connection_string(self) -> str:
        """
        Build database connection string.

        Returns:
            Database connection string
        """
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_name}"
        elif self.db_type == "duckdb":
            return f"duckdb:///{self.db_name}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port or 5432}/{self.db_name}"
        elif self.db_type == "mysql":
            return f"mysql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port or 3306}/{self.db_name}"
        else:
            raise ConfigValidationError(f"Unsupported database type: {self.db_type}")
