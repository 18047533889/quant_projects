"""
Factor Preprocess (FP) configuration.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from config.base_config import BaseConfig, ConfigValidationError


class FPConfig(BaseConfig):
    """
    Configuration for Factor Preprocess package.

    Handles data cleaning, normalization, and preprocessing.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize FP configuration."""
        # Default values
        self.project_name = "factor_preprocess"
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.cache_dir: Optional[Path] = None

        # Database configuration
        self.db_type = "duckdb"
        self.db_host: Optional[str] = None
        self.db_port: Optional[int] = None
        self.db_name = "preprocess.db"
        self.db_username: Optional[str] = None
        self.db_password: Optional[str] = None
        self.db_pool_size = 5
        self.db_timeout = 30

        # Cache configuration
        self.cache_backend = "disk"
        self.cache_max_size_mb = 2048
        self.cache_ttl_seconds = 7200
        self.redis_url: Optional[str] = None
        self.cache_disk_path: Optional[Path] = None

        # Logging configuration
        self.log_level = "INFO"
        self.log_file: Optional[Path] = None
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_max_bytes = 10485760
        self.log_backup_count = 5

        # Processing configuration
        self.chunk_size = 10000
        self.parallel_jobs = 4
        self.preprocessing_pipeline = ["clean", "normalize", "validate"]

        # Outlier detection
        self.outlier_method = "iqr"
        self.outlier_threshold = 3.0

        # Missing value handling
        self.missing_value_strategy = "forward_fill"

        super().__init__(config_dict)

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Project settings
        self.project_name = config_dict.get("project_name", self.project_name)
        self.input_dir = self.resolve_path(config_dict.get("input_dir"))
        self.output_dir = self.resolve_path(config_dict.get("output_dir"))
        self.cache_dir = self.resolve_path(config_dict.get("cache_dir"))

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
        self.chunk_size = config_dict.get("chunk_size", self.chunk_size)
        self.parallel_jobs = config_dict.get("parallel_jobs", self.parallel_jobs)
        self.preprocessing_pipeline = config_dict.get("preprocessing_pipeline", self.preprocessing_pipeline)

        # Outlier detection
        self.outlier_method = config_dict.get("outlier_method", self.outlier_method)
        self.outlier_threshold = config_dict.get("outlier_threshold", self.outlier_threshold)

        # Missing value handling
        self.missing_value_strategy = config_dict.get("missing_value_strategy", self.missing_value_strategy)

    def validate(self) -> None:
        """Validate configuration."""
        # Required paths
        if self.input_dir is None:
            raise ConfigValidationError("input_dir is required")
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
        if self.cache_backend == "disk":
            if not self.cache_disk_path and not self.cache_dir:
                raise ConfigValidationError("disk backend requires cache_disk_path or cache_dir")

        self.validate_positive_number(self.cache_max_size_mb, "cache_max_size_mb")
        if self.cache_ttl_seconds < 0:
            raise ConfigValidationError("cache_ttl_seconds must be non-negative")

        # Processing validation
        self.validate_positive_number(self.chunk_size, "chunk_size")
        self.validate_positive_number(self.parallel_jobs, "parallel_jobs")

        if self.chunk_size < 100:
            raise ConfigValidationError("chunk_size must be >= 100")
        if self.parallel_jobs > 64:
            raise ConfigValidationError("parallel_jobs must be <= 64")

        # Outlier validation
        valid_methods = ["iqr", "zscore", "isolation_forest", "none"]
        if self.outlier_method not in valid_methods:
            raise ConfigValidationError(
                f"Invalid outlier_method: {self.outlier_method}. Must be one of {valid_methods}"
            )

        self.validate_positive_number(self.outlier_threshold, "outlier_threshold")

        # Missing value validation
        valid_strategies = ["forward_fill", "backward_fill", "interpolate", "drop", "mean", "median", "zero"]
        if self.missing_value_strategy not in valid_strategies:
            raise ConfigValidationError(
                f"Invalid missing_value_strategy: {self.missing_value_strategy}. Must be one of {valid_strategies}"
            )

        # Pipeline validation
        valid_steps = ["clean", "normalize", "validate", "outlier_removal", "missing_value_handling", "winsorize"]
        for step in self.preprocessing_pipeline:
            if step not in valid_steps:
                raise ConfigValidationError(
                    f"Invalid pipeline step: {step}. Must be one of {valid_steps}"
                )

        # Logging validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ConfigValidationError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}"
            )

    def get_cache_path(self) -> Path:
        """
        Get effective cache path.

        Returns:
            Cache directory path
        """
        if self.cache_disk_path:
            return self.cache_disk_path
        elif self.cache_dir:
            return self.cache_dir
        else:
            return self.output_dir / "cache"
