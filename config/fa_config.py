"""
Factor Assets (FA) configuration.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from config.base_config import BaseConfig, ConfigValidationError


class FAConfig(BaseConfig):
    """
    Configuration for Factor Assets package.

    Handles factor storage, versioning, and archival.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize FA configuration."""
        # Default values
        self.project_name = "factor_assets"
        self.factors_dir: Optional[Path] = None
        self.metadata_dir: Optional[Path] = None
        self.archive_dir: Optional[Path] = None

        # Database configuration
        self.db_type = "duckdb"
        self.db_host: Optional[str] = None
        self.db_port: Optional[int] = None
        self.db_name = "factor_assets.db"
        self.db_username: Optional[str] = None
        self.db_password: Optional[str] = None
        self.db_pool_size = 5
        self.db_timeout = 30

        # Cache configuration
        self.cache_backend = "memory"
        self.cache_max_size_mb = 512
        self.cache_ttl_seconds = 1800
        self.redis_url: Optional[str] = None
        self.cache_disk_path: Optional[Path] = None

        # Logging configuration
        self.log_level = "INFO"
        self.log_file: Optional[Path] = None
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_max_bytes = 10485760
        self.log_backup_count = 5

        # Factor management
        self.max_factor_age_days = 365
        self.auto_archive = True
        self.compression = "gzip"

        # Storage formats
        self.factor_format = "parquet"
        self.metadata_format = "json"

        # Versioning
        self.enable_versioning = True
        self.max_versions = 10

        super().__init__(config_dict)

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Project settings
        self.project_name = config_dict.get("project_name", self.project_name)
        self.factors_dir = self.resolve_path(config_dict.get("factors_dir"))
        self.metadata_dir = self.resolve_path(config_dict.get("metadata_dir"))
        self.archive_dir = self.resolve_path(config_dict.get("archive_dir"))

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

        # Factor management
        self.max_factor_age_days = config_dict.get("max_factor_age_days", self.max_factor_age_days)
        self.auto_archive = config_dict.get("auto_archive", self.auto_archive)
        self.compression = config_dict.get("compression", self.compression)

        # Storage formats
        self.factor_format = config_dict.get("factor_format", self.factor_format)
        self.metadata_format = config_dict.get("metadata_format", self.metadata_format)

        # Versioning
        self.enable_versioning = config_dict.get("enable_versioning", self.enable_versioning)
        self.max_versions = config_dict.get("max_versions", self.max_versions)

    def validate(self) -> None:
        """Validate configuration."""
        # Required paths
        if self.factors_dir is None:
            raise ConfigValidationError("factors_dir is required")
        if self.metadata_dir is None:
            raise ConfigValidationError("metadata_dir is required")

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

        # Factor management validation
        self.validate_positive_number(self.max_factor_age_days, "max_factor_age_days")

        valid_compressions = ["gzip", "bzip2", "lz4", "zstd", "none"]
        if self.compression not in valid_compressions:
            raise ConfigValidationError(
                f"Invalid compression: {self.compression}. Must be one of {valid_compressions}"
            )

        # Storage format validation
        valid_factor_formats = ["parquet", "feather", "hdf5", "csv"]
        if self.factor_format not in valid_factor_formats:
            raise ConfigValidationError(
                f"Invalid factor_format: {self.factor_format}. Must be one of {valid_factor_formats}"
            )

        valid_metadata_formats = ["json", "yaml", "toml"]
        if self.metadata_format not in valid_metadata_formats:
            raise ConfigValidationError(
                f"Invalid metadata_format: {self.metadata_format}. Must be one of {valid_metadata_formats}"
            )

        # Versioning validation
        if self.enable_versioning:
            self.validate_positive_number(self.max_versions, "max_versions")

        # Archival validation
        if self.auto_archive and self.archive_dir is None:
            raise ConfigValidationError("auto_archive requires archive_dir")

        # Logging validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ConfigValidationError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}"
            )

    def get_factor_path(self, factor_name: str, version: Optional[int] = None) -> Path:
        """
        Build path to factor file.

        Args:
            factor_name: Name of the factor
            version: Optional version number

        Returns:
            Path to factor file
        """
        if version is not None and self.enable_versioning:
            filename = f"{factor_name}_v{version}.{self.factor_format}"
        else:
            filename = f"{factor_name}.{self.factor_format}"

        return self.factors_dir / filename

    def get_metadata_path(self, factor_name: str) -> Path:
        """
        Build path to factor metadata file.

        Args:
            factor_name: Name of the factor

        Returns:
            Path to metadata file
        """
        filename = f"{factor_name}.{self.metadata_format}"
        return self.metadata_dir / filename
