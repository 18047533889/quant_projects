"""
Pydantic schemas for configuration validation.

Provides typed schemas with automatic validation for all configuration types.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Database type enumeration."""
    POSTGRES = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"


class CacheBackend(str, Enum):
    """Cache backend enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class DatabaseSchema(BaseModel):
    """Database connection configuration schema."""

    type: DatabaseType = Field(default=DatabaseType.DUCKDB, description="Database type")
    host: Optional[str] = Field(default=None, description="Database host")
    port: Optional[int] = Field(default=None, description="Database port")
    database: str = Field(..., description="Database name or path")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    pool_size: int = Field(default=5, ge=1, le=100, description="Connection pool size")
    timeout: int = Field(default=30, ge=1, description="Connection timeout in seconds")

    @field_validator('port')
    @classmethod
    def validate_port(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @model_validator(mode='after')
    def validate_connection_params(self) -> 'DatabaseSchema':
        """Validate that required connection params are present for remote databases."""
        if self.type in [DatabaseType.POSTGRES, DatabaseType.MYSQL]:
            if not self.host:
                raise ValueError(f"{self.type.value} requires 'host' parameter")
            if not self.username:
                raise ValueError(f"{self.type.value} requires 'username' parameter")
        return self


class CacheSchema(BaseModel):
    """Cache configuration schema."""

    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend")
    max_size_mb: int = Field(default=1024, ge=1, description="Maximum cache size in MB")
    ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds (0 = no expiry)")
    redis_url: Optional[str] = Field(default=None, description="Redis URL (if using Redis backend)")
    disk_path: Optional[Path] = Field(default=None, description="Disk cache path (if using disk backend)")

    @model_validator(mode='after')
    def validate_backend_params(self) -> 'CacheSchema':
        """Validate backend-specific parameters."""
        if self.backend == CacheBackend.REDIS and not self.redis_url:
            raise ValueError("Redis backend requires 'redis_url'")
        if self.backend == CacheBackend.DISK and not self.disk_path:
            raise ValueError("Disk backend requires 'disk_path'")
        return self


class DataSourceSchema(BaseModel):
    """Data source configuration schema."""

    name: str = Field(..., description="Data source name")
    type: str = Field(..., description="Data source type (local, remote, api)")
    path: Optional[Path] = Field(default=None, description="Local path or base URL")
    credentials: Optional[Dict[str, str]] = Field(default=None, description="API credentials")
    timeout: int = Field(default=60, ge=1, description="Request timeout in seconds")
    retry_count: int = Field(default=3, ge=0, description="Number of retries")
    rate_limit: Optional[int] = Field(default=None, ge=1, description="Rate limit (requests per minute)")


class LoggingSchema(BaseModel):
    """Logging configuration schema."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    file: Optional[Path] = Field(default=None, description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    max_bytes: int = Field(default=10485760, ge=1024, description="Max log file size in bytes")
    backup_count: int = Field(default=5, ge=0, description="Number of backup log files")


class QEConfigSchema(BaseModel):
    """Quant Evaluator configuration schema."""

    project_name: str = Field(default="quant_evaluator", description="Project name")
    data_dir: Path = Field(..., description="Data directory path")
    output_dir: Path = Field(..., description="Output directory path")
    temp_dir: Optional[Path] = Field(default=None, description="Temporary directory path")

    database: DatabaseSchema = Field(default_factory=DatabaseSchema, description="Database configuration")
    cache: CacheSchema = Field(default_factory=CacheSchema, description="Cache configuration")
    logging: LoggingSchema = Field(default_factory=LoggingSchema, description="Logging configuration")

    parallel_workers: int = Field(default=4, ge=1, le=64, description="Number of parallel workers")
    batch_size: int = Field(default=1000, ge=1, description="Batch processing size")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")

    data_sources: List[DataSourceSchema] = Field(default_factory=list, description="Data source configurations")

    @field_validator('data_dir', 'output_dir', 'temp_dir')
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            return Path(v).expanduser().resolve()
        return v


class FPConfigSchema(BaseModel):
    """Factor Preprocess configuration schema."""

    project_name: str = Field(default="factor_preprocess", description="Project name")
    input_dir: Path = Field(..., description="Input data directory")
    output_dir: Path = Field(..., description="Processed output directory")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory")

    database: DatabaseSchema = Field(default_factory=DatabaseSchema, description="Database configuration")
    cache: CacheSchema = Field(default_factory=CacheSchema, description="Cache configuration")
    logging: LoggingSchema = Field(default_factory=LoggingSchema, description="Logging configuration")

    chunk_size: int = Field(default=10000, ge=100, description="Data chunk size for processing")
    parallel_jobs: int = Field(default=4, ge=1, le=64, description="Number of parallel jobs")

    preprocessing_pipeline: List[str] = Field(
        default_factory=lambda: ["clean", "normalize", "validate"],
        description="Preprocessing steps"
    )

    outlier_method: str = Field(default="iqr", description="Outlier detection method")
    outlier_threshold: float = Field(default=3.0, gt=0, description="Outlier threshold")

    missing_value_strategy: str = Field(default="forward_fill", description="Missing value handling strategy")

    @field_validator('input_dir', 'output_dir', 'cache_dir')
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            return Path(v).expanduser().resolve()
        return v


class FAConfigSchema(BaseModel):
    """Factor Assets configuration schema."""

    project_name: str = Field(default="factor_assets", description="Project name")
    factors_dir: Path = Field(..., description="Factors storage directory")
    metadata_dir: Path = Field(..., description="Factor metadata directory")
    archive_dir: Optional[Path] = Field(default=None, description="Factor archive directory")

    database: DatabaseSchema = Field(default_factory=DatabaseSchema, description="Database configuration")
    cache: CacheSchema = Field(default_factory=CacheSchema, description="Cache configuration")
    logging: LoggingSchema = Field(default_factory=LoggingSchema, description="Logging configuration")

    max_factor_age_days: int = Field(default=365, ge=1, description="Maximum factor age in days")
    auto_archive: bool = Field(default=True, description="Automatically archive old factors")
    compression: str = Field(default="gzip", description="Compression method for archived factors")

    factor_format: str = Field(default="parquet", description="Factor storage format")
    metadata_format: str = Field(default="json", description="Metadata storage format")

    enable_versioning: bool = Field(default=True, description="Enable factor versioning")
    max_versions: int = Field(default=10, ge=1, description="Maximum versions to keep")

    @field_validator('factors_dir', 'metadata_dir', 'archive_dir')
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            return Path(v).expanduser().resolve()
        return v


class FOConfigSchema(BaseModel):
    """Factor Optimizer configuration schema."""

    project_name: str = Field(default="factor_optimizer", description="Project name")
    input_dir: Path = Field(..., description="Input factors directory")
    output_dir: Path = Field(..., description="Optimized factors output directory")
    models_dir: Optional[Path] = Field(default=None, description="Trained models directory")

    database: DatabaseSchema = Field(default_factory=DatabaseSchema, description="Database configuration")
    cache: CacheSchema = Field(default_factory=CacheSchema, description="Cache configuration")
    logging: LoggingSchema = Field(default_factory=LoggingSchema, description="Logging configuration")

    optimization_method: str = Field(default="genetic", description="Optimization algorithm")
    max_iterations: int = Field(default=100, ge=1, description="Maximum optimization iterations")
    population_size: int = Field(default=50, ge=10, description="Population size for genetic algorithm")
    convergence_threshold: float = Field(default=1e-6, gt=0, description="Convergence threshold")

    feature_selection: bool = Field(default=True, description="Enable feature selection")
    max_features: Optional[int] = Field(default=None, ge=1, description="Maximum features to select")

    cross_validation_folds: int = Field(default=5, ge=2, description="Cross-validation folds")
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Test set size ratio")

    parallel_trials: int = Field(default=4, ge=1, le=64, description="Number of parallel optimization trials")

    @field_validator('input_dir', 'output_dir', 'models_dir')
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            return Path(v).expanduser().resolve()
        return v
