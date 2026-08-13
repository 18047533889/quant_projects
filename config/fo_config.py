"""
Factor Optimizer (FO) configuration.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from config.base_config import BaseConfig, ConfigValidationError


class FOConfig(BaseConfig):
    """
    Configuration for Factor Optimizer package.

    Handles factor optimization, feature selection, and model training.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize FO configuration."""
        # Default values
        self.project_name = "factor_optimizer"
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.models_dir: Optional[Path] = None

        # Database configuration
        self.db_type = "duckdb"
        self.db_host: Optional[str] = None
        self.db_port: Optional[int] = None
        self.db_name = "optimizer.db"
        self.db_username: Optional[str] = None
        self.db_password: Optional[str] = None
        self.db_pool_size = 5
        self.db_timeout = 30

        # Cache configuration
        self.cache_backend = "disk"
        self.cache_max_size_mb = 2048
        self.cache_ttl_seconds = 3600
        self.redis_url: Optional[str] = None
        self.cache_disk_path: Optional[Path] = None

        # Logging configuration
        self.log_level = "INFO"
        self.log_file: Optional[Path] = None
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log_max_bytes = 10485760
        self.log_backup_count = 5

        # Optimization configuration
        self.optimization_method = "genetic"
        self.max_iterations = 100
        self.population_size = 50
        self.convergence_threshold = 1e-6

        # Feature selection
        self.feature_selection = True
        self.max_features: Optional[int] = None

        # Cross-validation
        self.cross_validation_folds = 5
        self.test_size = 0.2

        # Parallel processing
        self.parallel_trials = 4

        super().__init__(config_dict)

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Project settings
        self.project_name = config_dict.get("project_name", self.project_name)
        self.input_dir = self.resolve_path(config_dict.get("input_dir"))
        self.output_dir = self.resolve_path(config_dict.get("output_dir"))
        self.models_dir = self.resolve_path(config_dict.get("models_dir"))

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

        # Optimization settings
        self.optimization_method = config_dict.get("optimization_method", self.optimization_method)
        self.max_iterations = config_dict.get("max_iterations", self.max_iterations)
        self.population_size = config_dict.get("population_size", self.population_size)
        self.convergence_threshold = config_dict.get("convergence_threshold", self.convergence_threshold)

        # Feature selection
        self.feature_selection = config_dict.get("feature_selection", self.feature_selection)
        self.max_features = config_dict.get("max_features", self.max_features)

        # Cross-validation
        self.cross_validation_folds = config_dict.get("cross_validation_folds", self.cross_validation_folds)
        self.test_size = config_dict.get("test_size", self.test_size)

        # Parallel processing
        self.parallel_trials = config_dict.get("parallel_trials", self.parallel_trials)

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
        if self.cache_backend == "disk" and not self.cache_disk_path:
            raise ConfigValidationError("disk backend requires cache_disk_path")

        self.validate_positive_number(self.cache_max_size_mb, "cache_max_size_mb")
        if self.cache_ttl_seconds < 0:
            raise ConfigValidationError("cache_ttl_seconds must be non-negative")

        # Optimization validation
        valid_methods = ["genetic", "bayesian", "grid_search", "random_search", "pso", "simulated_annealing"]
        if self.optimization_method not in valid_methods:
            raise ConfigValidationError(
                f"Invalid optimization_method: {self.optimization_method}. Must be one of {valid_methods}"
            )

        self.validate_positive_number(self.max_iterations, "max_iterations")
        self.validate_positive_number(self.population_size, "population_size")
        self.validate_positive_number(self.convergence_threshold, "convergence_threshold")

        if self.population_size < 10:
            raise ConfigValidationError("population_size must be >= 10")

        # Feature selection validation
        if self.max_features is not None:
            self.validate_positive_number(self.max_features, "max_features")

        # Cross-validation validation
        if self.cross_validation_folds < 2:
            raise ConfigValidationError("cross_validation_folds must be >= 2")

        if self.test_size <= 0 or self.test_size >= 1:
            raise ConfigValidationError("test_size must be between 0 and 1")

        # Parallel processing validation
        self.validate_positive_number(self.parallel_trials, "parallel_trials")
        if self.parallel_trials > 64:
            raise ConfigValidationError("parallel_trials must be <= 64")

        # Logging validation
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ConfigValidationError(
                f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}"
            )

    def get_model_path(self, model_name: str) -> Path:
        """
        Build path to model file.

        Args:
            model_name: Name of the model

        Returns:
            Path to model file
        """
        if self.models_dir:
            return self.models_dir / f"{model_name}.pkl"
        else:
            return self.output_dir / "models" / f"{model_name}.pkl"
