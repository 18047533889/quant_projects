"""
Base configuration class with validation and common utilities.
"""

from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
import os
from pathlib import Path


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class BaseConfig(ABC):
    """
    Abstract base class for all configuration objects.

    Provides:
    - Validation interface
    - Common utility methods
    - Environment variable resolution
    - Path normalization
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        self._raw_config = config_dict or {}
        self._validated = False
        self._load_from_dict(self._raw_config)

    @abstractmethod
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.

        Must be implemented by subclasses to populate their specific fields.

        Args:
            config_dict: Configuration dictionary
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """
        Validate configuration.

        Must be implemented by subclasses to check their specific constraints.

        Raises:
            ConfigValidationError: If validation fails
        """
        pass

    def validate_and_raise(self) -> None:
        """
        Validate configuration and mark as validated.

        Raises:
            ConfigValidationError: If validation fails
        """
        self.validate()
        self._validated = True

    def is_validated(self) -> bool:
        """Check if configuration has been validated."""
        return self._validated

    @staticmethod
    def resolve_env_var(value: Any, default: Any = None) -> Any:
        """
        Resolve environment variable references.

        If value is a string starting with '$', treat it as an environment
        variable name and return its value.

        Args:
            value: Value to resolve (can be "$VAR_NAME" or literal value)
            default: Default value if env var not found

        Returns:
            Resolved value or default

        Examples:
            resolve_env_var("$HOME") -> "/home/user"
            resolve_env_var("literal") -> "literal"
            resolve_env_var("$MISSING", "fallback") -> "fallback"
        """
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            return os.environ.get(var_name, default)
        return value if value is not None else default

    @staticmethod
    def resolve_path(path: Any, base_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Resolve and normalize file path.

        Handles:
        - Environment variable expansion
        - Relative path resolution against base_dir
        - ~ expansion
        - Absolute path normalization

        Args:
            path: Path to resolve (str, Path, or None)
            base_dir: Base directory for relative paths

        Returns:
            Resolved Path object or None
        """
        if path is None:
            return None

        if isinstance(path, str):
            path = BaseConfig.resolve_env_var(path)
            if path is None:
                return None

        path = Path(path).expanduser()

        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path

        return path.resolve()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with default fallback.

        Args:
            key: Configuration key (supports dot notation for nested)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._raw_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation
        """
        return self._raw_config.copy()

    @staticmethod
    def validate_required_fields(config: Dict[str, Any], required_fields: List[str], context: str = "") -> None:
        """
        Validate that required fields are present and non-empty.

        Args:
            config: Configuration dictionary
            required_fields: List of required field names
            context: Context string for error messages

        Raises:
            ConfigValidationError: If any required field is missing or empty
        """
        missing = []
        for field in required_fields:
            value = config.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)

        if missing:
            ctx = f" in {context}" if context else ""
            raise ConfigValidationError(
                f"Missing required fields{ctx}: {', '.join(missing)}"
            )

    @staticmethod
    def validate_positive_number(value: Any, field_name: str) -> None:
        """
        Validate that value is a positive number.

        Args:
            value: Value to validate
            field_name: Field name for error messages

        Raises:
            ConfigValidationError: If value is not a positive number
        """
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"{field_name} must be a number, got {type(value).__name__}"
            )
        if value <= 0:
            raise ConfigValidationError(
                f"{field_name} must be positive, got {value}"
            )

    @staticmethod
    def validate_path_exists(path: Optional[Path], field_name: str, must_exist: bool = False) -> None:
        """
        Validate that path exists if required.

        Args:
            path: Path to validate
            field_name: Field name for error messages
            must_exist: Whether path must exist

        Raises:
            ConfigValidationError: If path is required but doesn't exist
        """
        if path is None:
            return

        if must_exist and not path.exists():
            raise ConfigValidationError(
                f"{field_name} path does not exist: {path}"
            )

    def __repr__(self) -> str:
        """String representation."""
        validated = " (validated)" if self._validated else " (not validated)"
        return f"{self.__class__.__name__}{validated}"
