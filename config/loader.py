"""
Configuration loader with support for YAML, JSON, and environment variables.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path
from config.base_config import BaseConfig, ConfigValidationError


T = TypeVar('T', bound=BaseConfig)


class ConfigLoader:
    """
    Configuration loader with multiple source support.

    Supports:
    - YAML files
    - JSON files
    - Environment variables
    - Dictionary merging with priority
    """

    def __init__(self, env_prefix: str = "QUANT_"):
        """
        Initialize config loader.

        Args:
            env_prefix: Prefix for environment variable names
        """
        self.env_prefix = env_prefix

    def load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.

        Automatically detects format based on file extension.

        Args:
            file_path: Path to config file (.yaml, .yml, or .json)

        Returns:
            Configuration dictionary

        Raises:
            ConfigValidationError: If file format is unsupported or parsing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigValidationError(f"Config file not found: {file_path}")

        suffix = file_path.suffix.lower()

        try:
            with open(file_path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f)
                else:
                    raise ConfigValidationError(
                        f"Unsupported config file format: {suffix}. "
                        f"Use .yaml, .yml, or .json"
                    )
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigValidationError(f"Failed to parse {file_path}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load {file_path}: {e}")

    def load_from_env(self, nested: bool = True) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Environment variables should follow the pattern:
        {env_prefix}SECTION__KEY=value

        Args:
            nested: If True, create nested dict from __ separators

        Returns:
            Configuration dictionary

        Examples:
            QUANT_DATABASE__HOST=localhost -> {"database": {"host": "localhost"}}
            QUANT_LOG_LEVEL=INFO -> {"log_level": "INFO"}
        """
        config = {}

        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue

            # Remove prefix and convert to lowercase
            clean_key = key[len(self.env_prefix):].lower()

            if nested and '__' in clean_key:
                # Split into nested keys
                keys = clean_key.split('__')
                current = config

                # Navigate/create nested structure
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                # Set final value with type conversion
                current[keys[-1]] = self._convert_env_value(value)
            else:
                # Flat key
                config[clean_key] = self._convert_env_value(value)

        return config

    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """
        Convert environment variable string to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (bool, int, float, or str)
        """
        # Try boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.

        Later configs override earlier ones. Nested dicts are merged recursively.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        result = {}

        for config in configs:
            result = self._deep_merge(result, config)

        return result

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    def load(
        self,
        config_class: Type[T],
        file_path: Optional[Path] = None,
        defaults: Optional[Dict[str, Any]] = None,
        override: Optional[Dict[str, Any]] = None,
        use_env: bool = True,
        validate: bool = True
    ) -> T:
        """
        Load and build configuration with priority chain.

        Priority (lowest to highest):
        1. defaults
        2. file_path
        3. environment variables (if use_env=True)
        4. override

        Args:
            config_class: Configuration class to instantiate
            file_path: Optional config file path
            defaults: Optional default values
            override: Optional override values (highest priority)
            use_env: Whether to load from environment variables
            validate: Whether to validate after loading

        Returns:
            Configured and optionally validated config instance

        Raises:
            ConfigValidationError: If validation fails
        """
        config_parts = []

        # 1. Add defaults
        if defaults:
            config_parts.append(defaults)

        # 2. Load from file
        if file_path:
            config_parts.append(self.load_from_file(file_path))

        # 3. Load from environment
        if use_env:
            env_config = self.load_from_env()
            if env_config:
                config_parts.append(env_config)

        # 4. Add overrides
        if override:
            config_parts.append(override)

        # Merge all configs
        merged = self.merge_configs(*config_parts)

        # Instantiate config
        config = config_class(merged)

        # Validate if requested
        if validate:
            config.validate_and_raise()

        return config

    def save_to_file(self, config: BaseConfig, file_path: Path, format: Optional[str] = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration instance
            file_path: Output file path
            format: Format ('yaml' or 'json'). Auto-detected from extension if None

        Raises:
            ConfigValidationError: If format is unsupported
        """
        file_path = Path(file_path)

        # Determine format
        if format is None:
            suffix = file_path.suffix.lower()
            if suffix in ['.yaml', '.yml']:
                format = 'yaml'
            elif suffix == '.json':
                format = 'json'
            else:
                raise ConfigValidationError(
                    f"Cannot determine format from extension: {suffix}. "
                    f"Specify format explicitly."
                )

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict
        config_dict = config.to_dict()

        # Write file
        try:
            with open(file_path, 'w') as f:
                if format == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                elif format == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ConfigValidationError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to save config to {file_path}: {e}")
