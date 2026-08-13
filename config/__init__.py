"""
Configuration Management System

This package provides centralized configuration management with validation,
environment variable support, and multiple file format loading.

Quick Start:
-----------
    from config import ConfigLoader, QEConfig

    loader = ConfigLoader()
    config = loader.load(
        QEConfig,
        file_path="config/examples/qe_config.yaml",
        validate=True
    )

Available Configurations:
------------------------
- QEConfig: Quant Evaluator configuration
- FPConfig: Factor Preprocess configuration
- FAConfig: Factor Assets configuration
- FOConfig: Factor Optimizer configuration

Features:
---------
- Type-safe configuration with validation
- YAML and JSON file loading
- Environment variable support with nested keys
- Configuration merging with priority
- Pydantic schemas for automatic validation
- Path resolution and normalization

See CONFIG_GUIDE.md for detailed documentation and examples.
"""

__version__ = "1.0.0"

from config.base_config import BaseConfig, ConfigValidationError
from config.loader import ConfigLoader
from config.qe_config import QEConfig
from config.fp_config import FPConfig
from config.fa_config import FAConfig
from config.fo_config import FOConfig

__all__ = [
    "BaseConfig",
    "ConfigValidationError",
    "ConfigLoader",
    "QEConfig",
    "FPConfig",
    "FAConfig",
    "FOConfig",
    "__version__",
]
