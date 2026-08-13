"""
Integration example: Using configuration in a real application.

This example shows how to integrate the configuration system into an actual
quant project application with proper error handling and logging setup.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ConfigLoader, QEConfig, ConfigValidationError


class QuantEvaluatorApp:
    """Example application using configuration management."""

    def __init__(self, config: QEConfig):
        """Initialize application with configuration."""
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging from configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(self.config.log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.log_level)
        formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if configured)
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.log_max_bytes,
                backupCount=self.config.log_backup_count
            )
            file_handler.setLevel(self.config.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def initialize(self) -> None:
        """Initialize application resources."""
        self.logger.info(f"Initializing {self.config.project_name}")

        # Create necessary directories
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.temp_dir:
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Data directory: {self.config.data_dir}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Database: {self.config.db_type}")
        self.logger.info(f"Cache backend: {self.config.cache_backend}")
        self.logger.info(f"Parallel workers: {self.config.parallel_workers}")

    def run(self) -> None:
        """Run the application."""
        self.logger.info("Starting evaluation process")

        # Simulate work
        self.logger.info(f"Processing with batch size: {self.config.batch_size}")
        self.logger.info(f"Using {self.config.parallel_workers} parallel workers")

        if self.config.enable_profiling:
            self.logger.info("Profiling enabled")

        self.logger.info("Evaluation completed successfully")

    def shutdown(self) -> None:
        """Cleanup and shutdown."""
        self.logger.info("Shutting down application")


def load_config_with_fallback(
    config_file: Optional[Path] = None,
    env_prefix: str = "QUANT_"
) -> QEConfig:
    """
    Load configuration with fallback strategy.

    Priority:
    1. Command-line specified config file
    2. Environment-specific config (dev/staging/prod)
    3. Default config
    4. Environment variables only

    Args:
        config_file: Optional config file path
        env_prefix: Environment variable prefix

    Returns:
        Loaded and validated configuration

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    loader = ConfigLoader(env_prefix=env_prefix)

    # Determine config file to use
    if config_file is None:
        # Check for environment-specific config
        import os
        env = os.getenv("ENVIRONMENT", "dev")
        examples_dir = Path(__file__).parent
        config_file = examples_dir / f"qe_config_{env}.yaml"

        if not config_file.exists():
            # Fall back to default
            config_file = examples_dir / "qe_config.yaml"

    # Set default values
    defaults = {
        "parallel_workers": 4,
        "batch_size": 1000,
        "enable_profiling": False
    }

    try:
        if config_file.exists():
            config = loader.load(
                QEConfig,
                file_path=config_file,
                defaults=defaults,
                use_env=True,
                validate=True
            )
            print(f"✓ Configuration loaded from: {config_file}")
        else:
            # Load from environment only
            config = loader.load(
                QEConfig,
                defaults=defaults,
                use_env=True,
                validate=True
            )
            print("✓ Configuration loaded from environment variables")

        return config

    except ConfigValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)


def main():
    """Main entry point with proper error handling."""
    import argparse

    parser = argparse.ArgumentParser(description="Quant Evaluator Application")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config_with_fallback(config_file=args.config)

        if args.validate_only:
            print("✓ Configuration is valid")
            print(f"  Project: {config.project_name}")
            print(f"  Data dir: {config.data_dir}")
            print(f"  Output dir: {config.output_dir}")
            print(f"  Database: {config.db_type}")
            print(f"  Workers: {config.parallel_workers}")
            sys.exit(0)

        # Run application
        app = QuantEvaluatorApp(config)
        app.initialize()
        app.run()
        app.shutdown()

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
