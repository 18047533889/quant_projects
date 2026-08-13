"""
Example usage of the configuration management system.
"""

from pathlib import Path
import os
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ConfigLoader, QEConfig, FPConfig, FAConfig, FOConfig


def example_basic_loading():
    """Example: Basic configuration loading from file."""
    print("=" * 60)
    print("Example 1: Basic Configuration Loading")
    print("=" * 60)

    loader = ConfigLoader()
    config = loader.load(
        QEConfig,
        file_path=Path(__file__).parent / "qe_config.yaml",
        validate=True
    )

    print(f"Project: {config.project_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Parallel workers: {config.parallel_workers}")
    print(f"Database type: {config.db_type}")
    print(f"Cache backend: {config.cache_backend}")
    print(f"Validated: {config.is_validated()}")
    print()


def example_env_variables():
    """Example: Loading with environment variables."""
    print("=" * 60)
    print("Example 2: Environment Variable Loading")
    print("=" * 60)

    # Set some environment variables
    os.environ["QUANT_PROJECT_NAME"] = "custom_evaluator"
    os.environ["QUANT_PARALLEL_WORKERS"] = "8"
    os.environ["QUANT_ENABLE_PROFILING"] = "true"
    os.environ["QUANT_DATABASE__TYPE"] = "postgresql"
    os.environ["QUANT_DATABASE__HOST"] = "localhost"

    loader = ConfigLoader(env_prefix="QUANT_")

    # Load environment variables only
    env_config = loader.load_from_env()
    print(f"Environment config: {env_config}")
    print()

    # Clean up
    for key in list(os.environ.keys()):
        if key.startswith("QUANT_"):
            del os.environ[key]


def example_config_merging():
    """Example: Merging multiple configuration sources."""
    print("=" * 60)
    print("Example 3: Configuration Merging")
    print("=" * 60)

    loader = ConfigLoader()

    # Define different config layers
    defaults = {
        "project_name": "default_project",
        "parallel_workers": 2,
        "batch_size": 500
    }

    file_config = {
        "project_name": "file_project",
        "parallel_workers": 4,
        "data_dir": "/data"
    }

    overrides = {
        "parallel_workers": 16,
        "enable_profiling": True
    }

    # Merge: defaults < file < overrides
    merged = loader.merge_configs(defaults, file_config, overrides)

    print(f"Project name: {merged['project_name']}")  # from file
    print(f"Parallel workers: {merged['parallel_workers']}")  # from overrides
    print(f"Batch size: {merged['batch_size']}")  # from defaults
    print(f"Enable profiling: {merged['enable_profiling']}")  # from overrides
    print()


def example_all_packages():
    """Example: Loading configurations for all packages."""
    print("=" * 60)
    print("Example 4: All Package Configurations")
    print("=" * 60)

    loader = ConfigLoader()
    examples_dir = Path(__file__).parent

    # Load QE config
    try:
        qe_config = loader.load(
            QEConfig,
            file_path=examples_dir / "qe_config.yaml",
            validate=False  # Skip validation for demo
        )
        print(f"✓ QE Config loaded: {qe_config.project_name}")
    except Exception as e:
        print(f"✗ QE Config failed: {e}")

    # Load FP config
    try:
        fp_config = loader.load(
            FPConfig,
            file_path=examples_dir / "fp_config.yaml",
            validate=False
        )
        print(f"✓ FP Config loaded: {fp_config.project_name}")
        print(f"  - Outlier method: {fp_config.outlier_method}")
        print(f"  - Pipeline: {', '.join(fp_config.preprocessing_pipeline)}")
    except Exception as e:
        print(f"✗ FP Config failed: {e}")

    # Load FA config
    try:
        fa_config = loader.load(
            FAConfig,
            file_path=examples_dir / "fa_config.yaml",
            validate=False
        )
        print(f"✓ FA Config loaded: {fa_config.project_name}")
        print(f"  - Versioning enabled: {fa_config.enable_versioning}")
        print(f"  - Max versions: {fa_config.max_versions}")
    except Exception as e:
        print(f"✗ FA Config failed: {e}")

    # Load FO config
    try:
        fo_config = loader.load(
            FOConfig,
            file_path=examples_dir / "fo_config.yaml",
            validate=False
        )
        print(f"✓ FO Config loaded: {fo_config.project_name}")
        print(f"  - Optimization method: {fo_config.optimization_method}")
        print(f"  - Max iterations: {fo_config.max_iterations}")
    except Exception as e:
        print(f"✗ FO Config failed: {e}")

    print()


def example_validation():
    """Example: Configuration validation."""
    print("=" * 60)
    print("Example 5: Configuration Validation")
    print("=" * 60)

    from config.base_config import ConfigValidationError

    # Valid configuration
    try:
        config = QEConfig({
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "parallel_workers": 4,
            "batch_size": 1000
        })
        config.validate_and_raise()
        print("✓ Valid configuration passed")
    except ConfigValidationError as e:
        print(f"✗ Validation failed: {e}")

    # Invalid configuration - negative workers
    try:
        config = QEConfig({
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "parallel_workers": -5
        })
        config.validate_and_raise()
        print("✓ Invalid configuration passed (unexpected!)")
    except ConfigValidationError as e:
        print(f"✓ Correctly caught validation error: {e}")

    # Invalid configuration - missing required field
    try:
        config = QEConfig({
            "output_dir": "/tmp/output",
            # data_dir is missing
        })
        config.validate_and_raise()
        print("✓ Missing field passed (unexpected!)")
    except ConfigValidationError as e:
        print(f"✓ Correctly caught validation error: {e}")

    print()


def example_programmatic():
    """Example: Programmatic configuration building."""
    print("=" * 60)
    print("Example 6: Programmatic Configuration")
    print("=" * 60)

    # Build FO config programmatically
    config = FOConfig({
        "project_name": "genetic_optimizer",
        "input_dir": "/data/factors",
        "output_dir": "/data/optimized",
        "models_dir": "/data/models",
        "optimization_method": "genetic",
        "max_iterations": 200,
        "population_size": 100,
        "convergence_threshold": 1e-7,
        "feature_selection": True,
        "max_features": 30,
        "parallel_trials": 8
    })

    print(f"Project: {config.project_name}")
    print(f"Optimization: {config.optimization_method}")
    print(f"Population: {config.population_size}")
    print(f"Max features: {config.max_features}")

    # Get model path
    model_path = config.get_model_path("best_model")
    print(f"Model path: {model_path}")
    print()


def example_database_connection():
    """Example: Building database connection strings."""
    print("=" * 60)
    print("Example 7: Database Connection Strings")
    print("=" * 60)

    # SQLite/DuckDB
    config1 = QEConfig({
        "data_dir": "/tmp/data",
        "output_dir": "/tmp/output",
        "database": {
            "type": "duckdb",
            "database": "my_database.db"
        }
    })
    print(f"DuckDB: {config1.get_db_connection_string()}")

    # PostgreSQL
    config2 = QEConfig({
        "data_dir": "/tmp/data",
        "output_dir": "/tmp/output",
        "database": {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "quant_db",
            "username": "user",
            "password": "pass"
        }
    })
    print(f"PostgreSQL: {config2.get_db_connection_string()}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("Configuration Management System Examples")
    print("*" * 60)
    print("\n")

    example_basic_loading()
    example_env_variables()
    example_config_merging()
    example_all_packages()
    example_validation()
    example_programmatic()
    example_database_connection()

    print("*" * 60)
    print("All examples completed!")
    print("*" * 60)
    print()


if __name__ == "__main__":
    main()
