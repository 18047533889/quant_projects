#!/usr/bin/env python3
"""
Verification script for configuration management system.

Runs comprehensive checks to ensure all components are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BaseConfig,
    ConfigLoader,
    QEConfig,
    FPConfig,
    FAConfig,
    FOConfig,
    ConfigValidationError
)


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from config import schemas
        print("  ✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config_classes():
    """Test that all config classes can be instantiated."""
    print("\nTesting config class instantiation...")

    tests = [
        (QEConfig, {"data_dir": "/tmp/data", "output_dir": "/tmp/output"}),
        (FPConfig, {"input_dir": "/tmp/input", "output_dir": "/tmp/output"}),
        (FAConfig, {"factors_dir": "/tmp/factors", "metadata_dir": "/tmp/metadata"}),
        (FOConfig, {"input_dir": "/tmp/input", "output_dir": "/tmp/output"}),
    ]

    all_passed = True
    for config_class, config_dict in tests:
        try:
            config = config_class(config_dict)
            print(f"  ✓ {config_class.__name__}: {config.project_name}")
        except Exception as e:
            print(f"  ✗ {config_class.__name__} failed: {e}")
            all_passed = False

    return all_passed


def test_file_loading():
    """Test loading from YAML and JSON files."""
    print("\nTesting file loading...")

    import os
    examples_dir = Path(__file__).parent / "examples"
    loader = ConfigLoader()

    # Set required environment variables for production config
    os.environ["QUANT_DATA_DIR"] = "/tmp/quant_data"
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_USERNAME"] = "user"
    os.environ["DB_PASSWORD"] = "pass"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["API_KEY"] = "test_key"

    test_files = [
        (examples_dir / "qe_config.yaml", QEConfig),
        (examples_dir / "fp_config.yaml", FPConfig),
        (examples_dir / "fa_config.yaml", FAConfig),
        (examples_dir / "fo_config.yaml", FOConfig),
        (examples_dir / "qe_config_production.json", QEConfig),
    ]

    all_passed = True
    for file_path, config_class in test_files:
        try:
            if file_path.exists():
                config = loader.load(config_class, file_path=file_path, validate=True)
                print(f"  ✓ Loaded {file_path.name}")
            else:
                print(f"  ⚠ Skipped {file_path.name} (not found)")
        except Exception as e:
            print(f"  ✗ Failed to load {file_path.name}: {e}")
            all_passed = False

    # Clean up environment variables
    for key in ["QUANT_DATA_DIR", "DB_HOST", "DB_USERNAME", "DB_PASSWORD", "REDIS_URL", "API_KEY"]:
        os.environ.pop(key, None)

    return all_passed


def test_validation():
    """Test configuration validation."""
    print("\nTesting validation...")

    all_passed = True

    # Test valid configuration
    try:
        config = QEConfig({
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "parallel_workers": 4
        })
        config.validate_and_raise()
        print("  ✓ Valid configuration passed validation")
    except Exception as e:
        print(f"  ✗ Valid configuration failed: {e}")
        all_passed = False

    # Test invalid configuration (should raise error)
    try:
        config = QEConfig({
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/output",
            "parallel_workers": -5  # Invalid
        })
        config.validate_and_raise()
        print("  ✗ Invalid configuration should have failed validation")
        all_passed = False
    except ConfigValidationError:
        print("  ✓ Invalid configuration correctly rejected")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        all_passed = False

    return all_passed


def test_environment_loading():
    """Test loading from environment variables."""
    print("\nTesting environment variable loading...")

    import os

    # Set test environment variables
    os.environ["TEST_PROJECT_NAME"] = "test_project"
    os.environ["TEST_PARALLEL_WORKERS"] = "8"
    os.environ["TEST_DATABASE__TYPE"] = "postgresql"

    try:
        loader = ConfigLoader(env_prefix="TEST_")
        env_config = loader.load_from_env()

        if env_config.get("project_name") == "test_project":
            print("  ✓ String value loaded correctly")
        else:
            print("  ✗ String value incorrect")
            return False

        if env_config.get("parallel_workers") == 8:
            print("  ✓ Integer value converted correctly")
        else:
            print("  ✗ Integer value conversion failed")
            return False

        if env_config.get("database", {}).get("type") == "postgresql":
            print("  ✓ Nested value loaded correctly")
        else:
            print("  ✗ Nested value loading failed")
            return False

        # Clean up
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

        return True

    except Exception as e:
        print(f"  ✗ Environment loading failed: {e}")
        return False


def test_config_merging():
    """Test configuration merging."""
    print("\nTesting configuration merging...")

    try:
        loader = ConfigLoader()

        config1 = {"a": 1, "b": 2, "nested": {"x": 10}}
        config2 = {"b": 20, "c": 3, "nested": {"y": 20}}

        merged = loader.merge_configs(config1, config2)

        if merged["a"] == 1 and merged["b"] == 20 and merged["c"] == 3:
            print("  ✓ Top-level merging correct")
        else:
            print("  ✗ Top-level merging failed")
            return False

        if merged["nested"]["x"] == 10 and merged["nested"]["y"] == 20:
            print("  ✓ Nested merging correct")
        else:
            print("  ✗ Nested merging failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Config merging failed: {e}")
        return False


def test_path_resolution():
    """Test path resolution."""
    print("\nTesting path resolution...")

    try:
        # Absolute path
        path = BaseConfig.resolve_path("/absolute/path")
        if str(path) == "/absolute/path":
            print("  ✓ Absolute path resolved correctly")
        else:
            print("  ✗ Absolute path resolution failed")
            return False

        # Relative path
        base = Path("/base")
        path = BaseConfig.resolve_path("relative", base)
        if path == Path("/base/relative"):
            print("  ✓ Relative path resolved correctly")
        else:
            print("  ✗ Relative path resolution failed")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Path resolution failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("Configuration Management System Verification")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Config Classes": test_config_classes(),
        "File Loading": test_file_loading(),
        "Validation": test_validation(),
        "Environment Variables": test_environment_loading(),
        "Config Merging": test_config_merging(),
        "Path Resolution": test_path_resolution(),
    }

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())
    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if all_passed:
        print("\n✓ All verification tests passed!")
        return 0
    else:
        print("\n✗ Some verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
