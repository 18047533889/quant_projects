"""
Unit tests for configuration management system.
"""

import os
import pytest
from pathlib import Path
import tempfile
import yaml
import json

from config.base_config import BaseConfig, ConfigValidationError
from config.loader import ConfigLoader
from config.qe_config import QEConfig
from config.fp_config import FPConfig
from config.fa_config import FAConfig
from config.fo_config import FOConfig


class TestBaseConfig:
    """Test BaseConfig functionality."""

    def test_resolve_env_var(self):
        """Test environment variable resolution."""
        os.environ["TEST_VAR"] = "test_value"
        assert BaseConfig.resolve_env_var("$TEST_VAR") == "test_value"
        assert BaseConfig.resolve_env_var("literal") == "literal"
        assert BaseConfig.resolve_env_var("$MISSING", "default") == "default"
        del os.environ["TEST_VAR"]

    def test_resolve_path(self):
        """Test path resolution."""
        path = BaseConfig.resolve_path("/tmp/test")
        assert isinstance(path, Path)
        assert path == Path("/tmp/test")

        # Test relative path
        base_dir = Path("/base")
        path = BaseConfig.resolve_path("relative", base_dir)
        assert path == Path("/base/relative")

    def test_validate_required_fields(self):
        """Test required fields validation."""
        config = {"field1": "value1", "field2": "value2"}
        BaseConfig.validate_required_fields(config, ["field1", "field2"])

        with pytest.raises(ConfigValidationError):
            BaseConfig.validate_required_fields(config, ["field1", "missing"])

    def test_validate_positive_number(self):
        """Test positive number validation."""
        BaseConfig.validate_positive_number(5, "test_field")
        BaseConfig.validate_positive_number(1.5, "test_field")

        with pytest.raises(ConfigValidationError):
            BaseConfig.validate_positive_number(-5, "test_field")

        with pytest.raises(ConfigValidationError):
            BaseConfig.validate_positive_number("not_a_number", "test_field")


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def test_load_from_yaml(self, tmp_path):
        """Test loading from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "project_name": "test",
            "parallel_workers": 4,
            "database": {"type": "duckdb"}
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        loaded = loader.load_from_file(config_file)

        assert loaded["project_name"] == "test"
        assert loaded["parallel_workers"] == 4
        assert loaded["database"]["type"] == "duckdb"

    def test_load_from_json(self, tmp_path):
        """Test loading from JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "project_name": "test",
            "parallel_workers": 4
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        loader = ConfigLoader()
        loaded = loader.load_from_file(config_file)

        assert loaded["project_name"] == "test"
        assert loaded["parallel_workers"] == 4

    def test_load_from_env(self):
        """Test loading from environment variables."""
        os.environ["TEST_PROJECT_NAME"] = "env_project"
        os.environ["TEST_WORKERS"] = "8"
        os.environ["TEST_ENABLE"] = "true"
        os.environ["TEST_DB__HOST"] = "localhost"
        os.environ["TEST_DB__PORT"] = "5432"

        loader = ConfigLoader(env_prefix="TEST_")
        env_config = loader.load_from_env()

        assert env_config["project_name"] == "env_project"
        assert env_config["workers"] == 8
        assert env_config["enable"] is True
        assert env_config["db"]["host"] == "localhost"
        assert env_config["db"]["port"] == 5432

        # Clean up
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

    def test_merge_configs(self):
        """Test configuration merging."""
        loader = ConfigLoader()

        config1 = {"a": 1, "b": 2, "nested": {"x": 10}}
        config2 = {"b": 20, "c": 3, "nested": {"y": 20}}
        config3 = {"c": 30, "d": 4}

        merged = loader.merge_configs(config1, config2, config3)

        assert merged["a"] == 1
        assert merged["b"] == 20
        assert merged["c"] == 30
        assert merged["d"] == 4
        assert merged["nested"]["x"] == 10
        assert merged["nested"]["y"] == 20

    def test_convert_env_value(self):
        """Test environment value conversion."""
        assert ConfigLoader._convert_env_value("true") is True
        assert ConfigLoader._convert_env_value("false") is False
        assert ConfigLoader._convert_env_value("123") == 123
        assert ConfigLoader._convert_env_value("45.67") == 45.67
        assert ConfigLoader._convert_env_value("text") == "text"


class TestQEConfig:
    """Test QEConfig functionality."""

    def test_initialization(self):
        """Test QE config initialization."""
        config = QEConfig({
            "data_dir": "/data",
            "output_dir": "/output",
            "parallel_workers": 4
        })

        assert config.data_dir == Path("/data")
        assert config.output_dir == Path("/output")
        assert config.parallel_workers == 4

    def test_validation_success(self):
        """Test successful validation."""
        config = QEConfig({
            "data_dir": "/data",
            "output_dir": "/output",
            "parallel_workers": 4,
            "batch_size": 1000
        })

        config.validate_and_raise()
        assert config.is_validated()

    def test_validation_missing_required(self):
        """Test validation with missing required fields."""
        config = QEConfig({"output_dir": "/output"})

        with pytest.raises(ConfigValidationError, match="data_dir is required"):
            config.validate_and_raise()

    def test_validation_invalid_workers(self):
        """Test validation with invalid parallel workers."""
        config = QEConfig({
            "data_dir": "/data",
            "output_dir": "/output",
            "parallel_workers": -5
        })

        with pytest.raises(ConfigValidationError):
            config.validate_and_raise()

    def test_db_connection_string(self):
        """Test database connection string generation."""
        config = QEConfig({
            "data_dir": "/data",
            "output_dir": "/output",
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "user",
                "password": "pass"
            }
        })

        conn_str = config.get_db_connection_string()
        assert "postgresql://" in conn_str
        assert "localhost:5432" in conn_str
        assert "test_db" in conn_str


class TestFPConfig:
    """Test FPConfig functionality."""

    def test_initialization(self):
        """Test FP config initialization."""
        config = FPConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "chunk_size": 5000,
            "outlier_method": "zscore"
        })

        assert config.input_dir == Path("/input")
        assert config.chunk_size == 5000
        assert config.outlier_method == "zscore"

    def test_validation_invalid_outlier_method(self):
        """Test validation with invalid outlier method."""
        config = FPConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "outlier_method": "invalid_method",
            "cache": {"backend": "memory"}
        })

        with pytest.raises(ConfigValidationError, match="Invalid outlier_method"):
            config.validate_and_raise()

    def test_get_cache_path(self):
        """Test cache path resolution."""
        config = FPConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "cache_dir": "/cache"
        })

        assert config.get_cache_path() == Path("/cache")


class TestFAConfig:
    """Test FAConfig functionality."""

    def test_initialization(self):
        """Test FA config initialization."""
        config = FAConfig({
            "factors_dir": "/factors",
            "metadata_dir": "/metadata",
            "enable_versioning": True,
            "max_versions": 5
        })

        assert config.factors_dir == Path("/factors")
        assert config.enable_versioning is True
        assert config.max_versions == 5

    def test_get_factor_path(self):
        """Test factor path generation."""
        config = FAConfig({
            "factors_dir": "/factors",
            "metadata_dir": "/metadata",
            "enable_versioning": True
        })

        path = config.get_factor_path("my_factor", version=3)
        assert path == Path("/factors/my_factor_v3.parquet")

        path = config.get_factor_path("my_factor")
        assert path == Path("/factors/my_factor.parquet")

    def test_get_metadata_path(self):
        """Test metadata path generation."""
        config = FAConfig({
            "factors_dir": "/factors",
            "metadata_dir": "/metadata"
        })

        path = config.get_metadata_path("my_factor")
        assert path == Path("/metadata/my_factor.json")


class TestFOConfig:
    """Test FOConfig functionality."""

    def test_initialization(self):
        """Test FO config initialization."""
        config = FOConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "optimization_method": "bayesian",
            "max_iterations": 200
        })

        assert config.optimization_method == "bayesian"
        assert config.max_iterations == 200

    def test_validation_invalid_method(self):
        """Test validation with invalid optimization method."""
        config = FOConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "optimization_method": "invalid_method",
            "cache": {"backend": "memory"}
        })

        with pytest.raises(ConfigValidationError, match="Invalid optimization_method"):
            config.validate_and_raise()

    def test_get_model_path(self):
        """Test model path generation."""
        config = FOConfig({
            "input_dir": "/input",
            "output_dir": "/output",
            "models_dir": "/models"
        })

        path = config.get_model_path("my_model")
        assert path == Path("/models/my_model.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
