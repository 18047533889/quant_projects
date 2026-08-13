# Configuration Management Guide

## Overview

The `config/` module provides a centralized configuration management system for all quant_projects packages. It supports:

- Type-safe configuration classes with validation
- YAML, JSON, and environment variable loading
- Configuration merging with priority chains
- Pydantic schemas for automatic validation

## Architecture

### Core Components

1. **BaseConfig** (`base_config.py`) - Abstract base class with validation interface
2. **ConfigLoader** (`loader.py`) - Loads and merges configurations from multiple sources
3. **Schemas** (`schemas.py`) - Pydantic schemas for type validation
4. **Package Configs** - Specific configurations for QE, FP, FA, FO packages

### Configuration Priority

Configurations are merged in priority order (lowest to highest):

1. Default values (hardcoded in config classes)
2. Configuration file (YAML/JSON)
3. Environment variables
4. Explicit overrides (passed programmatically)

## Package Configurations

### QEConfig - Quant Evaluator

Configuration for factor evaluation, backtesting, and performance analysis.

**Key Settings:**
- `data_dir`: Input data directory
- `output_dir`: Results output directory
- `parallel_workers`: Number of parallel workers
- `batch_size`: Batch processing size

### FPConfig - Factor Preprocess

Configuration for data cleaning, normalization, and preprocessing.

**Key Settings:**
- `input_dir`: Raw data input directory
- `output_dir`: Processed data output directory
- `chunk_size`: Data chunk size for processing
- `outlier_method`: Outlier detection method (iqr, zscore, isolation_forest)
- `missing_value_strategy`: Missing value handling strategy

### FAConfig - Factor Assets

Configuration for factor storage, versioning, and archival.

**Key Settings:**
- `factors_dir`: Factor storage directory
- `metadata_dir`: Factor metadata directory
- `enable_versioning`: Enable factor versioning
- `max_versions`: Maximum versions to keep
- `auto_archive`: Automatically archive old factors

### FOConfig - Factor Optimizer

Configuration for factor optimization and feature selection.

**Key Settings:**
- `input_dir`: Input factors directory
- `output_dir`: Optimized factors output directory
- `optimization_method`: Optimization algorithm (genetic, bayesian, grid_search)
- `max_iterations`: Maximum optimization iterations
- `feature_selection`: Enable feature selection

## Usage Examples

### Basic Usage

```python
from config import ConfigLoader, QEConfig

# Load from YAML file
loader = ConfigLoader()
config = loader.load(
    QEConfig,
    file_path="config/examples/qe_config.yaml",
    validate=True
)

print(f"Data directory: {config.data_dir}")
print(f"Parallel workers: {config.parallel_workers}")
```

### Loading with Environment Variables

```python
import os
from config import ConfigLoader, FPConfig

# Set environment variables
os.environ["QUANT_INPUT_DIR"] = "/data/raw"
os.environ["QUANT_PARALLEL_JOBS"] = "8"

# Load with env vars
loader = ConfigLoader(env_prefix="QUANT_")
config = loader.load(
    FPConfig,
    file_path="config/examples/fp_config.yaml",
    use_env=True,
    validate=True
)

print(f"Input directory: {config.input_dir}")
print(f"Parallel jobs: {config.parallel_jobs}")
```

### Configuration Merging

```python
from config import ConfigLoader, FAConfig

loader = ConfigLoader()

# Define defaults
defaults = {
    "factors_dir": "/default/factors",
    "max_versions": 5
}

# Define overrides
overrides = {
    "max_versions": 20,
    "enable_versioning": True
}

# Load and merge: defaults < file < env < overrides
config = loader.load(
    FAConfig,
    file_path="config/examples/fa_config.yaml",
    defaults=defaults,
    override=overrides,
    use_env=True,
    validate=True
)

print(f"Max versions: {config.max_versions}")  # 20 (from override)
```

### Programmatic Configuration

```python
from config import FOConfig

# Build config from dictionary
config_dict = {
    "project_name": "my_optimizer",
    "input_dir": "/data/factors",
    "output_dir": "/data/optimized",
    "optimization_method": "bayesian",
    "max_iterations": 200,
    "parallel_trials": 8
}

config = FOConfig(config_dict)
config.validate_and_raise()

print(f"Optimization method: {config.optimization_method}")
print(f"Max iterations: {config.max_iterations}")
```

### Saving Configuration

```python
from config import ConfigLoader, QEConfig
from pathlib import Path

loader = ConfigLoader()
config = loader.load(QEConfig, file_path="config/examples/qe_config.yaml")

# Save to new file
loader.save_to_file(
    config,
    file_path=Path("/tmp/my_config.yaml"),
    format="yaml"
)
```

## Environment Variables

Environment variables follow the pattern: `{PREFIX}SECTION__KEY=value`

### Examples

```bash
# Simple values
export QUANT_PROJECT_NAME=my_project
export QUANT_LOG_LEVEL=DEBUG

# Nested values (use double underscore)
export QUANT_DATABASE__TYPE=postgresql
export QUANT_DATABASE__HOST=localhost
export QUANT_DATABASE__PORT=5432

# Cache configuration
export QUANT_CACHE__BACKEND=redis
export QUANT_CACHE__REDIS_URL=redis://localhost:6379

# Type conversion (automatic)
export QUANT_PARALLEL_WORKERS=8          # Converted to int
export QUANT_ENABLE_PROFILING=true       # Converted to bool
export QUANT_TEST_SIZE=0.25              # Converted to float
```

### Sensitive Values

For sensitive values (passwords, API keys), use environment variables:

```yaml
# config.yaml
database:
  type: postgresql
  host: localhost
  username: $DB_USERNAME     # Will be resolved from env
  password: $DB_PASSWORD     # Will be resolved from env
```

```bash
export DB_USERNAME=myuser
export DB_PASSWORD=secret123
```

## Configuration Files

### YAML Format (Recommended)

```yaml
# config.yaml
project_name: my_project

database:
  type: duckdb
  database: mydb.db
  pool_size: 5

cache:
  backend: memory
  max_size_mb: 1024

logging:
  level: INFO
  file: /var/log/app.log
```

### JSON Format

```json
{
  "project_name": "my_project",
  "database": {
    "type": "duckdb",
    "database": "mydb.db",
    "pool_size": 5
  },
  "cache": {
    "backend": "memory",
    "max_size_mb": 1024
  }
}
```

## Validation

### Automatic Validation

All configurations can be automatically validated:

```python
from config import QEConfig, ConfigValidationError

try:
    config = QEConfig({
        "data_dir": "/data",
        "output_dir": "/output",
        "parallel_workers": -5  # Invalid!
    })
    config.validate_and_raise()
except ConfigValidationError as e:
    print(f"Validation failed: {e}")
```

### Custom Validation

Extend config classes for custom validation:

```python
from config import BaseConfig, ConfigValidationError
from typing import Dict, Any

class MyConfig(BaseConfig):
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        self.min_value = config_dict.get("min_value", 0)
        self.max_value = config_dict.get("max_value", 100)
    
    def validate(self) -> None:
        if self.min_value >= self.max_value:
            raise ConfigValidationError(
                f"min_value ({self.min_value}) must be < max_value ({self.max_value})"
            )
```

## Pydantic Schemas

For stronger type validation, use Pydantic schemas:

```python
from config.schemas import QEConfigSchema
from pydantic import ValidationError

try:
    schema = QEConfigSchema(
        data_dir="/data",
        output_dir="/output",
        parallel_workers=4,
        batch_size=1000
    )
    print(schema.model_dump())
except ValidationError as e:
    print(f"Schema validation failed: {e}")
```

## Best Practices

### 1. Use Environment Variables for Secrets

Never hardcode sensitive values in config files:

```yaml
# Good
database:
  username: $DB_USERNAME
  password: $DB_PASSWORD

# Bad
database:
  username: myuser
  password: secret123
```

### 2. Validate Early

Always validate configurations at startup:

```python
config = loader.load(QEConfig, file_path="config.yaml", validate=True)
# Any validation errors will be caught immediately
```

### 3. Use Type Hints

Leverage type hints for better IDE support:

```python
from config import QEConfig
from pathlib import Path

def process_data(config: QEConfig) -> None:
    data_dir: Path = config.data_dir  # IDE knows this is a Path
    # ...
```

### 4. Separate Environments

Use different config files for different environments:

```
config/
  examples/
    qe_config_dev.yaml
    qe_config_staging.yaml
    qe_config_production.yaml
```

```python
import os
env = os.getenv("ENVIRONMENT", "dev")
config_file = f"config/examples/qe_config_{env}.yaml"
config = loader.load(QEConfig, file_path=config_file)
```

### 5. Document Configuration

Add comments to configuration files:

```yaml
# Data directories
data_dir: /data          # Raw input data
output_dir: /output      # Processed output

# Performance tuning
parallel_workers: 4      # Adjust based on CPU cores
batch_size: 1000         # Increase for better throughput
```

## Troubleshooting

### Configuration Not Loading

Check file path and format:

```python
from pathlib import Path

config_path = Path("config/examples/qe_config.yaml")
if not config_path.exists():
    print(f"Config file not found: {config_path}")
```

### Environment Variables Not Applied

Verify prefix and naming:

```python
loader = ConfigLoader(env_prefix="QUANT_")
env_config = loader.load_from_env()
print(f"Loaded env config: {env_config}")
```

### Validation Errors

Enable detailed error messages:

```python
try:
    config.validate_and_raise()
except ConfigValidationError as e:
    print(f"Validation error: {e}")
    print(f"Config state: {config.to_dict()}")
```

## API Reference

### BaseConfig

- `validate()` - Validate configuration (must be implemented by subclasses)
- `validate_and_raise()` - Validate and mark as validated
- `is_validated()` - Check if configuration has been validated
- `resolve_env_var(value, default)` - Resolve environment variable reference
- `resolve_path(path, base_dir)` - Resolve and normalize file path
- `get(key, default)` - Get configuration value with default fallback
- `to_dict()` - Convert configuration to dictionary

### ConfigLoader

- `load_from_file(file_path)` - Load configuration from file
- `load_from_env(nested)` - Load configuration from environment variables
- `merge_configs(*configs)` - Merge multiple configuration dictionaries
- `load(config_class, ...)` - Load and build configuration with priority chain
- `save_to_file(config, file_path, format)` - Save configuration to file

## Examples Directory

See `config/examples/` for complete configuration examples:

- `qe_config.yaml` - Quant Evaluator basic configuration
- `fp_config.yaml` - Factor Preprocess configuration
- `fa_config.yaml` - Factor Assets configuration
- `fo_config.yaml` - Factor Optimizer configuration
- `qe_config_production.json` - Production configuration with remote data sources
