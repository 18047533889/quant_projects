# Configuration Quick Reference

## Basic Usage

```python
from config import ConfigLoader, QEConfig

# Load from YAML file
loader = ConfigLoader()
config = loader.load(QEConfig, file_path="config.yaml", validate=True)

# Access configuration
print(config.data_dir)
print(config.parallel_workers)
```

## Loading Strategies

### From File
```python
config = loader.load(QEConfig, file_path="config.yaml")
```

### From Environment Variables
```bash
export QUANT_DATA_DIR=/data
export QUANT_PARALLEL_WORKERS=8
export QUANT_DATABASE__TYPE=postgresql
```

```python
loader = ConfigLoader(env_prefix="QUANT_")
config = loader.load(QEConfig, use_env=True)
```

### With Defaults and Overrides
```python
config = loader.load(
    QEConfig,
    file_path="config.yaml",
    defaults={"parallel_workers": 4},
    override={"enable_profiling": True},
    use_env=True,
    validate=True
)
```

## Available Configurations

| Config | Purpose | Key Settings |
|--------|---------|--------------|
| `QEConfig` | Quant Evaluator | data_dir, parallel_workers, database, cache |
| `FPConfig` | Factor Preprocess | outlier_method, pipeline, chunk_size |
| `FAConfig` | Factor Assets | versioning, storage_format, max_versions |
| `FOConfig` | Factor Optimizer | optimization_method, max_iterations, population_size |

## Configuration Files

### YAML Example
```yaml
project_name: my_project
data_dir: /data
parallel_workers: 8

database:
  type: postgresql
  host: localhost
  port: 5432
  database: quant_db

cache:
  backend: redis
  redis_url: redis://localhost:6379/0
```

### JSON Example
```json
{
  "project_name": "my_project",
  "data_dir": "/data",
  "parallel_workers": 8,
  "database": {
    "type": "postgresql",
    "host": "localhost"
  }
}
```

## Environment Variables

### Nested Keys
Use double underscore for nesting:
```bash
QUANT_DATABASE__TYPE=postgresql
QUANT_DATABASE__HOST=localhost
QUANT_CACHE__BACKEND=redis
```

### Secret References
Reference environment variables in config files:
```yaml
database:
  password: $DB_PASSWORD  # Will read from DB_PASSWORD env var
```

## Validation

```python
try:
    config.validate_and_raise()
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

### Common Validations
- Required fields (data_dir, output_dir)
- Positive numbers (parallel_workers, batch_size)
- Valid enums (database type, cache backend)
- Path existence
- Cross-field constraints

## Path Resolution

```python
# Absolute paths
data_dir: /absolute/path/to/data

# Relative paths (relative to config file)
data_dir: ./relative/data

# Home directory expansion
data_dir: ~/data

# Environment variable expansion
data_dir: $DATA_ROOT/data
```

## Database Connection

```python
# Get connection string
conn_str = config.get_db_connection_string()

# Examples:
# DuckDB: duckdb:///path/to/database.db
# PostgreSQL: postgresql://user:pass@host:5432/db
# MySQL: mysql://user:pass@host:3306/db
```

## Common Patterns

### Development vs Production
```python
import os
env = os.getenv("ENVIRONMENT", "dev")
config_file = f"config/config_{env}.yaml"
config = loader.load(QEConfig, file_path=config_file)
```

### With Fallback
```python
try:
    config = loader.load(QEConfig, file_path="config.yaml")
except FileNotFoundError:
    config = loader.load(QEConfig, use_env=True)
```

### Validation Only
```python
config = QEConfig(config_dict)
if not config.validate():
    errors = config.get_validation_errors()
    for error in errors:
        print(f"Error: {error}")
```

## Package-Specific Features

### QEConfig
```python
# Database connection
conn_str = config.get_db_connection_string()

# Cache path
cache_path = config.get_cache_path()
```

### FPConfig
```python
# Pipeline steps
for step in config.pipeline_steps:
    print(step)  # clean, outlier_removal, normalize, etc.

# Cache path
cache_path = config.get_cache_path()
```

### FAConfig
```python
# Factor path with versioning
path = config.get_factor_path("my_factor", version=3)
# -> /factors/my_factor_v3.parquet

# Metadata path
meta_path = config.get_metadata_path("my_factor")
# -> /metadata/my_factor.json
```

### FOConfig
```python
# Model path
model_path = config.get_model_path("best_model")
# -> /models/best_model.pkl

# Results path
results_path = config.get_results_path("optimization_results")
# -> /results/optimization_results.json
```

## Testing Your Config

```bash
# Validate configuration
python3 config/examples/integration_example.py --config myconfig.yaml --validate-only

# Run with configuration
python3 config/examples/integration_example.py --config myconfig.yaml
```

## Debugging

### Print Configuration
```python
print(config.to_dict())
```

### Check Validation Status
```python
if config.is_validated():
    print("Config is valid")
else:
    print("Config not validated yet")
```

### Get Validation Errors
```python
errors = config.get_validation_errors()
for error in errors:
    print(f"Validation error: {error}")
```

## More Information

- Full documentation: `config/CONFIG_GUIDE.md`
- Usage examples: `config/examples/usage_examples.py`
- Integration example: `config/examples/integration_example.py`
- Unit tests: `config/test_config.py`
