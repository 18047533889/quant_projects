# Configuration Management System

Centralized configuration management for quant_projects with validation, environment variable support, and YAML/JSON loading.

## Quick Start

```python
from config import ConfigLoader, QEConfig

loader = ConfigLoader()
config = loader.load(
    QEConfig,
    file_path="config/examples/qe_config.yaml",
    validate=True
)

print(f"Data directory: {config.data_dir}")
print(f"Parallel workers: {config.parallel_workers}")
```

## Features

- **Type-safe configurations** - Dedicated config classes for each package
- **Validation** - Built-in validation with clear error messages
- **Multiple sources** - Load from YAML, JSON, or environment variables
- **Configuration merging** - Priority-based merging: defaults < file < env < overrides
- **Path resolution** - Automatic path normalization and environment variable expansion
- **Pydantic schemas** - Optional automatic validation with Pydantic

## Available Configurations

### QEConfig - Quant Evaluator
Factor evaluation, backtesting, and performance analysis configuration.

### FPConfig - Factor Preprocess  
Data cleaning, normalization, and preprocessing configuration.

### FAConfig - Factor Assets
Factor storage, versioning, and archival configuration.

### FOConfig - Factor Optimizer
Factor optimization and feature selection configuration.

## Documentation

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for comprehensive documentation including:
- Detailed usage examples
- Environment variable configuration
- Validation guide
- Best practices
- API reference

## Examples

Run the examples:
```bash
python3 config/examples/usage_examples.py
```

See `config/examples/` for sample YAML and JSON configurations.

## Testing

```bash
pytest config/test_config.py -v
```

## Structure

```
config/
├── __init__.py              # Package exports
├── base_config.py           # Base configuration class
├── loader.py                # Configuration loader
├── schemas.py               # Pydantic schemas
├── qe_config.py            # Quant Evaluator config
├── fp_config.py            # Factor Preprocess config
├── fa_config.py            # Factor Assets config
├── fo_config.py            # Factor Optimizer config
├── test_config.py          # Unit tests
├── CONFIG_GUIDE.md         # Full documentation
├── README.md               # This file
└── examples/
    ├── qe_config.yaml
    ├── fp_config.yaml
    ├── fa_config.yaml
    ├── fo_config.yaml
    ├── qe_config_production.json
    └── usage_examples.py
```
