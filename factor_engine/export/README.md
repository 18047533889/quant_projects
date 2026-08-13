# Export Utilities

Comprehensive serialization, import, and conversion utilities for FactorEngine and DataAccess contracts.

## Installation

The export utilities are available in both `factor_engine/export` and `dataaccess/export` packages.

```python
# For FactorEngine contracts
from factor_engine.export import (
    serialize_to_json,
    serialize_to_parquet,
    serialize_to_feather,
    import_from_json,
    import_from_parquet,
    to_pandas,
    to_polars,
    to_arrow,
)

# For DataAccess contracts
from dataaccess.export import (
    serialize_to_json,
    import_from_json,
    to_pandas,
)
```

## Features

### 1. Serialization (`serializers.py`)

Serialize contracts to JSON, Parquet, and Feather formats:

```python
from factor_engine.modeling.contracts import ModelOperatorSpec, ModelExecutionClass
from factor_engine.export import serialize_to_json, serialize_to_parquet

# Single contract to JSON
spec = ModelOperatorSpec(
    canonical="my_operator",
    execution_class=ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR,
    semantic_role="model_feature",
)
json_str = serialize_to_json(spec, "spec.json")

# Batch contracts to Parquet
specs = [spec1, spec2, spec3]
serialize_to_parquet(specs, "specs.parquet", compression="snappy")

# Batch to directory
from factor_engine.export import serialize_batch
paths = serialize_batch(
    specs,
    "output_dir/",
    format="json",  # or "parquet", "feather"
    prefix="operator",
)
```

**Supported formats:**
- JSON: Human-readable, versioned, with type metadata
- Parquet: Columnar, compressed, efficient for batch processing
- Feather: Fast serialization, good for intermediate storage

### 2. Import (`importers.py`)

Deserialize contracts with automatic type reconstruction:

```python
from factor_engine.export import import_from_json, import_from_parquet
from factor_engine.modeling.contracts import ModelOperatorSpec

# Import with type reconstruction
spec = import_from_json("spec.json", target_type=ModelOperatorSpec)

# Import batch
specs = import_from_parquet("specs.parquet")

# Import from directory
from factor_engine.export import import_batch
contracts = import_batch("output_dir/", pattern="*.json")
```

**Type reconstruction:**
- Automatic if `__type__` metadata present
- Explicit via `target_type` parameter
- Handles nested dataclasses and enums

### 3. Data Conversion (`converters.py`)

Convert between pandas, polars, and arrow formats:

```python
from factor_engine.export import to_pandas, to_polars, to_arrow
import pandas as pd

# Dict to pandas
df = to_pandas({"a": 1, "b": 2})

# Pandas to arrow (zero-copy where possible)
table = to_arrow(df)

# Polars conversion
if polars_available:
    pl_df = to_polars(df)

# Batch conversion
from factor_engine.export import convert_batch
dfs = convert_batch([data1, data2, data3], target_format="pandas")
```

**Features:**
- Zero-copy conversions where possible
- Schema inference and validation
- Type mapping across formats
- Batch operations

### 4. Schema Versioning (`versioning.py`)

Manage schema evolution and migrations:

```python
from factor_engine.export import (
    SchemaVersion,
    VersionRegistry,
    register_migration,
    migrate_contract,
)

# Define version
version = SchemaVersion(
    version="2.0.0",
    contract_type="ModelOperatorSpec",
    schema_hash="abc123",
    created_at=datetime.now(timezone.utc),
    description="Added new_field",
    added_fields=("new_field",),
)

# Register migration
@register_migration("ModelOperatorSpec", "1.0.0", "2.0.0")
def migrate_v1_to_v2(data: dict) -> dict:
    data["new_field"] = "default_value"
    return data

# Migrate contract
old_data = {"canonical": "test", "version": "1.0.0"}
new_data = migrate_contract(old_data, "ModelOperatorSpec", "1.0.0", "2.0.0")
```

**Features:**
- Version registry with backward compatibility
- Migration path finding (multi-hop)
- Schema hash for integrity checking
- Breaking change tracking

## Supported Contracts

### FactorEngine Contracts

**Modeling:**
- `ModelOperatorSpec` - Operator specifications
- `RichModelTiming` - Timing contracts
- `SampleAdequacyContract` - Sample requirements
- `LabelContract` - Label definitions
- `DecisionClock` - Decision timing
- `PredictionBatch` - Prediction outputs
- `RegimeMetadata` - Regime information
- `FitFingerprint` - Model fingerprints
- `ParameterSearchPolicy` - Search policies

**Enums:**
- `ModelExecutionClass`
- `TimingKind`
- `ParamRole`
- `SessionPhase`

### DataAccess Contracts

**Read Contracts:**
- `DataSnapshot` - Data snapshots
- `FileVersion` - File versions
- `ReadLineage` - Data lineage
- `ReadStats` - Read statistics

**Schema:**
- `SchemaEpoch` - Schema versions
- `ContractIR` - Contract IR
- `RuntimeContract` - Runtime contracts

## Usage Examples

### Export Model Specifications

```python
from factor_engine.modeling.contracts import ModelOperatorSpec, ModelExecutionClass
from factor_engine.export import serialize_to_parquet

# Create specifications
specs = [
    ModelOperatorSpec(
        canonical=f"operator_{i}",
        execution_class=ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR,
        semantic_role="model_feature",
    )
    for i in range(100)
]

# Export to Parquet (efficient for large batches)
serialize_to_parquet(specs, "operator_specs.parquet")

# Later: import and use
from factor_engine.export import import_from_parquet
loaded_specs = import_from_parquet("operator_specs.parquet")
```

### Version Migration

```python
from factor_engine.export import VersionRegistry, migrate_contract

# Set up registry
registry = VersionRegistry()

# Define migrations
def v1_to_v2(data):
    data["cost_class"] = "medium"  # new required field
    return data

def v2_to_v3(data):
    if "deprecated_field" in data:
        data["new_field"] = data.pop("deprecated_field")
    return data

registry.register_migration("ModelOperatorSpec", "1.0", "2.0", v1_to_v2)
registry.register_migration("ModelOperatorSpec", "2.0", "3.0", v2_to_v3)

# Migrate from v1 to v3 (automatic path finding)
old_contract = {"canonical": "test", "__version__": "1.0"}
new_contract = registry.migrate(old_contract, "ModelOperatorSpec", "1.0", "3.0")
```

### Cross-Format Pipeline

```python
from factor_engine.export import (
    import_from_json,
    to_pandas,
    to_arrow,
    serialize_to_parquet,
)

# Import from JSON
contracts = [import_from_json(f"contract_{i}.json") for i in range(10)]

# Convert to DataFrame for analysis
df = to_pandas(contracts)

# Filter and transform
filtered = df[df["cost_class"] == "high"]

# Export to Parquet
serialize_to_parquet(filtered.to_dict("records"), "high_cost_ops.parquet")
```

## Testing

Run tests:

```bash
# FactorEngine
cd /home/shw/quant_projects/factor_engine
python3 -m pytest export/test_export.py -v

# DataAccess
cd /home/shw/quant_projects/dataaccess
python3 -m pytest export/test_export.py -v
```

## Performance Notes

- **JSON**: Human-readable, good for small contracts and debugging
- **Parquet**: Best for large batches (100+ contracts), supports compression
- **Feather**: Fast read/write, good for intermediate storage
- **Arrow**: Zero-copy interop between pandas/polars, good for pipelines

Compression recommendations:
- Parquet: `snappy` (default) for speed, `zstd` for size
- Feather: `lz4` (default) for speed, `zstd` for size

## API Reference

See module docstrings for detailed API documentation:
- `serializers.py` - Serialization functions
- `importers.py` - Import and deserialization
- `converters.py` - Format conversion
- `versioning.py` - Schema versioning
