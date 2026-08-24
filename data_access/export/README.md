# Export Utilities

Comprehensive serialization, import, and conversion utilities for DataAccess contracts.

## Installation

```python
from dataaccess.export import (
    serialize_to_json,
    serialize_to_parquet,
    serialize_to_feather,
    import_from_json,
    import_from_parquet,
    to_pandas,
    to_polars,
    to_arrow,
)
```

## Features

### 1. Serialization

Serialize DataAccess contracts to JSON, Parquet, and Feather formats:

```python
from dataaccess.export import serialize_to_json, serialize_to_parquet

# Snapshot to JSON
snapshot = {
    "snapshot_id": "snap_001",
    "dataset": "ashare_daily",
    "registry_hash": "abc123",
    "schema_hash": "def456",
    "files": [
        {"path": "/data/file1.parquet", "size": 1024},
        {"path": "/data/file2.parquet", "size": 2048},
    ],
}
serialize_to_json(snapshot, "snapshot.json")

# Batch to Parquet
snapshots = [snapshot1, snapshot2, snapshot3]
serialize_to_parquet(snapshots, "snapshots.parquet")
```

### 2. Import

Deserialize contracts with automatic type reconstruction:

```python
from dataaccess.export import import_from_json, import_from_parquet

# Import single contract
snapshot = import_from_json("snapshot.json")

# Import batch
snapshots = import_from_parquet("snapshots.parquet")

# Import directory
from dataaccess.export import import_batch
contracts = import_batch("output_dir/", pattern="*.json")
```

### 3. Data Conversion

Convert between pandas, polars, and arrow formats:

```python
from dataaccess.export import to_pandas, to_arrow

# Dict to pandas
df = to_pandas({"dataset": "ashare_daily", "rows": 1000})

# Pandas to arrow
table = to_arrow(df)
```

### 4. Schema Versioning

Manage schema evolution:

```python
from dataaccess.export import SchemaVersion, register_migration

# Define version
version = SchemaVersion(
    version="2.0.0",
    contract_type="DataSnapshot",
    schema_hash="xyz789",
    created_at=datetime.now(timezone.utc),
    description="Added file checksums",
)

# Register migration
@register_migration("DataSnapshot", "1.0.0", "2.0.0")
def migrate_snapshot_v1_to_v2(data: dict) -> dict:
    for file in data.get("files", []):
        file["checksum"] = None  # Add checksum field
    return data
```

## Supported Contracts

**Read Contracts:**
- `DataSnapshot` - Data snapshot with file manifest
- `FileVersion` - File version metadata
- `ReadLineage` - Data lineage tracking
- `ReadStats` - Read statistics

**Schema:**
- `SchemaEpoch` - Schema versions
- `ContractIR` - Contract intermediate representation
- `RuntimeContract` - Runtime contracts

## Usage Examples

### Export Data Snapshots

```python
from dataaccess.export import serialize_batch

snapshots = [
    {
        "snapshot_id": f"snap_{i}",
        "dataset": "ashare_daily",
        "registry_hash": "hash1",
        "schema_hash": "schema_v1",
    }
    for i in range(100)
]

# Export to Parquet
paths = serialize_batch(snapshots, "snapshots/", format="parquet")
```

### Track Lineage

```python
from dataaccess.export import serialize_to_json

lineage = {
    "dataset": "ashare_daily",
    "columns": ["open", "close", "volume"],
    "time_range": ("2020-01-01", "2023-12-31"),
    "instrument_filter": ["000001.SZ", "000002.SZ"],
    "build_sha": "abc123",
}

serialize_to_json(lineage, "lineage.json")
```

### Convert Read Results

```python
from dataaccess.export import to_pandas, to_arrow

# Read result as dict
result = {
    "rows": 1000,
    "bytes": 50000,
    "elapsed_ms": 125.5,
    "paths": ["/data/part1.parquet", "/data/part2.parquet"],
}

# Convert to DataFrame for analysis
df = to_pandas(result)

# Export as Arrow for interop
table = to_arrow(df)
```

## Testing

Run tests:

```bash
cd /home/shw/quant_projects/dataaccess
python3 -m pytest export/test_export.py -v
```

## API Reference

See module docstrings for detailed API documentation:
- `serializers.py` - Serialization functions
- `importers.py` - Import and deserialization
- `converters.py` - Format conversion
- `versioning.py` - Schema versioning
