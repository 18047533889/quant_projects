# -*- coding: utf-8 -*-
"""Tests for DataAccess export utilities."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from export.serializers import (
    ContractSerializer,
    serialize_batch,
    serialize_to_feather,
    serialize_to_json,
    serialize_to_parquet,
)
from export.importers import (
    ContractImporter,
    import_batch,
    import_from_feather,
    import_from_json,
    import_from_parquet,
)
from export.converters import (
    DataFrameConverter,
    convert_batch,
    to_arrow,
    to_pandas,
)
from export.versioning import (
    SchemaVersion,
    VersionRegistry,
    compute_schema_hash,
    get_current_version,
    migrate_contract,
    register_migration,
)


class TestDataAccessSerializer:
    """Test DataAccess contract serialization."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        data = {
            "dataset": "ashare_daily",
            "columns": ["open", "close", "volume"],
            "time_range": ("2020-01-01", "2023-12-31"),
        }

        serialized = ContractSerializer.to_dict(data)

        assert serialized["dataset"] == "ashare_daily"
        assert serialized["columns"] == ["open", "close", "volume"]

    def test_serialize_with_datetime(self):
        """Test serializing with datetime values."""
        data = {
            "snapshot_id": "snap_001",
            "created_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "dataset": "test",
        }

        serialized = ContractSerializer.to_dict(data)

        assert "2023-01-01T12:00:00" in serialized["created_at"]

    def test_serialize_nested_structure(self):
        """Test serializing nested structures."""
        data = {
            "snapshot": {
                "snapshot_id": "s1",
                "files": [
                    {"path": "/data/file1.parquet", "size": 1024},
                    {"path": "/data/file2.parquet", "size": 2048},
                ],
            },
            "metadata": {"version": "1.0"},
        }

        serialized = ContractSerializer.to_dict(data)

        assert serialized["snapshot"]["snapshot_id"] == "s1"
        assert len(serialized["snapshot"]["files"]) == 2

    def test_json_serialization(self):
        """Test JSON string serialization."""
        data = {
            "registry_hash": "abc123",
            "schema_hash": "def456",
            "params": {"market": "CN", "freq": "daily"},
        }

        json_str = serialize_to_json(data)
        parsed = json.loads(json_str)

        assert parsed["registry_hash"] == "abc123"
        assert parsed["params"]["market"] == "CN"


class TestDataAccessImporter:
    """Test DataAccess contract import."""

    def test_import_from_dict(self):
        """Test importing from dictionary."""
        data = {
            "snapshot_id": "test_snap",
            "dataset": "test_dataset",
            "registry_hash": "hash1",
            "schema_hash": "hash2",
        }

        imported = ContractImporter.from_dict(data)

        assert imported["snapshot_id"] == "test_snap"
        assert imported["dataset"] == "test_dataset"

    def test_import_nested_dict(self):
        """Test importing nested dictionary."""
        data = {
            "lineage": {
                "dataset": "ashare_daily",
                "columns": ["open", "close"],
                "params": [["market", "CN"], ["freq", "1d"]],
            }
        }

        imported = ContractImporter.from_dict(data)

        assert imported["lineage"]["dataset"] == "ashare_daily"


class TestFileFormatsDataAccess:
    """Test file format operations for DataAccess."""

    def test_json_file_round_trip(self):
        """Test JSON file round trip."""
        data = {
            "dataset": "test_data",
            "schema_hash": "hash123",
            "files": ["file1.parquet", "file2.parquet"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            serialize_to_json(data, path)

            loaded = import_from_json(path)

            assert loaded["dataset"] == "test_data"
            assert len(loaded["files"]) == 2

    def test_parquet_batch(self):
        """Test Parquet batch serialization."""
        snapshots = [
            {
                "snapshot_id": f"snap_{i}",
                "dataset": "test",
                "registry_hash": f"hash_{i}",
                "schema_hash": "schema_v1",
            }
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshots.parquet"
            serialize_to_parquet(snapshots, path)

            loaded = import_from_parquet(path)

            assert len(loaded) == 5

    def test_feather_batch(self):
        """Test Feather batch serialization."""
        lineages = [
            {
                "dataset": f"dataset_{i}",
                "columns": ["col1", "col2"],
                "time_range": ("2020-01-01", "2023-12-31"),
            }
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lineages.feather"
            serialize_to_feather(lineages, path)

            loaded = import_from_feather(path)

            assert len(loaded) == 3

    def test_batch_json_directory(self):
        """Test batch JSON directory export."""
        contracts = [
            {"contract_id": f"c{i}", "type": "read"} for i in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = serialize_batch(contracts, tmpdir, format="json", prefix="contract")

            assert len(paths) == 4
            loaded = import_batch(tmpdir, pattern="*.json")
            assert len(loaded) == 4


class TestDataFrameConversionDataAccess:
    """Test DataFrame conversion for DataAccess."""

    def test_dict_to_pandas(self):
        """Test converting dict to pandas."""
        data = {"a": 1, "b": 2, "c": 3}
        df = to_pandas(data)

        assert isinstance(df, pd.DataFrame)
        assert df.loc[0, "a"] == 1

    def test_list_to_pandas(self):
        """Test converting list to pandas."""
        data = [
            {"stock_id": "000001", "close": 10.5},
            {"stock_id": "000002", "close": 20.3},
        ]
        df = to_pandas(data)

        assert len(df) == 2
        assert "stock_id" in df.columns

    def test_arrow_conversion(self):
        """Test arrow conversion."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "value": [100, 200],
        })

        table = to_arrow(df)

        assert table.num_rows == 2
        assert "date" in table.column_names

    def test_batch_conversion(self):
        """Test batch conversion."""
        data_list = [
            {"idx": i, "value": i * 10} for i in range(3)
        ]

        dfs = convert_batch(data_list, target_format="pandas")

        assert len(dfs) == 3
        assert all(isinstance(df, pd.DataFrame) for df in dfs)


class TestVersioningDataAccess:
    """Test versioning for DataAccess contracts."""

    def test_schema_version_basic(self):
        """Test basic schema version."""
        version = SchemaVersion(
            version="1.0.0",
            contract_type="DataSnapshot",
            schema_hash="hash_v1",
            created_at=datetime.now(timezone.utc),
            description="Initial snapshot schema",
        )

        assert version.version == "1.0.0"
        assert version.contract_type == "DataSnapshot"

    def test_version_registry_basic(self):
        """Test basic version registry operations."""
        registry = VersionRegistry()

        v1 = SchemaVersion(
            version="1.0",
            contract_type="ReadContract",
            schema_hash="h1",
            created_at=datetime.now(timezone.utc),
        )

        registry.register_version(v1)
        registry.set_current_version("ReadContract", "1.0")

        current = registry.get_current_version("ReadContract")
        assert current == "1.0"

    def test_migration_path(self):
        """Test migration path finding."""
        registry = VersionRegistry()

        for ver in ["1.0", "1.1", "2.0"]:
            v = SchemaVersion(
                version=ver,
                contract_type="TestContract",
                schema_hash=f"h_{ver}",
                created_at=datetime.now(timezone.utc),
            )
            registry.register_version(v)

        def migrate_1_0_to_1_1(data: dict) -> dict:
            data["v1_1_field"] = "added"
            return data

        def migrate_1_1_to_2_0(data: dict) -> dict:
            data["v2_field"] = "new"
            return data

        registry.register_migration("TestContract", "1.0", "1.1", migrate_1_0_to_1_1)
        registry.register_migration("TestContract", "1.1", "2.0", migrate_1_1_to_2_0)
        registry.set_current_version("TestContract", "2.0")

        data = {"original": True}
        migrated = registry.migrate(data, "TestContract", "1.0", "2.0")

        assert migrated["v1_1_field"] == "added"
        assert migrated["v2_field"] == "new"
        assert migrated["original"] is True

    def test_registry_persistence(self):
        """Test saving and loading registry."""
        registry = VersionRegistry()

        v1 = SchemaVersion(
            version="1.0",
            contract_type="PersistTest",
            schema_hash="persist_hash",
            created_at=datetime.now(timezone.utc),
            description="Persistence test",
        )

        registry.register_version(v1)
        registry.set_current_version("PersistTest", "1.0")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            registry.save(path)

            loaded = VersionRegistry.load(path)

            assert loaded.get_current_version("PersistTest") == "1.0"
            loaded_version = loaded.get_version("PersistTest", "1.0")
            assert loaded_version.description == "Persistence test"

    def test_compute_schema_hash_consistency(self):
        """Test schema hash computation consistency."""
        schema1 = {"field_a": "int64", "field_b": "string"}
        schema2 = {"field_b": "string", "field_a": "int64"}

        hash1 = compute_schema_hash(schema1)
        hash2 = compute_schema_hash(schema2)

        assert hash1 == hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
