# -*- coding: utf-8 -*-
"""Tests for export utilities."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling.contracts import (
    DecisionClock,
    LabelContract,
    ModelExecutionClass,
    ModelOperatorSpec,
    ParamRole,
    ParameterSearchPolicy,
    PredictionBatch,
    RegimeMetadata,
    RichModelTiming,
    SampleAdequacyContract,
    TimingKind,
)
from factor_engine.export.serializers import (
    ContractSerializer,
    serialize_batch,
    serialize_to_feather,
    serialize_to_json,
    serialize_to_parquet,
)
from factor_engine.export.importers import (
    ContractImporter,
    import_batch,
    import_from_feather,
    import_from_json,
    import_from_parquet,
)
from factor_engine.export.converters import (
    DataFrameConverter,
    convert_batch,
    to_arrow,
    to_pandas,
)
from factor_engine.export.versioning import (
    SchemaVersion,
    VersionRegistry,
    compute_schema_hash,
    get_current_version,
    migrate_contract,
    register_migration,
)


class TestContractSerializer:
    """Test contract serialization."""

    def test_serialize_model_operator_spec(self):
        """Test serializing ModelOperatorSpec."""
        spec = ModelOperatorSpec(
            canonical="test_operator",
            execution_class=ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR,
            semantic_role="model_feature",
            cost_class="medium",
        )

        data = ContractSerializer.to_dict(spec)

        assert data["canonical"] == "test_operator"
        assert data["semantic_role"] == "model_feature"
        assert "__type__" in data
        assert "execution_class" in data

    def test_serialize_rich_model_timing(self):
        """Test serializing RichModelTiming."""
        timing = RichModelTiming(
            timing_kind=TimingKind.PRIOR_FIT_PREDICTIVE.value,
            decision_time="t_close",
            feature_cutoff_rule="features through t_close",
        )

        data = ContractSerializer.to_dict(timing)

        assert data["timing_kind"] == "prior_fit_predictive"
        assert data["decision_time"] == "t_close"

    def test_serialize_sample_adequacy_contract(self):
        """Test serializing SampleAdequacyContract."""
        contract = SampleAdequacyContract(
            min_raw_obs=100,
            min_effective_obs=80,
            min_unique_dates=20,
            min_unique_stocks=50,
            min_obs_per_parameter=10.0,
            min_regime_obs=30,
        )

        data = ContractSerializer.to_dict(contract)

        assert data["min_raw_obs"] == 100
        assert data["min_effective_obs"] == 80
        assert data["min_regime_obs"] == 30

    def test_serialize_label_contract(self):
        """Test serializing LabelContract."""
        label = LabelContract(
            label_name="ret_1d",
            horizon_bars=1,
            return_basis="vwap_to_vwap",
        )

        data = ContractSerializer.to_dict(label)

        assert data["label_name"] == "ret_1d"
        assert data["horizon_bars"] == 1
        assert data["return_basis"] == "vwap_to_vwap"

    def test_serialize_decision_clock(self):
        """Test serializing DecisionClock."""
        clock = DecisionClock(
            decision_at="t_close",
            label_available_at="t+1 close",
            execution_at="t+1 VWAP",
        )

        data = ContractSerializer.to_dict(clock)

        assert data["decision_at"] == "t_close"
        assert data["execution_at"] == "t+1 VWAP"

    def test_serialize_parameter_search_policy(self):
        """Test serializing ParameterSearchPolicy."""
        policy = ParameterSearchPolicy(
            role=ParamRole.ECONOMIC_HORIZON,
            searchable=True,
            certified_values=(1, 5, 10, 20),
        )

        data = ContractSerializer.to_dict(policy)

        assert data["searchable"] is True
        assert data["certified_values"] == [1, 5, 10, 20]

    def test_serialize_regime_metadata(self):
        """Test serializing RegimeMetadata."""
        regime = RegimeMetadata(
            mode="hard",
            n_regimes=3,
            temperature=1.0,
        )

        data = ContractSerializer.to_dict(regime)

        assert data["mode"] == "hard"
        assert data["n_regimes"] == 3

    def test_serialize_prediction_batch(self):
        """Test serializing PredictionBatch."""
        batch = PredictionBatch(
            values=np.array([0.1, 0.2, 0.3]),
            row_ids=("A", "B", "C"),
            status="ok",
        )

        data = ContractSerializer.to_dict(batch)

        assert data["values"] == [0.1, 0.2, 0.3]
        assert data["row_ids"] == ["A", "B", "C"]

    def test_json_round_trip(self):
        """Test JSON serialization round trip."""
        spec = ModelOperatorSpec(
            canonical="test_op",
            execution_class=ModelExecutionClass.PREDICTIVE_SUPERVISED,
            semantic_role="model_score",
            artifact_required=True,
            training_lifecycle="artifact_walk_forward",
            default_searchable=False,
        )

        json_str = serialize_to_json(spec)
        data = json.loads(json_str)

        assert data["canonical"] == "test_op"
        assert data["artifact_required"] is True


class TestContractImporter:
    """Test contract import and reconstruction."""

    def test_import_model_operator_spec(self):
        """Test importing ModelOperatorSpec."""
        data = {
            "__type__": "modeling.contracts.ModelOperatorSpec",
            "canonical": "test_op",
            "execution_class": ModelExecutionClass.SAME_TIME_CROSS_SECTIONAL,
            "semantic_role": "model_feature",
            "cost_class": "medium",
            "timing_contract_id": "auto",
            "sample_contract_id": "auto",
            "parameter_policy_id": "auto",
            "input_contract_id": "auto",
            "training_lifecycle": "none",
            "artifact_required": False,
            "stateful": False,
            "checkpoint_supported": False,
            "default_searchable": True,
        }

        obj = ContractImporter.from_dict(data, target_type=ModelOperatorSpec)

        assert obj.canonical == "test_op"
        assert obj.semantic_role == "model_feature"

    def test_import_stale_model_operator_spec_metadata(self):
        """Reject unresolved serialized type metadata instead of returning a dict."""
        data = {
            "__type__": "factor_engine.modeling.contracts_legacy.ModelOperatorSpec",
            "canonical": "stale_metadata",
        }

        with pytest.raises(ValueError, match="Unable to resolve serialized type"):
            ContractImporter.from_dict(data)

    def test_import_explicit_target_type_with_stale_metadata(self):
        """Explicit target_type remains authoritative over stale metadata."""
        data = {
            "__type__": "modeling.contracts.ModelOperatorSpec",
            "canonical": "explicit_target",
            "execution_class": ModelExecutionClass.SAME_TIME_CROSS_SECTIONAL,
            "semantic_role": "diagnostic",
        }

        obj = ContractImporter.from_dict(data, target_type=ModelOperatorSpec)

        assert obj.canonical == "explicit_target"
        assert obj.semantic_role == "diagnostic"

    def test_import_label_contract(self):
        """Test importing LabelContract."""
        data = {
            "label_name": "ret_5d",
            "horizon_bars": 5,
            "return_basis": "vwap_to_vwap",
            "origin_time": "t",
            "start_time_rule": "entry VWAP completes at t close",
            "end_time_rule": "exit VWAP completes at t+H close",
            "availability_time_rule": "label matured at t+H close",
            "overlapping": False,
            "entry_price_basis": "VWAP_t",
            "exit_price_basis": "VWAP_{t+H}",
            "purge_by_interval": True,
            "embargo_bars": 0,
        }

        obj = ContractImporter.from_dict(data, target_type=LabelContract)

        assert obj.label_name == "ret_5d"
        assert obj.horizon_bars == 5


class TestFileFormats:
    """Test file format serialization and import."""

    def test_json_file_round_trip(self):
        """Test JSON file round trip."""
        spec = ModelOperatorSpec(
            canonical="json_test",
            execution_class=ModelExecutionClass.SAME_TIME_CROSS_SECTIONAL,
            semantic_role="diagnostic",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            serialize_to_json(spec, path)

            loaded = import_from_json(path, target_type=ModelOperatorSpec)

            assert loaded.canonical == "json_test"

    def test_parquet_round_trip(self):
        """Test Parquet serialization round trip."""
        specs = [
            ModelOperatorSpec(
                canonical=f"op_{i}",
                execution_class=ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR,
                semantic_role="model_feature",
            )
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            serialize_to_parquet(specs, path)

            loaded = import_from_parquet(path)

            assert len(loaded) == 3
            assert loaded[0]["canonical"] == "op_0"

    def test_feather_round_trip(self):
        """Test Feather serialization round trip."""
        labels = [
            LabelContract(
                label_name=f"ret_{h}d",
                horizon_bars=h,
                return_basis="vwap_to_vwap",
            )
            for h in [1, 5, 10]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            serialize_to_feather(labels, path)

            loaded = import_from_feather(path)

            assert len(loaded) == 3

    def test_serialize_batch_json(self):
        """Test batch serialization to JSON."""
        objects = [
            DecisionClock(decision_at=f"t_{i}") for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = serialize_batch(objects, tmpdir, format="json", prefix="clock")

            assert len(paths) == 3
            assert all(p.suffix == ".json" for p in paths)

    def test_serialize_batch_parquet(self):
        """Test batch serialization to Parquet."""
        objects = [
            RegimeMetadata(n_regimes=n) for n in [2, 3, 4]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = serialize_batch(objects, tmpdir, format="parquet", prefix="regime")

            assert len(paths) == 1
            assert paths[0].suffix == ".parquet"

    def test_import_batch(self):
        """Test batch import."""
        specs = [
            ModelOperatorSpec(
                canonical=f"batch_op_{i}",
                execution_class=ModelExecutionClass.RESEARCH_STRUCTURAL,
                semantic_role="diagnostic",
            )
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            serialize_batch(specs, tmpdir, format="json", prefix="spec")

            loaded = import_batch(tmpdir, pattern="*.json")

            assert len(loaded) == 5


class TestDataFrameConverter:
    """Test data format conversions."""

    def test_to_pandas_from_dict(self):
        """Test converting dict to pandas."""
        data = {"a": 1, "b": 2, "c": 3}
        df = to_pandas(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ["a", "b", "c"]

    def test_to_pandas_from_list(self):
        """Test converting list to pandas."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = to_pandas(data)

        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_to_arrow_from_pandas(self):
        """Test converting pandas to arrow."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        table = to_arrow(df)

        assert table.num_rows == 3
        assert table.num_columns == 2

    def test_to_arrow_from_dict(self):
        """Test converting dict to arrow."""
        data = {"col1": 100, "col2": 200}
        table = to_arrow(data)

        assert table.num_rows == 1

    def test_convert_batch(self):
        """Test batch conversion."""
        data_list = [
            {"a": i, "b": i * 2} for i in range(5)
        ]

        dfs = convert_batch(data_list, target_format="pandas")

        assert len(dfs) == 5
        assert all(isinstance(df, pd.DataFrame) for df in dfs)

    def test_infer_schema(self):
        """Test schema inference."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })

        schema = DataFrameConverter.infer_schema(df)

        assert len(schema) == 3
        assert schema.field("int_col") is not None


class TestVersioning:
    """Test schema versioning and migration."""

    def test_schema_version_creation(self):
        """Test creating SchemaVersion."""
        version = SchemaVersion(
            version="1.0.0",
            contract_type="TestContract",
            schema_hash="abc123",
            created_at=datetime.now(timezone.utc),
            description="Initial version",
        )

        assert version.version == "1.0.0"
        assert not version.is_breaking

    def test_schema_version_with_breaking_changes(self):
        """Test SchemaVersion with breaking changes."""
        version = SchemaVersion(
            version="2.0.0",
            contract_type="TestContract",
            schema_hash="def456",
            created_at=datetime.now(timezone.utc),
            breaking_changes=("removed_field", "changed_type"),
        )

        assert version.is_breaking
        assert len(version.breaking_changes) == 2

    def test_version_registry_register(self):
        """Test registering versions."""
        registry = VersionRegistry()

        v1 = SchemaVersion(
            version="1.0",
            contract_type="MyContract",
            schema_hash="hash1",
            created_at=datetime.now(timezone.utc),
        )

        registry.register_version(v1)
        registry.set_current_version("MyContract", "1.0")

        assert registry.get_current_version("MyContract") == "1.0"

    def test_version_registry_migration(self):
        """Test migration registration and execution."""
        registry = VersionRegistry()

        v1 = SchemaVersion(
            version="1.0",
            contract_type="TestContract",
            schema_hash="h1",
            created_at=datetime.now(timezone.utc),
        )
        v2 = SchemaVersion(
            version="2.0",
            contract_type="TestContract",
            schema_hash="h2",
            created_at=datetime.now(timezone.utc),
        )

        registry.register_version(v1)
        registry.register_version(v2)
        registry.set_current_version("TestContract", "2.0")

        def migrate_v1_to_v2(data: dict) -> dict:
            data["new_field"] = "added"
            return data

        registry.register_migration("TestContract", "1.0", "2.0", migrate_v1_to_v2)

        data = {"old_field": "value"}
        migrated = registry.migrate(data, "TestContract", "1.0", "2.0")

        assert migrated["new_field"] == "added"
        assert migrated["old_field"] == "value"

    def test_compute_schema_hash(self):
        """Test schema hash computation."""
        schema1 = {"field1": "int", "field2": "str"}
        schema2 = {"field2": "str", "field1": "int"}

        hash1 = compute_schema_hash(schema1)
        hash2 = compute_schema_hash(schema2)

        assert hash1 == hash2

    def test_migration_decorator(self):
        """Test migration decorator."""
        @register_migration("DecoratorTest", "1.0", "2.0")
        def test_migration(data: dict) -> dict:
            data["migrated"] = True
            return data

        data = {"original": True}
        result = test_migration(data)

        assert result["migrated"] is True

    def test_get_current_version(self):
        """Test global get_current_version."""
        version = get_current_version("ModelOperatorSpec")
        assert version == "1.0.0"

    def test_registry_save_load(self):
        """Test saving and loading registry."""
        registry = VersionRegistry()

        v1 = SchemaVersion(
            version="1.0",
            contract_type="SaveLoadTest",
            schema_hash="test_hash",
            created_at=datetime.now(timezone.utc),
            description="Test version",
        )

        registry.register_version(v1)
        registry.set_current_version("SaveLoadTest", "1.0")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            registry.save(path)

            loaded_registry = VersionRegistry.load(path)

            assert loaded_registry.get_current_version("SaveLoadTest") == "1.0"
            version = loaded_registry.get_version("SaveLoadTest", "1.0")
            assert version.description == "Test version"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
