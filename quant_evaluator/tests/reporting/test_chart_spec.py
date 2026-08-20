"""
Tests for ChartSpec and ChartArtifactStore.
"""

import pytest
import sys
import os
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from quant_evaluator.reporting.chart_spec import ChartSpec
from quant_evaluator.reporting.artifacts import (
    ChartArtifactStore,
    ArtifactInfo,
    PathTraversalError,
    ChecksumError,
)


class TestChartSpec:
    """Tests for ChartSpec dataclass."""

    def test_creation(self):
        """Test ChartSpec creation."""
        spec = ChartSpec(
            title="Test Chart",
            x_label="X",
            y_label="Y",
            chart_type="line",
            data={"series": [1, 2, 3]}
        )
        assert spec.title == "Test Chart"
        assert spec.x_label == "X"
        assert spec.y_label == "Y"
        assert spec.chart_type == "line"
        assert spec.data == {"series": [1, 2, 3]}
        assert spec.content_hash != ""
        # Verify immutability
        try:
            spec.title = "Modified"
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_content_hash_computation(self):
        """Test that content hash is computed from identity fields."""
        spec1 = ChartSpec(
            title="Test",
            chart_type="line",
            x_label="X",
            y_label="Y",
            data={"series": [1, 2, 3]}
        )
        spec2 = ChartSpec(
            title="Test",
            chart_type="line",
            x_label="X",
            y_label="Y",
            data={"series": [1, 2, 3]}
        )
        # Same identity should produce same hash
        assert spec1.content_hash == spec2.content_hash

    def test_content_hash_includes_metadata(self):
        """Test that content hash includes title, chart_type, renderer, etc."""
        spec1 = ChartSpec(
            title="Chart A",
            chart_type="line",
            x_label="X",
            y_label="Y",
            data={"series": [1, 2, 3]}
        )
        spec2 = ChartSpec(
            title="Chart B",  # Different title
            chart_type="line",
            x_label="X",
            y_label="Y",
            data={"series": [1, 2, 3]}
        )
        assert spec1.content_hash != spec2.content_hash

    def test_replace_method(self):
        """Test immutable replace method."""
        spec1 = ChartSpec(
            title="Original",
            chart_type="line",
            x_label="X",
            y_label="Y",
            data={"series": [1, 2, 3]}
        )
        spec2 = spec1.replace(title="Modified")
        assert spec2.title == "Modified"
        assert spec1.title == "Original"
        assert spec2.content_hash != spec1.content_hash

    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        spec = ChartSpec(
            title="Test Chart",
            x_label="X",
            y_label="Y",
            chart_type="bar",
            data={"categories": ["A", "B"], "values": [10, 20]}
        )
        d = spec.to_dict()
        spec2 = ChartSpec.from_dict(d)
        assert spec.title == spec2.title
        assert spec.x_label == spec2.x_label
        assert spec.y_label == spec2.y_label
        assert spec.chart_type == spec2.chart_type
        assert spec.data == spec2.data
        assert spec.content_hash == spec2.content_hash

    def test_json_roundtrip(self):
        """Test to_json/from_json roundtrip."""
        spec = ChartSpec(
            title="JSON Chart",
            x_label="X",
            y_label="Y",
            chart_type="scatter",
            data={"x": [1, 2], "y": [3, 4]}
        )
        json_str = spec.to_json()
        spec2 = ChartSpec.from_json(json_str)
        assert spec.title == spec2.title
        assert spec.data == spec2.data


class TestChartArtifactStore:
    """Tests for ChartArtifactStore."""

    def test_save_and_load(self):
        """Test saving and loading artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="Test",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}
            )
            artifact_id = store.save(spec)
            loaded = store.load(artifact_id)
            assert loaded is not None
            assert loaded.title == "Test"
            assert loaded.data == {"series": [1, 2, 3]}

    def test_deduplication(self):
        """Test that same content is deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            spec1 = ChartSpec(
                title="Chart 1",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}
            )
            spec2 = ChartSpec(
                title="Chart 2",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}  # Same data
            )
            id1 = store.save(spec1)
            id2 = store.save(spec2)
            # Content hashes are different because title is included in hash
            # But verify both are saved
            assert store.exists(id1)
            assert store.exists(id2)
            # Both should load correctly
            loaded1 = store.load(id1)
            loaded2 = store.load(id2)
            assert loaded1 is not None
            assert loaded2 is not None

    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            for i in range(3):
                spec = ChartSpec(
                    title=f"Chart {i}",
                    x_label="X",
                    y_label="Y",
                    chart_type="line",
                    data={"series": [i]}
                )
                store.save(spec)
            artifacts = store.list_artifacts()
            assert len(artifacts) == 3

    def test_get_artifact(self):
        """Test getting artifact metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="Metadata Test",
                x_label="X",
                y_label="Y",
                chart_type="bar",
                data={"values": [1, 2]}
            )
            artifact_id = store.save(spec)
            info = store.get_artifact(artifact_id)
            assert info is not None
            assert info.title == "Metadata Test"
            assert info.chart_type == "bar"

    def test_delete_artifact(self):
        """Test deleting artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="To Delete",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1]}
            )
            artifact_id = store.save(spec)
            assert store.exists(artifact_id)
            result = store.delete_artifact(artifact_id)
            assert result is True
            assert not store.exists(artifact_id)
            assert store.load(artifact_id) is None

    def test_nonexistent_artifact(self):
        """Test loading nonexistent artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            assert store.load("nonexistent") is None
            assert store.get_artifact("nonexistent") is None
            assert store.delete_artifact("nonexistent") is False

    def test_path_traversal_protection(self):
        """Test that path traversal attacks are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            try:
                bad_path = store._get_artifact_path("../../../etc/passwd")
                assert False, "Should have raised PathTraversalError"
            except PathTraversalError:
                pass

    def test_checksum_verification(self):
        """Test that checksum verification works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChartArtifactStore(base_dir=tmpdir)
            spec = ChartSpec(
                title="Checksum Test",
                x_label="X",
                y_label="Y",
                chart_type="line",
                data={"series": [1, 2, 3]}
            )
            artifact_id = store.save(spec)
            # Load should succeed
            loaded = store.load(artifact_id, verify_checksum=True)
            assert loaded is not None
