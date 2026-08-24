"""
Test no-raw-values invariant.

Factor Assets MUST NOT store raw factor values.
Only references, metadata, and bounded fingerprints are allowed.
"""

import pytest

from factor_assets.contracts.asset import FactorAsset, AssetMetadata
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.evidence_ref import EvidenceRef, EvidenceBundleRef
from factor_assets.registry import AssetRepository


def test_factor_asset_has_no_value_fields():
    """FactorAsset should not have fields for raw factor values."""
    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at="2024-01-01T00:00:00Z",
    )

    # Check that asset has no attributes that could store values
    assert not hasattr(asset, "values")
    assert not hasattr(asset, "data")
    assert not hasattr(asset, "matrix")
    assert not hasattr(asset, "series")
    assert not hasattr(asset, "dataframe")


def test_evidence_ref_has_no_raw_evidence():
    """EvidenceRef should only have references, not full metric values."""
    evidence = EvidenceRef(
        evidence_id="ev_001",
        evaluation_run_id="run_001",
        metric_name="rank_ic",
        metric_version="1.0",
        timestamp="2024-01-01T00:00:00Z",
        factor_id="F001",
        summary_value=0.05,  # Bounded summary is OK
    )

    # Has reference fields
    assert evidence.evidence_id
    assert evidence.evaluation_run_id
    assert evidence.metric_name

    # Has bounded summary only
    assert evidence.summary_value is not None
    assert isinstance(evidence.summary_value, float)

    # Should not have full series/arrays
    assert not hasattr(evidence, "daily_ic_series")
    assert not hasattr(evidence, "values")
    assert not hasattr(evidence, "full_results")


def test_evidence_bundle_ref_has_no_raw_bundle():
    """EvidenceBundleRef should only have references, not full bundle data."""
    bundle = EvidenceBundleRef(
        bundle_id="bundle_001",
        evaluation_run_id="run_001",
        factor_ids=("F001",),
        timestamp="2024-01-01T00:00:00Z",
        qe_version="0.1.0",
        primary_metric="rank_ic",
        primary_value=0.05,  # Bounded summary is OK
    )

    # Has reference fields
    assert bundle.bundle_id
    assert bundle.evaluation_run_id

    # Has bounded summary only
    assert bundle.primary_value is not None

    # Should not have full evaluation data
    assert not hasattr(bundle, "metrics")
    assert not hasattr(bundle, "full_results")
    assert not hasattr(bundle, "evaluation_data")


def test_repository_stores_no_values():
    """AssetRepository should not store raw factor values."""
    repo = AssetRepository()

    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = repo.register(metadata, lineage)

    # Repository stores assets, not values
    assert repo.exists("F001")
    retrieved = repo.get("F001")
    assert retrieved.factor_id == "F001"

    # Repository has no value storage
    assert not hasattr(repo, "_values")
    assert not hasattr(repo, "_factor_data")
    assert not hasattr(repo, "_materialized_values")


def test_asset_metadata_has_no_computation_logic():
    """AssetMetadata should not contain computation logic or data."""
    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
        lookback_days=20,
        required_fields=("close",),
    )

    # Has descriptive metadata
    assert metadata.canonical_repr
    assert metadata.lookback_days == 20
    assert metadata.required_fields == ("close",)

    # Should not have computation or values
    assert not hasattr(metadata, "compute")
    assert not hasattr(metadata, "calculate")
    assert not hasattr(metadata, "values")
    assert not hasattr(metadata, "expression")  # Only repr, not executable


def test_structural_fingerprint_is_bounded():
    """Structural fingerprint must be bounded, not full values."""
    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at="2024-01-01T00:00:00Z",
        structural_fingerprint="fp_abc",  # String reference, not array
        semantic_fingerprint="sem_xyz",
    )

    # Fingerprints are strings (references or hashes), not arrays
    assert isinstance(asset.structural_fingerprint, str)
    assert isinstance(asset.semantic_fingerprint, str)

    # Not numeric arrays
    assert not isinstance(asset.structural_fingerprint, (list, tuple))
    assert not isinstance(asset.semantic_fingerprint, (list, tuple))


def test_lineage_stores_references_only():
    """Lineage should store parent IDs, not parent values."""
    from factor_assets.contracts.lineage import ParentRef

    parent1 = ParentRef(factor_id="F001", relationship="mutation")
    parent2 = ParentRef(factor_id="F002", relationship="combination")

    lineage = LineageRef(
        factor_id="F003",
        parents=(parent1, parent2),
        campaign_id="camp_001",
    )

    # Has parent IDs
    assert lineage.parent_ids == ("F001", "F002")

    # Parents are references (IDs), not data
    for parent in lineage.parents:
        assert isinstance(parent.factor_id, str)
        assert not hasattr(parent, "values")
        assert not hasattr(parent, "data")


def test_no_materialization_paths_in_asset():
    """FactorAsset should not store materialization paths or state."""
    metadata = AssetMetadata(
        factor_id="F001",
        canonical_repr="ts_rank(close, 20)",
        canonical_hash="abc123",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    lineage = LineageRef(factor_id="F001", parents=())

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at="2024-01-01T00:00:00Z",
    )

    # No materialization state
    assert not hasattr(asset, "materialization_path")
    assert not hasattr(asset, "storage_path")
    assert not hasattr(asset, "parquet_path")
    assert not hasattr(asset, "materialized_at")
    assert not hasattr(asset, "watermark")
