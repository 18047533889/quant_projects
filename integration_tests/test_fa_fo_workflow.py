"""
Integration test: FA + FO workflow.

Test the asset lifecycle, registry, and optimizer coordination.
"""

import pytest
from datetime import datetime
from typing import Optional


def test_fa_import_isolation():
    """Test that FA can be imported independently."""
    import factor_assets
    from factor_assets import (
        FactorAsset,
        AssetMetadata,
        FactorSet,
        EvidenceRef,
        AssetRepository,
        LifecycleState,
    )

    assert factor_assets.__version__ is not None
    assert FactorAsset is not None
    assert AssetRepository is not None


def test_fo_import_isolation():
    """Test that FO can be imported independently."""
    import factor_optimizer

    info = factor_optimizer.package_info()

    assert info["name"] == "factor-optimizer"
    assert "mutation_grammar" in info["capabilities"]
    assert "search_budget_tracking" in info["capabilities"]


def test_fa_factor_identity_creation():
    """Test creating a FactorIdentity."""
    from factor_assets import create_factor_id, FactorIdentity

    factor_id = create_factor_id(
        name="momentum_20d",
        version="v1",
        params={"window": 20, "method": "log_return"},
    )

    assert factor_id is not None
    assert isinstance(factor_id, str)
    assert len(factor_id) > 0


def test_fa_asset_creation():
    """Test creating a FactorAsset."""
    from factor_assets import FactorAsset, AssetMetadata, LineageRef, LifecycleState, create_factor_id

    factor_id = create_factor_id(
        name="test_factor",
        version="v1",
        params={"window": 10},
    )

    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="test_factor",
        canonical_hash="hash_test_factor_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test factor for integration",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    assert asset.metadata.factor_id == factor_id
    assert asset.metadata.canonical_repr == "test_factor"


def test_fa_lifecycle_states():
    """Test lifecycle state transitions."""
    from factor_assets import LifecycleState, StateTransition

    # Test all defined states
    states = [
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        LifecycleState.PRODUCTION_READY,
    ]

    assert len(states) == 4
    assert LifecycleState.REGISTERED.value == "REGISTERED"
    assert LifecycleState.PRODUCTION_READY.value == "PRODUCTION_READY"

    # Test transition creation
    transition = StateTransition(
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        description="Starting evaluation",
    )

    assert transition.from_state == LifecycleState.REGISTERED
    assert transition.to_state == LifecycleState.EVALUATED


def test_fa_evidence_ref():
    """Test EvidenceRef for linking evaluation results."""
    from factor_assets import EvidenceRef

    evidence_ref = EvidenceRef(
        evidence_id="evidence_001",
        evaluation_run_id="eval_2024_001",
        metric_name="rank_ic",
        metric_version="0.1.0",
        timestamp=datetime.now().isoformat(),
        factor_id="test_factor_id_001",
        summary_value=0.045,
    )

    assert evidence_ref.factor_id == "test_factor_id_001"
    assert evidence_ref.metric_name == "rank_ic"
    assert evidence_ref.summary_value == 0.045


def test_fa_factor_set():
    """Test FactorSet creation and management."""
    from factor_assets import FactorSet, FactorSetSpec

    spec = FactorSetSpec(
        set_id="momentum_suite_001",
        name="momentum_suite",
        selection_policy="manual",
        description="Collection of momentum factors",
    )

    factor_set = FactorSet(
        set_id="momentum_suite_001",
        name="momentum_suite",
        factor_ids=("factor_001", "factor_002", "factor_003"),
        created_at=datetime.now().isoformat(),
        spec=spec,
    )

    assert factor_set.name == "momentum_suite"
    assert len(factor_set.factor_ids) == 3
    assert "factor_001" in factor_set.factor_ids


def test_fa_repository_basic_operations():
    """Test AssetRepository basic CRUD operations."""
    from factor_assets import (
        AssetRepository,
        FactorAsset,
        AssetMetadata,
        create_factor_id,
        DuplicateIdentityError,
        AssetNotFoundError,
    )

    repo = AssetRepository()

    # Create asset
    from factor_assets import LineageRef, LifecycleState

    factor_id = create_factor_id(name="repo_test", version="v1", params={})
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="repo_test",
        canonical_hash="hash_repo_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test",
    )
    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )
    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    # Add to repository
    asset = repo.register(metadata, lineage)

    # Retrieve
    retrieved = repo.get(factor_id)
    assert retrieved.metadata.factor_id == factor_id
    assert retrieved.metadata.canonical_repr == "repo_test"

    # Test duplicate detection
    with pytest.raises(DuplicateIdentityError):
        repo.register(metadata, lineage)

    # Test not found
    with pytest.raises(AssetNotFoundError):
        repo.get("non_existent_factor_id")


def test_fa_seen_index():
    """Test SeenIndex for tracking evaluated factors."""
    from factor_assets import SeenIndex

    seen_index = SeenIndex()

    seen_index.record(
        canonical_hash="hash_abc123",
        factor_id="factor_001",
        origin="evaluation",
    )

    assert seen_index.is_seen("hash_abc123")

    retrieved = seen_index.get("hash_abc123")
    assert retrieved is not None
    assert retrieved.factor_id == "factor_001"
    assert retrieved.canonical_hash == "hash_abc123"


def test_fa_lineage_tracking():
    """Test LineageRef for factor derivation tracking."""
    from factor_assets import LineageRef, ParentRef

    parent1 = ParentRef(
        factor_id="parent_factor_001",
        relationship="derived_from",
    )

    parent2 = ParentRef(
        factor_id="parent_factor_002",
        relationship="combined_with",
    )

    lineage = LineageRef(
        factor_id="child_factor_001",
        parents=(parent1, parent2),
    )

    assert len(lineage.parents) == 2


def test_fo_package_capabilities():
    """Test FO declares correct capabilities."""
    import factor_optimizer

    info = factor_optimizer.package_info()

    assert "capabilities" in info
    capabilities = info["capabilities"]

    assert "mutation_grammar" in capabilities
    assert "legality_validation" in capabilities
    assert "search_budget_tracking" in capabilities
    assert "seen_cache" in capabilities


def test_fo_adapter_protocols():
    """Test FO adapter information is available."""
    import factor_optimizer

    info = factor_optimizer.package_info()

    assert "adapters" in info
    adapters = info["adapters"]

    assert "factor_engine" in adapters
    assert "quant_evaluator" in adapters
    assert adapters["factor_engine"] == "optional"
    assert adapters["quant_evaluator"] == "optional"


def test_fa_fo_contract_alignment():
    """Test that FA contracts align with FO expectations."""
    from factor_assets import FactorAsset, LineageRef, LifecycleState, create_factor_id, AssetMetadata

    # FO would operate on factor_id strings
    factor_id = create_factor_id(name="test", version="v1", params={"p": 1})

    # FA provides full asset metadata
    metadata = AssetMetadata(
        factor_id=factor_id,
        canonical_repr="test",
        canonical_hash="hash_test_v1",
        frequency="daily",
        domains=("equity",),
        timing="daily",
        description="Test factor",
    )

    lineage = LineageRef(
        factor_id=factor_id,
        parents=(),
    )

    asset = FactorAsset(
        metadata=metadata,
        lineage=lineage,
        lifecycle_state=LifecycleState.REGISTERED,
        registered_at=datetime.now().isoformat(),
    )

    # FO needs factor_id from metadata
    assert asset.metadata.factor_id is not None


def test_fa_repository_listing():
    """Test repository list and filter operations."""
    from factor_assets import (
        AssetRepository,
        FactorAsset,
        AssetMetadata,
        create_factor_id,
    )

    repo = AssetRepository()

    # Add multiple assets
    from factor_assets import LineageRef, LifecycleState

    for i in range(3):
        factor_id = create_factor_id(name=f"factor_{i}", version="v1", params={"idx": i})
        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_repr=f"factor_{i}",
            canonical_hash=f"hash_factor_{i}_v1",
            frequency="daily",
            domains=("equity",),
            timing="daily",
            description=f"Test factor {i}",
        )
        lineage = LineageRef(
            factor_id=factor_id,
            parents=(),
        )
        asset = repo.register(metadata, lineage)

    all_assets = repo.list_all()
    assert len(all_assets) >= 3
