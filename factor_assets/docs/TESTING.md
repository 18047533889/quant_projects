# FactorAssets Testing Guide

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Overview

FactorAssets uses pytest for testing. The suite covers contracts, repository operations, lifecycle state machine, lineage tracking, clustering, and architectural constraints.

## Quick Start

### Run All Tests

```bash
cd /home/shw/quant_projects/factor_assets
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=factor_assets --cov-report=term-missing
```

### Run Specific Categories

```bash
# Repository tests
pytest tests/test_repository.py

# Lifecycle tests
pytest tests/test_lifecycle_orchestration.py tests/lifecycle/

# Lineage tests
pytest tests/test_lineage_graph.py

# Clustering tests
pytest tests/test_clustering_families.py tests/test_clustering_lineage.py

# Architectural constraint tests
pytest tests/no_raw_values/
```

## Test Organization

```
tests/
├── __init__.py
├── conftest.py                       # Shared fixtures
├── test_repository.py                # AssetRepository tests
├── test_lifecycle_orchestration.py   # Lifecycle orchestrator
├── test_lineage_graph.py             # Lineage DAG tests
├── test_snapshots.py                 # Snapshot manager
├── test_seen_index.py                # Deduplication tests
├── test_clustering_families.py       # Family detection
├── test_clustering_lineage.py        # Parent-child detection
├── test_graph_sparse.py              # Sparse graph tests
├── test_graph_edges.py               # Edge filtering
├── lifecycle/
│   ├── __init__.py
│   └── test_state_machine.py        # Pure state machine
├── selection/
│   ├── __init__.py
│   ├── test_gates.py                # Gate evaluation
│   └── test_policy.py               # Selection policy
├── aggregation/
│   ├── __init__.py
│   ├── test_specs.py                # Aggregation specs
│   └── test_representatives.py      # Representative selection
└── no_raw_values/
    ├── __init__.py
    └── test_no_raw_values.py        # Architectural constraints
```

## Shared Fixtures

Located in `tests/conftest.py`:

### Basic Fixtures

```python
@pytest.fixture
def sample_metadata():
    """Sample AssetMetadata for testing."""
    return AssetMetadata(
        factor_id="F_test001",
        canonical_hash="abc123",
        frequency="daily",
        domains=("equity",),
        data_sources=("market",),
    )

@pytest.fixture
def empty_repo():
    """Empty AssetRepository."""
    return AssetRepository()

@pytest.fixture
def repo_with_factors():
    """Repository with 3 registered factors."""
    repo = AssetRepository()
    # Register test factors
    return repo

@pytest.fixture
def sample_evidence_ref():
    """Sample EvidenceRef."""
    return EvidenceRef(
        evaluation_id="eval_001",
        factor_id="F_test001",
        evidence_type="qe_bundle",
        metric_snapshot={"mean_ic": 0.05, "coverage": 0.95},
        timestamp="2026-08-14T10:00:00Z",
    )
```

## Writing New Tests

### Test Template

```python
import pytest
from factor_assets import (
    AssetRepository,
    AssetMetadata,
    LifecycleState,
    DuplicateIdentityError,
)


class TestYourFeature:
    """Test suite for your feature."""
    
    def test_happy_path(self, empty_repo, sample_metadata):
        """Test basic functionality."""
        asset = empty_repo.register(
            metadata=sample_metadata,
            lineage=None,
            tags={},
        )
        
        assert asset is not None
        assert asset.metadata.factor_id == "F_test001"
        assert asset.lifecycle_state == LifecycleState.REGISTERED
    
    def test_edge_case_duplicate(self, empty_repo, sample_metadata):
        """Test duplicate registration raises error."""
        empty_repo.register(sample_metadata, None, {})
        
        with pytest.raises(DuplicateIdentityError):
            empty_repo.register(sample_metadata, None, {})
    
    def test_contract_validation(self):
        """Test contract validation."""
        with pytest.raises(ValueError):
            AssetMetadata(
                factor_id="",  # Invalid empty ID
                canonical_hash="abc",
                frequency="daily",
                domains=(),
                data_sources=(),
            )
```

## Repository Tests

### Registration Tests

```python
def test_register_new_factor(empty_repo):
    """Test registering a new factor."""
    metadata = AssetMetadata(
        factor_id="F_001",
        canonical_hash="hash001",
        frequency="daily",
        domains=("equity",),
        data_sources=("market",),
    )
    
    asset = empty_repo.register(metadata, lineage=None, tags={})
    
    assert asset.factor_id == "F_001"
    assert asset.lifecycle_state == LifecycleState.REGISTERED
    assert len(asset.evidence_refs) == 0


def test_register_duplicate_raises_error(empty_repo, sample_metadata):
    """Duplicate canonical_hash raises DuplicateIdentityError."""
    empty_repo.register(sample_metadata, None, {})
    
    # Same hash, different factor_id - still duplicate
    duplicate = AssetMetadata(
        factor_id="F_different",
        canonical_hash=sample_metadata.canonical_hash,  # Same hash!
        frequency="daily",
        domains=("equity",),
        data_sources=("market",),
    )
    
    with pytest.raises(DuplicateIdentityError):
        empty_repo.register(duplicate, None, {})
```

### Query Tests

```python
def test_get_existing_factor(repo_with_factors):
    """Test retrieving existing factor."""
    asset = repo_with_factors.get("F_test001")
    
    assert asset is not None
    assert asset.factor_id == "F_test001"


def test_get_nonexistent_raises_error(empty_repo):
    """Test retrieving non-existent factor raises error."""
    from factor_assets import AssetNotFoundError
    
    with pytest.raises(AssetNotFoundError):
        empty_repo.get("F_nonexistent")


def test_exists(repo_with_factors):
    """Test existence check."""
    assert repo_with_factors.exists("F_test001")
    assert not repo_with_factors.exists("F_nonexistent")


def test_find_by_hash(repo_with_factors):
    """Test finding factor by canonical hash."""
    asset = repo_with_factors.find_by_hash("abc123")
    
    assert asset is not None
    assert asset.metadata.canonical_hash == "abc123"


def test_find_by_hash_nonexistent(empty_repo):
    """Test finding non-existent hash returns None."""
    assert empty_repo.find_by_hash("nonexistent") is None
```

### Transition Tests

```python
def test_transition_to_evaluated(empty_repo, sample_metadata, sample_evidence_ref):
    """Test transition REGISTERED -> EVALUATED."""
    asset = empty_repo.register(sample_metadata, None, {})
    assert asset.lifecycle_state == LifecycleState.REGISTERED
    
    # Transition to EVALUATED with evidence
    updated = empty_repo.transition(
        factor_id=asset.factor_id,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=[sample_evidence_ref],
        operator="system",
    )
    
    assert updated.lifecycle_state == LifecycleState.EVALUATED
    assert len(updated.evidence_refs) == 1


def test_illegal_transition_raises_error(empty_repo, sample_metadata):
    """Test illegal transition raises LifecycleConflictError."""
    from factor_assets import LifecycleConflictError
    
    asset = empty_repo.register(sample_metadata, None, {})
    
    # Cannot skip EVALUATED
    with pytest.raises(LifecycleConflictError):
        empty_repo.transition(
            factor_id=asset.factor_id,
            to_state=LifecycleState.APPROVED,  # Skip EVALUATED
            operator="system",
        )
```

### List and Stats Tests

```python
def test_list_by_state(repo_with_factors):
    """Test listing factors by lifecycle state."""
    registered = repo_with_factors.list_by_state(LifecycleState.REGISTERED)
    
    assert len(registered) > 0
    assert all(a.lifecycle_state == LifecycleState.REGISTERED for a in registered)


def test_list_all(repo_with_factors):
    """Test listing all factors."""
    all_assets = repo_with_factors.list_all()
    
    assert len(all_assets) >= 3


def test_stats(repo_with_factors):
    """Test repository statistics."""
    stats = repo_with_factors.stats()
    
    assert stats.total_assets >= 3
    assert LifecycleState.REGISTERED in stats.by_state
```

## Lifecycle Tests

### State Machine Tests

```python
from factor_assets.lifecycle import StateMachine

def test_legal_transition():
    """Test legal transition validation."""
    assert StateMachine.can_transition(
        LifecycleState.REGISTERED,
        LifecycleState.EVALUATED
    )


def test_illegal_transition():
    """Test illegal transition detection."""
    assert not StateMachine.can_transition(
        LifecycleState.REGISTERED,
        LifecycleState.APPROVED  # Skip EVALUATED
    )


def test_self_transition_evaluated_only():
    """Test self-transition only allowed for EVALUATED."""
    assert StateMachine.can_transition(
        LifecycleState.EVALUATED,
        LifecycleState.EVALUATED  # Re-evaluation
    )
    
    assert not StateMachine.can_transition(
        LifecycleState.REGISTERED,
        LifecycleState.REGISTERED  # Not allowed
    )
```

### Orchestrator Tests

```python
from factor_assets.registry import LifecycleOrchestrator, TransitionRequest

def test_orchestrator_validate():
    """Test orchestrator validation."""
    orchestrator = LifecycleOrchestrator()
    
    request = TransitionRequest(
        factor_id="F_001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=[],
        operator="system",
    )
    
    # Should not raise
    orchestrator.validate_transition(request)


def test_orchestrator_hooks():
    """Test transition hooks."""
    orchestrator = LifecycleOrchestrator()
    
    hook_called = []
    
    def my_hook(request):
        hook_called.append(request.factor_id)
    
    orchestrator.add_transition_hook(
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        hook=my_hook,
    )
    
    request = TransitionRequest(
        factor_id="F_001",
        from_state=LifecycleState.REGISTERED,
        to_state=LifecycleState.EVALUATED,
        evidence_refs=[],
        operator="system",
    )
    
    orchestrator.execute_transition(request)
    
    assert "F_001" in hook_called
```

## Lineage Tests

### DAG Construction

```python
from factor_assets.registry import LineageGraph
from factor_assets import LineageRef, ParentRef

def test_lineage_graph_root():
    """Test root factor (no parents)."""
    graph = LineageGraph()
    
    lineage = LineageRef(
        factor_id="F_root",
        parent_refs=(),
        generation=0,
    )
    
    graph.register_factor(lineage)
    
    assert graph.is_root("F_root")
    assert not graph.is_leaf("F_root")  # No children yet


def test_lineage_graph_parent_child():
    """Test parent-child relationship."""
    graph = LineageGraph()
    
    # Register parent
    parent_lineage = LineageRef(
        factor_id="F_parent",
        parent_refs=(),
        generation=0,
    )
    graph.register_factor(parent_lineage)
    
    # Register child
    child_lineage = LineageRef(
        factor_id="F_child",
        parent_refs=(ParentRef(parent_id="F_parent", contribution_weight=1.0),),
        generation=1,
    )
    graph.register_factor(child_lineage)
    
    # Verify relationships
    parents = graph.get_parents("F_child")
    assert "F_parent" in parents
    
    children = graph.get_children("F_parent")
    assert "F_child" in children


def test_lineage_depth():
    """Test lineage depth calculation."""
    graph = LineageGraph()
    
    # Root
    graph.register_factor(LineageRef("F_0", (), 0))
    
    # Generation 1
    graph.register_factor(LineageRef(
        "F_1",
        (ParentRef("F_0", 1.0),),
        1,
    ))
    
    # Generation 2
    graph.register_factor(LineageRef(
        "F_2",
        (ParentRef("F_1", 1.0),),
        2,
    ))
    
    assert graph.get_lineage_depth("F_0") == 0
    assert graph.get_lineage_depth("F_1") == 1
    assert graph.get_lineage_depth("F_2") == 2
```

## Clustering Tests

### Family Detection

```python
from factor_assets.clustering import ConnectedComponents
from factor_assets.graph import SparseCorrelationGraph

def test_connected_components():
    """Test connected component clustering."""
    # Create correlation graph
    edges = [
        ("F_1", "F_2", 0.8),
        ("F_2", "F_3", 0.75),
        ("F_4", "F_5", 0.9),  # Separate component
    ]
    
    graph = SparseCorrelationGraph.from_edges(edges)
    
    # Find components
    clusterer = ConnectedComponents(graph)
    result = clusterer.find_components()
    
    assert result.num_clusters == 2
    
    # F_1, F_2, F_3 in same cluster
    cluster_1 = result.factor_to_cluster["F_1"]
    assert result.factor_to_cluster["F_2"] == cluster_1
    assert result.factor_to_cluster["F_3"] == cluster_1
    
    # F_4, F_5 in different cluster
    cluster_2 = result.factor_to_cluster["F_4"]
    assert cluster_2 != cluster_1
    assert result.factor_to_cluster["F_5"] == cluster_2
```

## Selection Tests

### Gate Evaluation

```python
from factor_assets.selection import ThresholdGate, GateResult

def test_threshold_gate_pass():
    """Test gate passes when metric exceeds threshold."""
    gate = ThresholdGate(
        gate_name="min_ic_gate",
        gate_version="1.0",
        threshold=0.03,
    )
    
    evaluation = gate.evaluate(
        factor_id="F_001",
        evidence_id="eval_001",
        metric_name="mean_ic",
        metric_value=0.05,  # Above threshold
    )
    
    assert evaluation.result == GateResult.PASS
    assert evaluation.passed


def test_threshold_gate_fail():
    """Test gate fails when metric below threshold."""
    gate = ThresholdGate(
        gate_name="min_ic_gate",
        gate_version="1.0",
        threshold=0.03,
    )
    
    evaluation = gate.evaluate(
        factor_id="F_001",
        evidence_id="eval_001",
        metric_name="mean_ic",
        metric_value=0.01,  # Below threshold
    )
    
    assert evaluation.result == GateResult.FAIL
    assert not evaluation.passed
```

## Architectural Constraint Tests

### No Raw Values

```python
def test_no_numpy_arrays_in_contracts():
    """Verify no NumPy arrays in core contracts."""
    import numpy as np
    from factor_assets import FactorAsset, AssetMetadata
    
    metadata = AssetMetadata(
        factor_id="F_001",
        canonical_hash="hash",
        frequency="daily",
        domains=("equity",),
        data_sources=("market",),
    )
    
    asset = FactorAsset(
        metadata=metadata,
        lifecycle_state=LifecycleState.REGISTERED,
        lineage=None,
        evidence_refs=(),
        tags={},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    # Check no NumPy arrays in any field
    for field_name, field_value in asset.__dict__.items():
        assert not isinstance(field_value, np.ndarray), \
            f"Field {field_name} contains NumPy array"


def test_evidence_ref_bounded_snapshot():
    """Verify evidence refs contain only bounded snapshots."""
    from factor_assets import EvidenceRef
    
    ref = EvidenceRef(
        evaluation_id="eval_001",
        factor_id="F_001",
        evidence_type="qe_bundle",
        metric_snapshot={"mean_ic": 0.05, "coverage": 0.95},
        timestamp="2026-08-14T10:00:00Z",
    )
    
    # Snapshot should be dict with scalar values
    assert isinstance(ref.metric_snapshot, dict)
    assert all(isinstance(v, (int, float, str, bool, type(None))) 
               for v in ref.metric_snapshot.values())
```

## Performance Tests

```python
@pytest.mark.slow
def test_large_repository_performance():
    """Test repository with 10K factors."""
    import time
    
    repo = AssetRepository()
    
    # Register 10K factors
    start = time.time()
    for i in range(10000):
        metadata = AssetMetadata(
            factor_id=f"F_{i:06d}",
            canonical_hash=f"hash_{i}",
            frequency="daily",
            domains=("equity",),
            data_sources=("market",),
        )
        repo.register(metadata, None, {})
    
    register_time = time.time() - start
    
    # Query performance
    start = time.time()
    for i in range(1000):
        repo.get(f"F_{i:06d}")
    query_time = time.time() - start
    
    print(f"Register 10K: {register_time:.2f}s")
    print(f"Query 1K: {query_time:.4f}s ({query_time/1000*1000:.2f}ms avg)")
    
    # Should be fast
    assert query_time / 1000 < 0.001  # <1ms per query
```

## Coverage Goals

- **Contracts:** 100% line coverage
- **Repository:** 100% branch coverage
- **Lifecycle:** 100% state transition coverage
- **Lineage:** All DAG operations covered
- **Overall:** >95% coverage

## Running Coverage

```bash
# Generate HTML report
pytest tests/ --cov=factor_assets --cov-report=html

# View report
open htmlcov/index.html

# Check minimum coverage
pytest tests/ --cov=factor_assets --cov-fail-under=95
```

## Best Practices

1. **Immutability:** Test that frozen dataclasses cannot be modified
2. **State machine:** Test all legal and illegal transitions
3. **Append-only:** Verify history is never deleted
4. **Hash uniqueness:** Test duplicate detection
5. **Protocol boundaries:** Use mocks for FE/QE adapters
6. **Edge cases:** Empty repo, single factor, cycles in lineage

---

**Last Updated:** 2026-08-14  
**Contributors:** Quant Platform Team
