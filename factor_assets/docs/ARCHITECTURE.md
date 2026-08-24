# FactorAssets Architecture

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Overview

FactorAssets is an append-only factor registry with immutable identity, lifecycle state machine, and lineage tracking. It stores metadata and evidence references only—no raw factor values, no file I/O, no parser logic.

## Design Principles

### 1. Append-Only Repository

All operations add records, never delete:
- Factor registration creates immutable identity
- State transitions append events
- Evidence references accumulate
- History is never rewritten

**Rationale:** Reproducibility and audit trail. Every decision is traceable.

### 2. Immutable Identity

Once registered, factor identity never changes:
- `factor_id`: Unique identifier (derived from canonical hash)
- `canonical_hash`: Content-based identity from FactorEngine
- No renames, no deletions, only deprecation

**Rationale:** Stable references across systems. Evidence remains valid.

### 3. Protocol-Based Integration

Dependencies via protocols, not concrete classes:
- `FactorIdentityProvider`: Obtain canonical hash from FE
- `EvidenceProvider`: Retrieve QE evaluation results
- No hard coupling to FE/QE implementations

**Rationale:** Testability without full stack. Clear boundaries.

### 4. No Raw Values

Only metadata and references:
- Evidence refs point to QE evaluations
- No factor values stored
- No time series data
- Bounded metric snapshots only

**Rationale:** FA is governance layer, not data layer. DA owns storage.

### 5. Conservative State Machine

Explicit lifecycle with validation:
- REGISTERED → EVALUATED → APPROVED → PRODUCTION_READY
- Illegal transitions raise errors
- Evidence required for most transitions
- Self-transitions only for re-evaluation

**Rationale:** Fail-closed. Prevent accidental production deployment.

## Module Organization

```
factor_assets/
├── __init__.py              # Public API exports
├── contracts/               # Core data contracts
│   ├── asset.py            # FactorAsset, AssetMetadata
│   ├── factor_set.py       # FactorSet, FactorSetSpec
│   ├── evidence_ref.py     # EvidenceRef, EvidenceBundleRef
│   ├── lifecycle.py        # LifecycleState, StateEvent
│   ├── lineage.py          # LineageRef, ParentRef
│   └── envelope.py         # ContractEnvelope (versioning)
├── registry/                # Repository and metadata
│   ├── repository.py       # AssetRepository (main API)
│   ├── lifecycle.py        # LifecycleOrchestrator
│   ├── lineage.py          # LineageGraph (DAG)
│   └── snapshots.py        # SnapshotManager
├── identity/                # Canonical identity
│   └── canonical.py        # FactorIdentityProvider, create_factor_id
├── seen_index/              # Deduplication
│   └── exact.py            # SeenIndex, SeenRecord
├── lifecycle/               # State machine
│   └── state_machine.py    # Pure validation logic
├── selection/               # Admission gates
│   ├── gates.py            # EvidenceGate, ThresholdGate
│   └── policy.py           # SelectionPolicy
├── clustering/              # Factor families
│   ├── families.py         # Connected components, modularity
│   └── lineage.py          # Parent-child detection
├── aggregation/             # Family representatives
│   ├── specs.py            # AggregationSpec
│   └── representatives.py  # RepresentativeSelector
├── graph/                   # Correlation graphs
│   ├── sparse.py           # SparseCorrelationGraph
│   └── edges.py            # EdgeFilter (threshold, top-K)
├── similarity/              # Similarity measurement
│   └── exact.py            # SimilarityMeasure (protocol)
└── novelty/                 # Evidence protocol
    └── provider.py         # EvidenceProvider (QE boundary)
```

## Core Contracts

### FactorAsset

Complete factor record with metadata, lifecycle, lineage, and evidence.

```python
@dataclass(frozen=True)
class FactorAsset:
    metadata: AssetMetadata
    lifecycle_state: LifecycleState
    lineage: Optional[LineageRef]
    evidence_refs: Tuple[EvidenceRef, ...]
    tags: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**Key invariants:**
- `metadata.factor_id` never changes
- `lifecycle_state` only transitions via state machine
- `evidence_refs` is append-only (grows over time)
- All fields frozen (immutable)

### AssetMetadata

Core identity and domain information.

```python
@dataclass(frozen=True)
class AssetMetadata:
    factor_id: str              # Unique ID (e.g., "F_abc123")
    canonical_hash: str         # FE canonical identity
    frequency: str              # "daily", "minute", etc.
    domains: Tuple[str, ...]    # ("equity", "futures")
    data_sources: Tuple[str, ...]  # ("market", "fundamental")
    timing_kind: Optional[str] = None
    creation_mode: Optional[str] = None
```

### LifecycleState

Conservative state machine:

```python
class LifecycleState(Enum):
    REGISTERED = "REGISTERED"       # Initial registration
    EVALUATED = "EVALUATED"         # QE evidence obtained
    APPROVED = "APPROVED"           # Passed admission gates
    PRODUCTION_READY = "PRODUCTION_READY"  # Ready for production
    DEPRECATED = "DEPRECATED"       # Superseded by better factor
    RETIRED = "RETIRED"             # Permanently disabled
```

**Legal Transitions:**
- REGISTERED → EVALUATED (evidence added)
- EVALUATED → APPROVED (gates passed)
- EVALUATED → EVALUATED (re-evaluation)
- APPROVED → PRODUCTION_READY (final checks)
- Any state → DEPRECATED
- DEPRECATED → RETIRED

**Illegal Transitions:**
- Any backwards transition (except DEPRECATED)
- Skip EVALUATED (must have evidence)
- Direct REGISTERED → APPROVED

### LineageRef

Complete lineage metadata.

```python
@dataclass(frozen=True)
class LineageRef:
    factor_id: str
    parent_refs: Tuple[ParentRef, ...]
    generation: int             # 0 = root, 1+ = derived
    campaign_id: Optional[str]
    trial_id: Optional[str]
    mutation_type: Optional[str]
```

**Parent-Child Relationships:**
- Root factors: `generation=0`, no parents
- Mutations: `generation=1`, single parent
- Combinations: `generation=1`, multiple parents

## Repository Architecture

### AssetRepository

Main API for factor management.

**Core Operations:**
```python
class AssetRepository:
    def register(metadata, lineage, tags) -> FactorAsset
    def get(factor_id) -> FactorAsset
    def exists(factor_id) -> bool
    def find_by_hash(canonical_hash) -> Optional[FactorAsset]
    def transition(factor_id, to_state, evidence_refs, ...) -> FactorAsset
    def list_by_state(state) -> list[FactorAsset]
    def list_all() -> list[FactorAsset]
    def get_events(factor_id) -> list[StateEvent]
    def stats() -> RepositoryStats
```

**Internal Structure:**
```
_assets: Dict[str, FactorAsset]           # factor_id -> asset
_hash_index: Dict[str, str]               # canonical_hash -> factor_id
_events: Dict[str, list[StateEvent]]      # factor_id -> events
_state_index: Dict[LifecycleState, set[str]]  # state -> factor_ids
```

**Append-Only Semantics:**
1. Register adds to `_assets`
2. Transition creates new `FactorAsset` (frozen), appends event
3. Old asset remains in history (via events)
4. No deletions, only state changes

### LifecycleOrchestrator

Centralized state transition validation and execution.

```python
class LifecycleOrchestrator:
    def validate_transition(request) -> None
    def execute_transition(request) -> TransitionResult
    def add_event_listener(listener) -> None
    def add_transition_hook(from_state, to_state, hook) -> None
    def get_legal_next_states(current_state) -> tuple[LifecycleState, ...]
    def get_transition_path(from_state, to_state) -> Optional[tuple[...]]
```

**Validation Rules:**
1. Check if transition is legal
2. Verify evidence requirements
3. Run pre-transition hooks
4. Execute transition
5. Fire event listeners
6. Run post-transition hooks

### LineageGraph

Factor ancestry DAG.

```python
class LineageGraph:
    def register_factor(lineage) -> None
    def get_parents(factor_id) -> tuple[str, ...]
    def get_children(factor_id) -> tuple[str, ...]
    def get_ancestors(factor_id, max_depth) -> tuple[str, ...]
    def get_descendants(factor_id, max_depth) -> tuple[str, ...]
    def get_lineage_depth(factor_id) -> int
    def has_cycle(factor_id) -> bool
    def is_root(factor_id) -> bool
    def is_leaf(factor_id) -> bool
```

**Graph Structure:**
```
_edges: Dict[str, set[str]]  # parent_id -> {child_id, ...}
_reverse: Dict[str, set[str]]  # child_id -> {parent_id, ...}
_metadata: Dict[tuple[str, str], LineageEdge]  # (parent, child) -> edge
```

### SnapshotManager

Point-in-time repository views.

```python
class SnapshotManager:
    def create_snapshot(assets, events, query) -> SnapshotResult
    def get_state_at_time(factor_id, events, timestamp) -> LifecycleState
    def get_state_transitions(factor_id, events, start, end) -> tuple[StateEvent, ...]
    def compare_snapshots(assets, events, timestamp_a, timestamp_b) -> dict
```

**Use Cases:**
- Reproduce historical factor set
- Compare portfolio composition over time
- Audit trail for compliance

## Selection and Admission

### Gates

Evidence-based admission gates.

```python
# Protocol
class EvidenceGate(Protocol):
    def evaluate(factor_id, evidence_id, metric_name, metric_value) -> GateEvaluation

# Implementations
class ThresholdGate:  # metric >= threshold
class CompositeGate:  # AND/OR combination
```

**Gate Evaluation:**
```python
@dataclass
class GateEvaluation:
    gate_name: str
    gate_version: str
    result: GateResult  # PASS, FAIL, SKIP, ERROR
    passed: bool
    factor_id: str
    evidence_id: str
    metric_name: Optional[str]
    metric_value: Optional[float]
    threshold: Optional[float]
```

### SelectionPolicy

Admission decision logic.

```python
class SelectionPolicy:
    def make_decision(
        factor_id,
        gate_evaluations,
        evidence_refs,
        operator,
    ) -> SelectionDecision
```

**Decision Flow:**
1. Evaluate all gates
2. Check composite criteria
3. Record decision with rationale
4. Return APPROVED/REJECTED with reason

## Clustering and Similarity

### Family Detection

Group correlated factors.

```python
# Connected components (simple)
class ConnectedComponents:
    def find_components() -> ClusterResult

# Modularity clustering (Louvain)
class ModularityClustering:
    def cluster(max_iterations) -> ClusterResult
```

**Clustering Input:**
- Correlation graph (SparseCorrelationGraph)
- Threshold for edge inclusion
- Optional similarity measure

**Output:**
```python
@dataclass
class ClusterResult:
    factor_to_cluster: Dict[str, int]
    cluster_sizes: Dict[int, int]
    num_clusters: int
```

### Representative Selection

Choose best factor per family.

```python
class FamilyRepresentativeSelector:
    def select_representatives(
        family,
        factor_ids,
        method,  # MAX_IC, MIN_CORRELATION, etc.
        max_representatives,
        ic_provider,
        correlation_provider,
    ) -> RepresentativeSelection
```

**Selection Methods:**
- MAX_IC: Highest information coefficient
- MIN_CORRELATION: Most orthogonal to others
- EQUAL_WEIGHT: Equal-weight combination
- FIRST: First discovered
- RANDOM: Random selection

## Integration Boundaries

### FactorEngine (via Protocol)

```python
class FactorIdentityProvider(Protocol):
    def get_canonical_hash(expression) -> str
    def get_canonical_repr(expression) -> str
    def get_identity_ref(expression) -> str
```

**Usage:**
```python
# FA never parses expressions, delegates to FE
fe_adapter = create_fe_adapter()  # from FE package
canonical_hash = fe_adapter.get_canonical_hash(expression)
factor_id = create_factor_id(canonical_hash)
```

### QuantEvaluator (via Protocol)

```python
class EvidenceProvider(Protocol):
    def get_evidence(query) -> Optional[EvidenceResult]
    def has_evidence(factor_id, evaluation_run_id) -> bool
    def get_latest_evidence(factor_id) -> Optional[EvidenceResult]
```

**Evidence Flow:**
1. QE evaluates factor, produces EvaluationBundle
2. QE stores bundle (in QE's storage)
3. FA stores EvidenceRef with bundle ID and metric snapshot
4. FA queries QE via protocol when full evidence needed

### DataAccess (No Direct Dependency)

FA never fetches factor values:
- DA owns raw values
- FA owns metadata
- FE computes on demand via DA

## Testing Strategy

### Unit Tests (28 files, 276 tests)

**Coverage:**
- Contracts: Immutability, validation
- Repository: Register, transition, query
- Lifecycle: State machine rules
- Lineage: DAG construction, traversal
- Clustering: Algorithm correctness
- Snapshots: Time-travel queries
- Gates: Threshold evaluation

### Integration Tests

- Full workflow: Register → Evaluate → Approve → Production
- Multi-factor batch operations
- Lineage tracking across generations
- Snapshot consistency

### Architectural Constraint Tests

**`tests/no_raw_values/test_no_raw_values.py`:**
- Verify no NumPy arrays stored
- No Pandas DataFrames
- No raw time series
- Only references and bounded snapshots

## Performance Considerations

### Memory Usage

**In-Memory Repository:**
- O(N) for N factors
- Typical: 10K factors ≈ 10MB
- With full event history: 10K factors × 10 events ≈ 50MB

**Future: Persistent Storage**
- SQLite for local
- PostgreSQL for production
- S3/COS for event log

### Query Performance

**Current (In-Memory):**
- `get(factor_id)`: O(1)
- `find_by_hash(hash)`: O(1) via hash index
- `list_by_state(state)`: O(1) via state index
- `get_ancestors(factor_id)`: O(depth) via DAG

**Future Optimizations:**
- Index on tags
- Index on creation_time
- Full-text search on expressions (via FE)

## Design Decisions

### Why Append-Only?

**Alternatives considered:**
1. Mutable records (update in place)
2. Delete and recreate
3. Versioned entities

**Decision:** Append-only wins for:
- Complete audit trail
- Reproducibility (snapshots)
- Concurrent access (read-only after write)
- Simplicity (no update conflicts)

### Why No Raw Values?

**Alternatives:**
1. Store factor values in FA
2. Cache recent values

**Decision:** No raw values because:
- DA already stores values (duplication)
- FA is governance, not data layer
- Memory footprint explosion
- Clear boundary separation

### Why Protocol-Based?

**Alternatives:**
1. Hard dependency on FE/QE
2. No integration (pure isolation)

**Decision:** Protocols because:
- Testable without full stack
- Adapter pattern for different FE/QE
- Clear contracts at boundaries
- No circular dependencies

## Future Extensions

### Wave 2

- **Persistent storage:** SQLite/PostgreSQL backend
- **Distributed coordination:** Multi-writer consistency
- **Advanced similarity:** LSH, embedding-based
- **Automated re-evaluation:** Trigger based on data drift

### Wave 3

- **Real-time updates:** Stream processing
- **Multi-tenancy:** Org/team isolation
- **Factor marketplace:** Sharing and discovery
- **ML-based selection:** Learned admission policies

---

**Last Updated:** 2026-08-14  
**Authors:** Quant Platform Team
