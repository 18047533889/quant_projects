# FactorAssets API Reference

**Version:** 0.1.0  
**Last Updated:** 2026-08-14

## Table of Contents

1. [Contracts](#contracts)
2. [Registry](#registry)
3. [Identity](#identity)
4. [Seen Index](#seen-index)
5. [Selection](#selection)
6. [Clustering](#clustering)
7. [Graph](#graph)
8. [Lineage](#lineage)
9. [Aggregation](#aggregation)

---

## Contracts

### FactorAsset

Complete factor asset record.

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

**Properties:**
- `factor_id: str` - From metadata
- `canonical_hash: str` - From metadata

---

### AssetMetadata

Core identity and domain information.

```python
@dataclass(frozen=True)
class AssetMetadata:
    factor_id: str
    canonical_hash: str
    frequency: str
    domains: Tuple[str, ...]
    data_sources: Tuple[str, ...]
    timing_kind: Optional[str] = None
    creation_mode: Optional[str] = None
```

**Example:**
```python
metadata = AssetMetadata(
    factor_id="F_abc123",
    canonical_hash="abc123def456",
    frequency="daily",
    domains=("equity",),
    data_sources=("market", "fundamental"),
)
```

---

### LifecycleState

Factor lifecycle states.

```python
class LifecycleState(Enum):
    REGISTERED = "REGISTERED"
    EVALUATED = "EVALUATED"
    APPROVED = "APPROVED"
    PRODUCTION_READY = "PRODUCTION_READY"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"
```

**Legal Transitions:**
- REGISTERED → EVALUATED
- EVALUATED → EVALUATED (re-evaluation)
- EVALUATED → APPROVED
- APPROVED → PRODUCTION_READY
- Any → DEPRECATED
- DEPRECATED → RETIRED

---

### StateEvent

Immutable state change event.

```python
@dataclass(frozen=True)
class StateEvent:
    event_id: str
    factor_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: datetime
    operator: str
    reason: Optional[str]
    evidence_refs: Tuple[EvidenceRef, ...]
    metadata: Dict[str, Any]
```

---

### EvidenceRef

Reference to QE evaluation.

```python
@dataclass(frozen=True)
class EvidenceRef:
    evaluation_id: str
    factor_id: str
    evidence_type: str
    metric_snapshot: Dict[str, Any]
    timestamp: str
    source_system: str = "quant_evaluator"
    metadata: Optional[Dict[str, Any]] = None
```

**Example:**
```python
evidence_ref = EvidenceRef(
    evaluation_id="eval_20260814_001",
    factor_id="F_abc123",
    evidence_type="qe_bundle",
    metric_snapshot={
        "mean_ic": 0.045,
        "ic_std": 0.12,
        "coverage": 0.95,
        "turnover": 0.23,
    },
    timestamp="2026-08-14T10:30:00Z",
)
```

---

### LineageRef

Complete lineage metadata.

```python
@dataclass(frozen=True)
class LineageRef:
    factor_id: str
    parent_refs: Tuple[ParentRef, ...]
    generation: int
    campaign_id: Optional[str] = None
    trial_id: Optional[str] = None
    mutation_type: Optional[str] = None
```

**Example:**
```python
lineage = LineageRef(
    factor_id="F_child",
    parent_refs=(
        ParentRef(parent_id="F_parent1", contribution_weight=0.6),
        ParentRef(parent_id="F_parent2", contribution_weight=0.4),
    ),
    generation=1,
    campaign_id="campaign_001",
    trial_id="trial_042",
    mutation_type="linear_combination",
)
```

---

### FactorSet

Selected set of factors.

```python
@dataclass(frozen=True)
class FactorSet:
    set_id: str
    name: str
    spec: FactorSetSpec
    factor_ids: Tuple[str, ...]
    snapshot_time: str
    metadata: Dict[str, Any]
```

**Properties:**
- `size: int` - Number of factors

---

## Registry

### AssetRepository

Main repository class for factor management.

```python
class AssetRepository:
    def __init__(self):
        """Create empty repository."""
```

#### register

```python
def register(
    self,
    metadata: AssetMetadata,
    lineage: Optional[LineageRef],
    tags: Dict[str, Any],
) -> FactorAsset:
```

Register new factor in repository.

**Parameters:**
- `metadata`: Factor metadata with identity
- `lineage`: Optional lineage information
- `tags`: User-defined tags

**Returns:** Registered FactorAsset

**Raises:**
- `DuplicateIdentityError`: If canonical_hash already exists

**Example:**
```python
repo = AssetRepository()

metadata = AssetMetadata(
    factor_id="F_001",
    canonical_hash="hash001",
    frequency="daily",
    domains=("equity",),
    data_sources=("market",),
)

asset = repo.register(
    metadata=metadata,
    lineage=None,
    tags={"strategy": "momentum", "author": "quant_team"},
)
```

---

#### get

```python
def get(self, factor_id: str) -> FactorAsset:
```

Retrieve factor by ID.

**Raises:** `AssetNotFoundError` if not found

---

#### exists

```python
def exists(self, factor_id: str) -> bool:
```

Check if factor exists in repository.

---

#### find_by_hash

```python
def find_by_hash(self, canonical_hash: str) -> Optional[FactorAsset]:
```

Find factor by canonical hash (deduplication).

**Returns:** FactorAsset if found, None otherwise

**Example:**
```python
# Check if factor already registered
existing = repo.find_by_hash("abc123def456")
if existing:
    print(f"Already registered as {existing.factor_id}")
```

---

#### transition

```python
def transition(
    self,
    factor_id: str,
    to_state: LifecycleState,
    evidence_refs: Optional[List[EvidenceRef]] = None,
    operator: str = "system",
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FactorAsset:
```

Transition factor to new lifecycle state.

**Parameters:**
- `factor_id`: Factor to transition
- `to_state`: Target lifecycle state
- `evidence_refs`: Optional new evidence references
- `operator`: Who performed transition (default "system")
- `reason`: Optional reason for transition
- `metadata`: Additional metadata

**Returns:** Updated FactorAsset

**Raises:**
- `LifecycleConflictError`: If transition is illegal
- `AssetNotFoundError`: If factor not found

**Example:**
```python
# Add evaluation evidence
evidence = EvidenceRef(...)

asset = repo.transition(
    factor_id="F_001",
    to_state=LifecycleState.EVALUATED,
    evidence_refs=[evidence],
    operator="quant_evaluator",
    reason="Initial evaluation complete",
)
```

---

#### list_by_state

```python
def list_by_state(self, state: LifecycleState) -> List[FactorAsset]:
```

List all factors in given lifecycle state.

**Example:**
```python
approved = repo.list_by_state(LifecycleState.APPROVED)
print(f"Approved factors: {len(approved)}")
```

---

#### list_all

```python
def list_all(self) -> List[FactorAsset]:
```

List all factors in repository.

---

#### get_events

```python
def get_events(self, factor_id: str) -> List[StateEvent]:
```

Get lifecycle event history for factor.

**Example:**
```python
events = repo.get_events("F_001")
for event in events:
    print(f"{event.timestamp}: {event.from_state} → {event.to_state}")
```

---

#### stats

```python
def stats(self) -> RepositoryStats:
```

Get repository statistics.

**Returns:** RepositoryStats with total counts and breakdown by state

**Example:**
```python
stats = repo.stats()
print(f"Total: {stats.total_assets}")
print(f"By state: {stats.by_state}")
```

---

### LifecycleOrchestrator

Centralized state transition orchestration.

```python
class LifecycleOrchestrator:
    def __init__(self):
        """Create orchestrator."""
```

#### validate_transition

```python
def validate_transition(self, request: TransitionRequest) -> None:
```

Validate transition legality.

**Raises:** `LifecycleConflictError` if invalid

---

#### execute_transition

```python
def execute_transition(self, request: TransitionRequest) -> TransitionResult:
```

Execute state transition with hooks and listeners.

**Returns:** TransitionResult with event and warnings

---

#### add_transition_hook

```python
def add_transition_hook(
    self,
    from_state: LifecycleState,
    to_state: LifecycleState,
    hook: Callable[[TransitionRequest], None],
) -> None:
```

Register pre-transition hook.

**Example:**
```python
orchestrator = LifecycleOrchestrator()

def log_transition(request):
    print(f"Transitioning {request.factor_id}: {request.from_state} → {request.to_state}")

orchestrator.add_transition_hook(
    from_state=LifecycleState.EVALUATED,
    to_state=LifecycleState.APPROVED,
    hook=log_transition,
)
```

---

#### get_legal_next_states

```python
def get_legal_next_states(
    self,
    current_state: LifecycleState,
) -> Tuple[LifecycleState, ...]:
```

Get all legal next states from current state.

---

### LineageGraph

Factor lineage DAG.

```python
class LineageGraph:
    def __init__(self):
        """Create empty lineage graph."""
```

#### register_factor

```python
def register_factor(self, lineage: LineageRef) -> None:
```

Register factor and its parents in graph.

---

#### get_parents

```python
def get_parents(self, factor_id: str) -> Tuple[str, ...]:
```

Get immediate parent factor IDs.

**Example:**
```python
graph = LineageGraph()
graph.register_factor(lineage)

parents = graph.get_parents("F_child")
print(f"Parents: {parents}")
```

---

#### get_children

```python
def get_children(self, factor_id: str) -> Tuple[str, ...]:
```

Get immediate child factor IDs.

---

#### get_ancestors

```python
def get_ancestors(
    self,
    factor_id: str,
    max_depth: Optional[int] = None,
) -> Tuple[str, ...]:
```

Get all ancestor factor IDs (recursive).

**Parameters:**
- `factor_id`: Starting factor
- `max_depth`: Optional depth limit

**Returns:** All ancestor IDs

---

#### get_descendants

```python
def get_descendants(
    self,
    factor_id: str,
    max_depth: Optional[int] = None,
) -> Tuple[str, ...]:
```

Get all descendant factor IDs (recursive).

---

#### get_lineage_depth

```python
def get_lineage_depth(self, factor_id: str) -> int:
```

Get lineage depth (distance from root).

**Returns:** 0 for root, 1+ for derived factors

---

#### is_root

```python
def is_root(self, factor_id: str) -> bool:
```

Check if factor is a root (no parents).

---

#### is_leaf

```python
def is_leaf(self, factor_id: str) -> bool:
```

Check if factor is a leaf (no children).

---

#### has_cycle

```python
def has_cycle(self, factor_id: str) -> bool:
```

Check if lineage contains cycles.

---

### SnapshotManager

Point-in-time repository snapshots.

```python
class SnapshotManager:
    def __init__(self):
        """Create snapshot manager."""
```

#### create_snapshot

```python
def create_snapshot(
    self,
    assets: List[FactorAsset],
    events: Dict[str, List[StateEvent]],
    query: SnapshotQuery,
) -> SnapshotResult:
```

Create point-in-time snapshot.

**Parameters:**
- `assets`: All assets
- `events`: All events by factor_id
- `query`: Snapshot query with filters

**Returns:** SnapshotResult with matched assets

**Example:**
```python
manager = SnapshotManager()

query = SnapshotQuery(
    as_of_timestamp="2026-08-14T10:00:00Z",
    include_states=[LifecycleState.APPROVED],
)

snapshot = manager.create_snapshot(
    assets=repo.list_all(),
    events={f.factor_id: repo.get_events(f.factor_id) for f in repo.list_all()},
    query=query,
)

print(f"Snapshot: {len(snapshot.matched_assets)} factors")
```

---

## Identity

### create_factor_id

```python
def create_factor_id(canonical_hash: str, prefix: str = "F") -> str:
```

Generate factor ID from canonical hash.

**Parameters:**
- `canonical_hash`: Content-based hash from FE
- `prefix`: ID prefix (default "F")

**Returns:** Factor ID like "F_abc123de"

**Example:**
```python
from factor_assets import create_factor_id

factor_id = create_factor_id("abc123def456789", prefix="F")
print(factor_id)  # "F_abc123d"
```

---

### FactorIdentityProvider

Protocol for obtaining identity from FactorEngine.

```python
class FactorIdentityProvider(Protocol):
    def get_canonical_hash(self, expression: Any) -> str:
        """Get content-based hash."""
    
    def get_canonical_repr(self, expression: Any) -> str:
        """Get canonical representation."""
    
    def get_identity_ref(self, expression: Any) -> str:
        """Get complete identity reference."""
```

**Usage:**
```python
# Provided by FactorEngine package
fe_adapter = create_fe_adapter()

canonical_hash = fe_adapter.get_canonical_hash(expression)
factor_id = create_factor_id(canonical_hash)
```

---

## Seen Index

### SeenIndex

Deduplication cache.

```python
class SeenIndex:
    def __init__(self, fe_adapter=None):
        """Create seen index."""
```

#### record

```python
def record(
    self,
    canonical_hash: str,
    factor_id: str,
    origin: str,
    context: Optional[Dict] = None,
) -> SeenRecord:
```

Mark factor as seen.

**Parameters:**
- `canonical_hash`: Factor content hash
- `factor_id`: Registered factor ID
- `origin`: Source system (e.g., "quant_evaluator")
- `context`: Optional context metadata

**Returns:** SeenRecord

**Example:**
```python
from factor_assets import SeenIndex

seen = SeenIndex()

record = seen.record(
    canonical_hash="abc123",
    factor_id="F_abc123",
    origin="factor_optimizer",
    context={"trial_id": "trial_001"},
)
```

---

#### is_seen

```python
def is_seen(self, canonical_hash: str) -> bool:
```

Check if factor has been seen before.

**Example:**
```python
if seen.is_seen("abc123"):
    print("Already evaluated")
else:
    # Proceed with evaluation
    pass
```

---

#### get

```python
def get(self, canonical_hash: str) -> Optional[SeenRecord]:
```

Get seen record for hash.

**Returns:** SeenRecord if found, None otherwise

---

#### check_and_mark

```python
def check_and_mark(
    self,
    factor_definition: Any,
    trial_id: str,
) -> Tuple[bool, Optional[SeenRecord]]:
```

Check if seen and mark if not (requires FE adapter).

**Returns:** (is_duplicate, record_if_seen)

---

### SeenRecord

First-seen record.

```python
@dataclass
class SeenRecord:
    canonical_hash: str
    first_seen_at: datetime
    trial_id: str
    factor_id: Optional[str]
    metadata: Optional[Dict]
```

**Methods:**
- `to_dict() -> Dict`
- `from_dict(data: Dict) -> SeenRecord`

---

## Selection

### ThresholdGate

Simple threshold-based admission gate.

```python
class ThresholdGate:
    def __init__(
        self,
        gate_name: str,
        gate_version: str,
        threshold: float,
    ):
```

#### evaluate

```python
def evaluate(
    self,
    factor_id: str,
    evidence_id: str,
    metric_name: str,
    metric_value: float,
) -> GateEvaluation:
```

Evaluate metric against threshold.

**Returns:** GateEvaluation with PASS/FAIL

**Example:**
```python
from factor_assets.selection import ThresholdGate, GateResult

gate = ThresholdGate(
    gate_name="min_ic_gate",
    gate_version="1.0",
    threshold=0.03,
)

evaluation = gate.evaluate(
    factor_id="F_001",
    evidence_id="eval_001",
    metric_name="mean_ic",
    metric_value=0.045,
)

if evaluation.passed:
    print("Gate passed!")
```

---

### CompositeGate

Composite AND/OR gate.

```python
class CompositeGate:
    def __init__(
        self,
        gates: List[EvidenceGate],
        mode: str = "AND",  # "AND" or "OR"
    ):
```

#### evaluate_all

```python
def evaluate_all(
    self,
    factor_id: str,
    evidence_id: str,
    metrics: Dict[str, float],
) -> Tuple[GateEvaluation, List[GateEvaluation]]:
```

Evaluate all gates.

**Returns:** (composite_result, individual_results)

---

### SelectionPolicy

Admission decision policy.

```python
class SelectionPolicy:
    def __init__(self):
        """Create selection policy."""
```

#### make_decision

```python
def make_decision(
    self,
    factor_id: str,
    gate_evaluations: List[GateEvaluation],
    evidence_refs: List[EvidenceRef],
    operator: str,
) -> SelectionDecision:
```

Make admission decision based on gate results.

**Returns:** SelectionDecision with APPROVED/REJECTED

**Example:**
```python
from factor_assets.selection import SelectionPolicy

policy = SelectionPolicy()

decision = policy.make_decision(
    factor_id="F_001",
    gate_evaluations=[gate_eval],
    evidence_refs=[evidence_ref],
    operator="system",
)

if decision.approved:
    print(f"Admitted: {decision.reason}")
```

---

## Clustering

### ConnectedComponents

Connected component clustering.

```python
class ConnectedComponents:
    def __init__(self, graph: SparseCorrelationGraph):
        """Create clusterer."""
```

#### find_components

```python
def find_components(self) -> ClusterResult:
```

Find connected components in correlation graph.

**Returns:** ClusterResult with cluster assignments

**Example:**
```python
from factor_assets.clustering import ConnectedComponents
from factor_assets.graph import SparseCorrelationGraph

# Create graph
edges = [("F_1", "F_2", 0.8), ("F_2", "F_3", 0.75)]
graph = SparseCorrelationGraph.from_edges(edges)

# Find families
clusterer = ConnectedComponents(graph)
result = clusterer.find_components()

print(f"Found {result.num_clusters} families")
```

---

### ModularityClustering

Louvain-style community detection.

```python
class ModularityClustering:
    def __init__(self, graph: SparseCorrelationGraph):
        """Create clusterer."""
```

#### cluster

```python
def cluster(self, max_iterations: int = 100) -> ClusterResult:
```

Perform modularity-based clustering.

**Parameters:**
- `max_iterations`: Maximum iterations (default 100)

**Returns:** ClusterResult

---

## Graph

### SparseCorrelationGraph

Sparse correlation graph.

```python
class SparseCorrelationGraph:
    @classmethod
    def from_edges(
        cls,
        edges: List[Tuple[str, str, float]],
    ) -> SparseCorrelationGraph:
        """Create from edge list."""
```

**Properties:**
- `nodes: Set[str]` - All node IDs
- `node_count: int` - Number of nodes
- `edge_count: int` - Number of edges

**Methods:**
- `neighbors(factor_id) -> List[Tuple[str, float]]` - Get neighbors with weights
- `degree(factor_id) -> int` - Node degree
- `has_edge(factor_a, factor_b) -> bool` - Check edge
- `get_correlation(factor_a, factor_b) -> Optional[float]` - Get weight
- `subgraph(node_subset) -> SparseCorrelationGraph` - Extract subgraph
- `density() -> float` - Graph density

---

## Aggregation

### FamilyRepresentativeSelector

Select representative factors from families.

```python
class FamilyRepresentativeSelector:
    def __init__(self):
        """Create selector."""
```

#### select_representatives

```python
def select_representatives(
    self,
    family: str,
    factor_ids: List[str],
    method: RepresentativeSelectionMethod,
    max_representatives: int = 1,
    ic_provider: Optional[ICProvider] = None,
    correlation_provider: Optional[CorrelationProvider] = None,
) -> RepresentativeSelection:
```

Select best factors from family.

**Parameters:**
- `family`: Family identifier
- `factor_ids`: All factors in family
- `method`: Selection method (MAX_IC, MIN_CORRELATION, etc.)
- `max_representatives`: Maximum to select
- `ic_provider`: IC values (required for MAX_IC)
- `correlation_provider`: Correlations (required for MIN_CORRELATION)

**Returns:** RepresentativeSelection with chosen factors

---

**Last Updated:** 2026-08-14  
**Version:** 0.1.0
