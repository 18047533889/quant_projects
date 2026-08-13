# FactorAssets Quick Start

**5-Minute Guide to Factor Registry and Lifecycle**

## Installation

```bash
cd /home/shw/quant_projects/factor_assets
pip install -e .
```

## Basic Example

Register a factor and track its lifecycle:

```python
from factor_assets import (
    AssetRepository,
    FactorAsset,
    AssetMetadata,
    LifecycleState,
    create_factor_id,
)

# Create repository
repo = AssetRepository()

# Register a new factor
metadata = AssetMetadata(
    factor_id="F_abc123",
    canonical_hash="abc123def456",
    frequency="daily",
    domains=("equity",),
    data_sources=("market",),
)

asset = repo.register(
    metadata=metadata,
    lineage=None,  # Root factor (no parents)
    tags={"strategy": "momentum", "author": "quant_team"},
)

print(f"Registered: {asset.factor_id}")
print(f"State: {asset.lifecycle_state}")  # REGISTERED
```

## Lifecycle Transitions

Move factor through states based on evidence:

```python
from factor_assets import EvidenceRef

# Add evaluation evidence
evidence_ref = EvidenceRef(
    evaluation_id="eval_001",
    factor_id="F_abc123",
    evidence_type="qe_bundle",
    metric_snapshot={"mean_ic": 0.045, "ic_std": 0.12, "coverage": 0.95},
    timestamp="2026-08-14T10:00:00Z",
)

# Transition to EVALUATED
asset = repo.transition(
    factor_id="F_abc123",
    to_state=LifecycleState.EVALUATED,
    evidence_refs=[evidence_ref],
    operator="system",
)

print(f"New state: {asset.lifecycle_state}")  # EVALUATED

# Transition to APPROVED (if gates pass)
asset = repo.transition(
    factor_id="F_abc123",
    to_state=LifecycleState.APPROVED,
    operator="quant_team",
    reason="Passed all admission gates",
)

print(f"Now: {asset.lifecycle_state}")  # APPROVED
```

## Query Repository

```python
# Get specific factor
asset = repo.get("F_abc123")

# Check existence
exists = repo.exists("F_abc123")

# Find by hash (deduplication)
asset = repo.find_by_hash("abc123def456")

# List by state
approved = repo.list_by_state(LifecycleState.APPROVED)
print(f"Approved factors: {len(approved)}")

# Get all factors
all_factors = repo.list_all()

# Repository statistics
stats = repo.stats()
print(f"Total: {stats.total_assets}")
print(f"By state: {stats.by_state}")
```

## Seen Index (Deduplication)

Track which factors have been seen:

```python
from factor_assets import SeenIndex, SeenRecord

# Create seen index
seen = SeenIndex()

# Mark factor as seen
record = seen.record(
    canonical_hash="abc123def456",
    factor_id="F_abc123",
    origin="quant_evaluator",
    context={"trial_id": "trial_001"},
)

# Check if seen before
is_seen = seen.is_seen("abc123def456")
print(f"Seen before: {is_seen}")  # True

# Get seen record
record = seen.get("abc123def456")
print(f"First seen: {record.first_seen_at}")
print(f"Trial: {record.trial_id}")
```

## Factor Sets

Group factors for batch operations:

```python
from factor_assets import FactorSet, FactorSetSpec

# Define selection criteria
spec = FactorSetSpec(
    name="momentum_factors",
    description="All momentum-based factors",
    filters={
        "lifecycle_state": "APPROVED",
        "tags.strategy": "momentum",
        "min_ic": 0.03,
    },
    max_factors=50,
)

# Create factor set
factor_set = FactorSet(
    set_id="set_001",
    spec=spec,
    factor_ids=("F_abc123", "F_def456", "F_ghi789"),
    snapshot_time="2026-08-14T10:00:00Z",
)

print(f"Set: {factor_set.name}")
print(f"Size: {factor_set.size}")
```

## Lineage Tracking

Track factor ancestry:

```python
from factor_assets import LineageRef, ParentRef, LineageGraph

# Register parent-child relationship
parent_ref = ParentRef(
    parent_id="F_abc123",
    contribution_weight=1.0,
    relationship_type="mutation",
)

lineage = LineageRef(
    factor_id="F_child001",
    parent_refs=(parent_ref,),
    generation=1,
    campaign_id="campaign_001",
    trial_id="trial_042",
    mutation_type="window_adjust",
)

# Register with lineage
child_asset = repo.register(
    metadata=child_metadata,
    lineage=lineage,
    tags={},
)

# Query lineage graph
graph = LineageGraph()
graph.register_factor(lineage)

# Get ancestors
parents = graph.get_parents("F_child001")
print(f"Parents: {parents}")  # ('F_abc123',)

ancestors = graph.get_ancestors("F_child001")
print(f"All ancestors: {ancestors}")
```

## Selection Gates

Evaluate factors against admission criteria:

```python
from factor_assets.selection import ThresholdGate, GateResult

# Create gate
ic_gate = ThresholdGate(
    gate_name="min_ic_gate",
    gate_version="1.0",
    threshold=0.03,
)

# Evaluate
evaluation = ic_gate.evaluate(
    factor_id="F_abc123",
    evidence_id="eval_001",
    metric_name="mean_ic",
    metric_value=0.045,
)

print(f"Gate result: {evaluation.result}")  # PASS
print(f"Passed: {evaluation.passed}")  # True
```

## Selection Policy

Make admission decisions:

```python
from factor_assets.selection import SelectionPolicy, SelectionReason

# Create policy
policy = SelectionPolicy()

# Make decision
decision = policy.make_decision(
    factor_id="F_abc123",
    gate_evaluations=[evaluation],
    evidence_refs=[evidence_ref],
    operator="system",
)

print(f"Decision: {decision.approved}")  # True
print(f"Reason: {decision.reason}")  # APPROVED
```

## Lifecycle Events

Track state change history:

```python
# Get lifecycle events for a factor
events = repo.get_events("F_abc123")

for event in events:
    print(f"{event.timestamp}: {event.from_state} -> {event.to_state}")
    print(f"  Operator: {event.operator}")
    print(f"  Reason: {event.reason}")
```

## Snapshots

Point-in-time views:

```python
from factor_assets.registry import SnapshotManager, SnapshotQuery

manager = SnapshotManager()

# Create snapshot
query = SnapshotQuery(
    as_of_timestamp="2026-08-14T10:00:00Z",
    include_states=[LifecycleState.APPROVED, LifecycleState.PRODUCTION_READY],
)

snapshot = manager.create_snapshot(
    assets=repo.list_all(),
    events=all_events,
    query=query,
)

print(f"Snapshot at {snapshot.as_of_timestamp}")
print(f"Matched: {len(snapshot.matched_assets)}")
```

## Complete Workflow

```python
from factor_assets import (
    AssetRepository,
    AssetMetadata,
    LineageRef,
    EvidenceRef,
    LifecycleState,
    SeenIndex,
    create_factor_id,
)

# 1. Initialize
repo = AssetRepository()
seen = SeenIndex()

# 2. Check if factor already seen
canonical_hash = "abc123def456"
if seen.is_seen(canonical_hash):
    print("Factor already seen, skipping")
    exit()

# 3. Register new factor
factor_id = create_factor_id(canonical_hash, prefix="F")
metadata = AssetMetadata(
    factor_id=factor_id,
    canonical_hash=canonical_hash,
    frequency="daily",
    domains=("equity",),
    data_sources=("market", "fundamental"),
)

asset = repo.register(metadata=metadata, lineage=None, tags={})

# 4. Mark as seen
seen.record(canonical_hash, factor_id, origin="quant_evaluator")

# 5. Evaluate and add evidence
evidence_ref = EvidenceRef(
    evaluation_id="eval_001",
    factor_id=factor_id,
    evidence_type="qe_bundle",
    metric_snapshot={"mean_ic": 0.042, "coverage": 0.96},
    timestamp="2026-08-14T10:00:00Z",
)

asset = repo.transition(
    factor_id=factor_id,
    to_state=LifecycleState.EVALUATED,
    evidence_refs=[evidence_ref],
    operator="system",
)

# 6. Apply gates and approve
# (gates check metric thresholds)
asset = repo.transition(
    factor_id=factor_id,
    to_state=LifecycleState.APPROVED,
    operator="quant_team",
)

# 7. Query approved factors
approved = repo.list_by_state(LifecycleState.APPROVED)
print(f"Total approved: {len(approved)}")
```

## Common Patterns

### Pattern 1: Batch Registration

```python
# Register multiple factors
factor_hashes = ["hash1", "hash2", "hash3"]

for h in factor_hashes:
    if not seen.is_seen(h):
        factor_id = create_factor_id(h)
        metadata = AssetMetadata(
            factor_id=factor_id,
            canonical_hash=h,
            frequency="daily",
            domains=("equity",),
            data_sources=("market",),
        )
        repo.register(metadata=metadata, lineage=None, tags={})
        seen.record(h, factor_id, origin="batch_import")
```

### Pattern 2: Deduplication Before Evaluation

```python
# Before expensive evaluation, check if factor exists
canonical_hash = compute_hash(factor_expression)

if seen.is_seen(canonical_hash):
    record = seen.get(canonical_hash)
    print(f"Already evaluated as {record.factor_id}")
    asset = repo.get(record.factor_id)
    # Use existing evaluation
else:
    # Proceed with evaluation
    evaluate_factor(factor_expression)
```

### Pattern 3: Lifecycle-Based Queries

```python
# Get all production-ready factors
production = repo.list_by_state(LifecycleState.PRODUCTION_READY)

# Get recently approved
approved = repo.list_by_state(LifecycleState.APPROVED)
recent = [a for a in approved if a.created_at > cutoff_time]

# Get factors needing re-evaluation
evaluated = repo.list_by_state(LifecycleState.EVALUATED)
stale = [a for a in evaluated if a.last_evidence_age_days > 30]
```

## Next Steps

- **Full API:** See [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Testing:** See [TESTING.md](TESTING.md)
- **Platform Guide:** See [PLATFORM_GUIDE.md](../../PLATFORM_GUIDE.md)

## Tips

1. **Deduplication:** Always check `SeenIndex` before registration
2. **Immutability:** Factor identity and canonical hash never change
3. **Evidence refs:** Store references, not raw values
4. **State transitions:** Follow lifecycle rules (no backwards transitions)
5. **Lineage:** Track ancestry for factor families

## Troubleshooting

**Q: DuplicateIdentityError**  
A: Factor with this canonical_hash already registered. Use `find_by_hash()` to retrieve existing factor.

**Q: LifecycleConflictError**  
A: Invalid state transition. Check allowed transitions with state machine.

**Q: AssetNotFoundError**  
A: Factor ID not in repository. Verify ID or check if factor was registered.
