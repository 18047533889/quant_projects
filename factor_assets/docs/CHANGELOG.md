# FactorAssets Changelog

All notable changes to FactorAssets will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-14

### Added - Core Functionality

#### Contracts
- `FactorAsset`: Complete factor record with metadata, lifecycle, lineage, evidence refs
- `AssetMetadata`: Core identity (factor_id, canonical_hash, frequency, domains, sources)
- `FactorSet`: Selected set of factors by IDs
- `FactorSetSpec`: Specification for factor set selection
- `EvidenceRef`: Reference to single QE evaluation result
- `EvidenceBundleRef`: Reference to complete QE evaluation bundle
- `LifecycleState`: Enum (REGISTERED, EVALUATED, APPROVED, PRODUCTION_READY, DEPRECATED, RETIRED)
- `StateTransition`: Legal state transition specification
- `StateEvent`: Immutable lifecycle state change event
- `LifecycleConflictError`: Exception for illegal transitions
- `LineageRef`: Complete lineage reference with parents, campaign, trial info
- `ParentRef`: Reference to parent factor in lineage
- `ContractEnvelope`: Cross-package versioned envelope

#### Registry
- `AssetRepository`: Append-only in-memory repository
  - `register(metadata, lineage, tags)` - Register new factor
  - `get(factor_id)` - Retrieve factor by ID
  - `exists(factor_id)` - Check existence
  - `find_by_hash(canonical_hash)` - Find by content hash (deduplication)
  - `transition(factor_id, to_state, ...)` - State transition
  - `list_by_state(state)` - Query by lifecycle state
  - `list_all()` - Get all factors
  - `get_events(factor_id)` - Retrieve state history
  - `stats()` - Repository statistics
- `DuplicateIdentityError`: Exception for duplicate registration
- `AssetNotFoundError`: Exception for missing assets
- `RepositoryStats`: Repository statistics dataclass

#### Lifecycle Management
- `LifecycleOrchestrator`: Centralized state transition orchestration
  - `validate_transition(request)` - Validate transition legality
  - `execute_transition(request)` - Execute state change
  - `add_event_listener(listener)` - Register event listener
  - `add_transition_hook(from_state, to_state, hook)` - Register hook
  - `get_legal_next_states(current_state)` - Query legal transitions
  - `get_transition_path(from_state, to_state)` - Find transition path
- `StateMachine`: Pure validation logic
  - `can_transition(from_state, to_state)` - Check legality (static)
  - `validate(from_state, to_state, evidence_keys)` - Validate transition (static)
  - `create_event(...)` - Create state event (static)
- `TransitionRequest`: Transition request dataclass
- `TransitionResult`: Transition result with event and warnings

#### Lineage Tracking
- `LineageGraph`: Factor lineage DAG tracker
  - `register_factor(lineage)` - Register factor in graph
  - `register_campaign(campaign)` - Register search campaign
  - `get_parents(factor_id)` - Get immediate parents
  - `get_children(factor_id)` - Get immediate children
  - `get_edge(parent_id, child_id)` - Get lineage edge
  - `get_ancestors(factor_id, max_depth)` - Recursive ancestors
  - `get_descendants(factor_id, max_depth)` - Recursive descendants
  - `get_lineage_depth(factor_id)` - Depth from root
  - `get_campaign(campaign_id)` - Get campaign metadata
  - `get_campaign_factors(campaign_id)` - Factors in campaign
  - `get_factor_campaign(factor_id)` - Factor's campaign
  - `has_cycle(factor_id)` - Cycle detection
  - `is_root(factor_id)` - Check if root factor
  - `is_leaf(factor_id)` - Check if leaf factor
  - `get_roots()` - All root factors
  - `get_leaves()` - All leaf factors
  - `stats()` - Graph statistics
- `LineageEdge`: Directed lineage edge
- `CampaignMetadata`: Search campaign metadata

#### Snapshots
- `SnapshotManager`: Point-in-time registry snapshots
  - `create_snapshot(assets, events, query)` - Create snapshot
  - `get_state_at_time(factor_id, events, timestamp)` - Historical state
  - `get_state_transitions(factor_id, events, start, end)` - State history
  - `compare_snapshots(assets, events, timestamp_a, timestamp_b)` - Compare
- `SnapshotQuery`: Snapshot query specification
- `SnapshotResult`: Snapshot query result

#### Identity Management
- `FactorIdentityProvider`: Protocol for FE identity boundary
  - `get_canonical_hash(expression)` - Obtain content hash
  - `get_canonical_repr(expression)` - Canonical representation
  - `get_identity_ref(expression)` - Identity reference
- `FactorIdentity`: Complete factor identity dataclass
- `create_factor_id(canonical_hash, prefix)` - Generate factor ID from hash

#### Seen Index (Deduplication)
- `SeenIndex`: In-memory seen history
  - `record(canonical_hash, factor_id, ...)` - Mark as seen
  - `is_seen(canonical_hash)` - Check if seen
  - `get(canonical_hash)` - Get seen record
  - `get_all()` - All seen records
  - `count()` - Number of unique factors
- `SeenRecord`: First-seen record dataclass

#### Selection Gates
- `EvidenceGate`: Protocol for admission gates
- `ThresholdGate`: Simple threshold gate
  - `evaluate(factor_id, evidence_id, metric_name, metric_value)` - Evaluate
  - Properties: `gate_name`, `gate_version`, `threshold`
- `CompositeGate`: Composite AND/OR gate
  - `evaluate_all(factor_id, evidence_id, metrics)` - Evaluate all gates
- `GateEvaluation`: Gate evaluation result
- `GateResult`: Enum (PASS, FAIL, SKIP, ERROR)

#### Selection Policy
- `SelectionPolicy`: Selection decision policy
  - `make_decision(factor_id, gate_evaluations, ...)` - Make admission decision
  - `get_decision(factor_id)` - Retrieve decision
  - `get_all_decisions(factor_id)` - All decisions for factor
  - `count_approved()` - Count approved factors
  - `count_rejected()` - Count rejected factors
  - `count_total()` - Total decisions
- `SelectionDecision`: Immutable selection decision record
- `SelectionReason`: Enum (APPROVED, REJECTED_GATE_FAILURE, REJECTED_SIMILARITY, ...)

#### Aggregation
- `AggregationSpec`: Factor aggregation specification
- `AggregationResult`: Aggregation computation result
- `WeightingScheme`: Enum (EQUAL, IC_WEIGHTED, INVERSE_VARIANCE, RANK_IC, MEAN_ABSOLUTE_IC, CUSTOM)
- `FamilyRepresentativeSelector`: Representative selection
  - `select_representatives(family, factor_ids, method, ...)` - Select best factors
- `RepresentativeSelection`: Selection result
- `RepresentativeSelectionMethod`: Enum (MAX_IC, MIN_CORRELATION, EQUAL_WEIGHT, FIRST, RANDOM)
- `ICProvider`: Protocol for IC values
- `CorrelationProvider`: Protocol for correlations

#### Clustering
- `ConnectedComponents`: Connected component finder
  - `find_components()` - Find families via connectivity
- `ModularityClustering`: Louvain-style community detection
  - `cluster(max_iterations)` - Modularity-based clustering
- `ClusterResult`: Clustering result
  - Properties: `num_clusters`
  - Methods: `get_cluster_members(cluster_id)`, `get_all_clusters()`
- `LineageDetector`: Parent-child relationship detection
  - `detect_lineage(family_members, family_id)` - Detect family lineage
  - `detect_all_lineages(cluster_result)` - Lineage for all families
- `ParentChildRelation`: Parent-child relation record
- `FamilyLineage`: Complete family lineage structure
  - Properties: `size`, `depth`
  - Methods: `get_children(parent_id)`, `get_parents(child_id)`

#### Graph Operations
- `SparseCorrelationGraph`: Sparse adjacency list graph
  - `neighbors(factor_id)` - Get neighbors with weights
  - `degree(factor_id)` - Node degree
  - `has_edge(factor_a, factor_b)` - Check edge existence
  - `get_correlation(factor_a, factor_b)` - Get edge weight
  - `subgraph(node_subset)` - Extract subgraph
  - `to_edge_list()` - Convert to edge list
  - `density()` - Graph density
  - Properties: `nodes`, `node_count`, `edge_count`
- `CorrelationEdge`: Correlation edge dataclass
- `EdgeFilter`: Abstract edge filter (ABC)
- `ThresholdFilter`: Threshold-based filter
- `TopKFilter`: Top-K edges per node filter
- `CompositeFilter`: Sequential filter composition
- `SignFilter`: Sign-based filter (positive/negative/both)

#### Similarity Measurement
- `SimilarityMeasure`: Protocol for similarity computation
- `CorrelationSimilarity`: Stub implementation (cache-based)
  - `add_result(result)` - Add similarity result
  - `compute_similarity(factor_id_a, factor_id_b, ...)` - Compute similarity
  - `find_similar(factor_id, threshold, max_results, ...)` - Find similar factors
  - `count()` - Number of cached results
  - `clear()` - Clear cache
- `SimilarityResult`: Similarity computation result
- `SimilarityMethod`: Enum (PEARSON, SPEARMAN, KENDALL)

#### Novelty (QE Evidence Protocol)
- `EvidenceProvider`: Protocol for QE evidence boundary
  - `get_evidence(query)` - Retrieve evidence
  - `has_evidence(factor_id, evaluation_run_id)` - Check existence
  - `get_latest_evidence(factor_id)` - Get latest evaluation
- `MockEvidenceProvider`: Mock implementation for testing
  - `add_evidence(result)` - Add mock evidence
  - `count()` - Number of evidence records
  - `clear()` - Clear cache
- `EvidenceQuery`: Evidence lookup query
- `EvidenceResult`: Evidence retrieval result

### Testing
- 28 test files, 276 test functions
- Unit tests for all core contracts
- Repository operations (register, transition, query)
- Lifecycle state machine (legal/illegal transitions)
- Lineage graph (DAG construction, traversal)
- Clustering algorithms (connected components, modularity)
- Point-in-time snapshots
- Gate evaluation and selection policy
- Architectural constraint tests (no raw values)

### Documentation
- Package overview
- Design principles documented
- Protocol-based integration pattern
- Append-only semantics

### Known Limitations

#### Not Yet Implemented
- Persistent storage (in-memory only)
- Distributed coordination (single-writer only)
- Advanced similarity (stub only)
- Real-time sync across repositories
- Automated re-evaluation triggers

#### Architecture
- In-memory repository limits to ~100K factors
- No multi-tenancy support
- No access control / permissions
- No audit log export

#### Integration
- FE adapter is protocol only (no implementation)
- QE evidence provider is protocol only
- No DA integration (not needed)

### Breaking Changes

N/A - Initial release

### Dependencies
- No external dependencies (stdlib only)
- Optional: factor-engine (for FE adapter)
- Optional: quant-evaluator (for QE adapter)

### Deprecations

None

### Security

None

---

## [Unreleased]

### Planned for 0.2.0 (Wave 2)

#### Persistence
- [ ] SQLite backend for local persistence
- [ ] PostgreSQL backend for production
- [ ] Event log to S3/COS
- [ ] Snapshot export/import

#### Performance
- [ ] Index on tags for fast filtering
- [ ] Full-text search on expressions
- [ ] Batch operations optimization
- [ ] Query result caching

#### Features
- [ ] Similarity computation (not just stub)
- [ ] Automated re-evaluation scheduling
- [ ] Factor expiry policies
- [ ] Multi-tenancy support

#### Integration
- [ ] Complete FE adapter implementation
- [ ] Complete QE adapter implementation
- [ ] Webhook notifications for state changes

### Planned for 0.3.0 (Wave 3)

- [ ] Distributed repository coordination
- [ ] Real-time sync across instances
- [ ] Factor marketplace / sharing
- [ ] ML-based admission policies
- [ ] Web dashboard

---

## Version History

- **0.1.0** (2026-08-14): Initial bounded implementation

---

## Upgrade Guide

N/A - Initial release

---

## Contributors

Quant Platform Team

---

**Last Updated:** 2026-08-14
