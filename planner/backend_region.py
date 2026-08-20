# -*- coding: utf-8 -*-
"""Multi-backend physical region plan model (MB-P0-004, MB-P0-005).

Upgrades from single-backend PlanRoute to a true multi-region physical plan with
explicit backend assignments, transfer edges, and execution contracts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# R21-PLAN-HASH-STRENGTHEN: plan-hash payload version.  Bump when the set of
# hashed physical-plan semantics changes so stale hashes cannot be confused
# with hashes from a different binding contract.
_PLAN_HASH_PAYLOAD_VERSION = 1


class PhysicalBackend(str, Enum):
    """Physical execution backend identity."""

    PANDAS_NUMPY = "pandas_numpy"
    POLARS_PANEL = "polars_panel"
    POLARS_LONG = "polars_long"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"


class Representation(str, Enum):
    """Physical data representation."""

    PANDAS_LONG = "pandas_long"
    PANDAS_WIDE = "pandas_wide"
    NUMPY_PANEL = "numpy_panel"
    POLARS_LONG = "polars_long"
    POLARS_WIDE = "polars_wide"
    POLARS_LAZY_LONG = "polars_lazy_long"
    ARROW_TABLE = "arrow_table"
    DUCKDB_RELATION = "duckdb_relation"
    Q_TABLE = "q_table"
    Q_VECTOR = "q_vector"
    Q_KEYED_TABLE = "q_keyed_table"
    Q_SHARED_HANDLE = "q_shared_handle"


class TransferTransform(str, Enum):
    """Transfer operation type at region boundary.

    R21-TRANSFER-BOUNDARIES: Formal transfer boundaries for cross-backend data movement.
    Each transform has a specific cost model and semantic contract.
    """

    SAME_BACKEND_NATIVE = "same_backend_native"  # No conversion
    DUCKDB_TO_ARROW = "duckdb_to_arrow"  # Section 21: Preferred boundary
    Q_TO_ARROW = "q_to_arrow"  # R21: Q → Arrow for cross-backend consumption
    CLICKHOUSE_TO_ARROW = "clickhouse_to_arrow"  # R21: ClickHouse → Arrow boundary
    ARROW_TO_POLARS = "arrow_to_polars"  # Arrow → Polars DataFrame/LazyFrame
    ARROW_TO_PANDAS = "arrow_to_pandas"  # Arrow → Pandas DataFrame
    POLARS_TO_PANDAS = "polars_to_pandas"  # Polars → Pandas
    PANDAS_TO_POLARS = "pandas_to_polars"  # Pandas → Polars
    POLARS_TO_NUMPY = "polars_to_numpy"  # R21: Polars Series → NumPy ndarray
    NUMPY_TO_POLARS = "numpy_to_polars"  # R21: NumPy ndarray → Polars Series
    WIDE_TO_LONG = "wide_to_long"  # Reshape operation
    LONG_TO_WIDE = "long_to_wide"  # Reshape operation
    SORT = "sort"  # Explicit sort to satisfy downstream property
    REPARTITION = "repartition"  # Repartition by different key
    DTYPE_CAST = "dtype_cast"  # Type conversion


class ExecutionAxis(str, Enum):
    """Execution partitioning axis."""

    TIME_PER_INSTRUMENT = "time_per_instrument"
    CROSS_SECTION_PER_DATE = "cross_section_per_date"
    GROUP_PER_DATE = "group_per_date"
    GLOBAL_PANEL = "global_panel"
    EVENT_STREAM = "event_stream"
    RELATIONAL = "relational"
    RECURSIVE_TIME_PER_INSTRUMENT = "recursive_time_per_instrument"


@dataclass(frozen=True)
class PhysicalProperties:
    """Physical properties of region output (§35, MB-P2-005).

    Tracks ordering, partitioning, grouping, and uniqueness constraints.
    Downstream regions that don't satisfy requirements must insert enforcement ops.
    """
    sorted_by: tuple[str, ...] = ()
    partitioned_by: tuple[str, ...] = ()
    grouped_by: tuple[str, ...] = ()
    unique_key: tuple[str, ...] = ()
    grain: str | None = None

    def satisfies(self, required: PhysicalProperties) -> bool:
        """Check if these properties satisfy the required properties."""
        if required.sorted_by and self.sorted_by[:len(required.sorted_by)] != required.sorted_by:
            return False
        if required.partitioned_by and set(required.partitioned_by) != set(self.partitioned_by):
            return False
        if required.grouped_by and set(required.grouped_by) != set(self.grouped_by):
            return False
        if required.unique_key and set(required.unique_key) != set(self.unique_key):
            return False
        if required.grain and self.grain != required.grain:
            return False
        return True


@dataclass(frozen=True)
class StateContract:
    """State continuation contract for stateful operators (§40).

    MB-P0-012: For RECURSIVE_TIME execution, if requires_checkpoint is False,
    the operation must be sequential-only (cannot parallelize across time).
    """
    requires_checkpoint: bool = False
    checkpoint_seed: str | None = None
    stateful_operators: tuple[str, ...] = ()
    checkpoint_interval: int | None = None
    # MB-P0-012: Track if operation requires sequential execution
    sequential_only: bool = False

    def __post_init__(self):
        """Validate state contract constraints."""
        # MB-P0-012: If stateful but no checkpoint, must be sequential
        if self.stateful_operators and not self.requires_checkpoint:
            if not self.sequential_only:
                # Use object.__setattr__ for frozen dataclass
                object.__setattr__(self, 'sequential_only', True)


@dataclass(frozen=True)
class BackendRegion:
    """A contiguous execution region assigned to a single backend.

    MB-P0-004: Replaces the single-backend PlanRoute with explicit region boundaries.
    MB-P2-005: Includes physical properties (sorted/partitioned/grouped).
    MB-P0-011: CROSS_SECTION_PER_DATE cannot partition by instrument.
    MB-P0-012: RECURSIVE_TIME without checkpoint must be sequential_only.
    """

    region_id: str
    backend: PhysicalBackend
    representation: Representation
    node_ids: tuple[str, ...]
    execution_axis: ExecutionAxis
    estimated_rows: int
    estimated_compute_ms: float
    estimated_memory_bytes: int
    # Semantic contracts preserved across region boundary
    source_snapshot: str = ""
    universe_contract: str = ""
    grain: str = ""
    available_at: str = ""
    pit_safe: bool = True
    # MB-P2-005: Physical properties as first-class attributes
    required_properties: PhysicalProperties | None = None
    output_properties: PhysicalProperties | None = None
    state_contract: StateContract | None = None
    # MB-P1-016, MB-P1-017: Streaming and direct sink capabilities
    streaming_capable: bool = False
    supports_direct_sink: bool = False
    # R21-P023: Four-backend plan identity and resource tracking
    implementation_id: str = ""  # PhysicalImplementationID ("pi:v1:...") binding
    # R21-NODE-IMPL: Per-node implementation mapping for heterogeneous regions
    node_implementations: dict[str, str] = field(default_factory=dict)
    # node_id → PhysicalImplementationID for fine-grained implementation identity
    input_bytes: int = 0  # estimated input data volume
    output_bytes: int = 0  # estimated output data volume
    liveness: str = "eager"  # "eager" / "lazy" / "stream"
    parameter_domain_identity: str = ""  # parameter-domain hash identity

    def __post_init__(self):
        """Validate region constraints and compute manifest hash (MB-P0-011, MB-P0-012, R21-NODE-IMPL)."""
        # MB-P0-011: CROSS_SECTION_PER_DATE cannot partition by instrument
        if self.execution_axis == ExecutionAxis.CROSS_SECTION_PER_DATE:
            if self.required_properties and "instrument" in self.required_properties.partitioned_by:
                raise ValueError(
                    "CROSS_SECTION_PER_DATE cannot be partitioned by instrument: "
                    "cross-sectional operations require all instruments in each date partition"
                )

        # MB-P0-012: RECURSIVE_TIME without checkpoint must be sequential_only
        if self.execution_axis == ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT:
            if self.state_contract:
                if not self.state_contract.requires_checkpoint and not self.state_contract.sequential_only:
                    raise ValueError(
                        "RECURSIVE_TIME without checkpoint must be sequential_only: "
                        "cannot parallelize stateful operations without checkpointing"
                    )

        # R21-NODE-IMPL: Compute manifest hash for node implementations
        if self.node_implementations and not self.implementation_id:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'implementation_id', self._compute_manifest_hash_impl())
        elif self.node_implementations and self.implementation_id:
            # Verify consistency if both are provided
            expected = self._compute_manifest_hash_impl()
            if self.implementation_id != expected:
                raise ValueError(
                    f"implementation_id ({self.implementation_id}) does not match "
                    f"computed manifest hash ({expected}) for node_implementations"
                )

    def _compute_manifest_hash_impl(self) -> str:
        """Compute SHA256 manifest hash from node_implementations dict.

        R21-NODE-IMPL: Deterministic hash of sorted (node_id, pi) pairs.
        Returns "rimh:v1:<sha256>" format for RegionImplementationManifestHash.
        """
        if not self.node_implementations:
            return ""
        sorted_items = sorted(self.node_implementations.items())
        payload = json.dumps(dict(sorted_items), sort_keys=True, separators=(",", ":"))
        return "rimh:v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_region_implementation_manifest_hash(node_implementations: dict[str, str]) -> str:
        """Compute RegionImplementationManifestHash from node_implementations.

        R21-NODE-IMPL: SHA256(sorted node_implementations items) for stable
        identity of heterogeneous regions where different nodes may use
        different PhysicalImplementationIDs.
        """
        if not node_implementations:
            return ""
        sorted_items = sorted(node_implementations.items())
        payload = json.dumps(dict(sorted_items), sort_keys=True, separators=(",", ":"))
        return "rimh:v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "backend": self.backend.value if isinstance(self.backend, Enum) else str(self.backend),
            "representation": self.representation.value if isinstance(self.representation, Enum) else str(self.representation),
            "node_count": len(self.node_ids),
            "execution_axis": self.execution_axis.value if isinstance(self.execution_axis, Enum) else str(self.execution_axis),
            "estimated_compute_ms": round(self.estimated_compute_ms, 3),
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "streaming_capable": self.streaming_capable,
            "supports_direct_sink": self.supports_direct_sink,
            # R21-P023: identity/resource surface
            "implementation_id": self.implementation_id,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "liveness": self.liveness,
            "parameter_domain_identity": self.parameter_domain_identity,
            # R21-NODE-IMPL: per-node implementation mapping
            "node_implementations": self.node_implementations,
            "region_implementation_manifest_hash": self._compute_manifest_hash_impl(),
        }


@dataclass(frozen=True)
class TransferEdge:
    """Transfer between two backend regions.

    MB-P0-004: Explicit representation of cross-backend data movement.
    MB-P1-021: Transfer cost must include sort/repartition/reshape/dtype costs.
    MB-P2-002: Typed contract for all transfer edges.
    Section 22: SQL ordering cannot be assumed — downstream requiring sorted data
    must check producer properties and set requires_sort=True if needed.
    R21-TRANSFER-BOUNDARIES: Uses TransferTransform enum for formal transform type.
    """

    edge_id: str
    producer_region: str
    consumer_region: str
    source_backend: PhysicalBackend
    target_backend: PhysicalBackend
    source_representation: Representation
    target_representation: Representation
    transform: TransferTransform  # R21-TRANSFER-BOUNDARIES: formal transform type
    estimated_rows: int
    estimated_bytes: int
    estimated_transfer_ms: float
    # Required transformations (Section 22: must be explicit and costed)
    requires_sort: bool = False  # Section 22: SQL results may be unordered
    requires_repartition: bool = False
    requires_reshape: bool = False  # wide <-> long
    requires_dtype_cast: bool = False
    # Section 22: Producer's output properties (for downstream to check)
    producer_sorted_by: tuple[str, ...] = ()  # What ordering producer guarantees
    producer_guarantees_order: bool = False  # Whether producer promises stable order
    # Semantic contracts must be preserved (MB-P0-013)
    preserves_pit: bool = True
    preserves_universe: bool = True
    preserves_grain: bool = True
    source_snapshot_id: str | None = None
    # R21-P023: per-representation materialization cost decomposition
    materialization_cost_arrow_ms: float = 0.0  # Arrow IPC/conversion cost
    materialization_cost_numpy_ms: float = 0.0  # NumPy buffer materialization cost
    materialization_cost_polars_ms: float = 0.0  # Polars DataFrame/LazyFrame cost
    materialization_cost_sql_ms: float = 0.0  # SQL result materialization cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "producer_region": self.producer_region,
            "consumer_region": self.consumer_region,
            "source_representation": self.source_representation.value if isinstance(self.source_representation, Enum) else str(self.source_representation),
            "target_representation": self.target_representation.value if isinstance(self.target_representation, Enum) else str(self.target_representation),
            "transform": self.transform.value if isinstance(self.transform, Enum) else str(self.transform),
            "estimated_bytes": self.estimated_bytes,
            "requires_sort": self.requires_sort,
            "requires_repartition": self.requires_repartition,
            "requires_reshape": self.requires_reshape,
            "estimated_transfer_ms": round(self.estimated_transfer_ms, 3),
            # R21-P023: materialization cost decomposition
            "materialization_cost_arrow_ms": round(self.materialization_cost_arrow_ms, 3),
            "materialization_cost_numpy_ms": round(self.materialization_cost_numpy_ms, 3),
            "materialization_cost_polars_ms": round(self.materialization_cost_polars_ms, 3),
            "materialization_cost_sql_ms": round(self.materialization_cost_sql_ms, 3),
        }


@dataclass(frozen=True)
class PhysicalRegionPlan:
    """Complete physical execution plan with backend regions and transfer edges.

    MB-P0-005: Executable physical DAG that the planner produces and executor
    must follow exactly, with no runtime re-routing.
    MB-P2-002: Full typed contract for the complete plan.
    """

    plan_id: str
    regions: tuple[BackendRegion, ...]
    edges: tuple[TransferEdge, ...]
    topological_order: tuple[str, ...]  # Region IDs in execution order
    root_region_ids: tuple[str, ...]
    # Cost estimates (MB-P1-007: decomposed by component)
    total_compute_ms: float
    total_transfer_ms: float
    total_sort_ms: float = 0.0
    total_reshape_ms: float = 0.0
    total_ttdc_ms: float = 0.0
    peak_memory_bytes: int = 0
    # Plan metadata
    plan_hash: str = ""
    logical_node_count: int = 0
    backend_switch_count: int = 0
    native_fraction: float = 0.0  # 0.0-1.0
    # Evidence and certificates
    routing_basis: str = "estimated"  # "measured" | "estimated"
    certificate: Any | None = None

    def __post_init__(self) -> None:
        """Validate the physical DAG before it reaches an executor.

        The plan is the sole backend authority: an edge may only carry the
        representation actually produced by its producer into the residency
        expected by its consumer.  Previously these fields were merely
        descriptive, so a malformed plan could silently materialize or switch
        backend at runtime.
        """
        regions_by_id = {region.region_id: region for region in self.regions}
        if len(regions_by_id) != len(self.regions):
            raise ValueError("PhysicalRegionPlan contains duplicate region IDs")

        if self.regions and set(self.topological_order) != set(regions_by_id):
            raise ValueError("topological_order must contain every region exactly once")
        if len(set(self.topological_order)) != len(self.topological_order):
            raise ValueError("topological_order contains duplicate region IDs")
        if self.regions and not set(self.root_region_ids).issubset(regions_by_id):
            raise ValueError("root_region_ids contains an unknown region")

        order_index = {region_id: index for index, region_id in enumerate(self.topological_order)}
        edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.edge_id in edge_ids:
                raise ValueError(f"PhysicalRegionPlan contains duplicate edge ID: {edge.edge_id}")
            edge_ids.add(edge.edge_id)
            producer = regions_by_id.get(edge.producer_region)
            consumer = regions_by_id.get(edge.consumer_region)
            if producer is None or consumer is None:
                raise ValueError(f"TransferEdge {edge.edge_id} references an unknown region")
            if producer.region_id == consumer.region_id:
                raise ValueError(f"TransferEdge {edge.edge_id} cannot be self-referential")
            if (edge.source_backend, edge.source_representation) != (
                producer.backend, producer.representation
            ):
                # A transfer may explicitly materialize a boundary form (for
                # example DuckDB Relation -> Arrow), but it must remain on the
                # producer backend.  Backend changes are represented by the
                # target side of the edge, never inferred by the executor.
                allowed_boundary = {
                    (PhysicalBackend.DUCKDB_SQL, Representation.ARROW_TABLE),
                }
                if (edge.source_backend, edge.source_representation) not in allowed_boundary:
                    raise ValueError(
                        f"TransferEdge {edge.edge_id} source residency does not match producer {producer.region_id}"
                    )
            if edge.target_backend != consumer.backend or edge.target_representation != consumer.representation:
                raise ValueError(
                    f"TransferEdge {edge.edge_id} target residency does not match consumer {consumer.region_id}"
                )
            if self.regions and order_index[edge.producer_region] >= order_index[edge.consumer_region]:
                raise ValueError(
                    f"TransferEdge {edge.edge_id} violates topological_order"
                )

        expected_switches = sum(
            edge.source_backend != edge.target_backend for edge in self.edges
        )
        if self.backend_switch_count != expected_switches:
            raise ValueError(
                "backend_switch_count must equal backend-changing transfer edges"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "region_count": len(self.regions),
            "edge_count": len(self.edges),
            "backend_switch_count": self.backend_switch_count,
            "peak_memory_bytes": self.peak_memory_bytes,
            "total_compute_ms": round(self.total_compute_ms, 3),
            "total_transfer_ms": round(self.total_transfer_ms, 3),
            "total_sort_ms": round(self.total_sort_ms, 3),
            "total_reshape_ms": round(self.total_reshape_ms, 3),
            "total_ttdc_ms": round(self.total_ttdc_ms, 3),
            "native_fraction": round(self.native_fraction, 4),
            "regions": [r.to_dict() for r in self.regions],
            "edges": [e.to_dict() for e in self.edges],
        }

    @staticmethod
    def compute_plan_hash(
        regions: tuple[BackendRegion, ...],
        edges: tuple[TransferEdge, ...],
        logical_hash: str,
    ) -> str:
        """Compute stable hash for the physical plan (§68, MB-P2-014).

        R21-PLAN-HASH-STRENGTHEN: the hash must bind ALL physical plan
        semantics — two plans that would execute even one byte differently
        must not share a plan hash.  Previously the payload only covered
        region_id/backend/node_count and edge producer/consumer/source_repr/
        target_repr, so plans differing in node identity, representation,
        execution axis, implementation identity, semantic contracts,
        parameter domains, or transfer transforms collided silently.

        Binds (per region): region_id, backend, node_ids, representation,
        execution_axis, implementation_id (PhysicalImplementationID),
        liveness, parameter_domain_identity, source identity
        (source_snapshot/universe_contract), semantic contract
        (grain/available_at/pit_safe), required/output physical properties
        (sorted_by/partitioned_by/grouped_by/unique_key/grain), state
        contract (checkpoint/seed/stateful operators/sequential).

        Binds (per edge): edge_id, producer/consumer, source/target backend,
        source/target representation, transfer transforms
        (requires_sort/repartition/reshape/dtype_cast + producer ordering
        guarantees), and preserved semantic contracts
        (pit/universe/grain/source_snapshot).

        Binds (per plan): the logical DAG hash.

        Not included: cost/size estimates (estimated_rows, estimated_*_ms,
        *_bytes) — these are predictions, not plan identity (§68).
        """
        payload = {
            "version": _PLAN_HASH_PAYLOAD_VERSION,
            "logical_hash": logical_hash,
            "regions": [
                PhysicalRegionPlan._region_hash_payload(r) for r in regions
            ],
            "edges": [
                PhysicalRegionPlan._edge_hash_payload(e) for e in edges
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _region_hash_payload(r: BackendRegion) -> dict[str, Any]:
        """Full semantic identity of one BackendRegion for plan hashing."""

        def _enum(v: Any) -> str:
            return v.value if isinstance(v, Enum) else str(v)

        def _props(p: PhysicalProperties | None) -> dict[str, Any]:
            if p is None:
                return {"__absent__": True}
            return {
                "sorted_by": list(p.sorted_by),
                "partitioned_by": list(p.partitioned_by),
                "grouped_by": list(p.grouped_by),
                "unique_key": list(p.unique_key),
                "grain": p.grain,
            }

        def _state(s: StateContract | None) -> dict[str, Any]:
            if s is None:
                return {"__absent__": True}
            return {
                "requires_checkpoint": s.requires_checkpoint,
                "checkpoint_seed": s.checkpoint_seed,
                "stateful_operators": list(s.stateful_operators),
                "checkpoint_interval": s.checkpoint_interval,
                "sequential_only": s.sequential_only,
            }

        return {
            "region_id": r.region_id,
            "backend": _enum(r.backend),
            "node_ids": list(r.node_ids),
            "representation": _enum(r.representation),
            "execution_axis": _enum(r.execution_axis),
            # R21-P023: PhysicalImplementationID binding ("pi:v1:...")
            "implementation_id": r.implementation_id,
            # R21-NODE-IMPL: per-node implementation mapping for plan identity
            "node_implementations": r.node_implementations,
            "region_implementation_manifest_hash": r._compute_manifest_hash_impl(),
            # eager / lazy / stream
            "liveness": r.liveness,
            # parameter-domain hash identity
            "parameter_domain_identity": r.parameter_domain_identity,
            # source identity
            "source_snapshot": r.source_snapshot,
            "universe_contract": r.universe_contract,
            # semantic contract
            "grain": r.grain,
            "available_at": r.available_at,
            "pit_safe": r.pit_safe,
            "required_properties": _props(r.required_properties),
            "output_properties": _props(r.output_properties),
            "state_contract": _state(r.state_contract),
        }

    @staticmethod
    def _edge_hash_payload(e: TransferEdge) -> dict[str, Any]:
        """Full semantic identity of one TransferEdge for plan hashing."""

        def _enum(v: Any) -> str:
            return v.value if isinstance(v, Enum) else str(v)

        return {
            "edge_id": e.edge_id,
            "producer": e.producer_region,
            "consumer": e.consumer_region,
            "source_backend": _enum(e.source_backend),
            "target_backend": _enum(e.target_backend),
            "source_repr": _enum(e.source_representation),
            "target_repr": _enum(e.target_representation),
            "transform": _enum(e.transform),  # R21-TRANSFER-BOUNDARIES: bind transform type
            # transfer transforms (§22: explicit and costed — hence hashed)
            "requires_sort": e.requires_sort,
            "requires_repartition": e.requires_repartition,
            "requires_reshape": e.requires_reshape,
            "requires_dtype_cast": e.requires_dtype_cast,
            "producer_sorted_by": list(e.producer_sorted_by),
            "producer_guarantees_order": e.producer_guarantees_order,
            # semantic contracts preserved across the boundary (MB-P0-013)
            "preserves_pit": e.preserves_pit,
            "preserves_universe": e.preserves_universe,
            "preserves_grain": e.preserves_grain,
            "source_snapshot_id": e.source_snapshot_id,
        }


def normalize_backend_name(backend: str) -> PhysicalBackend:
    """Normalize backend string to PhysicalBackend enum.

    MB-P0-002: Ensure we use specific backends (duckdb_sql) not generic (sql).
    """
    normalized = backend.lower().strip()

    if normalized in {"pandas", "pandas_numpy"}:
        return PhysicalBackend.PANDAS_NUMPY
    if normalized in {"polars", "polars_panel"}:
        return PhysicalBackend.POLARS_PANEL
    if normalized in {"polars_long"}:
        return PhysicalBackend.POLARS_LONG
    if normalized in {"duckdb", "duckdb_sql", "sql"}:
        # MB-P0-002: Never return generic "sql"
        return PhysicalBackend.DUCKDB_SQL
    if normalized in {"clickhouse", "clickhouse_sql"}:
        return PhysicalBackend.CLICKHOUSE_SQL
    if normalized in {"q", "q_kdb", "kdb"}:
        return PhysicalBackend.Q_KDB

    raise ValueError(f"Unknown backend: {backend}")


def infer_representation(backend: PhysicalBackend, wide: bool = False) -> Representation:
    """Infer representation from backend and data layout.

    MB-P1-018 (Section 21): DuckDB Region boundary should prefer Arrow/Relation,
    not unnecessarily convert to Pandas via `.df()`. Keep DuckDB Relation / SQL
    within regions; cross-region boundary should prefer DuckDB → Arrow.
    """
    if backend == PhysicalBackend.PANDAS_NUMPY:
        return Representation.PANDAS_WIDE if wide else Representation.PANDAS_LONG
    if backend in {PhysicalBackend.POLARS_PANEL, PhysicalBackend.POLARS_LONG}:
        if backend == PhysicalBackend.POLARS_LONG:
            return Representation.POLARS_LAZY_LONG
        return Representation.POLARS_WIDE if wide else Representation.POLARS_LONG
    if backend == PhysicalBackend.DUCKDB_SQL:
        # MB-P1-018: Prefer DUCKDB_RELATION (native) or ARROW_TABLE (boundary)
        # over PANDAS_LONG to avoid unnecessary conversion
        return Representation.DUCKDB_RELATION
    if backend == PhysicalBackend.CLICKHOUSE_SQL:
        return Representation.ARROW_TABLE  # ClickHouse → Arrow boundary
    if backend == PhysicalBackend.Q_KDB:
        return Representation.Q_TABLE
    return Representation.PANDAS_LONG


def backend_supports_direct_sink(backend: PhysicalBackend, representation: Representation) -> bool:
    """Check if backend supports direct sink to Parquet (MB-P1-025, §50).

    Avoids native→Pandas→Parquet roundtrip for final output.
    """
    if backend in (PhysicalBackend.POLARS_PANEL, PhysicalBackend.POLARS_LONG):
        return True
    if backend == PhysicalBackend.DUCKDB_SQL and representation == Representation.DUCKDB_RELATION:
        return True
    # PyArrow not yet in PhysicalBackend enum but future-proofing
    return False


def supports_streaming(backend: PhysicalBackend, operators: tuple[str, ...]) -> bool:
    """Check if backend supports streaming execution for given operators.

    MB-P1-016: Low-memory servers must use streaming-first approach.
    Section 19: Not all Polars operators are streaming-safe.
    """
    if backend == PhysicalBackend.DUCKDB_SQL:
        # DuckDB has good streaming support for most SQL operations
        return True
    if backend in (PhysicalBackend.POLARS_PANEL, PhysicalBackend.POLARS_LONG):
        # Check if operators require global sort/full group
        # This is a simplified check - real implementation should query operator registry
        global_ops = {"rank", "quantile", "sort"}
        return not any(op in global_ops for op in operators)
    return False


def infer_transfer_transform(
    source_repr: Representation,
    target_repr: Representation,
) -> TransferTransform:
    """Infer transfer transform from source and target representations.

    R21-TRANSFER-BOUNDARIES: Maps representation pairs to formal TransferTransform values.
    """
    if source_repr == target_repr:
        return TransferTransform.SAME_BACKEND_NATIVE

    # DuckDB → Arrow (Section 21: preferred boundary)
    if source_repr == Representation.DUCKDB_RELATION and target_repr == Representation.ARROW_TABLE:
        return TransferTransform.DUCKDB_TO_ARROW

    # Q → Arrow (R21: Q → Arrow for cross-backend consumption)
    if source_repr in {Representation.Q_TABLE, Representation.Q_VECTOR, Representation.Q_KEYED_TABLE} and target_repr == Representation.ARROW_TABLE:
        return TransferTransform.Q_TO_ARROW

    # ClickHouse → Arrow (R21: ClickHouse → Arrow boundary)
    # Note: ClickHouse typically produces Arrow directly, but formalizing the boundary
    if source_repr == Representation.ARROW_TABLE and target_repr == Representation.ARROW_TABLE:
        # ClickHouse → Arrow is a no-op if already Arrow, but we keep the transform
        # for explicit tracking
        pass  # Fall through to other checks

    # Arrow → Polars
    if source_repr == Representation.ARROW_TABLE and target_repr in {
        Representation.POLARS_LONG, Representation.POLARS_WIDE, Representation.POLARS_LAZY_LONG
    }:
        return TransferTransform.ARROW_TO_POLARS

    # Arrow → Pandas
    if source_repr == Representation.ARROW_TABLE and target_repr in {
        Representation.PANDAS_LONG, Representation.PANDAS_WIDE
    }:
        return TransferTransform.ARROW_TO_PANDAS

    # Polars → Pandas
    if source_repr in {Representation.POLARS_LONG, Representation.POLARS_WIDE, Representation.POLARS_LAZY_LONG} and target_repr in {
        Representation.PANDAS_LONG, Representation.PANDAS_WIDE
    }:
        return TransferTransform.POLARS_TO_PANDAS

    # Pandas → Polars
    if source_repr in {Representation.PANDAS_LONG, Representation.PANDAS_WIDE} and target_repr in {
        Representation.POLARS_LONG, Representation.POLARS_WIDE, Representation.POLARS_LAZY_LONG
    }:
        return TransferTransform.PANDAS_TO_POLARS

    # Polars → NumPy (R21: Polars Series → NumPy ndarray)
    if source_repr in {Representation.POLARS_LONG, Representation.POLARS_WIDE, Representation.POLARS_LAZY_LONG} and target_repr == Representation.NUMPY_PANEL:
        return TransferTransform.POLARS_TO_NUMPY

    # NumPy → Polars (R21: NumPy ndarray → Polars Series)
    if source_repr == Representation.NUMPY_PANEL and target_repr in {
        Representation.POLARS_LONG, Representation.POLARS_WIDE, Representation.POLARS_LAZY_LONG
    }:
        return TransferTransform.NUMPY_TO_POLARS

    # Wide ↔ Long reshapes
    if "wide" in source_repr.value and "long" in target_repr.value:
        return TransferTransform.WIDE_TO_LONG
    if "long" in source_repr.value and "wide" in target_repr.value:
        return TransferTransform.LONG_TO_WIDE

    # Default to PANDAS_TO_POLARS for unknown combinations
    return TransferTransform.PANDAS_TO_POLARS


def estimate_transfer_cost_ms(
    source_repr: Representation,
    target_repr: Representation,
    estimated_bytes: int,
    requires_sort: bool = False,
    requires_repartition: bool = False,
    requires_reshape: bool = False,
) -> float:
    """Estimate transfer edge cost (MB-P1-021, §32).

    Includes representation conversion, sort, repartition, and reshape costs.
    R21-TRANSFER-BOUNDARIES: Now supports all formal transfer transforms.
    """
    # Base conversion cost by edge type (R21-TRANSFER-BOUNDARIES: expanded)
    conversion_costs = {
        ("duckdb_relation", "arrow_table"): 3.0,
        ("q_table", "arrow_table"): 4.0,  # R21: Q → Arrow
        ("q_vector", "arrow_table"): 4.0,  # R21: Q → Arrow
        ("arrow_table", "polars_long"): 2.0,
        ("arrow_table", "polars_lazy_long"): 2.0,
        ("arrow_table", "pandas_long"): 2.0,
        ("polars_long", "pandas_long"): 2.0,
        ("polars_lazy_long", "pandas_long"): 2.0,
        ("pandas_wide", "pandas_long"): 2.0,  # reshape
        ("pandas_long", "pandas_wide"): 2.0,  # reshape
        ("polars_long", "numpy_panel"): 1.5,  # R21: Polars → NumPy (zero-copy)
        ("polars_wide", "numpy_panel"): 1.5,  # R21: Polars → NumPy (zero-copy)
        ("polars_lazy_long", "numpy_panel"): 2.0,  # R21: Polars Lazy → NumPy
        ("numpy_panel", "polars_long"): 2.0,  # R21: NumPy → Polars
        ("numpy_panel", "polars_wide"): 2.0,  # R21: NumPy → Polars
        ("numpy_panel", "polars_lazy_long"): 2.5,  # R21: NumPy → Polars Lazy
    }

    src = source_repr.value if isinstance(source_repr, Enum) else str(source_repr)
    tgt = target_repr.value if isinstance(target_repr, Enum) else str(target_repr)

    base_ms = conversion_costs.get((src, tgt), 3.0)

    # Add bytes-based cost
    if estimated_bytes > 0:
        mb = estimated_bytes / (1024 * 1024)
        base_ms += mb / 200.0 * 1000.0  # 200 MB/s conversion throughput

    # Add transformation costs
    if requires_sort:
        # O(n log n) sort cost
        base_ms += max(1.5, estimated_bytes / (1024 * 1024) / 100.0 * 1000.0)

    if requires_repartition:
        base_ms += max(1.0, estimated_bytes / (1024 * 1024) / 300.0 * 1000.0)

    if requires_reshape:
        base_ms += max(2.0, estimated_bytes / (1024 * 1024) / 150.0 * 1000.0)

    return base_ms
