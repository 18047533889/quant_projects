# -*- coding: utf-8 -*-
"""MB-P1-009: Batch-global backend optimizer.

Replaces per-root backend selection with true batch-global optimization that:
- Considers shared DAG nodes only once
- Accounts for cross-root shared benefit from source scans and CSE
- Uses dynamic programming with parent transfer affinity
- Produces PhysicalRegionPlan with explicit regions and transfer edges

The optimizer operates on a batch of factors (roots) with shared subexpressions,
computing optimal backend assignment that minimizes total batch cost including:
  - Compute cost per node
  - Transfer cost at backend boundaries
  - Source scan sharing across roots
  - CSE materialization and reuse
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalProperties,
    PhysicalRegionPlan,
    Representation,
    TransferEdge,
    TransferTransform,
    infer_representation,
)
from factor_engine.planner.logical_plan import PlanNode


# R45: Numba gets its OWN PhysicalImplementationID, distinct from pandas_numpy's
# production capability.  The Numba candidate must not borrow pandas_numpy's
# production certification — it is a separate physical implementation with its
# own kernel identity (NumbaKernelImplementationID, "nki:v1:...") and its own
# accelerator (NUMBA_CPU).  When a certified Numba kernel is registered for the
# operator, we derive a deterministic PI-ID from the kernel's implementation id;
# otherwise the candidate is not production-certified.
_NUMBA_PI_ID_PREFIX = "pi:v3:numba:"


def _numba_pi_id(op: str, kernel_impl_id: str) -> str:
    """Deterministic PhysicalImplementationID for the Numba candidate.

    Binds the operator canonical and the certified Numba kernel implementation
    id so that a kernel swap changes the PI-ID (and thus the choice identity).
    """
    digest = hashlib.sha256(
        json.dumps(
            {"op": op, "kernel_impl_id": kernel_impl_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"{_NUMBA_PI_ID_PREFIX}{digest}"


@dataclass(frozen=True)
class OptimizationOpportunity:
    kind: str
    savings_bytes: int
    savings_work: float
    affected_roots: list[str]
    shared_node_id: str | None = None
    confidence: float = 0.8


@dataclass
class GlobalOptimizationResult:
    opportunities: list[OptimizationOpportunity] = field(default_factory=list)
    applied_count: int = 0
    total_savings_bytes: int = 0
    total_savings_work: float = 0.0
    rewritten_dag: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeBackendChoice:
    """A single node's backend choice with cost breakdown.

    R45: carries the EXACT physical implementation binding, not just a backend
    name.  ``physical_implementation_id`` identifies the precise selectable
    implementation; ``bound_parameter_identity`` binds the real parameter
    domain (window/span/...) used to evaluate the kernel signature;
    ``implementation_closure_hash`` binds the full semantic closure of the
    implementation; ``numeric_policy_identity`` binds the numerical semantic
    policy; ``kernel_signature`` records the concrete kernel signature the
    implementation was evaluated against.  These fields are additive — existing
    positional/keyword construction keeps working.
    """

    node_id: str
    backend: PhysicalBackend
    compute_cost_ms: float
    transfer_from_children_ms: float
    total_cost_ms: float
    representation: Representation
    execution_kind: ExecutionKind
    production_certified: bool
    # R45 exact-implementation binding (additive; defaults preserve ABI).
    physical_implementation_id: str = ""
    bound_parameter_identity: str = ""
    implementation_closure_hash: str = ""
    numeric_policy_identity: str = ""
    kernel_signature: str = ""


@dataclass(frozen=True)
class SharedNodeBenefit:
    """MB-P1-010: Real derivation of shared benefit from DAG structure.

    Replaces fixed 10%/5ms heuristic with actual avoided work calculation.
    """

    node_id: str
    consumer_count: int
    compute_cost_ms: float
    avoided_recompute_ms: float  # (consumer_count - 1) * compute_cost
    scan_bytes: int
    avoided_scan_bytes: int  # If source node
    benefit_ms: float  # Total benefit from sharing


def estimate_shared_benefits(
    shared_nodes: dict[str, PlanNode],
    consumer_counts: dict[str, int],
    node_costs: dict[str, float],
    source_nodes: set[str],
    scan_cost_per_mb: float = 0.01,
) -> tuple[dict[str, SharedNodeBenefit], float]:
    """MB-P1-010: Derive real shared benefit from DAG topology.

    Args:
        shared_nodes: Map of node_id -> PlanNode for shared expressions
        consumer_counts: Map of node_id -> number of consumers
        node_costs: Map of node_id -> compute cost in ms
        source_nodes: Set of node_ids that are source scans
        scan_cost_per_mb: Cost coefficient for scanning data

    Returns:
        (benefits_by_node, total_benefit_ms)
    """
    benefits: dict[str, SharedNodeBenefit] = {}
    total_benefit = 0.0

    for node_id, node in shared_nodes.items():
        consumers = consumer_counts.get(node_id, 0)
        if consumers <= 1:
            continue

        compute_cost = node_costs.get(node_id, 0.0)
        # Benefit: we compute once but N consumers use it
        avoided_recompute = compute_cost * (consumers - 1)

        # Source scan benefit
        scan_bytes = 0
        avoided_scan_bytes = 0
        if node_id in source_nodes:
            # Estimate scan bytes from node attributes
            attrs = getattr(node, "attrs", None) or {}
            scan_bytes = int(attrs.get("estimated_bytes", 0) or 0)
            if scan_bytes < 0:
                scan_bytes = 0
            avoided_scan_bytes = scan_bytes * (consumers - 1)

        scan_benefit = avoided_scan_bytes / 1_000_000.0 * scan_cost_per_mb
        total_node_benefit = avoided_recompute + scan_benefit

        benefit = SharedNodeBenefit(
            node_id=node_id,
            consumer_count=consumers,
            compute_cost_ms=compute_cost,
            avoided_recompute_ms=avoided_recompute,
            scan_bytes=scan_bytes,
            avoided_scan_bytes=avoided_scan_bytes,
            benefit_ms=total_node_benefit,
        )
        benefits[node_id] = benefit
        total_benefit += total_node_benefit

    return benefits, total_benefit


@dataclass(frozen=True)
class BatchOptimizationResult:
    """MB-P1-009: Result of batch-global optimization."""

    physical_plan: PhysicalRegionPlan
    per_node_choices: dict[str, NodeBackendChoice]
    shared_benefits: dict[str, SharedNodeBenefit]
    total_shared_benefit_ms: float
    optimization_basis: str  # "dp_global_exact" | "dp_global_approximate"
    production_ready: bool = False
    readiness_reason: str = ""
    optimization_elapsed_ms: float = 0.0
    candidate_plan_count: int = 0
    selected_incumbent: str = ""


class PhysicalBatchGlobalOptimizer:
    """MB-P1-009: Batch-global backend optimizer with shared DAG awareness.

    Uses dynamic programming to find optimal backend assignment considering:
    - Shared nodes computed once
    - Transfer costs at backend boundaries
    - Parent backend affinity (child->parent transfer costs)
    - Source scan sharing

    R21-ROUTING-AUTHORITY: this optimizer is the SOLE production routing
    authority.  ``BackendRouter``, ``get_best_backend`` and
    ``IntelligentBackendSelector`` are capability/cost CANDIDATE PROVIDERS
    only — they may propose a backend but never finalize the production route.
    The production path (``runtime.batch_service``) routes through
    ``optimize_batch_global`` and executes the admitted ``PhysicalRegionPlan``
    without per-operator rerouting.
    """

    def __init__(
        self,
        *,
        transfer_cost_per_mb: float = 0.05,
        delegate_penalty_ms: float = 3.0,
        scan_cost_per_mb: float = 0.01,
        exact_search_max_ambiguous_nodes: int = 12,
        approximate_max_passes: int = 4,
        max_optimization_ms: float = 250.0,
        max_candidate_plans: int = 128,
        forced_backend: str | None = None,
    ) -> None:
        if exact_search_max_ambiguous_nodes < 0:
            raise ValueError("exact_search_max_ambiguous_nodes must be non-negative")
        if approximate_max_passes <= 0:
            raise ValueError("approximate_max_passes must be positive")
        if max_optimization_ms <= 0:
            raise ValueError("max_optimization_ms must be positive")
        if max_candidate_plans <= 0:
            raise ValueError("max_candidate_plans must be positive")
        self.transfer_cost_per_mb = transfer_cost_per_mb
        self.delegate_penalty_ms = delegate_penalty_ms
        self.scan_cost_per_mb = scan_cost_per_mb
        self.exact_search_max_ambiguous_nodes = exact_search_max_ambiguous_nodes
        self.approximate_max_passes = approximate_max_passes
        self.max_optimization_ms = float(max_optimization_ms)
        self.max_candidate_plans = int(max_candidate_plans)
        self.forced_backend = forced_backend

        # Memo for DP: (node_id, output_backend) -> (cost, choice)
        self.memo: dict[tuple[str, PhysicalBackend], tuple[float, NodeBackendChoice]] = {}

    def optimize_batch(
        self,
        roots: dict[str, PlanNode],
        shared_nodes: dict[str, PlanNode],
        node_graph: dict[str, list[str]],  # node_id -> [child_ids]
        ctx: Any,
    ) -> BatchOptimizationResult:
        """MB-P1-009: Optimize backend assignment for entire batch.

        Args:
            roots: Map of factor_id -> root PlanNode
            shared_nodes: Map of sid -> shared PlanNode
            node_graph: DAG structure as adjacency list
            ctx: Execution context

        Returns:
            BatchOptimizationResult with physical plan and choices
        """
        optimization_started = time.monotonic()
        from factor_engine.backend.plan_cost_router import plan_occurrences
        from factor_engine.backend.operator_cost import estimate_backend_cost

        # Discover the complete graph from PlanNode.inputs.  Caller-supplied
        # adjacency maps are compatibility hints, not a complete source of truth.
        all_nodes: dict[str, PlanNode] = {}
        discovered_graph: dict[str, list[str]] = {}
        object_names = {id(node): node_id for node_id, node in shared_nodes.items()}
        object_names.update({id(node): root_id for root_id, node in roots.items()})
        used_ids: set[str] = set()
        visiting: set[str] = set()

        def discover(node: PlanNode, preferred: str | None = None) -> str:
            key = preferred or object_names.get(id(node)) or getattr(node, "node_id", None)
            key = str(key or f"node_{id(node)}")
            if key in used_ids and all_nodes.get(key) is not node:
                key = f"{key}_{id(node)}"
            if key in visiting:
                raise ValueError(f"cyclic plan dependency at node {key!r}")
            if key in all_nodes:
                return key
            visiting.add(key)
            used_ids.add(key)
            all_nodes[key] = node
            children = [discover(child) for child in (getattr(node, "inputs", ()) or ())]
            if getattr(node, "op", "") == "plan_ref":
                sid = str((getattr(node, "attrs", None) or {}).get("sid") or "")
                if not sid or sid not in shared_nodes:
                    raise ValueError(f"dangling plan_ref sid {sid!r}")
                children.append(discover(shared_nodes[sid], sid))
            discovered_graph[key] = children
            visiting.remove(key)
            return key

        for factor_id, root in roots.items():
            discover(root, factor_id)
        for shared_id, shared_node in shared_nodes.items():
            discover(shared_node, shared_id)
        node_graph = discovered_graph

        # Count consumers for shared benefit calculation
        consumer_counts: dict[str, int] = {}
        for children in node_graph.values():
            for child_id in children:
                consumer_counts[child_id] = consumer_counts.get(child_id, 0) + 1

        # Unknown estimates must not drive production routing or readiness.
        rows = self._known_rows(ctx)
        estimated_bytes = self._known_positive_stat(
            ctx, ("byte_count_estimate", "estimated_bytes", "input_bytes")
        )
        estimated_memory = self._known_positive_stat(
            ctx, ("memory_bytes_estimate", "estimated_memory_bytes", "peak_memory_bytes")
        )
        if (estimated_bytes is None or estimated_memory is None) and getattr(
            ctx, "data_source", None
        ) is not None:
            try:
                from factor_engine.planner.data_shape import estimate_shape_from_context

                shape = estimate_shape_from_context(ctx)
                estimated_bytes = estimated_bytes or int(shape.estimated_bytes or 0) or None
                estimated_memory = estimated_memory or estimated_bytes
            except Exception:
                pass
        estimate_rows = rows or 0
        node_costs: dict[str, float] = {}
        for node_id, node in all_nodes.items():
            occurrences = plan_occurrences(node)
            if occurrences:
                occ = occurrences[0]
                # Estimate for pandas as baseline
                cost = estimate_backend_cost(
                    occ.canonical,
                    "pandas_numpy",
                    row_count_estimate=estimate_rows,
                )
                node_costs[node_id] = cost
            else:
                node_costs[node_id] = 0.0

        # Identify source nodes
        source_nodes = {nid for nid, node in all_nodes.items()
                       if getattr(node, "op", "") == "column"}

        # MB-P1-010: Calculate initial shared benefits using Pandas baseline
        shared_benefits, total_benefit = estimate_shared_benefits(
            shared_nodes,
            consumer_counts,
            node_costs,
            source_nodes,
            self.scan_cost_per_mb,
        )

        node_estimates = self._derive_node_estimates(
            all_nodes,
            node_graph,
            roots=tuple(roots),
            global_rows=rows,
            global_bytes=estimated_bytes,
            global_memory=estimated_memory,
        )

        # Solve one assignment problem over the complete discovered DAG.  A
        # shared node is represented by one graph variable, so all roots observe
        # the same persisted assignment.
        self._choices, optimization_basis, candidate_plan_count, selected_incumbent = self._optimize_graph(
            all_nodes, node_graph, estimate_rows, ctx, node_estimates
        )
        per_node_choices = dict(self._choices)

        # MB-P1-010, §44: Recompute shared benefit based on selected physical implementation.
        # The initial estimate used Pandas baseline costs; after physical assignment,
        # we recalculate avoided compute based on each shared node's selected backend cost.
        shared_benefits, total_benefit = self._recompute_shared_benefits_after_assignment(
            shared_nodes=shared_nodes,
            consumer_counts=consumer_counts,
            per_node_choices=per_node_choices,
            source_nodes=source_nodes,
            node_estimates=node_estimates,
            rows=estimate_rows,
            ctx=ctx,
        )

        total_compute = sum(choice.compute_cost_ms for choice in per_node_choices.values())
        total_transfer = sum(choice.transfer_from_children_ms for choice in per_node_choices.values())

        # Build physical region plan
        plan = self._build_physical_plan(
            per_node_choices,
            shared_benefits,
            total_benefit,
            total_compute,
            total_transfer,
            ctx,
            node_graph,
            tuple(roots),
            node_estimates,
        )
        # P2 split-brain closure: a production region must never be routed to a
        # backend the runtime cannot execute.  The optimizer's candidate filter
        # already drops CLICKHOUSE_SQL / Q_KDB (no wired runtime executor), but
        # this fail-closed gate makes it impossible for ANY plan carrying one to
        # reach the runtime — even one built by an older/newer caller path.
        from factor_engine.runtime.engine import assert_all_backends_runtime_capable

        assert_all_backends_runtime_capable(plan)
        production_ready, readiness_reason = self._readiness(
            roots=roots,
            all_nodes=all_nodes,
            choices=per_node_choices,
            plan=plan,
            rows=rows,
            estimated_bytes=estimated_bytes,
            estimated_memory=estimated_memory,
            node_graph=node_graph,
            run_mode=getattr(ctx, "run_mode", None),
        )

        return BatchOptimizationResult(
            physical_plan=plan,
            per_node_choices=per_node_choices,
            shared_benefits=shared_benefits,
            total_shared_benefit_ms=total_benefit,
            optimization_basis=optimization_basis,
            production_ready=production_ready,
            readiness_reason=readiness_reason,
            optimization_elapsed_ms=(time.monotonic() - optimization_started) * 1000.0,
            candidate_plan_count=candidate_plan_count,
            selected_incumbent=selected_incumbent,
        )

    @staticmethod
    def _known_positive_stat(ctx: Any, keys: tuple[str, ...]) -> int | None:
        """Return a positive evidenced estimate from runtime statistics."""
        runtime = dict(getattr(ctx, "runtime_stats", None) or {})
        for key in keys:
            try:
                value = int(runtime.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
        return None

    @classmethod
    def _known_rows(cls, ctx: Any) -> int | None:
        """Return an evidenced row estimate, never a routing fallback."""
        value = cls._known_positive_stat(
            ctx, ("row_count_estimate", "input_row_count", "estimated_rows")
        )
        if value is not None:
            return value
        if getattr(ctx, "data_source", None) is None:
            return None
        try:
            from factor_engine.planner.data_shape import estimate_shape_from_context
            value = int(estimate_shape_from_context(ctx).estimated_rows or 0)
            return value if value > 0 else None
        except Exception:
            return None

    @staticmethod
    def _bound_parameter_identity(node: PlanNode) -> str:
        """Canonical identity of the node's bound parameter domain.

        Uses the real window/span (and any other numeric params) from the node's
        attrs so the Numba candidate is evaluated against the actual kernel
        signature, not a hard-coded window=0.
        """
        attrs = dict(getattr(node, "attrs", None) or {})
        params = dict(getattr(node, "params", None) or {})
        merged: dict[str, Any] = {}
        for key in ("window", "span", "min_periods", "fast", "slow", "signal"):
            if key in attrs:
                merged[key] = attrs[key]
            elif key in params:
                merged[key] = params[key]
        return json.dumps(merged, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _kernel_signature(node: PlanNode) -> str:
        """Concrete kernel signature string for the node's bound parameters."""
        attrs = dict(getattr(node, "attrs", None) or {})
        params = dict(getattr(node, "params", None) or {})
        window = attrs.get("window", params.get("window", attrs.get("span", params.get("span", 0))))
        try:
            window = int(window or 0)
        except (TypeError, ValueError):
            window = 0
        return f"window={window}"

    @staticmethod
    def _bound_window(node: PlanNode) -> int:
        """Real bound window/span from the node's params (R45)."""
        attrs = dict(getattr(node, "attrs", None) or {})
        params = dict(getattr(node, "params", None) or {})
        window = attrs.get("window", params.get("window", attrs.get("span", params.get("span", 0))))
        try:
            return int(window or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _capability_pi_id(capability: Any) -> str:
        """Best-effort PhysicalImplementationID from a BackendCapability.

        The capability record may carry an explicit ``implementation_id``
        (PhysicalImplementationID).  When absent, derive a deterministic
        PI-ID from the capability's canonical/backend/execution_kind so the
        choice still binds an exact implementation identity.
        """
        impl_id = getattr(capability, "implementation_id", None)
        if impl_id is not None:
            value = getattr(impl_id, "value", None)
            if value:
                return str(value)
        canonical = getattr(capability, "canonical", "")
        backend = getattr(capability, "backend", "")
        backend_value = getattr(backend, "value", backend)
        exec_kind = getattr(capability, "execution_kind", "")
        exec_value = getattr(exec_kind, "value", exec_kind)
        digest = hashlib.sha256(
            json.dumps(
                {"canonical": str(canonical), "backend": str(backend_value),
                 "execution_kind": str(exec_value)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"pi:v3:capability:{digest}"

    def _eligible_choices(
        self, node_id: str, node: PlanNode, rows: int, ctx: Any
    ) -> tuple[NodeBackendChoice, ...]:
        """Return explicitly evidenced physical candidates for one logical node."""
        from factor_engine.backend.operator_capability import (
            UnsupportedOperatorBackendError,
            capability_for,
            supports_pandas,
            supports_polars,
            supports_sql,
        )
        from factor_engine.backend.operator_cost import estimate_backend_cost
        from factor_engine.backend.polars_backend_kind import canonical_polars_is_delegate

        op = getattr(node, "op", "")
        if op in {"column", "literal", "plan_ref"}:
            # R45: ``literal`` and ``plan_ref`` are zero-cost reference nodes and
            # stay production-certified.  ``column`` is a PhysicalSourceBinding
            # node: it is ONLY production-certified when it carries a real
            # physical source binding (DataReadIdentity + field semantics + PIT +
            # universe + snapshot).  Without a binding it is NOT certified.
            if op == "column":
                certified = self._column_has_source_binding(node)
                source_binding = dict(getattr(node, "attrs", None) or {}).get(
                    "physical_source_binding"
                )
            else:
                certified = True
            if self._has_source_residency(node):
                backends = (self._source_residency(node),)
            else:
                backends = tuple(
                    (backend, infer_representation(backend))
                    for backend in (
                        PhysicalBackend.PANDAS_NUMPY,
                        PhysicalBackend.POLARS_PANEL,
                    )
                )
            return tuple(
                NodeBackendChoice(
                    node_id=node_id,
                    backend=backend,
                    compute_cost_ms=0.0,
                    transfer_from_children_ms=0.0,
                    total_cost_ms=0.0,
                    representation=representation,
                    execution_kind=ExecutionKind.REFERENCE,
                    production_certified=certified,
                    physical_implementation_id=(
                        f"pi:v3:source_binding:{source_binding.digest}"
                        if op == "column" and certified else ""
                    ),
                    bound_parameter_identity=self._bound_parameter_identity(node),
                )
                for backend, representation in backends
            )

        mode = str(getattr(ctx, "run_mode", "research") or "research").lower()
        candidates: list[NodeBackendChoice] = []
        bound_params = self._bound_parameter_identity(node)
        kernel_signature = self._kernel_signature(node)
        if supports_pandas(op, mode=mode):
            capability = capability_for(op, "pandas_numpy")
            candidates.append(NodeBackendChoice(
                node_id=node_id,
                backend=PhysicalBackend.PANDAS_NUMPY,
                compute_cost_ms=estimate_backend_cost(
                    op, PhysicalBackend.PANDAS_NUMPY.value, row_count_estimate=rows
                ),
                transfer_from_children_ms=0.0,
                total_cost_ms=0.0,
                representation=infer_representation(PhysicalBackend.PANDAS_NUMPY),
                execution_kind=capability.execution_kind,
                production_certified=capability.is_production_eligible() or mode != "production",
                physical_implementation_id=self._capability_pi_id(capability),
                bound_parameter_identity=bound_params,
                kernel_signature=kernel_signature,
            ))
        if supports_polars(op, mode=mode):
            capability = capability_for(op, "polars")
            is_delegate = canonical_polars_is_delegate(
                op, production_mode=mode == "production"
            )
            if not is_delegate or mode != "production":
                candidates.append(NodeBackendChoice(
                    node_id=node_id,
                    backend=PhysicalBackend.POLARS_PANEL,
                    compute_cost_ms=estimate_backend_cost(
                        op, PhysicalBackend.POLARS_PANEL.value,
                        row_count_estimate=rows,
                    ),
                    transfer_from_children_ms=0.0,
                    total_cost_ms=0.0,
                    representation=infer_representation(PhysicalBackend.POLARS_PANEL),
                    execution_kind=(
                        ExecutionKind.POLARS_PANDAS_DELEGATE
                        if is_delegate else capability.execution_kind
                    ),
                    production_certified=(
                        capability.is_production_eligible() and not is_delegate
                    ) or mode != "production",
                    physical_implementation_id=self._capability_pi_id(capability),
                    bound_parameter_identity=bound_params,
                    kernel_signature=kernel_signature,
                ))
        # Add DuckDB/ClickHouse/Q/Numba candidates
        if supports_sql(op, mode=mode):
            capability = capability_for(op, "duckdb_sql")
            candidates.append(NodeBackendChoice(
                node_id=node_id,
                backend=PhysicalBackend.DUCKDB_SQL,
                compute_cost_ms=estimate_backend_cost(
                    op, PhysicalBackend.DUCKDB_SQL.value, row_count_estimate=rows
                ),
                transfer_from_children_ms=0.0,
                total_cost_ms=0.0,
                representation=infer_representation(PhysicalBackend.DUCKDB_SQL),
                execution_kind=capability.execution_kind,
                production_certified=capability.is_production_eligible() or mode != "production",
                physical_implementation_id=self._capability_pi_id(capability),
                bound_parameter_identity=bound_params,
                kernel_signature=kernel_signature,
            ))
        # ClickHouse (CLICKHOUSE_SQL) candidate — P2 split-brain closure: disabled
        # until the runtime actually wires a CLICKHOUSE_SQL executor in
        # ``_physical_backend_for_region``.  ``ClickHousePushdownBackend`` exists in
        # ``factor_engine.backend`` but is NOT wired into the runtime resolver, so a
        # region routed here would fail at execution time.  Keeping it off the
        # candidate list means the optimizer can never emit it, and
        # ``assert_all_backends_runtime_capable`` (applied after plan build) fails
        # loudly if anything else routes a region to it.
        if False and supports_sql(op, mode=mode, data_source_kind="clickhouse"):
            capability = capability_for(op, "clickhouse_sql")
            candidates.append(NodeBackendChoice(
                node_id=node_id,
                backend=PhysicalBackend.CLICKHOUSE_SQL,
                compute_cost_ms=estimate_backend_cost(
                    op, PhysicalBackend.CLICKHOUSE_SQL.value, row_count_estimate=rows
                ),
                transfer_from_children_ms=0.0,
                total_cost_ms=0.0,
                representation=infer_representation(PhysicalBackend.CLICKHOUSE_SQL),
                execution_kind=capability.execution_kind,
                production_certified=capability.is_production_eligible() or mode != "production",
                physical_implementation_id=self._capability_pi_id(capability),
                bound_parameter_identity=bound_params,
                kernel_signature=kernel_signature,
            ))
        # Q/KDB (Q_KDB) candidate — DISABLED.  This is the truthful fail-closed
        # gate the q backend actually has in this environment:
        #
        #  * ``QBackend`` is a REAL integration path (backend/q_backend/ has a
        #    full compiler/executor/adapter/process-manager), but it requires a
        #    live q runtime.  This environment has NO q/kdb+ binary, NO pykx /
        #    qpython / pyq client, and NO Q_LICENSED env — so
        #    ``QProcessManager.check_availability()`` honestly reports
        #    UNAVAILABLE / LICENSE_MISSING and every executor path raises a
        #    typed, fail-closed error (BackendUnavailableError /
        #    QProcessUnavailableError / QPlanningFallbackAllowed).
        #  * Routing a production region here would produce a plan the runtime
        #    cannot execute (split-brain).  So we never emit the Q_KDB candidate
        #    and ``assert_all_backends_runtime_capable`` (applied after plan
        #    build) rejects any plan that still carries a Q_KDB region.
        #
        # If a real q runtime is provisioned later, ``is_q_available()`` — the
        # SAME honest gate used everywhere else — will report AVAILABLE and this
        # block will start proposing Q_KDB regions again automatically.
        try:
            from factor_engine.backend.q_backend.q_physical_implementation_registry import (
                get_q_physical_implementation_registry,
            )
            registry = get_q_physical_implementation_registry()
            from factor_engine.backend.q_backend.q_process_manager import (
                is_q_available,
            )
            def _q_available_for_optimizer() -> bool:
                try:
                    return is_q_available()
                except Exception:
                    return False
            # Fail-closed: a Q_KDB candidate is only ever admitted when the
            # SAME honest runtime gate the executor will use reports AVAILABLE.
            if (
                _q_available_for_optimizer()
                and registry.has_lowering(op)
            ):
                capability = capability_for(op, "q_kdb")
                candidates.append(NodeBackendChoice(
                    node_id=node_id,
                    backend=PhysicalBackend.Q_KDB,
                    compute_cost_ms=estimate_backend_cost(
                        op, PhysicalBackend.Q_KDB.value, row_count_estimate=rows
                    ),
                    transfer_from_children_ms=0.0,
                    total_cost_ms=0.0,
                    representation=infer_representation(PhysicalBackend.Q_KDB),
                    execution_kind=capability.execution_kind,
                    production_certified=capability.is_production_eligible() or mode != "production",
                    physical_implementation_id=self._capability_pi_id(capability),
                    bound_parameter_identity=bound_params,
                    kernel_signature=kernel_signature,
                ))
        except Exception:
            pass
        # Numba backend
        try:
            from factor_engine.backend.routing import numba_enabled_for_op
            # R45: evaluate the Numba candidate against the node's REAL bound
            # parameters (window/span from the node's params), not a hard-coded
            # window=0.  The kernel signature is derived from the actual bound
            # parameter domain.
            bound_params = self._bound_parameter_identity(node)
            kernel_signature = self._kernel_signature(node)
            real_window = self._bound_window(node)
            if numba_enabled_for_op(op, window=real_window, panel_rows=rows):
                # R45: Numba gets its OWN PhysicalImplementationID and evidence,
                # derived from the certified Numba kernel implementation id.  It
                # does NOT borrow pandas_numpy's production capability.
                numba_pi_id = ""
                numba_certified = False
                try:
                    from factor_engine.backend.numba_kernel_registry import (
                        NumbaKernelRegistry,
                        get_implementation_registry,
                    )
                    if not NumbaKernelRegistry.kernels():
                        import factor_engine.backend.numba_kernels  # noqa: F401
                    impl = get_implementation_registry().get(op)
                    if impl is not None and impl.get("implementation_id"):
                        numba_pi_id = _numba_pi_id(
                            op, str(impl["implementation_id"])
                        )
                        numba_certified = True
                except Exception:
                    numba_pi_id = ""
                    numba_certified = False
                candidates.append(NodeBackendChoice(
                    node_id=node_id,
                    backend=PhysicalBackend.PANDAS_NUMPY,  # Numba runs on the pandas_numpy data plane
                    compute_cost_ms=estimate_backend_cost(
                        op, "numba", row_count_estimate=rows
                    ),
                    transfer_from_children_ms=0.0,
                    total_cost_ms=0.0,
                    representation=infer_representation(PhysicalBackend.PANDAS_NUMPY),
                    execution_kind=ExecutionKind.NUMBA_CPU_KERNEL,
                    production_certified=numba_certified or mode != "production",
                    physical_implementation_id=numba_pi_id,
                    bound_parameter_identity=bound_params,
                    kernel_signature=kernel_signature,
                ))
        except Exception:
            pass
        if not candidates:
            raise UnsupportedOperatorBackendError(
                f"no {'production-certified ' if mode == 'production' else ''}"
                f"backend for operator {op!r} in batch-global optimizer"
            )
        if self.forced_backend not in (None, "", "auto"):
            allowed = {
                "pandas": {PhysicalBackend.PANDAS_NUMPY},
                "pandas_numpy": {PhysicalBackend.PANDAS_NUMPY},
                "polars_long": {PhysicalBackend.POLARS_LONG, PhysicalBackend.POLARS_PANEL},
                "duckdb_sql": {PhysicalBackend.DUCKDB_SQL},
            }.get(self.forced_backend)
            if allowed is None:
                raise ValueError(f"unknown forced backend {self.forced_backend!r}")
            candidates = [choice for choice in candidates if choice.backend in allowed]
            if not candidates:
                raise UnsupportedOperatorBackendError(
                    f"operator {op!r} has no candidate for forced backend {self.forced_backend!r}"
                )
        return tuple(candidates)

    def _optimize_graph(
        self,
        all_nodes: dict[str, PlanNode],
        node_graph: dict[str, list[str]],
        rows: int,
        ctx: Any,
        node_estimates: dict[str, tuple[int, int, int]],
    ) -> tuple[dict[str, NodeBackendChoice], str, int, str]:
        """Minimize one objective over every logical node and dependency edge.

        Search is wall-clock and candidate bounded.  Before exploring mixed
        assignments we materialize every complete single-residency incumbent
        (including Pandas) plus a cheapest-per-node legal assignment.  A timeout
        therefore returns a real, executable plan; it never skips capability or
        production-readiness validation.
        """
        deadline = time.monotonic() + self.max_optimization_ms / 1000.0
        candidates = {
            node_id: self._eligible_choices(node_id, node, rows, ctx)
            for node_id, node in all_nodes.items()
        }
        ambiguous = [node_id for node_id in sorted(candidates) if len(candidates[node_id]) > 1]
        fixed = {
            node_id: choices[0]
            for node_id, choices in candidates.items()
            if len(choices) == 1
        }
        incumbents = self._complete_incumbents(candidates)
        if not incumbents:
            raise ValueError("batch-global optimizer produced no legal incumbent")
        incumbent_name, incumbent = min(
            incumbents,
            key=lambda item: (
                self._assignment_objective(item[1], node_graph, node_estimates),
                item[0],
            ),
        )
        evaluated = len(incumbents)
        if len(ambiguous) <= self.exact_search_max_ambiguous_nodes:
            best, explored, timed_out = self._exact_assignment(
                candidates, ambiguous, fixed, node_graph, node_estimates,
                incumbent=incumbent, deadline=deadline,
                candidate_budget=max(0, self.max_candidate_plans - evaluated),
            )
            evaluated += explored
            basis = "bounded_incumbent_timeout" if timed_out else "dp_global_exact"
        else:
            best, explored, timed_out = self._approximate_assignment(
                candidates, ambiguous, fixed, node_graph, node_estimates,
                incumbent=incumbent, deadline=deadline,
                candidate_budget=max(0, self.max_candidate_plans - evaluated),
            )
            evaluated += explored
            basis = "bounded_incumbent_timeout" if timed_out else "dp_global_approximate"

        result: dict[str, NodeBackendChoice] = {}
        for node_id, choice in best.items():
            transfer = sum(
                self._estimate_transfer_cost_bytes(
                    best[child_id].backend,
                    choice.backend,
                    node_estimates[child_id][1],
                )
                for child_id in node_graph.get(node_id, ())
                if (best[child_id].backend, best[child_id].representation)
                != (choice.backend, choice.representation)
            )
            result[node_id] = NodeBackendChoice(
                node_id=node_id,
                backend=choice.backend,
                compute_cost_ms=choice.compute_cost_ms,
                transfer_from_children_ms=transfer,
                total_cost_ms=choice.compute_cost_ms + transfer,
                representation=choice.representation,
                execution_kind=choice.execution_kind,
                production_certified=choice.production_certified,
                physical_implementation_id=choice.physical_implementation_id,
                bound_parameter_identity=choice.bound_parameter_identity,
                implementation_closure_hash=choice.implementation_closure_hash,
                numeric_policy_identity=choice.numeric_policy_identity,
                kernel_signature=choice.kernel_signature,
            )
        selected = incumbent_name if best == incumbent else "mixed"
        return result, basis, evaluated, selected

    @staticmethod
    def _complete_incumbents(
        candidates: dict[str, tuple[NodeBackendChoice, ...]],
    ) -> list[tuple[str, dict[str, NodeBackendChoice]]]:
        """Return legal whole-residency baselines and one legal mixed fallback."""
        if not candidates:
            return [("empty", {})]
        residencies = set.intersection(*(
            {(choice.backend, choice.representation) for choice in choices}
            for choices in candidates.values()
        ))
        out: list[tuple[str, dict[str, NodeBackendChoice]]] = []
        for backend, representation in sorted(
            residencies, key=lambda item: (item[0].value, item[1].value)
        ):
            assignment = {
                node_id: min(
                    (choice for choice in choices if (choice.backend, choice.representation)
                     == (backend, representation)),
                    key=lambda choice: (choice.compute_cost_ms, choice.execution_kind.value),
                )
                for node_id, choices in candidates.items()
            }
            out.append((f"single:{backend.value}:{representation.value}", assignment))
        mixed = {
            node_id: min(
                choices,
                key=lambda choice: (
                    choice.compute_cost_ms, choice.backend.value,
                    choice.representation.value,
                ),
            )
            for node_id, choices in candidates.items()
        }
        out.append(("legal_mixed", mixed))
        return out

    def _assignment_objective(
        self,
        assignment: dict[str, NodeBackendChoice],
        node_graph: dict[str, list[str]],
        node_estimates: dict[str, tuple[int, int, int]],
    ) -> float:
        cost = sum(choice.compute_cost_ms for choice in assignment.values())
        for consumer_id, children in node_graph.items():
            consumer = assignment[consumer_id]
            for child_id in children:
                producer = assignment[child_id]
                if (producer.backend, producer.representation) != (
                    consumer.backend, consumer.representation
                ):
                    cost += self._estimate_transfer_cost_bytes(
                        producer.backend,
                        consumer.backend,
                        node_estimates[child_id][1],
                    )
        return cost

    def _exact_assignment(
        self,
        candidates: dict[str, tuple[NodeBackendChoice, ...]],
        ambiguous: list[str],
        fixed: dict[str, NodeBackendChoice],
        node_graph: dict[str, list[str]],
        node_estimates: dict[str, tuple[int, int, int]],
        *,
        incumbent: dict[str, NodeBackendChoice],
        deadline: float,
        candidate_budget: int,
    ) -> tuple[dict[str, NodeBackendChoice], int, bool]:
        best = dict(incumbent)
        best_cost = self._assignment_objective(best, node_graph, node_estimates)
        explored = 0
        timed_out = False

        def search(index: int, assignment: dict[str, NodeBackendChoice]) -> None:
            nonlocal best, best_cost, explored, timed_out
            if timed_out or time.monotonic() >= deadline or explored >= candidate_budget:
                timed_out = True
                return
            compute_floor = sum(choice.compute_cost_ms for choice in assignment.values())
            if compute_floor > best_cost:
                return
            if index == len(ambiguous):
                explored += 1
                value = self._assignment_objective(
                    assignment, node_graph, node_estimates
                )
                signature = self._assignment_signature(assignment)
                best_signature = self._assignment_signature(best) if best is not None else ()
                if value < best_cost or (value == best_cost and signature < best_signature):
                    best_cost = value
                    best = dict(assignment)
                return
            node_id = ambiguous[index]
            for choice in self._sorted_choices(candidates[node_id]):
                assignment[node_id] = choice
                search(index + 1, assignment)
            assignment.pop(node_id, None)

        search(0, dict(fixed))
        return best, explored, timed_out

    def _approximate_assignment(
        self,
        candidates: dict[str, tuple[NodeBackendChoice, ...]],
        ambiguous: list[str],
        fixed: dict[str, NodeBackendChoice],
        node_graph: dict[str, list[str]],
        node_estimates: dict[str, tuple[int, int, int]],
        *,
        incumbent: dict[str, NodeBackendChoice],
        deadline: float,
        candidate_budget: int,
    ) -> tuple[dict[str, NodeBackendChoice], int, bool]:
        """Deterministic bounded coordinate descent over the global objective."""
        assignment = dict(incumbent)
        explored = 0
        timed_out = False

        parents: dict[str, list[str]] = {node_id: [] for node_id in assignment}
        for consumer_id, child_ids in node_graph.items():
            for child_id in child_ids:
                parents[child_id].append(consumer_id)
        for node_id in parents:
            parents[node_id].sort()

        for _ in range(self.approximate_max_passes):
            changed = False
            for node_id in ambiguous:
                if time.monotonic() >= deadline or explored >= candidate_budget:
                    timed_out = True
                    break
                current = assignment[node_id]
                best_choice = current
                best_key: tuple[float, str, str] | None = None
                for choice in self._sorted_choices(candidates[node_id]):
                    explored += 1
                    if explored > candidate_budget or time.monotonic() >= deadline:
                        timed_out = True
                        break
                    key = (
                        self._local_choice_cost(
                            node_id,
                            choice,
                            assignment,
                            node_graph,
                            parents,
                            node_estimates,
                        ),
                        choice.backend.value,
                        choice.representation.value,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_choice = choice
                assignment[node_id] = best_choice
                changed = changed or best_choice != current
                if timed_out:
                    break
            if not changed:
                break
            if timed_out:
                break
        incumbent_cost = self._assignment_objective(incumbent, node_graph, node_estimates)
        assignment_cost = self._assignment_objective(assignment, node_graph, node_estimates)
        if assignment_cost > incumbent_cost:
            assignment = dict(incumbent)
        return assignment, min(explored, candidate_budget), timed_out

    def _local_choice_cost(
        self,
        node_id: str,
        choice: NodeBackendChoice,
        assignment: dict[str, NodeBackendChoice],
        node_graph: dict[str, list[str]],
        parents: dict[str, list[str]],
        node_estimates: dict[str, tuple[int, int, int]],
    ) -> float:
        """Score only terms incident to one node for bounded coordinate descent."""
        cost = choice.compute_cost_ms
        for child_id in node_graph.get(node_id, ()):
            child = assignment[child_id]
            if (child.backend, child.representation) != (
                choice.backend, choice.representation
            ):
                cost += self._estimate_transfer_cost_bytes(
                    child.backend, choice.backend, node_estimates[child_id][1]
                )
        for parent_id in parents[node_id]:
            parent = assignment[parent_id]
            if (choice.backend, choice.representation) != (
                parent.backend, parent.representation
            ):
                cost += self._estimate_transfer_cost_bytes(
                    choice.backend, parent.backend, node_estimates[node_id][1]
                )
        return cost

    @staticmethod
    def _sorted_choices(
        choices: tuple[NodeBackendChoice, ...],
    ) -> tuple[NodeBackendChoice, ...]:
        return tuple(sorted(
            choices,
            key=lambda choice: (choice.backend.value, choice.representation.value),
        ))

    @staticmethod
    def _assignment_signature(
        assignment: dict[str, NodeBackendChoice],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (assignment[node_id].backend.value, assignment[node_id].representation.value)
            for node_id in sorted(assignment)
        )

    @staticmethod
    def _derive_node_estimates(
        all_nodes: dict[str, PlanNode],
        node_graph: dict[str, list[str]],
        *,
        roots: tuple[str, ...],
        global_rows: int | None,
        global_bytes: int | None,
        global_memory: int | None,
    ) -> dict[str, tuple[int, int, int]]:
        """Collect node-specific row/byte/memory evidence without fabricating it."""
        estimates: dict[str, tuple[int, int, int]] = {}
        source_ids = [
            node_id
            for node_id, node in all_nodes.items()
            if getattr(node, "op", "") == "column"
        ]
        sole_source = source_ids[0] if len(source_ids) == 1 else None
        for node_id, node in all_nodes.items():
            attrs = dict(getattr(node, "attrs", None) or {})
            def positive(*keys: str) -> int:
                for key in keys:
                    try:
                        value = int(attrs.get(key) or 0)
                    except (TypeError, ValueError):
                        value = 0
                    if value > 0:
                        return value
                return 0
            rows = positive("estimated_rows", "row_count_estimate")
            byte_count = positive("estimated_bytes", "byte_count_estimate")
            memory = positive("estimated_memory_bytes", "memory_bytes_estimate")
            if node_id in roots:
                rows = rows or int(global_rows or 0)
                byte_count = byte_count or int(global_bytes or 0)
                memory = memory or int(global_memory or 0)
            if node_id == sole_source:
                rows = rows or int(global_rows or 0)
                byte_count = byte_count or int(global_bytes or 0)
                memory = memory or int(global_memory or 0)
            estimates[node_id] = (rows, byte_count, memory)
        return estimates

    @staticmethod
    def _has_source_residency(node: PlanNode) -> bool:
        attrs = getattr(node, "attrs", None) or {}
        return bool(attrs.get("source_backend") or attrs.get("source_representation"))

    @staticmethod
    def _column_has_source_binding(node: PlanNode) -> bool:
        """R45: a ``column`` node is production-certified ONLY with a real
        physical source binding.

        A column is a PhysicalSourceBinding node.  It is certified when it
        carries the full read identity: DataReadIdentity (dataset/revision),
        field semantics, PIT (calendar identity / availability cutoff),
        universe snapshot, and source snapshot.  A bare ``column`` with only a
        name (no binding) is NOT production-certified.
        """
        from factor_engine.runtime.physical_source_binding import (
            is_authoritative_source_binding,
        )

        attrs = dict(getattr(node, "attrs", None) or {})
        return is_authoritative_source_binding(attrs.get("physical_source_binding"))

    @classmethod
    def _source_residency(
        cls, node: PlanNode
    ) -> tuple[PhysicalBackend, Representation]:
        attrs = getattr(node, "attrs", None) or {}
        backend_name = str(attrs.get("source_backend") or "").lower()
        representation_name = str(attrs.get("source_representation") or "").lower()
        representation_map = {item.value: item for item in Representation}
        representation = representation_map.get(representation_name)
        backend_by_representation = {
            Representation.POLARS_LONG: PhysicalBackend.POLARS_LONG,
            Representation.POLARS_WIDE: PhysicalBackend.POLARS_PANEL,
            Representation.POLARS_LAZY_LONG: PhysicalBackend.POLARS_LONG,
            Representation.DUCKDB_RELATION: PhysicalBackend.DUCKDB_SQL,
            Representation.PANDAS_LONG: PhysicalBackend.PANDAS_NUMPY,
            Representation.PANDAS_WIDE: PhysicalBackend.PANDAS_NUMPY,
            Representation.NUMPY_PANEL: PhysicalBackend.PANDAS_NUMPY,
        }
        backend_map = {item.value: item for item in PhysicalBackend}
        backend = backend_map.get(backend_name)
        if backend_name and backend is None:
            raise ValueError(f"unknown source_backend {backend_name!r}")
        if representation_name and representation is None:
            raise ValueError(f"unknown source_representation {representation_name!r}")
        inferred_backend = (
            backend_by_representation.get(representation)
            if representation is not None
            else None
        )
        if backend is not None and inferred_backend is not None and backend != inferred_backend:
            raise ValueError(
                "source backend and representation conflict: "
                f"{backend.value!r} vs {representation.value!r}"
            )
        if backend is None:
            backend = inferred_backend
        backend = backend or PhysicalBackend.PANDAS_NUMPY
        return backend, representation or infer_representation(backend)

    def _estimate_transfer_cost_bytes(
        self, source: PhysicalBackend, target: PhysicalBackend, bytes_estimate: int
    ) -> float:
        """Estimate transfer cost from evidenced payload bytes."""
        if source == target or bytes_estimate <= 0:
            return 0.0
        return self.delegate_penalty_ms + (
            bytes_estimate / 1_000_000.0 * self.transfer_cost_per_mb
        )

    def _estimate_transfer_cost(
        self, source: PhysicalBackend, target: PhysicalBackend, rows: int
    ) -> float:
        """Estimate transfer cost between backends."""
        if source == target:
            return 0.0
        bytes_estimate = rows * 8  # Simplified
        mb = bytes_estimate / 1_000_000.0
        return self.delegate_penalty_ms + (mb * self.transfer_cost_per_mb)

    @staticmethod
    def _readiness(
        *,
        roots: dict[str, PlanNode],
        all_nodes: dict[str, PlanNode],
        choices: dict[str, NodeBackendChoice],
        plan: PhysicalRegionPlan,
        rows: int | None,
        estimated_bytes: int | None,
        estimated_memory: int | None,
        node_graph: dict[str, list[str]],
        run_mode: str | None,
    ) -> tuple[bool, str]:
        """Apply complete fail-closed production-readiness gates."""
        if not roots or not all_nodes:
            return False, "empty logical plan"
        if not choices or not plan.regions or not plan.root_region_ids:
            return False, "empty physical plan"
        missing = sorted(set(all_nodes) - set(choices))
        if missing:
            return False, f"unassigned logical nodes: {', '.join(missing)}"
        delegates = sorted(
            node_id
            for node_id, choice in choices.items()
            if choice.execution_kind
            in {
                ExecutionKind.DELEGATE_PYTHON,
                ExecutionKind.DELEGATE_PANDAS,
                ExecutionKind.POLARS_PANDAS_DELEGATE,
            }
        )
        if delegates:
            return False, f"delegate fallback assigned: {', '.join(delegates)}"
        if rows is None:
            return False, "row-count estimate unavailable"
        if estimated_bytes is None:
            return False, "byte estimate unavailable"
        if estimated_memory is None:
            return False, "memory estimate unavailable"
        if any(not choice.production_certified for choice in choices.values()):
            return False, "non-production-certified backend assignment"
        if any(
            choice.execution_kind == ExecutionKind.UNSUPPORTED
            for choice in choices.values()
        ):
            return False, "unsupported backend assignment"
        if any(
            not region.node_ids
            or region.required_properties is None
            or region.output_properties is None
            for region in plan.regions
        ):
            return False, "region execution contract incomplete"
        expected_boundaries = {
            (child_id, consumer_id)
            for consumer_id, child_ids in node_graph.items()
            for child_id in child_ids
            if child_id in choices
            and consumer_id in choices
            and (
                choices[child_id].backend != choices[consumer_id].backend
                or choices[child_id].representation
                != choices[consumer_id].representation
            )
        }
        actual_boundaries = {
            (producer_id, consumer_id)
            for producer_id, consumer_id in expected_boundaries
            if any(
                edge.producer_region
                == next(
                    region.region_id
                    for region in plan.regions
                    if producer_id in region.node_ids
                )
                and edge.consumer_region
                == next(
                    region.region_id
                    for region in plan.regions
                    if consumer_id in region.node_ids
                )
                for edge in plan.edges
            )
        }
        if actual_boundaries != expected_boundaries:
            return False, "backend switch lacks transfer edge"
        if rows is None:
            return False, "row-count estimate unavailable"
        if estimated_bytes is None:
            return False, "byte estimate unavailable"
        if estimated_memory is None:
            return False, "memory estimate unavailable"
        if any(region.estimated_rows <= 0 for region in plan.regions):
            return False, "region row estimate unavailable"
        if any(region.estimated_memory_bytes <= 0 for region in plan.regions):
            return False, "region memory estimate unavailable"
        if any(edge.estimated_rows <= 0 or edge.estimated_bytes <= 0 for edge in plan.edges):
            return False, "transfer estimate unavailable"
        if plan.peak_memory_bytes <= 0 or plan.logical_node_count != len(all_nodes):
            return False, "plan estimates or node coverage incomplete"
        if run_mode != "production":
            return False, "production readiness requires production run mode"
        return True, ""

    def _recompute_shared_benefits_after_assignment(
        self,
        *,
        shared_nodes: dict[str, PlanNode],
        consumer_counts: dict[str, int],
        per_node_choices: dict[str, NodeBackendChoice],
        source_nodes: set[str],
        node_estimates: dict[str, tuple[int, int, int]],
        rows: int,
        ctx: Any,
    ) -> tuple[dict[str, SharedNodeBenefit], float]:
        """MB-P1-010, §44: Recompute shared benefit using selected backend costs.

        After physical assignment, the actual compute cost for a shared node
        depends on its assigned backend (Polars, Q, Numba, etc.), not the
        Pandas baseline used in the initial estimate.  This method recalculates
        avoided compute based on the selected implementation.
        """
        from factor_engine.backend.operator_cost import estimate_backend_cost

        benefits: dict[str, SharedNodeBenefit] = {}
        total_benefit = 0.0

        for node_id, node in shared_nodes.items():
            consumers = consumer_counts.get(node_id, 0)
            if consumers <= 1:
                continue

            # Use the assigned backend cost if available, else fall back to
            # the initially-estimated cost (which used Pandas baseline).
            choice = per_node_choices.get(node_id)
            if choice is not None:
                # Recompute cost with the selected backend
                from factor_engine.backend.plan_cost_router import plan_occurrences

                occurrences = plan_occurrences(node)
                if occurrences:
                    occ = occurrences[0]
                    compute_cost = estimate_backend_cost(
                        occ.canonical,
                        choice.backend.value,
                        row_count_estimate=rows,
                    )
                else:
                    compute_cost = choice.compute_cost_ms
            else:
                # Shared node not directly assigned; use initial estimate
                from factor_engine.backend.plan_cost_router import plan_occurrences

                occurrences = plan_occurrences(node)
                if occurrences:
                    occ = occurrences[0]
                    compute_cost = estimate_backend_cost(
                        occ.canonical,
                        "pandas_numpy",
                        row_count_estimate=rows,
                    )
                else:
                    compute_cost = 0.0

            # Benefit: we compute once but N consumers use it
            avoided_recompute = compute_cost * (consumers - 1)

            # Source scan benefit
            scan_bytes = 0
            avoided_scan_bytes = 0
            if node_id in source_nodes:
                attrs = getattr(node, "attrs", None) or {}
                scan_bytes = int(attrs.get("estimated_bytes", 0) or 0)
                if scan_bytes < 0:
                    scan_bytes = 0
                avoided_scan_bytes = scan_bytes * (consumers - 1)

            scan_benefit = avoided_scan_bytes / 1_000_000.0 * self.scan_cost_per_mb
            total_node_benefit = avoided_recompute + scan_benefit

            benefit = SharedNodeBenefit(
                node_id=node_id,
                consumer_count=consumers,
                compute_cost_ms=compute_cost,
                avoided_recompute_ms=avoided_recompute,
                scan_bytes=scan_bytes,
                avoided_scan_bytes=avoided_scan_bytes,
                benefit_ms=total_node_benefit,
            )
            benefits[node_id] = benefit
            total_benefit += total_node_benefit

        return benefits, total_benefit

    def _build_physical_plan(
        self,
        per_node_choices: dict[str, NodeBackendChoice],
        shared_benefits: dict[str, SharedNodeBenefit],
        total_benefit: float,
        total_compute: float,
        total_transfer: float,
        ctx: Any,
        node_graph: dict[str, list[str]],
        root_ids: tuple[str, ...],
        node_estimates: dict[str, tuple[int, int, int]],
    ) -> PhysicalRegionPlan:
        """Build PhysicalRegionPlan from optimization results."""
        import hashlib
        import json

        # Build maximal connected residency regions, not one global bucket per
        # backend. Re-entering a backend later in a chain must form a new region.
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in per_node_choices}
        for parent_id, child_ids in node_graph.items():
            parent = per_node_choices.get(parent_id)
            if parent is None:
                continue
            for child_id in child_ids:
                child = per_node_choices.get(child_id)
                if child is None:
                    continue
                if (parent.backend, parent.representation) == (
                    child.backend,
                    child.representation,
                ):
                    adjacency[parent_id].add(child_id)
                    adjacency[child_id].add(parent_id)

        region_components: list[tuple[PhysicalBackend, Representation, list[str]]] = []
        unassigned = set(per_node_choices)
        while unassigned:
            seed = min(unassigned)
            residency = (
                per_node_choices[seed].backend,
                per_node_choices[seed].representation,
            )
            component: list[str] = []
            stack = [seed]
            unassigned.remove(seed)
            while stack:
                node_id = stack.pop()
                component.append(node_id)
                for neighbor in sorted(adjacency[node_id]):
                    if neighbor in unassigned:
                        unassigned.remove(neighbor)
                        stack.append(neighbor)
            region_components.append((*residency, component))

        topological_nodes: list[str] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for child_id in node_graph.get(node_id, ()):
                visit(child_id)
            topological_nodes.append(node_id)

        for node_id in sorted(per_node_choices):
            visit(node_id)
        node_rank = {node_id: index for index, node_id in enumerate(topological_nodes)}
        region_components.sort(
            key=lambda component: min(node_rank[node_id] for node_id in component[2])
        )

        regions: list[BackendRegion] = []
        region_for_node: dict[str, str] = {}
        # A region carries explicit contracts; unknown resource estimates stay
        # zero so readiness can reject the plan rather than inventing values.
        for idx, (backend, representation, node_ids) in enumerate(region_components):
            region_id = f"region_{idx}_{backend.value}_{representation.value}"
            for node_id in node_ids:
                region_for_node[node_id] = region_id
            region_rows = max(
                (node_estimates[node_id][0] for node_id in node_ids), default=0
            )
            region_memory = max(
                (node_estimates[node_id][2] for node_id in node_ids), default=0
            )
            region = BackendRegion(
                region_id=region_id,
                backend=backend,
                representation=representation,
                node_ids=tuple(node_ids),
                execution_axis=ExecutionAxis.GLOBAL_PANEL,
                estimated_rows=region_rows,
                estimated_compute_ms=sum(
                    per_node_choices[nid].compute_cost_ms for nid in node_ids
                ),
                estimated_memory_bytes=region_memory,
                required_properties=PhysicalProperties(),
                output_properties=PhysicalProperties(),
                state_contract=None,
            )
            regions.append(region)

        # Build one explicit edge for every cross-backend dependency.
        edges: list[TransferEdge] = []
        seen_boundaries: set[tuple[str, str]] = set()
        for consumer_id, child_ids in node_graph.items():
            consumer = per_node_choices.get(consumer_id)
            if consumer is None:
                continue
            for child_id in child_ids:
                producer = per_node_choices.get(child_id)
                if producer is None or (
                    producer.backend == consumer.backend
                    and producer.representation == consumer.representation
                ):
                    continue
                boundary = (child_id, consumer_id)
                if boundary in seen_boundaries:
                    continue
                seen_boundaries.add(boundary)
                edge_rows, edge_bytes, _ = node_estimates[child_id]
                edge_id = f"edge_{len(edges)}_{child_id}_to_{consumer_id}"
                edges.append(
                    TransferEdge(
                        edge_id=edge_id,
                        producer_region=region_for_node[child_id],
                        consumer_region=region_for_node[consumer_id],
                        source_backend=producer.backend,
                        target_backend=consumer.backend,
                        source_representation=producer.representation,
                        target_representation=consumer.representation,
                        transform=TransferTransform.SAME_BACKEND_NATIVE,
                        estimated_rows=edge_rows,
                        estimated_bytes=edge_bytes,
                        estimated_transfer_ms=(
                            self.delegate_penalty_ms
                            + edge_bytes / 1_000_000.0 * self.transfer_cost_per_mb
                            if producer.backend != consumer.backend and edge_bytes > 0
                            else 0.0
                        ),
                    )
                )

        # Create plan hash
        plan_dict = {
            "nodes": sorted(per_node_choices.keys()),
            "backends": {nid: choice.backend.value
                        for nid, choice in per_node_choices.items()},
        }
        plan_hash = hashlib.sha256(
            json.dumps(plan_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

        backend_switches = sum(
            edge.source_backend != edge.target_backend for edge in edges
        )
        total_ttdc = total_compute + total_transfer - total_benefit

        return PhysicalRegionPlan(
            plan_id=plan_hash,
            regions=tuple(regions),
            edges=tuple(edges),
            topological_order=tuple(r.region_id for r in regions),
            root_region_ids=tuple(
                dict.fromkeys(region_for_node[root_id] for root_id in root_ids if root_id in region_for_node)
            ),
            total_compute_ms=total_compute,
            total_transfer_ms=total_transfer,
            total_ttdc_ms=max(0.0, total_ttdc),
            peak_memory_bytes=sum(region.estimated_memory_bytes for region in regions),
            plan_hash=plan_hash,
            logical_node_count=len(per_node_choices),
            backend_switch_count=backend_switches,
            native_fraction=(
                sum(
                    choice.execution_kind
                    in {
                        ExecutionKind.NATIVE_EXPR,
                        ExecutionKind.NATIVE_GROUP,
                        ExecutionKind.NATIVE_STREAMING,
                        ExecutionKind.POLARS_NATIVE_EXPR,
                        ExecutionKind.POLARS_NUMPY_KERNEL,
                        ExecutionKind.DUCKDB_NATIVE_SQL,
                    }
                    for choice in per_node_choices.values()
                ) / len(per_node_choices)
                if per_node_choices else 0.0
            ),
            routing_basis="estimated",
        )


def optimize_batch_global(
    roots: dict[str, PlanNode],
    shared_nodes: dict[str, PlanNode],
    node_graph: dict[str, list[str]],
    ctx: Any,
    *,
    max_optimization_ms: float = 250.0,
    max_candidate_plans: int = 128,
    forced_backend: str | None = None,
) -> BatchOptimizationResult:
    """MB-P1-009: Entry point for batch-global optimization.

    This replaces the per-root loop in plan_batch_route with true global optimization.
    """
    optimizer = PhysicalBatchGlobalOptimizer(
        max_optimization_ms=max_optimization_ms,
        max_candidate_plans=max_candidate_plans,
        forced_backend=forced_backend,
    )
    return optimizer.optimize_batch(roots, shared_nodes, node_graph, ctx)


def validate_root_physical_support(
    root: PlanNode, ctx: Any, *, forced_backend: str | None = None,
    shared_nodes: dict[str, PlanNode] | None = None,
) -> None:
    """Capability-only root validation; this does not optimize a plan."""
    optimizer = PhysicalBatchGlobalOptimizer(forced_backend=forced_backend)
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        optimizer._eligible_choices(node.node_id, node, 1, ctx)
        if node.op == "plan_ref":
            sid = str((node.attrs or {}).get("sid") or "")
            definition = (shared_nodes or {}).get(sid)
            if definition is not None:
                stack.append(definition)
        stack.extend(node.inputs)


class BatchGlobalOptimizer(PhysicalBatchGlobalOptimizer):
    """Compatibility facade for physical and legacy DAG optimization APIs."""

    def __init__(
        self,
        *,
        transfer_cost_per_mb: float = 0.05,
        delegate_penalty_ms: float = 3.0,
        scan_cost_per_mb: float = 0.01,
        exact_search_max_ambiguous_nodes: int = 12,
        approximate_max_passes: int = 4,
        max_optimization_ms: float = 250.0,
        max_candidate_plans: int = 128,
        forced_backend: str | None = None,
        min_savings_bytes: int = 10 * 1024 * 1024,
        min_savings_work: float = 50_000.0,
        min_reuse_count: int = 2,
    ) -> None:
        super().__init__(
            transfer_cost_per_mb=transfer_cost_per_mb,
            delegate_penalty_ms=delegate_penalty_ms,
            scan_cost_per_mb=scan_cost_per_mb,
            exact_search_max_ambiguous_nodes=exact_search_max_ambiguous_nodes,
            approximate_max_passes=approximate_max_passes,
            max_optimization_ms=max_optimization_ms,
            max_candidate_plans=max_candidate_plans,
            forced_backend=forced_backend,
        )
        self._min_savings_bytes = min_savings_bytes
        self._min_savings_work = min_savings_work
        self._min_reuse_count = min_reuse_count

    def optimize_batch(self, roots: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(roots, dict):
            return super().optimize_batch(roots, *args, **kwargs)
        return self._optimize_legacy(roots, **kwargs)

    def _optimize_legacy(self, dag: Any, **kwargs: Any) -> GlobalOptimizationResult:
        opportunities: list[OptimizationOpportunity] = []
        cost_fn = kwargs.get("cost_fn")
        if kwargs.get("enable_cse", True):
            opportunities.extend(self._detect_cross_root_cse(dag, cost_fn))
        if kwargs.get("enable_predicate_push", True):
            opportunities.extend(self._detect_predicate_pushdown(dag, cost_fn))
        if kwargs.get("enable_constant_fold", True):
            opportunities.extend(self._detect_constant_folding(dag, cost_fn))
        opportunities.sort(key=lambda item: (item.savings_bytes, item.savings_work), reverse=True)
        rewritten = dag
        applied: list[OptimizationOpportunity] = []
        for opportunity in opportunities:
            if opportunity.savings_bytes < self._min_savings_bytes and opportunity.savings_work < self._min_savings_work:
                continue
            candidate = self._apply_optimization(rewritten, opportunity)
            if self._dag_structure_signature(candidate) != self._dag_structure_signature(rewritten):
                rewritten = candidate
                applied.append(opportunity)
        return GlobalOptimizationResult(
            opportunities=opportunities,
            applied_count=len(applied),
            total_savings_bytes=sum(item.savings_bytes for item in applied),
            total_savings_work=sum(item.savings_work for item in applied),
            rewritten_dag=rewritten if applied else None,
        )

    def _detect_cross_root_cse(self, dag: Any, cost_fn: Callable[[Any], dict[str, Any]] | None) -> list[OptimizationOpportunity]:
        fingerprints: dict[str, list[str]] = {}
        for root_id, task in self._iter_root_tasks(dag):
            for node in self._traverse_plan(task):
                fingerprint = self._structural_fingerprint(node)
                if fingerprint:
                    fingerprints.setdefault(fingerprint, []).append(str(root_id))
        opportunities = []
        for fingerprint, roots in fingerprints.items():
            if len(roots) < self._min_reuse_count:
                continue
            node_cost = self._estimate_node_cost(self._find_root_node(dag, roots[0]), cost_fn)
            reuse = len(roots) - 1
            opportunities.append(OptimizationOpportunity(
                kind="cse",
                savings_bytes=int(node_cost.get("peak_live_memory_bytes", 0) * reuse),
                savings_work=float(node_cost.get("total_work", 0.0) * reuse),
                affected_roots=roots,
                shared_node_id=f"shared_{fingerprint[:8]}",
                confidence=0.85,
            ))
        return opportunities

    def _detect_predicate_pushdown(self, dag: Any, cost_fn: Callable[[Any], dict[str, Any]] | None) -> list[OptimizationOpportunity]:
        return []

    def _detect_constant_folding(self, dag: Any, cost_fn: Callable[[Any], dict[str, Any]] | None) -> list[OptimizationOpportunity]:
        return []

    def _apply_optimization(self, dag: Any, opportunity: OptimizationOpportunity) -> Any:
        return dag

    @staticmethod
    def _iter_root_tasks(dag: Any) -> list[tuple[str, Any]]:
        tasks = getattr(dag, "tasks", {})
        return [(str(task_id), task) for task_id, task in tasks.items() if getattr(task, "task_type", None) == "root"]

    @staticmethod
    def _traverse_plan(task: Any) -> list[Any]:
        result = []
        def visit(node: Any) -> None:
            if node is None:
                return
            result.append(node)
            for child in getattr(node, "inputs", ()) or ():
                visit(child)
        visit(getattr(task, "node_ref", None))
        return result

    @staticmethod
    def _structural_fingerprint(node: Any) -> str | None:
        op = getattr(node, "op", None)
        if op is None:
            return None
        children = tuple(BatchGlobalOptimizer._structural_fingerprint(child) for child in getattr(node, "inputs", ()) or ())
        params = tuple(sorted((getattr(node, "params", {}) or {}).items()))
        return repr((op, children, params))

    @staticmethod
    def _find_root_node(dag: Any, root_id: str) -> Any:
        return getattr(getattr(dag, "tasks", {}).get(root_id), "node_ref", None)

    @staticmethod
    def _estimate_node_cost(node: Any, cost_fn: Callable[[Any], dict[str, Any]] | None) -> dict[str, Any]:
        if cost_fn is not None:
            try:
                return cost_fn(node)
            except Exception:
                pass
        return {"peak_live_memory_bytes": 0, "total_work": 0.0}

    @staticmethod
    def _dag_structure_signature(dag: Any) -> tuple[Any, ...]:
        tasks = getattr(dag, "tasks", {})
        if not hasattr(tasks, "items"):
            return (type(dag).__qualname__, repr(dag))
        return tuple(
            (str(task_id), getattr(task, "task_type", None),
             tuple(sorted(map(str, getattr(task, "dependencies", ())))),
             getattr(getattr(task, "node_ref", None), "op", None))
            for task_id, task in sorted(tasks.items(), key=lambda item: str(item[0]))
        )
