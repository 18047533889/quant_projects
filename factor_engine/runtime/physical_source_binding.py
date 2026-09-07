"""Authoritative DataAccess bindings for logical source columns."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from factor_engine.planner.logical_plan import PlanNode


@dataclass(frozen=True)
class PhysicalSourceBinding:
    read_identity: Any
    logical_field: str
    semantic_catalog_identity: str

    @property
    def digest(self) -> str:
        import hashlib
        payload = "|".join((
            str(getattr(self.read_identity, "digest", "") or ""),
            self.logical_field,
            self.semantic_catalog_identity,
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RootPhysicalPreflightError:
    code: str
    error_type: str
    message: str


@dataclass(frozen=True)
class PreparedSourceSnapshotExpectation:
    dataset: str
    columns: tuple[str, ...]
    params: tuple[tuple[str, Any], ...]
    time_range: tuple[Any, Any]
    instrument_filter: tuple[Any, ...]
    content_digest: str


@dataclass(frozen=True)
class BatchSourcePreflightResult:
    root_plans: dict[str, PlanNode]
    shared_nodes: dict[str, PlanNode]
    per_root_errors: dict[str, RootPhysicalPreflightError]
    scope_error: RootPhysicalPreflightError | None = None
    prepare_count: int = 0
    unique_source_count: int = 0
    snapshot_expectations: tuple[PreparedSourceSnapshotExpectation, ...] = ()

    @property
    def valid_root_names(self) -> tuple[str, ...]:
        return tuple(self.root_plans)


def is_authoritative_source_binding(value: Any) -> bool:
    """Validate a binding by reconstructing its DA-issued content identity."""
    if not isinstance(value, PhysicalSourceBinding):
        return False
    try:
        from data_access.read.data_read_identity import DataReadIdentity
        identity = value.read_identity
        if not isinstance(identity, DataReadIdentity):
            return False
        rebuilt = DataReadIdentity.from_dict(identity.to_dict())
    except Exception:
        return False
    return bool(
        value.logical_field and value.semantic_catalog_identity and identity.digest
        and rebuilt.digest == identity.digest
        and identity.provenance_status == "available"
        and identity.dataset and identity.revision and identity.calendar_identity
        and identity.universe_snapshot and identity.source_snapshot and identity.fields
    )


def _build_binding(
    source: Any, logical_field: str, scope: Any, *, catalog_field: str | None = None,
    snapshot: Any,
    store: Any, prepared: Any,
    decision_clock: str | None, availability_cutoff: str | None,
) -> PhysicalSourceBinding:
    """Build one strict identity from DataAccess authorities, never plan attrs."""
    from data_access.read.data_read_identity import build_data_read_identity
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    if not isinstance(source, DataAccessSource):
        raise TypeError("production physical source binding requires DataAccessSource")
    if not source.production or not source.strict_unknown_fields or not source.pit_enforce:
        raise ValueError("production physical source binding requires strict PIT DataAccessSource")
    resolved_name = catalog_field or logical_field
    field_plan = source._ensure_field_plans([resolved_name])[resolved_name]
    if field_plan.source == "raw":
        raise ValueError(f"field {logical_field!r} has no authoritative semantic plan")
    semantic_field = (source._resolve_catalog_fields([resolved_name]) or {}).get(resolved_name)
    if semantic_field is None:
        raise ValueError(f"field {resolved_name!r} has no DataAccess semantic-catalog binding")
    universe = str(getattr(scope, "universe_id", "") or "")
    if not universe:
        raise ValueError("production physical source binding requires an execution universe")
    identity = build_data_read_identity(
        store, dataset=source.dataset,
        columns=tuple(field_plan.physical_fields) or (resolved_name,),
        time_range=(source.start_date, source.end_date),
        instrument_filter=source.instrument_filter, universe=universe,
        decision_clock=decision_clock,
        availability_cutoff=availability_cutoff,
        fields=(semantic_field,), prepared=prepared, strict=True,
    )
    binding = PhysicalSourceBinding(identity, logical_field, source.semantic_catalog_identity())
    if not is_authoritative_source_binding(binding):
        raise ValueError(f"DataAccess issued an incomplete identity for {logical_field!r}")
    pit_relevant = any(
        field.availability not in (None, "same_day")
        or field.pit_fidelity not in (None, "effective_only", "unsupported")
        or any(token in str(field.temporal_model or "").lower()
               for token in ("financial", "event"))
        for field in identity.fields
    )
    if pit_relevant and not (identity.decision_clock or identity.availability_cutoff):
        raise ValueError(
            f"PIT-relevant field {logical_field!r} requires an authoritative decision clock"
        )
    return binding


def bind_plan_sources(
    plan: PlanNode, *, source: Any, scope: Any,
    snapshot: Any,
    store: Any | None = None,
    prepared: Any = None,
    decision_clock: str | None = None,
    availability_cutoff: str | None = None,
    _binding_cache: dict[tuple[str, ...], PhysicalSourceBinding] | None = None,
    column_authorities: dict[str, tuple[Any, str, Any, Any]] | None = None,
) -> PlanNode:
    """Return an immutable plan whose columns carry DA-issued source bindings."""
    memo: dict[int, PlanNode] = {}
    if store is None:
        from data_access import get_store
        store = get_store()
    binding_cache = _binding_cache if _binding_cache is not None else {}
    def visit(node: PlanNode) -> PlanNode:
        cached = memo.get(id(node))
        if cached is not None:
            return cached
        attrs = dict(node.attrs)
        if node.op == "column":
            name = str(attrs.get("name") or "")
            if not name:
                raise ValueError("source column name is required")
            actual_source, actual_field, actual_snapshot, actual_prepared = (
                column_authorities.get(name, (source, name, snapshot, prepared))
                if column_authorities is not None else (source, name, snapshot, prepared)
            )
            key = (
                name, str(getattr(scope, "scope_key")()),
                str(getattr(actual_source, "dataset", "") or ""), actual_field,
                str(getattr(actual_snapshot, "content_digest", "") or ""),
                str(decision_clock or ""), str(availability_cutoff or ""),
            )
            if key not in binding_cache:
                binding_cache[key] = _build_binding(
                    actual_source, name, scope, catalog_field=actual_field,
                    snapshot=actual_snapshot,
                    store=store,
                    prepared=actual_prepared,
                    decision_clock=decision_clock,
                    availability_cutoff=availability_cutoff,
                )
            attrs["physical_source_binding"] = binding_cache[key]
        rebound = PlanNode(node.op, inputs=tuple(visit(c) for c in node.inputs), attrs=attrs,
                           semantic_attrs=node.semantic_attrs, node_id=node.node_id)
        memo[id(node)] = rebound
        return rebound
    return visit(plan)


def bind_batch_sources(
    dag: Any, *, source: Any, execution_context: Any,
    _diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, PlanNode], dict[str, PlanNode]]:
    """Bind a compiled DAG using its actual per-root execution scope."""
    from data_access import get_store
    from factor_engine.planner.batch_data_request import BatchSourceResolver
    from factor_engine.planner.source_binding import discover_column_source_bindings

    store = get_store()
    logical_fields: set[str] = set()
    plans = [fp.root for fp in dag.roots] + list((dag.shared_nodes or {}).values())
    discovered = discover_column_source_bindings(plans)
    source_refs = discovered.bindings
    if discovered.source_ref_columns_seen != discovered.source_ref_columns_bound:
        raise ValueError("SourceRef column lacks a complete typed ColumnSourceBinding")
    def collect(node: PlanNode) -> None:
        if node.op == "column":
            name = str(node.attrs.get("name") or "")
            if name not in source_refs:
                logical_fields.add(name)
        for child in node.inputs:
            collect(child)
    for fp in dag.roots:
        collect(fp.root)
    for node in (dag.shared_nodes or {}).values():
        collect(node)
    if (not logical_fields and not source_refs) or "" in logical_fields:
        raise ValueError("production batch requires named source columns")
    prepared_reads: list[tuple[Any, Any]] = []
    column_authorities: dict[str, tuple[Any, str, Any, Any]] = {}

    def prepare_authority(actual_source: Any, fields: set[str]) -> tuple[Any, Any]:
        from factor_engine.storage.sources.data_access_source import DataAccessSource
        if not isinstance(actual_source, DataAccessSource):
            raise TypeError("resolved source scope is not a strict DataAccessSource")
        if not actual_source.production or not actual_source.pit_enforce:
            raise ValueError("resolved source scope is not production PIT-enforced")
        field_plans = actual_source._ensure_field_plans(sorted(fields))
        physical = sorted({column for name in fields
                           for column in (tuple(field_plans[name].physical_fields) or (name,))})
        prepared_read = store.prepare_read(
            actual_source.dataset, columns=physical,
            time_range=(actual_source.start_date, actual_source.end_date),
            instrument_filter=actual_source.instrument_filter,
            params=dict(actual_source.params), run_mode="production",
            snapshot_policy="fail_if_changed",
        )
        if _diagnostics is not None:
            _diagnostics["prepare_count"] = _diagnostics.get("prepare_count", 0) + 1
            keys = _diagnostics.setdefault("prepared_source_keys", set())
            source_key = (
                str(actual_source.dataset),
                tuple(sorted((str(k), repr(v)) for k, v in dict(actual_source.params).items())),
                str(actual_source.start_date), str(actual_source.end_date),
                tuple(actual_source.instrument_filter or ()),
            )
            keys.add(source_key)
        frozen = prepared_read.resolved_source_snapshot
        if not getattr(frozen, "content_digest", None):
            store._pipeline.release_reservation(
                getattr(prepared_read, "resource_reservation", None)
            )
            raise ValueError("batch source snapshot lacks a content digest")
        prepared_reads.append((prepared_read, store))
        if _diagnostics is not None:
            expectations = _diagnostics.setdefault("snapshot_expectations", {})
            expectation = PreparedSourceSnapshotExpectation(
                dataset=str(actual_source.dataset),
                columns=tuple(physical),
                params=tuple(sorted(dict(actual_source.params).items())),
                time_range=(actual_source.start_date, actual_source.end_date),
                instrument_filter=tuple(actual_source.instrument_filter or ()),
                content_digest=str(frozen.content_digest),
            )
            expectations[source_key] = expectation
        return prepared_read, frozen

    prepared = snapshot = None
    if logical_fields:
        prepared, snapshot = prepare_authority(source, logical_fields)
    by_secondary: dict[Any, list[Any]] = {}
    for binding in source_refs.values():
        by_secondary.setdefault(binding.source_scope, []).append(binding)
    try:
        resolver_anchor = getattr(execution_context, "data_source", None) or source
        resolver = BatchSourceResolver(
            resolver_anchor,
            market=str(getattr(execution_context, "market", "") or ""),
        )
        for source_scope, bindings in by_secondary.items():
            adapter = resolver.resolve_source(source_scope)
            if adapter is None:
                raise ValueError(f"no authoritative adapter for source scope {source_scope.key()}")
            secondary_prepared, secondary_snapshot = prepare_authority(
                adapter, {binding.field for binding in bindings}
            )
            for binding in bindings:
                column_authorities[binding.encoded_column] = (
                    adapter, binding.field, secondary_snapshot, secondary_prepared,
                )
    except Exception:
        for prepared_read, prepared_store in prepared_reads:
            prepared_store._pipeline.release_reservation(
                getattr(prepared_read, "resource_reservation", None)
            )
        raise
    decision_clock = getattr(execution_context, "decision_clock", None)
    availability_cutoff = getattr(execution_context, "availability_cutoff", None)
    binding_cache: dict[tuple[str, ...], PhysicalSourceBinding] = {}
    try:
        roots = {
            fp.factor_name: bind_plan_sources(
                fp.root, source=source, scope=fp.execution_scope,
                snapshot=snapshot, decision_clock=decision_clock, store=store,
                prepared=prepared, availability_cutoff=availability_cutoff,
                _binding_cache=binding_cache, column_authorities=column_authorities,
            ) for fp in dag.roots
        }
        shared = dict(dag.shared_nodes or {})
        if shared:
            scopes = {str(fp.execution_scope.scope_key()): fp.execution_scope for fp in dag.roots}
            if len(scopes) != 1:
                raise ValueError("shared production DAG must have one authoritative execution scope")
            scope = next(iter(scopes.values()))
            shared = {sid: bind_plan_sources(
                node, source=source, scope=scope, snapshot=snapshot, store=store,
                prepared=prepared, decision_clock=decision_clock,
                availability_cutoff=availability_cutoff, _binding_cache=binding_cache,
                column_authorities=column_authorities,
            ) for sid, node in shared.items()}
        return roots, shared
    finally:
        for prepared_read, prepared_store in prepared_reads:
            reservation = getattr(prepared_read, "resource_reservation", None)
            prepared_store._pipeline.release_reservation(reservation)


def _walk_plan_nodes(root: PlanNode):
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(node.inputs)


def _shared_dependency_closure(
    roots: list[Any], shared_nodes: dict[str, PlanNode],
) -> dict[str, PlanNode]:
    required: set[str] = set()
    pending: list[str] = []
    for fp in roots:
        for node in _walk_plan_nodes(fp.root):
            if node.op == "plan_ref":
                sid = str(node.attrs.get("sid") or "")
                if sid:
                    pending.append(sid)
    while pending:
        sid = pending.pop()
        if sid in required:
            continue
        definition = shared_nodes.get(sid)
        if definition is None:
            raise ValueError(f"surviving root has dangling shared plan_ref {sid!r}")
        required.add(sid)
        for node in _walk_plan_nodes(definition):
            if node.op == "plan_ref":
                child_sid = str(node.attrs.get("sid") or "")
                if child_sid and child_sid not in required:
                    pending.append(child_sid)
    return {sid: shared_nodes[sid] for sid in required}


def preflight_batch_sources(
    dag: Any, *, source: Any, execution_context: Any,
    forced_backend: str | None = None,
) -> BatchSourcePreflightResult:
    """Bind once while isolating root-local field and operator defects."""
    from factor_engine.planner.source_binding import discover_column_source_bindings
    from factor_engine.storage.sources.data_access_source import (
        MissingDataDependencyError,
        UnknownField,
        UnknownFieldSemanticError,
    )
    local_field_errors = (
        MissingDataDependencyError, UnknownField, UnknownFieldSemanticError,
    )

    from factor_engine.planner.batch_data_request import BatchSourceResolver

    root_errors: dict[str, RootPhysicalPreflightError] = {}
    ordinary_consumers: dict[str, set[str]] = {}
    secondary_consumers: dict[tuple[Any, str], set[str]] = {}
    for fp in dag.roots:
        try:
            discovered = discover_column_source_bindings([fp.root])
            if discovered.source_ref_columns_seen != discovered.source_ref_columns_bound:
                raise MissingDataDependencyError(
                    "SourceRef column lacks a complete typed ColumnSourceBinding"
                )
            ordinary: set[str] = set()
            for node in _walk_plan_nodes(fp.root):
                if node.op == "column":
                    name = str(node.attrs.get("name") or "")
                    if name and name not in discovered.bindings:
                        ordinary.add(name)
            for field in sorted(ordinary):
                ordinary_consumers.setdefault(field, set()).add(fp.factor_name)
            for binding in discovered.bindings.values():
                secondary_consumers.setdefault(
                    (binding.source_scope, binding.field), set()
                ).add(fp.factor_name)
        except local_field_errors as exc:
            root_errors[fp.factor_name] = RootPhysicalPreflightError(
                "UNKNOWN_FIELD", type(exc).__name__, str(exc)
            )
        except Exception as exc:
            return BatchSourcePreflightResult(
                {}, {}, root_errors,
                RootPhysicalPreflightError(
                    "SOURCE_SCOPE_UNAVAILABLE", type(exc).__name__, str(exc)
                ),
            )
    for field, consumers in ordinary_consumers.items():
        try:
            source._ensure_field_plans([field])
        except local_field_errors as exc:
            for name in consumers:
                root_errors[name] = RootPhysicalPreflightError(
                    "UNKNOWN_FIELD", type(exc).__name__, str(exc)
                )
        except Exception as exc:
            return BatchSourcePreflightResult(
                {}, {}, root_errors,
                RootPhysicalPreflightError(
                    "SOURCE_SCOPE_UNAVAILABLE", type(exc).__name__, str(exc)
                ),
            )
    resolver = BatchSourceResolver(
        getattr(execution_context, "data_source", None) or source,
        market=str(getattr(execution_context, "market", "") or ""),
    )
    adapter_cache: dict[Any, Any] = {}
    for (source_scope, field), consumers in secondary_consumers.items():
        try:
            if source_scope not in adapter_cache:
                adapter_cache[source_scope] = resolver.resolve_source(source_scope)
            adapter = adapter_cache[source_scope]
            if adapter is None:
                raise ValueError(
                    f"no authoritative adapter for source scope {source_scope.key()}"
                )
            adapter._ensure_field_plans([field])
        except local_field_errors as exc:
            for name in consumers:
                root_errors[name] = RootPhysicalPreflightError(
                    "UNKNOWN_FIELD", type(exc).__name__, str(exc)
                )
        except Exception as exc:
            return BatchSourcePreflightResult(
                {}, {}, root_errors,
                RootPhysicalPreflightError(
                    "SOURCE_SCOPE_UNAVAILABLE", type(exc).__name__, str(exc)
                ),
            )
    valid_fps = [fp for fp in dag.roots if fp.factor_name not in root_errors]
    if not valid_fps:
        return BatchSourcePreflightResult({}, {}, root_errors)
    try:
        surviving_shared = _shared_dependency_closure(
            valid_fps, dict(dag.shared_nodes or {})
        )
    except Exception as exc:
        return BatchSourcePreflightResult(
            {}, {}, root_errors,
            RootPhysicalPreflightError(
                "SOURCE_SCOPE_UNAVAILABLE", type(exc).__name__, str(exc)
            ),
        )
    filtered = SimpleNamespace(roots=tuple(valid_fps), shared_nodes=surviving_shared)
    try:
        diagnostics: dict[str, Any] = {}
        roots, shared = bind_batch_sources(
            filtered, source=source, execution_context=execution_context,
            _diagnostics=diagnostics,
        )
    except Exception as exc:
        error = RootPhysicalPreflightError(
            "SOURCE_SCOPE_UNAVAILABLE", type(exc).__name__, str(exc)
        )
        return BatchSourcePreflightResult({}, {}, root_errors, error)

    from factor_engine.runtime.multibackend.batch_global_optimizer import validate_root_physical_support
    supported: dict[str, PlanNode] = {}
    for name, root in roots.items():
        try:
            validate_root_physical_support(
                root, execution_context, forced_backend=forced_backend,
                shared_nodes=shared,
            )
            supported[name] = root
        except Exception as exc:
            from factor_engine.backend.operator_capability import UnsupportedOperatorBackendError
            if not isinstance(exc, UnsupportedOperatorBackendError):
                raise
            root_errors[name] = RootPhysicalPreflightError(
                "UNSUPPORTED_OPERATOR", type(exc).__name__, str(exc)
            )
    prepared_source_count = diagnostics.get("prepare_count", 0)
    unique_source_count = len(diagnostics.get("prepared_source_keys", set()))
    return BatchSourcePreflightResult(
        supported, shared if len(supported) == len(roots) else {}, root_errors,
        prepare_count=prepared_source_count,
        unique_source_count=unique_source_count,
        snapshot_expectations=tuple(
            diagnostics.get("snapshot_expectations", {}).values()
        ),
    )
