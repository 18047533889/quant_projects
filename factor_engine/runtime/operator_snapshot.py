"""Runtime-derived operator identity, contracts, capabilities, and evidence ledger.

This module is deliberately a *view* over ``OperatorRegistry``.  It does not
register operators, certify names, or maintain a second capability authority.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "factor_engine.runtime_operator_snapshot.v4"
EVIDENCE_STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN", "UNSUPPORTED", "STALE"})
_SNAPSHOT_CACHE: dict[str, dict[str, Any]] = {}
_SNAPSHOT_CACHE_LIMIT = 2
_TRUSTED_EVIDENCE_TOKEN = object()
_SNAPSHOT_CACHE_LOCK = threading.RLock()


def _json(value: Any) -> Any:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "value"):
        return _json(value.value)
    if is_dataclass(value):
        return {k: _json(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_json(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if value.__class__.__name__ == "_MissingDefaultType":
        return {"declared": False}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bound_kernel_identities(op: Any, file_digests: dict[str, str | None]) -> list[dict[str, Any]]:
    """Bind generic wrappers to the authored kernels they actually invoke."""
    candidates = [getattr(op, "_contract_callable", None), getattr(op, "_fn", None)]
    wrapper = getattr(op, "_calculate_series", None)
    if wrapper is not None:
        try:
            parameter = inspect.signature(wrapper).parameters.get("_fn")
            if parameter is not None:
                candidates.append(parameter.default)
        except (TypeError, ValueError):
            pass
    result = []
    seen = set()
    for fn in candidates:
        if not callable(fn) or id(fn) in seen:
            continue
        seen.add(id(fn))
        try:
            source = inspect.getsourcefile(fn) or inspect.getfile(fn)
        except (TypeError, OSError):
            source = None
        path = str(Path(source).resolve()) if source else None
        if path is not None and path not in file_digests:
            try:
                file_digests[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                file_digests[path] = None
        result.append({
            "qualified_name": f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', type(fn).__qualname__)}",
            "source_file": path,
            "file_digest": file_digests.get(path),
        })
    # _contract_callable and the transport _fn can refer to the same function
    # through different bound-method objects; normalize the evidence identity.
    return sorted({ _digest(item): item for item in result }.values(), key=_digest)


def _implementation_identity(op: Any, file_digests: dict[str, str | None] | None = None) -> dict[str, Any]:
    cls = type(op)
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    try:
        source_file = inspect.getsourcefile(cls) or inspect.getfile(cls)
    except (TypeError, OSError):
        source_file = None
    file_digest = None
    if source_file:
        path = Path(source_file).resolve()
        key = str(path)
        if file_digests is not None and key in file_digests:
            file_digest = file_digests[key]
        else:
            try:
                file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                source_file = str(path)
            if file_digests is not None:
                file_digests[key] = file_digest
    # The loaded module file is the immutable source unit shipped in the wheel.
    # Avoid re-reading/parsing every class body (thousands of bindings can share
    # one file); qualified name + whole-file digest still invalidates any edit.
    source_digest = file_digest
    kernels = _bound_kernel_identities(op, file_digests if file_digests is not None else {})
    return {
        "qualified_name": qualified,
        "source_file": str(Path(source_file).resolve()) if source_file else None,
        "file_digest": file_digest,
        "source_digest": source_digest,
        "bound_kernels": kernels,
        "implementation_digest": _digest({
            "qualified_name": qualified,
            "file_digest": file_digest,
            "source_digest": source_digest,
            "bound_kernels": kernels,
        }),
    }


def _declared_parameter_schema(ps: Any) -> tuple[dict[str,Any],bool]:
    """JSON Schema for an explicitly authored scalar/sequence/union contract."""
    item={"x-factor-engine-role":_json(getattr(ps,"param_role",None)),
          "x-searchable":bool(getattr(ps,"searchable",True))}
    alternatives=getattr(ps,"alternatives",None)
    child=getattr(ps,"items",None)
    verified=True
    if alternatives is not None:
        branches=[_declared_parameter_schema(branch) for branch in alternatives]
        item["anyOf"]=[schema for schema,_ in branches]
        verified=all(ok for _,ok in branches)
    elif child is not None and getattr(ps,"dtype",None) in (tuple,list):
        child_schema,verified=_declared_parameter_schema(child)
        item.update(type="array",items=child_schema)
        if ps.min_items is not None:
            item["minItems"]=ps.min_items
        if ps.max_items is not None:
            item["maxItems"]=ps.max_items
    else:
        from factor_engine.runtime.session_calendar import SessionCalendar
        from factor_engine.cleaned_operators.common.static_adjacency import StaticAdjacency
        typemap={
            int:"integer", float:"number", str:"string", bool:"boolean",
            SessionCalendar:"object",
            StaticAdjacency:"object",
        }
        dtype=getattr(ps,"dtype",None)
        if dtype in typemap:
            item["type"]=typemap[dtype]
            if dtype is StaticAdjacency:
                item["x-factor-engine-object"]="static_adjacency"
                item["x-factor-engine-axis-semantics"]="source_symbol_x_destination_symbol"
        else:
            verified=False
    if getattr(ps,"min",None) is not None:
        item["minimum"]=ps.min
    if getattr(ps,"max",None) is not None:
        item["maximum"]=ps.max
    if getattr(ps,"choices",None) is not None:
        item["enum"]=_json(ps.choices)
    if getattr(ps,"default",object()) is None:
        if "type" in item:
            item["type"]=[item["type"],"null"]
        elif "anyOf" in item:
            item["anyOf"].append({"type":"null"})
        if "enum" in item and None not in item["enum"]:
            item["enum"].append(None)
    if not verified:
        item["x-factor-engine-verification"]="unknown"
    return item,verified


def _parameter_contract(op: Any, panel_params: Iterable[str]) -> tuple[dict[str, Any], str, bool]:
    meta = getattr(op, "metadata", None)
    names = list(getattr(meta, "param_names", None) or ())
    specs = getattr(meta, "param_specs", None) or {}
    relations = getattr(meta, "relational_specs", None) or ()
    panel = set(panel_params)
    mixed = set(getattr(meta, "mixed_params", None) or ())
    nullable_mixed = set(getattr(meta, "nullable_mixed_params", None) or ())
    from factor_engine.cleaned_operators.operator_spec import _operator_contract_signature
    signature = _operator_contract_signature(op)
    kernel_defaults = {
        name: parameter.default
        for name, parameter in (signature.parameters.items() if signature else ())
        if parameter.default is not inspect.Parameter.empty
    }
    properties: dict[str, Any] = {}
    required: list[str] = []
    verified = True
    for name in names:
        if name in mixed:
            branches = [{"type": "array"}, {"type": "number"}]
            if name in nullable_mixed:
                branches.append({"type": "null"})
            properties[name] = {
                "x-factor-engine-role": "panel_or_scalar",
                "anyOf": branches,
            }
            if name not in kernel_defaults:
                required.append(name)
            else:
                properties[name]["default"] = _json(kernel_defaults[name])
            continue
        if name in panel:
            properties[name] = {"x-factor-engine-role": "panel", "type": "array"}
            if name not in kernel_defaults:
                required.append(name)
            elif kernel_defaults[name] is None:
                properties[name]["type"] = ["array", "null"]
                properties[name]["default"] = None
            continue
        ps = specs.get(name)
        if ps is None:
            properties[name] = {"x-factor-engine-verification": "unknown"}
            verified = False
            continue
        item,item_verified = _declared_parameter_schema(ps)
        verified = verified and item_verified
        default = getattr(ps, "default", None)
        if default.__class__.__name__ == "_MissingDefaultType":
            if name not in kernel_defaults:
                required.append(name)
            elif kernel_defaults[name] is not None:
                item["default"] = _json(kernel_defaults[name])
            # An omitted kernel default of None need not mean explicit None
            # is admitted by ParamSpec. Do not invent a nullable type here.
        else:
            item["default"] = _json(default)
            if default is None and "type" in item and not isinstance(item["type"],list):
                item["type"] = [item["type"], "null"]
                if "enum" in item and None not in item["enum"]:
                    item["enum"].append(None)
        properties[name] = item
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(set(required)),
        "allOf": [{"x-factor-engine-relation": str(r.expression)} for r in relations],
        "x-factor-engine-contract-status": "declared" if verified else "unknown",
    }
    variadic_mixed = getattr(meta, "variadic_mixed_param", None)
    if variadic_mixed:
        schema["x-factor-engine-variadic"] = {
            "name": variadic_mixed,
            "role": "panel_or_scalar",
            "anyOf": [{"type": "array"}, {"type": "number"}, {"type": "null"}],
        }
    return schema, _digest(schema), verified


@dataclass(frozen=True)
class EvidenceRecord:
    canonical: str
    backend: str
    semantic_version: int
    implementation_digest: str
    parameter_schema_digest: str
    status: str = "NOT_RUN"
    oracle_kind: str | None = None
    test_node_id: str | None = None
    artifact_reference: str | None = None
    fixture_digest: str | None = None
    dependency_lock_digest: str | None = None
    runtime_fallbacks: tuple[str, ...] = ()


def _evidence_for(identity: dict[str, Any], supplied: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    key = (identity["canonical"], identity["backend"])
    prior = dict(supplied.get(key, supplied.get((identity["canonical"], "*"), {})))
    status = str(prior.get("status", "NOT_RUN")).upper()
    if status not in EVIDENCE_STATUSES:
        status = "NOT_RUN"
    bound = ("semantic_version", "implementation_digest", "parameter_schema_digest")
    if prior and status in {"PASS", "FAIL"} and any(prior.get(k) != identity[k] for k in bound):
        status = "STALE"
    # PASS is only accepted from a concrete linked test artifact.
    if status == "PASS" and not (
        prior.get("_trusted_provenance") is _TRUSTED_EVIDENCE_TOKEN
        and prior.get("test_node_id") and prior.get("artifact_reference")
        and len(str(prior.get("artifact_digest") or "")) == 64
    ):
        status = "NOT_RUN"
    return {
        **{k: prior.get(k) for k in (
            "dependency_lock_digest", "fixture_digest", "test_node_id", "oracle_kind",
            "runtime_fallbacks", "artifact_reference", "artifact_digest")},
        **{k: identity[k] for k in ("canonical", "backend", *bound)},
        "status": status,
    }


def _native_accelerated(backends: Iterable[Mapping[str, Any]], passing: set[str]) -> bool:
    native_kinds = {"polars_native_expr", "polars_numpy_kernel", "duckdb_native_sql"}
    return any(
        b.get("backend") in passing and b.get("physical_execution_kind") in native_kinds
        and b.get("evidence", {}).get("runtime_fallbacks") == [] for b in backends
    )


def _research_callable(backends: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        b.get("research_supported") is True and b.get("contract_status") == "declared"
        for b in backends
    )


def build_runtime_operator_snapshot(
    *, evidence_records: Iterable[Mapping[str, Any]] = (), profile: str = "runtime"
) -> dict[str, Any]:
    """Export every loaded canonical/backend plus aliases and explicit evidence state."""
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.backend.operator_semantic_version import semantic_version
    from factor_engine.cleaned_operators.operator_spec import build_operator_spec
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.planner.composite_lowering import infer_execution_kind

    ensure_cleaned_loaded()
    operators, aliases, catalog = OperatorRegistry._read_state()
    supplied = {
        (str(r.get("canonical")), str(r.get("backend"))): r for r in evidence_records
    }
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    implementation_cache: dict[int, dict[str, Any]] = {}
    file_digest_cache: dict[str, str | None] = {}
    for canonical in sorted(set(operators) | set(catalog)):
        impls = operators.get(canonical, {})
        alias_names = sorted(a for a, target in aliases.items() if target == canonical)
        if not impls:
            rows.append({
                "canonical": canonical, "aliases": alias_names, "profile": profile,
                "runtime_bound": False, "capabilities": {"discoverable": True,
                "research_registered": False, "research_callable": False,
                "production_callable": False,
                "native_accelerated": False}, "execution_kind": "unknown",
                "contract_status": "unknown", "backends": [],
                "implementation_available": False,
                "research_callable": False,
                "execution_state": "NO_IMPLEMENTATION",
                "assurance": "UNVERIFIED",
                "production_callable": False,
                "blockers": ["NO_IMPLEMENTATION"],
                "warnings": [],
            })
            continue
        # OperatorSpec owns panel/scalar topology, including its signature-based
        # fallback when legacy metadata does not declare ``panel_params``.
        spec = build_operator_spec(canonical)
        representative = next(iter(impls.values()))
        representative_meta = getattr(representative, "metadata", None)
        panel_params = tuple(getattr(spec, "panel_params", ()) or ())
        backends: list[dict[str, Any]] = []
        for backend, op in sorted(impls.items()):
            impl = implementation_cache.get(id(op))
            if impl is None:
                impl = _implementation_identity(op, file_digest_cache)
                implementation_cache[id(op)] = impl
            schema, schema_digest, schema_verified = _parameter_contract(
                op, panel_params
            )
            if spec is None:
                # Missing topology authority must not turn an otherwise empty
                # backend schema into a declared contract.
                schema["x-factor-engine-contract-status"] = "unknown"
                schema_digest, schema_verified = _digest(schema), False
            identity = {
                "canonical": canonical, "backend": backend,
                "semantic_version": semantic_version(canonical),
                "parameter_schema_digest": schema_digest, **impl,
            }
            evidence = _evidence_for(identity, supplied)
            ledger.append(evidence)
            from factor_engine.backend.operator_capability import supports_pandas, supports_polars
            if backend in {"pandas", "pandas_numpy"}:
                research_supported = supports_pandas(canonical, mode="research")
            elif backend in {"polars", "polars_panel", "polars_long"}:
                research_supported = supports_polars(canonical, mode="research")
            else:
                research_supported = False
            from factor_engine.backend.polars_backend_kind import get_physical_spec
            physical = get_physical_spec(op)
            physical_kind = str(getattr(getattr(physical, "execution_kind", None), "value", "unsupported"))
            backends.append({**identity, "parameter_schema": schema,
                             "contract_status": "declared" if schema_verified else "unknown",
                             "research_supported": bool(research_supported),
                             "physical_execution_kind": physical_kind,
                             "evidence": evidence})
        passing = {b["backend"] for b in backends if b["evidence"]["status"] == "PASS"}
        native = _native_accelerated(backends, passing)
        research_callable = _research_callable(backends)
        evidence_statuses = {b["evidence"]["status"] for b in backends}
        blockers = [] if research_callable else [
            "INPUT_CONTRACT_MISMATCH"
            if any(b["research_supported"] for b in backends)
            else "NO_IMPLEMENTATION"
        ]
        warnings = [] if evidence_statuses == {"PASS"} else [
            "NUMERICAL_CERTIFICATION_PENDING"
        ]
        rows.append({
            "canonical": canonical, "aliases": alias_names, "profile": profile,
            "runtime_bound": True,
            "execution_kind": infer_execution_kind(canonical),
            "status": str(catalog.get(canonical, {}).get("status") or "research"),
            "frequency": str(
                catalog.get(canonical, {}).get("output_grain")
                or getattr(representative_meta, "output_grain", None)
                or catalog.get(canonical, {}).get("input_grain")
                or getattr(representative_meta, "input_grain", None) or "any"
            ),
            "markets": sorted(str(x) for x in (catalog.get(canonical, {}).get("markets") or ())),
            "data_capabilities": sorted(str(x) for x in (catalog.get(canonical, {}).get("data_capabilities") or ())),
            "budget_class": catalog.get(canonical, {}).get("budget_class"),
            "contract_status": "declared" if all(b["contract_status"] == "declared" for b in backends) else "unknown",
            "implementation_available": True,
            "research_callable": research_callable,
            "execution_state": (
                "EXECUTABLE_VERIFIED" if research_callable and evidence_statuses == {"PASS"}
                else "EXECUTABLE_UNVERIFIED" if research_callable
                else "BLOCKED"
            ),
            "assurance": "VERIFIED" if evidence_statuses == {"PASS"} else "UNVERIFIED",
            "production_callable": bool(getattr(spec, "allow_in_production", False) and passing),
            "blockers": blockers,
            "warnings": warnings,
            "capabilities": {
                "discoverable": True,
                "research_registered": True,
                "research_callable": research_callable,
                "production_callable": bool(getattr(spec, "allow_in_production", False) and passing),
                "native_accelerated": native,
            },
            "backends": backends,
        })
    alias_rows = [{"alias": a, "canonical": OperatorRegistry.resolve_canonical(a)} for a in sorted(aliases)]
    import factor_engine.factor_recipes  # noqa: F401 - populate public recipe registry
    from factor_engine.factor_recipes.registry import FactorRecipeRegistry
    recipe_rows = [
        {"name": name, **payload}
        for name, payload in FactorRecipeRegistry.catalog().items()
    ]
    body = {"schema_version": SCHEMA_VERSION, "profile": profile,
            "registry_version": getattr(OperatorRegistry, "_version", None),
            "operators": rows, "aliases": alias_rows, "recipes": recipe_rows,
            "evidence_ledger": ledger}
    body["catalog_digest"] = _digest(body)
    body["generated_at"] = datetime.now(timezone.utc).isoformat()
    body["counts"] = {
        "canonical": len(rows), "aliases": len(alias_rows),
        "public_recipes": len(recipe_rows), "runtime_bindings": len(ledger),
        "implementation_available": sum(bool(r["implementation_available"]) for r in rows),
        "research_callable": sum(bool(r["research_callable"]) for r in rows),
        "executable_unverified": sum(r["execution_state"] == "EXECUTABLE_UNVERIFIED" for r in rows),
        "known_failed_bindings": sum(
            b["evidence"]["status"] == "FAIL" for r in rows for b in r["backends"]
        ),
        "production_callable": sum(bool(r["production_callable"]) for r in rows),
    }
    return body


def _live_runtime_digest() -> str:
    """Digest loaded bindings and contracts so source/metadata edits bust cache."""
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.backend.operator_semantic_version import semantic_version
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    operators, aliases, _catalog = OperatorRegistry._read_state()
    files: dict[str, str | None] = {}
    bindings = []
    for canonical, impls in sorted(operators.items()):
        for backend, op in sorted(impls.items()):
            impl = _implementation_identity(op, files)
            meta = getattr(op, "metadata", None)
            panel_params = tuple(getattr(meta, "panel_params", None) or ())
            _schema, schema_digest, _verified = _parameter_contract(op, panel_params)
            bindings.append((canonical, backend, impl["implementation_digest"],
                             schema_digest, semantic_version(canonical)))
    return _digest({"registry_version": getattr(OperatorRegistry, "_version", None),
                    "aliases": dict(aliases), "bindings": bindings})


def get_cached_runtime_operator_snapshot(
    *, evidence_records: Iterable[Mapping[str, Any]] = (), profile: str = "runtime"
) -> dict[str, Any]:
    """Return a bounded cached snapshot, invalidated by actual code/schema identity."""
    records = [dict(r) for r in evidence_records]
    with _SNAPSHOT_CACHE_LOCK:
        evidence_key = [
            {**{k: v for k, v in r.items() if k != "_trusted_provenance"},
             "trusted_provenance": r.get("_trusted_provenance") is _TRUSTED_EVIDENCE_TOKEN}
            for r in records
        ]
        key = _digest({"runtime": _live_runtime_digest(), "profile": profile, "evidence": evidence_key})
        cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None:
            return cached
        snapshot = build_runtime_operator_snapshot(evidence_records=records, profile=profile)
        _SNAPSHOT_CACHE[key] = snapshot
        while len(_SNAPSHOT_CACHE) > _SNAPSHOT_CACHE_LIMIT:
            _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)))
        return snapshot


def query_runtime_operator_snapshot(
    snapshot: Mapping[str, Any], *, actor: str = "discoverable",
    backend: str | None = None, frequency: str | None = None,
    execution_kind: str | None = None, market: str | None = None,
    data_capability: str | None = None, budget_class: str | None = None,
    canonical: str | None = None, offset: int = 0, limit: int = 100,
) -> dict[str, Any]:
    """Filter/paginate operator descriptions; unknown context never matches.

    ``research_registered`` means an actual runtime binding exists;
    ``research_callable`` additionally requires existing backend support and a
    fully declared typed contract. ``production_callable`` and
    ``native_accelerated`` require linked PASS evidence.
    """
    if actor not in {"discoverable", "research_registered", "research_callable", "production_callable", "native_accelerated"}:
        raise ValueError(f"unknown actor/capability {actor!r}")
    if offset < 0 or limit < 1 or limit > 500:
        raise ValueError("offset must be >= 0 and limit must be between 1 and 500")
    def actor_matches(row: Mapping[str, Any]) -> bool:
        selected = [b for b in row.get("backends", ()) if backend is None or b.get("backend") == backend]
        if actor == "discoverable":
            return True if backend is None else bool(selected)
        if actor == "research_registered":
            return bool(selected)
        if actor == "research_callable":
            return any(b.get("research_supported") is True and b.get("contract_status") == "declared" for b in selected)
        if actor == "production_callable":
            return bool(row.get("capabilities", {}).get(actor, False)) and any(
                b.get("evidence", {}).get("status") == "PASS" for b in selected
            )
        native_kinds = {"polars_native_expr", "polars_numpy_kernel", "duckdb_native_sql"}
        return any(
            b.get("physical_execution_kind") in native_kinds
            and b.get("evidence", {}).get("status") == "PASS"
            and b.get("evidence", {}).get("runtime_fallbacks") == [] for b in selected
        )

    matches = []
    for row in snapshot.get("operators", ()):
        if canonical is not None and row.get("canonical") != canonical:
            continue
        if not actor_matches(row):
            continue
        if frequency and row.get("frequency") != frequency:
            continue
        if execution_kind and row.get("execution_kind") != execution_kind:
            continue
        if market and market not in (row.get("markets") or ()):
            continue
        if data_capability and data_capability not in (row.get("data_capabilities") or ()):
            continue
        if budget_class and row.get("budget_class") != budget_class:
            continue
        matches.append(row)
    return {
        "schema_version": snapshot.get("schema_version"),
        "catalog_digest": snapshot.get("catalog_digest"),
        "actor": actor, "canonical": canonical,
        "total_matches": len(matches), "offset": offset,
        "limit": limit, "items": matches[offset:offset + limit],
    }


def load_evidence_records(
    path: Path, *, trusted_artifact_digest: str | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("evidence_ledger", []))
    # Compatibility ingestion for the 2026-09-06 intraday stale-binding
    # artifact.  It is a block record, not a test run, and therefore expands to
    # wildcard-backend STALE rows; it can never manufacture PASS.
    if not records and isinstance(payload.get("affected"), list):
        records = [
            {"canonical": str(row["canonical"]), "backend": "*", "status": "STALE",
             "artifact_reference": str(path)}
            for row in payload["affected"]
            if isinstance(row, Mapping) and row.get("canonical")
        ]
    if not isinstance(records, list):
        raise ValueError("evidence artifact records must be a list")
    out = [dict(r) for r in records if isinstance(r, Mapping)]
    artifact_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    trusted = (
        payload.get("schema_version") == "factor_engine.operator_test_run.v3"
        and trusted_artifact_digest == artifact_digest
    )
    for row in out:
        if trusted:
            row["_trusted_provenance"] = _TRUSTED_EVIDENCE_TOKEN
        row.setdefault("artifact_digest", artifact_digest)
    return out
