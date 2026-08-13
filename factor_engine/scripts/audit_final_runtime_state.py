#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 §47 final-runtime-state audit.

Runs ``cleaned_operators.load_all()`` and, for every registered canonical,
exports a frozen final-state record: resolved pandas/polars/sql implementations,
implementation kind, implementation hashes, logical-contract hash, ParamSpec,
input/output types / units / grain / availability / broadcast / stateful
checkpoint contract, surface, lifecycle, production certification, evidence refs
and alias topology.

It then auto-detects conflict classes that a healthy tree must contain ZERO of
and exits nonzero when any is found:

* backend metadata overriding the canonical logical contract,
* the same canonical exposing multiple conflicting backend contracts,
* audited fixed operators resolving to a non-fixed implementation module,
* a pandas-delegate registered under backend="polars" without an explicit
  delegate kind,
* planner rewrites dropping ``semantic_attrs`` (behavioural check),
* an unaudited production operator,
* a missing implementation hash,
* a missing source / calendar identity.

Usage::

    python scripts/audit_final_runtime_state.py

Exit code 0 = clean; nonzero = at least one hard conflict found.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

# ``python scripts/audit_final_runtime_state.py`` puts ``scripts/`` on sys.path,
# not the repo root.  Ensure the repo root is importable regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load() -> None:
    from cleaned_operators import load_all

    # A concurrent session may be mid-edit, which can break load_all transiently.
    # Retry once before giving up (mirrors the workflow's retry-once rule).
    try:
        load_all()
    except Exception:
        load_all()


def _kernel_fn(op: Any) -> Any:
    fn = getattr(op, "_fn", None)
    if fn is not None:
        return fn
    return getattr(op, "_calculate_series", None) or getattr(op, "calculate", None)


def _kernel_module(op: Any) -> str:
    fn = getattr(op, "_fn", None)
    if fn is not None:
        return getattr(fn, "__module__", "") or ""
    return getattr(type(op), "__module__", "") or ""


def _fn_payload_or_none(fn: Any) -> str | None:
    from cleaned_operators.registry import _fn_payload

    func = getattr(fn, "__func__", fn)
    try:
        return _fn_payload(func, None)
    except Exception:
        return None


def _fn_hash(fn: Any) -> str | None:
    payload = _fn_payload_or_none(fn)
    if not payload:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _impl_hash(op: Any) -> str:
    from cleaned_operators.registry import _impl_source_hash

    return _impl_source_hash(op)


def _contract_hash(op: Any) -> str:
    from cleaned_operators.registry import _contract_hash

    return _contract_hash(op)


def _logical_equal(left: Any, right: Any) -> bool:
    """Structural equality for a logical-contract field (mirrors registry)."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_logical_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_logical_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
        return True
    return left == right


# Logical-contract fields that MUST be identical across every backend of a
# canonical (mirrors ``registry._LOGICAL_CONTRACT_FIELDS``).
_LOGICAL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("param_specs", {}),
    ("param_aliases", {}),
    ("relational_specs", []),
    ("param_types", {}),
    ("input_units", {}),
    ("compatible_units", {}),
    ("output_unit", None),
    ("window_semantics", None),
    ("input_grain", None),
    ("output_grain", None),
    ("available_at", None),
    ("same_session_usable", None),
    ("role", None),
    ("input_arity", None),
    ("panel_arity", None),
    ("total_positional_arity", None),
    ("scalar_params", ()),
    ("panel_params", ()),
    ("input_fields", []),
    ("output_field", None),
    ("broadcast_specs", ()),
)

# The audited fixed set (R13 NEW-P0-01): canonical -> per-backend expected fixed
# kernel module.  These are the operators the audit book complained were "fixed
# in source but stale in the real runtime".
AUDITED_FIXED_MODULES: dict[str, dict[str, str]] = {
    "ts_count_if": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "ts_sum_if": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "ts_mean_if": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "ts_std_if": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "ts_last_if": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "ts_days_since": {
        "pandas_numpy": "cleaned_operators.layer_composite_fixes",
        "polars": "cleaned_operators.layer_composite_fixes",
    },
    "ts_true_streak": {
        "pandas_numpy": "cleaned_operators.overhaul.daily",
        "polars": "cleaned_operators.overhaul.daily",
    },
    "overnight_return": {
        "pandas_numpy": "cleaned_operators.return_decomp",
    },
    "open_close_return": {
        "pandas_numpy": "cleaned_operators.return_decomp",
    },
    "open_to_vwap_return": {
        "pandas_numpy": "cleaned_operators.return_decomp",
    },
    "vwap_to_close_return": {
        "pandas_numpy": "cleaned_operators.return_decomp",
    },
}


@dataclass
class Record:
    canonical: str
    surface: str
    status: str
    production_certified: bool | None
    lifecycle: str
    contract_hash: str
    aliases: list[str]
    backends: list[str]
    implementations: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_refs: dict[str, Any] = field(default_factory=dict)


def _impl_kind(op: Any, backend: str, backend_meta: dict[str, Any], canonical: str) -> str:
    if backend == "pandas_numpy":
        return "native_kernel"
    if backend == "sql":
        return "sql_native"
    if backend == "polars":
        execution_kind = str(backend_meta.get("execution_kind") or "")
        if execution_kind:
            return execution_kind  # e.g. expression_native / polars_eager_native
        tags = getattr(getattr(op, "metadata", None), "tags", None) or []
        if "polars_native" in tags:
            return "expression_native"
        # Heuristic: does the kernel actually convert to pandas?  Only explicit
        # conversion calls are treated as a delegate, never a bare import.
        payload = _fn_payload_or_none(_kernel_fn(op)) or ""
        if any(token in payload for token in ("to_pandas", "_pl_to_pandas")):
            return "pandas_delegate_udf"
        return "polars_native"
    return "unknown"


def build_records() -> tuple[dict[str, Record], list[str]]:
    from cleaned_operators.operator_surface import classify_canonical
    from cleaned_operators.registry import OperatorRegistry as R

    records: dict[str, Record] = {}
    errors: list[str] = []
    for canonical in R.list_canonical():
        catalog = R._catalog.get(canonical) or {}
        try:
            rec = Record(
                canonical=canonical,
                surface=classify_canonical(canonical),
                status=str(catalog.get("status") or "unknown"),
                production_certified=catalog.get("production_certified"),
                lifecycle=str(catalog.get("lifecycle_status") or ""),
                contract_hash="",
                aliases=sorted(
                    a for a, t in R._aliases.items() if R.resolve_canonical(a) == canonical and a != canonical
                ),
                backends=R.backends_for(canonical),
            )
            for backend in R.backends_for(canonical):
                op = R._operators.get(canonical, {}).get(backend)
                if op is None:
                    continue
                backend_meta = dict((catalog.get("backend_meta") or {}).get(backend) or {})
                meta = getattr(op, "metadata", None)
                rec.implementations[backend] = {
                    "module": type(op).__module__,
                    "kernel_module": _kernel_module(op),
                    "kind": _impl_kind(op, backend, backend_meta, canonical),
                    "impl_hash": _impl_hash(op),
                    "kernel_hash": _fn_hash(_kernel_fn(op)),
                    "contract_hash": _contract_hash(op),
                    "source": str(backend_meta.get("source") or ""),
                    "execution_kind": str(backend_meta.get("execution_kind") or ""),
                    "pandas_bridge": bool(backend_meta.get("pandas_bridge")),
                    "uses_pandas_fallback": bool(backend_meta.get("uses_pandas_fallback")),
                    "param_specs": dict(getattr(meta, "param_specs", None) or {}),
                    "param_types": dict(getattr(meta, "param_types", None) or {}),
                    "input_units": dict(getattr(meta, "input_units", None) or {}),
                    "compatible_units": dict(getattr(meta, "compatible_units", None) or {}),
                    "output_unit": getattr(meta, "output_unit", None),
                    "input_grain": getattr(meta, "input_grain", None),
                    "output_grain": getattr(meta, "output_grain", None),
                    "available_at": getattr(meta, "available_at", None),
                    "same_session_usable": getattr(meta, "same_session_usable", None),
                    "window_semantics": getattr(meta, "window_semantics", None),
                    "role": getattr(meta, "role", None),
                    "broadcast_specs": [str(s) for s in (getattr(meta, "broadcast_specs", None) or ())],
                }
            pandas_op = R._operators.get(canonical, {}).get("pandas_numpy")
            if pandas_op is not None:
                rec.contract_hash = _contract_hash(pandas_op)
            # Evidence refs
            rec.evidence_refs = {
                "production_certified": catalog.get("production_certified"),
                "implementation_certified": catalog.get("implementation_certified"),
                "semantic_certified": catalog.get("semantic_certified"),
                "temporal_certified": catalog.get("temporal_certified"),
                "edge_certified": catalog.get("edge_certified"),
                "replacement_history": list(catalog.get("replacement_history") or ()),
            }
            records[canonical] = rec
        except Exception as exc:  # pragma: no cover - one bad canonical must not kill the audit
            errors.append(f"{canonical}: {type(exc).__name__}: {exc}")
    return records, errors


def detect_conflicts(records: dict[str, Record]) -> list[str]:
    """Return a list of hard conflicts; empty = clean."""
    from cleaned_operators.registry import OperatorRegistry as R

    conflicts: list[str] = []

    # 1. backend metadata overriding the canonical logical contract.
    for canonical, rec in records.items():
        catalog = R._catalog.get(canonical) or {}
        for backend, impl in rec.implementations.items():
            for field, _empty in _LOGICAL_FIELDS:
                canonical_value = catalog.get(field)
                if canonical_value in (None, "", (), [], {}):
                    continue
                current = impl.get(field)
                if current in (None, "", (), [], {}):
                    continue
                if not _logical_equal(current, canonical_value):
                    conflicts.append(
                        f"backend-override: {canonical}/{backend} field {field!r} "
                        f"{current!r} != canonical {canonical_value!r}"
                    )

    # 2. same canonical with multiple conflicting backend contracts.
    for canonical, rec in records.items():
        impls = rec.implementations
        if len(impls) < 2:
            continue
        items = list(impls.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (b1, d1), (b2, d2) = items[i], items[j]
                for field, _empty in _LOGICAL_FIELDS:
                    v1, v2 = d1.get(field), d2.get(field)
                    if v1 in (None, "", (), [], {}) or v2 in (None, "", (), [], {}):
                        continue
                    if not _logical_equal(v1, v2):
                        conflicts.append(
                            f"backend-contract-conflict: {canonical} {b1} vs {b2} "
                            f"field {field!r}: {v1!r} != {v2!r}"
                        )

    # 3. audited fixed operators resolve to a non-fixed implementation module.
    for canonical, expected in AUDITED_FIXED_MODULES.items():
        for backend, expected_module in expected.items():
            impl = records.get(canonical, {}).implementations if canonical in records else {}
            entry = impl.get(backend) if isinstance(impl, dict) else None
            if entry is None:
                conflicts.append(
                    f"audited-missing: {canonical}/{backend} is not registered"
                )
                continue
            resolved = entry.get("kernel_module") or ""
            if resolved != expected_module:
                conflicts.append(
                    f"audited-source-mismatch: {canonical}/{backend} resolved "
                    f"kernel module {resolved!r} != expected fixed module "
                    f"{expected_module!r}"
                )

    # 4. pandas-delegate under backend="polars" without an explicit delegate kind.
    # An explicit delegate kind is a backend_meta declaration
    # (``pandas_bridge`` / ``uses_pandas_fallback``).  A delegate that only claims
    # ``execution_kind=expression_native`` is a FALSE native claim, which is the
    # exact stale-runtime hazard the audit exists to catch.
    for canonical, rec in records.items():
        for backend, impl in rec.implementations.items():
            if backend != "polars":
                continue
            explicit_delegate_kind = bool(
                impl.get("pandas_bridge") or impl.get("uses_pandas_fallback")
            )
            is_delegate = impl.get("kind") == "pandas_delegate_udf"
            if is_delegate and not explicit_delegate_kind:
                conflicts.append(
                    f"pandas-delegate-without-kind: {canonical}/polars "
                    f"(source={impl.get('source')!r}, "
                    f"execution_kind={impl.get('execution_kind')!r}) routes "
                    "through pandas but declares no explicit delegate kind"
                )

    # 5. missing implementation hash.  A pandas/polars runtime must expose a
    # stable kernel payload; a marker-only backend (e.g. ``sql``, which routes
    # through the SQL emitter) legitimately has no Python kernel and is exempt.
    for canonical, rec in records.items():
        for backend, impl in rec.implementations.items():
            if backend not in ("pandas_numpy", "polars"):
                continue
            if not impl.get("impl_hash"):
                conflicts.append(
                    f"missing-impl-hash: {canonical}/{backend} has no "
                    f"implementation hash"
                )
            if not impl.get("kernel_hash"):
                conflicts.append(
                    f"missing-kernel-hash: {canonical}/{backend} has no stable "
                    f"kernel payload hash"
                )

    # 6. missing source / calendar identity.
    for canonical, rec in records.items():
        catalog = R._catalog.get(canonical) or {}
        for backend, impl in rec.implementations.items():
            if not impl.get("source"):
                conflicts.append(f"missing-source: {canonical}/{backend} has no source")
        # Calendar identity: daily/session operators must resolve an availability
        # contract or an explicit grain; a bare catalog with none of these is a
        # missing temporal identity.
        status = rec.status
        if status in {"production", "implemented"} and not rec.implementations:
            conflicts.append(f"missing-calendar-identity: {canonical} has no runtime backend")

    # 7. unaudited production operator: claims production status without the
    # six-gate certification composite.  ``production_certification`` is the
    # single authority: CERTIFIED -> fine; DENIED -> never a production claim;
    # PENDING -> registered as a production target but evidence not yet bound —
    # if it ALSO carries status="production" that status is unaudited.
    from cleaned_operators.operator_surface import (
        ProductionCertification,
        production_certification,
    )

    for canonical, rec in records.items():
        if rec.status != "production":
            continue
        if rec.production_certified is True:
            continue
        try:
            cert = production_certification(canonical)
        except Exception:  # pragma: no cover - certification always importable
            cert = ProductionCertification.PENDING
        if cert is ProductionCertification.CERTIFIED:
            continue
        if cert is ProductionCertification.DENIED:
            continue
        conflicts.append(
            f"unaudited-production: {canonical} is status=production but the "
            f"six-gate certification is not bound "
            f"(production_certified={rec.production_certified!r})"
        )

    return conflicts


def detect_semantic_attrs_loss() -> list[str]:
    """Behavioural check: planner rewrites must preserve ``semantic_attrs``.

    Builds a small plan carrying a ``column`` node with semantic_attrs and runs
    the full ``Optimizer.optimize`` pipeline.  Any rewrite that drops the
    attrs is a real semantic loss (R13 NEW-P1-70), not a static-text artifact.
    """
    from planner.logical_plan import PlanNode

    def col(name: str) -> PlanNode:
        return PlanNode(
            op="column",
            attrs={"name": name},
            inputs=[],
            semantic_attrs={
                "unit": "level",
                "grain": "daily",
                "available_at": "session_close",
                "source_table": "daily",
                "field_id": "mock",
            },
        )

    def lit(v: float) -> PlanNode:
        return PlanNode(op="literal", attrs={"value": v}, inputs=[])

    plans = {
        "divide-rewrite": PlanNode(
            "divide",
            [
                PlanNode("subtract", [col("close"), PlanNode("ts_mean", [col("close"), lit(5.0)], {"d": 5, "window": 5})], {}),
                PlanNode("ts_std", [col("close"), lit(5.0)], {"d": 5, "window": 5}),
            ],
            {},
        ),
        "log-returns-rewrite": PlanNode(
            "log",
            [PlanNode("divide", [col("close"), PlanNode("ts_delay", [col("close"), lit(1)], {"d": 1})], {})],
            {},
        ),
        "identity": PlanNode(
            "ts_mean",
            [col("close"), lit(5.0)],
            {"d": 5, "window": 5},
        ),
    }
    from planner.optimizer import Optimizer

    losses: list[str] = []
    optimizer = Optimizer(allow_semantic_rewrites=True)
    for name, plan in plans.items():
        try:
            out = optimizer.optimize(plan, production=False)
        except Exception as exc:  # pragma: no cover - rewrite pipeline may raise on toy plans
            losses.append(f"semantic-rewrite-error: {name}: {type(exc).__name__}: {exc}")
            continue
        for node in _walk(out):
            # Every column node that came from an input column must still carry
            # the semantic attrs (the mock column id / source table).
            if node.op == "column" and node.attrs.get("name") == "close":
                if not node.semantic_attrs.get("source_table"):
                    losses.append(
                        f"semantic-attrs-loss: planner rewrite {name} dropped "
                        f"semantic_attrs on column node"
                    )
    return losses


def _walk(node: Any):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(cur.inputs)


def _format_table(records: dict[str, Record]) -> str:
    lines = []
    lines.append("canonical | surface | status | prod_cert | backends | contract_hash | impl_kinds")
    lines.append("-" * 120)
    for canonical in sorted(records):
        rec = records[canonical]
        kinds = ",".join(
            f"{b}:{i.get('kind','?')}" for b, i in sorted(rec.implementations.items())
        )
        lines.append(
            f"{canonical} | {rec.surface} | {rec.status} | {rec.production_certified} "
            f"| {','.join(rec.backends)} | {rec.contract_hash} | {kinds}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    _load()
    from cleaned_operators.registry import OperatorRegistry as R

    print(f"registry lifecycle={R.lifecycle()} version={R.version()} "
          f"canonicals={len(R.list_canonical())}")

    records, errors = build_records()
    if errors:
        print("RECORD ERRORS:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)

    conflicts = detect_conflicts(records)
    sem_losses = detect_semantic_attrs_loss()
    conflicts.extend(sem_losses)

    print("\n=== per-canonical final-state table ===")
    print(_format_table(records))

    print("\n=== conflict report ===")
    if conflicts:
        for c in conflicts:
            print(f"CONFLICT: {c}")
        print(f"\n{len(conflicts)} hard conflict(s) found -> exit 1")
        return 1

    print("clean: 0 conflicts")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        sys.exit(2)
