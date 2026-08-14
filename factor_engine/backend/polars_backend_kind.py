# -*- coding: utf-8 -*-
"""Explicit backend-kind taxonomy for "polars" registry slots (audit R13 P0-63).

`load_all()` ends with :func:`cleaned_operators.polars_gap_coverage.register_polaris_gap_coverage`,
which wraps nearly every remaining pandas-only canonical as a `backend="polars"`
slot via :func:`cleaned_operators.rolling_pack.register_polars_udf`.  Those slots
are *not* native Polars at all — every call round-trips Polars → Pandas → certified
pandas kernel → Polars.  Treating them as "polars supported" on the backend name
alone pollutes coverage / capability / router decisions.

This module is the single authority for classifying a registered implementation.
Kinds:

* ``polars_native``               — real Polars expressions or per-column numpy
                                    UDF over Polars columns (no pandas round-trip);
* ``polars_udf_pandas_delegate``  — pl → pandas → pandas reference → pl delegation
                                    (gap coverage / ``register_polars_udf``);
* ``pandas_reference``            — the certified ``pandas_numpy`` reference;
* ``sql_native``                  — DuckDB / ClickHouse SQL lowering;
* ``unsupported``                 — no implementation of that kind.

The router / capability / coverage reports must read this kind instead of
``backend == "polars"``.
**ExecutionKind Declaration (2026-08-13 explicit capability contract)**:

Operators should declare their execution characteristics via
:class:`PhysicalImplementationSpec` rather than relying on source inspection.
The classification logic falls back to heuristics with a warning when the
declaration is missing, but explicit declaration is the authoritative path forward.
"""
from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

# Import from unified contract authority (FE-P0-003, FE-P0-004)
from backend.contracts import ExecutionKind, PhysicalImplementationSpec




class BackendKind(str, Enum):
    """Explicit backend-kind classification (legacy compatibility)."""

    POLARS_NATIVE = "polars_native"
    POLARS_UDF_PANDAS_DELEGATE = "polars_udf_pandas_delegate"
    PANDAS_REFERENCE = "pandas_reference"
    SQL_NATIVE = "sql_native"
    UNSUPPORTED = "unsupported"



#: Source markers whose ``polars`` backend is a Pandas materialisation round-trip.
PANDAS_DELEGATE_SOURCES: frozenset[str] = frozenset({
    "polars_udf",             # register_polars_udf (gap coverage)
    "pandas_bridge",          # register_polars_bridge
    "polars_geometry_math",   # polars I/O around the same numpy kernels
    "factor_dsl_polars_bridge",
    "polars_misc_utils",
})

#: Module-level bridges that round-trip through ``to_pandas()``.  Kept minimal:
#: ``cleaned_operators.common.polars_ops`` also hosts GENUINE native expression
#: operators (e.g. ``AbsPolars``), so module membership alone must not imply a
#: delegate — the kernel-body inspection decides the rest.
PANDAS_DELEGATE_MODULES: frozenset[str] = frozenset({
    "cleaned_operators.rolling_pack",          # register_polars_udf / register_polars_bridge
})


def _module_name(operator: Any) -> str:
    cls = getattr(operator, "__class__", None)
    return str(getattr(cls, "__module__", "") or "") if cls is not None else ""


def _source_is_delegate(source: str) -> bool:
    return str(source or "").lower() in PANDAS_DELEGATE_SOURCES


def _module_is_delegate(operator: Any, module: str = "") -> bool:
    mod = module or _module_name(operator)
    return any(mod.startswith(m) for m in PANDAS_DELEGATE_MODULES)


def _kernel_is_delegate(operator: Any) -> bool:
    """A real implementation builds Polars expressions/columns; a delegate ends
    by round-tripping through pandas (``to_pandas()`` / ``_pl_to_pd`` and
    ``_pl_rebuild``).  Inspects both the framework ``_calculate_series`` and the
    concrete ``calc`` method so polars_ops-style native operators are not
    mislabelled by class module alone."""
    for method_name in ("_calculate_series", "calc"):
        kernel = getattr(operator, method_name, None)
        if kernel is None:
            continue
        import inspect

        try:
            body = inspect.getsource(kernel)
        except (OSError, TypeError):
            body = ""
        if any(
            tok in body
            for tok in ("to_pandas()", ".to_pandas(", "_pl_to_pd", "_pl_rebuild")
        ):
            return True
    return False


def get_physical_spec(operator: Any) -> PhysicalImplementationSpec | None:
    """Extract PhysicalImplementationSpec from an operator if declared.

    Checks for:
    1. ``operator._physical_spec`` class/instance attribute
    2. ``operator.physical_spec()`` method

    Returns None if no explicit declaration found.
    """
    # Check for _physical_spec attribute
    spec = getattr(operator, "_physical_spec", None)
    if isinstance(spec, PhysicalImplementationSpec):
        return spec

    # Check for physical_spec() method
    method = getattr(operator, "physical_spec", None)
    if callable(method):
        try:
            result = method()
            if isinstance(result, PhysicalImplementationSpec):
                return result
        except Exception:
            pass

    return None


def polars_backend_kind(
    operator: Any,
    *,
    source: str = "",
    module: str = "",
    production_mode: bool = True,  # FE-P0-005: Default to fail-closed
) -> BackendKind:
    """Classify a registered ``polars`` implementation.

    Returns one of the :class:`BackendKind` values.  ``source`` / ``module`` are
    the registry ``backend_meta["source"]`` and the operator class module; when
    they are known the classification is exact without source inspection.

    **FE-P0-005: Production classification requires explicit validated spec**.
    Defaults to production_mode=True (fail closed). Set production_mode=False
    explicitly for research/diagnostic heuristic classification.

    **Explicit declaration preferred**: Operators should declare
    :class:`PhysicalImplementationSpec` via ``_physical_spec`` attribute or
    ``physical_spec()`` method. Source inspection is only for research diagnostics.
    """
    # Try explicit declaration first
    spec = get_physical_spec(operator)
    if spec is not None:
        # Map ExecutionKind to BackendKind (legacy compatibility)
        if spec.execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE:
            return BackendKind.POLARS_UDF_PANDAS_DELEGATE
        elif spec.execution_kind in {
            ExecutionKind.POLARS_NATIVE_EXPR,
            ExecutionKind.POLARS_NUMPY_KERNEL,
        }:
            return BackendKind.POLARS_NATIVE
        elif spec.execution_kind == ExecutionKind.PANDAS_REFERENCE:
            return BackendKind.PANDAS_REFERENCE
        elif spec.execution_kind in {
            ExecutionKind.DUCKDB_NATIVE_SQL,
            ExecutionKind.SQL_PYTHON_UDF,
        }:
            return BackendKind.SQL_NATIVE
        else:
            return BackendKind.UNSUPPORTED

    # FE-P0-005: Production requires explicit spec; fail closed
    if production_mode:
        canonical = getattr(operator, "canonical", "unknown")
        warnings.warn(
            f"Production mode: Operator {canonical} (backend=polars) lacks explicit "
            f"PhysicalImplementationSpec; classification=UNSUPPORTED. "
            f"Add _physical_spec attribute or physical_spec() method for production eligibility.",
            stacklevel=2,
        )
        return BackendKind.UNSUPPORTED

    # Research/diagnostic mode: fallback to heuristics with warning
    canonical = getattr(operator, "canonical", "unknown")
    warnings.warn(
        f"Research mode: Operator {canonical} (backend=polars) lacks explicit PhysicalImplementationSpec; "
        f"falling back to source inspection heuristics. "
        f"Add _physical_spec attribute for authoritative classification.",
        stacklevel=2,
    )

    if _source_is_delegate(source):
        return BackendKind.POLARS_UDF_PANDAS_DELEGATE
    if _module_is_delegate(operator, module=module):
        return BackendKind.POLARS_UDF_PANDAS_DELEGATE
    if _kernel_is_delegate(operator):
        return BackendKind.POLARS_UDF_PANDAS_DELEGATE
    if getattr(operator, "_calculate_series", None) is not None:
        # Genuine polars implementation (expression or per-column numpy UDF).
        return BackendKind.POLARS_NATIVE
    return BackendKind.POLARS_NATIVE


def canonical_polars_kind(canonical: str, *, production_mode: bool = True) -> BackendKind:
    """Return the backend-kind of a canonical's registered ``polars`` slot.

    FE-P0-005: Defaults to production_mode=True (fail closed).
    """
    try:
        from cleaned_operators.registry import OperatorRegistry
    except Exception:  # pragma: no cover - registry unavailable
        return BackendKind.UNSUPPORTED
    if "polars" not in OperatorRegistry.backends_for(canonical):
        return BackendKind.UNSUPPORTED
    op = OperatorRegistry.get(canonical, "polars")
    if op is None:
        return BackendKind.UNSUPPORTED
    meta = (
        (OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta") or {}
    ).get("polars", {}) or {}
    source = str(meta.get("source") or "")
    return polars_backend_kind(op, source=source, production_mode=production_mode)


def canonical_polars_is_delegate(canonical: str, *, production_mode: bool = True) -> bool:
    """True when the canonical's polars slot is a pandas-delegating UDF.

    FE-P0-005: Defaults to production_mode=True (fail closed).
    """
    return canonical_polars_kind(canonical, production_mode=production_mode) == BackendKind.POLARS_UDF_PANDAS_DELEGATE


def capability_quality(canonical: str, backend: str) -> str:
    """Capability-quality label for a canonical × backend slot (audit R13 P1-84).

    Replaces the boolean "Pandas/Polars/SQL supported" slot report with a quality
    classification:

    * ``pandas_native``            — certified pandas reference;
    * ``polars_native_expression`` — real Polars expression path;
    * ``polars_native_kernel``     — per-column numpy UDF over Polars columns;
    * ``polars_pandas_delegate``   — pl→pandas→pl delegation (gap coverage);
    * ``duckdb_native_sql`` / ``clickhouse_native_sql``;
    * ``unsupported``.

    A delegate-only polars slot is reported as ``polars_pandas_delegate``, never
    as a native "Polars supported" boolean.

    **Explicit declaration preferred**: Uses PhysicalImplementationSpec when
    available; falls back to source inspection with warning otherwise.
    """
    b = str(backend or "").lower()
    if b == "pandas_numpy":
        return "pandas_native"
    if b in {"polars", "polars_panel"}:
        kind = canonical_polars_kind(canonical)
        if kind == BackendKind.POLARS_UDF_PANDAS_DELEGATE:
            return "polars_pandas_delegate"

        # Try explicit spec first
        try:
            from cleaned_operators.registry import OperatorRegistry
            op = OperatorRegistry.get(canonical, "polars")
            if op is not None:
                spec = get_physical_spec(op)
                if spec is not None:
                    if spec.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR:
                        return "polars_native_expression"
                    elif spec.execution_kind == ExecutionKind.POLARS_NUMPY_KERNEL:
                        return "polars_native_kernel"
                    elif spec.execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE:
                        return "polars_pandas_delegate"

                # Fallback: distinguish expression vs per-column numpy kernel by source inspection
                kernel = getattr(op, "_calculate_series", None)
                if kernel is not None:
                    import inspect

                    try:
                        body = inspect.getsource(kernel)
                    except (OSError, TypeError):
                        body = ""
                    if "pl." in body or "import polars" in body:
                        return "polars_native_expression"
        except Exception:
            pass
        return "polars_native_kernel"
    if b in {"duckdb_sql", "sql", "duckdb"}:
        return "duckdb_native_sql"
    if b in {"clickhouse_sql", "clickhouse", "ch"}:
        return "clickhouse_native_sql"
    return "unsupported"


__all__ = [
    "BackendKind",
    "ExecutionKind",
    "PhysicalImplementationSpec",
    "get_physical_spec",
    "PANDAS_DELEGATE_SOURCES",
    "PANDAS_DELEGATE_MODULES",
    "polars_backend_kind",
    "canonical_polars_kind",
    "canonical_polars_is_delegate",
    "capability_quality",
]
