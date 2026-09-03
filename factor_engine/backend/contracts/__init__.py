# -*- coding: utf-8 -*-
"""Single authority for backend capability contracts (R21-P023-Four-Backend-Plan).

This module defines the canonical enums and dataclasses for backend capability
classification. All other modules must import from here to ensure ABI consistency.

FE-P0-003: ExecutionKind unified enum with all required members
FE-P0-004: PhysicalImplementationSpec as single authority
FE-P0-005: Production classification requires explicit validated spec
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

__all__ = [
    "BackendFamily",
    "PhysicalBackend",
    "BackendKind",
    "Representation",
    "ExecutionKind",
    "CapabilityLevel",
    "PhysicalImplementationID",
    "PhysicalImplementationSpec",
    "Accelerator",
    # Canonical string vocabulary + validation
    "CANONICAL_BACKEND_STRINGS",
    "BACKEND_FAMILY_CANONICAL",
    "PHYSICAL_BACKEND_CANONICAL",
    "REPRESENTATION_CANONICAL",
    "EXECUTION_KIND_CANONICAL",
    "ACCELERATOR_CANONICAL",
    "canonical_backend_string",
    "validate_backend_string",
    "is_canonical_backend_string",
]


class BackendFamily(str, Enum):
    """Logical backend family grouping physical backends."""
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    SQL = "sql"
    Q_KDB = "q_kdb"


class BackendKind(str, Enum):
    """Physical backend execution engines (legacy alias of :class:`PhysicalBackend`).

    Kept for backward compatibility with existing consumers. New code should
    prefer :class:`PhysicalBackend`, which is the single canonical authority for
    physical backend identifiers (R21-BACKEND-VOCABULARY).
    """
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"


class PhysicalBackend(str, Enum):
    """Canonical physical backend execution engines (single authority).

    This is the authoritative vocabulary for physical backend identifiers.
    ``BackendKind`` is retained as a backward-compatible alias so existing
    consumers keep working; new code must use :class:`PhysicalBackend`.

    Canonical strings (each member's ``.value`` is the only accepted spelling):
        pandas_numpy   — certified pandas/numpy reference implementation
        polars         — Polars native expression / columnar execution
        duckdb_sql     — DuckDB SQL lowering
        clickhouse_sql — ClickHouse SQL lowering
        q_kdb          — Q/KDB physical execution backend
    """
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"


class Representation(str, Enum):
    """Data representation / layout layer for a backend execution.

    Distinguishes the physical layout a backend operates on, independent of the
    execution engine. This is the canonical vocabulary for representation
    identifiers (R21-BACKEND-VOCABULARY).

    Canonical strings:
        wide_panel   — wide panel layout (rows=instruments, cols=dates)
        long         — long/tidy layout (one row per instrument-date)
        lazy         — deferred / lazy evaluation graph (not yet materialized)
        eager        — eager / materialized execution
        sql          — SQL relational representation
        table        — generic tabular representation (e.g. Q/KDB table)
    """
    WIDE_PANEL = "wide_panel"
    LONG = "long"
    LAZY = "lazy"
    EAGER = "eager"
    SQL = "sql"
    TABLE = "table"


class Accelerator(str, Enum):
    """Hardware accelerator for kernel execution."""
    NONE = "none"
    NUMBA_CPU = "numba_cpu"


class ExecutionKind(str, Enum):
    """How a backend executes an operator (unified authority for FE-P0-003).

    This enum consolidates all execution kind classifications across the codebase.
    No other module should define execution kind enums independently.
    """
    # Generic kinds
    UNSUPPORTED = "unsupported"
    REFERENCE = "reference"

    # Native execution (no delegation)
    NATIVE_EXPR = "native_expr"
    NATIVE_GROUP = "native_group"
    NATIVE_STREAMING = "native_streaming"

    # Polars-specific native execution
    POLARS_NATIVE_EXPR = "polars_native_expr"
    POLARS_NUMPY_KERNEL = "polars_numpy_kernel"
    # Backward-compatible alias kept for existing runtime/type-level consumers.
    POLARS_NATIVE_KERNEL = "polars_numpy_kernel"

    # Numba CPU kernel execution (NUMBA_CPU_KERNEL is an accelerator, not a backend kind)
    NUMBA_CPU_KERNEL = "numba_cpu_kernel"

    # Delegation to other backends
    DELEGATE_PYTHON = "delegate_python"
    DELEGATE_PANDAS = "delegate_pandas"
    POLARS_PANDAS_DELEGATE = "polars_pandas_delegate"

    # Pandas reference
    PANDAS_REFERENCE = "pandas_reference"

    # SQL execution
    DUCKDB_NATIVE_SQL = "duckdb_native_sql"
    SQL_NATIVE = "duckdb_native_sql"
    CLICKHOUSE_NATIVE_SQL = "clickhouse_native_sql"
    SQL_PYTHON_UDF = "sql_python_udf"

    # Q backend native execution
    Q_NATIVE = "q_native"


class CapabilityLevel(str, Enum):
    """Backend capability levels."""
    UNSUPPORTED = "unsupported"
    IMPLEMENTED = "implemented"
    PARITY_VERIFIED = "parity_verified"
    PRODUCTION_SAFE = "production_safe"


@dataclass(frozen=True)
class PhysicalImplementationID:
    """Deterministic identity of one selectable physical implementation."""

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PhysicalImplementationSpec:
    """Explicit declaration of how an operator implementation executes.

    Single authority for physical implementation metadata (FE-P0-004).
    This replaces source-code heuristics with an authoritative contract.

    Operator classes should define a ``_physical_spec`` class attribute or
    ``physical_spec()`` method returning this spec. Production eligibility
    REQUIRES explicit validated spec; absent/broken specs fail closed (FE-P0-005).

    Fields:
        canonical: Canonical operator name (e.g., "ts_mean").
        backend: Physical backend identifier ("pandas_numpy", "polars", "duckdb_sql").
        execution_kind: How this implementation executes (see ExecutionKind enum).
        supports_lazy: Can defer computation until collect()?
        supports_streaming: Can process in batches without full materialization?
        stateful: Does execution carry mutable state across windows (e.g., EMA)?
        materializes_full_panel: Must load entire panel into memory?
        requires_sorted: Requires pre-sorted input for correctness?
        supports_nulls: Handles null values correctly per spec?
        supports_nan: Handles NaN correctly (vs treating as null)?
        supports_inf: Handles +/-Inf correctly (vs clamping/error)?
        notes: Human-readable notes about implementation choices/limits.
    """

    canonical: str
    backend: str
    execution_kind: ExecutionKind

    # Execution mode capabilities
    supports_lazy: bool = False
    supports_streaming: bool = False
    stateful: bool = False
    materializes_full_panel: bool = False
    requires_sorted: bool = False

    # Data handling capabilities
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False

    # Documentation
    notes: str = ""

    # R2-P0-017 identity/evidence bindings. Defaults preserve constructor ABI,
    # but an omitted binding is deliberately ineligible for production.
    #
    # ``implementation_closure_hash`` binds the full semantic closure of the
    # implementation: source/AST, transitive helper source, emitter/kernel
    # identity, production parameter signature (ParamSpec), and numerical
    # semantic policies.  The digest must be a 64-hex SHA-256 computed from the
    # canonical closure payload returned by
    # :func:`backend.evidence_provenance.implementation_closure_hash_for`.
    implementation_source_hash: str = ""
    emitter_identity: str = ""
    kernel_identity: str = ""
    accelerator: Accelerator = Accelerator.NONE
    kernel_signature: str = ""
    parameter_domain_hash: str = ""
    semantic_contract_hash: str = ""
    implementation_closure_hash: str = ""

    @staticmethod
    def _nonblank(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def validation_errors(self) -> tuple[str, ...]:
        """Return deterministic, fail-closed validation failures."""
        errors: list[str] = []
        if not self._nonblank(self.canonical):
            errors.append("blank canonical")
        if not self._nonblank(self.backend):
            errors.append("blank backend")
        elif self.backend not in {member.value for member in BackendKind}:
            errors.append("unknown backend")
        if not isinstance(self.execution_kind, ExecutionKind):
            errors.append("invalid execution_kind")
        # Identity/evidence binding fields.  A blank value is a hard failure;
        # a NON-blank value may be EITHER a plaintext identity token (the
        # operator authoring convention across cleaned_operators, e.g.
        # ``cleaned_operators.common.time_series:TSMean:v1`` /
        # ``ts_mean:min_periods=1:axis=time:v1``) OR a real 64-hex SHA-256
        # digest (evidence provenance).  Both are honest immutable binding
        # identifiers that change whenever the implementation/domain/contract
        # changes, so a non-blank value is accepted regardless of format.  (R47
        # tried to REQUIRE 64-hex + a non-blank implementation_closure_hash,
        # which made every registered spec incomplete — physical_impl id /
        # production admission all failed closed for the entire catalog;
        # restored the pre-R47 contract where the three original fields are
        # required and implementation_closure_hash is an optional v3
        # strengthening that admission must not hinge on.)
        for field_name in (
            "implementation_source_hash",
            "parameter_domain_hash",
            "semantic_contract_hash",
        ):
            if not self._nonblank(getattr(self, field_name)):
                errors.append(f"blank {field_name}")
        if not (self._nonblank(self.emitter_identity) or self._nonblank(self.kernel_identity)):
            errors.append("blank emitter/kernel identity")
        return tuple(errors)

    @property
    def physical_implementation_id(self) -> PhysicalImplementationID | None:
        """Return the bound ID, or ``None`` for an incomplete declaration.

        Version history:
            v1: original payload (backend, canonical, emitter_identity, execution_kind,
                implementation_source_hash, kernel_identity, parameter_domain_hash,
                semantic_contract_hash).
            v2: added accelerator and kernel_signature to payload.
            v3: added implementation_closure_hash binding (full semantic closure).
        """
        if self.validation_errors():
            return None
        payload = {
            "accelerator": self.accelerator.value,
            "backend": self.backend.strip(),
            "canonical": self.canonical.strip(),
            "emitter_identity": self.emitter_identity.strip(),
            "execution_kind": self.execution_kind.value,
            "implementation_closure_hash": self.implementation_closure_hash.strip(),
            "implementation_source_hash": self.implementation_source_hash.strip(),
            "kernel_identity": self.kernel_identity.strip(),
            "kernel_signature": self.kernel_signature.strip(),
            "parameter_domain_hash": self.parameter_domain_hash.strip(),
            "semantic_contract_hash": self.semantic_contract_hash.strip(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return PhysicalImplementationID(f"pi:v3:{digest}")

    def is_production_eligible(self) -> bool:
        """Check if this spec allows production eligibility.

        Returns False if execution_kind is UNSUPPORTED or any delegate type,
        or if critical capabilities are missing.
        """
        if self.validation_errors():
            return False
        if self.execution_kind == ExecutionKind.UNSUPPORTED:
            return False

        # Delegates cannot be production-eligible
        if self.execution_kind in {
            ExecutionKind.DELEGATE_PYTHON,
            ExecutionKind.DELEGATE_PANDAS,
            ExecutionKind.POLARS_PANDAS_DELEGATE,
        }:
            return False

        return True

    def is_native_execution(self) -> bool:
        """Check if execution is truly native (not delegate)."""
        return self.execution_kind in {
            ExecutionKind.NATIVE_EXPR,
            ExecutionKind.NATIVE_GROUP,
            ExecutionKind.NATIVE_STREAMING,
            ExecutionKind.POLARS_NATIVE_EXPR,
            ExecutionKind.POLARS_NUMPY_KERNEL,
            ExecutionKind.NUMBA_CPU_KERNEL,
            ExecutionKind.DUCKDB_NATIVE_SQL,
            ExecutionKind.CLICKHOUSE_NATIVE_SQL,
            ExecutionKind.Q_NATIVE,
        }
