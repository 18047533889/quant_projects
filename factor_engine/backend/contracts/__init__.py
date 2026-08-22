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
    "BackendKind",
    "ExecutionKind",
    "CapabilityLevel",
    "PhysicalImplementationID",
    "PhysicalImplementationSpec",
    "Accelerator",
]


class BackendFamily(str, Enum):
    """Logical backend family grouping physical backends."""
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    SQL = "sql"
    Q_KDB = "q_kdb"


class BackendKind(str, Enum):
    """Physical backend execution engines."""
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"


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
    implementation_source_hash: str = ""
    emitter_identity: str = ""
    kernel_identity: str = ""
    accelerator: Accelerator = Accelerator.NONE
    kernel_signature: str = ""
    parameter_domain_hash: str = ""
    semantic_contract_hash: str = ""

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
        """Return the bound ID, or ``None`` for an incomplete declaration."""
        if self.validation_errors():
            return None
        payload = {
            "backend": self.backend.strip(),
            "canonical": self.canonical.strip(),
            "emitter_identity": self.emitter_identity.strip(),
            "execution_kind": self.execution_kind.value,
            "implementation_source_hash": self.implementation_source_hash.strip(),
            "kernel_identity": self.kernel_identity.strip(),
            "parameter_domain_hash": self.parameter_domain_hash.strip(),
            "semantic_contract_hash": self.semantic_contract_hash.strip(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return PhysicalImplementationID(f"pi:v1:{digest}")

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
