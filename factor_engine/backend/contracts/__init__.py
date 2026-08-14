# -*- coding: utf-8 -*-
"""Single authority for backend capability contracts.

This module defines the canonical enums and dataclasses for backend capability
classification. All other modules must import from here to ensure ABI consistency.

FE-P0-003: ExecutionKind unified enum with all required members
FE-P0-004: PhysicalImplementationSpec as single authority
FE-P0-005: Production classification requires explicit validated spec
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

__all__ = [
    "BackendKind",
    "ExecutionKind",
    "CapabilityLevel",
    "PhysicalImplementationSpec",
]


class BackendKind(str, Enum):
    """Physical backend execution engines."""
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"


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

    # Delegation to other backends
    DELEGATE_PYTHON = "delegate_python"
    DELEGATE_PANDAS = "delegate_pandas"
    POLARS_PANDAS_DELEGATE = "polars_pandas_delegate"

    # Pandas reference
    PANDAS_REFERENCE = "pandas_reference"

    # SQL execution
    DUCKDB_NATIVE_SQL = "duckdb_native_sql"
    SQL_PYTHON_UDF = "sql_python_udf"


class CapabilityLevel(str, Enum):
    """Backend capability levels."""
    UNSUPPORTED = "unsupported"
    IMPLEMENTED = "implemented"
    PARITY_VERIFIED = "parity_verified"
    PRODUCTION_SAFE = "production_safe"


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

    def is_production_eligible(self) -> bool:
        """Check if this spec allows production eligibility.

        Returns False if execution_kind is UNSUPPORTED or any delegate type,
        or if critical capabilities are missing.
        """
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
            ExecutionKind.DUCKDB_NATIVE_SQL,
        }
