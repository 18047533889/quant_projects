# -*- coding: utf-8 -*-
"""Focused tests for FE-P0-003, FE-P0-004, FE-P0-005 capability contract unification.

Tests verify:
- Single authority for ExecutionKind/BackendKind/CapabilityLevel
- PhysicalImplementationSpec single authority
- Production mode defaults to fail closed (no heuristics)
- No nonexistent enum member failures
"""
from __future__ import annotations

import pytest


def test_unified_execution_kind_import():
    """FE-P0-003: ExecutionKind has single authority with all required members."""
    from backend.contracts import ExecutionKind

    # Verify all required members exist (no AttributeError)
    required_members = [
        "UNSUPPORTED",
        "REFERENCE",
        "NATIVE_EXPR",
        "NATIVE_GROUP",
        "NATIVE_STREAMING",
        "POLARS_NATIVE_EXPR",
        "POLARS_NUMPY_KERNEL",
        "DELEGATE_PYTHON",
        "DELEGATE_PANDAS",
        "POLARS_PANDAS_DELEGATE",
        "PANDAS_REFERENCE",
        "DUCKDB_NATIVE_SQL",
        "SQL_PYTHON_UDF",
    ]

    for member in required_members:
        assert hasattr(ExecutionKind, member), f"ExecutionKind missing {member}"
        getattr(ExecutionKind, member)  # Will raise AttributeError if missing


def test_unified_backend_kind_import():
    """FE-P0-003: BackendKind has single authority."""
    from backend.contracts import BackendKind

    required_members = [
        "PANDAS_NUMPY",
        "POLARS",
        "DUCKDB_SQL",
        "CLICKHOUSE_SQL",
        "Q_KDB",
    ]

    for member in required_members:
        assert hasattr(BackendKind, member), f"BackendKind missing {member}"


def test_physical_implementation_spec_authority():
    """FE-P0-004: PhysicalImplementationSpec single authority."""
    from backend.contracts import ExecutionKind, PhysicalImplementationSpec

    # Can construct spec
    spec = PhysicalImplementationSpec(
        canonical="test_op",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=True,
        supports_streaming=True,
        supports_nulls=True,
    )

    assert spec.canonical == "test_op"
    assert spec.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR
    assert spec.is_native_execution() is True
    assert spec.is_production_eligible() is True


def test_polars_backend_kind_imports_from_contracts():
    """FE-P0-003/004: polars_backend_kind imports from unified authority."""
    from backend.polars_backend_kind import ExecutionKind, PhysicalImplementationSpec
    from backend.contracts import ExecutionKind as ContractsEK
    from backend.contracts import PhysicalImplementationSpec as ContractsPIS

    # Same class (identity check, not just equality)
    assert ExecutionKind is ContractsEK
    assert PhysicalImplementationSpec is ContractsPIS


def test_operator_capability_imports_from_contracts():
    """FE-P0-003/004: operator_capability imports from unified authority."""
    from backend.operator_capability import (
        BackendKind,
        CapabilityLevel,
        ExecutionKind,
    )
    from backend.contracts import (
        BackendKind as ContractsBK,
        CapabilityLevel as ContractsCL,
        ExecutionKind as ContractsEK,
    )

    # Same classes
    assert BackendKind is ContractsBK
    assert CapabilityLevel is ContractsCL
    assert ExecutionKind is ContractsEK


def test_production_mode_defaults_to_fail_closed():
    """FE-P0-005: Production mode defaults to True (fail closed)."""
    from backend.polars_backend_kind import polars_backend_kind, BackendKind
    from backend.contracts import ExecutionKind, PhysicalImplementationSpec

    # Mock operator without _physical_spec
    class MockOperatorNoSpec:
        canonical = "test_no_spec"

        def _calculate_series(self, df, params):
            return df

    op = MockOperatorNoSpec()

    # Default behavior: production mode (fail closed)
    with pytest.warns(UserWarning, match="Production mode.*UNSUPPORTED"):
        kind = polars_backend_kind(op)  # No production_mode arg
        assert kind == BackendKind.UNSUPPORTED


def test_explicit_research_mode():
    """FE-P0-005: Explicit production_mode=False allows heuristic fallback."""
    from backend.polars_backend_kind import polars_backend_kind, BackendKind

    class MockOperatorNoSpec:
        canonical = "test_no_spec"

        def _calculate_series(self, df, params):
            return df

    op = MockOperatorNoSpec()

    # Research mode: allows heuristic fallback
    with pytest.warns(UserWarning, match="Research mode.*heuristics"):
        kind = polars_backend_kind(op, production_mode=False)
        # Will be POLARS_NATIVE from heuristic fallback
        assert kind in {BackendKind.POLARS_NATIVE, BackendKind.POLARS_UDF_PANDAS_DELEGATE}


def test_explicit_spec_works_in_production_mode():
    """FE-P0-005: Operators with explicit spec work in production mode."""
    from backend.polars_backend_kind import polars_backend_kind, BackendKind, get_physical_spec
    from backend.contracts import ExecutionKind, PhysicalImplementationSpec

    class MockOperatorWithSpec:
        canonical = "test_with_spec"
        _physical_spec = PhysicalImplementationSpec(
            canonical="test_with_spec",
            backend="polars",
            execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
            supports_lazy=True,
        )

    op = MockOperatorWithSpec()

    # Can extract spec
    spec = get_physical_spec(op)
    assert spec is not None
    assert spec.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR

    # Production mode: works fine (no warning)
    kind = polars_backend_kind(op, production_mode=True)
    assert kind == BackendKind.POLARS_NATIVE


def test_backend_capability_record_native_execution():
    """FE-P0-003: BackendCapabilityRecord.is_native_execution uses updated enum."""
    from backend.operator_capability import BackendCapabilityRecord
    from backend.contracts import BackendKind, CapabilityLevel, ExecutionKind

    # Native expr
    record = BackendCapabilityRecord(
        canonical="test",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.PRODUCTION_SAFE,
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    )
    assert record.is_native_execution() is True

    # Delegate
    record_delegate = BackendCapabilityRecord(
        canonical="test",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.IMPLEMENTED,
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
    )
    assert record_delegate.is_native_execution() is False


def test_no_duplicate_enum_definitions():
    """FE-P0-003: Ensure no duplicate ExecutionKind definitions."""
    from backend.contracts import ExecutionKind as ContractsEK

    # Import after contracts to ensure they get the same one
    from backend.polars_backend_kind import ExecutionKind as PolarsEK
    from backend.operator_capability import ExecutionKind as CapabilityEK

    # All should be the exact same class
    assert PolarsEK is ContractsEK
    assert CapabilityEK is ContractsEK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
