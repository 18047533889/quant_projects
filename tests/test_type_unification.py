# -*- coding: utf-8 -*-
"""Regression tests for R21-TYPE-UNIFICATION.

Verifies:
1. BackendFamily enum exists with correct members
2. Accelerator enum exists with correct members
3. PhysicalImplementationSpec has accelerator and kernel_signature fields
4. BackendName Literal includes q_kdb
5. BackendKind enum includes Q_KDB
"""
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.contracts import (
    BackendFamily,
    BackendKind,
    Accelerator,
    PhysicalImplementationSpec,
    ExecutionKind,
)
from backend.operator_capability import BackendName


def test_backend_family_enum():
    """Test that BackendFamily enum has all required members."""
    assert hasattr(BackendFamily, "PANDAS_NUMPY"), "BackendFamily.PANDAS_NUMPY is missing"
    assert BackendFamily.PANDAS_NUMPY.value == "pandas_numpy"

    assert hasattr(BackendFamily, "POLARS"), "BackendFamily.POLARS is missing"
    assert BackendFamily.POLARS.value == "polars"

    assert hasattr(BackendFamily, "SQL"), "BackendFamily.SQL is missing"
    assert BackendFamily.SQL.value == "sql"

    assert hasattr(BackendFamily, "Q_KDB"), "BackendFamily.Q_KDB is missing"
    assert BackendFamily.Q_KDB.value == "q_kdb"


def test_accelerator_enum():
    """Test that Accelerator enum has all required members."""
    assert hasattr(Accelerator, "NONE"), "Accelerator.NONE is missing"
    assert Accelerator.NONE.value == "none"

    assert hasattr(Accelerator, "NUMBA_CPU"), "Accelerator.NUMBA_CPU is missing"
    assert Accelerator.NUMBA_CPU.value == "numba_cpu"


def test_physical_implementation_spec_has_accelerator_and_kernel_signature():
    """Test that PhysicalImplementationSpec has accelerator and kernel_signature fields."""
    _SHA = "0123456789abcdef" * 4  # 64-char lowercase hex SHA-256
    spec = PhysicalImplementationSpec(
        canonical="test_op",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.NATIVE_EXPR,
        implementation_source_hash=_SHA,
        emitter_identity="test_emitter",
        kernel_identity="test_kernel",
        parameter_domain_hash=_SHA,
        semantic_contract_hash=_SHA,
    )

    # Check default values
    assert spec.accelerator == Accelerator.NONE, f"Expected Accelerator.NONE, got {spec.accelerator}"
    assert spec.kernel_signature == "", f"Expected empty string, got '{spec.kernel_signature}'"

    # Check that we can set values
    spec_with_accel = PhysicalImplementationSpec(
        canonical="test_op",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.NUMBA_CPU_KERNEL,
        implementation_source_hash=_SHA,
        emitter_identity="test_emitter",
        kernel_identity="test_kernel",
        accelerator=Accelerator.NUMBA_CPU,
        kernel_signature="(float64[:], float64[:]) -> float64[:]",
        parameter_domain_hash=_SHA,
        semantic_contract_hash=_SHA,
        implementation_closure_hash=_SHA,
    )

    assert spec_with_accel.accelerator == Accelerator.NUMBA_CPU
    assert spec_with_accel.kernel_signature == "(float64[:], float64[:]) -> float64[:]"

    # Check that validation still works
    errors = spec_with_accel.validation_errors()
    assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    # Check that implementation_id includes new fields
    pi_id = spec_with_accel.physical_implementation_id
    assert pi_id is not None, "physical_implementation_id should not be None"


def test_backend_name_literal_includes_q_kdb():
    """Test that BackendName Literal type includes q_kdb."""
    # This is a type-level check - BackendName should be a Literal type
    # We can verify it's a valid Literal by checking its __args__
    import typing
    args = typing.get_args(BackendName)
    assert "q_kdb" in args, f"BackendName should include 'q_kdb', got {args}"
    assert "pandas_numpy" in args, f"BackendName should include 'pandas_numpy', got {args}"
    assert "polars" in args, f"BackendName should include 'polars', got {args}"
    assert "duckdb_sql" in args, f"BackendName should include 'duckdb_sql', got {args}"
    assert "clickhouse_sql" in args, f"BackendName should include 'clickhouse_sql', got {args}"


def test_backend_kind_includes_q_kdb():
    """Test that BackendKind enum includes Q_KDB."""
    assert hasattr(BackendKind, "Q_KDB"), "BackendKind.Q_KDB is missing"
    assert BackendKind.Q_KDB.value == "q_kdb", f"Expected 'q_kdb', got '{BackendKind.Q_KDB.value}'"

    # Verify all expected members
    expected_members = {"PANDAS_NUMPY", "POLARS", "DUCKDB_SQL", "CLICKHOUSE_SQL", "Q_KDB"}
    actual_members = {m.name for m in BackendKind}
    assert expected_members == actual_members, f"Expected {expected_members}, got {actual_members}"


if __name__ == "__main__":
    # Run tests manually if executed directly
    import pytest
    pytest.main([__file__, "-v"])
