# -*- coding: utf-8 -*-
"""Tests for unified backend capability contracts (FE-BE-P0-001 through FE-BE-P0-004).

This module tests the unified backend capability contract system:
- Single authority for BackendKind, ExecutionKind, CapabilityLevel
- Merged BackendCapability/BackendCapabilityRecord into single typed record
- Production fail-closed on missing/invalid specs
- Heuristics explicitly marked research-only
"""
from __future__ import annotations

import pytest

from backend.contracts import (
    BackendKind,
    CapabilityLevel,
    ExecutionKind,
    PhysicalImplementationSpec,
)
from backend.operator_capability import (
    BackendCapability,
    BackendCapabilityRecord,
    backend_status,
    capability_for,
)


def test_single_authority_enums():
    """FE-BE-P0-001: Verify single authority for enums across modules."""
    from backend.polars_backend_kind import (
        ExecutionKind as PolarsExecutionKind,
        PhysicalImplementationSpec as PolarsPhysicalSpec,
    )

    # All modules must import from backend.contracts
    assert PolarsExecutionKind is ExecutionKind
    assert PolarsPhysicalSpec is PhysicalImplementationSpec

    # Verify key enum members exist
    assert ExecutionKind.POLARS_NATIVE_EXPR
    assert ExecutionKind.POLARS_NUMPY_KERNEL
    assert ExecutionKind.POLARS_PANDAS_DELEGATE
    assert ExecutionKind.DUCKDB_NATIVE_SQL
    assert ExecutionKind.UNSUPPORTED


def test_merged_capability_record():
    """FE-BE-P0-002: Verify BackendCapabilityRecord is alias for BackendCapability."""
    # BackendCapabilityRecord should be an alias
    assert BackendCapabilityRecord is BackendCapability

    # Test instantiation with enum types
    cap = BackendCapability(
        canonical="test_op",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.PRODUCTION_SAFE,
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_nulls=True,
        supports_lazy=True,
    )

    # Verify enum types are preserved
    assert isinstance(cap.backend, BackendKind)
    assert isinstance(cap.level, CapabilityLevel)
    assert isinstance(cap.execution_kind, ExecutionKind)
    assert cap.backend == BackendKind.POLARS
    assert cap.level == CapabilityLevel.PRODUCTION_SAFE
    assert cap.execution_kind == ExecutionKind.POLARS_NATIVE_EXPR


def test_capability_for_returns_unified_type():
    """FE-BE-P0-002: capability_for returns BackendCapability with enum types."""
    from cleaned_operators import load_all

    load_all()

    cap = capability_for("ts_mean", "pandas_numpy")

    # Should return BackendCapability (not BackendCapabilityRecord)
    assert isinstance(cap, BackendCapability)

    # All classification fields should be enums
    assert isinstance(cap.backend, BackendKind)
    assert isinstance(cap.level, CapabilityLevel)
    assert isinstance(cap.execution_kind, ExecutionKind)

    # Legacy property for backward compatibility
    assert isinstance(cap.status, str)
    assert cap.status == cap.level.value


def test_production_mode_requires_explicit_spec():
    """FE-BE-P0-003: Production mode requires explicit PhysicalImplementationSpec."""
    from backend.polars_backend_kind import (
        PolarsImplementationKind,
        polars_backend_kind,
    )

    class OperatorWithoutSpec:
        canonical = "test_without_spec"

        def _calculate_series(self, df):
            return df  # Would normally be classified as native

    # Production mode (default): should return UNSUPPORTED and warn
    with pytest.warns(UserWarning, match="Production mode.*lacks explicit"):
        kind = polars_backend_kind(OperatorWithoutSpec(), production_mode=True)
        assert kind == PolarsImplementationKind.UNSUPPORTED

    # Research mode: falls back to heuristics with warning
    with pytest.warns(UserWarning, match="Research mode.*lacks explicit"):
        kind = polars_backend_kind(OperatorWithoutSpec(), production_mode=False)
        # Should classify based on heuristics
        assert kind == PolarsImplementationKind.POLARS_NATIVE


def test_explicit_spec_overrides_heuristics():
    """FE-BE-P0-003: Explicit spec takes precedence over source inspection."""
    from backend.polars_backend_kind import (
        PolarsImplementationKind,
        polars_backend_kind,
    )

    class OperatorWithExplicitSpec:
        canonical = "explicit_delegate"
        _physical_spec = PhysicalImplementationSpec(
            canonical="explicit_delegate",
            backend="polars",
            execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        )

        def _calculate_series(self, df):
            # Even though this looks native, explicit spec says delegate
            return df

    # Should use explicit spec, no warnings
    kind = polars_backend_kind(OperatorWithExplicitSpec(), production_mode=True)
    assert kind == PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE


def test_physical_spec_production_eligibility():
    """FE-BE-P0-004: PhysicalImplementationSpec.is_production_eligible() works correctly."""
    # Native implementations are production-eligible
    valid_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    spec_native = PhysicalImplementationSpec(
        canonical="test",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        implementation_source_hash=valid_hash,
        emitter_identity="test.polars.expr:v1",
        parameter_domain_hash=valid_hash,
        semantic_contract_hash=valid_hash,
        implementation_closure_hash=valid_hash,
    )
    assert spec_native.is_production_eligible()

    # Delegates are NOT production-eligible
    spec_delegate = PhysicalImplementationSpec(
        canonical="test",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
    )
    assert not spec_delegate.is_production_eligible()

    # Unsupported is NOT production-eligible
    spec_unsupported = PhysicalImplementationSpec(
        canonical="test",
        backend="polars",
        execution_kind=ExecutionKind.UNSUPPORTED,
    )
    assert not spec_unsupported.is_production_eligible()


def test_capability_quality_marked_research_only():
    """FE-BE-P0-003: capability_quality is marked as research/diagnostic only."""
    from backend.polars_backend_kind import capability_quality
    import inspect

    # Check docstring contains warning
    doc = inspect.getdoc(capability_quality)
    assert doc is not None
    assert "RESEARCH/DIAGNOSTIC USE ONLY" in doc
    assert "production_mode" in doc


def test_backend_status_production_mode_default():
    """FE-BE-P0-003: backend_status defaults to production_mode=True."""
    from cleaned_operators import load_all
    import inspect

    load_all()

    # Check signature
    sig = inspect.signature(backend_status)
    assert "production_mode" in sig.parameters
    assert sig.parameters["production_mode"].default is True

    # Test actual behavior - should fail closed for operators without explicit spec
    # (most operators currently lack _physical_spec, so production_mode=True is stricter)


def test_csv_export_handles_enum_types():
    """FE-BE-P0-002: to_csv_row() correctly serializes enum types."""
    cap = BackendCapability(
        canonical="test",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.PRODUCTION_SAFE,
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    )

    row = cap.to_csv_row()

    # Should convert enums to their string values
    assert isinstance(row["backend"], str)
    assert isinstance(row["level"], str)
    assert isinstance(row["execution_kind"], str)
    assert row["backend"] == "polars"
    assert row["level"] == "production_safe"
    assert row["execution_kind"] == "polars_native_expr"


def test_is_native_execution():
    """FE-BE-P0-002: is_native_execution() works with unified enum types."""
    cap_native = BackendCapability(
        canonical="test",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.PRODUCTION_SAFE,
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    )
    assert cap_native.is_native_execution()

    cap_delegate = BackendCapability(
        canonical="test",
        backend=BackendKind.POLARS,
        level=CapabilityLevel.IMPLEMENTED,
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
    )
    assert not cap_delegate.is_native_execution()


# ---------------------------------------------------------------------------
# R21-IS-NATIVE-EXECUTION-SYNC regression tests
# ---------------------------------------------------------------------------

def test_is_native_execution_covers_numba_cpu_kernel():
    """R21-IS-NATIVE-EXECUTION-SYNC: NUMBA_CPU_KERNEL is native."""
    cap = BackendCapability(
        canonical="test",
        backend=BackendKind.PANDAS_NUMPY,
        level=CapabilityLevel.IMPLEMENTED,
        execution_kind=ExecutionKind.NUMBA_CPU_KERNEL,
    )
    assert cap.is_native_execution()


def test_is_native_execution_covers_clickhouse_native_sql():
    """R21-IS-NATIVE-EXECUTION-SYNC: CLICKHOUSE_NATIVE_SQL is native."""
    cap = BackendCapability(
        canonical="test",
        backend=BackendKind.CLICKHOUSE_SQL,
        level=CapabilityLevel.IMPLEMENTED,
        execution_kind=ExecutionKind.CLICKHOUSE_NATIVE_SQL,
    )
    assert cap.is_native_execution()


def test_is_native_execution_covers_q_native():
    """R21-IS-NATIVE-EXECUTION-SYNC: Q_NATIVE is native."""
    cap = BackendCapability(
        canonical="test",
        backend=BackendKind.Q_KDB,
        level=CapabilityLevel.IMPLEMENTED,
        execution_kind=ExecutionKind.Q_NATIVE,
    )
    assert cap.is_native_execution()


# ---------------------------------------------------------------------------
# R21-HASH-FORMAT-ENFORCE regression tests
# ---------------------------------------------------------------------------

def test_valid_sha256_hashes_pass_validation():
    """R21-HASH-FORMAT-ENFORCE: Valid 64-char lowercase hex SHA256 hashes pass."""
    valid_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    assert len(valid_hash) == 64
    assert all(c in "0123456789abcdef" for c in valid_hash)

    spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.NATIVE_GROUP,
        implementation_source_hash=valid_hash,
        parameter_domain_hash=valid_hash,
        semantic_contract_hash=valid_hash,
        implementation_closure_hash=valid_hash,
        emitter_identity="test.pandas:v1",
    )
    errors = spec.validation_errors()
    # Should have no hash-related errors
    hash_errors = [e for e in errors if "hash" in e.lower()]
    assert hash_errors == [], f"Unexpected hash errors: {hash_errors}"
    assert spec.is_production_eligible()


def test_invalid_hash_format_with_label_prefix():
    """R21-HASH-FORMAT-ENFORCE: Hash with label prefix 'ts_mean:v2' fails validation."""
    valid_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.NATIVE_GROUP,
        implementation_source_hash="ts_mean:v2",
        parameter_domain_hash=valid_hash,
        semantic_contract_hash=valid_hash,
        implementation_closure_hash=valid_hash,
        emitter_identity="test.pandas:v1",
    )
    errors = spec.validation_errors()
    # Should fail validation due to invalid implementation_source_hash format
    assert "implementation_source_hash" in errors[0]
    assert "invalid" in errors[0].lower()
    assert "format" in errors[0].lower()
    # Should NOT be production eligible
    assert not spec.is_production_eligible()


def test_invalid_hash_format_uppercase_hex():
    """R21-HASH-FORMAT-ENFORCE: Uppercase hex hash fails (must be lowercase)."""
    uppercase_hash = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2"
    assert len(uppercase_hash) == 64

    spec = PhysicalImplementationSpec(
        canonical="ts_mean",
        backend="pandas_numpy",
        execution_kind=ExecutionKind.NATIVE_GROUP,
        implementation_source_hash=uppercase_hash,
        parameter_domain_hash=uppercase_hash,
        semantic_contract_hash=uppercase_hash,
        implementation_closure_hash=uppercase_hash,
        emitter_identity="test.pandas:v1",
    )
    errors = spec.validation_errors()
    # Should fail validation due to uppercase characters
    hash_errors = [e for e in errors if "hash" in e.lower()]
    assert len(hash_errors) == 4  # All four hash fields should fail
    assert not spec.is_production_eligible()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
