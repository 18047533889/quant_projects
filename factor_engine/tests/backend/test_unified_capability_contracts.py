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

from factor_engine.backend.contracts import (
    BackendKind,
    CapabilityLevel,
    ExecutionKind,
    PhysicalImplementationSpec,
)
from factor_engine.backend.operator_capability import (
    BackendCapability,
    BackendCapabilityRecord,
    backend_status,
    capability_for,
)


def test_single_authority_enums():
    """FE-BE-P0-001: Verify single authority for enums across modules."""
    from factor_engine.backend.polars_backend_kind import (
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
    from factor_engine.cleaned_operators import load_all

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
    from factor_engine.backend.polars_backend_kind import (
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
    from factor_engine.backend.polars_backend_kind import (
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
    spec_native = PhysicalImplementationSpec(
        canonical="test",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        implementation_source_hash="test-source-sha256",
        emitter_identity="test.polars.expr:v1",
        parameter_domain_hash="test-params-sha256",
        semantic_contract_hash="test-contract-sha256",
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
    from factor_engine.backend.polars_backend_kind import capability_quality
    import inspect

    # Check docstring contains warning
    doc = inspect.getdoc(capability_quality)
    assert doc is not None
    assert "RESEARCH/DIAGNOSTIC USE ONLY" in doc
    assert "production_mode" in doc


def test_backend_status_production_mode_default():
    """FE-BE-P0-003: backend_status defaults to production_mode=True."""
    from factor_engine.cleaned_operators import load_all
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
