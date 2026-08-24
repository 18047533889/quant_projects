# -*- coding: utf-8 -*-
"""Regression tests for R21-NUMBA-COST-MODEL (post R21-PLANNER-TYPE-UNIFICATION).

R21-PLANNER-TYPE-UNIFICATION: Numba is an Accelerator (``backend.contracts.
Accelerator.NUMBA_CPU`` / ``ExecutionKind.NUMBA_CPU_KERNEL``), NOT a
``PhysicalBackend``.  This test suite verifies the corrected invariant and
keeps the TTDC cost-model integration check (no NUMBA_CPU backend dependency).
"""
import pytest
from factor_engine.planning.backend_selector import (
    IntelligentBackendSelector,
    DataScale,
    OperatorProfile,
    MemoryPressure,
    RoutingContext,
    RoutingDecision,
    BackendCharacteristics,
    _BACKEND_CHARACTERISTICS,
)
from factor_engine.planning.backend_region import PhysicalBackend, ExecutionAxis
from factor_engine.backend.contracts import Accelerator, ExecutionKind


class TestNumbaIsAcceleratorNotBackend:
    """Test that Numba is an accelerator, not a PhysicalBackend (R21-PLANNER-TYPE-UNIFICATION)."""

    def test_numba_not_a_physical_backend(self):
        """PhysicalBackend must NOT contain NUMBA_CPU (Numba = Accelerator)."""
        assert not hasattr(PhysicalBackend, "NUMBA_CPU"), \
            "NUMBA_CPU must not be a PhysicalBackend (Numba is an Accelerator)"

    def test_numba_accelerator_contracts_exist(self):
        """Accelerator.NUMBA_CPU and ExecutionKind.NUMBA_CPU_KERNEL must exist."""
        assert hasattr(Accelerator, "NUMBA_CPU"), "Accelerator.NUMBA_CPU is missing"
        assert Accelerator.NUMBA_CPU.value == "numba_cpu"
        assert hasattr(ExecutionKind, "NUMBA_CPU_KERNEL"), "ExecutionKind.NUMBA_CPU_KERNEL is missing"
        assert ExecutionKind.NUMBA_CPU_KERNEL.value == "numba_cpu_kernel"

    def test_no_numba_backend_characteristics_entry(self):
        """_BACKEND_CHARACTERISTICS must not contain a NUMBA_CPU key."""
        assert all(b.name != "NUMBA_CPU" for b in _BACKEND_CHARACTERISTICS), \
            "_BACKEND_CHARACTERISTICS must not contain a NUMBA_CPU backend entry"


class TestBackendCharacteristicsFields:
    """Test BackendCharacteristics still exposes the JIT cost model capability fields."""

    def test_jit_cost_fields_are_capability_fields(self):
        """BackendCharacteristics carries optional JIT cost fields (defaulted to 0.0)."""
        chars = BackendCharacteristics(
            backend=PhysicalBackend.PANDAS_NUMPY,
            startup_cost_ms=2.0,
            per_row_throughput_mrows_per_sec=0.5,
            memory_overhead_factor=1.2,
            supports_streaming=False,
            preferred_scale=(DataScale.TINY,),
            preferred_profiles=(OperatorProfile.ELEMENTWISE_HEAVY,),
            parallel_efficiency=0.3,
            conversion_penalty_ms=5.0,
        )
        assert chars.cold_jit_ms == 0.0
        assert chars.warm_ms == 0.0
        assert chars.cache_hit_probability == 0.0


class TestBackendSelection:
    """Test backend selection still routes correctly with the unified enum."""

    def test_selection_runs_with_unified_backends(self):
        """Selection must run and return a valid unified PhysicalBackend."""
        ctx = RoutingContext(
            estimated_rows=10_000,
            estimated_columns=5,
            estimated_bytes=10_000 * 5 * 8,
            available_memory_bytes=1024**3,
            operator_profile=OperatorProfile.ELEMENTWISE_HEAVY,
            execution_axis=ExecutionAxis.GLOBAL_PANEL,
            window_params=[],
            has_regression=False,
            parent_backend=None,
            allows_streaming=True,
            performance_priority="balanced",
        )
        selector = IntelligentBackendSelector()
        decision = selector.select_backend(ctx)
        assert decision.chosen_backend in PhysicalBackend
        assert decision.estimated_cost_ms > 0


class TestNumbaTTDCIntegration:
    """Test TTDC calculation keeps working (no NUMBA_CPU backend dependency)."""

    def test_ttdc_includes_jit_cost(self):
        """TTDC predictor should still work for a normal backend."""
        from factor_engine.backend.plan_cost_router import predict_ttdc, TtdcEstimate

        # Create a mock shape object
        class MockShape:
            selected_bytes = 1_000_000
            total_bytes = 1_000_000
            estimated_rows = 100_000
            projected_columns = 10
            total_columns = 10
            remote = False

        shape = MockShape()

        # Test with pandas_numpy backend
        estimate = predict_ttdc(shape, "pandas_numpy")
        assert isinstance(estimate, TtdcEstimate)
        assert estimate.total_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
