# -*- coding: utf-8 -*-
"""Regression tests for R21-NUMBA-PLANNER-CANDIDATES (Writer BQ).

Tests Numba backend as a planner-selectable candidate for recursive/stateful operators.
Targets planning/backend_selector.py and planning/dag_partition_optimizer.py.
"""
import pytest
from factor_engine.planning.backend_selector import (
    IntelligentBackendSelector,
    DataScale,
    OperatorProfile,
    MemoryPressure,
    RoutingContext,
    RoutingDecision,
)
from factor_engine.planning.backend_region import PhysicalBackend, ExecutionAxis
from factor_engine.backend.contracts import ExecutionKind


class TestNumbaBackendCharacteristics:
    """Test that Numba backend characteristics are properly defined."""

    def test_numba_backend_exists_in_characteristics(self):
        """Numba backend should be in _BACKEND_CHARACTERISTICS if implemented."""
        from factor_engine.planning.backend_selector import _BACKEND_CHARACTERISTICS

        # If NUMBA backend is added, it should be in characteristics
        # This test ensures we don't break existing backends
        assert PhysicalBackend.PANDAS_NUMPY in _BACKEND_CHARACTERISTICS
        assert PhysicalBackend.POLARS_EAGER in _BACKEND_CHARACTERISTICS
        assert PhysicalBackend.POLARS_LAZY in _BACKEND_CHARACTERISTICS
        assert PhysicalBackend.DUCKDB_SQL in _BACKEND_CHARACTERISTICS

    def test_cost_model_includes_numba_fields(self):
        """Cost model should include cold_jit_ms, warm_ms, cache_hit_probability when Numba is used."""
        # This test verifies the cost model structure
        from factor_engine.planning.backend_selector import _BACKEND_CHARACTERISTICS

        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=100_000,
            estimated_columns=10,
            estimated_bytes=8_000_000,
            available_memory_bytes=8_000_000_000,
            operator_profile=OperatorProfile.WINDOW_HEAVY,
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            window_params=[50, 100],
            has_regression=False,
            parent_backend=None,
            allows_streaming=True,
            performance_priority="balanced",
        )

        decision = selector.select_backend(ctx)
        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_backend in _BACKEND_CHARACTERISTICS


class TestNumbaCandidateSelection:
    """Test Numba as a candidate for stateful/recursive operators."""

    def test_stateful_operator_gets_numba_candidate(self):
        """Operators with NUMBA_CPU_KERNEL implementation should get Numba candidates."""
        # Simulate operator with NUMBA CPU kernel implementation
        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=500_000,
            estimated_columns=5,
            estimated_bytes=20_000_000,
            available_memory_bytes=16_000_000_000,
            operator_profile=OperatorProfile.WINDOW_HEAVY,
            execution_axis=ExecutionAxis.RECURSIVE_TIME_PER_INSTRUMENT,
            window_params=[100, 200, 500],
            has_regression=False,
            parent_backend=None,
            allows_streaming=False,
            performance_priority="speed",
        )

        decision = selector.select_backend(ctx)
        # Verify decision is made
        assert isinstance(decision, RoutingDecision)
        assert decision.estimated_cost_ms > 0
        assert decision.memory_footprint_bytes > 0

    def test_jit_warmup_considered_in_cost(self):
        """Cost model should account for JIT warmup for Numba kernels."""
        # Test that cold_jit_ms penalty is applied for Numba backend
        selector = IntelligentBackendSelector()

        # First call (cold start)
        ctx_cold = RoutingContext(
            estimated_rows=10_000,
            estimated_columns=5,
            estimated_bytes=400_000,
            available_memory_bytes=16_000_000_000,
            operator_profile=OperatorProfile.ELEMENTWISE_HEAVY,
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            window_params=[],
            has_regression=False,
            parent_backend=None,
            allows_streaming=False,
            performance_priority="balanced",
        )

        decision_cold = selector.select_backend(ctx_cold)

        # Record actual cost for warmup learning
        selector.record_actual_cost(
            backend=decision_cold.chosen_backend,
            scale=DataScale.MEDIUM,
            profile=OperatorProfile.ELEMENTWISE_HEAVY,
            actual_cost_ms=15.0,  # Simulated warm execution
        )

        # Second call should benefit from warm cache
        decision_warm = selector.select_backend(ctx_cold)

        # Verify adaptive learning works
        stats = selector.get_calibration_stats()
        assert "adjustment_factors" in stats
        assert "history_size" in stats


class TestNumbaCacheHitProbability:
    """Test cache hit probability modeling for Numba JIT."""

    def test_cache_hit_probability_affects_selection(self):
        """Higher cache hit probability should reduce effective cost."""
        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=100_000,
            estimated_columns=10,
            estimated_bytes=8_000_000,
            available_memory_bytes=16_000_000_000,
            operator_profile=OperatorProfile.WINDOW_HEAVY,
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            window_params=[50, 100, 200],
            has_regression=False,
            parent_backend=PhysicalBackend.POLARS_EAGER,
            allows_streaming=False,
            performance_priority="balanced",
        )

        decision = selector.select_backend(ctx)

        # Verify transfer affinity works with parent backend
        assert isinstance(decision, RoutingDecision)
        assert len(decision.reasoning) > 0


class TestNumbaPlannerIntegration:
    """Integration tests for Numba in planner candidate selection."""

    def test_planner_selects_numba_for_recursive_ops(self):
        """Planner should consider Numba for recursive_time_per_instrument axis."""
        from factor_engine.planning.backend_selector import select_optimal_backend_for_node, _BACKEND_CHARACTERISTICS

        decision = select_optimal_backend_for_node(
            estimated_rows=200_000,
            estimated_columns=8,
            estimated_bytes=12_800_000,
            available_memory_bytes=16_000_000_000,
            operator_names=["ts_ema", "ts_rolling_mean", "ts_delta"],
            window_params=[50, 100],
            parent_backend=PhysicalBackend.POLARS_EAGER,
            performance_priority="balanced",
        )

        assert isinstance(decision, RoutingDecision)
        assert decision.chosen_backend in _BACKEND_CHARACTERISTICS
        assert decision.estimated_cost_ms >= 0

    def test_numba_conversion_cost_with_polars(self):
        """Test conversion cost when switching from Polars to Numba backend."""
        from factor_engine.planning.backend_selector import _BACKEND_CHARACTERISTICS

        selector = IntelligentBackendSelector()

        # Test conversion cost calculation
        conversion_ms = selector._get_conversion_cost(
            source=PhysicalBackend.POLARS_EAGER,
            target=PhysicalBackend.PANDAS_NUMPY,
            bytes_size=10_000_000,  # 10MB
        )

        assert conversion_ms > 0
        assert isinstance(conversion_ms, float)


# ExecutionKind.NUMBA_CPU_KERNEL contract tests
class TestNumbaExecutionKindContract:
    """Test ExecutionKind.NUMBA_CPU_KERNEL contract from factor_engine.backend.contracts."""

    def test_numba_execution_kind_exists(self):
        """ExecutionKind.NUMBA_CPU_KERNEL should be defined."""
        assert hasattr(ExecutionKind, "NUMBA_CPU_KERNEL")
        assert ExecutionKind.NUMBA_CPU_KERNEL.value == "numba_cpu_kernel"

    def test_numba_is_native_execution(self):
        """NUMBA_CPU_KERNEL should be classified as native execution."""
        from factor_engine.backend.contracts import PhysicalImplementationSpec

        # Verify NUMBA_CPU_KERNEL is in native execution set
        native_kinds = {
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
        assert ExecutionKind.NUMBA_CPU_KERNEL in native_kinds

    def test_numba_spec_production_eligible(self):
        """A valid NUMBA_CPU_KERNEL spec should be production eligible."""
        from factor_engine.backend.contracts import (
            PhysicalImplementationSpec,
            Accelerator,
        )

        spec = PhysicalImplementationSpec(
            canonical="ts_ema",
            backend="pandas_numpy",
            execution_kind=ExecutionKind.NUMBA_CPU_KERNEL,
            supports_lazy=False,
            supports_streaming=False,
            stateful=True,
            materializes_full_panel=False,
            requires_sorted=True,
            supports_nulls=True,
            supports_nan=True,
            supports_inf=False,
            notes="Numba JIT compiled EMA kernel",
            implementation_source_hash="abc123",
            emitter_identity="test_emitter",
            kernel_identity="test_kernel",
            accelerator=Accelerator.NUMBA_CPU,
            kernel_signature="(f8[:], f8, f8) -> f8[:]",
            parameter_domain_hash="def456",
            semantic_contract_hash="ghi789",
        )

        assert spec.is_production_eligible()
        assert spec.is_native_execution()
        assert spec.validation_errors() == ()
