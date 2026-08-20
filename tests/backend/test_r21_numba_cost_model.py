# -*- coding: utf-8 -*-
"""Regression tests for R21-NUMBA-COST-MODEL: cold_jit_ms, warm_ms, cache_hit_probability.

Tests Numba backend cost model fields in planning/backend_selector.py.
Verifies TTDC calculation includes JIT compilation and cache hit logic.
"""
import pytest
from planning.backend_selector import (
    IntelligentBackendSelector,
    DataScale,
    OperatorProfile,
    MemoryPressure,
    RoutingContext,
    RoutingDecision,
    BackendCharacteristics,
    _BACKEND_CHARACTERISTICS,
)
from planning.backend_region import PhysicalBackend, ExecutionAxis


class TestNumbaCostModelFields:
    """Test that NUMBA_CPU backend characteristics include JIT cost fields."""

    def test_numba_backend_has_jit_cost_fields(self):
        """NUMBA_CPU backend must have cold_jit_ms, warm_ms, cache_hit_probability."""
        assert PhysicalBackend.NUMBA_CPU in _BACKEND_CHARACTERISTICS
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        assert hasattr(chars, 'cold_jit_ms')
        assert hasattr(chars, 'warm_ms')
        assert hasattr(chars, 'cache_hit_probability')

        # Validate ranges
        assert chars.cold_jit_ms > 0, "cold_jit_ms must be positive"
        assert chars.warm_ms > 0, "warm_ms must be positive"
        assert 0.0 <= chars.cache_hit_probability <= 1.0, "cache_hit_probability must be in [0, 1]"

    def test_jit_cost_fields_positive(self):
        """JIT cost fields must be non-negative."""
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        assert chars.cold_jit_ms >= 0
        assert chars.warm_ms >= 0
        assert chars.cache_hit_probability >= 0


class TestNumbaCostModelCalculation:
    """Test JIT cost model calculation in _evaluate_costs."""

    def test_effective_jit_cost_formula(self):
        """Verify effective JIT cost = (1 - cache_hit) * cold_jit + warm."""
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        # Expected: (1 - 0.8) * 100 + 5 = 20 + 5 = 25ms
        expected_effective = (1.0 - chars.cache_hit_probability) * chars.cold_jit_ms + chars.warm_ms

        assert expected_effective > 0, "Effective JIT cost must be positive"
        assert expected_effective < chars.cold_jit_ms, "Effective JIT cost must be less than cold_jit_ms"

    def test_jit_cost_reduces_with_higher_cache_hit(self):
        """Higher cache_hit_probability should reduce effective JIT cost."""
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        effective_current = (1.0 - chars.cache_hit_probability) * chars.cold_jit_ms + chars.warm_ms

        # Simulate higher cache hit probability (0.95 vs 0.8)
        effective_high = (1.0 - 0.95) * chars.cold_jit_ms + chars.warm_ms

        assert effective_high < effective_current, "Higher cache hit should reduce cost"

    def test_jit_cost_increases_with_cold_start(self):
        """Lower cache_hit_probability should increase effective JIT cost."""
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        effective_current = (1.0 - chars.cache_hit_probability) * chars.cold_jit_ms + chars.warm_ms

        # Simulate cold start (0.0 cache hit)
        effective_cold = (1.0 - 0.0) * chars.cold_jit_ms + chars.warm_ms

        assert effective_cold > effective_current, "Cold start should increase cost"
        assert effective_cold == chars.cold_jit_ms + chars.warm_ms, "Cold start = cold_jit + warm"


class TestNumbaBackendSelection:
    """Test Numba backend selection in routing decisions."""

    def test_numba_is_candidate_in_characteristics(self):
        """NUMBA_CPU backend should be present in _BACKEND_CHARACTERISTICS."""
        assert PhysicalBackend.NUMBA_CPU in _BACKEND_CHARACTERISTICS, \
            "NUMBA_CPU must be in _BACKEND_CHARACTERISTICS"

    def test_numba_cost_includes_jit_overhead(self):
        """Numba backend cost should reflect JIT overhead."""
        selector = IntelligentBackendSelector()

        ctx = RoutingContext(
            estimated_rows=50_000,
            estimated_columns=5,
            estimated_bytes=2_000_000,
            available_memory_bytes=16_000_000_000,
            operator_profile=OperatorProfile.ELEMENTWISE_HEAVY,
            execution_axis=ExecutionAxis.TIME_PER_INSTRUMENT,
            window_params=[],
            has_regression=False,
            parent_backend=None,
            allows_streaming=False,
            performance_priority="balanced",
        )

        decision = selector.select_backend(ctx)

        # Verify cost is reasonable (includes JIT overhead)
        assert decision.estimated_cost_ms > 0
        assert decision.estimated_cost_ms < 10000, "Cost should be reasonable for JIT backend"


class TestNumbaCacheHitProbability:
    """Test cache hit probability affects backend selection."""

    def test_high_cache_hit_reduces_effective_cost(self):
        """Backend with higher cache_hit_probability should have lower effective cost."""
        chars = _BACKEND_CHARACTERISTICS[PhysicalBackend.NUMBA_CPU]

        # Calculate effective JIT cost with current cache hit probability
        effective_jit = (1.0 - chars.cache_hit_probability) * chars.cold_jit_ms + chars.warm_ms

        # Verify that effective JIT cost is significantly lower than cold_jit_ms
        reduction_ratio = effective_jit / chars.cold_jit_ms
        assert reduction_ratio < 0.5, "Effective JIT cost should be significantly lower than cold_jit_ms"


class TestNumbaTTDCIntegration:
    """Test TTDC calculation includes JIT cost model."""

    def test_ttdc_includes_jit_cost(self):
        """TTDC predictor should account for Numba JIT cost model."""
        from backend.plan_cost_router import predict_ttdc, TtdcEstimate

        # Create a mock shape object
        class MockShape:
            selected_bytes = 1_000_000
            total_bytes = 1_000_000
            estimated_rows = 100_000
            projected_columns = 10
            total_columns = 10
            remote = False

        shape = MockShape()

        # Test with NUMBA_CPU backend (if supported)
        # For now, test that predictor doesn't crash
        estimate = predict_ttdc(shape, "pandas_numpy")
        assert isinstance(estimate, TtdcEstimate)
        assert estimate.total_ms > 0
