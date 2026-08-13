"""
Test fixtures for integration testing.

Provides:
- synthetic_data: Generate realistic factor/label/exposure data
- mock_adapters: Mock DA/FE adapters for testing
- test_helpers: Assertion helpers for comparing results
"""

from .synthetic_data import (
    generate_factor_panel,
    generate_label_bundle,
    generate_exposure_matrix,
    generate_correlated_factors,
    generate_realistic_market_data,
    PanelConfig,
    DataCharacteristics,
)

from .mock_adapters import (
    MockDataAccessAdapter,
    MockFactorEngineAdapter,
    MockEvaluationBackend,
    AdapterResponse,
    AdapterStatus,
    create_stable_adapters,
    create_flaky_adapters,
    create_slow_adapters,
)

from .test_helpers import (
    assert_panel_shape,
    assert_factor_properties,
    assert_timing_consistency,
    assert_numeric_close,
    assert_no_lookahead,
    compare_evaluation_results,
    assert_correlation_structure,
    assert_cross_sectional_neutrality,
    assert_panel_aligned,
    assert_monotonic_increasing,
    assert_stationary,
    summarize_panel,
)

__all__ = [
    # Synthetic data generation
    "generate_factor_panel",
    "generate_label_bundle",
    "generate_exposure_matrix",
    "generate_correlated_factors",
    "generate_realistic_market_data",
    "PanelConfig",
    "DataCharacteristics",
    # Mock adapters
    "MockDataAccessAdapter",
    "MockFactorEngineAdapter",
    "MockEvaluationBackend",
    "AdapterResponse",
    "AdapterStatus",
    "create_stable_adapters",
    "create_flaky_adapters",
    "create_slow_adapters",
    # Test helpers
    "assert_panel_shape",
    "assert_factor_properties",
    "assert_timing_consistency",
    "assert_numeric_close",
    "assert_no_lookahead",
    "compare_evaluation_results",
    "assert_correlation_structure",
    "assert_cross_sectional_neutrality",
    "assert_panel_aligned",
    "assert_monotonic_increasing",
    "assert_stationary",
    "summarize_panel",
]
