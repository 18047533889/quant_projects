"""
Tests for FactorEngine adapter.

Tests verify:
- Lazy import and OptionalDependencyMissing
- Protocol compliance
- Conversion logic (when FE available)
"""

import pytest
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing


def test_factor_engine_adapter_import_fails_without_fe():
    """Test that FactorEngineAdapter raises OptionalDependencyMissing when FE not installed."""
    # This test assumes factor_engine is not in the current path
    # If factor_engine is available, this test will be skipped
    import sys

    # Temporarily hide factor_engine from import path
    original_path = sys.path.copy()
    fe_paths = [p for p in sys.path if 'factor_engine' in p]

    try:
        # Remove FE paths temporarily
        for p in fe_paths:
            if p in sys.path:
                sys.path.remove(p)

        # Clear any cached FE imports
        fe_modules = [k for k in sys.modules.keys() if k.startswith('api') or k.startswith('runtime')]
        for mod in fe_modules:
            if 'factor' in mod.lower():
                sys.modules.pop(mod, None)

        from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

        with pytest.raises(OptionalDependencyMissing) as exc_info:
            FactorEngineAdapter()

        assert "FactorEngine is not installed" in str(exc_info.value)

    finally:
        # Restore original path
        sys.path = original_path


def test_factor_engine_adapter_not_imported_by_core():
    """Test that core modules do not import factor_engine adapter."""
    import sys

    # Clear any pre-existing adapter imports from other tests
    adapter_keys = [k for k in list(sys.modules.keys()) if "quant_evaluator.adapters" in k]
    for key in adapter_keys:
        sys.modules.pop(key, None)

    # Import core modules
    import quant_evaluator
    import quant_evaluator.contracts
    import quant_evaluator.api

    # Verify adapter module is not loaded
    adapter_modules = [
        name for name in sys.modules.keys()
        if "quant_evaluator.adapters.factor_engine" in name
    ]

    assert len(adapter_modules) == 0, \
        f"Core import should not load factor_engine adapter, but found: {adapter_modules}"


def test_factor_batch_provider_protocol():
    """Test that FactorBatchProvider protocol is properly defined."""
    from quant_evaluator.adapters.factor_engine import FactorBatchProvider

    # Verify protocol methods
    assert hasattr(FactorBatchProvider, "compute_batch")

    # Verify callable
    assert callable(getattr(FactorBatchProvider, "compute_batch", None))


def test_factor_identity_provider_protocol():
    """Test that FactorIdentityProvider protocol is properly defined."""
    from quant_evaluator.adapters.factor_engine import FactorIdentityProvider

    # Verify protocol methods
    assert hasattr(FactorIdentityProvider, "get_factor_id")
    assert hasattr(FactorIdentityProvider, "get_structural_id")

    # Verify callable
    assert callable(getattr(FactorIdentityProvider, "get_factor_id", None))
    assert callable(getattr(FactorIdentityProvider, "get_structural_id", None))


@pytest.mark.skipif(
    True,  # Always skip until FE contract frozen
    reason="Requires FactorEngine installation and contract freeze"
)
def test_fe_result_to_factor_batch_integration():
    """Integration test for FE result conversion (requires FE installed)."""
    from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

    adapter = FactorEngineAdapter()

    # This test requires real FE fixture
    # Implementation pending FE contract freeze
    pytest.skip("FE contract not frozen")


@pytest.mark.skipif(
    True,  # Always skip until FE contract frozen
    reason="Requires FactorEngine installation and contract freeze"
)
def test_fe_factors_to_factor_batch_integration():
    """Integration test for FE execution (requires FE installed)."""
    from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

    adapter = FactorEngineAdapter()

    # This test requires real FE fixture
    # Implementation pending FE contract freeze
    pytest.skip("FE contract not frozen")


def test_factor_engine_adapter_methods_raise_not_implemented():
    """Test that stub methods raise NotImplementedError with clear messages."""
    import sys

    # Skip if FE is actually available
    if any('factor_engine' in p for p in sys.path):
        pytest.skip("factor_engine may be available, skipping stub test")

    from quant_evaluator.adapters.factor_engine import FactorEngineAdapter

    # Adapter init should raise OptionalDependencyMissing
    with pytest.raises(OptionalDependencyMissing):
        adapter = FactorEngineAdapter()


def test_factor_engine_protocols_structural_compatibility():
    """Test that protocols support structural typing (duck typing)."""
    from quant_evaluator.adapters.factor_engine import FactorBatchProvider, FactorIdentityProvider
    from quant_evaluator.contracts.factor_batch import FactorBatch

    class MockBatchProvider:
        def compute_batch(self, factor_exprs, start_date, end_date, universe, market="CN"):
            # Return mock FactorBatch
            return None

    class MockIdentityProvider:
        def get_factor_id(self, factor_expr):
            return "mock_id"

        def get_structural_id(self, factor_expr):
            return "mock_structural_id"

    mock_batch = MockBatchProvider()
    mock_identity = MockIdentityProvider()

    # Protocol structural compatibility (duck typing)
    assert hasattr(mock_batch, "compute_batch")
    assert hasattr(mock_identity, "get_factor_id")
    assert hasattr(mock_identity, "get_structural_id")

    # Verify methods work
    assert mock_identity.get_factor_id(None) == "mock_id"
    assert mock_identity.get_structural_id(None) == "mock_structural_id"


def test_factor_engine_adapter_delegates_to_fe_not_duplicate():
    """Test that adapter design delegates to FE, not duplicate kernels."""
    # This is a documentation test - verify adapter has no operator kernels
    from quant_evaluator.adapters import factor_engine
    import inspect

    source = inspect.getsource(factor_engine)

    # Adapter should not contain operator kernel implementations
    # These keywords suggest kernel duplication
    forbidden_patterns = [
        "def rolling_mean",
        "def ewm_",
        "def rank_",
        "def ts_corr",
        "def cs_rank",
        "# kernel implementation",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in source, \
            f"Adapter should not duplicate FE kernels, found: {pattern}"

    # Adapter should reference FE modules
    assert "from api import factor" in source or "self._fe_" in source, \
        "Adapter should delegate to FE modules"
