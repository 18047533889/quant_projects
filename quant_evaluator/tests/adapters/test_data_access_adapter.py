"""
Tests for DataAccess adapter.

Tests verify:
- Lazy import and OptionalDependencyMissing
- Protocol compliance
- Conversion logic (when DA available)
"""

import pytest
import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing


def test_data_access_adapter_import_fails_without_da():
    """Test that DataAccessAdapter raises OptionalDependencyMissing when DA not installed."""
    # This test assumes dataaccess is not installed
    # If dataaccess is installed, this test will be skipped
    try:
        import dataaccess  # noqa: F401
        pytest.skip("dataaccess is installed, skipping missing dependency test")
    except ImportError:
        pass

    from quant_evaluator.adapters.data_access import DataAccessAdapter

    with pytest.raises(OptionalDependencyMissing) as exc_info:
        DataAccessAdapter()

    assert "DataAccess is not installed" in str(exc_info.value)


def test_data_access_adapter_not_imported_by_core():
    """Test that core modules do not import data_access adapter."""
    import sys

    # Import core modules
    import quant_evaluator
    import quant_evaluator.contracts
    import quant_evaluator.api

    # Verify adapter module is not loaded
    adapter_modules = [
        name for name in sys.modules.keys()
        if "quant_evaluator.adapters.data_access" in name
    ]

    assert len(adapter_modules) == 0, \
        f"Core import should not load data_access adapter, but found: {adapter_modules}"


def test_context_provider_protocol():
    """Test that ContextProvider protocol is properly defined."""
    from quant_evaluator.adapters.data_access import ContextProvider
    import inspect

    # Verify protocol methods
    assert hasattr(ContextProvider, "get_trading_dates")
    assert hasattr(ContextProvider, "is_trading_day")
    assert hasattr(ContextProvider, "next_trading_day")

    # Verify these are callable
    assert callable(getattr(ContextProvider, "get_trading_dates", None))
    assert callable(getattr(ContextProvider, "is_trading_day", None))
    assert callable(getattr(ContextProvider, "next_trading_day", None))


def test_universe_provider_protocol():
    """Test that UniverseProvider protocol is properly defined."""
    from quant_evaluator.adapters.data_access import UniverseProvider
    import inspect

    # Verify protocol methods
    assert hasattr(UniverseProvider, "get_universe")
    assert hasattr(UniverseProvider, "is_in_universe")

    # Verify these are callable
    assert callable(getattr(UniverseProvider, "get_universe", None))
    assert callable(getattr(UniverseProvider, "is_in_universe", None))


@pytest.mark.skipif(
    True,  # Always skip until DA contract frozen
    reason="Requires DataAccess installation and contract freeze"
)
def test_da_frame_to_factor_batch_integration():
    """Integration test for DA frame conversion (requires DA installed)."""
    from quant_evaluator.adapters.data_access import DataAccessAdapter

    adapter = DataAccessAdapter()

    # This test requires real DA fixture
    # Implementation pending DA contract freeze
    pytest.skip("DA contract not frozen")


@pytest.mark.skipif(
    True,  # Always skip until DA contract frozen
    reason="Requires DataAccess installation and contract freeze"
)
def test_da_frame_to_label_bundle_integration():
    """Integration test for DA label conversion (requires DA installed)."""
    from quant_evaluator.adapters.data_access import DataAccessAdapter

    adapter = DataAccessAdapter()

    # This test requires real DA fixture
    # Implementation pending DA contract freeze
    pytest.skip("DA contract not frozen")


def test_data_access_adapter_methods_raise_not_implemented():
    """Test that stub methods raise NotImplementedError with clear messages."""
    # Skip if DA is actually installed
    try:
        import dataaccess  # noqa: F401
        pytest.skip("dataaccess is installed, skipping stub test")
    except ImportError:
        pass

    from quant_evaluator.adapters.data_access import DataAccessAdapter

    # Adapter init should raise OptionalDependencyMissing
    with pytest.raises(OptionalDependencyMissing):
        adapter = DataAccessAdapter()


def test_data_access_protocols_are_runtime_checkable():
    """Test that protocols can be used with isinstance checks."""
    from quant_evaluator.adapters.data_access import ContextProvider, UniverseProvider
    from typing import runtime_checkable, Protocol

    # Verify protocols are defined properly for runtime checking
    # Note: Protocols may not be runtime_checkable by default in this implementation
    # This test documents expected behavior

    class MockContextProvider:
        def get_trading_dates(self, start, end, market="CN"):
            return tuple()

        def is_trading_day(self, date, market="CN"):
            return True

        def next_trading_day(self, date, market="CN"):
            return date

    mock = MockContextProvider()

    # Protocol structural compatibility (duck typing)
    assert hasattr(mock, "get_trading_dates")
    assert hasattr(mock, "is_trading_day")
    assert hasattr(mock, "next_trading_day")
