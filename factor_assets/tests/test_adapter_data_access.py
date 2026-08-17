"""
Tests for DataAccess adapter.

Tests DA integration protocols and error handling.
"""

import pytest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from factor_assets.adapters import OptionalDependencyMissing


class TestDAFactorValueReader:
    """Test DA factor value reader."""

    def test_factor_read_uses_read_factors_and_materializes(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        class Handle:
            def to_arrow(self):
                return {"rows": 1}

        calls = []
        store = SimpleNamespace(
            read_factors=lambda *args, **kwargs: (calls.append((args, kwargs)) or Handle())
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        assert reader.read_factor_values("F1", date(2020, 1, 1), date(2020, 1, 2)) == {"rows": 1}
        assert calls[0][0] == (["F1"],)
        assert calls[0][1]["time_range"] == (date(2020, 1, 1), date(2020, 1, 2))


    def test_missing_da_raises_error(self):
        """Test that missing DA raises OptionalDependencyMissing when DA not available."""
        import factor_assets.adapters.data_access as da_mod

        # Mock _try_import_da to do nothing and keep DA_AVAILABLE = False
        def mock_try_import():
            da_mod.DA_AVAILABLE = False
            da_mod._DataAccessStore = None

        with patch.object(da_mod, '_try_import_da', mock_try_import):
            from factor_assets.adapters.data_access import DAFactorValueReader

            with pytest.raises(OptionalDependencyMissing) as exc_info:
                DAFactorValueReader()

            assert exc_info.value.package_name == "data_access"
            assert exc_info.value.adapter_name == "DAFactorValueReader"

    def test_not_implemented_error_message(self):
        """Test that error message mentions DA when not available."""
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters import OptionalDependencyMissing

        # Mock _try_import_da to do nothing and keep DA_AVAILABLE = False
        def mock_try_import():
            da_mod.DA_AVAILABLE = False
            da_mod._DataAccessStore = None

        with patch.object(da_mod, '_try_import_da', mock_try_import):
            with pytest.raises(OptionalDependencyMissing) as exc_info:
                from factor_assets.adapters.data_access import DAFactorValueReader
                DAFactorValueReader()

            assert "data_access" in str(exc_info.value)


class TestDACatalogReader:
    """Test DA catalog reader."""

    def test_catalog_metadata_and_filters_are_authoritative(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DACatalogReader

        meta = SimpleNamespace(
            status="active",
            start_time="2020-01-01",
            end_time="2020-12-31",
            to_dict=lambda: {"factor_id": "F1", "status": "active", "universe": "US"},
        )
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta, "F2": meta})
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DACatalogReader(store=store)
        assert reader.get_catalog_entry("F1")["status"] == "active"
        assert reader.get_catalog_entry("missing") is None
        assert reader.list_available_factors({"universe": "US"}) == ("F1", "F2")

    def test_list_available_factors_excludes_inactive_records(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DACatalogReader

        active = SimpleNamespace(status="active", to_dict=lambda: {"factor_id": "F1", "status": "active"})
        archived = SimpleNamespace(status="archived", to_dict=lambda: {"factor_id": "F2", "status": "archived"})
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": active, "F2": archived})
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DACatalogReader(store=store)
        assert reader.list_available_factors() == ("F1",)

    def test_availability_uses_catalog_period_without_wall_clock(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        meta = SimpleNamespace(status="active", start_time="2020-01-01", end_time="2020-12-31")
        store = SimpleNamespace(get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta}))
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        assert reader.check_factor_availability("F1", date(2020, 6, 1)) is True
        assert reader.check_factor_availability("F1", date(2021, 1, 1)) is False
        assert reader.check_factor_availability("missing") is False

    def test_universe_argument_fails_closed_until_row_filter_is_supported(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        store = SimpleNamespace()
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        with pytest.raises(ValueError, match="universe filtering is not supported"):
            reader.read_factor_values("F1", date(2020, 1, 1), date(2020, 1, 2), universe="US")

    def test_missing_da_raises_error(self):
        """Test that missing DA raises OptionalDependencyMissing when DA not available."""
        import factor_assets.adapters.data_access as da_mod

        # Mock _try_import_da to do nothing and keep DA_AVAILABLE = False
        def mock_try_import():
            da_mod.DA_AVAILABLE = False
            da_mod._DataAccessStore = None

        with patch.object(da_mod, '_try_import_da', mock_try_import):
            from factor_assets.adapters.data_access import DACatalogReader

            with pytest.raises(OptionalDependencyMissing) as exc_info:
                DACatalogReader()

            assert exc_info.value.package_name == "data_access"
            assert exc_info.value.adapter_name == "DACatalogReader"


class TestProtocols:
    """Test protocol definitions."""

    def test_factor_value_reader_protocol(self):
        """Test that FactorValueReader protocol is defined correctly."""
        from factor_assets.adapters.data_access import FactorValueReader
        from typing import get_type_hints
        import inspect

        # Verify protocol has required methods
        assert hasattr(FactorValueReader, 'read_factor_values')
        assert hasattr(FactorValueReader, 'check_factor_availability')

        # Verify read_factor_values signature
        sig = inspect.signature(FactorValueReader.read_factor_values)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'factor_id' in params
        assert 'start_date' in params
        assert 'end_date' in params
        assert 'universe' in params

    def test_catalog_reader_protocol(self):
        """Test that CatalogReader protocol is defined correctly."""
        from factor_assets.adapters.data_access import CatalogReader
        import inspect

        # Verify protocol has required methods
        assert hasattr(CatalogReader, 'get_catalog_entry')
        assert hasattr(CatalogReader, 'list_available_factors')

        # Verify get_catalog_entry signature
        sig = inspect.signature(CatalogReader.get_catalog_entry)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'factor_id' in params

    def test_mock_implementation(self):
        """Test that we can create mock implementations of protocols."""
        from factor_assets.adapters.data_access import FactorValueReader, CatalogReader
        from typing import Optional, Dict, Any, Tuple

        # Create mock implementations
        class MockFactorValueReader:
            def read_factor_values(
                self,
                factor_id: str,
                start_date: date,
                end_date: date,
                universe: Optional[str] = None,
            ) -> Any:
                return {"mock": "data"}

            def check_factor_availability(
                self,
                factor_id: str,
                as_of_date: Optional[date] = None,
            ) -> bool:
                return True

        class MockCatalogReader:
            def get_catalog_entry(
                self,
                factor_id: str,
            ) -> Optional[Dict[str, Any]]:
                return {"factor_id": factor_id}

            def list_available_factors(
                self,
                filters: Optional[Dict[str, Any]] = None,
            ) -> Tuple[str, ...]:
                return ("F1", "F2")

        # Verify mock implementations satisfy protocols
        mock_reader = MockFactorValueReader()
        mock_catalog = MockCatalogReader()

        # Test mock behavior
        assert mock_reader.read_factor_values(
            "F123",
            date(2020, 1, 1),
            date(2020, 12, 31),
        ) == {"mock": "data"}
        assert mock_reader.check_factor_availability("F123") is True
        assert mock_catalog.get_catalog_entry("F123") == {"factor_id": "F123"}
        assert mock_catalog.list_available_factors() == ("F1", "F2")


class TestAdapterIntegration:
    """Test adapter integration patterns."""

    def test_graceful_import_handling(self):
        """Test that missing DA can be handled gracefully."""
        import factor_assets.adapters.data_access as da_mod

        # Mock _try_import_da to do nothing and keep DA_AVAILABLE = False
        def mock_try_import():
            da_mod.DA_AVAILABLE = False
            da_mod._DataAccessStore = None

        with patch.object(da_mod, '_try_import_da', mock_try_import):
            try:
                from factor_assets.adapters.data_access import DAFactorValueReader
                reader = DAFactorValueReader()
                assert False, "Should have raised OptionalDependencyMissing"
            except OptionalDependencyMissing as e:
                # This is expected
                assert "data_access" in str(e)
                assert "pip install factor_assets[adapters]" in str(e)

    def test_protocol_based_design(self):
        """Test that protocols enable testing without real DA."""
        from factor_assets.adapters.data_access import FactorValueReader
        from typing import Optional, Any

        # Create test implementation
        class TestFactorValueReader:
            def __init__(self):
                self.calls = []

            def read_factor_values(
                self,
                factor_id: str,
                start_date: date,
                end_date: date,
                universe: Optional[str] = None,
            ) -> Any:
                self.calls.append(("read", factor_id, start_date, end_date, universe))
                return {"factor_id": factor_id, "rows": 100}

            def check_factor_availability(
                self,
                factor_id: str,
                as_of_date: Optional[date] = None,
            ) -> bool:
                self.calls.append(("check", factor_id, as_of_date))
                return True

        # Use test implementation
        reader = TestFactorValueReader()

        result = reader.read_factor_values(
            "F123",
            date(2020, 1, 1),
            date(2020, 12, 31),
            universe="US_STOCKS",
        )

        assert result["factor_id"] == "F123"
        assert len(reader.calls) == 1
        assert reader.calls[0][0] == "read"
        assert reader.calls[0][1] == "F123"

        available = reader.check_factor_availability("F123", date(2020, 1, 1))
        assert available is True
        assert len(reader.calls) == 2


class TestErrorMessages:
    """Test error message quality."""

    def test_optional_dependency_missing_message(self):
        """Test that error messages are helpful."""
        from factor_assets.adapters import OptionalDependencyMissing

        error = OptionalDependencyMissing("data_access", "DAFactorValueReader")

        message = str(error)
        assert "data_access" in message
        assert "DAFactorValueReader" in message
        assert "pip install factor_assets[adapters]" in message

    def test_different_adapters_different_messages(self):
        """Test that different adapters have distinct error messages."""
        from factor_assets.adapters import OptionalDependencyMissing

        qe_error = OptionalDependencyMissing("quant_evaluator", "QEEvidenceProvider")
        fe_error = OptionalDependencyMissing("factor_engine", "FEIdentityProvider")
        da_error = OptionalDependencyMissing("data_access", "DAFactorValueReader")

        assert "quant_evaluator" in str(qe_error)
        assert "factor_engine" in str(fe_error)
        assert "data_access" in str(da_error)

        assert "QEEvidenceProvider" in str(qe_error)
        assert "FEIdentityProvider" in str(fe_error)
        assert "DAFactorValueReader" in str(da_error)
