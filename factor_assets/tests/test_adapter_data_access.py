"""
Tests for DataAccess adapter.

Tests DA integration protocols and error handling.
"""

import pytest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from factor_assets.adapters import OptionalDependencyMissing


class _DatedUniverseStore:
    def _resolve_universe_instruments(self, universe, time_range=None, instruments=None):
        assert universe == "universe:history"
        day = time_range[0]
        members = {
            date(2020, 1, 2): ["LIVE", "DELISTED_LATER"],
            date(2021, 1, 2): ["LIVE", "NEW_MEMBER"],
        }[day]
        return members


def test_resolve_universe_snapshot_membership_changes_with_date():
    from factor_assets.adapters.data_access import resolve_universe_snapshot
    store = _DatedUniverseStore()
    old = resolve_universe_snapshot(store, "universe:history", as_of=date(2020, 1, 2))
    new = resolve_universe_snapshot(store, "universe:history", as_of=date(2021, 1, 2))
    assert old.snapshot_id != new.snapshot_id
    assert old.eligible_members == ("DELISTED_LATER", "LIVE")
    assert new.eligible_members == ("LIVE", "NEW_MEMBER")


def test_historically_eligible_delisted_member_is_preserved():
    from factor_assets.adapters.data_access import resolve_universe_snapshot
    resolved = resolve_universe_snapshot(
        _DatedUniverseStore(), "universe:history", as_of=date(2020, 1, 2)
    )
    assert "DELISTED_LATER" in resolved.eligible_members


def test_unresolved_universe_snapshot_fails_closed():
    from factor_assets.adapters.data_access import (
        DataAccessUnavailableAsOfError, resolve_universe_snapshot,
    )
    with pytest.raises(DataAccessUnavailableAsOfError):
        resolve_universe_snapshot(SimpleNamespace(), "universe:missing", as_of=date(2020, 1, 2))


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
        assert "universe" not in calls[0][1]

    def test_import_type_error_is_not_reported_as_missing_dependency(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        def broken_import():
            raise TypeError("broken DataAccess ABI")

        with patch.object(da_mod, "_try_import_da", broken_import):
            with pytest.raises(TypeError, match="broken DataAccess ABI"):
                DAFactorValueReader()

    def test_non_materializable_handle_is_typed_schema_error(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader, DataAccessSchemaError

        store = SimpleNamespace(read_factors=lambda *args, **kwargs: object())
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        with pytest.raises(DataAccessSchemaError):
            reader.read_factor_values("F1", date(2020, 1, 1), date(2020, 1, 2))

    def test_missing_factor_data_error_is_typed_not_found(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import (
            DAFactorValueReader,
            DataAccessNotFoundError,
        )

        class DataError(Exception):
            pass

        def read_missing_factor(*args, **kwargs):
            raise DataError("read_factors: factor ['missing_factor'] 都没有可读文件")

        store = SimpleNamespace(read_factors=read_missing_factor)
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)

        with pytest.raises(DataAccessNotFoundError) as exc_info:
            reader.read_factor_values("missing_factor", date(2020, 1, 1), date(2020, 1, 2))
        assert isinstance(exc_info.value.cause, DataError)

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
        meta_two = SimpleNamespace(
            status="active",
            to_dict=lambda: {"factor_id": "F2", "status": "active", "universe": "US"},
        )
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta, "F2": meta_two})
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DACatalogReader(store=store)
        assert reader.get_catalog_entry("F1")["status"] == "active"
        assert reader.get_catalog_entry("missing") is None
        assert reader.list_available_factors({"universe": "US"}) == ("F1", "F2")

    def test_catalog_key_payload_factor_id_mismatch_is_schema_error(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DACatalogReader, DataAccessSchemaError

        meta = SimpleNamespace(to_dict=lambda: {"factor_id": "OTHER", "status": "active"})
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta})
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DACatalogReader(store=store)

        with pytest.raises(DataAccessSchemaError, match="disagrees"):
            reader.get_catalog_entry("F1")
        with pytest.raises(DataAccessSchemaError, match="disagrees"):
            reader.list_available_factors()

    def test_catalog_serialization_failure_is_typed(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DACatalogReader, DataAccessAdapterError

        def broken_to_dict():
            raise RuntimeError("catalog serialization failed")

        meta = SimpleNamespace(to_dict=broken_to_dict)
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta})
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DACatalogReader(store=store)
        with pytest.raises(DataAccessAdapterError) as exc_info:
            reader.get_catalog_entry("F1")
        assert exc_info.value.cause is not None
        assert str(exc_info.value.cause) == "catalog serialization failed"

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
        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta}),
            read_factors=lambda *args, **kwargs: object(),
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        assert reader.check_factor_availability("F1", date(2020, 6, 1)) is True
        assert reader.check_factor_availability("F1", date(2021, 1, 1)) is False
        assert reader.check_factor_availability("missing") is False

    def test_availability_fails_closed_when_catalog_is_stale(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        class DataError(Exception):
            pass

        meta = SimpleNamespace(status="active", start_time="2020-01-01", end_time="2020-12-31")

        def read_missing_factor(*args, **kwargs):
            raise DataError("read_factors: factor ['F1'] missing underlying data")

        store = SimpleNamespace(
            get_factor_catalog=lambda: SimpleNamespace(records={"F1": meta}),
            read_factors=read_missing_factor,
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)

        assert reader.check_factor_availability("F1", date(2020, 6, 1)) is False

    def test_universe_filters_long_layout_rows(self):
        import factor_assets.adapters.data_access as da_mod
        from factor_assets.adapters.data_access import DAFactorValueReader

        class Handle:
            def __init__(self, rows):
                self.rows = rows

            def to_arrow(self):
                return self.rows

        calls = []

        def read_factors(*args, **kwargs):
            calls.append(("read_factors", args, kwargs))
            return Handle([{"asset": "A", "universe": "A"}, {"asset": "B", "universe": "B"}])

        def read_joined(*args, **kwargs):
            calls.append(("read_joined", args, kwargs))
            return Handle([{"asset": "A", "universe": "A"}])

        store = SimpleNamespace(
            read_factors=read_factors,
            read_joined=read_joined,
            _resolve_universe_instruments=lambda universe, time_range, instruments: ["A"],
        )
        with patch.object(da_mod, "_try_import_da", lambda: setattr(da_mod, "DA_AVAILABLE", True)):
            reader = DAFactorValueReader(store=store)
        result = reader.read_factor_values(
            "F1", date(2020, 1, 1), date(2020, 1, 2), universe="A"
        )
        assert result == [{"asset": "A", "universe": "A"}]
        assert calls[0][0] == "read_joined"
        assert calls[0][2]["universe"] == "A"
        assert calls[0][2]["params"] == {"factor_id": "F1"}

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
                assert "pip install factor_assets[data_access]" in str(e)

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
        assert "pip install factor_assets[data_access]" in message

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
