"""
DataAccess adapter for factor value and catalog integration.

Provides optional integration with DataAccess for reading factor values
and catalog metadata. This is NOT required for FA core functionality.

FA stores identity, lineage, and evidence references — not raw factor values.
This adapter enables optional reads when factor values are needed.

This is an OPTIONAL adapter — FA core does not depend on DA.
"""

from typing import Protocol, Optional, Dict, Any, Tuple
from datetime import date, datetime
from collections.abc import Mapping

from factor_assets.adapters import OptionalDependencyMissing


class DataAccessAdapterError(RuntimeError):
    """Typed failure raised when DataAccess cannot satisfy an adapter contract."""

    code = "INFRASTRUCTURE_ERROR"

    def __init__(self, message: str, *, cause: BaseException | None = None):
        super().__init__(message)
        self.cause = cause


class DataAccessNotFoundError(DataAccessAdapterError):
    code = "NOT_FOUND"


class DataAccessUnavailableAsOfError(DataAccessAdapterError):
    code = "UNAVAILABLE_ASOF"


class DataAccessPermissionError(DataAccessAdapterError):
    code = "PERMISSION"


class DataAccessPITRejectedError(DataAccessAdapterError):
    code = "PIT_REJECTED"


class DataAccessSchemaError(DataAccessAdapterError):
    code = "SCHEMA_ERROR"


def _is_missing_factor_error(exc: BaseException) -> bool:
    """Identify DataAccess's typed error for a factor with no readable data."""
    if type(exc).__name__.lower() != "dataerror":
        return False
    text = str(exc).lower()
    return "factor" in text and any(
        marker in text
        for marker in (
            "not found",
            "missing",
            "no readable",
            "没有可读文件",
            "不存在",
        )
    )


def _raise_typed(message: str, exc: BaseException) -> None:
    if _is_missing_factor_error(exc):
        raise DataAccessNotFoundError(message, cause=exc) from exc
    name = type(exc).__name__.lower()
    if "permission" in name or "authoriz" in name:
        raise DataAccessPermissionError(message, cause=exc) from exc
    if "pit" in name or "asof" in name:
        raise DataAccessPITRejectedError(message, cause=exc) from exc
    if "schema" in name or "validation" in name:
        raise DataAccessSchemaError(message, cause=exc) from exc
    raise DataAccessAdapterError(message, cause=exc) from exc


# Try to import DataAccess - use lazy import to avoid module conflicts
DA_AVAILABLE = False
_DataAccessStore = None
_get_store = None
_ReadHandle = None
_DataRequest = None

def _try_import_da():
    """Lazy import DataAccess to avoid conflicts."""
    global DA_AVAILABLE, _DataAccessStore, _get_store, _ReadHandle, _DataRequest
    if DA_AVAILABLE or _DataAccessStore is not None:
        return
    try:
        from data_access import DataAccessStore, get_store, ReadHandle, DataRequest
        DA_AVAILABLE = True
        _DataAccessStore = DataAccessStore
        _get_store = get_store
        _ReadHandle = ReadHandle
        _DataRequest = DataRequest
    except ImportError:
        DA_AVAILABLE = False


class FactorValueReader(Protocol):
    """
    Protocol for reading factor values from DataAccess.

    FA defines this protocol; adapters implement it for DA integration.
    FA never materializes or owns factor values — this is for optional reads.
    """

    def read_factor_values(
        self,
        factor_id: str,
        start_date: date,
        end_date: date,
        universe: Optional[str] = None,
    ) -> Any:
        """
        Read factor values for a time range.

        Args:
            factor_id: Factor identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            universe: Optional universe filter

        Returns:
            Factor values (implementation-specific type)
        """
        ...

    def check_factor_availability(
        self,
        factor_id: str,
        as_of_date: Optional[date] = None,
    ) -> bool:
        """
        Check if factor values are available.

        Args:
            factor_id: Factor identifier
            as_of_date: Optional as-of date for point-in-time check

        Returns:
            True if factor is available
        """
        ...


class CatalogReader(Protocol):
    """
    Protocol for reading catalog metadata from DataAccess.

    FA may use this to validate factor availability and metadata consistency.
    """

    def get_catalog_entry(
        self,
        factor_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get catalog entry for a factor.

        Args:
            factor_id: Factor identifier

        Returns:
            Catalog entry dict or None if not found
        """
        ...

    def list_available_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        """
        List available factors matching filters.

        Args:
            filters: Optional filter criteria

        Returns:
            Tuple of factor IDs
        """
        ...


def _catalog_payload(factor_id: str, meta: Any) -> dict[str, Any]:
    to_dict = getattr(meta, "to_dict", None)
    if not callable(to_dict):
        raise DataAccessSchemaError(
            f"DataAccess catalog record {factor_id!r} has no to_dict()"
        )
    try:
        payload = to_dict()
    except Exception as exc:
        _raise_typed(f"DataAccess catalog record {factor_id!r} failed to serialize", exc)
    if not isinstance(payload, Mapping):
        raise DataAccessSchemaError(
            f"DataAccess catalog record {factor_id!r} did not produce a mapping"
        )
    payload = dict(payload)
    if payload.get("factor_id") != factor_id:
        raise DataAccessSchemaError(
            f"DataAccess catalog record key {factor_id!r} disagrees with "
            f"payload factor_id {payload.get('factor_id')!r}"
        )
    return payload


def _catalog_records(store: Any) -> Mapping[str, Any]:
    try:
        catalog = store.get_factor_catalog()
        records = getattr(catalog, "records", None)
    except Exception as exc:
        _raise_typed("DataAccess factor catalog lookup failed", exc)
    if not isinstance(records, Mapping):
        raise DataAccessSchemaError("DataAccess factor catalog has no records mapping")
    return records
def _coerce_catalog_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return date.fromisoformat(text[:10])


class DAFactorValueReader:
    """
    FactorValueReader implementation for DataAccess.

    This adapter is OPTIONAL and only needed when FA needs to read
    factor values. Core FA functionality does not require this.
    """

    def __init__(self, store: Optional["DataAccessStore"] = None):
        """
        Initialize DA factor value reader.

        Args:
            store: Optional DataAccessStore instance (uses get_store() if None)

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        _try_import_da()
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DAFactorValueReader")

        self._store = store if store is not None else _get_store()

    def read_factor_values(
        self,
        factor_id: str,
        start_date: date,
        end_date: date,
        universe: Optional[str] = None,
    ) -> Any:
        """Read and terminally materialize one factor through DataAccess."""
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        try:
            if universe is None:
                handle = self._store.read_factors(
                    [factor_id],
                    time_range=(start_date, end_date),
                )
            else:
                # DataAccess's long read_factors path accepts ``universe`` but
                # does not apply membership filtering.  read_joined is the
                # public path that performs the date/asset inner join.
                handle = self._store.read_joined(
                    "factor_lake",
                    {"factor_lake": ["datetime", "asset", "value"]},
                    time_range=(start_date, end_date),
                    universe=universe,
                    params={"factor_id": factor_id},
                )
        except Exception as exc:
            _raise_typed(f"DataAccess factor read failed for {factor_id!r}", exc)
        materialize = getattr(handle, "to_arrow", None)
        if not callable(materialize):
            raise DataAccessSchemaError(
                "DataAccess factor read did not return a materializable handle"
            )
        try:
            return materialize()
        except Exception as exc:
            _raise_typed(f"DataAccess factor materialization failed for {factor_id!r}", exc)

    def check_factor_availability(
        self,
        factor_id: str,
        as_of_date: Optional[date] = None,
    ) -> bool:
        """Check catalog metadata and confirm the factor is readable at a PIT date."""
        records = _catalog_records(self._store)
        meta = records.get(factor_id)
        if meta is None:
            return False
        status = getattr(meta, "status", None)
        if status is not None and str(status).lower() not in {"", "active", "available", "published"}:
            return False
        if as_of_date is None:
            return True
        start = _coerce_catalog_date(getattr(meta, "start_time", None))
        end = _coerce_catalog_date(getattr(meta, "end_time", None))
        if not (start is None or start <= as_of_date) or not (end is None or as_of_date <= end):
            return False
        try:
            self._store.read_factors(
                [factor_id],
                time_range=(as_of_date, as_of_date),
            )
        except Exception as exc:
            if _is_missing_factor_error(exc):
                return False
            raise
        return True



class DACatalogReader:
    """
    CatalogReader implementation for DataAccess.

    This adapter is OPTIONAL and only needed when FA needs to read
    catalog metadata. Core FA functionality does not require this.
    """

    def __init__(self, store: Optional["DataAccessStore"] = None):
        """
        Initialize DA catalog reader.

        Args:
            store: Optional DataAccessStore instance (uses get_store() if None)

        Raises:
            OptionalDependencyMissing: If DA is not available
        """
        _try_import_da()
        if not DA_AVAILABLE:
            raise OptionalDependencyMissing("data_access", "DACatalogReader")

        self._store = store if store is not None else _get_store()

    def get_catalog_entry(
        self,
        factor_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the authoritative DataAccess ``FactorMeta`` record."""
        records = _catalog_records(self._store)
        meta = records.get(factor_id)
        if meta is None:
            return None
        return _catalog_payload(factor_id, meta)


    def list_available_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        """List IDs from the authoritative catalog, applying exact filters."""
        records = _catalog_records(self._store)
        filters = filters or {}
        ids = []
        for factor_id, meta in records.items():
            payload = _catalog_payload(factor_id, meta)
            status = str(payload.get("status") or "").lower()
            if status and status not in {"active", "available", "published"}:
                continue
            if all(payload.get(key) == value for key, value in filters.items()):
                ids.append(factor_id)
        return tuple(sorted(ids))



__all__ = [
    "FactorValueReader",
    "CatalogReader",
    "DAFactorValueReader",
    "DACatalogReader",
    "DataAccessAdapterError",
    "DataAccessNotFoundError",
    "DataAccessUnavailableAsOfError",
    "DataAccessPermissionError",
    "DataAccessPITRejectedError",
    "DataAccessSchemaError",
]
