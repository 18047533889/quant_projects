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

from factor_assets.adapters import OptionalDependencyMissing


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
    except (ImportError, TypeError):
        # TypeError can occur due to dataclass conflicts
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
        if universe is not None:
            raise ValueError(
                "universe filtering is not supported by the DataAccess factor-read contract"
            )
        handle = self._store.read_factors(
            [factor_id],
            time_range=(start_date, end_date),
            universe=universe,
        )
        materialize = getattr(handle, "to_arrow", None)
        if not callable(materialize):
            raise TypeError("DataAccess factor read did not return a materializable handle")
        return materialize()

    def check_factor_availability(
        self,
        factor_id: str,
        as_of_date: Optional[date] = None,
    ) -> bool:
        """Check authoritative factor metadata, optionally at an explicit PIT date."""
        catalog = self._store.get_factor_catalog()
        meta = catalog.records.get(factor_id)
        if meta is None:
            return False
        status = getattr(meta, "status", None)
        if status is not None and str(status).lower() not in {"", "active", "available", "published"}:
            return False
        if as_of_date is None:
            return True
        start = _coerce_catalog_date(getattr(meta, "start_time", None))
        end = _coerce_catalog_date(getattr(meta, "end_time", None))
        return (start is None or start <= as_of_date) and (end is None or as_of_date <= end)



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
        catalog = self._store.get_factor_catalog()
        meta = catalog.records.get(factor_id)
        if meta is None:
            return None
        return meta.to_dict() if callable(getattr(meta, "to_dict", None)) else dict(meta)


    def list_available_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ...]:
        """List IDs from the authoritative catalog, applying exact filters."""
        catalog = self._store.get_factor_catalog()
        filters = filters or {}
        ids = []
        for factor_id, meta in catalog.records.items():
            payload = meta.to_dict() if callable(getattr(meta, "to_dict", None)) else dict(meta)
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
]
