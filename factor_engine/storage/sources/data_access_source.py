"""通过 ``data_access.get_store()`` 读取登记数据集并绑定快照缓存。"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from typing import Any, Iterable

from logging_utils import get_logger
from workspace_paths import quant_projects_root

from .datasource import DataSource

logger = get_logger("storage.data_access_source")


class DataAccessColumnPreflightError(ValueError):
    """A logical formula dependency is not a valid physical column request."""


class MissingDataDependencyError(DataAccessColumnPreflightError):
    """A formula requires another logical dataset or an explicitly derived field."""


def _ensure_data_access_importable() -> None:
    root = str(quant_projects_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_store():
    _ensure_data_access_importable()
    from data_access import get_store
    return get_store()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(1, value)


class DataAccessSource(DataSource):
    """FactorEngine DataAccess source with snapshot-bound caches and preflight."""

    def __init__(
        self,
        *,
        dataset: str,
        fields: dict[str, str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        instrument_filter: list[str] | None = None,
        normalize_timestamp: bool | None = None,
        timestamp_unit: str | None = None,
        read_auto: bool | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.fields = dict(fields or {})
        self.start_date = start_date
        self.end_date = end_date
        self.instrument_filter = list(instrument_filter) if instrument_filter else None
        self.normalize_timestamp = normalize_timestamp
        self.timestamp_unit = timestamp_unit
        self.params = dict(params or {})
        self.read_auto = (
            bool(read_auto)
            if read_auto is not None
            else bool(self.params.pop("read_auto", False))
        )
        self._lazy_scan = bool(self.params.pop("lazy_scan", False))
        self._column_cache: OrderedDict[str, Any] = OrderedDict()
        self._panel_cache: OrderedDict[str, Any] = OrderedDict()
        self._lazy_bundle: Any | None = None
        self._data_snapshot_id: str | None = None
        self._snapshot_checked_at = 0.0
        self._snapshot_ttl_seconds = float(
            os.environ.get("FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS", "60") or 60
        )
        self._max_cache_columns = _positive_int_env(
            "FACTOR_ENGINE_DATA_CACHE_MAX_COLUMNS", 64
        )
        self._closed = False

    @property
    def data_snapshot_id(self) -> str | None:
        return self._data_snapshot_id

    @property
    def lazy_scan(self) -> bool:
        return bool(self._lazy_scan)

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("DataAccessSource is closed")

    def _time_range(self) -> tuple[Any, Any] | None:
        if self.start_date is None and self.end_date is None:
            return None
        return (self.start_date, self.end_date)

    def _preflight_logical_columns(self, names: Iterable[str]) -> None:
        """Reject known semantic mistakes before they become DuckDB Binder errors.

        The dataset registry schema is intentionally *not* used as a hard gate
        here because the LQTP StockDailyBar mirror can contain real columns that
        lag the checked-in schema documentation.  We instead fail on two cases
        that are unambiguously wrong:

        * an active FactorEngine operator name was emitted as a bare column;
        * a known LQTP derived/multi-source dependency was emitted as a daily-bar
          physical column.
        """
        requested = [str(name) for name in names]
        from cleaned_operators.production_tiers import LQTP_SOURCE_DEPENDENT_NAMES

        derived = sorted(
            name for name in requested
            if name in LQTP_SOURCE_DEPENDENT_NAMES and name not in self.fields
        )
        if derived:
            raise MissingDataDependencyError(
                f"dataset={self.dataset!r} cannot satisfy derived/source-backed fields "
                f"{derived}; configure a composite/financial/valuation source or a "
                "versioned derived-field definition instead of querying StockDailyBar"
            )

        try:
            from backend.cleaned_bridge import ensure_cleaned_loaded
            from cleaned_operators.registry import OperatorRegistry

            ensure_cleaned_loaded()
            operator_names = set(OperatorRegistry.list_canonical()) | set(OperatorRegistry._aliases)
        except Exception:
            operator_names = set()
        mistaken_ops = sorted(
            name for name in requested
            if name in operator_names and name not in self.fields
        )
        if mistaken_ops:
            raise DataAccessColumnPreflightError(
                f"dataset={self.dataset!r} received operator name(s) as physical columns: "
                f"{mistaken_ops}. Expand the formula/template before data access."
            )

    def _resolve_columns(self, names: Iterable[str]) -> tuple[list[str], dict[str, str]]:
        names = list(names)
        self._preflight_logical_columns(names)
        physical: list[str] = []
        output_names: dict[str, str] = {}
        for name in names:
            src = self.fields.get(name, name)
            physical.append(src)
            if src != name:
                output_names[src] = name
        return physical, output_names

    def _record_read_snapshot(self, snapshot_id: str | None) -> None:
        if not snapshot_id:
            return
        if self._data_snapshot_id and snapshot_id != self._data_snapshot_id:
            self.clear_cache(reset_snapshot=False)
        self._data_snapshot_id = snapshot_id
        self._snapshot_checked_at = time.monotonic()

    def refresh_snapshot(self, *, force: bool = False) -> str | None:
        self._assert_open()
        now = time.monotonic()
        if (
            not force
            and self._data_snapshot_id is not None
            and now - self._snapshot_checked_at < self._snapshot_ttl_seconds
        ):
            return self._data_snapshot_id
        store = _get_store()
        snapshot = store.describe_dataset(
            self.dataset,
            params=dict(self.params),
            instrument_filter=self.instrument_filter,
        )
        current = snapshot.snapshot_id
        if self._data_snapshot_id and current != self._data_snapshot_id:
            logger.info(
                "data_access snapshot changed dataset=%s old=%s new=%s; clearing caches",
                self.dataset,
                self._data_snapshot_id,
                current,
            )
            self.clear_cache(reset_snapshot=False)
        self._data_snapshot_id = current
        self._snapshot_checked_at = now
        return current

    def clear_cache(self, *, reset_snapshot: bool = True) -> None:
        self._column_cache.clear()
        self._panel_cache.clear()
        self._lazy_bundle = None
        if reset_snapshot:
            self._data_snapshot_id = None
            self._snapshot_checked_at = 0.0

    def close(self) -> None:
        self.clear_cache()
        self._closed = True

    def _put_cache(self, cache: OrderedDict[str, Any], name: str, value: Any) -> None:
        cache[name] = value
        cache.move_to_end(name)
        while len(cache) > self._max_cache_columns:
            cache.popitem(last=False)

    def column_cache_stats(self) -> dict[str, int]:
        return {
            "cached_columns": len(self._column_cache),
            "cached_panels": len(self._panel_cache),
            "max_cache_columns": self._max_cache_columns,
        }

    def enable_lazy_scan(self, enabled: bool = True) -> None:
        self._assert_open()
        enabled = bool(enabled)
        if enabled != self._lazy_scan:
            self.clear_cache(reset_snapshot=False)
        self._lazy_scan = enabled
        if enabled:
            self.read_auto = True

    def read_session(self) -> "DataSourceReadSession":
        from .read_session import DataSourceReadSession
        return DataSourceReadSession(self)

    def load_column(self, name: str):
        self.refresh_snapshot()
        if name in self._column_cache:
            self._column_cache.move_to_end(name)
            return self._column_cache[name]
        return self.load_columns([name])[name]

    def _adapter_options(self, store):
        from data_access.store import adapter_options_for_dataset

        ds = store.get_dataset(self.dataset)
        options = adapter_options_for_dataset(ds)
        normalize = (
            self.normalize_timestamp
            if self.normalize_timestamp is not None
            else bool(options.get("normalize_timestamp", False))
        )
        unit = self.timestamp_unit or options.get("timestamp_unit")
        return ds, normalize, unit

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        self.refresh_snapshot()
        needed = [name for name in names if name not in self._column_cache]
        if not needed:
            for name in names:
                self._column_cache.move_to_end(name)
            return {name: self._column_cache[name] for name in names}

        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds, normalize, unit = self._adapter_options(store)
        logger.info(
            "data_access dataset=%s columns=%s time_range=%s lazy=%s",
            self.dataset,
            needed,
            self._time_range(),
            self._lazy_scan,
        )

        if self.read_auto and self._lazy_scan:
            from backend.polars_lazy import scan_dataset_columns

            fetched = scan_dataset_columns(
                store,
                self.dataset,
                physical_columns=physical,
                time_column=ds.time_column,
                instrument_column=ds.instrument_column,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                output_names=output_names or None,
                normalize_timestamp=normalize,
                timestamp_unit=unit,
                params=dict(self.params),
                bundle=self._lazy_bundle,
            )
            if self._lazy_bundle is not None:
                self._record_read_snapshot(self._lazy_bundle.snapshot_id)
        else:
            from data_access.read.adapters import arrow_table_to_multiindex_columns

            all_columns = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            result = store.read_result(
                self.dataset,
                columns=all_columns,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                **self.params,
            )
            self._record_read_snapshot(result.snapshot.snapshot_id)
            fetched = arrow_table_to_multiindex_columns(
                result.table,
                timestamp_column=ds.time_column,
                instrument_column=ds.instrument_column,
                value_columns=physical,
                output_names=output_names or None,
                normalize_timestamp=normalize,
                timestamp_unit=unit,
            )

        for name in needed:
            self._put_cache(self._column_cache, name, fetched[name])
        return {name: self._column_cache[name] for name in names}

    def prefetch_columns(self, names: list[str]) -> None:
        if not names:
            return
        self.refresh_snapshot()
        if self.read_auto and self._lazy_scan:
            self._prefetch_lazy_bundle(names)
        else:
            self.load_columns(names)

    def _prefetch_lazy_bundle(self, names: list[str]) -> None:
        needed = [name for name in names if name not in self._column_cache]
        if not needed:
            return
        physical, output_names = self._resolve_columns(needed)
        store = _get_store()
        ds, normalize, unit = self._adapter_options(store)
        from backend.polars_lazy import build_lazy_column_bundle

        if self._lazy_bundle is None:
            merged_physical = physical
            merged_output = output_names
        else:
            merged_physical = list(dict.fromkeys([*self._lazy_bundle.physical_columns, *physical]))
            merged_output = dict(self._lazy_bundle.output_names)
            merged_output.update(output_names)
            if merged_physical == list(self._lazy_bundle.physical_columns):
                fetched = self._lazy_bundle.materialize_columns(
                    physical, output_names=output_names or None
                )
                for name in needed:
                    self._put_cache(self._column_cache, name, fetched[name])
                return

        self._lazy_bundle = build_lazy_column_bundle(
            store,
            self.dataset,
            physical_columns=merged_physical,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            output_names=merged_output or None,
            normalize_timestamp=normalize,
            timestamp_unit=unit,
            params=dict(self.params),
        )
        self._record_read_snapshot(self._lazy_bundle.snapshot_id)
        fetched = self._lazy_bundle.materialize_columns(
            physical, output_names=output_names or None
        )
        for name in needed:
            self._put_cache(self._column_cache, name, fetched[name])

    def prefetch_panels(self, names: list[str]) -> None:
        self.refresh_snapshot()
        needed = [name for name in names if name not in self._panel_cache]
        if not needed:
            return
        batch = self.load_columns(needed)
        for name, series in batch.items():
            level = series.index.names[-1] or "instrument"
            self._put_cache(self._panel_cache, name, series.unstack(level=level))

    def load_column_panel(self, name: str):
        self.refresh_snapshot()
        if name in self._panel_cache:
            self._panel_cache.move_to_end(name)
            return self._panel_cache[name]
        series = self.load_column(name)
        level = series.index.names[-1] or "instrument"
        panel = series.unstack(level=level)
        self._put_cache(self._panel_cache, name, panel)
        return panel

    def dataset_axis_columns(self) -> tuple[str, str]:
        return _get_store().dataset_axis_columns(self.dataset)

    def scan_polars_long(self, columns: list[str]):
        self.refresh_snapshot()
        physical, output_names = self._resolve_columns(columns)
        store = _get_store()
        ds = store.get_dataset(self.dataset)
        from backend.polars_lazy import build_scan_polars_long

        return build_scan_polars_long(
            store,
            self.dataset,
            logical_columns=columns,
            physical_columns=physical,
            output_names=output_names,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
        )

    def scan_index_long(self):
        self.refresh_snapshot()
        store = _get_store()
        ds = store.get_dataset(self.dataset)
        from backend.polars_lazy import build_scan_index_long

        return build_scan_index_long(
            store,
            self.dataset,
            time_column=ds.time_column,
            instrument_column=ds.instrument_column,
            time_range=self._time_range(),
            instrument_filter=self.instrument_filter,
            params=dict(self.params),
        )
