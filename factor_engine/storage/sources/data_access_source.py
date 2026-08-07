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


class UnknownFieldSemanticError(DataAccessColumnPreflightError):
    """A requested field has no contract in the field registry for this dataset.

    Raised only when the source runs with ``strict_unknown_fields`` (production).
    Research mode may explicitly allow raw physical columns instead.
    """


class FieldNormalizationError(DataAccessColumnPreflightError):
    """A registered field could not be normalized to its canonical unit/scale."""


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
        semantic_filters: dict[str, Any] | None = None,
        read_mode: str = "panel",
        strict_unknown_fields: bool | None = None,
    ) -> None:
        self.dataset = dataset
        self.fields = dict(fields or {})
        # Fail-closed field contracts in production (unknown fields and
        # normalization errors raise).  Default None → auto-detect the engine run
        # mode so research stays lenient (raw physical-column fallback allowed).
        if strict_unknown_fields is None:
            try:
                from runtime.production_policy import is_production_mode

                strict_unknown_fields = bool(is_production_mode())
            except Exception:
                strict_unknown_fields = False
        self.strict_unknown_fields = bool(strict_unknown_fields)
        self.start_date = start_date
        self.end_date = end_date
        self.instrument_filter = list(instrument_filter) if instrument_filter else None
        self.normalize_timestamp = normalize_timestamp
        self.timestamp_unit = timestamp_unit
        self.params = dict(params or {})
        self.semantic_filters = dict(semantic_filters or {})
        self.read_mode = str(read_mode or "panel").lower()
        self._validate_semantic_contract()
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
        #: 廉价的 manifest 版本 token（读 _manifest.json sidecar），用于 TTL 内
        #: 判断数据是否变化；与真实 DataSnapshot id 分开跟踪（格式不同，不能互比）。
        self._manifest_token: str | None = None
        self._snapshot_checked_at = 0.0
        self._snapshot_ttl_seconds = float(
            os.environ.get("FACTOR_ENGINE_DATA_SNAPSHOT_TTL_SECONDS", "60") or 60
        )
        self._max_cache_columns = _positive_int_env(
            "FACTOR_ENGINE_DATA_CACHE_MAX_COLUMNS", 64
        )
        self._closed = False

    def _validate_semantic_contract(self) -> None:
        """Apply COS panel/event and required-filter policy at construction."""
        try:
            _ensure_data_access_importable()
            from data_access.cos_contract import (
                get_cos_contract,
                resolve_event_clock,
                validate_event_filters,
                validate_panel_request,
            )
        except ImportError:
            return
        contract = get_cos_contract(self.dataset)
        if contract is None:
            return
        if self.read_mode == "panel":
            validate_panel_request(
                self.dataset,
                semantic_filters=self.semantic_filters,
            )
        elif self.read_mode in {"event", "pit"}:
            resolve_event_clock(
                self.dataset,
                allow_effective_time=self.read_mode == "event",
            )
            validate_event_filters(contract, self.semantic_filters)
        else:
            raise ValueError("read_mode must be panel, event, or pit")
        for name, value in self.semantic_filters.items():
            existing = self.params.get(name)
            if existing is not None and existing != value:
                raise ValueError(
                    f"conflicting semantic filter {name}: params={existing!r} filter={value!r}"
                )
            self.params[name] = value

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
            src = self.fields.get(name)
            spec = None
            if src is None:
                try:
                    from fields import FIELD_REGISTRY

                    spec = FIELD_REGISTRY.get(name, table=self.dataset)
                except Exception:
                    spec = None
                if spec is not None:
                    if spec.dataset == self.dataset:
                        src = spec.source_name
                    elif self.strict_unknown_fields:
                        raise UnknownFieldSemanticError(
                            f"dataset={self.dataset!r} has no field {name!r}: it belongs "
                            f"to dataset {spec.dataset!r}"
                        )
            if src is None and self.strict_unknown_fields and name not in self.fields:
                # Production fail-closed: a request that matches neither an explicit
                # alias mapping nor a registered field of this dataset has no unit /
                # PIT / temporal contract and must not silently pass through as a raw
                # physical column.  Research keeps the raw fallback via
                # strict_unknown_fields=False.
                raise UnknownFieldSemanticError(
                    f"dataset={self.dataset!r} has no registered field {name!r}; "
                    "set strict_unknown_fields=False (research) to allow the raw "
                    "physical column"
                )
            src = src or name
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

    def _query_scoped_snapshot_token(self, store) -> str | None:
        """廉价 query-scoped snapshot token：优先读 ``_manifest.json`` sidecar。

        几十微秒级（不 glob 全量文件、不读 footer）。返回 ``None`` 表示没有可用
        manifest（调用方回退到 ``describe_dataset``）。
        """
        try:
            version = store.manifest_version(self.dataset, **self.params)
        except Exception:
            return None
        if not version.get("has_manifest") or not version.get("fresh"):
            return None
        dv = version.get("dataset_version")
        pv = version.get("partition_version")
        if not dv or not pv:
            return None
        return f"manifest:{dv}:{pv}"

    def refresh_snapshot(self, *, force: bool = False) -> str | None:
        """proactive 快照刷新（TTL 内短路）。

        **优先走廉价 manifest token**（``store.manifest_version``），避免每隔 TTL
        对整个 dataset 做昂贵 ``describe_dataset`` 再执行一次实际 read；只有没有
        manifest 时才回退 describe。返回最近一次的数据快照身份（生产路径只消费
        这里的缓存失效副作用）。
        """
        self._assert_open()
        now = time.monotonic()
        if (
            not force
            and self._data_snapshot_id is not None
            and now - self._snapshot_checked_at < self._snapshot_ttl_seconds
        ):
            return self._data_snapshot_id
        store = _get_store()
        token = self._query_scoped_snapshot_token(store)
        if token is not None:
            # 廉价路径：manifest 版本变了才清缓存（token 与真实 snapshot id 分开跟踪）
            if self._manifest_token is not None and token != self._manifest_token:
                logger.info(
                    "data_access manifest version changed dataset=%s old=%s new=%s; clearing caches",
                    self.dataset,
                    self._manifest_token,
                    token,
                )
                self.clear_cache(reset_snapshot=False)
            self._manifest_token = token
        else:
            # 无 manifest：回退全量 describe（旧行为，仅此路径昂贵）
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
        return self._data_snapshot_id

    def clear_cache(self, *, reset_snapshot: bool = True) -> None:
        self._column_cache.clear()
        self._panel_cache.clear()
        self._lazy_bundle = None
        if reset_snapshot:
            self._data_snapshot_id = None
            self._manifest_token = None
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

        if self._lazy_scan:
            # 显式 polars-lazy 优化（collect 前表达式仍在 DuckDB 内下推）
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
            # 引擎/结果形态交给 DataAccess 成本路由（read_auto 语义下沉到 DataAccess）。
            from data_access.read.adapters import arrow_table_to_multiindex_columns

            all_columns = list(dict.fromkeys([ds.time_column, ds.instrument_column, *physical]))
            handle = store.read(
                self.dataset,
                columns=all_columns,
                time_range=self._time_range(),
                instrument_filter=self.instrument_filter,
                **self.params,
            )
            self._record_read_snapshot(getattr(handle.snapshot, "snapshot_id", None))
            fetched = arrow_table_to_multiindex_columns(
                handle.to_arrow(),
                timestamp_column=ds.time_column,
                instrument_column=ds.instrument_column,
                value_columns=physical,
                output_names=output_names or None,
                normalize_timestamp=normalize,
                timestamp_unit=unit,
            )

        # DataAccess exposes raw COS values for ``load_columns``.  FactorEngine
        # must use the semantic contract before caching a logical factor input;
        # otherwise A-share Return remains in 1/10000 units and silently
        # contaminates every downstream return-based operator.
        self._normalize_contract_columns(fetched, needed)
        for name in needed:
            self._put_cache(self._column_cache, name, fetched[name])
        return {name: self._column_cache[name] for name in names}

    def _normalize_contract_columns(self, fetched: dict[str, Any], names: list[str]) -> None:
        """Normalize every registered logical field before it enters the cache.

        Registered fields are scaled from ``source_unit`` to the canonical unit
        (e.g. A-share Return bp → ratio, TurnoverRatio/ROE/ShareRatio/Weight % →
        ratio).  Production is fail-closed: a unit/scale failure raises instead of
        silently caching the raw vendor value (which would contaminate every
        downstream operator with a 10000× or 100× error).
        """
        normalized: set[str] = set()
        try:
            from fields import FIELD_REGISTRY

            for name in names:
                spec = FIELD_REGISTRY.get(name)
                if spec is None or spec.dataset != self.dataset or name not in fetched:
                    continue
                scale = float(spec.scale_to_canonical or 1.0)
                if scale != 1.0:
                    fetched[name] = fetched[name] * scale
                normalized.add(name)
        except Exception as exc:
            if self.strict_unknown_fields:
                raise FieldNormalizationError(
                    f"failed to normalize field contracts for dataset={self.dataset!r}: {exc}"
                ) from exc
            logger.warning("field normalization skipped for %s: %s", self.dataset, exc)

        # Compatibility for external DataAccess contracts that are not represented
        # in FactorEngine's field registry yet.
        try:
            from data_access.cos_contract import get_cos_contract, normalize_return_values
        except Exception:
            return
        contract = get_cos_contract(self.dataset)
        if contract is None or not contract.return_column:
            return
        for name in names:
            if name in normalized or name not in fetched:
                continue
            physical_name = self.fields.get(name)
            if physical_name is None:
                try:
                    from fields import FIELD_REGISTRY

                    spec = FIELD_REGISTRY.get(name, table=self.dataset)
                    if spec is not None and spec.dataset == self.dataset:
                        physical_name = spec.source_name
                except Exception:
                    physical_name = None
            physical_name = physical_name or name
            if physical_name != contract.return_column:
                continue
            if float(contract.return_scale) != 1.0:
                fetched[name] = normalize_return_values(fetched[name], self.dataset)

    def prefetch_columns(self, names: list[str]) -> None:
        if not names:
            return
        self.refresh_snapshot()
        if self._lazy_scan:
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
                self._normalize_contract_columns(fetched, needed)
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
        self._normalize_contract_columns(fetched, needed)
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

        lf = build_scan_polars_long(
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
        try:
            import polars as pl
            from fields import FIELD_REGISTRY

            expressions = []
            for name in columns:
                spec = FIELD_REGISTRY.get(name, table=self.dataset)
                if spec is None or spec.dataset != self.dataset:
                    continue
                scale = float(spec.scale_to_canonical or 1.0)
                if scale != 1.0:
                    expressions.append((pl.col(name).cast(pl.Float64) * scale).alias(name))
            if expressions:
                lf = lf.with_columns(expressions)
        except ImportError:
            pass
        return lf

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
