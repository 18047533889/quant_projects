# -*- coding: utf-8 -*-
"""R33 §7/§8/§9：BufferRef + SourceWaveExecutor —— task 之间传引用，不传大对象。

- :class:`BufferRef`：统一数据引用（representation / schema / rows / bytes /
  source_snapshot / ordering / refcount）。task 之间传 ref，materialize 由
  consumer 要求；同一 buffer 可被多个 root 消费。
- :class:`SourceWaveExecutor`：R33-P0-016 —— 把 read wave 真正接进 scheduler
  主路径。``execute_wave`` 一次 scan 该 wave 的 union 物理列（经
  ``source.prefetch_columns`` 填入共享 column cache，scan once → consumers
  many），产出 :class:`SourceBufferRef`；记录 runtime event（wave 真实执行，
  不是 explain）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BufferRef:
    """R33 §8：统一数据引用。

    ``representation`` 描述值表示（arrow_batches / polars_lazy / pandas_column /
    numpy_block / spill_file）；``location`` 是取值句柄（cache key / object
    reference）。task 之间只传本 ref。
    """

    representation: str
    schema: tuple[str, ...] = ()
    grain: str = ""            # panel / long / event / scalar
    rows: int = 0
    bytes: int = 0
    source_snapshot: str = ""
    ordering: str = ""         # "instrument,time:asc" | ""
    location: Any = None       # 取值句柄（cache key / duckdb relation）
    ownership: str = "shared"
    refcount: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "schema": list(self.schema),
            "grain": self.grain,
            "rows": self.rows,
            "bytes": self.bytes,
            "source_snapshot": self.source_snapshot,
            "ordering": self.ordering,
            "ownership": self.ownership,
            "refcount": self.refcount,
        }


@dataclass(frozen=True)
class SourceBufferRef(BufferRef):
    """SOURCE_SCAN 产出的 buffer（R33-P0-009 的 BufferRef 形态）。

    ``column_cache_keys`` 指向共享列缓存（DataAccessSource._column_cache）——
    真正物化的数据驻留在那，多个 root 复用同一份。
    """

    column_cache_keys: tuple[str, ...] = ()
    wave_id: int = -1
    scan_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["column_cache_keys"] = list(self.column_cache_keys)
        out["wave_id"] = self.wave_id
        out["scan_bytes"] = self.scan_bytes
        return out


class SourceWaveExecutor:
    """R33-P0-016：把 read wave 接进 scheduler 主路径（scan once → consumers many）。

    ``execute_wave`` 对 wave 的 union 物理列调用 ``source.prefetch_columns``
    （内部 refresh_snapshot + resolve + 一次 scan + 共享 cache），记录 runtime
    event 并返回 :class:`SourceBufferRef`。scheduler 在 wave 覆盖的 SOURCE_SCAN
    task 就绪前执行它，之后所有 root 的 backend.execute 命中共享列缓存。
    """

    def __init__(self, source: Any, ctx: Any | None = None) -> None:
        self._source = source
        self._ctx = ctx
        self._lock = threading.Lock()
        # Serialize check, scan, and publication so concurrent callers cannot
        # both observe a missing wave and perform duplicate reads.
        self._execution_lock = threading.Lock()
        self._executed: dict[int, SourceBufferRef] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def source(self) -> Any:
        return self._source

    def release_wave(self, wave_id: int) -> SourceBufferRef | None:
        """Drop the executor's strong reference after the final consumer."""
        with self._lock:
            ref = self._executed.pop(int(wave_id), None)
        if self._ctx is not None:
            adapter = getattr(self._ctx, "data_source", None)
            release = getattr(adapter, "release_wave", None)
            if callable(release):
                release(int(wave_id))
        return ref

    def execute_wave(
        self,
        wave: Any,
        *,
        consumer_ids: Iterable[str] = (),
    ) -> SourceBufferRef | None:
        """Execute a wave at most once, including under concurrent callers."""
        with self._execution_lock:
            return self._execute_wave(
                wave, consumer_ids=consumer_ids
            )

    def _validate_snapshot_expectations(self) -> None:
        expectations = tuple(
            getattr(self._ctx, "physical_snapshot_expectations", ()) or ()
        )
        if not expectations:
            return
        from data_access import get_store
        from factor_engine.storage.sources.composite_source import SnapshotVerificationError

        store = get_store()
        for expected in expectations:
            prepared = None
            try:
                prepared = store.prepare_read(
                    expected.dataset,
                    columns=tuple(expected.columns),
                    time_range=tuple(expected.time_range),
                    instrument_filter=tuple(expected.instrument_filter),
                    params=dict(expected.params),
                    run_mode="production",
                    snapshot_policy="fail_if_changed",
                )
                actual = str(
                    getattr(prepared.resolved_source_snapshot, "content_digest", "") or ""
                )
                if not actual or actual != expected.content_digest:
                    raise SnapshotVerificationError(
                        f"source snapshot changed for {expected.dataset}: "
                        f"expected={expected.content_digest} actual={actual or '<missing>'}"
                    )
            finally:
                if prepared is not None:
                    store._pipeline.release_reservation(
                        getattr(prepared, "resource_reservation", None)
                    )

    def _execute_wave(
        self,
        wave: Any,
        *,
        consumer_ids: Iterable[str] = (),
    ) -> SourceBufferRef | None:
        """执行一个 read wave：prefetch union 物理列一次，返回 :class:`SourceBufferRef`。

        幂等：同一 wave_id 只执行一次。真实 scan 失败时返回 ``None``（调度器走
        逐 task 兜底读，root 内部仍会按需 load），但 runtime event 记录失败原因
        ——不静默。
        """
        wave_id = int(getattr(wave, "wave_id", -1))
        with self._lock:
            if wave_id in self._executed:
                return self._executed[wave_id]
        columns = sorted(getattr(wave, "columns", frozenset()) or frozenset())
        t0 = time.monotonic()
        event: dict[str, Any] = {
            "wave_id": wave_id,
            "columns": columns,
            "consumer_count": len(set(consumer_ids)),
            "ok": False,
            "reason": "",
            "ms": 0.0,
            "scan_bytes": int(getattr(wave, "estimated_scan_bytes", 0) or 0),
            "memory_bytes": int(getattr(wave, "estimated_memory_bytes", 0) or 0),
        }
        ref: SourceBufferRef | None = None
        try:
            self._validate_snapshot_expectations()
            source = self._source
            prefetch = getattr(source, "prefetch_columns", None)
            load = getattr(source, "load_columns", None)
            # R39-P0-PERF-011: choose the physical path before scanning. Native
            # consumers must not be forced through pandas load_columns.
            representation = self._representation_for_wave(wave)
            pandas_representation = representation in ("pandas_columns", "pandas_column")
            loaded_columns: dict[str, Any] = {}
            native_buffer: Any = None
            native_scan = getattr(source, "scan_polars_long", None)
            if columns and not pandas_representation and callable(native_scan):
                native_buffer = native_scan(list(columns))
                collect = getattr(native_buffer, "collect", None)
                if callable(collect):
                    # A LazyFrame is only a query description. Materialize it
                    # inside the READ_WAVE lease and before the post-scan
                    # snapshot check, otherwise a later collect can read a new
                    # file generation after this wave was certified.
                    native_buffer = collect()
            elif columns and pandas_representation and callable(load):
                loaded = load(list(columns))
                if isinstance(loaded, dict):
                    loaded_columns = loaded
            elif columns and callable(prefetch):
                prefetched = prefetch(list(columns))
                if isinstance(prefetched, dict):
                    loaded_columns = prefetched
            elif columns and callable(load):
                loaded = load(list(columns))
                if isinstance(loaded, dict):
                    loaded_columns = loaded
            # R39-P0-PERF-011：read wave 输出不再固定 pandas-column cache——
            # 按 wave.preferred_representation 直接产出对应 BufferRef。
            location = (
                "source.column_cache"
                if representation in ("pandas_columns", "pandas_column")
                else "source.native"
            )
            self._validate_snapshot_expectations()
            if self._ctx is not None and columns:
                from factor_engine.storage.sources.wave_prefetched_source import (
                    ensure_wave_prefetched_source,
                )
                adapter = ensure_wave_prefetched_source(self._ctx, source)
                if pandas_representation:
                    adapter.publish_columns(wave_id, columns, loaded_columns)
                else:
                    if native_buffer is None:
                        raise RuntimeError("native read wave produced no materialized buffer")
                    adapter.publish_native(wave_id, columns, native_buffer)
            ref = SourceBufferRef(
                representation=representation,
                schema=tuple(columns),
                grain="panel",
                bytes=int(getattr(wave, "estimated_memory_bytes", 0) or 0),
                source_snapshot=str(getattr(source, "snapshot_token", "") or ""),
                ordering="",
                location=location,
                ownership="shared",
                refcount=len(set(consumer_ids)),
                column_cache_keys=tuple(columns),
                wave_id=wave_id,
                scan_bytes=int(getattr(wave, "physical_union_scan_bytes", 0)
                               or getattr(wave, "estimated_scan_bytes", 0) or 0),
                meta={
                    "loaded_columns": loaded_columns,
                    "native_buffer": native_buffer,
                },
            )
            event["ok"] = True
            with self._lock:
                self._executed[wave_id] = ref
        except Exception as exc:  # noqa: BLE001
            event["reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["ms"] = round((time.monotonic() - t0) * 1000, 3)
            with self._lock:
                self._events.append(event)
            self._record_runtime_event(event)
        return ref

    @staticmethod
    def _representation_for_wave(wave: Any) -> str:
        """PERF-011：取 wave.preferred_representation（枚举或字符串），缺省 pandas。"""
        rep = getattr(wave, "preferred_representation", "pandas_columns") or "pandas_columns"
        if hasattr(rep, "value"):  # SourceRepresentation enum
            rep = rep.value
        return str(rep)

    def _record_runtime_event(self, event: dict[str, Any]) -> None:
        """写进 ctx.runtime_stats（read wave 真实执行 runtime evidence）。"""
        if self._ctx is None:
            return
        try:
            runtime = dict(getattr(self._ctx, "runtime_stats", None) or {})
            waves = runtime.setdefault("read_waves", [])
            if isinstance(waves, list):
                waves.append(event)
                runtime["read_wave_count"] = len(waves)
                runtime["read_wave_executed"] = sum(
                    1 for w in waves if w.get("ok")
                )
            self._ctx.runtime_stats = runtime  # type: ignore[attr-defined]
        except Exception:
            pass

    def summary(self) -> dict[str, Any]:
        with self._lock:
            ok = sum(1 for e in self._events if e.get("ok"))
            return {
                "waves_planned": len(self._events),
                "waves_executed": ok,
                "waves_failed": len(self._events) - ok,
                "events": list(self._events),
            }
