"""Execution-context source view backed by snapshot-verified read waves."""
from __future__ import annotations

from threading import RLock
from typing import Any, Iterable


class WavePrefetchedSourceAdapter:
    """Serve wave-bound columns without reopening the physical source."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._columns: dict[str, dict[int, Any]] = {}
        self._native_frames: dict[int, tuple[frozenset[str], Any]] = {}
        self._wave_columns: dict[int, frozenset[str]] = {}
        self._lock = RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def publish_columns(self, wave_id: int, columns: Iterable[str], loaded: dict[str, Any]) -> None:
        names = set(columns)
        if not names.issubset(loaded):
            missing = sorted(names.difference(loaded))
            raise RuntimeError(f"wave materialization omitted bound columns: {missing}")
        with self._lock:
            self._wave_columns[int(wave_id)] = frozenset(names)
            for name in names:
                self._columns.setdefault(name, {})[int(wave_id)] = loaded[name]

    def publish_native(self, wave_id: int, columns: Iterable[str], frame: Any) -> None:
        names = frozenset(columns)
        with self._lock:
            self._wave_columns[int(wave_id)] = names
            self._native_frames[int(wave_id)] = (names, frame)

    def release_wave(self, wave_id: int) -> None:
        wid = int(wave_id)
        with self._lock:
            names = self._wave_columns.pop(wid, frozenset())
            self._native_frames.pop(wid, None)
            for name in names:
                owners = self._columns.get(name)
                if owners is None:
                    continue
                owners.pop(wid, None)
                if not owners:
                    self._columns.pop(name, None)

    def _is_bound(self, name: str) -> bool:
        return any(name in names for names in self._wave_columns.values())

    def load_column(self, name: str) -> Any:
        with self._lock:
            if name in self._columns:
                return next(reversed(self._columns[name].values()))
            if self._is_bound(name):
                raise RuntimeError(f"bound wave column {name!r} has no materialized pandas value")
        return self.inner.load_column(name)

    def load_columns(self, names: Iterable[str]) -> dict[str, Any]:
        requested = list(names)
        with self._lock:
            missing_bound = [name for name in requested if self._is_bound(name) and name not in self._columns]
            if missing_bound:
                raise RuntimeError(
                    f"bound wave columns have no materialized pandas values: {sorted(missing_bound)}"
                )
            cached = {name: next(reversed(self._columns[name].values()))
                      for name in requested if name in self._columns}
        missing = [name for name in requested if name not in cached]
        if missing:
            cached.update(self.inner.load_columns(missing))
        return cached

    def scan_polars_long(self, columns: list[str]):
        requested = frozenset(columns)
        with self._lock:
            for _wave_id, (available, frame) in reversed(self._native_frames.items()):
                if requested.issubset(available):
                    lazy = frame.lazy() if hasattr(frame, "lazy") else frame
                    schema = set(getattr(frame, "columns", ()) or ())
                    keys = [name for name in ("ts", "inst", "session") if name in schema]
                    projection = [*keys, *(name for name in columns if name not in keys)]
                    return lazy.select(projection)
            bound = set().union(*self._wave_columns.values()) if self._wave_columns else set()
            if requested.intersection(bound):
                raise RuntimeError(
                    f"bound native wave columns are unavailable: {sorted(requested)}"
                )
        return self.inner.scan_polars_long(columns)


def ensure_wave_prefetched_source(ctx: Any, source: Any) -> WavePrefetchedSourceAdapter:
    current = getattr(ctx, "data_source", None)
    if isinstance(current, WavePrefetchedSourceAdapter):
        return current
    adapter = WavePrefetchedSourceAdapter(source)
    ctx.data_source = adapter
    return adapter
