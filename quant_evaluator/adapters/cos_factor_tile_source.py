"""Bounded ordered COS FactorTileSource backed by DataAccess bound reads.

``from_data_access`` binds a manifest through explicitly injected helpers and
uses DataAccess context factories for registered dataset descriptors. QE has
no runtime dependency on the library providing those helpers. At most two
independent worker contexts read ahead; tile assembly stays ordered and caller
supplied.
This module does not import benchmark scripts.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from threading import Lock
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from quant_evaluator.runtime.source_memory_budget import estimate_source_memory

from quant_evaluator.contracts.factor_tile_source import FactorTile


@dataclass(frozen=True)
class BoundCosFactor:
    """One manifest-selected factor and its expected remote identity."""

    factor_id: str
    uri: str
    sha256: str
    size_bytes: int
    params: Mapping[str, object] | None = None


@dataclass(frozen=True)
class BoundManifestHelpers:
    """Existing bound-manifest operations supplied by an integration layer.

    QE defines this small callable contract so its COS adapter stays usable
    without depending on a particular manifest-owning library. The injected
    functions remain responsible for all binding and identity validation.
    """

    read_bound_manifest: Callable[..., object]
    read_bound_factor: Callable[..., object]
    verify_bound_manifest_unchanged: Callable[..., object]

    def __post_init__(self):
        if not all(callable(helper) for helper in (
                self.read_bound_manifest, self.read_bound_factor,
                self.verify_bound_manifest_unchanged)):
            raise TypeError("bound manifest helpers must be callable")


@dataclass
class DataAccessReadContext:
    """Explicit registered DataAccess target plus its owned resource cleanup.

    Factories must create a fresh context per worker read. The manifest factory
    is called once on the controller thread and retained through final verify.
    """

    store: object
    manifest_dataset: str
    factor_dataset: str | None = None
    manifest_params: Mapping[str, object] | None = None
    factor_params: Mapping[str, object] | None = None
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class VerifiedFactorPayload:
    """One decoded factor panel paired with its bound source identity."""

    payload: object
    identity: Mapping[str, object]


class CosFactorTileSource:
    """FactorTileSource backed by verified COS reads and bounded prefetch.

    ``read_factor(record, manifest_snapshot)`` must return ``(payload, identity)``;
    identity must contain ``uri``, ``sha256``, ``size_bytes`` and
    ``manifest_sha256`` matching the selected record/snapshot. ``make_tile``
    assembles the ordered payloads into a FactorBatch.

    Prefer ``from_data_access`` for the production path. It binds and verifies
    the manifest through explicitly injected bound-manifest helpers and creates
    one DataAccess store/engine context per read. QE does not depend on the
    library that provides those helpers. Memory admission is a conservative
    logical-payload estimate for source frames, mutable dense assembly,
    immutable FactorBatch ownership, validity copies, bounded prefetch, and
    declared callback scratch; it is not a hard RSS guarantee because Python
    and pandas allocator/object overhead is outside the model.
    """

    def __init__(self, *, records: Sequence[BoundCosFactor], time_axis,
                 asset_axis, dtype: str, manifest_snapshot,
                 manifest_sha256: str, max_tile_size: int,
                 read_factor: Callable, verify_manifest: Callable,
                 make_tile: Callable, prefetch: str = "auto",
                 prefetch_workers: int = 2,
                 max_source_memory_bytes: int = 4 * 1024**3,
                 max_prefetch_memory_bytes: int = 512 * 1024**2,
                 extra_assembly_bytes_per_cell: int = 0,
                 _close_controller: Callable[[], None] | None = None):
        if prefetch not in {"off", "on", "auto"}:
            raise ValueError("prefetch must be 'off', 'on' or 'auto'")
        if type(prefetch_workers) is not int or prefetch_workers not in {1, 2, 4}:
            raise ValueError("prefetch_workers must be one of 1, 2 or 4")
        if not records or len({r.factor_id for r in records}) != len(records):
            raise ValueError("records must have unique factor ids")
        if type(max_tile_size) is not int or not 1 <= max_tile_size <= 32:
            raise ValueError("max_tile_size must be in 1..32")
        if type(max_source_memory_bytes) is not int or max_source_memory_bytes <= 0:
            raise ValueError("max_source_memory_bytes must be positive")
        if type(max_prefetch_memory_bytes) is not int or max_prefetch_memory_bytes < 0:
            raise ValueError("max_prefetch_memory_bytes must be nonnegative")
        if type(extra_assembly_bytes_per_cell) is not int or extra_assembly_bytes_per_cell < 0:
            raise ValueError("extra_assembly_bytes_per_cell must be a nonnegative integer")
        if not callable(read_factor) or not callable(verify_manifest) or not callable(make_tile):
            raise TypeError("read_factor, verify_manifest and make_tile must be callable")
        normalized_dtype = np.dtype(dtype)
        if (not np.issubdtype(normalized_dtype, np.number)
                or np.issubdtype(normalized_dtype, np.complexfloating)):
            raise ValueError("factor dtype must be real numeric")
        if (time_axis.values is None or asset_axis.values is None
                or time_axis.size <= 0 or asset_axis.size <= 0):
            raise ValueError("complete nonempty source axes are required")
        self.records = tuple(records)
        self.factor_ids = tuple(r.factor_id for r in records)
        self.time_axis, self.asset_axis = time_axis, asset_axis
        self.dtype = dtype
        self.snapshot_id = self._request_snapshot_id(
            manifest_sha256, self.factor_ids, time_axis, asset_axis, dtype, self.records)
        self.max_tile_size = max_tile_size
        self.manifest_snapshot, self.manifest_sha256 = manifest_snapshot, manifest_sha256
        self.read_factor, self.verify_manifest, self.make_tile = read_factor, verify_manifest, make_tile
        # auto is deliberately enabled for this COS-only adapter.
        self.prefetch_mode = prefetch
        self.prefetch_enabled = prefetch != "off"
        # Keep queued payloads bounded to active workers: a larger queue could
        # retain completed payloads beyond the worker-based memory estimate.
        self.prefetch_workers = prefetch_workers if self.prefetch_enabled else 0
        self.prefetch_window = prefetch_workers if self.prefetch_enabled else 1
        self.next_start = 0
        self._closed = False
        self._lock = Lock()
        self._executor = None
        self._pending: deque[tuple[int, Future]] = deque()
        self._submitted = 0
        self._verified_manifest = False
        self._failed = False
        self._final_verified = False
        self.reads: list[tuple[int, int]] = []
        self.tile_read_timings: list[dict[str, object]] = []
        self.max_source_memory_bytes = max_source_memory_bytes
        self.max_prefetch_memory_bytes = max_prefetch_memory_bytes
        self.extra_assembly_bytes_per_cell = extra_assembly_bytes_per_cell
        self._controller_close = _close_controller
        # During the standard dense materializer, source frames coexist with a
        # mutable assembled array and FactorBatch's immutable value owner (3
        # panel copies). The validity mask and its immutable owner add 2 bytes
        # per cell. Custom make_tile callbacks must declare any additional
        # simultaneously-live scratch via extra_assembly_bytes_per_cell. This
        # is a logical payload estimate, not an RSS guarantee: allocator and
        # pandas object/index overhead are outside the bound.
        estimate = estimate_source_memory(
            time_size=time_axis.size, asset_size=asset_axis.size,
            factor_count=len(records), max_tile_size=max_tile_size,
            dtype_itemsize=normalized_dtype.itemsize,
            prefetch_workers=self.prefetch_workers,
            prefetch_enabled=self.prefetch_enabled,
            extra_assembly_bytes_per_cell=extra_assembly_bytes_per_cell)
        # DataAccess research reads cap Arrow results at 128 MiB. Allow an
        # additional equal-sized pandas conversion buffer per active worker.
        estimated_prefetch = estimate.prefetch_bytes
        estimated_bytes = estimate.total_bytes
        if estimated_prefetch > max_prefetch_memory_bytes:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            raise MemoryError("bounded prefetch exceeds max_prefetch_memory_bytes")
        if estimated_bytes > max_source_memory_bytes:
            raise MemoryError("tile assembly plus bounded prefetch exceeds max_source_memory_bytes")
        self._executor = (ThreadPoolExecutor(max_workers=self.prefetch_workers,
                                             thread_name_prefix="qe-cos-prefetch")
                          if self.prefetch_enabled else None)
        self.estimated_assembly_bytes = estimate.assembly_bytes
        self.estimated_peak_source_bytes = estimated_bytes
        self.estimated_prefetch_bytes = estimated_prefetch

    @staticmethod
    def _request_snapshot_id(manifest_sha256, factor_ids, time_axis, asset_axis, dtype,
                             records=()):
        digest = hashlib.sha256()
        fields = {"manifest_sha256": manifest_sha256, "factor_ids": factor_ids,
                  "time": [time_axis.name, time_axis.dtype, time_axis.size],
                  "asset": [asset_axis.name, asset_axis.dtype, asset_axis.size],
                  "dtype": str(dtype),
                  "sources": [(r.factor_id, r.uri, r.sha256, r.size_bytes) for r in records]}
        digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":"),
                                 default=str).encode())
        for axis in (time_axis, asset_axis):
            values = axis.values
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes() if not values.dtype.hasobject else
                          json.dumps(values.tolist(), separators=(",", ":"),
                                     default=str).encode())
        return digest.hexdigest()

    @classmethod
    def from_data_access(cls, *, factor_ids: Sequence[str], time_axis, asset_axis,
                         dtype: str, make_tile: Callable,
                         bound_manifest_helpers: BoundManifestHelpers,
                         manifest_context_factory: Callable[[], DataAccessReadContext],
                         factor_context_factory: Callable[[BoundCosFactor], DataAccessReadContext],
                         max_object_mib: int = 128, max_tile_size: int = 16,
                         prefetch_workers: int = 2,
                         max_source_memory_bytes: int = 4 * 1024**3,
                         max_prefetch_memory_bytes: int = 512 * 1024**2,
                         extra_assembly_bytes_per_cell: int = 0,
                         prefetch: str = "auto", expected_manifest_sha256: str | None = None):
        """Build a COS source using caller-injected bound-manifest helpers.

        The context factories provide explicit registered dataset descriptors;
        each factor context owns a thread-local store/engine and cleanup method.
        The selected factor URI, SHA and byte count are taken from the captured
        manifest, never accepted as caller-authored identity claims. The source
        budget accounts for source frames, dense assembly, immutable FactorBatch
        copies, validity mask copies, and bounded prefetch. Custom tile builders
        with extra simultaneous scratch must declare it per T*N*factor cell.
        """
        if not isinstance(bound_manifest_helpers, BoundManifestHelpers):
            raise TypeError("bound_manifest_helpers must be BoundManifestHelpers")
        read_bound_factor = bound_manifest_helpers.read_bound_factor
        read_bound_manifest = bound_manifest_helpers.read_bound_manifest
        verify_bound_manifest_unchanged = (
            bound_manifest_helpers.verify_bound_manifest_unchanged)

        if type(max_object_mib) is not int or not 1 <= max_object_mib <= 128:
            raise ValueError("max_object_mib must be in 1..128")
        if (not factor_ids or len(set(factor_ids)) != len(factor_ids)
                or any(not isinstance(fid, str) or not fid for fid in factor_ids)):
            raise ValueError("factor_ids must be ordered, nonempty and unique")
        controller = manifest_context_factory()
        controller_closed = False

        def close_controller():
            nonlocal controller_closed
            if not controller_closed:
                controller_closed = True
                controller.close()

        try:
            snapshot = read_bound_manifest(
                controller.store, controller.manifest_dataset,
                manifest_params=controller.manifest_params, allow_research=True)
            manifest_sha = snapshot.manifest_sha256
            if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
                raise ValueError("bound manifest digest differs from expected digest")
            records = []
            for factor_id in factor_ids:
                row = snapshot.factors.get(factor_id)
                if not isinstance(row, dict):
                    raise ValueError("selected factor is absent from bound manifest")
                uri, sha, size = row.get("uri"), row.get("sha256"), row.get("bytes")
                if (not isinstance(uri, str) or not isinstance(sha, str)
                        or type(size) is not int or size <= 0
                        or size > max_object_mib * 1024**2):
                    raise ValueError("selected factor exceeds object bound or lacks identity")
                records.append(BoundCosFactor(factor_id, uri, sha, size))

            def read_factor(record, manifest_snapshot):
                context = factor_context_factory(record)
                if (context.manifest_dataset != controller.manifest_dataset
                        or context.factor_dataset is None):
                    context.close()
                    raise ValueError("factor context must target the bound manifest and a factor dataset")
                try:
                    bound = read_bound_factor(
                        context.store, context.manifest_dataset, context.factor_dataset,
                        record.factor_id, manifest_params=context.manifest_params,
                        factor_params=context.factor_params, allow_research=True,
                        max_object_mib=max_object_mib, manifest_snapshot=manifest_snapshot)
                    factor = bound.factor
                    identity = {"uri": factor.source_uri, "sha256": factor.content_sha256,
                                "size_bytes": factor.downloaded_bytes,
                                "manifest_sha256": bound.manifest_sha256,
                                "source_etag": factor.source_etag}
                    return factor.table.to_pandas(), identity
                finally:
                    context.close()

            def verify_manifest(manifest_snapshot):
                verify_bound_manifest_unchanged(
                    controller.store, controller.manifest_dataset, manifest_snapshot,
                    manifest_params=controller.manifest_params)

            return cls(
                records=records, time_axis=time_axis, asset_axis=asset_axis,
                dtype=dtype, manifest_snapshot=snapshot,
                manifest_sha256=manifest_sha, max_tile_size=max_tile_size,
                read_factor=read_factor, verify_manifest=verify_manifest,
                make_tile=make_tile, prefetch=prefetch,
                prefetch_workers=prefetch_workers,
                max_source_memory_bytes=max_source_memory_bytes,
                max_prefetch_memory_bytes=max_prefetch_memory_bytes,
                extra_assembly_bytes_per_cell=extra_assembly_bytes_per_cell,
                _close_controller=close_controller)
        except BaseException:
            close_controller()
            raise

    def _read_one(self, index: int):
        record = self.records[index]
        payload, identity = self.read_factor(record, self.manifest_snapshot)
        expected = (record.uri, record.sha256, record.size_bytes, self.manifest_sha256)
        actual = (identity.get("uri"), identity.get("sha256"),
                  identity.get("size_bytes"), identity.get("manifest_sha256"))
        if actual != expected:
            raise ValueError("COS factor identity differs from selected manifest")
        return VerifiedFactorPayload(payload, identity)

    def _fill(self, stop: int) -> None:
        while self._submitted < stop and len(self._pending) < self.prefetch_window:
            index = self._submitted
            self._pending.append((index, self._executor.submit(self._read_one, index)))
            self._submitted += 1

    def read_tile(self, start: int, end: int) -> FactorTile:
        executor = None
        started = time.perf_counter()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("COS factor tile source is closed")
                if (type(start) is not int or type(end) is not int or start != self.next_start
                        or end <= start or end > len(self.records)
                        or end - start > self.max_tile_size):
                    raise ValueError("tile range must be the next bounded source range")
                if not self._verified_manifest:
                    self.verify_manifest(self.manifest_snapshot)
                    self._verified_manifest = True
                payloads = []
                if self.prefetch_enabled:
                    # Keep one full worker window beyond this tile so these
                    # bounded reads can finish while the consumer evaluates it.
                    self._fill(min(len(self.records), end + self.prefetch_window))
                    for expected_index in range(start, end):
                        index, future = self._pending.popleft()
                        if index != expected_index:
                            raise RuntimeError("prefetch queue order was corrupted")
                        payloads.append(future.result())
                        self._fill(min(len(self.records), end + self.prefetch_window))
                else:
                    for index in range(start, end):
                        payloads.append(self._read_one(index))
                batch = self.make_tile(start, end, self.records[start:end], payloads)
                self.next_start = end
                if end == len(self.records):
                    self.verify_manifest(self.manifest_snapshot)
                    self._final_verified = True
                self.reads.append((start, end))
                self.tile_read_timings.append({
                    "tile_range": [start, end],
                    "wall_seconds": time.perf_counter() - started,
                    "prefetch_mode": self.prefetch_mode,
                    "prefetch_window": self.prefetch_window,
                })
                return FactorTile(start, end, batch, self.snapshot_id)
        except BaseException:
            with self._lock:
                self._failed = True
                self._closed = True
                for _, future in self._pending:
                    future.cancel()
                self._pending.clear()
                executor, self._executor = self._executor, None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self._close_controller()
            raise

    def _close_controller(self):
        callback = getattr(self, "_controller_close", None)
        if callback is not None:
            self._controller_close = None
            callback()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            should_verify = not self._failed and not self._final_verified
            for _, future in self._pending:
                future.cancel()
            self._pending.clear()
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        verify_error = None
        if should_verify:
            try:
                self.verify_manifest(self.manifest_snapshot)
                self._final_verified = True
            except BaseException as exc:
                verify_error = exc
        try:
            self._close_controller()
        finally:
            if verify_error is not None:
                raise verify_error

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
