# -*- coding: utf-8 -*-
"""P0-10..12: RemoteFactorBlockWriter —— FactorEngine 因子结果直写 COS（bounded-memory）。

生产路径 = factor → bounded RAM → serialize → COS upload → release RAM。**没有**
本地 parquet → SSD → upload → delete 的中间落盘。

设计
    - 每个 factor 的 1D float32 值先落在内存缓冲（per-partition）。
    - 攒满 ``columns_per_block``（复用 :func:`feature_block.choose_columns_per_block`
      的块大小逻辑）即 flush 成一个 ``(rows, ncols)`` float32 block，序列化为字节，
      经 ``ObjectStore`` 协议上传到 COS，上传完成后**立即释放**该 block 的 RAM。
    - 发布走不可变代次（immutable generation + CURRENT 指针最后翻转）——
      优先复用 ``data_access.write.object_store_generation_publisher.
      ObjectStoreGenerationPublisher``（对象上传后 head/verify size+sha256，manifest
      不可变，CURRENT 最后翻转，读者永远看不到部分代次）。
    - bounded-memory：``buffered_cols × rows × 4 + 当前 block 序列化字节`` 受
      ``ResourceBroker`` 租约约束（FEATURE_BLOCK_ASSEMBLY 组装 + COS_UPLOAD_INFLIGHT
      上传中）。租约不足时 writer 阻塞等待（绝不把整个 10GB artifact 一次放进 RAM）。

公开 API（供 planner / sink 消费，与 FeatureBlockWriter 形状对齐）::

    submit(factor_id, values, partition=None)
    flush()
    finish()          -> 发布代次 + 翻转 CURRENT
    abort()           -> 删除未晋升代次
    resolve_current() -> 当前 active generation_id | None
    manifest()        -> 本 writer 的 BlockManifest（或 None）
    summary()         -> 遥测

dtype 契约：因子值一律 float32（``FACTOR_VALUE_DTYPE``），与 feature_block 一致。
"""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.feature_block import (
    FACTOR_VALUE_DTYPE,
    BlockManifest,
    BlockSpec,
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    choose_columns_per_block,
    _partition_token,
)
from factor_engine.runtime.resource_broker import ResourceBroker

__all__ = [
    "RemoteFactorBlockWriter",
    "serialize_block",
    "deserialize_block",
    "make_remote_sink_writer",
]

_BYTES_PER_CELL = np.dtype(FACTOR_VALUE_DTYPE).itemsize  # 4
_BLOCK_SUFFIX = ".fb"  # 远程 block 对象后缀（feature-block 值）
_REMOTE_MANIFEST_KEY = "manifest.json"
#: 租约不足时的等待步长（毫秒）与超时。
_ACQUIRE_POLL_S = 0.01
_ACQUIRE_TIMEOUT_S = 300.0


def serialize_block(block: np.ndarray) -> bytes:
    """把一个 (rows, ncols) float32 block 序列化为字节载荷。

    返回的 bytes 是自描述载荷：头部 magic + shape，便于 reader 无 manifest 也能
    解析（测试回读用）。值域保证 float32。
    """
    arr = np.ascontiguousarray(block, dtype=np.dtype(FACTOR_VALUE_DTYPE))
    rows, cols = arr.shape
    header = json.dumps(
        {"magic": "FB1", "rows": int(rows), "cols": int(cols), "dtype": FACTOR_VALUE_DTYPE},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    head = len(header).to_bytes(4, "big")
    return head + header + arr.tobytes()


def deserialize_block(payload: bytes) -> np.ndarray:
    """解析 :func:`serialize_block` 的载荷，返回 float32 (rows, cols) ndarray。"""
    head = int.from_bytes(payload[:4], "big")
    header = json.loads(payload[4 : 4 + head].decode("utf-8"))
    rows, cols = int(header["rows"]), int(header["cols"])
    body = np.frombuffer(payload[4 + head : 4 + head + rows * cols * _BYTES_PER_CELL], dtype=np.dtype(FACTOR_VALUE_DTYPE))
    return np.ascontiguousarray(body.reshape(rows, cols))


class RemoteFactorBlockWriter:
    """COS 直写 factor-block writer（bounded-memory，P0-10..12）。

    ``store`` 是 ``data_access.read.object_store.ObjectStore`` 实例（本地测试用
    ``LocalObjectStore``，生产用 ``COSObjectStore``）。``broker`` 是
    ``ResourceBroker``，两者都可省略（省略时用内置 no-op broker / 内存 store，
    便于最小化测试）。
    """

    def __init__(
        self,
        store: Any,
        prefix: str,
        *,
        broker: ResourceBroker | None = None,
        columns_per_block: int | None = None,
        writer_memory_bytes: int | None = None,
        default_partition: str = "default",
        publisher: Any | None = None,
        on_lease_none: Callable[[MemoryLeaseKind, int], None] | None = None,
    ) -> None:
        self.store = store
        self.prefix = str(prefix).strip("/") or "factors"
        self.broker = broker
        self.columns_per_block = columns_per_block
        self.writer_memory_bytes = writer_memory_bytes
        self.default_partition = str(default_partition)
        self.on_lease_none = on_lease_none

        if publisher is not None:
            self._publisher = publisher
        else:
            from data_access.write.object_store_generation_publisher import (
                ObjectStoreGenerationPublisher,
            )

            self._publisher = ObjectStoreGenerationPublisher(store)

        # per-partition 缓冲：{partition: {factor_id: 1D float32 ndarray}}
        self._buffer: dict[str, dict[str, np.ndarray]] = {}
        self._order: dict[str, list[str]] = {}
        self._partition_rows: dict[str, int] = {}
        self._blocks: dict[str, BlockSpec] = {}
        self._factor_to_block: dict[str, dict[str, str]] = {}
        self._block_seq = 0
        self._finished = False
        self._aborted = False
        self._generation_id: str | None = None
        self._manifest: BlockManifest | None = None

        # telemetry
        self._serializer_peak_bytes = 0
        self._upload_buffer_bytes = 0
        self._inflight_parts = 0
        self._compressed_bytes = 0
        self._factor_blocks_written = 0
        self._total_factors_written = 0
        self._leases: list[Any] = []  # 当前持有的内存租约
        self._lock = threading.RLock()

    # -- 状态 ---------------------------------------------------------------

    def buffered_factors(self) -> int:
        return sum(len(v) for v in self._buffer.values())

    def block_count(self) -> int:
        return len(self._blocks)

    def generation_id(self) -> str | None:
        return self._generation_id

    # -- 主写入 -------------------------------------------------------------

    def submit(self, factor_id: str, values: Any, partition: str | None = None) -> None:
        """把一个因子的一段值（1D float32）加入缓冲，满了即 flush 成 block 上传。"""
        if self._finished:
            raise RuntimeError("RemoteFactorBlockWriter.finish() 已调用，不能再 submit")
        if self._aborted:
            raise RuntimeError("RemoteFactorBlockWriter.abort() 已调用，不能再 submit")
        partition = str(partition) if partition is not None else self.default_partition
        arr = self._coerce_values(values, factor_id)
        # bounded-memory：先为这列的 RAM 拿 FEATURE_BLOCK_ASSEMBLY 租约（阻塞等待）。
        self._acquire_or_wait(MemoryLeaseKind.FEATURE_BLOCK_ASSEMBLY, arr.nbytes)

        part_buf = self._buffer.setdefault(partition, {})
        part_order = self._order.setdefault(partition, [])
        if factor_id not in part_buf:
            part_order.append(factor_id)
        part_buf[factor_id] = arr
        self._partition_rows[partition] = int(arr.shape[0])
        target = self._resolve_columns_per_block(
            row_count=int(arr.shape[0]), factor_total=len(part_buf)
        )
        if len(part_buf) >= target:
            self.flush_partition(partition)

    def write_results(self, items: Iterable[Any]) -> None:
        """StreamingResultSink 接线：item.name=factor_id, item.value=1D 值,
        item.meta["partition"] 可选覆盖（与 FeatureBlockWriter.write_results 对齐）。"""
        for item in items:
            name = getattr(item, "name", None)
            value = getattr(item, "value", None)
            meta = getattr(item, "meta", None) or {}
            if name is None or value is None:
                continue
            self.submit(str(name), value, partition=meta.get("partition"))

    def flush(self) -> None:
        """flush 全部 partition 的缓冲（落成 block 并上传，释放 RAM）。"""
        for partition in list(self._buffer.keys()):
            self.flush_partition(partition)

    def flush_partition(self, partition: str) -> None:
        """把某个 partition 的缓冲落成（多个）block 并上传。"""
        part_buf = self._buffer.get(partition)
        if not part_buf:
            return
        part_order = self._order.get(partition, list(part_buf.keys()))
        fids = [f for f in part_order if f in part_buf]
        if not fids:
            return
        rows = int(part_buf[fids[0]].shape[0])
        self._partition_rows[partition] = rows
        cols = self._resolve_columns_per_block(row_count=rows, factor_total=len(fids))
        for chunk_start in range(0, len(fids), cols):
            chunk = fids[chunk_start : chunk_start + cols]
            self._upload_one_block(partition, rows, chunk, part_buf)
        # 上传完成后清空该 partition 缓冲并归还 RAM 租约。
        for fid in fids:
            self._release_lease_for(part_buf[fid])
        self._buffer[partition] = {}
        self._order[partition] = []

    def _upload_one_block(
        self,
        partition: str,
        rows: int,
        fids: Sequence[str],
        part_buf: Mapping[str, np.ndarray],
    ) -> None:
        block_id = f"b{_partition_token(partition)}_{self._block_seq:04d}"
        seq = self._block_seq
        self._block_seq += 1
        rel = f"{partition}/{seq:04d}{_BLOCK_SUFFIX}"

        # 组装 block 值（rows × ncols float32）。组装工作区本身也受租约约束，
        # 与 buffered 列 RAM 一起计入 bounded-memory 预算。
        block = np.empty((rows, len(fids)), dtype=self.dtype)
        column_map: dict[str, int] = {}
        for col, fid in enumerate(fids):
            col_arr = part_buf[fid]
            block[:, col] = col_arr if col_arr.shape[0] == rows else col_arr[:rows]
            column_map[fid] = col
        serialized = serialize_block(block)
        with self._lock:
            self._serializer_peak_bytes = max(
                self._serializer_peak_bytes, len(serialized)
            )
            self._upload_buffer_bytes = len(serialized)
        # 序列化字节也占 RAM：为上传 in-flight 拿租约（阻塞等待）。
        self._acquire_or_wait(MemoryLeaseKind.COS_UPLOAD_INFLIGHT, len(serialized))
        with self._lock:
            self._inflight_parts += 1
        try:
            self._publisher.add_object(
                self._require_generation(), rel, serialized,
                metadata={
                    "block_id": block_id,
                    "partition": partition,
                    "row_count": rows,
                    "factor_ids": list(fids),
                    "column_map": column_map,
                },
            )
        finally:
            self._release_lease_for(serialized)
            with self._lock:
                self._inflight_parts = max(0, self._inflight_parts - 1)
                self._upload_buffer_bytes = 0
                self._compressed_bytes += len(serialized)
                self._factor_blocks_written += 1
                self._total_factors_written += len(fids)
        spec = BlockSpec(
            block_id=block_id,
            partition=partition,
            file=rel,
            row_count=rows,
            factor_ids=list(fids),
            column_map=column_map,
        )
        self._blocks[block_id] = spec
        for fid in fids:
            self._factor_to_block.setdefault(fid, {})[partition] = block_id
        self._partition_rows[partition] = rows

    def finish(self) -> str:
        """flush 残余缓冲 + 写本 writer manifest + 发布代次（翻转 CURRENT）。

        返回 current generation_id（CURRENT 已指向它）。部分代次在 CURRENT 翻转前
        永远不可见。
        """
        if self._finished:
            return self._generation_id or ""
        self.flush()
        self._manifest = BlockManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            dtype=self.dtype,
            blocks=dict(self._blocks),
            factor_to_block={k: dict(v) for k, v in self._factor_to_block.items()},
            partition_rows=dict(self._partition_rows),
            columns_per_block=self.columns_per_block,
        )
        # 把本 writer 的 feature-block manifest 作为代次内对象发布（元数据与值分离）。
        manifest_bytes = json.dumps(
            self._manifest.to_dict(), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        self._acquire_or_wait(MemoryLeaseKind.MANIFEST_BUFFER, len(manifest_bytes))
        try:
            self._publisher.add_object(
                self._require_generation(), _REMOTE_MANIFEST_KEY, manifest_bytes,
                metadata={"role": "feature_block_manifest"},
            )
        finally:
            self._release_lease_for(manifest_bytes)
        # 最后翻转 CURRENT —— 部分代次绝不可见。
        self._publisher.finish_generation(self._require_generation())
        self._finished = True
        return self._generation_id or ""

    def abort(self) -> None:
        """删除未晋升代次（CURRENT 不受影响）。"""
        if self._finished:
            return
        if self._generation_id is not None:
            self._publisher.abort_generation(self._generation_id)
        self._aborted = True
        self._buffer = {}
        self._order = {}
        self._blocks = {}
        self._factor_to_block = {}

    def resolve_current(self) -> str | None:
        """当前 active generation_id；无 CURRENT 指针返回 None。"""
        return self._publisher.resolve_current(self.prefix)

    def manifest(self) -> BlockManifest | None:
        return self._manifest

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "prefix": self.prefix,
                "generation_id": self._generation_id,
                "finished": self._finished,
                "aborted": self._aborted,
                "block_count": len(self._blocks),
                "factor_blocks_written": self._factor_blocks_written,
                "total_factors_written": self._total_factors_written,
                "serializer_peak_bytes": self._serializer_peak_bytes,
                "upload_buffer_bytes": self._upload_buffer_bytes,
                "inflight_parts": self._inflight_parts,
                "compressed_bytes": self._compressed_bytes,
                "buffered_factors": self.buffered_factors(),
                "current_generation": self.resolve_current(),
            }

    # -- 内部工具 -----------------------------------------------------------

    @property
    def dtype(self) -> str:
        return FACTOR_VALUE_DTYPE

    def _resolve_columns_per_block(self, *, row_count: int, factor_total: int) -> int:
        if self.columns_per_block is not None:
            return max(1, int(self.columns_per_block))
        return choose_columns_per_block(
            total_factors=factor_total,
            row_count=row_count,
            writer_memory_bytes=self.writer_memory_bytes,
        )

    def _require_generation(self) -> str:
        """惰性 begin_generation（首个 block/对象写入前）。"""
        if self._generation_id is None:
            self._generation_id = self._publisher.begin_generation(
                self.prefix,
                metadata={"layout": "factor_block", "dtype": FACTOR_VALUE_DTYPE},
            )
        return self._generation_id

    def _coerce_values(self, values: Any, factor_id: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError(
                    f"RemoteFactorBlockWriter: 因子 {factor_id} 值必须为 1D，got ndim={arr.ndim}"
                )
        return np.ascontiguousarray(arr)

    def _acquire_or_wait(self, kind: MemoryLeaseKind, nbytes: int) -> None:
        """按 kind 拿内存租约；无 broker 直接通过。租约不足时阻塞等待，直到拿到
        或超时（fail-closed）。bounded-memory 的硬保证：buffered + in-flight 总和
        受 broker ExecutionBudget 约束，绝不把整块 artifact 一次性放进 RAM。"""
        if nbytes <= 0:
            return
        broker = self.broker
        if broker is None:
            return
        deadline = time.monotonic() + _ACQUIRE_TIMEOUT_S
        while True:
            lease = broker.acquire_memory(kind, nbytes)
            if lease is not None:
                with self._lock:
                    self._leases.append(lease)
                return
            if self.on_lease_none is not None:
                try:
                    self.on_lease_none(kind, nbytes)
                except Exception:
                    pass
            if time.monotonic() >= deadline:
                raise MemoryError(
                    f"RemoteFactorBlockWriter: 无法获得 {kind} 租约 {nbytes}B "
                    f"（{_ACQUIRE_TIMEOUT_S}s 超时），bounded-memory 拒绝继续"
                )
            time.sleep(_ACQUIRE_POLL_S)

    def _release_lease_for(self, obj: Any) -> None:
        """归还与某对象大小匹配的内存租约（best-effort）。"""
        broker = self.broker
        if broker is None:
            return
        try:
            nbytes = int(getattr(obj, "nbytes", None)) or len(obj)
        except Exception:
            return
        with self._lock:
            idx = next(
                (i for i, l in enumerate(self._leases) if l.nbytes == nbytes),
                None,
            )
            if idx is not None:
                lease = self._leases.pop(idx)
                try:
                    lease.release()
                except Exception:
                    pass


def make_remote_sink_writer(block_writer: RemoteFactorBlockWriter) -> Callable[[list[Any]], None]:
    """构造 StreamingResultSink 兼容的 ``Callable[[list[ResultItem]], None]``。"""

    def _sink_writer(items: list[Any]) -> None:
        block_writer.write_results(items)

    return _sink_writer
