# -*- coding: utf-8 -*-
"""P11+P12: FeatureBlock —— 100k 因子不能是 100k 文件。

A-share ~5000 只股票。100k 因子 × 5000 × float32 ≈ 2GB/日，~2500 交易日 ≈ 5TB
全历史。因此**全历史永久物化所有 100k 因子不可行**，且物理落盘必须分组，绝不
逐因子一文件。

物理布局（date/time partition × factor block × 全 universe）：:

    base_dir/
      <partition>/block_<i:04d>.npy   值（float32，rows×ncols，ncols<=512）
      manifest.json                  元数据（与值分离）

manifest 建立 ``FactorID -> BlockID -> Column`` 映射，reader 据此回读。块大小由
row_count / writer 内存 / 读模式动态选择，绝不固定为 100k 文件。多个因子共享一个
值文件（同一 block），每因子一列，由 manifest 的 column_map 定位。

dtype 契约：因子值一律 ``float32``（``FACTOR_VALUE_DTYPE``）。写入强制 cast 到
float32；回读返回 float32 ndarray。元数据（manifest）单独存放，与值文件分离。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# ---- dtype contract ---------------------------------------------------------
FACTOR_VALUE_DTYPE = "float32"  # 显式 dtype 契约：因子值一律 float32
_BYTES_PER_CELL = np.dtype(FACTOR_VALUE_DTYPE).itemsize  # 4

# ---- P0-12：row-axis identity 契约 -------------------------------------------
# 行键（row identity）按 partition 存一次（不按 factor 重复）：每 partition 一
# 条 row_key 记录 —— 时序键（date/bar timestamp）与截面键（instrument universe
# 有序哈希）分开。reader 读取时以 partition 内任一块的 row_key 校验
# ``ordered_instrument_hash``：universe 顺序/成员被 shuffle 时立即 DETECT 失配并
# 抛错，绝不静默读错值。
ROW_AXIS_SCHEMA_VERSION = 1
#: 时序键列名（feature_block 行键最常用的时间标识）。
ROW_KEY_TIME_COL = "timestamp"

# ---- 块大小动态选择边界 -------------------------------------------------------
MIN_COLUMNS_PER_BLOCK = 64
MAX_COLUMNS_PER_BLOCK = 512
_DEFAULT_WRITER_MEMORY_BYTES = 1 << 30  # 1GiB 缺省 writer 内存预算
_HEADROOM_FACTOR = 4  # 保守系数：把 block 字节上限压到预算的 1/4 以下
_ROWS_PER_GROWTH_STEP = 100  # 每行数增量 → 每块多一列的增速

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
VALUES_SUFFIX = ".npy"


def choose_columns_per_block(
    total_factors: int | None = None,
    row_count: int | None = None,
    writer_memory_bytes: int | None = None,
    *,
    min_cols: int = MIN_COLUMNS_PER_BLOCK,
    max_cols: int = MAX_COLUMNS_PER_BLOCK,
) -> int:
    """动态块大小（列数）：row_count / writer 内存 / 读模式共同决定。

    两条相反的作用力：
      * 行驱动增长 —— 行越多单块列越多（分摊 axis/header，读模式偏好大块）；
      * 内存上限 —— block 值字节 = rows×cols×4 必须 <= writer 内存预算，行太多
        时压回小块，绝不超预算。

    最终值夹在 ``[min_cols, max_cols]``。纯函数，便于单测。
    """
    max_cols = max(min_cols, int(max_cols))
    rows = max(1, int(row_count if row_count is not None else (total_factors or 1)))
    budget = max(1, int(writer_memory_bytes or _DEFAULT_WRITER_MEMORY_BYTES))

    # 内存上限（硬约束）：cols <= budget / (rows × bytes_per_cell)，留 1/4 headroom。
    # 内存逼到 min_cols 以下时以内存为准（绝不让单 block 字节超预算）。
    mem_capped = max(1, budget // max(1, rows * _BYTES_PER_CELL * _HEADROOM_FACTOR))
    # 行驱动增长（软下限）：行越多 → 单块列越多（读模式偏好整块读取）。
    row_based = min(max_cols, max(min_cols, min_cols + rows // _ROWS_PER_GROWTH_STEP))

    cols = min(max_cols, mem_capped, row_based)
    return max(1, cols)


def block_bytes(row_count: int, columns: int, dtype: str = FACTOR_VALUE_DTYPE) -> int:
    """一个 block 值文件的字节估算（float32 契约）。"""
    return max(1, int(row_count)) * max(1, int(columns)) * np.dtype(dtype).itemsize


# ---- manifest 结构 ----------------------------------------------------------
@dataclass
class RowAxisIdentity:
    """P0-12：block 文件的行轴身份。按 partition 存一次，不按 factor 重复。

    - 时序（row-order）键：``partition_time``（分区代表时间）、``calendar_snapshot_id``
      （交易历快照）、``source_snapshot_id``（上游数据快照）。
    - 截面（universe）键：``market``、``universe_snapshot_id``、有序仪器列
      哈希 ``ordered_instrument_hash``（列名顺序敏感；shuffle universe →
      哈希失配）。
    - ``row_axis_ref`` 是版本化行轴引用（内部 base64），行键内容一致性由
      reader 在读取时重新生成该哈希来校验。
    """

    schema_version: int = ROW_AXIS_SCHEMA_VERSION
    partition_time: str = ""
    market: str = ""
    universe_snapshot_id: str = ""
    ordered_instrument_hash: str = ""
    row_axis_ref: str = ""
    calendar_snapshot_id: str = ""
    source_snapshot_id: str = ""
    #: 行键记录本体（每 partition 存一次；不按 factor）。
    row_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "partition_time": self.partition_time,
            "market": self.market,
            "universe_snapshot_id": self.universe_snapshot_id,
            "ordered_instrument_hash": self.ordered_instrument_hash,
            "row_axis_ref": self.row_axis_ref,
            "calendar_snapshot_id": self.calendar_snapshot_id,
            "source_snapshot_id": self.source_snapshot_id,
            "row_keys": list(self.row_keys),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RowAxisIdentity":
        return cls(
            schema_version=int(d.get("schema_version", ROW_AXIS_SCHEMA_VERSION)),
            partition_time=str(d.get("partition_time", "")),
            market=str(d.get("market", "")),
            universe_snapshot_id=str(d.get("universe_snapshot_id", "")),
            ordered_instrument_hash=str(d.get("ordered_instrument_hash", "")),
            row_axis_ref=str(d.get("row_axis_ref", "")),
            calendar_snapshot_id=str(d.get("calendar_snapshot_id", "")),
            source_snapshot_id=str(d.get("source_snapshot_id", "")),
            row_keys=list(d.get("row_keys", [])),
        )

    @property
    def row_order_fingerprint(self) -> str:
        """行键排列指纹（row_keys 顺序敏感）。reader 用它检测行顺序被 shuffle。

        空 row_keys 返回空串（后退到 ordered_instrument_hash 校验）。
        """
        if not self.row_keys:
            return ""
        return compute_row_axis_ref(
            ordered_names=True,
            row_keys=self.row_keys,
        )


def compute_row_axis_ref(
    *,
    market: str = "",
    universe: Sequence[str] = (),
    ordered_names: bool = True,
    row_keys: Sequence[str] = (),
) -> str:
    """生成版本化行轴引用（base64 哈希）。

    - ``universe``：仪器有序列表。``ordered_names=True`` 时哈希对顺序敏感
      （shuffle → 哈希失配）。为保持对"历史 manifest 无 row-axis"的兼容，
      ``ordered_names=False`` 时对排序后的名称求值（universe 成员集校验，
      不校验顺序）。
    - ``row_keys``：行键列表。有则把行键排列一并纳入指纹（顺序敏感）。
    """
    h = hashlib.sha256()
    h.update(b"feature_block_row_axis/v1")
    h.update(b"market=" + str(market).encode("utf-8"))
    h.update(b"\x00")
    if ordered_names:
        names = list(universe)
    else:
        names = sorted(str(n) for n in universe)
    for n in names:
        h.update(str(n).encode("utf-8"))
        h.update(b"\x1f")
    for n in row_keys:
        h.update(str(n).encode("utf-8"))
        h.update(b"\x1e")
    return base64.b64encode(h.digest()).decode("ascii")


class RowAxisMismatchError(RuntimeError):
    """P0-12：row-axis identity 校验失败（universe 顺序/成员/行键排列失配）。"""


import base64  # noqa: E402


@dataclass
class BlockSpec:
    """一个物理 block 的元数据（与值文件分离存放）。"""

    block_id: str
    partition: str
    file: str  # 相对 base_dir 的值文件路径
    row_count: int
    factor_ids: list[str]
    column_map: dict[str, int]  # FactorID -> 列号（该 block 内）

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "partition": self.partition,
            "file": self.file,
            "row_count": self.row_count,
            "factor_ids": list(self.factor_ids),
            "column_map": dict(self.column_map),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockSpec":
        return cls(
            block_id=d["block_id"],
            partition=d["partition"],
            file=d["file"],
            row_count=int(d["row_count"]),
            factor_ids=list(d["factor_ids"]),
            column_map={k: int(v) for k, v in d["column_map"].items()},
        )


@dataclass
class BlockManifest:
    """根 manifest：把 FactorID -> BlockID -> Column 全图映射。

    Wave1-F identity: ``identities`` (factor_id -> FactorValueIdentity dict) 让
    manifest 携带因子的值身份（definition/implementation/treatment/market/freq/
    universe/data/calendar/numeric_policy/dtype/build_sha/block_content_hash），
    reader 据此校验 identity <-> 实际 block 内容一致，模型层据此消费身份而非裸因子名。
    ``feature_set_identity`` (FeatureSetIdentity dict) 是给定回测因子集的集合级身份。
    """

    schema_version: int
    dtype: str
    blocks: dict[str, BlockSpec]  # block_id -> spec
    factor_to_block: dict[str, dict[str, str]]  # FactorID -> {partition: block_id}
    partition_rows: dict[str, int]  # partition -> 行数
    columns_per_block: int | None = None
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)  # factor_id -> FactorValueIdentity.to_dict()
    feature_set_identity: dict[str, Any] | None = None  # FeatureSetIdentity.to_dict()
    # P0-12 row-axis identity：partition -> RowAxisIdentity（行键每 partition 存一次）。
    row_axis: dict[str, RowAxisIdentity] = field(default_factory=dict)

    def resolve(self, factor_id: str) -> list[tuple[str, BlockSpec, int]]:
        """FactorID -> [(partition, BlockSpec, column), ...]（跨全部分区）。"""
        out: list[tuple[str, BlockSpec, int]] = []
        per_partition = self.factor_to_block.get(factor_id)
        if not per_partition:
            return out
        for partition, block_id in per_partition.items():
            block = self.blocks.get(block_id)
            if block is None:
                continue
            column = block.column_map.get(factor_id)
            if column is None:
                continue
            out.append((partition, block, column))
        return out

    def verify_row_axis(
        self,
        *,
        expected_universe: Sequence[str] | None = None,
        expected_order: Sequence[str] | None = None,
        strict_members: bool = True,
    ) -> None:
        """校验 row-axis identity（partition 级）。

        - 对每个 partition 的 :class:`RowAxisIdentity`：若提供了
          ``expected_universe``，用 ``compute_row_axis_ref`` 重算并比对
          ``ordered_instrument_hash`` —— 失配抛 :class:`RowAxisMismatchError`。
        - 若 partition 记录了 ``row_keys``，其行键排列指纹也参与校验
          （行键 shuffle 被 DETECT）。
        - ``strict_members=False`` 只校验成员集（universe 顺序不敏感）。
        - 无 row_axis（旧 manifest）→ 跳过（向后兼容，不静默读错值风险）。
        """
        if not self.row_axis:
            return
        for partition, identity in self.row_axis.items():
            if expected_universe is not None:
                # ordered_instrument_hash 只代表 universe（列顺序）—— row_keys
                # 是独立的行键排列指纹，不混入 universe 哈希。
                ref = compute_row_axis_ref(
                    market=identity.market,
                    universe=expected_universe,
                    ordered_names=strict_members,
                )
                if ref != identity.ordered_instrument_hash:
                    raise RowAxisMismatchError(
                        f"partition={partition!r} row-axis 失配："
                        f"ordered_instrument_hash={identity.ordered_instrument_hash[:16]}… "
                        f"expected={ref[:16]}…（universe 顺序/成员变化被 DETECT）"
                    )
            # 可选行键排列校验（expected_order 提供时）：行键 shuffle 也被 DETECT。
            if expected_order is not None and identity.row_keys:
                order_ref = compute_row_axis_ref(
                    ordered_names=True,
                    row_keys=expected_order,
                )
                if order_ref != identity.row_order_fingerprint:
                    raise RowAxisMismatchError(
                        f"partition={partition!r} row-key 排列失配："
                        f"row_keys 被 shuffle（行键顺序变化被 DETECT）"
                    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "dtype": self.dtype,
            "columns_per_block": self.columns_per_block,
            "partition_rows": dict(self.partition_rows),
            "factor_to_block": {
                k: dict(v) for k, v in self.factor_to_block.items()
            },
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
        }
        if self.identities:
            payload["identities"] = dict(self.identities)
        if self.feature_set_identity is not None:
            payload["feature_set_identity"] = self.feature_set_identity
        if self.row_axis:
            payload["row_axis"] = {
                k: v.to_dict() for k, v in self.row_axis.items()
            }
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockManifest":
        return cls(
            schema_version=int(d["schema_version"]),
            dtype=str(d["dtype"]),
            columns_per_block=d.get("columns_per_block"),
            partition_rows=dict(d.get("partition_rows", {})),
            factor_to_block={
                k: dict(v) for k, v in d.get("factor_to_block", {}).items()
            },
            blocks={
                k: BlockSpec.from_dict(v) for k, v in d.get("blocks", {}).items()
            },
            identities={
                k: dict(v) for k, v in d.get("identities", {}).items()
            },
            feature_set_identity=(
                dict(d["feature_set_identity"])
                if d.get("feature_set_identity")
                else None
            ),
            row_axis={
                k: RowAxisIdentity.from_dict(v)
                for k, v in d.get("row_axis", {}).items()
            },
        )


# ---- writer -----------------------------------------------------------------
def _partition_token(partition: str) -> str:
    """把 partition 压成短 token 用于 block_id（同 partition 内块连续编号）。"""
    import hashlib

    return hashlib.blake2b(str(partition).encode("utf-8"), digest_size=3).hexdigest()


class FeatureBlockWriter:
    """物理 writer：把多个因子分组进一个 block 值文件，绝不逐因子一文件。

    每个 partition 缓冲因子值；缓冲满 ``columns_per_block`` 个即 flush 成一个
    ``(rows, ncols)`` float32 block 文件，manifest 记录每因子 → block → 列。
    ``finish()`` 落残余块 + 写根 manifest。元数据（manifest）与值分离。

    块大小由 ``choose_columns_per_block`` 动态决定（row_count / writer 内存 /
    读模式），并遵守 writer 内存上限 —— 不固定为 100k 文件。
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        dtype: str = FACTOR_VALUE_DTYPE,
        columns_per_block: int | None = None,
        writer_memory_bytes: int | None = None,
        default_partition: str = "default",
        build_sha: str = "",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.dtype = str(dtype or FACTOR_VALUE_DTYPE)
        self.default_partition = str(default_partition)
        self.columns_per_block = columns_per_block
        self.writer_memory_bytes = writer_memory_bytes
        self.build_sha = str(build_sha)
        # partition -> {factor_id: 1D float32 ndarray}
        self._buffer: dict[str, dict[str, np.ndarray]] = {}
        self._order: dict[str, list[str]] = {}  # partition -> 插入顺序 factor_id
        self._partition_rows: dict[str, int] = {}
        self._blocks: dict[str, BlockSpec] = {}
        self._factor_to_block: dict[str, dict[str, str]] = {}
        self._block_seq = 0
        self._manifest: BlockManifest | None = None
        self._finished = False
        # Wave1-F identity: factor_id -> FactorValueIdentity dict（写入时逐因子供给）。
        self._identities: dict[str, dict[str, Any]] = {}
        self._feature_set_identity: dict[str, Any] | None = None
        # P0-12 row-axis：partition -> RowAxisIdentity（行键每 partition 存一次）。
        self._row_axis: dict[str, Any] = {}

    # -- 状态 ---------------------------------------------------------------
    def buffered_factors(self) -> int:
        return sum(len(v) for v in self._buffer.values())

    def block_count(self) -> int:
        return len(self._blocks)

    # -- Wave1-F identity ---------------------------------------------------
    def set_identity(self, factor_id: str, identity: Any) -> None:
        """把某因子的 FactorValueIdentity 交给 writer，随 manifest 落盘。

        identity 可以是 ``FactorValueIdentity`` 对象或 ``to_dict()`` 结果。
        writer 只接受一次（重复 set 幂等忽略）。
        """
        if self._finished:
            raise RuntimeError("FeatureBlockWriter.finish() 已调用，不能再 set_identity")
        if identity is None:
            return
        if hasattr(identity, "to_dict"):
            payload = identity.to_dict()
        elif isinstance(identity, dict):
            payload = dict(identity)
        else:  # pragma: no cover - 防御
            raise TypeError(
                f"set_identity 需 FactorValueIdentity 或其 dict，got {type(identity)}"
            )
        self._identities.setdefault(str(factor_id), payload)

    def set_feature_set_identity(self, identity: Any) -> None:
        """把 FeatureSetIdentity 交给 writer，随 manifest 落盘。"""
        if self._finished:
            raise RuntimeError(
                "FeatureBlockWriter.finish() 已调用，不能再 set_feature_set_identity"
            )
        if identity is None:
            return
        if hasattr(identity, "to_dict"):
            self._feature_set_identity = identity.to_dict()
        else:
            self._feature_set_identity = dict(identity)

    def _build_missing_identities(self) -> dict[str, dict[str, Any]]:
        """为尚未显式 set_identity 的因子补齐 FactorValueIdentity（幂等）。

        候补哈希用轻量 descriptor（factor_id/freq/dtype/build_sha + partition），
        block_content_hash 从 block 真实字节计算 —— 保证「identity 的 content hash
        与物理内容一致」这条契约对无显式身份因子也成立。

        另外，显式 set_identity 但未携带 block_content_hash 的因子，也用真实 block
        字节补算 content hash —— 这样 reader 的 verify 永远有内容可校验（非空）。
        """
        from factor_engine.runtime.factor_value_identity import (
            UNKNOWN_SNAPSHOT,
            compute_block_content_hash,
            factor_value_identity_fields,
        )

        def _content_hash_of(spec: BlockSpec) -> str:
            if spec is None:
                return ""
            try:
                arr = np.load(self.base_dir / spec.file, allow_pickle=False)
                return compute_block_content_hash(
                    arr,
                    factor_ids=list(spec.factor_ids),
                    numeric_policy=self.dtype,
                )
            except Exception:  # pragma: no cover - 值文件尚未落盘等
                return ""

        out = dict(self._identities)
        for fid, per_partition in self._factor_to_block.items():
            partition = next(iter(per_partition.keys()))
            block_id = per_partition[partition]
            spec = self._blocks.get(block_id)
            if fid in out:
                # 显式 identity：若 content hash 为空则用真实字节补算。
                if not out[fid].get("block_content_hash"):
                    out[fid] = dict(out[fid])
                    out[fid]["block_content_hash"] = _content_hash_of(spec)
                continue
            out[fid] = factor_value_identity_fields(
                factor_id=fid,
                market="",
                frequency="",
                partition_time=str(partition),
                universe_snapshot_id=UNKNOWN_SNAPSHOT,
                ordered_instrument_hash="",
                data_snapshot_id=UNKNOWN_SNAPSHOT,
                calendar_snapshot_id=UNKNOWN_SNAPSHOT,
                decision_time_policy="",
                numeric_policy=self.dtype,
                dtype=self.dtype,
                build_sha=self.build_sha,
                block_content_hash=_content_hash_of(spec),
            )
        return out

    # -- 主写入 -------------------------------------------------------------
    # P0-12 row-axis：partition -> RowAxisIdentity（行键每 partition 存一次）。
    def set_row_axis(self, identity: RowAxisIdentity | dict[str, Any]) -> None:
        """把一个 partition 的 row-axis identity 交给 writer，随 manifest 落盘。

        identity 可以是 :class:`RowAxisIdentity` 或 ``to_dict()`` 结果。
        默认按 default_partition 生效（partition 参数未给时）。P0-12：每
        partition 存一份行键，不按 factor 重复。
        """
        if self._finished:
            raise RuntimeError("FeatureBlockWriter.finish() 已调用，不能再 set_row_axis")
        if identity is None:
            return
        if hasattr(identity, "to_dict"):
            payload = identity.to_dict()
        elif isinstance(identity, dict):
            payload = dict(identity)
        else:  # pragma: no cover - 防御
            raise TypeError(
                f"set_row_axis 需 RowAxisIdentity 或其 dict，got {type(identity)}"
            )
        self._row_axis[self.default_partition] = RowAxisIdentity.from_dict(payload)

    def add(self, factor_id: str, values: Any, partition: str | None = None) -> None:
        """把一个因子的一段值（1D float32）加入缓冲，满了即 flush 成整块。"""
        if self._finished:
            raise RuntimeError("FeatureBlockWriter.finish() 已调用，不能再 add")
        partition = str(partition) if partition is not None else self.default_partition
        arr = self._coerce_values(values, factor_id)
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
        """StreamingResultSink / writer 路径接线：item.name=factor_id，
        item.value=1D 值，item.meta["partition"] 可选覆盖。"""
        for item in items:
            name = getattr(item, "name", None)
            value = getattr(item, "value", None)
            meta = getattr(item, "meta", None) or {}
            if name is None or value is None:
                continue
            self.add(str(name), value, partition=meta.get("partition"))

    def flush_partition(self, partition: str) -> None:
        """把某个 partition 的缓冲落成一个（或多个）block 值文件。"""
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
            chunk = fids[chunk_start:chunk_start + cols]
            self._write_one_block(partition, rows, chunk, part_buf)
        # flush 后清空该 partition 缓冲
        self._buffer[partition] = {}
        self._order[partition] = []

    def _write_one_block(
        self,
        partition: str,
        rows: int,
        fids: Sequence[str],
        part_buf: Mapping[str, np.ndarray],
    ) -> None:
        block_id = f"b{_partition_token(partition)}_{self._block_seq:04d}"
        seq = self._block_seq
        self._block_seq += 1
        rel = f"{partition}/block_{seq:04d}{VALUES_SUFFIX}"
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # 一个 block 值文件：rows × len(fids) float32。列顺序与 fids 对齐。
        block = np.empty((rows, len(fids)), dtype=self.dtype)
        column_map: dict[str, int] = {}
        for col, fid in enumerate(fids):
            col_arr = part_buf[fid]
            if col_arr.shape[0] == rows:
                block[:, col] = col_arr
            else:
                # 行数不一致兜底：取重叠（不应发生；同 partition 行数一致）。
                block[:, col] = col_arr[:rows]
            column_map[fid] = col
        np.save(path, np.ascontiguousarray(block))
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

    def _resolve_columns_per_block(self, *, row_count: int, factor_total: int) -> int:
        if self.columns_per_block is not None:
            return max(1, int(self.columns_per_block))
        return choose_columns_per_block(
            total_factors=factor_total,
            row_count=row_count,
            writer_memory_bytes=self.writer_memory_bytes,
        )

    def finish(self) -> Path:
        """落残余缓冲 + 写根 manifest。返回 manifest 路径。"""
        if self._finished:
            return self.manifest_path
        for partition in list(self._buffer.keys()):
            self.flush_partition(partition)
        self._manifest = BlockManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            dtype=self.dtype,
            blocks=dict(self._blocks),
            factor_to_block={
                k: dict(v) for k, v in self._factor_to_block.items()
            },
            partition_rows=dict(self._partition_rows),
            columns_per_block=self.columns_per_block,
            identities=self._build_missing_identities(),
            feature_set_identity=self._feature_set_identity,
            row_axis=dict(self._row_axis),
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.manifest_path, self._manifest.to_dict())
        self._finished = True
        return self.manifest_path

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / MANIFEST_NAME

    @property
    def manifest(self) -> BlockManifest | None:
        return self._manifest

    # -- 内部工具 -----------------------------------------------------------
    def _coerce_values(self, values: Any, factor_id: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError(
                    f"FeatureBlockWriter: 因子 {factor_id} 值必须为 1D，got ndim={arr.ndim}"
                )
        if arr.dtype != np.dtype(FACTOR_VALUE_DTYPE):
            arr = arr.astype(FACTOR_VALUE_DTYPE)  # 显式 dtype 契约强制 float32
        return np.ascontiguousarray(arr)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---- reader -----------------------------------------------------------------
def load_manifest(base_dir: str | Path) -> BlockManifest:
    """读取根 manifest。"""
    path = Path(base_dir) / MANIFEST_NAME
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if int(data.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema 不匹配：{data.get('schema_version')} != {MANIFEST_SCHEMA_VERSION}"
        )
    return BlockManifest.from_dict(data)


class FeatureBlockReader:
    """reader：FactorID -> BlockID -> Column，回读因子值（float32 契约）。

    Wave1-F identity：``read_block`` 之后可调用 ``verify_manifest_identities``
    校验 manifest 内每因子的 identity.block_content_hash 与物理 block 内容一致；
    不一致返回 mismatch 列表（调用方据此拒绝读取）。

    P0-12 row-axis：构造时提供 ``expected_universe`` 后，每次 ``read_block`` /
    ``resolve`` 都会调用 ``manifest.verify_row_axis`` —— universe 顺序/成员被
    shuffle 时抛 :class:`RowAxisMismatchError`（绝不会静默读错值）。
    """

    def __init__(
        self,
        base_dir: str | Path,
        manifest: BlockManifest | None = None,
        *,
        expected_universe: Sequence[str] | None = None,
        verify_row_axis: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.manifest = manifest or load_manifest(base_dir)
        self.expected_universe = (
            list(expected_universe) if expected_universe is not None else None
        )
        self.verify_row_axis = bool(verify_row_axis)

    def _maybe_verify_row_axis(self) -> None:
        if self.verify_row_axis and self.expected_universe is not None:
            self.manifest.verify_row_axis(expected_universe=self.expected_universe)

    def resolve(self, factor_id: str) -> list[tuple[str, BlockSpec, int]]:
        self._maybe_verify_row_axis()
        return self.manifest.resolve(factor_id)

    def verify_manifest_identities(self) -> list[str]:
        """校验 manifest 里所有 identity 的 content hash 与真实 block 内容一致。

        返回 mismatch 的 factor_id 列表；空列表即全部通过。只校验携带
        ``block_content_hash`` 的 identity（无 identity 的旧 manifest 跳过 ——
        向后兼容，不破坏旧读路径）。
        """
        from factor_engine.runtime.factor_value_identity import (
            verify_manifest_factors_against_content,
        )

        return verify_manifest_factors_against_content(
            self.manifest, reader=self
        )

    def identities(self) -> dict[str, dict[str, Any]]:
        """manifest 内的 ``{factor_id: FactorValueIdentity dict, ...}``。"""
        return dict(self.manifest.identities)

    def feature_set_identity(self) -> dict[str, Any] | None:
        """manifest 内的 FeatureSetIdentity dict（若有）。"""
        return self.manifest.feature_set_identity

    def read_block(self, block_id: str) -> np.ndarray:
        spec = self.manifest.blocks.get(block_id)
        if spec is None:
            raise KeyError(f"block 不存在: {block_id}")
        self._maybe_verify_row_axis()
        arr = np.load(self.base_dir / spec.file, allow_pickle=False)
        if arr.dtype != np.dtype(FACTOR_VALUE_DTYPE):
            arr = arr.astype(FACTOR_VALUE_DTYPE)
        return np.ascontiguousarray(arr)

    def read_factor(self, factor_id: str) -> dict[str, np.ndarray]:
        """返回 ``{partition: 1D float32 ndarray}``。"""
        out: dict[str, np.ndarray] = {}
        for partition, spec, column in self.resolve(factor_id):
            block = self.read_block(spec.block_id)
            out[partition] = np.ascontiguousarray(block[:, column])
        return out

    def read_factor_full(self, factor_id: str) -> np.ndarray:
        """跨全部分区拼接（partition 排序）—— TIER2 全历史读路径。"""
        per = self.read_factor(factor_id)
        if not per:
            raise KeyError(f"因子不存在: {factor_id}")
        parts = [per[k] for k in sorted(per)]
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def factor_count(self) -> int:
        return len(self.manifest.factor_to_block)

    def block_count(self) -> int:
        return len(self.manifest.blocks)


# ---- sink 接线：result-complete -> hand to writer ---------------------------
def make_sink_writer(block_writer: FeatureBlockWriter) -> Any:
    """构造 StreamResultSink 兼容的 ``Callable[[list[ResultItem]], None]``。

    用法（result-complete → hand to writer）：:

        sink = StreamingResultSink(writer=make_sink_writer(block_writer), ...)
        sink.start(); ... sink.submit(name=fid, value=1d_array, partition=p) ...
        sink.finish()
        block_writer.finish()   # 落残余块 + 写 manifest
    """

    def _sink_writer(items: list[Any]) -> None:
        block_writer.write_results(items)

    return _sink_writer
