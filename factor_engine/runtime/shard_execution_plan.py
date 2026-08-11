# -*- coding: utf-8 -*-
"""R38 P0-001/002（§4）：真实 AutoShard 执行计划 —— 从「改内存数字」到
「切成 N 个可执行 shard + merge barrier」。

R36 的 AutoShard 只把 ``resource_contract.peak_memory_bytes`` 改小（伪 sharding），
真实 workload 仍整份执行 → 更容易 OOM（R38-P0-001）。本模块定义真正可执行的
分片计划：

    - :class:`ShardDescriptor`：单个 shard 的输入切片 / warmup overlap / 输出切片
      / checkpoint / merge 顺序（§4 必须字段）。
    - :class:`ShardMergeContract`（R38-P0-002）：index union / duplicate key /
      shard ordering / overlap trim / dtype / timezone / column order / missing
      cells / state checkpoint continuity —— 全部显式声明。
    - :class:`ShardExecutionPlan`：original task → N shard children + MERGE task。
    - :func:`shape_signature`：shard 形状签名（R38-P0-005：OOM 后新 shape 签名
      必须 != 失败 shape 签名，否则禁止重试）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


#: 分片维度
SHARD_ASSET = "asset"
SHARD_TIME = "time"
SHARD_SESSION = "session"


@dataclass(frozen=True)
class ShardDescriptor:
    """一个可执行 shard（§4：``ShardDescriptor`` 必须字段全部给出）。"""

    shard_id: str
    dimension: str                 # asset / time / session
    #: 输入切片：asset → 仪器子集（tuple）；time/session → (start, end) 时间窗。
    input_slice: object
    #: warmup overlap（time-shard rolling/stateful 需要 lookback 历史）。
    warmup_slice: object | None = None
    #: 输出切片：本 shard 结果覆盖的范围（time → (block_start, block_end)）。
    output_slice: object | None = None
    checkpoint_ref: object | None = None
    preserves_full_cross_section: bool = False
    merge_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "dimension": self.dimension,
            "input_slice": _serialize_slice(self.input_slice),
            "warmup_slice": _serialize_slice(self.warmup_slice),
            "output_slice": _serialize_slice(self.output_slice),
            "checkpoint_ref": self.checkpoint_ref is not None,
            "preserves_full_cross_section": self.preserves_full_cross_section,
            "merge_order": self.merge_order,
        }


@dataclass(frozen=True)
class ShardMergeContract:
    """merge 显式契约（R38-P0-002）。每个字段都有值——不留隐式默认。"""

    index_union: str = "concat_sort"          # concat_sort | union
    duplicate_key_policy: str = "error"       # error | first_wins | last_wins
    shard_ordering: str = "merge_order_asc"   # merge_order_asc | by_index
    overlap_trim: str = "warmup_trim"         # warmup_trim | none
    dtype: str = "preserve"                   # preserve | float64 | ...
    timezone: str = "preserve"                # preserve | utc | ...
    column_order: str = "preserve_first"      # preserve_first | sorted
    missing_cells: str = "nan"                # nan | error
    state_checkpoint_continuity: str = "required_if_time_shard_stateful"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_union": self.index_union,
            "duplicate_key_policy": self.duplicate_key_policy,
            "shard_ordering": self.shard_ordering,
            "overlap_trim": self.overlap_trim,
            "dtype": self.dtype,
            "timezone": self.timezone,
            "column_order": self.column_order,
            "missing_cells": self.missing_cells,
            "state_checkpoint_continuity": self.state_checkpoint_continuity,
        }


@dataclass(frozen=True)
class ShardExecutionPlan:
    """original task → shard children + MERGE（§4 必须的编排结构）。"""

    original_task_id: str
    dimension: str
    shards: tuple[ShardDescriptor, ...]
    merge_policy: ShardMergeContract
    merge_task_id: str
    shard_task_ids: tuple[str, ...]
    per_shard_peak_bytes: int
    original_peak_bytes: int
    #: OOM 后 replan 用：本次分片已失败的 shape 签名（§R38-P0-005 禁止同 shape 重试）。
    failed_shape_signature: str = ""
    reason: str = ""
    #: OOM replan 重建用（真实切片信息，R38-P0-005）。
    time_range: tuple[Any, Any] | None = None
    instrument_universe: tuple[Any, ...] | None = None
    #: 该计划对应的 SafeEnvelope（OOM 后按同一 envelope replan，不随 live 波动）。
    safe_envelope_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_task_id": self.original_task_id,
            "dimension": self.dimension,
            "shard_count": len(self.shards),
            "shards": [s.to_dict() for s in self.shards],
            "merge_policy": self.merge_policy.to_dict(),
            "merge_task_id": self.merge_task_id,
            "shard_task_ids": list(self.shard_task_ids),
            "per_shard_peak_bytes": self.per_shard_peak_bytes,
            "original_peak_bytes": self.original_peak_bytes,
            "failed_shape_signature": self.failed_shape_signature,
            "reason": self.reason,
            "time_range": list(self.time_range) if self.time_range else None,
            "instrument_universe": (
                list(self.instrument_universe) if self.instrument_universe else None
            ),
        }


def shape_signature(
    *,
    task_id: str,
    dimension: str,
    shard_count: int,
    per_shard_peak_bytes: int,
) -> str:
    """分片形状签名：同 shape 重试禁止（§R38-P0-005 hard invariant）。"""
    payload = f"{task_id}|{dimension}|{shard_count}|{per_shard_peak_bytes}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _serialize_slice(slice_: Any) -> Any:
    """把 input/output slice 序列化为 JSON-safe（tuple 展开）。"""
    if isinstance(slice_, (tuple, list)):
        return list(slice_)
    return slice_


#: 便于自动 import 的默认（无）merge contract。
DEFAULT_MERGE_CONTRACT = ShardMergeContract()
