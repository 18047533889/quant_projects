# -*- coding: utf-8 -*-
"""R44-P0: 远程 IO 成本模型 —— 节点级增量 planner 的轻量估算器。

增量 planner 在决定「逐对象 range-read / 整对象 GET / multipart sink」之前，
先花 O(1) 时间估算一次 RemoteScanPlan 的远程 IO 账单。这是**只读估算器**：

- 不修改现有 ``data_access.read.scan_cost``（那是 executor 路由的权威成本模型）；
- 不真正访问对象存储；全部由 ``RemoteScanPlan`` 的字段 + 每行字节假设推导。

R44-P0 字段语义：
    - ``range_read_bytes``       ：只读必要字节范围的字节数（列投影 + row-group 裁剪）。
    - ``get_count``              ：整对象 GET 次数（footer 探测 / 全量读 / 小对象）。
    - ``network_transfer_bytes`` ：实际落进网络传输的字节数（range_read + 预估行数据）。
    - ``multipart_sink_bytes``   ：经 multipart 写入远程的字节数。
    - ``time_to_durable_commit_ms``：durable commit 的近似时延（估算）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .remote_dataset_scanner import RemoteScanPlan

__all__ = ["RemoteIOCost", "estimate_remote_io"]

#: 每对象固定开销（HEAD / LIST / footer 探测）的估算字节。
_DEFAULT_OBJECT_OVERHEAD_BYTES = 512
#: durable commit 的固定时延（单对象 PUT + fsync + manifest 指针，毫秒）。
_DURABLE_COMMIT_MS_PER_OBJECT = 4.0
#: 网络传输每字节估算毫秒（Gbps 级链路，粗略）。
_MS_PER_NETWORK_BYTE = 1e-6


@dataclass(frozen=True)
class RemoteIOCost:
    """一次远程扫描的 IO 成本账单（R44-P0，只读估算）。"""

    range_read_bytes: int
    get_count: int
    network_transfer_bytes: int
    multipart_sink_bytes: int
    time_to_durable_commit_ms: int

    def total_bytes(self) -> int:
        """网络传输 + multipart 写出的总字节（range_read 已含在 network 内不重复计）。"""
        return self.network_transfer_bytes + self.multipart_sink_bytes

    def estimate(self) -> dict[str, Any]:
        return {
            "range_read_bytes": self.range_read_bytes,
            "get_count": self.get_count,
            "network_transfer_bytes": self.network_transfer_bytes,
            "multipart_sink_bytes": self.multipart_sink_bytes,
            "time_to_durable_commit_ms": self.time_to_durable_commit_ms,
            "total_bytes": self.total_bytes(),
        }


def estimate_remote_io(
    plan: "RemoteScanPlan",
    *,
    bytes_per_row: float = 32.0,
    object_overhead_bytes: int = 512,
) -> RemoteIOCost:
    """估算 ``RemoteScanPlan`` 的远程 IO（R44-P0，纯函数）。

    假设：
        - 每个对象需 1 次 HEAD（在 get_count 中计为 footer/元数据探测）；
        - 投影列占全列集合的比例用 ``len(fields) / 总列数`` 近似（无 schema 时
          用 1.0）；此处保守用「每行 bytes_per_row × 行数」作为网络主体；
        - range-read 优化只当 ``object_keys`` 非空且字段子集存在时生效，且
          保守地不承诺 row-group 裁剪收益（估算只比全量 GET 省 object 头）。
        - multipart_sink_bytes：估算器不含写入量，默认 0（调用方若走 multipart
          sink 自行累加）。

    ``bytes_per_row`` 与 ``object_overhead_bytes`` 可调，供 planner 校准。
    """
    object_count = max(1, len(plan.object_keys))
    overhead = max(0, int(object_overhead_bytes))

    # 网络主体：每对象 overhead + 行数据（无行数信息时以对象数×overhead 兜底）。
    rows_estimate = _plan_rows_estimate(plan)
    data_bytes = int(rows_estimate * max(0.0, float(bytes_per_row)))
    network_transfer = data_bytes + object_count * overhead

    # range-read：字段子集越小，列裁剪收益越大。保守取 60% 上限（还含 row-group
    # 内未命中 date 的浪费），至少省掉每对象的固定 head 开销。
    fields = tuple(plan.fields)
    range_bytes = network_transfer
    if fields:
        range_bytes = max(
            data_bytes // 2,
            int(data_bytes * _column_selectivity(fields)),
        ) + object_count * overhead

    # get_count：每个对象至少 1 次（footer/全量）；range-read 会多 1 次 footer 探测。
    get_count = object_count * (2 if fields else 1)

    commit_ms = int(
        object_count * _DURABLE_COMMIT_MS_PER_OBJECT
        + network_transfer * _MS_PER_NETWORK_BYTE
    )
    return RemoteIOCost(
        range_read_bytes=max(0, range_bytes),
        get_count=get_count,
        network_transfer_bytes=network_transfer,
        multipart_sink_bytes=0,
        time_to_durable_commit_ms=commit_ms,
    )


def _plan_rows_estimate(plan: "RemoteScanPlan") -> int:
    """从 plan 字段估行数：无显式行数时按对象数×每对象行数近似。"""
    rows = getattr(plan, "estimated_rows", None)
    if isinstance(rows, int) and rows > 0:
        return rows
    return max(1, len(plan.object_keys)) * 8192


def _column_selectivity(fields: tuple[str, ...]) -> float:
    """字段子集 → 列裁剪比例（1 列 ~30%，2~4 列 ~50%，5+ ~70%，封顶）。"""
    n = len(fields)
    if n <= 1:
        return 0.3
    if n <= 4:
        return 0.5
    return 0.7
