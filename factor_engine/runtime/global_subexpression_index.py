# -*- coding: utf-8 -*-
"""P78-P7: GlobalSubexpressionIndex —— 全局共享子表达式索引（跨 cohort CSE）。

设计目标（用户 P7/P8）：
    - 全局 CSE 不能因 cohort 切分而丢失：一个子表达式被 cohort A 与 cohort B
      的因子共享时，必须**物化一次**（内存或 spill）并复用，而不是每个 cohort
      各重算一次。
    - 本索引以 ``subexpression_semantic_hash`` 为键，记录：
        ``reuse_count``（全局消费次数，跨 cohort）、``recompute_cost``（重算成本）、
        ``materialized_bytes``（物化后内存占用）。
    - 决策：对每个共享子表达式，基于「复用收益 vs 内存租金 + spill 成本 +
      传输成本」决定 ``MATERIALIZE_MEMORY`` / ``MATERIALIZE_SPILL`` / ``RECOMPUTE``
      —— **不是**「缓存每一个重复」。

与既有框架的关系（不重建）：
    - 复用 :class:`~factor_engine.runtime.multibackend.cse_cache_optimizer.CSECacheOptimizer`
      的 pin/unpin/evict 语义与 :class:`~factor_engine.runtime.multibackend.spill_strategy.SpillStore`
      的 spill/restore 语义；本索引是**决策层**（决定物化方式），不重复实现缓存。
    - 引用计数（refcount）跨 cohort 累计：最后一个消费者完成时 refcount 归零 →
      立即 unpin/free（last-consumer free）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

#: 物化决策（cost-aware CSE，非「缓存每一个重复」）。
MATERIALIZE_MEMORY = "MATERIALIZE_MEMORY"
MATERIALIZE_SPILL = "MATERIALIZE_SPILL"
RECOMPUTE = "RECOMPUTE"


@dataclass
class SubexpressionEntry:
    """单个共享子表达式的全局索引条目。"""

    semantic_hash: str
    reuse_count: int = 0
    recompute_cost: float = 0.0
    materialized_bytes: int = 0
    #: 当前物化状态（None = 未物化 / 已释放）。
    action: str | None = None
    #: 跨 cohort 累计的引用计数（最后一个消费者完成 → 0 → 释放）。
    refcount: int = 0
    #: 是否已物化（内存或 spill）。
    materialized: bool = False
    #: 物化位置（"memory" | "spill"）。
    location: str | None = None
    #: 物化值（内存态；spill 时由 SpillStore 持有，此处仅存 spill_id）。
    value: Any = None
    spill_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_hash": self.semantic_hash,
            "reuse_count": self.reuse_count,
            "recompute_cost": self.recompute_cost,
            "materialized_bytes": self.materialized_bytes,
            "action": self.action,
            "refcount": self.refcount,
            "materialized": self.materialized,
            "location": self.location,
            "spill_id": self.spill_id,
        }


@dataclass
class CSEDecision:
    """一次 cost-aware CSE 决策的结果。"""

    semantic_hash: str
    action: str
    reuse_count: int
    recompute_cost: float
    materialized_bytes: int
    #: 决策依据（可解释性）。
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_hash": self.semantic_hash,
            "action": self.action,
            "reuse_count": self.reuse_count,
            "recompute_cost": self.recompute_cost,
            "materialized_bytes": self.materialized_bytes,
            "reason": self.reason,
        }


@dataclass
class CSELivenessTelemetry:
    """CSE liveness 遥测（任务指定）。"""

    peak_cache_bytes: int = 0
    eviction_count: int = 0
    spill_count: int = 0
    recompute_count: int = 0
    reuse_saved_ms: float = 0.0
    materialize_memory_count: int = 0
    materialize_spill_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_cache_bytes": self.peak_cache_bytes,
            "eviction_count": self.eviction_count,
            "spill_count": self.spill_count,
            "recompute_count": self.recompute_count,
            "reuse_saved_ms": round(self.reuse_saved_ms, 3),
            "materialize_memory_count": self.materialize_memory_count,
            "materialize_spill_count": self.materialize_spill_count,
        }


#: 默认成本参数（可覆盖）。
#: 内存租金：每字节每单位时间的内存占用成本（相对单位）。
_DEFAULT_MEMORY_RENT_PER_BYTE = 1e-9
#: spill 成本：每字节 spill 的固定成本（序列化 + 写盘 + 读回）。
_DEFAULT_SPILL_COST_PER_BYTE = 2e-9
#: 传输成本：每字节跨 backend/cohort 传输的成本。
_DEFAULT_TRANSFER_COST_PER_BYTE = 1e-9
#: 物化固定成本（一次物化的固定开销）。
_DEFAULT_MATERIALIZE_FIXED_COST = 1.0


class GlobalSubexpressionIndex:
    """全局共享子表达式索引（跨 cohort CSE 决策 + 引用计数 + 遥测）。

    线程安全（``threading.RLock``），供 cohort executor 并发消费。
    """

    def __init__(
        self,
        *,
        memory_rent_per_byte: float = _DEFAULT_MEMORY_RENT_PER_BYTE,
        spill_cost_per_byte: float = _DEFAULT_SPILL_COST_PER_BYTE,
        transfer_cost_per_byte: float = _DEFAULT_TRANSFER_COST_PER_BYTE,
        materialize_fixed_cost: float = _DEFAULT_MATERIALIZE_FIXED_COST,
    ) -> None:
        self._entries: dict[str, SubexpressionEntry] = {}
        self._lock = threading.RLock()
        self._telemetry = CSELivenessTelemetry()
        self._memory_rent_per_byte = float(memory_rent_per_byte)
        self._spill_cost_per_byte = float(spill_cost_per_byte)
        self._transfer_cost_per_byte = float(transfer_cost_per_byte)
        self._materialize_fixed_cost = float(materialize_fixed_cost)

    # -- 注册 / 引用计数 --

    def register(
        self,
        semantic_hash: str,
        *,
        recompute_cost: float,
        materialized_bytes: int,
        reuse_count: int = 1,
    ) -> None:
        """注册一个共享子表达式（幂等：重复注册累加 reuse_count）。"""
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None:
                entry = SubexpressionEntry(
                    semantic_hash=semantic_hash,
                    recompute_cost=float(recompute_cost),
                    materialized_bytes=int(materialized_bytes),
                    reuse_count=int(reuse_count),
                )
                self._entries[semantic_hash] = entry
            else:
                entry.reuse_count += int(reuse_count)
                entry.recompute_cost = float(recompute_cost)
                entry.materialized_bytes = int(materialized_bytes)

    def add_consumer(self, semantic_hash: str) -> None:
        """一个消费者（跨 cohort）开始消费 → refcount +1。"""
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is not None:
                entry.refcount += 1

    def release_consumer(self, semantic_hash: str) -> bool:
        """一个消费者完成 → refcount -1；归零则释放（last-consumer free）。

        Returns:
            True 表示 refcount 已归零且条目被释放（unpin/free）。
        """
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None:
                return False
            entry.refcount = max(0, entry.refcount - 1)
            if entry.refcount == 0 and entry.materialized:
                self._free_entry(entry)
                return True
            return False

    def _free_entry(self, entry: SubexpressionEntry) -> None:
        """释放物化条目（内存态丢弃 value；spill 态标记未物化）。"""
        if entry.location == "memory":
            entry.value = None
        entry.materialized = False
        entry.location = None
        entry.action = None
        entry.spill_id = None

    # -- cost-aware CSE 决策 --

    def decide(self, semantic_hash: str) -> CSEDecision:
        """对共享子表达式做 cost-aware 物化决策。

        决策模型（任务指定：复用收益 vs 内存租金 + spill 成本 + 传输成本）：
            - 复用收益 ``reuse_benefit = (reuse_count - 1) * recompute_cost``
              （原需 reuse_count 次重算，物化后只算 1 次）。
            - 内存租金 ``memory_rent = materialized_bytes * memory_rent_per_byte``。
            - spill 成本 ``spill_cost = materialized_bytes * spill_cost_per_byte``。
            - 传输成本 ``transfer_cost = materialized_bytes * transfer_cost_per_byte``。

        规则：
            - ``reuse_count < 2`` → RECOMPUTE（无复用，物化无意义）。
            - ``reuse_benefit <= materialize_fixed_cost`` → RECOMPUTE（收益不抵固定开销）。
            - 内存租金可承受（``reuse_benefit > memory_rent + fixed``）→ MATERIALIZE_MEMORY。
            - 否则若 spill 后仍划算（``reuse_benefit > spill_cost + transfer_cost + fixed``）
              → MATERIALIZE_SPILL。
            - 否则 → RECOMPUTE。

        注意：**不是**「缓存每一个重复」——低复用 / 高租金 / 高成本的子表达式
        明确选择 RECOMPUTE。
        """
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None:
                return CSEDecision(
                    semantic_hash=semantic_hash,
                    action=RECOMPUTE,
                    reuse_count=0,
                    recompute_cost=0.0,
                    materialized_bytes=0,
                    reason="unknown_subexpression",
                )
            reuse = entry.reuse_count
            recompute = entry.recompute_cost
            mbytes = entry.materialized_bytes
            fixed = self._materialize_fixed_cost
            if reuse < 2:
                return CSEDecision(
                    semantic_hash=semantic_hash,
                    action=RECOMPUTE,
                    reuse_count=reuse,
                    recompute_cost=recompute,
                    materialized_bytes=mbytes,
                    reason=f"reuse_count={reuse}<2",
                )
            reuse_benefit = (reuse - 1) * recompute
            if reuse_benefit <= fixed:
                return CSEDecision(
                    semantic_hash=semantic_hash,
                    action=RECOMPUTE,
                    reuse_count=reuse,
                    recompute_cost=recompute,
                    materialized_bytes=mbytes,
                    reason=f"reuse_benefit={reuse_benefit:.2f}<=fixed={fixed:.2f}",
                )
            memory_rent = mbytes * self._memory_rent_per_byte
            if reuse_benefit > memory_rent + fixed:
                return CSEDecision(
                    semantic_hash=semantic_hash,
                    action=MATERIALIZE_MEMORY,
                    reuse_count=reuse,
                    recompute_cost=recompute,
                    materialized_bytes=mbytes,
                    reason=(
                        f"reuse_benefit={reuse_benefit:.2f}>memory_rent={memory_rent:.2f}"
                        f"+fixed={fixed:.2f}"
                    ),
                )
            spill_cost = mbytes * self._spill_cost_per_byte
            transfer_cost = mbytes * self._transfer_cost_per_byte
            if reuse_benefit > spill_cost + transfer_cost + fixed:
                return CSEDecision(
                    semantic_hash=semantic_hash,
                    action=MATERIALIZE_SPILL,
                    reuse_count=reuse,
                    recompute_cost=recompute,
                    materialized_bytes=mbytes,
                    reason=(
                        f"reuse_benefit={reuse_benefit:.2f}>spill={spill_cost:.2f}"
                        f"+transfer={transfer_cost:.2f}+fixed={fixed:.2f}"
                    ),
                )
            return CSEDecision(
                semantic_hash=semantic_hash,
                action=RECOMPUTE,
                reuse_count=reuse,
                recompute_cost=recompute,
                materialized_bytes=mbytes,
                reason=(
                    f"reuse_benefit={reuse_benefit:.2f}<=spill+transfer+fixed"
                    f"({spill_cost:.2f}+{transfer_cost:.2f}+{fixed:.2f})"
                ),
            )

    # -- 物化 / 复用 --

    def materialize(
        self,
        semantic_hash: str,
        value: Any,
        *,
        action: str,
        spill_store: Any = None,
    ) -> None:
        """物化一个共享子表达式（内存或 spill）。

        Args:
            semantic_hash: 子表达式语义哈希
            value: 物化值
            action: 决策（MATERIALIZE_MEMORY / MATERIALIZE_SPILL）
            spill_store: 可选 SpillStore（MATERIALIZE_SPILL 时用于落盘）
        """
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None:
                return
            if action == MATERIALIZE_SPILL and spill_store is not None:
                meta = spill_store.spill(
                    semantic_hash, value, entry.materialized_bytes
                )
                if meta is not None:
                    entry.spill_id = meta.spill_id
                    entry.location = "spill"
                    entry.materialized = True
                    entry.action = MATERIALIZE_SPILL
                    self._telemetry.spill_count += 1
                    self._telemetry.materialize_spill_count += 1
                    return
                # spill 失败 → 回退内存物化（诚实，不静默丢弃）。
            entry.value = value
            entry.location = "memory"
            entry.materialized = True
            entry.action = MATERIALIZE_MEMORY
            self._telemetry.materialize_memory_count += 1
            self._telemetry.peak_cache_bytes = max(
                self._telemetry.peak_cache_bytes, entry.materialized_bytes
            )

    def get_materialized(self, semantic_hash: str) -> Any | None:
        """读取已物化值（内存态直接返回；spill 态由调用方 restore）。"""
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None or not entry.materialized:
                return None
            if entry.location == "memory":
                return entry.value
            return None  # spill 态：调用方经 spill_store.restore(spill_id)

    def spill_id_for(self, semantic_hash: str) -> str | None:
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is None or entry.location != "spill":
                return None
            return entry.spill_id

    def is_materialized(self, semantic_hash: str) -> bool:
        with self._lock:
            entry = self._entries.get(semantic_hash)
            return entry is not None and entry.materialized

    def refcount(self, semantic_hash: str) -> int:
        with self._lock:
            entry = self._entries.get(semantic_hash)
            return entry.refcount if entry is not None else 0

    # -- 遥测 --

    def record_recompute(self, semantic_hash: str) -> None:
        with self._lock:
            self._telemetry.recompute_count += 1

    def record_reuse_saved(self, semantic_hash: str) -> None:
        with self._lock:
            entry = self._entries.get(semantic_hash)
            if entry is not None:
                self._telemetry.reuse_saved_ms += (
                    entry.recompute_cost * (entry.reuse_count - 1)
                )

    def record_eviction(self) -> None:
        with self._lock:
            self._telemetry.eviction_count += 1

    def telemetry(self) -> CSELivenessTelemetry:
        with self._lock:
            return CSELivenessTelemetry(
                peak_cache_bytes=self._telemetry.peak_cache_bytes,
                eviction_count=self._telemetry.eviction_count,
                spill_count=self._telemetry.spill_count,
                recompute_count=self._telemetry.recompute_count,
                reuse_saved_ms=self._telemetry.reuse_saved_ms,
                materialize_memory_count=self._telemetry.materialize_memory_count,
                materialize_spill_count=self._telemetry.materialize_spill_count,
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "subexpression_count": len(self._entries),
                "materialized": sum(1 for e in self._entries.values() if e.materialized),
                "telemetry": self.telemetry().to_dict(),
            }
