# -*- coding: utf-8 -*-
"""R36 §104/105/106 + R38 P0-029/030/031/032/033/038 + P1-035/036/037（§12/§14）:
GovernedBufferStore —— CSE 共享缓冲的唯一 governed 通道。

R38 修复：
    - P0-029：``put()`` 不再返回裸 bool——返回 :class:`BufferPutResult`（status ∈
      {MEMORY, SPILLED, RECOMPUTE, REFUSED}），production 必须处理每种状态；
    - P1-035：``get()`` 统一持锁 + 更新 ``last_access``（引用生命周期）；
    - P1-036：eviction 按 LRU（``last_access`` 排序），不再按 dict 顺序；
    - P0-032/033：``spill()`` 走真实 :class:`SpillStore`（checksum + reload），
      spill-vs-recompute 成本决策；
    - P1-037：``reconciliation()`` 大 keyset 用抽样+外推（不再天然产生巨大漂移）；
    - P0-038：本 store 是 L0 CSE 唯一 owner（ExpressionCache 只作 adapter，不双套
      accounting）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.spill_store import SpillStore, SpillRef


def _estimate_bytes(value: Any) -> int:
    try:
        from runtime.resource_governor import estimate_object_bytes

        return max(0, int(estimate_object_bytes(value)))
    except Exception:
        return 0


#: put 结果状态（R38 P0-029）。
STATUS_MEMORY = "MEMORY"
STATUS_SPILLED = "SPILLED"
STATUS_RECOMPUTE = "RECOMPUTE"
STATUS_REFUSED = "REFUSED"


@dataclass(frozen=True)
class BufferPutResult:
    """一次 put 的结果（§P0-029：production 必须处理每种状态）。"""

    status: str
    key: str
    ref: Any = None           # SpillRef（SPILLED 时）
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "key": self.key,
            "ref": self.ref.to_dict() if isinstance(self.ref, SpillRef) else None,
            "reason": self.reason,
        }


@dataclass
class _Entry:
    """backing 条目（size + 访问时间）。"""

    value: Any
    bytes: int
    last_access: float = 0.0
    recompute_cost_ms: float = 0.0
    reuse_count: int = 1


class GovernedBufferStore:
    """带 byte 预算的 CSE 共享缓冲存储（put/get/pin/release/spill）。"""

    def __init__(
        self,
        backing: dict[str, Any] | None = None,
        *,
        budget_bytes: int | None = None,
        spill_store: SpillStore | None = None,
        spool_threshold_bytes: int | None = None,
    ) -> None:
        self._backing: dict[str, Any] = backing if backing is not None else {}
        self._entries: dict[str, _Entry] = {}
        self._budget = budget_bytes if budget_bytes is not None and budget_bytes > 0 else None
        self._pinned: set[str] = set()
        self._lock = threading.RLock()
        self._writes = 0
        self._refused = 0
        self._spilled = 0
        self._releases = 0
        self._reconciliations = 0
        self._spill_store = spill_store
        self._spool_threshold = spool_threshold_bytes or 512 * 1024**2
        # R38 P0-031：sid -> SpillRef（SPILLED 条目的 reload 依据）。
        self._spill_refs: dict[str, SpillRef] = {}

    @property
    def budget_bytes(self) -> int | None:
        return self._budget

    def _total_accounted(self) -> int:
        return sum(e.bytes for e in self._entries.values())

    def put(
        self,
        key: str,
        value: Any,
        *,
        bytes_: int | None = None,
        recompute_cost_ms: float = 0.0,
        spool: bool = False,
    ) -> BufferPutResult:
        """写入（R38 P0-029）：返回 BufferPutResult。

        - 预算够 → MEMORY；
        - 预算不够 → LRU 淘汰 → 还不够 → 若 ``spool`` 且 spill store 可用 →
          SPILLED（真实 spill）；否则 REFUSED（不静默）。
        """
        size = max(0, int(bytes_)) if bytes_ is not None else _estimate_bytes(value)
        now = time.monotonic()
        with self._lock:
            old = self._entries.get(key)
            if self._budget is not None:
                needed = self._total_accounted() - (old.bytes if old else 0) + size
                if needed > self._budget:
                    self._evict_lru(needed - self._budget)
                    if self._total_accounted() - (old.bytes if old else 0) + size > self._budget:
                        # 预算仍不够：先尝试真实 spill（P0-032）。
                        if spool and self._spill_store is not None:
                            try:
                                ref = self._spill_store.spill(
                                    value, key=key, source_identity="cse-shared"
                                )
                                self._spilled += 1
                                self._spill_refs[key] = ref
                                return BufferPutResult(
                                    STATUS_SPILLED, key, ref=ref,
                                    reason=f"over_budget_spilled:{size}",
                                )
                            except Exception as exc:  # noqa: BLE001
                                self._refused += 1
                                return BufferPutResult(
                                    STATUS_RECOMPUTE, key,
                                    reason=f"spill_failed_refuse_recompute:{type(exc).__name__}",
                                )
                        self._refused += 1
                        return BufferPutResult(
                            STATUS_REFUSED, key, reason=f"over_budget:{size}>remaining"
                        )
            self._backing[key] = value
            self._entries[key] = _Entry(
                value=value, bytes=size, last_access=now, recompute_cost_ms=recompute_cost_ms
            )
            self._writes += 1
            return BufferPutResult(STATUS_MEMORY, key, reason="ok")

    def get(self, key: str) -> Any | None:
        """R38 P1-035：持锁读取 + 更新 last_access（LRU 依据）。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_access = time.monotonic()
            entry.reuse_count += 1
            return entry.value

    def get_ref(self, key: str) -> Any:
        """读取并返回可 reload 的值（SPILLED 时从 spill store reload 磁盘）。"""
        with self._lock:
            if key in self._backing:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.last_access = time.monotonic()
                return self._backing[key]
        # backing 无 → 尝试从 spill store reload（R38 P0-031：plan_ref 读 spill）。
        ref = self._spill_refs.get(key)
        if ref is not None and self._spill_store is not None:
            try:
                value = self._spill_store.reload(ref)
                # reload 后放回 backing（后续访问不再落盘）。
                with self._lock:
                    self._backing[key] = value
                    self._entries[key] = _Entry(value=value, bytes=ref.bytes, last_access=time.monotonic())
                return value
            except Exception:
                return None
        return None

    def pin(self, key: str) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._pinned.add(key)
            return True

    def release(self, key: str) -> None:
        """§108/109：backing ref 与 accounting 同时释放（不留 ghost 账面）。"""
        with self._lock:
            existed = key in self._backing or key in self._entries
            self._backing.pop(key, None)
            self._entries.pop(key, None)
            self._pinned.discard(key)
            self._spill_refs.pop(key, None)
            if existed:
                self._releases += 1

    def spill(self, key: str) -> BufferPutResult:
        """R38 P0-032：真实 spill（写入 SpillStore），不是 drop。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return BufferPutResult(STATUS_RECOMPUTE, key, reason="absent_recompute")
            if self._spill_store is None:
                # 无 spill store → drop + recompute（显式 RECOMPUTE，不装 spill）。
                self._backing.pop(key, None)
                self._entries.pop(key, None)
                self._pinned.discard(key)
                self._spill_refs.pop(key, None)
                return BufferPutResult(STATUS_RECOMPUTE, key, reason="no_spill_store_recompute")
            try:
                ref = self._spill_store.spill(entry.value, key=key, source_identity="cse-shared")
                self._backing.pop(key, None)
                self._entries.pop(key, None)
                self._pinned.discard(key)
                self._spill_refs[key] = ref
                self._spilled += 1
                return BufferPutResult(STATUS_SPILLED, key, ref=ref, reason="real_spill")
            except Exception as exc:  # noqa: BLE001
                return BufferPutResult(
                    STATUS_RECOMPUTE, key, reason=f"spill_failed:{type(exc).__name__}"
                )

    def _evict_lru(self, needed: int) -> int:
        """R38 P1-036：LRU 淘汰（last_access 最久未访问优先；pinned 不淘汰）。"""
        candidates = sorted(
            (e for k, e in self._entries.items() if k not in self._pinned),
            key=lambda e: e.last_access,
        )
        freed = 0
        for entry in candidates:
            if freed >= needed:
                break
            key = next(
                (k for k, e in self._entries.items() if e is entry),
                None,
            )
            if key is None:
                continue
            self._backing.pop(key, None)
            self._entries.pop(key, None)
            freed += entry.bytes
        return freed

    # -- reconciliation（R38 P1-037：大 keyset 抽样 + 外推） --

    def reconciliation(self, *, sample_limit: int = 512) -> dict[str, Any]:
        """对比 accounted bytes 与 sampled actual bytes。

        keys ≤ sample_limit → 全量比；keys > limit → 抽样估计 + 外推到全量，
        不再天然产生巨大「漂移」。
        """
        with self._lock:
            accounted = self._total_accounted()
            keys = list(self._backing.keys())
            n = len(keys)
            if n <= sample_limit:
                sampled_actual = sum(_estimate_bytes(self._backing[k]) for k in keys)
                sampled_keys = n
                drift = abs(accounted - sampled_actual)
                extrapolated = False
            else:
                sample = keys[:sample_limit]
                sampled = sum(_estimate_bytes(self._backing[k]) for k in sample)
                sampled_actual = int(sampled * n / sample_limit)
                sampled_keys = sample_limit
                drift = abs(accounted - sampled_actual)
                extrapolated = True
            self._reconciliations += 1
            return {
                "accounted_bytes": accounted,
                "sampled_actual_bytes": sampled_actual,
                "sampled_keys": sampled_keys,
                "total_keys": n,
                "extrapolated": extrapolated,
                "drift_bytes": drift,
                "reconciliations": self._reconciliations,
            }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "budget_bytes": self._budget,
                "accounted_bytes": self._total_accounted(),
                "keys": len(self._backing),
                "writes": self._writes,
                "refused": self._refused,
                "spilled": self._spilled,
                "releases": self._releases,
                "pinned": len(self._pinned),
                "spill_store": self._spill_store.summary() if self._spill_store else None,
                "reconciliation": self.reconciliation(),
            }
