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

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

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


class BufferEntryState(str, enum.Enum):
    """R40 #11：条目生命周期状态机。

    - ``PINNED``       显式 pin（CSE 消费者在用），禁止淘汰；
    - ``RECLAIMABLE``  空闲、可被淘汰（默认）；
    - ``SPILLED``      backing 已落盘（SpillRef 保留，reload 依据）；
    - ``RECOMPUTABLE`` 已 drop（淘汰路径选择 recompute，不再 spill）；
    - ``EVICTED``      已从 entries 移除（终态）。
    """

    PINNED = "PINNED"
    RECLAIMABLE = "RECLAIMABLE"
    SPILLED = "SPILLED"
    RECOMPUTABLE = "RECOMPUTABLE"
    EVICTED = "EVICTED"


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
    """backing 条目（size + 访问时间 + R40 #11 状态机/refcount + #14 淘汰因子 + R42-016）。"""

    value: Any
    bytes: int
    last_access: float = 0.0
    recompute_cost_ms: float = 0.0
    reuse_count: int = 1
    # R40 #11：条目状态机 + 原子 refcount（在锁内读写）。
    state: str = BufferEntryState.RECLAIMABLE
    refcount: int = 0
    # R40 #14：多因子淘汰决策的输入维度。
    reload_cost_ms: float = 2.0
    write_cost_ms: float = 5.0
    future_consumers: int = 1
    # R42-016：reuse distance（下次使用距离，Belady-like eviction）。
    next_use_distance: int = 0  # 0=unknown, >0=已知消费序列距离


class SpillDecisionEngine:
    """R40 #14 + R42-016：多因子 eviction 打分（Belady-like reuse distance）。

    ``evict_candidate(entries, pressure)`` 对可淘汰条目（refcount==0 且非 PINNED）
    按 recompute_cost / reload_cost / write_cost / future_consumers / stale_ms /
    **next_use_distance** 综合打分，返回 ``[(key, score)]`` **升序**（分数越低 =
    淘汰成本越低 = 越先淘汰）。``pressure`` 是 disk_pressure ∈ [0,1]：磁盘越紧，
    写盘（spill）越贵，该维度用 ``write_cost * (1 + 3*pressure)`` 放大 → spill
    型条目排名靠后（更倾向 drop/recompute）。

    R42-016: next_use_distance > 0 时，distance 越大的条目越应先淘汰（Belady最优
    页面替换算法变种）；distance == 0 时 fallback 到 LRU（stale_ms）。
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "recompute_cost": 1.0,
            "reload_cost": 0.6,
            "write_cost": 0.4,
            "future_consumers": 0.2,
            "stale_ms": 0.01,
            "next_use_distance": 2.0,  # R42-016: Belady权重（优先级高于LRU）
        }

    def evict_candidate(
        self,
        entries: Iterable[tuple[str, _Entry]],
        *,
        pressure: float = 0.0,
        now: float | None = None,
    ) -> list[tuple[str, float]]:
        """返回 ``[(key, score)]`` 升序（最低分最先淘汰）。

        refcount>0 或 PINNED 的条目一律跳过（不可淘汰）——这是 #11 的核心：
        在用的共享 CSE 条目不能被字节预算误伤。

        R42-016: 已知 next_use_distance 的条目按 Belady 策略（distance 越大越先
        淘汰），未知的按 LRU（stale_ms）。
        """
        now = time.monotonic() if now is None else now
        disk = max(0.0, min(1.0, float(pressure)))
        w = self.weights
        scored: list[tuple[str, float]] = []
        for key, entry in entries:
            if entry.refcount > 0 or entry.state == BufferEntryState.PINNED:
                continue  # in-use：禁止淘汰
            stale_ms = max(0.0, now - entry.last_access)
            eff_write = max(0.0, entry.write_cost_ms) * (1.0 + 3.0 * disk)

            # R42-016: next_use_distance > 0 时优先使用 Belady；否则 fallback LRU
            if entry.next_use_distance > 0:
                # Belady: distance 越大，分数越低（越应先淘汰）
                # 用负数：-distance_weight * distance
                distance_score = -w.get("next_use_distance", 2.0) * entry.next_use_distance
            else:
                # Unknown distance: fallback to LRU（stale_ms 越大越先淘汰）
                distance_score = -w.get("stale_ms", 0.01) * stale_ms

            # 分数越低越先淘汰：
            #   + recompute/reload/write  → 淘汰成本越高排名越靠后；
            #   + distance_score          → R42-016: Belady distance 或 LRU fallback
            #   - future_consumers        → 未来消费越多越不该淘汰；
            score = (
                w.get("recompute_cost", 1.0) * max(0.0, entry.recompute_cost_ms)
                + w.get("reload_cost", 0.6) * max(0.0, entry.reload_cost_ms)
                + w.get("write_cost", 0.4) * eff_write
                + distance_score  # R42-016: 已经是负数（distance大→score低→先淘汰）
                - w.get("future_consumers", 0.2) * max(0, entry.future_consumers)
            )
            scored.append((key, score))
        scored.sort(key=lambda kv: (kv[1], kv[0]))
        return scored


class GovernedBufferStore:
    """带 byte 预算的 CSE 共享缓冲存储（put/get/pin/release/spill）。"""

    def __init__(
        self,
        backing: dict[str, Any] | None = None,
        *,
        budget_bytes: int | None = None,
        spill_store: SpillStore | None = None,
        spool_threshold_bytes: int | None = None,
        execution_id: str = "batch",
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
        # P0-013：execution 级 spill 生命周期（release/清理按 execution 收敛）。
        self._execution_id = str(execution_id or "batch")
        # R40 #14：多因子淘汰决策引擎（recompute/reload/write/future/stale）。
        self._decision_engine = SpillDecisionEngine()

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
        next_use_distance: int = 0,
        future_consumers: int = 1,
    ) -> BufferPutResult:
        """写入（R38 P0-029 + R42-016）：返回 BufferPutResult。

        - 预算够 → MEMORY；
        - 预算不够 → LRU 淘汰 → 还不够 → 若 ``spool`` 且 spill store 可用 →
          SPILLED（真实 spill）；否则 REFUSED（不静默）。

        R42-016 新增参数：
            next_use_distance: 距离下次使用的执行序数差（0=unknown, >0=已知）
            future_consumers: 未来剩余消费者数量（供 eviction 决策）
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
                                    value, key=key, source_identity="cse-shared",
                                    execution_id=self._execution_id,
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
                value=value,
                bytes=size,
                last_access=now,
                recompute_cost_ms=recompute_cost_ms,
                next_use_distance=next_use_distance,
                future_consumers=future_consumers,
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
        """读取并返回可 reload 的值（SPILLED 时从 spill store reload 磁盘）。

        P0-012：reload 后放回 backing 前**必须能进预算**——放不下先 LRU 逐出腾位，
        仍放不下就保持 disk-backed（本次返回不缓存），绝不无账本把大对象塞回内存
        （4GB spill 在只剩 1GB headroom 时不允许完整读回）。
        """
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
            except Exception:
                return None
            with self._lock:
                if self._budget is None:
                    self._backing[key] = value
                    self._entries[key] = _Entry(
                        value=value, bytes=ref.bytes, last_access=time.monotonic()
                    )
                    return value
                used = self._total_accounted()
                if used + ref.bytes > self._budget:
                    self._evict_lru(used + ref.bytes - self._budget)
                    used = self._total_accounted()
                if used + ref.bytes <= self._budget:
                    self._backing[key] = value
                    self._entries[key] = _Entry(
                        value=value, bytes=ref.bytes, last_access=time.monotonic()
                    )
            # 预算放不下 → 保持 disk-backed（不缓存），本次调用仍返回。
            return value
        return None

    def pin(self, key: str) -> bool:
        """R40 #11：pin 条目（状态 → PINNED，refcount 置 1）。

        pin 是**幂等**的——多个消费者对同一 sid 各 pin 一次，内部 refcount 只
        计一个 hold（多消费者计数由外部 ``ctx._cse_refcounts`` 负责，本 store
        不再重复记账）。
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            entry.state = BufferEntryState.PINNED
            entry.refcount = max(1, entry.refcount)
            self._pinned.add(key)
            return True

    def acquire_ref(self, key: str) -> bool:
        """R40 #11：原子 refcount +1（消费者在**持有值期间**显式占用）。

        占用中的条目（refcount>0）在 ``evict_if_over_budget`` 里被跳过——
        在用的共享 CSE 条目不会被字节预算误淘汰。
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            entry.refcount += 1
            return True

    def release_ref(self, key: str) -> None:
        """R40 #11：原子 refcount -1（与 :meth:`acquire_ref` 配对）。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.refcount > 0:
                entry.refcount -= 1
            if entry.refcount == 0 and entry.state == BufferEntryState.PINNED:
                entry.state = BufferEntryState.RECLAIMABLE
                self._pinned.discard(key)

    def release(self, key: str) -> None:
        """§108/109：backing ref 与 accounting 同时释放（不留 ghost 账面）。

        R40 #11：``release`` 幂等地归还一个 hold——refcount 归零才真正 drop
        （多消费者场景由外部 refcount 计数，最后一次 release 才落到本方法）。
        P0-013：drop 时删除 spill 文件（execution 级生命周期——release/失败/
        cancel 都收敛，不再积累 orphan parquet）。
        """
        with self._lock:
            existed = key in self._backing or key in self._entries
            entry = self._entries.get(key)
            if entry is not None and entry.refcount > 0:
                entry.refcount -= 1
                if entry.refcount > 0:
                    # 仍有 hold：仅归还一个引用，条目保留。
                    return
            self._pinned.discard(key)
            ref = self._spill_refs.pop(key, None)
            if ref is not None and self._spill_store is not None:
                try:
                    self._spill_store.delete(ref)
                except Exception:
                    pass
            self._backing.pop(key, None)
            self._entries.pop(key, None)
            self._pinned.discard(key)
            if existed:
                self._releases += 1

    def _disk_pressure(self) -> float:
        """R40 #14：当前磁盘压力（∈ [0,1]）。

        用 spill store 已落盘字节相对一个 safe ceiling 估算；无 spill store 或
        无法探测 → 0（无额外压力）。压力高时 SpillDecisionEngine 抬高写盘成本，
        淘汰更倾向 drop/recompute 而不是再写一份 spill。
        """
        if self._spill_store is None:
            return 0.0
        try:
            s = self._spill_store.summary()
            on_disk = max(0, int(s.get("bytes", 0)))
        except Exception:
            return 0.0
        ceiling = max(1, self._spool_threshold or 512 * 1024**2)
        return max(0.0, min(1.0, on_disk / ceiling))

    def evict_if_over_budget(self, target: int = 0, *, pressure: float | None = None) -> int:
        """MemoryGovernor L0 evict hook（R38 P0-038：L0 唯一 owner 的逐出入口）。

        governor 高压时从真实 store 逐出（pinned/refcount>0 不逐；R40 #11；
        R40 #14 多因子打分 + disk pressure），返回释放的**内存**字节数。旧实现
        把 hook 绑到 ExpressionCache（自己另记一套 accounting），对
        GovernedBufferStore 写入的值完全失效——split-brain。
        """
        with self._lock:
            if pressure is None:
                pressure = self._disk_pressure()
            return self._evict_lru(max(0, target), pressure=pressure)

    def cleanup_execution(self) -> int:
        """清理本 execution 的全部 spill 文件（P0-013：execution-scoped 收敛）。"""
        if self._spill_store is None:
            return 0
        with self._lock:
            ids = [self._execution_id]
        removed = 0
        for eid in ids:
            removed += self._spill_store.cleanup_execution(eid)
        return removed

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
                ref = self._spill_store.spill(
                    entry.value, key=key, source_identity="cse-shared",
                    execution_id=self._execution_id,
                )
                entry.state = BufferEntryState.SPILLED
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

    def _evict_lru(self, needed: int, *, pressure: float = 0.0) -> int:
        """R38 P1-036 + P0-033 + R40 #11/#14：多因子淘汰（pinned/in-use 不淘汰）。

        R40 #14：淘汰顺序由 :class:`SpillDecisionEngine` 按 recompute/reload/
        write_cost/future_consumers/stale 综合打分（不再只按 LRU 字节预算）。
        R40 #11：refcount>0 或 PINNED 的条目被引擎跳过——在用的共享 CSE 条目
        绝不会被字节预算误伤。

        每个候选仍做 **spill-vs-recompute 成本决策**（``SpillStore.should_spill``）：
            - 高 recompute 成本（PCA / GARCH / 大 source block）→ 真实 spill（写盘，
              保留 reload 依据），不 drop；
            - 低成本 elementwise（recompute 便宜）→ drop（并清掉旧 spill ref，
              防止 reload 复活过期数据）。
        返回释放的**内存**字节数（spill 与 drop 都释放内存）。
        """
        candidates = [
            (k, e) for k, e in self._entries.items()
            if e.refcount <= 0 and e.state != BufferEntryState.PINNED
        ]
        ranked = self._decision_engine.evict_candidate(candidates, pressure=pressure)
        freed = 0
        for key, _score in ranked:
            if freed >= needed:
                break
            entry = self._entries[key]
            # P0-033：成本决策（spill 优于 drop 当 recompute 显著比 reload 贵）。
            decision = None
            if self._spill_store is not None:
                try:
                    decision = self._spill_store.should_spill(
                        recompute_cost_ms=float(entry.recompute_cost_ms),
                        reload_cost_ms=float(entry.reload_cost_ms),
                        reuse_count=max(1, entry.reuse_count),
                    )
                except Exception:
                    decision = None
            if decision is not None and decision.should_spill:
                try:
                    ref = self._spill_store.spill(
                        entry.value, key=key, source_identity="cse-shared",
                        execution_id=self._execution_id,
                    )
                except Exception:
                    ref = None
                if ref is not None:
                    self._spill_refs[key] = ref
                    entry.state = BufferEntryState.SPILLED
                    self._spilled += 1
                    self._backing.pop(key, None)
                    self._entries.pop(key, None)
                    freed += entry.bytes
                    continue
            # drop（recompute 便宜 / spill 不可用）：清掉旧 spill ref 防止复活。
            old_ref = self._spill_refs.pop(key, None)
            if old_ref is not None and self._spill_store is not None:
                try:
                    self._spill_store.delete(old_ref)
                except Exception:
                    pass
            entry.state = BufferEntryState.RECOMPUTABLE
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
            states: dict[str, int] = {}
            in_use = 0
            for e in self._entries.values():
                states[str(e.state)] = states.get(str(e.state), 0) + 1
                if e.refcount > 0:
                    in_use += 1
            return {
                "budget_bytes": self._budget,
                "accounted_bytes": self._total_accounted(),
                "keys": len(self._backing),
                "writes": self._writes,
                "refused": self._refused,
                "spilled": self._spilled,
                "releases": self._releases,
                "pinned": len(self._pinned),
                "in_use_refcount": in_use,
                "states": states,
                "spill_store": self._spill_store.summary() if self._spill_store else None,
                "reconciliation": self.reconciliation(),
            }
