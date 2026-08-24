# -*- coding: utf-8 -*-
"""MB-P1-014: Concurrent token management to avoid overload.

ResourceBroker CPU/IO token admission 的并发安全与过载保护：
    - Token 总和上限（target_cpu_tokens / target_io_tokens）
    - 在跑任务动态统计（scheduler 每次 admission 前刷新）
    - Token 抢占（高优先级任务可抢占低优先级 token）
    - Token lease 超时自动回收（任务卡死不释放 → 自动 reclaim）
    - Overload 检测（token 超发 → 主动降并发）

防止 CPU oversubscription：DuckDB threads=8 × 4 tasks = 32 threads 在 8 核机器
上 thrashing（context switch 开销 > 实际计算）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

# Token lease 默认超时（毫秒）：任务超过此时间未完成自动回收 token
DEFAULT_TOKEN_LEASE_TIMEOUT_MS = 300_000  # 5 minutes

# 默认 CPU/IO token 预算：由真实资源探测决定（不再硬编码 8/4）
_DEFAULT_CPU_TOKENS = 8
_DEFAULT_IO_TOKENS = 4


def _default_token_budgets() -> tuple[int, int, int]:
    """探测真实 (cpu_tokens, io_tokens, memory_bytes)。

    显式环境变量优先（``FACTOR_ENGINE_CPU_BUDGET`` / ``FACTOR_ENGINE_MAX_MEMORY_BYTES``）；
    否则复用 ``runtime.resource_governor`` 的真实资源探测（cgroup CPU quota /
    cpuset / affinity / cgroup v2/v1 / SLURM / RLIMIT / host RAM 取最严格 min）。
    探测失败回退 ``(8, 4, 32GiB)``——回退值绝不做 4 倍放大。
    """
    try:
        from factor_engine.runtime.resource_governor import effective_cpu_slots, effective_memory_limit_bytes

        cpu = effective_cpu_slots()
        io = max(1, cpu)
        mem = effective_memory_limit_bytes()
        return cpu, io, mem
    except Exception:  # pragma: no cover - 探测不可用时保守回退
        return _DEFAULT_CPU_TOKENS, _DEFAULT_IO_TOKENS, 32 * 1024**3


@dataclass
class TokenReservation:
    """单次 token 预留记录（可撤销、可超时）。

    ``generation`` 用于抢占后的代际检查：抢占方通过 :meth:`ConcurrentTokenManager
    .preempt_tokens` 递增 ``revoked_generation``，被抢占任务在 checkpoint 处比较
    自己拿到的 ``generation`` 与管理器当前代际，不一致即主动停止。
    """

    reservation_id: str
    cpu_tokens: int
    io_tokens: int
    memory_bytes: int
    acquired_at_ms: float
    timeout_ms: float
    priority: int = 0  # 优先级（高优先级可抢占低优先级）
    task_id: str = ""
    generation: int = 0  # 抢代际：0 = 未抢占

    def is_expired(self, now_ms: float) -> bool:
        """判断 lease 是否超时（自动回收）。"""
        return (now_ms - self.acquired_at_ms) >= self.timeout_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "cpu_tokens": self.cpu_tokens,
            "io_tokens": self.io_tokens,
            "memory_bytes": self.memory_bytes,
            "acquired_at_ms": round(self.acquired_at_ms, 2),
            "timeout_ms": round(self.timeout_ms, 2),
            "priority": self.priority,
            "task_id": self.task_id,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class RevocationToken:
    """资源抢占凭据：任务在 checkpoint 处校验是否已被抢占。

    - ``reservation_id``：被抢占的 reservation
    - ``generation``：被抢占时刻的代际（调用 ``preempt_tokens`` 后管理器代际会递增）
    - ``priority``：抢占方的优先级
    - ``reason``：抢占原因（诊断）
    """

    reservation_id: str
    generation: int
    priority: int
    reason: str = "preempted"


@dataclass
class TokenManagerMetrics:
    """Token 管理统计。"""

    total_reservations: int = 0
    active_reservations: int = 0
    total_releases: int = 0
    timeout_reclaims: int = 0
    preemptions: int = 0
    overload_rejections: int = 0
    preempted: int = 0  # 被抢占（且已撤销）的任务数
    revoked_generation: int = 0  # 当前抢占代际

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reservations": self.total_reservations,
            "active_reservations": self.active_reservations,
            "total_releases": self.total_releases,
            "timeout_reclaims": self.timeout_reclaims,
            "preemptions": self.preemptions,
            "overload_rejections": self.overload_rejections,
            "preempted": self.preempted,
            "revoked_generation": self.revoked_generation,
        }


class ConcurrentTokenManager:
    """并发 token 管理器（CPU/IO/Memory 统一治理）。

    集成点：
        - ResourceBroker.try_reserve() 内部调用 manager.acquire_tokens()
        - AdaptiveBatchScheduler 定期调用 manager.reclaim_expired()
        - ResourceController 根据 PSI/memory pressure 动态调整 token budget
    """

    def __init__(
        self,
        *,
        cpu_budget: int | None = None,
        io_budget: int | None = None,
        memory_budget_bytes: int | None = None,
        lease_timeout_ms: float = DEFAULT_TOKEN_LEASE_TIMEOUT_MS,
        overload_threshold: float = 1.1,
    ) -> None:
        """
        Args:
            cpu_budget: CPU token 总预算（逻辑核心数）。None 时探测真实 CPU 硬上限。
            io_budget: IO token 总预算（并发 IO 任务数）。None 时 = cpu_budget。
            memory_budget_bytes: 内存总预算（字节）。None 时探测真实内存上限
                （cgroup/host 最严格 min），**绝不做 ×4 放大**。
            lease_timeout_ms: Token lease 超时自动回收（毫秒）
            overload_threshold: 超发阈值（actual / budget > threshold 拒绝新请求）
        """
        cpu, io, mem = _default_token_budgets()
        if memory_budget_bytes is None:
            memory_budget_bytes = mem
        if cpu_budget is None:
            cpu_budget = cpu
        if io_budget is None:
            io_budget = io
        # R21：负数 / 显式 0 视为非法配置，显式 fail-fast。
        if not isinstance(cpu_budget, int) or isinstance(cpu_budget, bool) or cpu_budget <= 0:
            raise ValueError(f"cpu_budget 必须为正整数，得到 {cpu_budget!r}")
        if not isinstance(io_budget, int) or isinstance(io_budget, bool) or io_budget <= 0:
            raise ValueError(f"io_budget 必须为正整数，得到 {io_budget!r}")
        if (
            not isinstance(memory_budget_bytes, int)
            or isinstance(memory_budget_bytes, bool)
            or memory_budget_bytes <= 0
        ):
            raise ValueError(
                f"memory_budget_bytes 必须为正整数，得到 {memory_budget_bytes!r}"
            )

        self.cpu_budget = cpu_budget
        self.io_budget = io_budget
        self.memory_budget_bytes = memory_budget_bytes
        self.lease_timeout_ms = lease_timeout_ms
        self.overload_threshold = overload_threshold

        self._reservations: dict[str, TokenReservation] = {}
        self._metrics = TokenManagerMetrics()
        self._lock = threading.RLock()
        self._reservation_counter = 0
        self._revoked_generation = 0

    def acquire_tokens(
        self,
        cpu_tokens: int,
        io_tokens: int,
        memory_bytes: int,
        *,
        task_id: str = "",
        priority: int = 0,
        timeout_ms: float | None = None,
    ) -> str | None:
        """尝试获取 token（成功返回 reservation_id，失败返回 None）。

        Args:
            cpu_tokens: 需要的 CPU token 数
            io_tokens: 需要的 IO token 数
            memory_bytes: 需要的内存字节数
            task_id: 任务标识（诊断用）
            priority: 优先级（高优先级可抢占低优先级）
            timeout_ms: Token lease 超时（None 使用默认）

        Returns:
            reservation_id（成功）或 None（被拒绝）
        """
        with self._lock:
            # 1) 先回收过期 lease
            self._reclaim_expired_internal(time.monotonic() * 1000.0)

            # R21：负数 token 显式拒绝（不静默 clamp 成 0/跳过）。
            for name, val in (
                ("cpu_tokens", cpu_tokens),
                ("io_tokens", io_tokens),
                ("memory_bytes", memory_bytes),
            ):
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    raise ValueError(
                        f"{name} 必须为非负整数，得到 {val!r}（禁止负 token）"
                    )

            # 2) 计算当前占用
            used_cpu = sum(r.cpu_tokens for r in self._reservations.values())
            used_io = sum(r.io_tokens for r in self._reservations.values())
            used_mem = sum(r.memory_bytes for r in self._reservations.values())

            # 3) 过载检测（超发 > threshold 拒绝）
            if self._is_overloaded(used_cpu, used_io, used_mem):
                self._metrics.overload_rejections += 1
                _logger.debug(
                    "token acquisition rejected: overload (cpu=%d/%d, io=%d/%d, mem=%d/%d)",
                    used_cpu, self.cpu_budget,
                    used_io, self.io_budget,
                    used_mem, self.memory_budget_bytes,
                )
                return None

            # 4) 判断是否可准入
            if (
                used_cpu + cpu_tokens > self.cpu_budget
                or used_io + io_tokens > self.io_budget
                or used_mem + memory_bytes > self.memory_budget_bytes
            ):
                # 尝试抢占低优先级 token
                if priority > 0:
                    freed = self._try_preempt(cpu_tokens, io_tokens, memory_bytes, priority)
                    if not freed:
                        return None
                else:
                    return None

            # 5) 分配 reservation
            self._reservation_counter += 1
            rid = f"token_res_{self._reservation_counter}"
            now_ms = time.monotonic() * 1000.0
            timeout = timeout_ms if timeout_ms is not None else self.lease_timeout_ms

            reservation = TokenReservation(
                reservation_id=rid,
                cpu_tokens=cpu_tokens,
                io_tokens=io_tokens,
                memory_bytes=memory_bytes,
                acquired_at_ms=now_ms,
                timeout_ms=timeout,
                priority=priority,
                task_id=task_id,
                generation=self._revoked_generation,
            )
            self._reservations[rid] = reservation
            self._metrics.total_reservations += 1
            self._metrics.active_reservations = len(self._reservations)

            _logger.debug(
                "token acquired: rid=%s, cpu=%d, io=%d, mem=%d, task=%s",
                rid, cpu_tokens, io_tokens, memory_bytes, task_id,
            )
            return rid

    def preempt_tokens(
        self,
        reservation_id: str,
        *,
        priority: int = 0,
        reason: str = "preempted",
    ) -> RevocationToken | None:
        """主动撤销一个低优先级 reservation（R21 资源抢占）。

        - 从 ``_reservations`` 移除该 reservation（释放其 CPU/IO/memory 占用），
          供更高优先级任务使用——这是真正的抢占，不再是「只删 dict 条目」。
        - 递增 ``revoked_generation``，并返回 :class:`RevocationToken`。
        - 被抢占任务在 checkpoint 处调用 :meth:`is_revoked` 比较代际，
          发现已被抢占即主动停止，不再继续执行。

        Returns:
            RevocationToken（成功）或 None（reservation 不存在 / 已释放）。
        """
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                _logger.debug("preempt_tokens: reservation %s not found", reservation_id)
                return None
            # 递增代际 + 把被抢占 reservation 标记为 revoked（供 checkpoint 检查）。
            self._revoked_generation += 1
            reservation.generation = -1
            self._metrics.preemptions += 1
            self._metrics.preempted += 1
            self._metrics.active_reservations = len(self._reservations)
            _logger.info(
                "token revoked: rid=%s, task=%s, gen=%d",
                reservation_id,
                reservation.task_id,
                self._revoked_generation,
            )
            return RevocationToken(
                reservation_id=reservation_id,
                generation=self._revoked_generation,
                priority=priority,
                reason=reason,
            )

    def is_revoked(self, reservation: TokenReservation | RevocationToken | None) -> bool:
        """checkpoint 代际检查：该 reservation 是否已被抢占（应停止执行）。

        - ``TokenReservation``：被抢占时其 ``generation`` 被置为 ``-1``（revoked
          哨兵），``generation < 0`` 即表示已被抢占。
        - ``RevocationToken``：只在抢占发生时签发，本身即代表「已撤销」→ True。
        - ``None``：视为已撤销（无 token = 无许可，fail-closed）。
        """
        if reservation is None:
            return True
        if isinstance(reservation, RevocationToken):
            return True
        return reservation.generation < 0

    def _try_preempt(
        self, need_cpu: int, need_io: int, need_mem: int, priority: int
    ) -> bool:
        """尝试抢占低优先级 token（高优先级任务优先执行）。

        Args:
            need_cpu: 需要的 CPU token
            need_io: 需要的 IO token
            need_mem: 需要的内存
            priority: 当前任务优先级

        Returns:
            True 表示抢占成功（已释放足够低优先级 token），False 表示无法抢占
        """
        # 找出所有低优先级 reservation
        candidates = [
            (rid, r)
            for rid, r in self._reservations.items()
            if r.priority < priority
        ]
        if not candidates:
            return False

        # 按优先级从低到高排序（优先抢占最低优先级）
        candidates.sort(key=lambda x: x[1].priority)

        freed_cpu = 0
        freed_io = 0
        freed_mem = 0
        to_remove: list[str] = []

        for rid, r in candidates:
            to_remove.append(rid)
            freed_cpu += r.cpu_tokens
            freed_io += r.io_tokens
            freed_mem += r.memory_bytes

            # 检查是否已释放足够
            if freed_cpu >= need_cpu and freed_io >= need_io and freed_mem >= need_mem:
                break

        # 最终检查（抢占后是否满足）
        used_cpu = sum(r.cpu_tokens for r in self._reservations.values())
        used_io = sum(r.io_tokens for r in self._reservations.values())
        used_mem = sum(r.memory_bytes for r in self._reservations.values())

        if (
            used_cpu - freed_cpu + need_cpu <= self.cpu_budget
            and used_io - freed_io + need_io <= self.io_budget
            and used_mem - freed_mem + need_mem <= self.memory_budget_bytes
        ):
            # 真正移除被抢占的 reservation（并递增代际，通知被抢占任务停止）
            for rid in to_remove:
                reservation = self._reservations.pop(rid, None)
                self._metrics.preemptions += 1
                self._metrics.preempted += 1
                if reservation is not None:
                    self._revoked_generation += 1
                    reservation.generation = -1
                    _logger.info(
                        "token preempted: rid=%s, task=%s, gen=%d",
                        rid,
                        reservation.task_id,
                        self._revoked_generation,
                    )
                else:
                    _logger.info("token preempted: rid=%s", rid)
            self._metrics.active_reservations = len(self._reservations)
            return True

        return False

    def release_tokens(self, reservation_id: str) -> None:
        """释放 token（任务完成时调用）。

        Args:
            reservation_id: acquire_tokens() 返回的 reservation_id
        """
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                _logger.debug("release_tokens: reservation %s not found", reservation_id)
                return

            self._metrics.total_releases += 1
            self._metrics.active_reservations = len(self._reservations)

            _logger.debug(
                "token released: rid=%s, cpu=%d, io=%d, mem=%d",
                reservation_id,
                reservation.cpu_tokens,
                reservation.io_tokens,
                reservation.memory_bytes,
            )

    def _is_overloaded(self, used_cpu: int, used_io: int, used_mem: int) -> bool:
        """判断当前是否过载（已占用 / 预算 > threshold）。"""
        if self.cpu_budget > 0:
            cpu_ratio = used_cpu / self.cpu_budget
            if cpu_ratio > self.overload_threshold:
                return True
        if self.io_budget > 0:
            io_ratio = used_io / self.io_budget
            if io_ratio > self.overload_threshold:
                return True
        if self.memory_budget_bytes > 0:
            mem_ratio = used_mem / self.memory_budget_bytes
            if mem_ratio > self.overload_threshold:
                return True
        return False

    def reclaim_expired(self) -> int:
        """回收过期 token lease（任务超时未完成）。

        Returns:
            回收的 reservation 数量
        """
        with self._lock:
            now_ms = time.monotonic() * 1000.0
            return self._reclaim_expired_internal(now_ms)

    def _reclaim_expired_internal(self, now_ms: float) -> int:
        """内部回收实现（已持锁）。"""
        expired: list[str] = []
        for rid, r in self._reservations.items():
            if r.is_expired(now_ms):
                expired.append(rid)

        for rid in expired:
            self._reservations.pop(rid, None)
            self._metrics.timeout_reclaims += 1
            _logger.warning("token lease expired: rid=%s", rid)

        if expired:
            self._metrics.active_reservations = len(self._reservations)

        return len(expired)

    def current_usage(self) -> dict[str, Any]:
        """返回当前 token 占用情况。"""
        with self._lock:
            used_cpu = sum(r.cpu_tokens for r in self._reservations.values())
            used_io = sum(r.io_tokens for r in self._reservations.values())
            used_mem = sum(r.memory_bytes for r in self._reservations.values())

            return {
                "cpu_used": used_cpu,
                "cpu_budget": self.cpu_budget,
                "cpu_utilization": round(used_cpu / max(1, self.cpu_budget), 3),
                "io_used": used_io,
                "io_budget": self.io_budget,
                "io_utilization": round(used_io / max(1, self.io_budget), 3),
                "memory_used": used_mem,
                "memory_budget": self.memory_budget_bytes,
                "memory_utilization": round(
                    used_mem / max(1, self.memory_budget_bytes), 3
                ),
                "active_reservations": len(self._reservations),
            }

    def adjust_budgets(
        self,
        *,
        cpu_budget: int | None = None,
        io_budget: int | None = None,
        memory_budget_bytes: int | None = None,
    ) -> None:
        """动态调整 token budget（ResourceController 反馈控制）。

        Args:
            cpu_budget: 新 CPU token 预算（None 保持不变）
            io_budget: 新 IO token 预算
            memory_budget_bytes: 新内存预算
        """
        with self._lock:
            if cpu_budget is not None:
                self.cpu_budget = max(1, cpu_budget)
            if io_budget is not None:
                self.io_budget = max(1, io_budget)
            if memory_budget_bytes is not None:
                self.memory_budget_bytes = max(1, memory_budget_bytes)

    def metrics(self) -> TokenManagerMetrics:
        """返回累计统计。"""
        with self._lock:
            return TokenManagerMetrics(
                total_reservations=self._metrics.total_reservations,
                active_reservations=self._metrics.active_reservations,
                total_releases=self._metrics.total_releases,
                timeout_reclaims=self._metrics.timeout_reclaims,
                preemptions=self._metrics.preemptions,
                overload_rejections=self._metrics.overload_rejections,
                preempted=self._metrics.preempted,
                revoked_generation=self._metrics.revoked_generation or self._revoked_generation,
            )

    def summary(self) -> dict[str, Any]:
        """返回完整状态摘要（诊断用）。"""
        with self._lock:
            usage = self.current_usage()
            metrics = self.metrics().to_dict()
            return {
                "usage": usage,
                "metrics": metrics,
                "lease_timeout_ms": self.lease_timeout_ms,
                "overload_threshold": self.overload_threshold,
                "revoked_generation": self._revoked_generation,
            }


# 全局单例
_global_token_manager: ConcurrentTokenManager | None = None
_global_lock = threading.Lock()


def get_global_token_manager() -> ConcurrentTokenManager:
    """返回全局 ConcurrentTokenManager（进程级单例）。

    首次调用时按真实资源探测预算（cgroup/host min，不做 ×4 放大）。
    """
    global _global_token_manager
    if _global_token_manager is None:
        with _global_lock:
            if _global_token_manager is None:
                cpu, io, mem = _default_token_budgets()
                _global_token_manager = ConcurrentTokenManager(
                    cpu_budget=cpu,
                    io_budget=io,
                    memory_budget_bytes=mem,
                )
    return _global_token_manager
