"""data_access.r30.resolution_lease —— R40 #57：两相租约（resolution → execution）。

problem
-------
现有 :class:`data_access.r30.execution_lease.ExecutionLease` 是**单相**租约——
昂贵的 source discovery（LIST / HEAD / snapshot / manifest）在 governor
admission 之前就发生了。批处理对同一数据集反复解析时，discovery 没有独立的
并发/资源治理，可能在没有执行预算的情况下疯狂发元数据请求。

design
------
:class:`ResolutionLease` 把「发现相」与「执行相」分离为两相状态机：

    RESOLUTION_PENDING → RESOLUTION_ACTIVE → EXECUTION_ACTIVE → RELEASED

- :meth:`acquire_resolution` 门禁 LIST/HEAD/snapshot/manifest 发现相（可消费
  GlobalResourceGovernor 的 remote-discovery slot，见 R40 #58）；
- :meth:`transition_to_execution` 在发现成功后取得 :class:`ExecutionLease`
  进入执行相（同一共享 resource envelope）；
- 任何时间点都可以 :meth:`release`（幂等）。

**明确**：DataAccess 不建立第二个 auto-sharder——ResolutionLease 只**消费**
共享 envelope（FE HostResourceCoordinator 提供，防御性 import；不可用则保守
默认），与 ExecutionLease 完全一致。
"""
from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

from data_access.r30.execution_lease import ExecutionLease

__all__ = [
    "ResolutionPhase",
    "ResolutionLease",
    "two_phase_resolution_then_execution",
]


class ResolutionPhase(str, enum.Enum):
    """两相租约的状态机相位。"""

    RESOLUTION_PENDING = "resolution_pending"
    RESOLUTION_ACTIVE = "resolution_active"
    EXECUTION_ACTIVE = "execution_active"
    RELEASED = "released"


@dataclass
class ResolutionLease:
    """两相租约：发现相（LIST/HEAD/snapshot/manifest）→ 执行相。"""

    lease_id: str
    #: 发现相并发上限（同时允许的 discovery 调用数）。
    max_resolution_slots: int = 4
    #: 可选的 GlobalResourceGovernor（消费其 remote-discovery slot，R40 #58）。
    governor: Any = None
    absolute_deadline: float | None = None

    phase: ResolutionPhase = field(default=ResolutionPhase.RESOLUTION_PENDING, init=False)
    _resolution_inflight: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _execution_lease: ExecutionLease | None = field(default=None, init=False, repr=False)

    # ---- 发现相（resolution / discovery） ----

    def acquire_resolution(self, slots: int = 1) -> bool:
        """门禁一次 source discovery（LIST / HEAD / snapshot / manifest）。

        返回 True 表示取得发现 slot。同时消费 governor 的
        ``acquire_remote_discovery_slot``（若注入）。发现相未开始 / 已进入执行相
        / 已释放 → 拒绝。

        R32-P0-001：回滚时只释放**本次**取得的 slot，不释放进入本调用前已持有的。
        R32-P0-002：governor 调用在锁外，避免 lock-order inversion / callback deadlock。
        R32-P0-003：deadline 检查（expired → 拒绝）。
        """
        slots = max(1, int(slots))
        # R32-P0-003：deadline 已过 → 拒绝。
        if self.absolute_deadline is not None:
            import time
            if time.monotonic() >= self.absolute_deadline:
                return False
        # 锁内：检查状态 + 预留 slot。
        with self._lock:
            if self.phase is ResolutionPhase.RELEASED:
                return False
            if self.phase is ResolutionPhase.EXECUTION_ACTIVE:
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} 已进入执行相，不能再发起 discovery"
                )
            if self.phase is ResolutionPhase.RESOLUTION_PENDING:
                self.phase = ResolutionPhase.RESOLUTION_ACTIVE
            if self._resolution_inflight + slots > self.max_resolution_slots:
                return False
            # 预留本地 slot（锁内），governor 调用在锁外。
            old_inflight = self._resolution_inflight
        # R32-P0-002：锁外调用 governor（避免持锁调用外部资源系统）。
        acquired_now = 0
        if self.governor is not None:
            try:
                acquire_fn = getattr(self.governor, "acquire_remote_discovery_slot", None)
                if callable(acquire_fn):
                    for i in range(slots):
                        if not acquire_fn():
                            # R32-P0-001：部分成功回滚**本次**已取得的 governor slot。
                            release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                            if callable(release_fn):
                                for _ in range(acquired_now):
                                    release_fn()
                            return False
                        acquired_now += 1
            except Exception:
                # 异常回滚本次已取得的 governor slot。
                release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                if callable(release_fn):
                    for _ in range(acquired_now):
                        try:
                            release_fn()
                        except Exception:
                            pass
                return False
        # 锁内：原子 commit 本地 slot。
        with self._lock:
            if self.phase is ResolutionPhase.RELEASED:
                # 阶段被其他线程改变，回滚 governor slot。
                if self.governor is not None and acquired_now > 0:
                    release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                    if callable(release_fn):
                        for _ in range(acquired_now):
                            try:
                                release_fn()
                            except Exception:
                                pass
                return False
            self._resolution_inflight += slots
            return True

    def release_resolution(self, slots: int = 1) -> None:
        """归还 discovery slot（与 :meth:`acquire_resolution` 配对）。

        R32-P0-002：governor 调用在锁外。
        """
        slots = max(1, int(slots))
        # 锁内：减少本地计数。
        with self._lock:
            if self._resolution_inflight <= 0:
                return
            n = min(slots, self._resolution_inflight)
            self._resolution_inflight -= n
        # 锁外：释放 governor slot。
        if self.governor is not None and n > 0:
            try:
                release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                if callable(release_fn):
                    for _ in range(n):
                        release_fn()
            except Exception:
                pass

    def release_all_resolution(self) -> None:
        """归还全部剩余 discovery slot（发现相完成 → 执行相前的标准收尾）。

        R32-P0-002：governor 调用在锁外。
        """
        # 锁内：读取并清零本地计数。
        with self._lock:
            n = self._resolution_inflight
            self._resolution_inflight = 0
        # 锁外：释放 governor slot。
        if n > 0 and self.governor is not None:
            try:
                release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                if callable(release_fn):
                    for _ in range(n):
                        release_fn()
            except Exception:
                pass

    @property
    def resolution_active(self) -> bool:
        with self._lock:
            return self.phase is ResolutionPhase.RESOLUTION_ACTIVE

    # ---- 执行相 ----

    def transition_to_execution(
        self,
        *,
        envelope: Mapping[str, Any] | None = None,
        execution_lease: ExecutionLease | None = None,
    ) -> ExecutionLease:
        """发现成功后进入执行相：取得 :class:`ExecutionLease`。

        - 发现相必须有至少一次成功 discovery（RESOLUTION_ACTIVE 且 inflight==0）；
        - 返回 ExecutionLease（用同一 envelope 构造）；之后本租约即执行相。

        R32-P0-003：deadline 已过 → 禁止进入执行相。
        R32-P0-004：外部注入 execution_lease 必须验证。
        """
        # R32-P0-003：deadline 已过 → 禁止 transition。
        if self.absolute_deadline is not None:
            import time
            if time.monotonic() >= self.absolute_deadline:
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} deadline 已过，禁止进入执行相"
                )
        with self._lock:
            if self.phase is ResolutionPhase.RELEASED:
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} 已释放，无法进入执行相"
                )
            if self.phase is ResolutionPhase.EXECUTION_ACTIVE:
                if self._execution_lease is not None:
                    return self._execution_lease
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} 执行相缺少 execution lease"
                )
            if self.phase is ResolutionPhase.RESOLUTION_PENDING:
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} 尚未开始 discovery，不能直接进入"
                    "执行相（R40 #57：昂贵 discovery 必须发生在 execution admission 前）"
                )
            if self._resolution_inflight > 0:
                raise RuntimeError(
                    f"ResolutionLease {self.lease_id} 仍有 {self._resolution_inflight} "
                    "个 inflight discovery，不能进入执行相"
                )
            # R32-P0-004：外部注入 execution_lease 必须验证。
            if execution_lease is not None:
                if not getattr(execution_lease, "_acquired", False):
                    raise ValueError(
                        f"外部注入 ExecutionLease 未 acquired（R32-P0-004）"
                    )
                if getattr(execution_lease, "_released", False):
                    raise ValueError(
                        f"外部注入 ExecutionLease 已 released（R32-P0-004）"
                    )
                # 验证 deadline 一致性。
                lease_deadline = getattr(execution_lease, "absolute_deadline", None)
                if self.absolute_deadline is not None and lease_deadline != self.absolute_deadline:
                    raise ValueError(
                        f"外部注入 ExecutionLease deadline={lease_deadline} 与 "
                        f"ResolutionLease deadline={self.absolute_deadline} 不一致"
                    )
            else:
                execution_lease = ExecutionLease.from_resource_envelope(
                    envelope, lease_id=f"{self.lease_id}/exec"
                )
                execution_lease.absolute_deadline = self.absolute_deadline
                execution_lease.acquire(envelope)
            self._execution_lease = execution_lease
            self.phase = ResolutionPhase.EXECUTION_ACTIVE
            return execution_lease

    def execution_lease(self) -> ExecutionLease | None:
        with self._lock:
            return self._execution_lease

    # ---- 生命周期 ----

    def release(self) -> None:
        """释放全部（幂等）：归还 discovery slot + 释放 execution lease。

        R32-P0-002：governor 调用在锁外。
        """
        # 锁内：标记 RELEASED，读取状态。
        with self._lock:
            if self.phase is ResolutionPhase.RELEASED:
                return
            n = self._resolution_inflight
            self._resolution_inflight = 0
            exec_lease = self._execution_lease
            self._execution_lease = None
            self.phase = ResolutionPhase.RELEASED
        # 锁外：释放 governor slot + execution lease。
        if n > 0 and self.governor is not None:
            try:
                release_fn = getattr(self.governor, "release_remote_discovery_slot", None)
                if callable(release_fn):
                    for _ in range(n):
                        release_fn()
            except Exception:
                pass
        if exec_lease is not None:
            try:
                exec_lease.release()
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "lease_id": self.lease_id,
                "phase": self.phase.value,
                "resolution_inflight": self._resolution_inflight,
                "max_resolution_slots": self.max_resolution_slots,
                "execution_lease": (
                    self._execution_lease.to_dict() if self._execution_lease is not None else None
                ),
                "absolute_deadline": self.absolute_deadline,
            }


def two_phase_resolution_then_execution(
    *,
    lease_id: str = "r40-two-phase",
    max_resolution_slots: int = 4,
    governor: Any = None,
    envelope: Mapping[str, Any] | None = None,
    resolution_work: Any,
    execution_work: Any,
) -> Any:
    """两相租约的便利执行器：``resolution_work(lease)`` 成功后
    ``execution_work(exec_lease)``；任一失败/异常都释放租约。

    返回 execution_work 的返回值。resolution_work 抛异常 → 不进入执行相。
    """
    lease = ResolutionLease(
        lease_id=lease_id,
        max_resolution_slots=max_resolution_slots,
        governor=governor,
    )
    try:
        result = resolution_work(lease)
        # 发现相完成后归还全部 discovery slot，再进入执行相。
        lease.release_all_resolution()
        exec_lease = lease.transition_to_execution(envelope=envelope)
        return execution_work(exec_lease)
    finally:
        lease.release()
