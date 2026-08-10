# -*- coding: utf-8 -*-
"""R36 §52/53/54/56/57/196：HostResourceCoordinator —— FE/DA/service 唯一资源权威。

R36 P0-016（§50/51/52）：FE（ResourceBroker）与 DA（GlobalResourceGovernor）各自
有「全局资源治理」，彼此不知道完整占用 → 可能 double admission。修复：进程内只有
**一个** :class:`HostResourceCoordinator`，FE broker / DA governor / service job
admission 全部围绕它。

    - :meth:`apply_da_envelope`：把 Safe Envelope 派生 DA governor 的
      ``max_total_reserved_memory`` / ``max_total_scan_bytes_inflight``——DA 不再
      独立决定「我还有多少全局内存」（§54），而是请求 child lease（§56）。
    - :class:`HostResourceLease`（§307）：job/block 级租约（parent_lease 支持
      §53 child lease 树）；所有 child 之和不能超过 host envelope。
    - service job admission（§57/59/60）：先问 coordinator 能否 lease，不再只靠
      固定 ``max_running`` 数并发。

注意（§302）：不要再建立第三/第四套独立 resource truth——本模块围绕
``ResourceBroker``（真实 token admission 的执行者）提供协调层。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from runtime.resource_broker import ResourceBroker


@dataclass
class HostResourceLease:
    """§307：一块主机资源的租约。"""

    lease_id: str
    owner: str
    memory_bytes: int
    cpu_tokens: int
    io_tokens: int = 0
    spill_bytes: int = 0
    parent_lease_id: str | None = None
    released: bool = field(default=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner": self.owner,
            "memory_bytes": self.memory_bytes,
            "cpu_tokens": self.cpu_tokens,
            "io_tokens": self.io_tokens,
            "spill_bytes": self.spill_bytes,
            "parent_lease_id": self.parent_lease_id,
            "released": self.released,
        }


class HostResourceCoordinator:
    """FE/DA/service 共享的唯一主机资源权威（进程级单例，§58）。"""

    def __init__(self, *, broker: ResourceBroker | None = None) -> None:
        self._broker = broker or ResourceBroker()
        self._lock = threading.RLock()
        self._leases: dict[str, HostResourceLease] = {}
        self._rejected: list[str] = []
        # 显式 job 内存 lease 上限（dry admission / 外部调度器注入）。
        self._job_lease_bytes: int | None = None

    @property
    def broker(self) -> ResourceBroker:
        return self._broker

    def set_job_lease(self, memory_bytes: int | None) -> None:
        self._job_lease_bytes = memory_bytes if memory_bytes is not None and memory_bytes > 0 else None

    def envelope(self) -> Any:
        return self._broker.resource_envelope()

    def decision(self, *, sink_backpressure: float = 0.0) -> Any:
        return self._broker.resource_decision(
            job_memory_lease_bytes=self._job_lease_bytes,
            sink_backpressure=sink_backpressure,
        )

    # -- DA 统一（P0-016/017，§54/56） --

    def _da_governor(self) -> Any | None:
        try:
            from data_access.runtime.resource_governor import get_global_governor

            return get_global_governor()
        except Exception:
            return None

    def apply_da_envelope(self) -> dict[str, Any]:
        """把 Safe Envelope 应用到 DA governor（§54：DA 不独立决定全局内存）。

        返回应用结果（含被拒绝情况）；DA 不可用时返回 ``{"applied": False}``。
        """
        gov = self._da_governor()
        if gov is None:
            return {"applied": False, "reason": "da_governor_unavailable"}
        env = self.envelope()
        safe = max(0, env.safe_memory_bytes)
        out: dict[str, Any] = {"applied": True, "safe_memory_bytes": safe}
        setter = getattr(gov, "set_max_total_reserved_memory", None)
        if callable(setter):
            setter(safe)
            out["max_total_reserved_memory"] = safe
        scan_setter = getattr(gov, "set_max_total_scan_bytes_inflight", None)
        if callable(scan_setter):
            # scan inflight ≤ safe envelope 的 50%（scan 只是中间物化，不独占）。
            scan = max(0, int(safe * 0.5))
            scan_setter(scan)
            out["max_total_scan_bytes_inflight"] = scan
        return out

    # -- child lease（§53/196） --

    def request_lease(
        self,
        *,
        owner: str,
        memory_bytes: int,
        cpu_tokens: int = 1,
        io_tokens: int = 0,
        spill_bytes: int = 0,
        parent_lease_id: str | None = None,
    ) -> HostResourceLease | None:
        """§53/60：请求一块内存的 child lease；与当前全部 active lease 一起不能
        超过 Safe Envelope（sum(child) ≤ envelope）。
        """
        with self._lock:
            env = self.envelope()
            used = sum(l.memory_bytes for l in self._leases.values() if not l.released)
            if parent_lease_id is not None:
                parent = self._leases.get(parent_lease_id)
                if parent is None or parent.released:
                    return None
                # child 不能超过 parent 剩余额度。
                child_used = sum(
                    l.memory_bytes
                    for l in self._leases.values()
                    if not l.released and l.parent_lease_id == parent_lease_id
                )
                available = max(0, parent.memory_bytes - child_used)
                if memory_bytes > available:
                    self._rejected.append(
                        f"child_mem:{owner}:{memory_bytes}>parent_avail:{available}"
                    )
                    return None
            if used + memory_bytes > max(0, env.safe_memory_bytes):
                self._rejected.append(
                    f"mem:{owner}:{memory_bytes} used:{used} safe:{env.safe_memory_bytes}"
                )
                return None
            lease = HostResourceLease(
                lease_id=uuid.uuid4().hex[:12],
                owner=owner,
                memory_bytes=memory_bytes,
                cpu_tokens=cpu_tokens,
                io_tokens=io_tokens,
                spill_bytes=spill_bytes,
                parent_lease_id=parent_lease_id,
            )
            self._leases[lease.lease_id] = lease
            return lease

    def release_lease(self, lease_id: str) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            lease.released = True
            # 递归释放 child。
            for child in [l for l in self._leases.values() if l.parent_lease_id == lease_id]:
                child.released = True
            return True

    def reconcile(self) -> dict[str, Any]:
        """§53：sum(active leases) 不能超过 host envelope（超了要告警）。"""
        with self._lock:
            env = self.envelope()
            used = sum(l.memory_bytes for l in self._leases.values() if not l.released)
            return {
                "active_leases": sum(1 for l in self._leases.values() if not l.released),
                "active_memory_bytes": used,
                "safe_memory_bytes": max(0, env.safe_memory_bytes),
                "over_commit": used > max(0, env.safe_memory_bytes),
                "rejected_count": len(self._rejected),
            }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "envelope": self.envelope().to_dict(),
                "decision": self.decision().to_dict(),
                "leases": [l.to_dict() for l in self._leases.values()],
                "reconcile": self.reconcile(),
                "da_envelope": self.apply_da_envelope(),
                "rejected_recent": self._rejected[-10:],
            }


#: 进程级单例（§58：不需要 per-job 覆盖 global pointer；HostCoordinator 全局共享）。
_COORDINATOR: HostResourceCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


def get_host_coordinator() -> HostResourceCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _COORDINATOR_LOCK:
            if _COORDINATOR is None:
                _COORDINATOR = HostResourceCoordinator()
    return _COORDINATOR


def reset_host_coordinator() -> None:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        _COORDINATOR = None
