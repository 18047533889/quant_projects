# -*- coding: utf-8 -*-
"""R36 §52/53/54/56/57/196 + R38 P0-017..022（§9）：HostResourceCoordinator 重构。

R36 版本把 FE/DA/service 统一到进程级 coordinator，但 lease 账本有缺陷（R38）：
    - **double count**（P0-017）：``sum(active leases.memory)`` 会把 parent 与 child
      都算进 host reserved（16GB job + 8GB child → 24GB）。
    - **CPU/IO/spill 只记录不约束**（P0-018）。
    - **job 内存 lease 是全局 scalar**（P0-019），并发 job 共享一个字段。
    - **release parent 只标记 direct children**（P0-020）。
    - **released lease 永久留在主 dict**（P1-021），长运行 service 无限增长。
    - **summary() 有 observability 副作用**（P0-022），顺手改 DA governor。

R38 修复 —— **root reservation accounting（§34 方案A，推荐）**：
    - 只有 root lease（``parent_lease_id is None``）计入 host reserved（memory /
      CPU / IO / spill 四维都约束）；
    - child 只在 parent 剩余额度内分配，**不再重复计 host**；
    - :class:`JobLease`：per-job 根租约对象，显式传递给 FE scheduler / DA /
      writer；``request_job_lease`` 创建 job 级 root lease；
    - ``release_lease`` 递归释放整棵子树 + 幂等 + 移到 terminal ring buffer；
    - ``summary()`` 纯读快照；``sync_da_limits()`` 显式动作。
"""
from __future__ import annotations

import contextvars
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from factor_engine.runtime.resource_broker import ResourceBroker

#: lease kind（§42 资源树：Job → Compute/DA Scan/Cache/Writer）。
KIND_JOB = "job"
KIND_COMPUTE = "compute"
KIND_NUMBA = "numba"
KIND_DA_SCAN = "da_scan"
KIND_CACHE = "cache"
KIND_WRITER = "writer"
KIND_REMOTE = "remote"

#: 终端 lease 保留数（P1-021：避免 _leases 无限增长）。
TERMINAL_RING_SIZE = 64

#: 当前 job 的活跃 JobLease（P0-018：service worker 在 job 执行期间设置；
#: FE scheduler / DA scan / writer 从当前 context 拿 job lease 申请 child）。
_ACTIVE_JOB_LEASE: contextvars.ContextVar = contextvars.ContextVar(
    "fe_active_job_lease", default=None
)


def get_active_job_lease() -> Any | None:
    """Read the current job binding without creating/switching a coordinator."""
    return _ACTIVE_JOB_LEASE.get()


@dataclass
class HostResourceLease:
    """一块主机资源的租约（§307 + R38 四维约束）。"""

    lease_id: str
    owner: str
    kind: str = KIND_JOB
    memory_bytes: int = 0
    cpu_tokens: int = 0
    io_tokens: int = 0
    spill_bytes: int = 0
    parent_lease_id: str | None = None
    released: bool = field(default=False)
    #: R38 P0-020：child lease 列表（递归释放用）。
    child_lease_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "owner": self.owner,
            "kind": self.kind,
            "memory_bytes": self.memory_bytes,
            "cpu_tokens": self.cpu_tokens,
            "io_tokens": self.io_tokens,
            "spill_bytes": self.spill_bytes,
            "parent_lease_id": self.parent_lease_id,
            "released": self.released,
        }


class JobLease:
    """R38 P0-019（§9）：per-job 根租约对象。

    显式持有 job 级 resource budget，并派生 child lease（compute / DA scan /
    cache / writer）。parent/child 都落在 coordinator 的 **root reservation
    accounting** 里：只有 job（root）计入 host，children 只在 job 内分配。
    """

    def __init__(
        self,
        coordinator: "HostResourceCoordinator",
        lease: HostResourceLease,
    ) -> None:
        self._coordinator = coordinator
        self._lease = lease

    @property
    def lease_id(self) -> str:
        return self._lease.lease_id

    @property
    def memory_bytes(self) -> int:
        return self._lease.memory_bytes

    @property
    def broker(self) -> ResourceBroker:
        return self._coordinator.broker

    @property
    def released(self) -> bool:
        return self._lease.released or self._coordinator.job_lease_closing(
            self._lease.lease_id
        )

    def request_child(
        self,
        *,
        owner: str,
        kind: str = KIND_COMPUTE,
        memory_bytes: int = 0,
        cpu_tokens: int = 0,
        io_tokens: int = 0,
        spill_bytes: int = 0,
    ) -> HostResourceLease | None:
        """在 job 内申请 child lease（不计 host，只占 job 剩余额度）。"""
        return self._coordinator.request_lease(
            owner=owner,
            kind=kind,
            memory_bytes=memory_bytes,
            cpu_tokens=cpu_tokens,
            io_tokens=io_tokens,
            spill_bytes=spill_bytes,
            parent_lease_id=self._lease.lease_id,
        )

    @property
    def remaining_memory_bytes(self) -> int:
        """Current uncommitted memory in this job's child-lease tree."""
        return self._coordinator.remaining_child_memory(self._lease.lease_id)

    def release_child(self, lease: HostResourceLease) -> bool:
        """Release one child lease without releasing the whole job tree."""
        if lease.parent_lease_id != self._lease.lease_id:
            return False
        return self._coordinator.release_lease(lease.lease_id)

    @contextmanager
    def bind_context(self):
        """Bind this concrete job while code runs in an executor thread."""
        token = _ACTIVE_JOB_LEASE.set(self)
        try:
            yield self
        finally:
            _ACTIVE_JOB_LEASE.reset(token)

    def release(self) -> None:
        """Close admission and release the root after live children drain."""
        self._coordinator.close_job_lease(self._lease.lease_id)

    def to_dict(self) -> dict[str, Any]:
        return self._lease.to_dict()


class HostResourceCoordinator:
    """FE/DA/service 共享的唯一主机资源权威（进程级单例，§58）。

    R38：root reservation accounting —— 只有 root lease 计入 host reserved；
    child 只在 parent 内分配，不重复计 host（§34 硬不变量）。
    """

    def __init__(self, *, broker: ResourceBroker | None = None) -> None:
        # P0-11 单权威 gate：production 下缺失 broker → raise MissingResourceBroker
        # （fail-closed，不再 ``broker or ResourceBroker()`` 静默新建第二套）。
        from factor_engine.runtime.resource_broker import require_broker

        self._broker = require_broker(broker, raise_on_missing=True)
        self._lock = threading.RLock()
        self._leases: dict[str, HostResourceLease] = {}
        self._rejected: list[str] = []
        # P1-021：released lease 移到 terminal ring（避免无限增长）。
        self._terminal: list[HostResourceLease] = []
        self._closing_job_roots: set[str] = set()
        # 显式 job 内存 lease 上限（旧 API 兼容：全局 scalar；R38 推荐 JobLease）。
        self._job_lease_bytes: int | None = None

    @property
    def broker(self) -> ResourceBroker:
        return self._broker

    def set_job_lease(self, memory_bytes: int | None) -> None:
        """旧 API 兼容（全局 scalar）。新代码请用 :meth:`request_job_lease`。"""
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

    def sync_da_limits(self) -> dict[str, Any]:
        """R38 P0-022：**显式动作**——把 Safe Envelope 应用到 DA governor。

        P0-018：同时注入 host child-lease 桥——DA 扫描的 admission 先向当前
        JobLease 请求 DAScanLease child（同一棵 lease 树），不再只走独立的
        GlobalResourceGovernor（双权威 → 资源账本分裂）。
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
            scan = max(0, int(safe * 0.5))
            scan_setter(scan)
            out["max_total_scan_bytes_inflight"] = scan
        bridge = getattr(gov, "set_host_lease_request", None)
        if callable(bridge):
            bridge(self.request_da_child_lease)
            out["host_lease_bridge"] = "installed"
        return out

    def set_active_job_lease(self, job_lease: Any | None) -> None:
        """P0-018：设置当前 context 的活跃 JobLease（service worker 在 job 执行
        期间设置；线程/协程间互不干扰，ContextVar request-scoped）。"""
        _ACTIVE_JOB_LEASE.set(job_lease)

    def active_job_lease(self) -> Any | None:
        return _ACTIVE_JOB_LEASE.get()

    def request_da_child_lease(
        self,
        memory_bytes: int,
        scan_bytes: int = 0,
    ) -> Any | None:
        """P0-018：DA 扫描从当前 JobLease 申请 DAScanLease child。

        无活跃 job（standalone DA / 非 service 路径）→ 返回 None（admit 回退
        本地 governor）。child 只在 job 剩余额度内分配，不重复计 host。返回
        :class:`_HostLeaseRef`（带 ``release()``，DA governor release 时一并释放）。
        """
        job = _ACTIVE_JOB_LEASE.get()
        requested = max(0, int(memory_bytes)) + max(0, int(scan_bytes))
        if requested <= 0:
            requested = 1
        lease_id = f"da-query-workspace:{getattr(job, 'lease_id', 'standalone')}:{uuid.uuid4().hex}"
        if job is None:
            # Spawned durable workers do not own a local JobLease tree. Charge
            # their DA query workspace directly to the installed parent broker
            # proxy instead of falling through to DA's process-local governor.
            # This lease is deliberately separate from READ_WAVE residency:
            # query/decode workspace and resident output may coexist.
            from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind

            lease = self._broker.acquire_memory(
                MemoryLeaseKind.SOURCE_READ,
                requested,
                lease_id=lease_id,
            )
            if lease is None:
                from data_access.core.exceptions import HostLeaseAdmissionDenied
                raise HostLeaseAdmissionDenied(
                    f"parent broker denied DA query workspace bytes={requested}"
                )
            return lease
        from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
        from factor_engine.runtime.job_scoped_lease import acquire_job_scoped_memory

        lease = acquire_job_scoped_memory(
            self._broker, job, MemoryLeaseKind.SOURCE_READ, requested,
            lease_id=lease_id,
        )
        if lease is None:
            from data_access.core.exceptions import HostLeaseAdmissionDenied
            raise HostLeaseAdmissionDenied(
                f"job lease denied DA query workspace bytes={requested}"
            )
        return lease

    def apply_da_envelope(self) -> dict[str, Any]:
        """R36 兼容别名：内部调用 :meth:`sync_da_limits`。"""
        return self.sync_da_limits()

    # -- child lease（R38：root reservation accounting，§34） --

    def request_job_lease(
        self,
        *,
        owner: str,
        memory_bytes: int,
        cpu_tokens: int = 1,
        io_tokens: int = 0,
        spill_bytes: int = 0,
    ) -> JobLease | None:
        """R38 P0-019：申请一个 job 级 root lease（唯一计入 host 的节点）。"""
        lease = self.request_lease(
            owner=owner,
            kind=KIND_JOB,
            memory_bytes=memory_bytes,
            cpu_tokens=cpu_tokens,
            io_tokens=io_tokens,
            spill_bytes=spill_bytes,
            parent_lease_id=None,
        )
        if lease is None:
            return None
        return JobLease(self, lease)

    def request_lease(
        self,
        *,
        owner: str,
        kind: str = KIND_JOB,
        memory_bytes: int = 0,
        cpu_tokens: int = 0,
        io_tokens: int = 0,
        spill_bytes: int = 0,
        parent_lease_id: str | None = None,
    ) -> HostResourceLease | None:
        """申请一块主机资源（R38：四维约束 + root accounting）。

        只有 root（``parent_lease_id is None``）计入 host reserved；child 只在
        parent 剩余额度内分配。host 不变量（§34）：

            root_reserved_memory <= safe_memory
            root_cpu_tokens <= target_cpu_tokens
            io_reserved <= io_capacity
            spill_reserved <= usable_spill
        """
        with self._lock:
            env = self.envelope()
            if parent_lease_id is not None:
                parent = self._leases.get(parent_lease_id)
                if (parent is None or parent.released
                        or parent_lease_id in self._closing_job_roots):
                    self._rejected.append(f"parent_released:{owner}:{parent_lease_id}")
                    return None
                ok, why = self._parent_has_room(
                    parent, memory_bytes, cpu_tokens, io_tokens, spill_bytes
                )
                if not ok:
                    self._rejected.append(f"child_{why}:{owner}")
                    return None
            else:
                # root accounting：只算 root lease（parent_lease_id is None）。
                used_mem = self._root_reserved("memory")
                if used_mem + memory_bytes > max(0, env.safe_memory_bytes):
                    self._rejected.append(
                        f"mem:{owner}:{memory_bytes} root_used:{used_mem} safe:{env.safe_memory_bytes}"
                    )
                    return None
                if not self._host_has_room(cpu_tokens, io_tokens, spill_bytes, owner):
                    return None
            lease = HostResourceLease(
                lease_id=uuid.uuid4().hex[:12],
                owner=owner,
                kind=kind,
                memory_bytes=max(0, int(memory_bytes)),
                cpu_tokens=max(0, int(cpu_tokens)),
                io_tokens=max(0, int(io_tokens)),
                spill_bytes=max(0, int(spill_bytes)),
                parent_lease_id=parent_lease_id,
            )
            self._leases[lease.lease_id] = lease
            if parent_lease_id is not None:
                parent = self._leases[parent_lease_id]
                parent.child_lease_ids.append(lease.lease_id)
            return lease

    def remaining_child_memory(self, parent_lease_id: str) -> int:
        """Return remaining memory under one live root lease, atomically."""
        with self._lock:
            parent = self._leases.get(parent_lease_id)
            if (parent is None or parent.released or parent.parent_lease_id is not None
                    or parent_lease_id in self._closing_job_roots):
                return 0
            child_mem = sum(
                child.memory_bytes
                for child_id in parent.child_lease_ids
                if (child := self._leases.get(child_id)) is not None
                and not child.released
            )
            return max(0, parent.memory_bytes - child_mem)

    def close_job_lease(self, lease_id: str) -> bool:
        """Reject new children now; release the root after existing children drain."""
        with self._lock:
            root = self._leases.get(lease_id)
            if root is None:
                return any(item.lease_id == lease_id for item in self._terminal)
            if root.parent_lease_id is not None:
                return False
            self._closing_job_roots.add(lease_id)
            if not any(
                (child := self._leases.get(child_id)) is not None
                and not child.released
                for child_id in root.child_lease_ids
            ):
                self._closing_job_roots.discard(lease_id)
                return self.release_lease(lease_id)
            return True

    def job_lease_closing(self, lease_id: str) -> bool:
        with self._lock:
            return lease_id in self._closing_job_roots

    def _parent_has_room(
        self,
        parent: HostResourceLease,
        memory_bytes: int,
        cpu_tokens: int,
        io_tokens: int,
        spill_bytes: int,
    ) -> tuple[bool, str]:
        """child 必须落在 parent 剩余额度内（不计 host）。"""
        child_ids = [
            c for c in parent.child_lease_ids if c in self._leases and not self._leases[c].released
        ]
        child_mem = sum(self._leases[c].memory_bytes for c in child_ids)
        child_cpu = sum(self._leases[c].cpu_tokens for c in child_ids)
        child_io = sum(self._leases[c].io_tokens for c in child_ids)
        child_spill = sum(self._leases[c].spill_bytes for c in child_ids)
        if memory_bytes > max(0, parent.memory_bytes - child_mem):
            return False, "mem"
        if cpu_tokens > max(0, parent.cpu_tokens - child_cpu):
            return False, "cpu"
        if io_tokens > max(0, parent.io_tokens - child_io):
            return False, "io"
        if spill_bytes > max(0, parent.spill_bytes - child_spill):
            return False, "spill"
        return True, ""

    def _host_has_room(
        self, cpu_tokens: int, io_tokens: int, spill_bytes: int, owner: str
    ) -> bool:
        """R38 P0-018：CPU / IO / spill 真实约束（root 维度）。"""
        cpu_used = self._root_reserved("cpu")
        io_used = self._root_reserved("io")
        spill_used = self._root_reserved("spill")
        target_cpu = max(1, self._broker.cpu_budget())
        io_cap = max(1, self._broker.hard_cpu_slots)
        usable_spill = self._broker._usable_spill() if hasattr(self._broker, "_usable_spill") else 0
        if cpu_used + cpu_tokens > target_cpu:
            self._rejected.append(
                f"cpu:{owner}:{cpu_tokens} used:{cpu_used} target:{target_cpu}"
            )
            return False
        if io_used + io_tokens > io_cap:
            self._rejected.append(f"io:{owner}:{io_tokens} used:{io_used} cap:{io_cap}")
            return False
        if spill_used + spill_bytes > max(0, usable_spill):
            self._rejected.append(
                f"spill:{owner}:{spill_bytes} used:{spill_used} usable:{usable_spill}"
            )
            return False
        return True

    def _root_reserved(self, dim: str) -> int:
        total = 0
        for l in self._leases.values():
            if l.released or l.parent_lease_id is not None:
                continue
            if dim == "memory":
                total += l.memory_bytes
            elif dim == "cpu":
                total += l.cpu_tokens
            elif dim == "io":
                total += l.io_tokens
            elif dim == "spill":
                total += l.spill_bytes
        return total

    def release_lease(self, lease_id: str) -> bool:
        """R38 P0-020：递归释放整棵 lease 树（幂等），并移到 terminal ring。"""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                # 已在 terminal ring（幂等）：仍返回 True（无害）。
                return any(l.lease_id == lease_id for l in self._terminal)
            parent_id = lease.parent_lease_id
            self._release_subtree(lease)
            # 顶层 root 从主 dict 移到 terminal ring（P1-021）。
            self._leases.pop(lease_id, None)
            self._terminal.append(lease)
            if len(self._terminal) > TERMINAL_RING_SIZE:
                self._terminal = self._terminal[-TERMINAL_RING_SIZE:]
            if parent_id is not None:
                parent = self._leases.get(parent_id)
                if parent is not None:
                    parent.child_lease_ids = [
                        child_id for child_id in parent.child_lease_ids
                        if child_id != lease_id
                    ]
                if parent_id in self._closing_job_roots and parent is not None \
                        and not parent.child_lease_ids:
                    self._closing_job_roots.discard(parent_id)
                    self.release_lease(parent_id)
            return True

    def _release_subtree(self, lease: HostResourceLease) -> None:
        """P0-019：递归标记 released，并**把 child 也从主 dict 移出**。

        旧实现只把 root 移出 ``_leases``，child 长期留在 dict 里积累 released
        对象（长时间 service 内存泄漏）。现在每个 child 都进 terminal ring。
        """
        lease.released = True
        for cid in list(lease.child_lease_ids):
            child = self._leases.get(cid)
            if child is not None:
                if not child.released:
                    self._release_subtree(child)
                # 子节点移入 terminal ring（不再留主 dict）。
                self._leases.pop(cid, None)
                self._terminal.append(child)
                if len(self._terminal) > TERMINAL_RING_SIZE:
                    self._terminal = self._terminal[-TERMINAL_RING_SIZE:]
        lease.child_lease_ids.clear()

    def reconcile(self) -> dict[str, Any]:
        """§34：root reserved 不能超过 host envelope（超了要告警）。"""
        with self._lock:
            env = self.envelope()
            used = self._root_reserved("memory")
            cpu = self._root_reserved("cpu")
            io = self._root_reserved("io")
            spill = self._root_reserved("spill")
            return {
                "active_root_leases": sum(
                    1
                    for l in self._leases.values()
                    if not l.released and l.parent_lease_id is None
                ),
                "active_leases": sum(1 for l in self._leases.values() if not l.released),
                "active_memory_bytes": used,
                "safe_memory_bytes": max(0, env.safe_memory_bytes),
                "over_commit": used > max(0, env.safe_memory_bytes),
                "cpu_reserved": cpu,
                "target_cpu_tokens": max(1, self._broker.cpu_budget()),
                "cpu_over_commit": cpu > max(1, self._broker.cpu_budget()),
                "io_reserved": io,
                "spill_reserved": spill,
                "terminal_count": len(self._terminal),
                "rejected_count": len(self._rejected),
            }

    def summary(self) -> dict[str, Any]:
        """R38 P0-022：**纯读**快照（不再顺手改 DA governor）。"""
        with self._lock:
            return {
                "envelope": self.envelope().to_dict(),
                "decision": self.decision().to_dict(),
                "leases": [l.to_dict() for l in self._leases.values()],
                "reconcile": self.reconcile(),
                "terminal_count": len(self._terminal),
                "rejected_recent": self._rejected[-10:],
            }


#: 进程级单例（§58：不需要 per-job 覆盖 global pointer；HostCoordinator 全局共享）。
_COORDINATOR: HostResourceCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


def get_host_coordinator(*, broker: Any | None = None) -> HostResourceCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _COORDINATOR_LOCK:
            if _COORDINATOR is None:
                from factor_engine.runtime.resource_broker import peek_v2_resource_broker

                authority = broker if broker is not None else peek_v2_resource_broker()
                _COORDINATOR = HostResourceCoordinator(broker=authority)
    if broker is not None and _COORDINATOR.broker is not broker:
        raise RuntimeError("host coordinator is already bound to another broker authority")
    return _COORDINATOR


def reset_host_coordinator() -> None:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        _COORDINATOR = None


class _HostLeaseRef:
    """host-backed DA 扫描租约引用（P0-018：可 release，DA governor 释放用）。"""

    __slots__ = ("_coordinator", "lease_id", "released")

    def __init__(self, coordinator: "HostResourceCoordinator", lease: HostResourceLease) -> None:
        self._coordinator = coordinator
        self.lease_id = lease.lease_id
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            try:
                self._coordinator.release_lease(self.lease_id)
            except Exception:
                pass
