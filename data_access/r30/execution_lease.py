"""data_access.r30.execution_lease —— R30-P1-014：ExecutionLease。

一次执行（job / 批次）持有的资源租约。**明确**：DataAccess 不建立第二个
auto-sharder——本租约只**消费**共享 resource envelope（由 FE
HostResourceCoordinator 的 ``resource_envelope()`` 提供，防御性 import；不可用
则保守默认），绝不自己决定全局资源上限。

资源维度（与 ResourceGovernor / QueryBudget 对齐）：
    - ``memory``          可保留内存（bytes）
    - ``scan_bytes``      inflight scan bytes（bytes）
    - ``remote_slots``    远端并发 slot 数
    - ``duckdb_slots``    DuckDB 并发 slot 数
    - ``temp_disk``       临时磁盘预算（bytes）
    - ``spill_budget``    spill 预算（bytes，如源块缓存溢出的落盘额度）
    - ``absolute_deadline`` 绝对 deadline（``time.monotonic()`` 纪元）

API：``acquire(envelope)`` → ``request_child(amounts)`` → ``release()``；
``remaining(now)`` 返回剩余时间（无 deadline → inf）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "DEFAULT_ENVELOPE",
    "ExecutionLease",
]

#: FE envelope 不可用时回退的保守默认值（R25 §26/27 语义）。
DEFAULT_ENVELOPE: dict[str, int] = {
    "memory": 1 * 1024 * 1024 * 1024,  # 1 GiB
    "scan_bytes": 512 * 1024 * 1024,  # 512 MiB
    "remote_slots": 16,
    "duckdb_slots": 8,
    "temp_disk": 1 * 1024 * 1024 * 1024,  # 1 GiB
    "spill_budget": 512 * 1024 * 1024,  # 512 MiB
}

#: 所有资源维度（request_child 校验用）。
_RESOURCE_KEYS = tuple(DEFAULT_ENVELOPE.keys())


def _env_int(env: Mapping[str, Any], key: str, default: int) -> int:
    value = env.get(key)
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _env_float(env: Mapping[str, Any], key: str, default: float | None) -> float | None:
    value = env.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _try_fetch_resource_envelope() -> dict[str, Any] | None:
    """从 FE HostResourceCoordinator 取共享 resource envelope（防御性）。

    FE 是 flat package（``runtime.*``）；DA/FE 同进程时懒加载。不可用（未部署 /
    无该方法）→ None（调用方回退保守默认）。绝不抛异常。
    """
    try:
        from factor_engine.runtime.host_resource_coordinator import get_host_coordinator

        coord = get_host_coordinator()
        fn = getattr(coord, "resource_envelope", None)
        if callable(fn):
            env = fn()
            if isinstance(env, Mapping):
                return dict(env)
    except Exception:
        pass
    return None


@dataclass
class ExecutionLease:
    """一次执行的资源租约（消费共享 envelope，不建立第二个 auto-sharder）。

    R32-P0-005：child release 必须归还父预算。
    R32-P0-006：全生命周期线程安全。
    """

    lease_id: str
    memory: int
    scan_bytes: int
    remote_slots: int
    duckdb_slots: int
    temp_disk: int
    spill_budget: int
    absolute_deadline: float | None = None

    _acquired: bool = field(default=False, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _children: list["ExecutionLease"] = field(default_factory=list, init=False, repr=False)
    _remaining: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _envelope: dict[str, Any] | None = field(default=None, init=False, repr=False)
    # R32-P0-006：内部锁保证线程安全。
    _lock: Any = field(default=None, init=False, repr=False)
    # R32-P0-005：parent ref 用于归还预算。
    _parent: "ExecutionLease | None" = field(default=None, init=False, repr=False)
    _parent_allocation: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """R32-P0-006：初始化内部锁和 parent refs。"""
        import threading
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_parent_allocation", {})

    def acquire(self, envelope: dict[str, Any] | None = None) -> "ExecutionLease":
        """按 envelope 占住本租约；envelope 缺省时本地无界（仍受调用方治理）。

        envelope 提供任一资源维度的上限时，请求量超过上限 → RuntimeError
        （fail-closed，不等执行完才发现超限）。返回 self（可链式）。

        R32-P0-006：线程安全。
        """
        with self._lock:
            if self._acquired or self._released:
                raise RuntimeError(
                    f"ExecutionLease {self.lease_id} 已占用或已释放，无法再次 acquire"
                )
            if envelope is not None:
                for key in _RESOURCE_KEYS:
                    limit = envelope.get(key)
                    if limit is not None and getattr(self, key) > limit:
                        raise RuntimeError(
                            f"ExecutionLease {self.lease_id} {key}={getattr(self, key)} "
                            f"超过 envelope 上限 {limit}（R30-P1-014：租约只消费共享 "
                            "envelope，不自我放大）。"
                        )
                self._envelope = dict(envelope)
            else:
                self._envelope = None
            self._remaining = {
                key: int(getattr(self, key)) for key in _RESOURCE_KEYS
            }
            self._acquired = True
            return self

    def release(self) -> None:
        """释放本租约及全部子租约；幂等。

        R32-P0-005：归还父预算（exactly-once）。
        R32-P0-006：线程安全。
        """
        with self._lock:
            if self._released:
                return
            self._released = True
            self._acquired = False
            children = list(self._children)
            self._children.clear()
            parent = self._parent
            parent_allocation = dict(self._parent_allocation)
            self._parent = None
            self._parent_allocation = {}
            self._remaining = {}
            self._envelope = None
        # 锁外：释放子租约。
        for child in children:
            try:
                child.release()
            except Exception:
                pass
        # R32-P0-005：归还父预算（exactly-once，锁外调用父的 _return_child_budget）。
        if parent is not None and parent_allocation:
            try:
                parent._return_child_budget(parent_allocation)
            except Exception:
                pass

    def remaining(self, now: float | None = None) -> float:
        """剩余时间（秒）。无 absolute_deadline → inf；已过 deadline → <= 0。"""
        if self.absolute_deadline is None:
            return float("inf")
        now = time.monotonic() if now is None else now
        return self.absolute_deadline - now

    def request_child(self, amounts: Mapping[str, int]) -> "ExecutionLease":
        """请求一个子租约；amounts 各维度不能超过父租约剩余额度。

        子租约继承父租约的 absolute_deadline。父租约记账：成功即扣减对应
        ``_remaining``；父租约 release 时递归释放全部子租约。

        R32-P0-005：子租约保存 parent ref + allocation token，release 时归还。
        R32-P0-006：线程安全。
        """
        with self._lock:
            if not self._acquired or self._released:
                raise RuntimeError(
                    f"父租约 {self.lease_id} 未占用（acquire 后），无法请求子租约"
                )
            clean: dict[str, int] = {}
            for key, value in dict(amounts).items():
                if key not in _RESOURCE_KEYS:
                    raise ValueError(f"未知资源维度 {key!r}（合法: {_RESOURCE_KEYS}）")
                value = max(0, int(value))
                if self._remaining[key] < value:
                    raise RuntimeError(
                        f"子租约 {key} 请求 {value} 超过父租约 {self.lease_id} "
                        f"剩余 {self._remaining[key]}"
                    )
                clean[key] = value
            child = ExecutionLease(
                lease_id=f"{self.lease_id}/child-{len(self._children) + 1}",
                memory=clean.get("memory", 0),
                scan_bytes=clean.get("scan_bytes", 0),
                remote_slots=clean.get("remote_slots", 0),
                duckdb_slots=clean.get("duckdb_slots", 0),
                temp_disk=clean.get("temp_disk", 0),
                spill_budget=clean.get("spill_budget", 0),
                absolute_deadline=self.absolute_deadline,
            )
            child._acquired = True
            child._remaining = dict(clean)
            # R32-P0-005：子租约保存父引用和分配记录。
            child._parent = self
            child._parent_allocation = dict(clean)
            for key, value in clean.items():
                self._remaining[key] -= value
            self._children.append(child)
            return child

    def _return_child_budget(self, allocation: dict[str, int]) -> None:
        """R32-P0-005：子租约 release 时归还预算（内部方法，由 child.release 调用）。"""
        with self._lock:
            if self._released:
                return
            for key, value in allocation.items():
                if key in self._remaining:
                    self._remaining[key] += value

    # ---- 状态 ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "memory": self.memory,
            "scan_bytes": self.scan_bytes,
            "remote_slots": self.remote_slots,
            "duckdb_slots": self.duckdb_slots,
            "temp_disk": self.temp_disk,
            "spill_budget": self.spill_budget,
            "absolute_deadline": self.absolute_deadline,
            "acquired": self._acquired,
            "released": self._released,
            "remaining": dict(self._remaining),
            "envelope": self._envelope,
            "children": [c.to_dict() for c in self._children],
        }

    @classmethod
    def from_resource_envelope(
        cls,
        envelope: dict[str, Any] | None = None,
        *,
        lease_id: str = "r30-lease",
    ) -> "ExecutionLease":
        """从 FE HostResourceCoordinator 的 resource_envelope() 构造（防御性）。

        envelope 为 None → 尝试懒加载 FE coordinator；仍不可用 → 保守默认值。
        维度 key 兼容 ``memory_bytes`` / ``scan_bytes`` / ``remote_slots`` /
        ``duckdb_slots`` / ``temp_disk_bytes`` / ``spill_budget_bytes`` /
        ``absolute_deadline``。
        """
        if envelope is None:
            envelope = _try_fetch_resource_envelope()
        env = dict(envelope) if isinstance(envelope, Mapping) else {}
        return cls(
            lease_id=lease_id,
            memory=_env_int(env, "memory_bytes", DEFAULT_ENVELOPE["memory"]),
            scan_bytes=_env_int(env, "scan_bytes", DEFAULT_ENVELOPE["scan_bytes"]),
            remote_slots=_env_int(env, "remote_slots", DEFAULT_ENVELOPE["remote_slots"]),
            duckdb_slots=_env_int(env, "duckdb_slots", DEFAULT_ENVELOPE["duckdb_slots"]),
            temp_disk=_env_int(env, "temp_disk_bytes", DEFAULT_ENVELOPE["temp_disk"]),
            spill_budget=_env_int(env, "spill_budget_bytes", DEFAULT_ENVELOPE["spill_budget"]),
            absolute_deadline=_env_float(env, "absolute_deadline", None),
        )
