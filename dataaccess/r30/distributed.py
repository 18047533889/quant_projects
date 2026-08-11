"""data_access.r30.distributed —— R30-P2-003 Distributed providers 接口铺底。

为未来多机部署(Redis / Postgres / etcd 等)定义三类分布式 provider 的 Protocol,
并给出进程内本地默认实现:

  - ``DistributedLeaseProvider`` —— 租约(带 TTL,可 renew / 按 token release);
  - ``DistributedLockProvider`` —— 互斥锁(可带超时);
  - ``DistributedMetadataStore`` —— KV 元数据(带 prefix 枚举)。

当前**不引入复杂基础设施**,默认全部本地(threading,进程内);等未来多机再
把 ``get_default_providers`` 替换成远端实现。
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DistributedLeaseProvider",
    "DistributedLockProvider",
    "DistributedMetadataStore",
    "LocalLeaseProvider",
    "LocalLockProvider",
    "LocalMetadataStore",
    "get_default_providers",
]


@runtime_checkable
class DistributedLeaseProvider(Protocol):
    """租约 provider:acquire 带 TTL 的 lease,持有者可 renew / 显式 release。

    ``acquire`` 返回租约 token(持有者凭证)或 None(已被他人持有且未过期)。
    """

    def acquire(self, name: str, ttl: float) -> str | None: ...
    def renew(self, name: str, token: str, ttl: float) -> bool: ...
    def release(self, name: str, token: str) -> bool: ...


@runtime_checkable
class DistributedLockProvider(Protocol):
    """互斥锁 provider:``acquire(name, timeout)`` 非阻塞语义。

    ``timeout=0`` 立即返回:拿到返回 True,已被持有点返回 False。
    """

    def acquire(self, name: str, timeout: float = 0.0) -> bool: ...
    def release(self, name: str) -> bool: ...


@runtime_checkable
class DistributedMetadataStore(Protocol):
    """KV 元数据 store:get/put/delete + prefix list。"""

    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...


# ---------------------------------------------------------------------------
# 进程内本地实现(threading;进程外不共享)。
# ---------------------------------------------------------------------------


def _new_token() -> str:
    return uuid.uuid4().hex


class LocalLeaseProvider:
    """线程安全的进程内租约表:``name -> (token, expires_at)``。"""

    def __init__(self, clock: Any = None) -> None:
        self._lock = threading.Lock()
        self._leases: dict[str, tuple[str, float]] = {}
        self._clock = clock or time.monotonic

    def acquire(self, name: str, ttl: float) -> str | None:
        ttl = max(float(ttl), 0.0)
        now = self._clock()
        with self._lock:
            existing = self._leases.get(name)
            if existing is not None and existing[1] > now:
                return None  # 已被持有且未过期
            token = _new_token()
            self._leases[name] = (token, now + ttl)
            return token

    def renew(self, name: str, token: str, ttl: float) -> bool:
        ttl = max(float(ttl), 0.0)
        now = self._clock()
        with self._lock:
            rec = self._leases.get(name)
            if rec is None or rec[0] != token or rec[1] <= now:
                return False
            self._leases[name] = (token, now + ttl)
            return True

    def release(self, name: str, token: str) -> bool:
        with self._lock:
            rec = self._leases.get(name)
            if rec is None or rec[0] != token:
                return False
            del self._leases[name]
            return True


class LocalLockProvider:
    """线程安全的进程内命名锁(``threading.Lock`` 字典)。"""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, name: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._locks[name] = lock
            return lock

    def acquire(self, name: str, timeout: float = 0.0) -> bool:
        return self._lock_for(name).acquire(timeout=timeout)

    def release(self, name: str) -> bool:
        try:
            self._lock_for(name).release()
            return True
        except RuntimeError:
            # 未持有/未加锁就 release → 本地锁语义,返回 False 不抛。
            return False


class LocalMetadataStore:
    """线程安全的进程内 KV store(前缀 list 排序返回)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def list(self, prefix: str = "") -> list[str]:
        with self._lock:
            return sorted(k for k in self._data if k.startswith(prefix))


def get_default_providers() -> dict[str, Any]:
    """返回默认(本地)provider 组:``{"lease": ..., "lock": ..., "metadata": ...}``。

    等未来多机部署再替换成 Redis / Postgres / etcd 实现,当前不引入复杂
    基础设施。
    """
    return {
        "lease": LocalLeaseProvider(),
        "lock": LocalLockProvider(),
        "metadata": LocalMetadataStore(),
    }
