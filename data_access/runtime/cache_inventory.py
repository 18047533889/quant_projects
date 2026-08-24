"""R39 #74 —— 统一可回收缓存字节账本（CacheInventory）。

多个内存 owner 存在：FE GovernedBufferStore、FE PanelCache、DA QueryResultCache、
DuckDB buffer、Polars。ResourceAutopilot 需要看到**总可回收 cache 字节**来决定
cache 预算收缩是否值得。本模块是一个极小的注册表/聚合器：

    - :func:`register_cache_owner`：owner 注册 ``current_bytes()`` /
      ``shrink_to(target)`` / ``max_bytes()`` 回调；
    - :func:`total_reclaimable_bytes`：全部 owner 当前字节总和；
    - :func:`reclaim_to`：把 owner 逐出（LRU）到总字节 <= target；
    - DA QueryResultCache 在启用时注册自身（见 ``data_access.read.query_cache``）。

线程安全；owner 可以是进程内任何包（DA/FE 都懒加载本模块，FE 通过
``data_access.runtime.cache_inventory`` 访问）。
"""

from __future__ import annotations

import threading
from typing import Any, Callable

_lock = threading.Lock()
_owners: dict[str, dict[str, Any]] = {}


def register_cache_owner(
    name: str,
    *,
    current_bytes: Callable[[], int],
    shrink_to: Callable[[int], None] | None = None,
    max_bytes: Callable[[], int | None] | None = None,
) -> None:
    """注册一个可回收内存 owner（幂等覆盖同名注册）。"""
    with _lock:
        _owners[str(name)] = {
            "current_bytes": current_bytes,
            "shrink_to": shrink_to,
            "max_bytes": max_bytes,
        }


def unregister_cache_owner(name: str) -> None:
    with _lock:
        _owners.pop(str(name), None)


def registered_owners() -> list[str]:
    with _lock:
        return sorted(_owners)


def owner_bytes(name: str) -> int:
    with _lock:
        spec = _owners.get(str(name))
    if spec is None:
        return 0
    try:
        return max(0, int(spec["current_bytes"]()))
    except Exception:
        return 0


def total_reclaimable_bytes() -> int:
    """全部注册 owner 的当前缓存字节总和。"""
    with _lock:
        names = list(_owners)
    total = 0
    for name in names:
        total += owner_bytes(name)
    return max(0, total)


def reclaim_to(target_bytes: int) -> dict[str, int]:
    """把 owner 逐出到总字节 <= target_bytes，返回每 owner 回收字节。

    每个 owner 若提供 ``shrink_to`` 就按其当前字节 - 剩余配额调用；不提供
    shrink 的 owner 只参与读数，不参与回收。回收顺序按 owner 名排序（确定性）。
    """
    target = max(0, int(target_bytes))
    with _lock:
        names = sorted(_owners)
        specs = {n: _owners[n] for n in names}
    current = {n: owner_bytes(n) for n in names}
    total = sum(current.values())
    if total <= target:
        return {n: 0 for n in names}
    # 每 owner 剩余配额：按当前字节占比分配 target，减少到 target。
    reclaimed: dict[str, int] = {}
    if total > 0:
        for n in names:
            share = int(target * current[n] / total) if current[n] else 0
            spec = specs[n]
            shrink = spec.get("shrink_to")
            if callable(shrink) and current[n] > share:
                try:
                    shrink(max(0, share))
                except Exception:
                    pass
            reclaimed[n] = max(0, current[n] - owner_bytes(n))
    return reclaimed


def inventory_summary() -> dict[str, Any]:
    """全量快照（autopilot telemetry / 调试用）。"""
    with _lock:
        names = sorted(_owners)
    rows: list[dict[str, Any]] = []
    for n in names:
        rows.append(
            {
                "name": n,
                "bytes": owner_bytes(n),
                "max_bytes": _owner_max(n),
                "has_shrink": _owner_has_shrink(n),
            }
        )
    return {
        "owners": rows,
        "total_reclaimable_bytes": total_reclaimable_bytes(),
    }


def _owner_max(name: str) -> int | None:
    with _lock:
        spec = _owners.get(name)
    if spec is None or not callable(spec.get("max_bytes")):
        return None
    try:
        v = spec["max_bytes"]()
        return int(v) if v is not None else None
    except Exception:
        return None


def _owner_has_shrink(name: str) -> bool:
    with _lock:
        spec = _owners.get(name)
    return bool(spec is not None and callable(spec.get("shrink_to")))


__all__ = [
    "register_cache_owner",
    "unregister_cache_owner",
    "registered_owners",
    "owner_bytes",
    "total_reclaimable_bytes",
    "reclaim_to",
    "inventory_summary",
]
