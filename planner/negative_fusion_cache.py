# -*- coding: utf-8 -*-
"""R39-P0-PERF-022: NegativeFusionCache —— 已知失败融合组合的负缓存。

fusion 失败不再整组退 per-root（binary-split 只拆失败子组）；但「已知失败的
组合」不应该在后续每次先付一次失败成本。负缓存键：

    NegativeFusionCacheKey(
        backend, backend_version, plan_family_hash, parameter_shape,
        source_shape,
    )

命中 → 跳过 multi-root 尝试，直接 per-root（对整组已知坏组合，不再付失败成本）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NegativeFusionCacheKey:
    """负缓存键（R39-P0-PERF-022）。

    - ``plan_family_hash``：计划结构族哈希（只含算子结构，不含参数值）；
    - ``parameter_shape``：参数化形状（window 等参数值，按根展开）；
    - ``source_shape``：source 形状（source_scope / snapshot / required_columns）。
    """

    backend: str
    backend_version: str
    plan_family_hash: str
    parameter_shape: tuple[str, ...] = ()
    source_shape: tuple[Any, ...] = ()

    def as_tuple(self) -> tuple:
        return (
            self.backend,
            self.backend_version,
            self.plan_family_hash,
            self.parameter_shape,
            self.source_shape,
        )


class NegativeFusionCache:
    """进程内负缓存（模块级 dict；可替换为 file-backed 持久化）。"""

    def __init__(self) -> None:
        self._blocked: dict[tuple, bool] = {}
        self._lock = threading.Lock()

    def mark_blocked(self, key: NegativeFusionCacheKey) -> None:
        with self._lock:
            self._blocked[key.as_tuple()] = True

    def is_blocked(self, key: NegativeFusionCacheKey) -> bool:
        with self._lock:
            return self._blocked.get(key.as_tuple(), False)

    def clear(self) -> None:
        with self._lock:
            self._blocked.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._blocked)


#: 模块级单例（进程内共享）。
NEGATIVE_FUSION_CACHE = NegativeFusionCache()
