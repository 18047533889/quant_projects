# -*- coding: utf-8 -*-
"""表示（representation）转换 KPI 追踪器 —— R39-PERF-031 Gate-11 可观测性。

``cleaned_bridge`` 在每一算子边界强制回 pandas（不是本文件/本 cluster 的职责），
本 tracker 在真正发生转换的消费点（``backend.panel_polars`` 的 fast/fallback 路径）
记录跨表示边界的数据量，暴露：

- ``transition_count``：按 (from, to) 表示对统计的转换次数；
- ``transition_bytes``：按 (from, to) 表示对累计的字节估算。

线程安全：简单 ``threading.Lock`` 保护计数器（统计目的，不追求精确并发语义）。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

TransitionKey = Tuple[str, str]


class RepresentationTransitionTracker:
    """跨表示转换的计数/字节追踪器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[TransitionKey, int] = {}
        self._bytes: Dict[TransitionKey, int] = {}

    def record_transition(self, from_rep: str, to_rep: str, bytes_: int) -> None:
        """记录一次 (from_rep → to_rep) 转换，累积字节估算。"""
        key = (from_rep, to_rep)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._bytes[key] = self._bytes.get(key, 0) + int(bytes_)

    def snapshot(self) -> Dict[str, Dict[TransitionKey, int]]:
        """返回 KPI dict：``transition_count`` / ``transition_bytes``。"""
        with self._lock:
            return {
                "transition_count": dict(self._counts),
                "transition_bytes": dict(self._bytes),
            }

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._bytes.clear()


# 进程级单例。各消费点（panel_polars fast/fallback 路径）直接使用模块级函数。
_GLOBAL = RepresentationTransitionTracker()


def get_rep_transition_tracker() -> RepresentationTransitionTracker:
    """返回进程级 tracker（测试可用它 reset/断言）。"""
    return _GLOBAL


def record_transition(from_rep: str, to_rep: str, bytes_: int) -> None:
    """记录一次表示转换（便捷函数，写入全局 tracker）。"""
    _GLOBAL.record_transition(from_rep, to_rep, bytes_)


def rep_transition_snapshot() -> Dict[str, Dict[TransitionKey, int]]:
    """返回全局 KPI 快照。"""
    return _GLOBAL.snapshot()


def reset_rep_transitions_for_tests() -> None:
    """仅测试：清空全局 tracker。"""
    _GLOBAL.reset()


# ---------------------------------------------------------------------------
# 便捷：字节估算辅助（避免在 panel_polars 内重复实现）
# ---------------------------------------------------------------------------
def estimate_panel_bytes(panel: Any) -> int:
    """估算 pandas 宽表转出数据的字节量（不含 deep object 内容，作为 KPI 估算）。"""
    mem = getattr(panel, "memory_usage", None)
    if mem is not None:
        try:
            return int(mem(deep=False).sum())
        except Exception:
            pass
    try:
        return int(sum(getattr(v, "nbytes", 0) for v in getattr(panel, "values", ())))
    except Exception:
        return 0
