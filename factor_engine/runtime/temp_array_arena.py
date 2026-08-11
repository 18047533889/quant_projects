# -*- coding: utf-8 -*-
"""临时数值 buffer 的 arena/pool —— R39-PERF-032。

滚动 / 排名 / neutralize 等 NumPy/Numba 热路径会反复 ``np.empty`` / 释放同样的
shape。``TemporaryArrayArena`` 按 ``(shape_bucket, dtype)`` 缓存 scratch 数组：

- ``acquire(shape, dtype)``：命中池直接返回（复用），未命中 ``np.empty``；
- ``release(arr)``：**先投毒**（填哨兵值）再回池，防止别名悄悄污染最终输出 ——
  任何把已释放数组误当结果写出/再读的 bug 会以明显错误暴露而非静默损坏；
- ``with_scratch(...)``：上下文管理器，异常安全。

语义约束：scratch 不跨 semantic result 生命周期。池中的数组所有权归 arena，
调用方不得在 ``release`` 之后继续持有引用并期待内容保持。
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

ShapeKey = Tuple[int, ...]
PoolKey = Tuple[ShapeKey, str]


def _shape_key(shape: Any) -> ShapeKey:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


def _poison_value(dtype: Any) -> Any:
    """按 dtype 选择哨兵值：float→NaN、int→max、bool→True，其它→0。

    NaN/int-max 在统计/因子计算里几乎不可能作为合法中间值出现，用作投毒哨兵
    能让「已释放数组被误用」的情况立刻可见。
    """
    dt = np.dtype(dtype)
    if dt.kind in "fc":
        return np.nan
    if dt.kind in "iu":
        return np.iinfo(dt).max
    if dt.kind == "b":
        return True
    return 0


class TemporaryArrayArena:
    """按 (shape, dtype) 复用的临时 NumPy 数组池。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Dict[PoolKey, list] = {}
        self._reuse_count = 0
        self._alloc_count = 0

    # -- stats ---------------------------------------------------------------
    @property
    def scratch_alloc_count(self) -> int:
        """累计新建分配次数。"""
        with self._lock:
            return self._alloc_count

    @property
    def scratch_array_reuse_count(self) -> int:
        """累计复用次数（从池中命中返回）。"""
        with self._lock:
            return self._reuse_count

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "scratch_alloc_count": self._alloc_count,
                "scratch_array_reuse_count": self._reuse_count,
                "pooled_arrays": sum(len(v) for v in self._pool.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._pool.clear()

    # -- main API ------------------------------------------------------------
    def acquire(self, shape: Any, dtype: Any) -> np.ndarray:
        """取一个 shape/dtype 的 scratch 数组（优先复用）。"""
        key = (_shape_key(shape), str(np.dtype(dtype)))
        with self._lock:
            stack = self._pool.get(key)
            if stack:
                arr = stack.pop()
                self._reuse_count += 1
            else:
                self._alloc_count += 1
                arr = np.empty(shape, dtype=dtype)
        # 确保返回数组的形状精确匹配请求（池化时可能被 resize 过）。
        if arr.shape != _shape_key(shape):
            arr = np.empty(shape, dtype=dtype)
        return arr

    def release(self, arr: np.ndarray) -> None:
        """回收 scratch 数组。先投毒再入池，别名污染可被检测。"""
        arr = np.asarray(arr)
        try:
            arr[...] = _poison_value(arr.dtype)
        except (ValueError, TypeError):
            # 只读数组或无法写入：丢弃而不是把脏数组放回池。
            return
        # 只池化拥有自己数据的 C-contiguous 数组；视图（base 存在）会污染底层
        # buffer，复制出一份独立 buffer 再池化，避免同一内存被双份出租。
        if arr.base is not None or not arr.flags.c_contiguous:
            try:
                arr = np.array(arr, dtype=arr.dtype, order="C", copy=True)
            except Exception:
                return
        key = (_shape_key(arr.shape), str(arr.dtype))
        with self._lock:
            self._pool.setdefault(key, []).append(arr)


@contextlib.contextmanager
def with_scratch(arena: TemporaryArrayArena, shape: Any, dtype: Any) -> Iterator[np.ndarray]:
    """上下文管理器：进入 acquire、退出 release（异常也释放）。"""
    arr = arena.acquire(shape, dtype)
    try:
        yield arr
    finally:
        arena.release(arr)


# 进程级默认 arena：供尚未接入的 NumPy 热路径直接使用。
_GLOBAL_ARENA: Optional[TemporaryArrayArena] = None


def get_scratch_arena() -> TemporaryArrayArena:
    """返回进程级 arena（惰性创建；测试可 reset 或直接用新实例）。"""
    global _GLOBAL_ARENA
    if _GLOBAL_ARENA is None:
        _GLOBAL_ARENA = TemporaryArrayArena()
    return _GLOBAL_ARENA
