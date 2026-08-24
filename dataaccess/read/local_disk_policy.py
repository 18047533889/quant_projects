# -*- coding: utf-8 -*-
"""R44-P0: 零本地磁盘策略 —— 节点级增量 FactorEngine 的落盘治理。

STRICT_REMOTE（production）下，**任何**本地持久字节写入都是违规：增量
checkpoint / spill / 中间产物必须走内存、tmpfs 或远程对象存储。HIGH_PERFORMANCE
（research）允许加密临时 NVMe spill，但**永远不权威**（authoritative 数据只存
远端对象）。

本模块提供可测试的 fail-closed 机制：

- ``LocalDiskPolicy``：STRICT_REMOTE / HIGH_PERFORMANCE。
- ``track_local_persistent_bytes_written``：contextmanager，进入 STRICT_REMOTE
  模式后把每次「即将落盘」的字节数累进 contextvar 计数器。
- ``assert_no_local_persistent_write``：STRICT_REMOTE 下累计 > 0 → 抛
  ``LocalDiskPolicyViolation``（fail-closed）；HIGH_PERFORMANCE 放行。
- ``spill_policy_for``：按环境解析策略。

R44-P0 语义：
    1. RAM 缓冲与 tmpfs 不是本地磁盘；真实持久块设备才是。
    2. 策略判定是**调用方权威**（env），不读 ``query_budget`` 的 strict 开关——
       本模块自持一套 env 解析，避免跨包耦合。
    3. 计数器是 contextvar（非全局）：并发 run 之间不互相污染，也便于测试隔离。
"""
from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Generator, Mapping

__all__ = [
    "LocalDiskPolicy",
    "LocalDiskPolicyViolation",
    "track_local_persistent_bytes_written",
    "assert_no_local_persistent_write",
    "spill_policy_for",
]


class LocalDiskPolicy(str, Enum):
    """本地磁盘策略（R44-P0）。

    - ``STRICT_REMOTE``     ：本地持久字节 == 0；只允许内存 / 远程 spill。
    - ``HIGH_PERFORMANCE``  ：允许加密临时 NVMe spill，但永远不权威。
    """

    STRICT_REMOTE = "STRICT_REMOTE"
    HIGH_PERFORMANCE = "HIGH_PERFORMANCE"


class LocalDiskPolicyViolation(RuntimeError):
    """STRICT_REMOTE 下尝试本地持久写入（R44-P0 fail-closed）。"""


#: 当前策略下的累计「本地持久写入字节」contextvar 计数器。
#: ``None`` = 未进入 track 上下文（默认放行，research）。
_local_persistent_bytes: contextvars.ContextVar[int] = contextvars.ContextVar(
    "r44_local_persistent_bytes", default=0
)


@contextmanager
def track_local_persistent_bytes_written() -> Generator[Callable[[int], None], None, None]:
    """进入 STRICT_REMOTE 写入跟踪：yield 一个 tracker（每次调它 = 一次本地落盘）。

    用法::

        with track_local_persistent_bytes_written() as track:
            if spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE:
                track(len(blob))     # 模拟一次持久写入
            assert_no_local_persistent_write()   # STRICT_REMOTE → 抛违规

    tracker 只累加字节；实际落盘与否由调用方判定（策略是调用方权威）。
    """
    token = _local_persistent_bytes.set(0)

    def _track(bytes_written: int) -> None:
        cur = _local_persistent_bytes.get()
        _local_persistent_bytes.set(cur + max(0, int(bytes_written)))

    try:
        yield _track
    finally:
        _local_persistent_bytes.reset(token)


def assert_no_local_persistent_write() -> None:
    """STRICT_REMOTE 下累计本地持久写入 > 0 → 抛 ``LocalDiskPolicyViolation``。

    策略判定看 ``spill_policy_for()``；HIGH_PERFORMANCE / 未进入 track 上下文
    一律放行（不追踪即不强制）。
    """
    if spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE:
        total = _local_persistent_bytes.get()
        if total > 0:
            raise LocalDiskPolicyViolation(
                "STRICT_REMOTE 零本地磁盘违规：已累计写入 "
                f"{total} 字节本地持久存储（R44-P0 fail-closed）。"
                "内存 / tmpfs / 远程对象存储之外不允许落盘。"
            )


def spill_policy_for(env: Mapping[str, str] | None = None) -> LocalDiskPolicy:
    """解析本地磁盘 spill 策略（R44-P0）。

    优先级：
        1. ``FACTOR_ENGINE_LOCAL_DISK_POLICY=STRICT_REMOTE|HIGH_PERFORMANCE`` 显式指定
           （大小写不敏感；非法值抛 ValueError，fail-closed 不静默降级）。
        2. ``FACTOR_ENGINE_PRODUCTION=1``（含 true/yes）→ STRICT_REMOTE。
        3. 缺省 → HIGH_PERFORMANCE（research 默认；production 由上层注入 env）。

    不读进程全局 ``os.environ`` 之外的东西；``env=None`` 时读 ``os.environ``。
    """
    source: Mapping[str, str] = os.environ if env is None else env
    raw = str(source.get("FACTOR_ENGINE_LOCAL_DISK_POLICY", "")).strip().upper()
    if raw:
        if raw in {member.value for member in LocalDiskPolicy}:
            return LocalDiskPolicy(raw)
        raise ValueError(
            f"非法 FACTOR_ENGINE_LOCAL_DISK_POLICY={raw!r}；"
            f"允许: {sorted(m.value for m in LocalDiskPolicy)}（R44-P0 fail-closed）"
        )
    production_raw = str(source.get("FACTOR_ENGINE_PRODUCTION", "")).strip().lower()
    if production_raw in {"1", "true", "yes"}:
        return LocalDiskPolicy.STRICT_REMOTE
    return LocalDiskPolicy.HIGH_PERFORMANCE
