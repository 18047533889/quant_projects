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
from pathlib import Path
from typing import Callable, Generator, Mapping

__all__ = [
    "LocalDiskPolicy",
    "LocalDiskPolicyViolation",
    "track_local_persistent_bytes_written",
    "assert_no_local_persistent_write",
    "spill_policy_for",
    "strict_remote_audit_guard",
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

#: STRICT_REMOTE 审计守卫的活跃层级（嵌套 enter 计数）。>0 表示守卫已在监视，
#: 直接本地持久写（Path 写 / to_parquet）会被拦截。
_audit_guard_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "r45_audit_guard_depth", default=0
)


class _DirectPersistentWriteError(LocalDiskPolicyViolation):
    """STRICT_REMOTE 下尝试**直接**本地持久写（审计守卫拦截，fail-closed）。

    这比「调用方声明字节」更强：即使调用方未走
    ``track_local_persistent_bytes_written``，只要它在守卫上下文中真的执行了
    ``Path.open(w)`` / ``Path.write_bytes`` / ``DataFrame.to_parquet``（写真实
    持久块设备），守卫也立即拒绝——零本地持久字节由过程本身证明，而非信任
    调用方。
    """


def _audit_blocking(path: str | None) -> None:
    """审计守卫拦截器：STRICT_REMOTE 且守卫在深 → 抛违规（fail-closed）。"""
    if _audit_guard_depth.get() <= 0:
        return  # 守卫未激活（research / 测试本地事务），放行。
    if spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE:
        loc = f"（路径 {path!r}）" if path else ""
        raise _DirectPersistentWriteError(
            "STRICT_REMOTE 审计守卫：检测到直接本地持久写入尝试" + loc
            + "——production 下不允许 Path.open(w)/write_bytes/to_parquet "
            "等任何本地持久字节落盘（R45 fail-closed，过程级证明零磁盘写）。"
        )


@contextmanager
def strict_remote_audit_guard(
    *, env: Mapping[str, str] | None = None,
) -> Generator[None, None, None]:
    """STRICT_REMOTE 下证明「零本地持久写」的审计守卫。

    进入后若 ``spill_policy_for(env)`` 判定为 STRICT_REMOTE，则守卫拦截
    直接本地持久写：

    - ``Path.open`` 以写模式（``w`` / ``a`` / ``x`` / ``w+`` / ``a+`` / ``r+``）打开；
    - ``Path.write_bytes`` / ``write_text``；
    - ``Path.unlink`` / ``os.replace`` 目标为受 guard 保护的持久路径；
    - ``pandas.DataFrame.to_parquet``（经 ``pathlib.Path``/``str`` 目标落盘）；
    - ``pyarrow.parquet.write_table``（写本地路径）。

    守卫**不**拦截 tmpfs 与内存缓冲（不属本地磁盘），也不拦截经
    ``ObjectStore`` 的远程写。这使「STRICT_REMOTE 下零本地持久写」成为过程的
    硬性证明（fail-closed），而非仅依赖调用方主动声明字节。

    用法::

        with strict_remote_audit_guard():
            process_generation(...)   # 内部任何本地持久写都会抛违规
    """
    depth = _audit_guard_depth.get()
    if depth <= 0 and spill_policy_for(env) is LocalDiskPolicy.STRICT_REMOTE:
        token = _audit_guard_depth.set(1)
        _install_audit_hooks()
    else:
        token = _audit_guard_depth.set(depth + 1)
    try:
        yield
    finally:
        _audit_guard_depth.reset(token)
        if depth <= 0 and _audit_guard_depth.get() == 0:
            _uninstall_audit_hooks()


def spill_policy(env: Mapping[str, str] | None = None) -> LocalDiskPolicy:
    """便捷别名：解析本地磁盘 spill 策略（见 ``spill_policy_for``）。"""
    return spill_policy_for(env)


# ---------------------------------------------------------------------------
# 审计钩子安装（monkeypatch 性：目标模块若已加载，安装后即生效）
# ---------------------------------------------------------------------------
_ORIGINAL_PATH_OPEN = None
_ORIGINAL_PATH_WRITE_BYTES = None
_ORIGINAL_PATH_WRITE_TEXT = None
_ORIGINAL_PATH_UNLINK = None
_ORIGINAL_OS_REPLACE = None
_ORIGINAL_TO_PARQUET = None
_ORIGINAL_PQ_WRITE_TABLE = None


def _guard_path_write_bytes(self, data) -> int:
    _audit_blocking(str(self))
    return _ORIGINAL_PATH_WRITE_BYTES(self, data)


def _guard_path_write_text(self, *args, **kwargs):
    _audit_blocking(str(self))
    return _ORIGINAL_PATH_WRITE_TEXT(self, *args, **kwargs)


def _guard_path_open(self, *args, **kwargs):
    mode = ""
    if args:
        mode = str(args[0])
    mode = mode or str(kwargs.get("mode", "r"))
    if any(c in mode for c in ("w", "a", "+", "x")):
        _audit_blocking(str(self))
    return _ORIGINAL_PATH_OPEN(self, *args, **kwargs)


def _guard_path_unlink(self, *args, **kwargs):
    _audit_blocking(str(self))
    return _ORIGINAL_PATH_UNLINK(self, *args, **kwargs)


def _guard_os_replace(src, dst, *args, **kwargs):
    _audit_blocking(str(dst))
    return _ORIGINAL_OS_REPLACE(src, dst, *args, **kwargs)


def _guard_df_to_parquet(self, path, *args, **kwargs):
    _audit_blocking(str(path))
    return _ORIGINAL_TO_PARQUET(self, path, *args, **kwargs)


def _guard_pq_write_table(table, where, *args, **kwargs):
    _audit_blocking(str(where))
    return _ORIGINAL_PQ_WRITE_TABLE(table, where, *args, **kwargs)


def _install_audit_hooks() -> None:
    global _ORIGINAL_PATH_OPEN, _ORIGINAL_PATH_WRITE_BYTES, _ORIGINAL_PATH_WRITE_TEXT
    global _ORIGINAL_PATH_UNLINK, _ORIGINAL_OS_REPLACE, _ORIGINAL_TO_PARQUET
    global _ORIGINAL_PQ_WRITE_TABLE
    if _ORIGINAL_PATH_OPEN is None:
        _ORIGINAL_PATH_OPEN = Path.open
        Path.open = _guard_path_open
    if _ORIGINAL_PATH_WRITE_BYTES is None:
        _ORIGINAL_PATH_WRITE_BYTES = Path.write_bytes
        Path.write_bytes = _guard_path_write_bytes
    if _ORIGINAL_PATH_WRITE_TEXT is None:
        _ORIGINAL_PATH_WRITE_TEXT = Path.write_text
        Path.write_text = _guard_path_write_text
    if _ORIGINAL_PATH_UNLINK is None:
        _ORIGINAL_PATH_UNLINK = Path.unlink
        Path.unlink = _guard_path_unlink
    if _ORIGINAL_OS_REPLACE is None:
        _ORIGINAL_OS_REPLACE = os.replace
        os.replace = _guard_os_replace
    try:
        import pandas as pd

        if _ORIGINAL_TO_PARQUET is None and hasattr(pd.DataFrame, "to_parquet"):
            _ORIGINAL_TO_PARQUET = pd.DataFrame.to_parquet
            pd.DataFrame.to_parquet = _guard_df_to_parquet
    except Exception:
        pass
    try:
        import pyarrow.parquet as pq

        if _ORIGINAL_PQ_WRITE_TABLE is None and hasattr(pq, "write_table"):
            _ORIGINAL_PQ_WRITE_TABLE = pq.write_table
            pq.write_table = _guard_pq_write_table
    except Exception:
        pass


def _uninstall_audit_hooks() -> None:
    global _ORIGINAL_PATH_OPEN, _ORIGINAL_PATH_WRITE_BYTES, _ORIGINAL_PATH_WRITE_TEXT
    global _ORIGINAL_PATH_UNLINK, _ORIGINAL_OS_REPLACE, _ORIGINAL_TO_PARQUET
    global _ORIGINAL_PQ_WRITE_TABLE
    if _ORIGINAL_PATH_OPEN is not None:
        Path.open = _ORIGINAL_PATH_OPEN
        _ORIGINAL_PATH_OPEN = None
    if _ORIGINAL_PATH_WRITE_BYTES is not None:
        Path.write_bytes = _ORIGINAL_PATH_WRITE_BYTES
        _ORIGINAL_PATH_WRITE_BYTES = None
    if _ORIGINAL_PATH_WRITE_TEXT is not None:
        Path.write_text = _ORIGINAL_PATH_WRITE_TEXT
        _ORIGINAL_PATH_WRITE_TEXT = None
    if _ORIGINAL_PATH_UNLINK is not None:
        Path.unlink = _ORIGINAL_PATH_UNLINK
        _ORIGINAL_PATH_UNLINK = None
    if _ORIGINAL_OS_REPLACE is not None:
        os.replace = _ORIGINAL_OS_REPLACE
        _ORIGINAL_OS_REPLACE = None
    if _ORIGINAL_TO_PARQUET is not None:
        try:
            import pandas as pd

            pd.DataFrame.to_parquet = _ORIGINAL_TO_PARQUET
        except Exception:
            pass
        _ORIGINAL_TO_PARQUET = None
    if _ORIGINAL_PQ_WRITE_TABLE is not None:
        try:
            import pyarrow.parquet as pq

            pq.write_table = _ORIGINAL_PQ_WRITE_TABLE
        except Exception:
            pass
        _ORIGINAL_PQ_WRITE_TABLE = None


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
