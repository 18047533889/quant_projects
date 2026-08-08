"""
data_access.retry —— IO 抖动重试装饰器

背景：
    共享盘（可能被 NFS 远程 mount）上偶发的 OSError / 瞬时读失败是常态。
    原 ParquetSource 在 reader 内部写死了两次重试 + pyarrow 降级，耦合过重。
    本模块把重试抽成装饰器，谁需要谁加，和 engine/reader 解耦。

职责：
    只处理「瞬时 IO 错误」，不处理业务错误（文件不存在、schema 不匹配等不重试）。

#P1-4 ``ExceptionClassifier``：把 DuckDB / pyarrow 包装过的异常也分类到
TRANSIENT_IO / THROTTLED / NETWORK_RESET / DEADLINE / SCHEMA / INVALID_QUERY /
CORRUPTION / AUTH——之前只捕 OSError/IOError/TimeoutError，DuckDB remote/NFS/
httpfs 的错误经常包装在自己的 exception hierarchy 里，retry 捕不到。

#P1-5 默认不 ``gc.collect()``（高并发抖动时大量任务一起 gc 会形成同步抖动），
只重试前指数退避 + jitter + 可选 absolute deadline。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import enum
import functools
import gc
import logging
import time
from typing import Callable, TypeVar

from .exceptions import DataError, ValidationError

T = TypeVar("T")

logger = logging.getLogger("data_access.retry")


class ErrorClass(str, enum.Enum):
    TRANSIENT_IO = "transient_io"       # 文件/网络瞬时抖动，可安全重试
    THROTTLED = "throttled"             # 限流（HTTP 429 / 背压）
    NETWORK_RESET = "network_reset"     # 连接被重置
    DEADLINE = "deadline"               # 查询超时（不重试——重试只会更慢）
    SCHEMA = "schema"                   # schema/类型不匹配（确定性错误）
    INVALID_QUERY = "invalid_query"     # SQL/查询本身错误（确定性错误）
    CORRUPTION = "corruption"           # 文件损坏（确定性错误）
    AUTH = "auth"                       # 凭证/权限（重试无意义）
    UNKNOWN = "unknown"


# WHY：只对「可能是瞬时抖动」的错误重试；业务/确定性错误重试无意义反而拖慢失败反馈。
_RETRYABLE_CLASSES = frozenset(
    {ErrorClass.TRANSIENT_IO, ErrorClass.THROTTLED, ErrorClass.NETWORK_RESET}
)
_NON_RETRYABLE_EXCEPTIONS = (ValidationError, DataError, KeyError, TypeError)


def classify_exception(exc: BaseException) -> ErrorClass:
    """#P1-4 把异常分类；DuckDB 的包装 hierarchy 也覆盖。"""
    import duckdb

    if isinstance(exc, (OSError, IOError)):
        text = str(exc).lower()
        if any(t in text for t in ("timed out", "timeout", "reset", "broken pipe",
                                   "connection reset", "network")):
            return ErrorClass.NETWORK_RESET
        if any(t in text for t in ("throttl", "too many", "quota", "rate limit",
                                   "busy", "slow down")):
            return ErrorClass.THROTTLED
        return ErrorClass.TRANSIENT_IO
    if isinstance(exc, TimeoutError):
        return ErrorClass.DEADLINE
    if isinstance(exc, duckdb.Error):
        text = str(exc).lower()
        if any(t in text for t in ("interrupt", "deadline", "cancelled",
                                   "canceled", "query aborted")):
            return ErrorClass.DEADLINE
        if any(t in text for t in ("httpfs", "s3", "connection", "io error",
                                   "unable to open", "network", "socket",
                                   "timeout", "reset")):
            return ErrorClass.TRANSIENT_IO
        if any(t in text for t in ("credentials", "unauthorized", "forbidden",
                                   "access denied", "permission", "403", "401")):
            return ErrorClass.AUTH
        if any(t in text for t in ("corrupt", "magic number", "truncated",
                                   "footer", "crc", "parquet error")):
            return ErrorClass.CORRUPTION
        if any(t in text for t in ("binder error", "catalog error", "schema",
                                   "column", "not found", "duplicate column")):
            return ErrorClass.SCHEMA
        if any(t in text for t in ("syntax", "parser error", "invalid input")):
            return ErrorClass.INVALID_QUERY
        # DuckDB IOError（httpfs 包装）默认按 transient 处理
        if "io" in text or "filesystem" in text:
            return ErrorClass.TRANSIENT_IO
        return ErrorClass.UNKNOWN
    return ErrorClass.UNKNOWN


def retry_io(
    max_attempts: int = 3,
    backoff_sec: float = 0.05,
    gc_before_retry: bool = False,
    jitter: float = 0.1,
    backoff_factor: float = 2.0,
    deadline_sec: float | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """给 IO 密集函数套上重试逻辑。

    参数：
        max_attempts: 最多尝试几次（含首次），默认 3 次
        backoff_sec: 首次重试前 sleep（指数倍增长）
        gc_before_retry: **默认 False**（#P1-5）；需要时显式开
        jitter: 退避的随机抖动比例 [0,1)，避免高并发同步重试
        backoff_factor: 指数退避倍率（默认 2.0）
        deadline_sec: 可选 absolute deadline——超过后即使还有重试机会也放弃

    #P1-4 只对 ``ErrorClass`` 属于 TRANSIENT_IO/THROTTLED/NETWORK_RESET 的重试；
    DEADLINE/SCHEMA/INVALID_QUERY/CORRUPTION/AUTH/UNKNOWN 一律不重试。
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            start = time.monotonic()
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except _NON_RETRYABLE_EXCEPTIONS:
                    # 业务错误：不重试，直接抛
                    raise
                except Exception as exc:
                    cls = classify_exception(exc)
                    if cls not in _RETRYABLE_CLASSES:
                        raise
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    if deadline_sec is not None and (time.monotonic() - start) >= deadline_sec:
                        break
                    logger.warning(
                        "IO 抖动重试 %d/%d: func=%s cls=%s err=%s",
                        attempt, max_attempts, func.__name__, cls.value, exc,
                    )
                    if gc_before_retry:
                        gc.collect()
                    delay = backoff_sec * (backoff_factor ** (attempt - 1))
                    if jitter and delay > 0:
                        delay *= 1.0 + (hash(func.__name__) % 1000) / 1000.0 * min(jitter, 0.5)
                    time.sleep(delay)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
