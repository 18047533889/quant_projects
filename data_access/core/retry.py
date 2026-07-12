"""
data_access.retry —— IO 抖动重试装饰器

背景：
    共享盘（可能被 NFS 远程 mount）上偶发的 OSError / 瞬时读失败是常态。
    原 ParquetSource 在 reader 内部写死了两次重试 + pyarrow 降级，耦合过重。
    本模块把重试抽成装饰器，谁需要谁加，和 engine/reader 解耦。

职责：
    只处理「瞬时 IO 错误」，不处理业务错误（文件不存在、schema 不匹配等不重试）。

维护人：quant 基础平台组    最后更新：2026-04-19
"""

from __future__ import annotations

import functools
import gc
import logging
import time
from typing import Callable, TypeVar

from .exceptions import DataError, ValidationError

T = TypeVar("T")

logger = logging.getLogger("data_access.retry")


# WHY：只对「可能是共享盘抖动」的错误重试；ValidationError / DataError 属于
#      确定性错误，重试无意义反而拖慢失败反馈。
_RETRYABLE_EXCEPTIONS = (OSError, IOError, TimeoutError)
_NON_RETRYABLE_EXCEPTIONS = (ValidationError, DataError, KeyError, TypeError)


def retry_io(
    max_attempts: int = 3,
    backoff_sec: float = 0.05,
    gc_before_retry: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """给 IO 密集函数套上重试逻辑。

    参数：
        max_attempts: 最多尝试几次（含首次），默认 3 次
        backoff_sec: 每次失败后 sleep 多久，简单线性退避
        gc_before_retry: 重试前 gc.collect()。pyarrow 在共享盘上偶发内存/
                         句柄问题时，手动 gc 有时能救场（原代码经验）

    用法：
        @retry_io()
        def read_file(path): ...

    WHY：不做指数退避 —— 这里对付的是毫秒级瞬时抖动，不是网络服务抖动；
         sleep 太久反而拖慢整体 pipeline。
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except _NON_RETRYABLE_EXCEPTIONS:
                    # 业务错误：不重试，直接抛
                    raise
                except _RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    logger.warning(
                        "IO 抖动重试 %d/%d: func=%s err=%s",
                        attempt, max_attempts, func.__name__, exc,
                    )
                    if gc_before_retry:
                        gc.collect()
                    time.sleep(backoff_sec)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
