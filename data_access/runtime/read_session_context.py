"""R38 P0-050（§20）：request-scoped resolution cache —— 用 ContextVar，不再改
Store 全局属性。

R29 的 ``DataReadSession.__enter__`` 直接 ``store._resolution_cache = my_cache``：
两个线程 A/B overlap 时，A exit 恢复 None 会把仍在跑的 B 的 cache 清掉（与
service broker module-global 竞态本质相同）。

修复：resolution cache 放 **ContextVar**（request context），Store prepare 时从
execution context 取，**不**修改 Store 全局属性。并发 session 互不覆盖。
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

#: 进程级 request-scoped resolution cache（None = 未启用）。
_resolution_cache_var: ContextVar[dict | None] = ContextVar(
    "data_access_resolution_cache", default=None
)


def get_resolution_cache() -> dict | None:
    """当前 request context 的 resolution cache（并发 session 各自持有）。"""
    return _resolution_cache_var.get()


def set_resolution_cache(cache: dict | None):
    """绑定当前线程的 resolution cache，返回 token（退出时 reset）。"""
    return _resolution_cache_var.set(cache)


def reset_resolution_cache(token: Any) -> None:
    _resolution_cache_var.reset(token)


def clear_resolution_cache() -> None:
    """清空当前 request context 的 resolution cache（job 内显式失效）。"""
    cache = _resolution_cache_var.get()
    if cache is not None:
        cache.clear()
