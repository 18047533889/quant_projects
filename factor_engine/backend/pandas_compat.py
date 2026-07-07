"""Pandas API 入口：可选 ``modin.pandas``，否则标准 ``pandas``。

通过环境变量 ``FACTOR_ENGINE_USE_MODIN=1``（或 ``build_backend(\"pandas_modin\")`` 在首次解析前设置）
启用 Modin。首次解析后模块会缓存实现；单进程内请勿依赖运行时反复切换。

``pd`` 为惰性代理：属性访问时解析真实模块，以便 ``isinstance(x, pd.Series)`` 等行为正常。
"""

from __future__ import annotations

import os
from typing import Any

_cached_impl: Any | None = None


def _env_use_modin() -> bool:
    return os.environ.get("FACTOR_ENGINE_USE_MODIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def resolve_pandas_module() -> Any:
    """返回 ``modin.pandas`` 或 ``pandas``；可被测试用来预热或检查。"""
    global _cached_impl
    if _cached_impl is not None:
        return _cached_impl
    if _env_use_modin():
        try:
            import modin.pandas as mpd  # type: ignore

            _cached_impl = mpd
            return _cached_impl
        except ImportError:
            pass
    import pandas as pd

    _cached_impl = pd
    return _cached_impl


def reset_pandas_module_cache_for_tests() -> None:
    """仅测试：清空缓存，便于在同一进程内切换环境变量后重新解析。"""
    global _cached_impl
    _cached_impl = None


class _LazyPd:
    """将 ``pd.Series`` 等解析到 :func:`resolve_pandas_module`。"""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(resolve_pandas_module(), name)


pd: Any = _LazyPd()
