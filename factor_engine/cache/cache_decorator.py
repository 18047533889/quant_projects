"""缓存装饰器：替代 @lru_cache，支持 TTL、智能失效、指标。

R43 缓存装饰器优化：
    - 替代 functools.lru_cache（功能增强）
    - 支持 TTL 自动过期
    - 支持 key 函数（自定义键生成）
    - 支持条件缓存（predicate）
    - 支持智能失效（invalidate_on）
    - 线程安全
    - 统计指标（hits/misses）
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
from typing import Any, Callable, TypeVar

from .unified_cache import UnifiedCache, EvictionPolicy

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _default_key_func(*args: Any, **kwargs: Any) -> str:
    """默认键生成函数（args + kwargs 哈希）。"""
    # 构建稳定的键
    parts = []

    # 位置参数
    for arg in args:
        try:
            # 尝试 JSON 序列化（稳定且可读）
            parts.append(json.dumps(arg, sort_keys=True, default=str))
        except (TypeError, ValueError):
            # Fallback: repr
            parts.append(repr(arg))

    # 关键字参数（排序保证稳定性）
    for k in sorted(kwargs.keys()):
        try:
            parts.append(f"{k}={json.dumps(kwargs[k], sort_keys=True, default=str)}")
        except (TypeError, ValueError):
            parts.append(f"{k}={repr(kwargs[k])}")

    # 哈希（避免键过长）
    raw = "|".join(parts)
    if len(raw) > 200:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw


def cached(
    maxsize: int | None = 128,
    ttl: float | None = None,
    typed: bool = False,
    key_func: Callable[..., str] | None = None,
    cache: UnifiedCache | None = None,
    predicate: Callable[[Any], bool] | None = None,
    namespace: str | None = None,
) -> Callable[[F], F]:
    """缓存装饰器（增强版 lru_cache）。

    Args:
        maxsize: 最大缓存条目数（None = 无限制）
        ttl: TTL（秒，None = 永不过期）
        typed: 是否区分参数类型（True = int(1) 和 float(1.0) 不同）
        key_func: 自定义键生成函数（None = 使用默认）
        cache: 外部缓存实例（None = 创建新实例）
        predicate: 缓存条件函数（返回 False 不缓存，None = 总是缓存）
        namespace: 命名空间前缀（多个函数共享缓存时区分）

    Returns:
        装饰后的函数

    Example:
        ```python
        @cached(maxsize=100, ttl=60.0)
        def expensive_compute(x: int) -> int:
            return x ** 2

        # 自定义键函数
        @cached(key_func=lambda obj, field: f"{obj.id}:{field}")
        def get_field(obj, field):
            return obj.data[field]

        # 条件缓存
        @cached(predicate=lambda result: result is not None)
        def fetch_data(key):
            return database.get(key)  # 只缓存非 None 结果
        ```
    """
    def decorator(func: F) -> F:
        # 创建或使用提供的缓存
        func_cache = cache or UnifiedCache(
            max_size=maxsize,
            eviction_policy=EvictionPolicy.LRU,
            default_ttl=ttl,
            name=f"cached:{func.__module__}.{func.__qualname__}",
        )

        # 键生成器
        keygen = key_func or _default_key_func

        # 统计
        stats = {"hits": 0, "misses": 0}
        stats_lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 生成缓存键
            try:
                # 如果有 key_func，直接使用
                if key_func:
                    cache_key = keygen(*args, **kwargs)
                else:
                    # 默认键生成：函数名 + 参数
                    arg_key = keygen(*args, **kwargs)
                    if typed:
                        # 区分类型：添加类型信息
                        type_sig = "|".join(type(a).__name__ for a in args)
                        cache_key = f"{func.__module__}:{func.__qualname__}:{type_sig}:{arg_key}"
                    else:
                        cache_key = f"{func.__module__}:{func.__qualname__}:{arg_key}"

                # 添加命名空间前缀
                if namespace:
                    cache_key = f"{namespace}:{cache_key}"

            except Exception:
                # 键生成失败：不缓存，直接调用
                return func(*args, **kwargs)

            # 查找缓存
            result = func_cache.get(cache_key)
            if result is not None:
                with stats_lock:
                    stats["hits"] += 1
                return result

            # Cache miss：调用原函数
            with stats_lock:
                stats["misses"] += 1
            result = func(*args, **kwargs)

            # 检查缓存条件
            if predicate is None or predicate(result):
                func_cache.set(cache_key, result, ttl=ttl)

            return result

        # 附加管理方法
        def cache_info() -> dict[str, Any]:
            """返回缓存统计信息。"""
            cache_stats = func_cache.stats()
            with stats_lock:
                return {
                    "function": f"{func.__module__}.{func.__qualname__}",
                    "hits": stats["hits"],
                    "misses": stats["misses"],
                    "hit_rate": stats["hits"] / max(1, stats["hits"] + stats["misses"]),
                    "cache_size": cache_stats.entry_count,
                    "cache_bytes": cache_stats.size_bytes,
                    "evictions": cache_stats.evictions,
                    "expirations": cache_stats.expirations,
                }

        def cache_clear() -> None:
            """清空缓存。"""
            func_cache.clear()
            with stats_lock:
                stats["hits"] = 0
                stats["misses"] = 0

        def cache_invalidate(*args: Any, **kwargs: Any) -> bool:
            """使特定参数的缓存失效。"""
            try:
                if key_func:
                    cache_key = keygen(*args, **kwargs)
                else:
                    arg_key = keygen(*args, **kwargs)
                    cache_key = f"{func.__module__}:{func.__qualname__}:{arg_key}"
                if namespace:
                    cache_key = f"{namespace}:{cache_key}"
                return func_cache.invalidate(cache_key)
            except Exception:
                return False

        # 附加到函数
        wrapper.cache_info = cache_info  # type: ignore
        wrapper.cache_clear = cache_clear  # type: ignore
        wrapper.cache_invalidate = cache_invalidate  # type: ignore
        wrapper.__wrapped__ = func  # type: ignore
        wrapper._cache = func_cache  # type: ignore

        return wrapper  # type: ignore

    return decorator


def cached_property(
    ttl: float | None = None,
    cache: UnifiedCache | None = None,
) -> Callable[[F], Any]:
    """缓存属性装饰器（替代 functools.cached_property，支持 TTL）。

    Args:
        ttl: TTL（秒，None = 永不过期）
        cache: 外部缓存实例（None = 使用实例属性）

    Returns:
        装饰后的属性

    Example:
        ```python
        class MyClass:
            @cached_property(ttl=60.0)
            def expensive_property(self):
                return self.compute()
        ```
    """
    def decorator(func: F) -> Any:
        attr_name = f"_cached_{func.__name__}"

        @functools.wraps(func)
        def wrapper(self: Any) -> Any:
            # 使用实例级缓存
            if cache is None:
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, UnifiedCache(
                        max_size=128,
                        default_ttl=ttl,
                        name=f"cached_property:{func.__name__}",
                    ))
                instance_cache = getattr(self, attr_name)
            else:
                instance_cache = cache

            # 缓存键：实例 id + 函数名
            cache_key = f"{id(self)}:{func.__name__}"

            # 查找或计算
            result = instance_cache.get(
                cache_key,
                factory=lambda: func(self),
                ttl=ttl,
            )
            return result

        return property(wrapper)

    return decorator


def invalidate_cache_on(
    *trigger_funcs: str,
    pattern: str | None = None,
) -> Callable[[F], F]:
    """智能失效装饰器（某些方法调用时自动失效缓存）。

    Args:
        *trigger_funcs: 触发失效的方法名列表
        pattern: 失效模式（None = 清空全部，否则按模式清空）

    Returns:
        装饰后的类

    Example:
        ```python
        @invalidate_cache_on("update_data", "delete_data")
        class MyClass:
            @cached()
            def get_data(self):
                return self._data

            def update_data(self, new_data):
                self._data = new_data
                # 自动失效 get_data 缓存
        ```
    """
    def decorator(cls: type) -> type:
        # 包装触发方法
        for method_name in trigger_funcs:
            if not hasattr(cls, method_name):
                continue

            original_method = getattr(cls, method_name)

            @functools.wraps(original_method)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original_method(self, *args, **kwargs)

                # 失效所有 @cached 方法
                for attr_name in dir(self):
                    attr = getattr(type(self), attr_name, None)
                    if attr and hasattr(attr, "_cache"):
                        if pattern:
                            attr._cache.clear(pattern=pattern)
                        else:
                            attr._cache.clear()

                return result

            setattr(cls, method_name, wrapper)

        return cls

    return decorator


def memoize(func: F) -> F:
    """简单记忆化装饰器（无大小限制，永久缓存）。

    Args:
        func: 要缓存的函数

    Returns:
        装饰后的函数

    Example:
        ```python
        @memoize
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
    """
    return cached(maxsize=None, ttl=None)(func)
