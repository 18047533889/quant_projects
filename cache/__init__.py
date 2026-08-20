"""缓存模块：统一缓存接口、分层缓存、装饰器。

R43 缓存策略优化：
    - UnifiedCache: 统一缓存接口（TTL、LRU、并发安全）
    - TieredCache: 分层缓存（L1/L2/L3）
    - @cached: 缓存装饰器（替代 lru_cache）
    - GlobalCacheManager: 全局缓存管理器
"""
from cache.cache_policy import CachePolicy

# 延迟导入，避免循环依赖
# from cache.unified_cache import UnifiedCache, EvictionPolicy
# from cache.tiered_cache import TieredCache
# from cache.cache_decorator import cached, cached_property, memoize
# from cache.cache_manager import GlobalCacheManager, get_global_cache_manager

# 保留原有导入（向后兼容）
from cache.column_cache import ColumnCacheScope, column_cache_scope
from cache.expression_cache import ExpressionCache
from cache.layers import CacheHitStats, CacheLayer
from cache.panel_cache import PanelCache
from cache.session import ExecutionCacheSession

__all__ = [
    "CachePolicy",
    "ColumnCacheScope",
    "column_cache_scope",
    "ExpressionCache",
    "CacheHitStats",
    "CacheLayer",
    "PanelCache",
    "ExecutionCacheSession",
]
