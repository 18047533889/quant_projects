"""R25 runtime —— 全局资源治理与统一 PreparedRead。

    - ``resource_governor``  GlobalResourceGovernor / ResourceReservation（§27）
    - ``prepared_read``      PreparedRead（§24 多时间轴 planner 对象）
    - ``cache_manager``      CacheManager（§28 pin/GC/quota）
"""

from .resource_governor import (
    GlobalResourceGovernor,
    ResourceReservation,
    get_global_governor,
)
from .cache_manager import (
    CacheManager,
    CacheEntryState,
    get_cache_manager,
)

__all__ = [
    "GlobalResourceGovernor",
    "ResourceReservation",
    "get_global_governor",
    "CacheManager",
    "CacheEntryState",
    "get_cache_manager",
]
