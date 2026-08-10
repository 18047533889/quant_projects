"""R25 runtime —— 全局资源治理与统一 PreparedRead。

    - ``resource_governor``  GlobalResourceGovernor / ResourceReservation（§27）
    - ``prepared_read``      PreparedRead / PredicateConstraint / TemporalPlan
                            （R26-P0-003 immutable executable plan）
    - ``read_pipeline``      ReadPipeline（R26-P0-004 统一执行链 + counters）
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
from .prepared_read import (
    PreparedRead,
    PredicateConstraint,
    TemporalPlan,
)
from .read_pipeline import (
    ReadPipeline,
    PipelineCounters,
    StreamingReservation,
    resolved_snapshot_from_files,
)

__all__ = [
    "GlobalResourceGovernor",
    "ResourceReservation",
    "get_global_governor",
    "CacheManager",
    "CacheEntryState",
    "get_cache_manager",
    "PreparedRead",
    "PredicateConstraint",
    "TemporalPlan",
    "ReadPipeline",
    "PipelineCounters",
    "StreamingReservation",
    "resolved_snapshot_from_files",
]
