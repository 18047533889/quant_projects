"""缓存层级定义与命中统计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CacheLayer(str, Enum):
    """因子引擎读/算缓存分层。"""

    L0_CSE = "l0_cse"  # run_many shared_result_cache
    L1_PANEL = "l1_panel"  # Series→panel unstack
    L1_COLUMN = "l1_column"  # DataSource 列/panel 预加载
    L2_SUBPLAN = "l2_subplan"  # CacheManager 内存
    L3_DISK = "l3_disk"  # PersistentPlanCache parquet


@dataclass
class CacheHitStats:
    """单次 run / run_many 的缓存命中计数。"""

    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    polars_ops: list[str] = field(default_factory=list)

    def record_hit(self, layer: CacheLayer) -> None:
        key = layer.value
        self.hits[key] = self.hits.get(key, 0) + 1

    def record_miss(self, layer: CacheLayer) -> None:
        key = layer.value
        self.misses[key] = self.misses.get(key, 0) + 1

    def record_polars_op(self, op: str) -> None:
        self.polars_ops.append(str(op))

    def to_dict(self) -> dict:
        return {
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "polars_ops": list(self.polars_ops),
            "polars_op_count": len(self.polars_ops),
        }
