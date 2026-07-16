"""执行期缓存会话：统一 L0–L3 与 runtime 统计。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from cache.expression_cache import ExpressionCache
from cache.layers import CacheHitStats, CacheLayer
from cache.panel_cache import PanelCache
from storage.cache import CacheManager, PersistentPlanCache


@dataclass
class ExecutionCacheSession:
    """绑定一次因子执行的缓存与统计。

    - **L0**：``shared_result_cache``（CSE / run_many）
    - **L1**：``panel_cache``（unstack 宽表）
    - **L2/L3**：``plan_cache``（``CacheManager`` / ``PersistentPlanCache``）
    """

    plan_cache: CacheManager | PersistentPlanCache | None = None
    shared_result_cache: dict[str, Any] | None = None
    panel_cache: dict[Any, Any] | None = None
    stats: CacheHitStats | None = None

    def __post_init__(self) -> None:
        """初始化默认 dict 与 ``ExpressionCache`` / ``PanelCache`` 包装。"""
        if self.panel_cache is None:
            self.panel_cache = {}
        if self.shared_result_cache is None:
            self.shared_result_cache = {}
        if self.stats is None:
            self.stats = CacheHitStats()
        self._expression_cache = ExpressionCache(self.shared_result_cache, stats=self.stats)
        self._panel_cache = PanelCache(self.panel_cache)

    @property
    def expression_cache(self) -> ExpressionCache:
        """L0 CSE 共享结果缓存包装。"""
        return self._expression_cache

    @property
    def panel_cache_store(self) -> PanelCache:
        """L1 panel unstack 缓存包装。"""
        return self._panel_cache

    def wrap_context(self, ctx: Any) -> Any:
        """把分层缓存挂到 ``ExecutionContext``。"""
        from backend.context import ExecutionContext

        if not isinstance(ctx, ExecutionContext):
            raise TypeError(f"expected ExecutionContext, got {type(ctx)!r}")
        return replace(
            ctx,
            cache=self.plan_cache,
            shared_result_cache=self.shared_result_cache,
            panel_cache=self.panel_cache,
            runtime_stats={"cache": self.stats},
        )

    def get_shared(self, sid: str) -> Any | None:
        """L0：读取 CSE 共享子树结果。"""
        return self._expression_cache.get(sid)

    def set_shared(self, sid: str, value: Any) -> None:
        """L0：写入 CSE 共享子树结果。"""
        return self._expression_cache.set(sid, value)

    def get_panel(self, key: Any) -> Any | None:
        """L1：读取 panel 缓存并更新命中统计。"""
        hit = self._panel_cache.get(key)
        if hit is not None and self.stats:
            self.stats.record_hit(CacheLayer.L1_PANEL)
        elif self.stats:
            self.stats.record_miss(CacheLayer.L1_PANEL)
        return hit

    def set_panel(self, key: Any, panel: Any) -> None:
        """L1：写入 panel 缓存。"""
        self._panel_cache.set(key, panel)

    def get_subplan(self, key: str) -> Any | None:
        """L2/L3：读取子计划缓存（内存或磁盘）。"""
        if self.plan_cache is None:
            return None
        hit = self.plan_cache.get(key)
        if hit is not None:
            if self.stats:
                layer = (
                    CacheLayer.L3_DISK
                    if isinstance(self.plan_cache, PersistentPlanCache)
                    else CacheLayer.L2_SUBPLAN
                )
                self.stats.record_hit(layer)
            return hit
        if self.stats:
            layer = (
                CacheLayer.L3_DISK
                if isinstance(self.plan_cache, PersistentPlanCache)
                else CacheLayer.L2_SUBPLAN
            )
            self.stats.record_miss(layer)
        return None

    def set_subplan(self, key: str, value: Any) -> None:
        """L2/L3：写入子计划缓存。"""
        if self.plan_cache is not None:
            self.plan_cache.set(key, value)

    def column_cache_stats(self, data_source: Any) -> dict[str, int]:
        """汇总 DataSource 侧列/panel 缓存条目数（L1_COLUMN）。"""
        fn = getattr(data_source, "column_cache_stats", None)
        if callable(fn):
            return dict(fn())
        cache = getattr(data_source, "_column_cache", None)
        panels = getattr(data_source, "_panel_cache", None)
        return {
            "cached_columns": len(cache) if isinstance(cache, dict) else 0,
            "cached_panels": len(panels) if isinstance(panels, dict) else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """导出本会话缓存摘要（供 audit / metrics）。"""
        out: dict[str, Any] = {}
        if self.stats is not None:
            out["stats"] = self.stats.to_dict()
        out["l0_keys"] = len(self.shared_result_cache or {})
        out["l1_keys"] = len(self.panel_cache or {})
        return out
