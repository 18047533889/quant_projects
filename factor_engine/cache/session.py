"""执行期缓存会话：统一 L0–L3 与 runtime 统计。"""

from __future__ import annotations

import os
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
    cse_budget_bytes: int | None = None
    panel_budget_bytes: int | None = None
    execution_id: str = "session"
    # R36 P0-022：governor 层注册失败时 fail-closed（production）/ warning（research）。
    strict: bool | None = None
    # R38 P0-039（§15）：run_mode 从 Engine execution context 显式传入，不再只看
    # 环境变量（env 没设时 config/ExecutionContext 可能已是 production）。
    run_mode: str | None = None

    def __post_init__(self) -> None:
        """初始化默认 dict 与 ``ExpressionCache`` / ``PanelCache`` / ``BufferStore``。"""
        if self.panel_cache is None:
            self.panel_cache = {}
        if self.shared_result_cache is None:
            self.shared_result_cache = {}
        if self.stats is None:
            self.stats = CacheHitStats()
        if self.strict is None:
            from runtime.production_policy import is_production_mode

            self.strict = bool(
                is_production_mode(self.run_mode)
                or os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production"
            )
        # 审计 #334：session 层名带 execution_id，不同 session 不再互踩记账；
        # 同一 execution_id 重复注册时后者覆盖（可接受）。
        self._l0_layer = f"{self.execution_id}:l0_cse"
        self._l1_layer = f"{self.execution_id}:l1_panel"
        # R36 P0-021（§104/105）：CSE 共享缓冲的 governed 写入通道——取代 raw
        # ``shared_result_cache[sid]=value`` 作为唯一权威写入路径。
        # R38 P0-032（§13）：挂真实 SpillStore，spill() 是真 spill 不是 drop。
        from runtime.buffer_store import GovernedBufferStore
        from runtime.spill_store import SpillStore

        self._buffer_store = GovernedBufferStore(
            self.shared_result_cache,
            budget_bytes=self.cse_budget_bytes,
            spill_store=SpillStore(),
            execution_id=self.execution_id,
        )
        # R38 P0-038（P0-014）：L0 **唯一 owner 是 GovernedBufferStore**。
        # ExpressionCache 只作 adapter（governed_store 模式）——不再双套 accounting。
        self._expression_cache = ExpressionCache(
            self.shared_result_cache,
            stats=self.stats,
            budget_bytes=self.cse_budget_bytes,
            layer_name=self._l0_layer,
            governed_store=self._buffer_store,
        )
        self._panel_cache = PanelCache(
            self.panel_cache,
            budget_bytes=self.panel_budget_bytes,
            layer_name=self._l1_layer,
        )
        # Phase 5 R6：把 CSE / panel 两层注册到全局 MemoryGovernor 的 evict hooks，
        # RSS 高压档时由 governor 主动逐出。
        # R36 P0-022（§107）：注册失败不能 ``except: pass``——生产下「以为有内存
        # 治理、实际没有」比没有治理更危险，必须 fail-closed。
        try:
            from runtime.resource_governor import global_memory_governor

            gov = global_memory_governor()
            # R38 P0-038：L0 高压逐出直接指向 GovernedBufferStore（唯一 owner，
            # 含 pinned/refcount/spill 语义），不再绑到 ExpressionCache 自己的
            # 一套 accounting（split-brain 修复）。
            gov.register_layer(self._l0_layer, self._buffer_store.evict_if_over_budget)
            gov.register_layer(self._l1_layer, self._panel_cache.evict_if_over_budget)
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                raise RuntimeError(
                    "ExecutionCacheSession: 无法注册 governor evict layer "
                    f"（R36 P0-022 fail-closed）——{type(exc).__name__}: {exc}"
                ) from exc
            import logging

            logging.getLogger(__name__).warning(
                "cache session 注册 governor layer 失败（research 模式降级 warning）: %s",
                exc,
            )

    @property
    def buffer_store(self) -> Any:
        """R36 P0-021：governed CSE buffer store（put/get/release/spill）。"""
        return self._buffer_store

    def release(self) -> None:
        """释放本 session 在 MemoryGovernor 中的层注册与记账（审计 #334）。

        R36 P0-023（§108）：unregister_layer 只移除 governor accounting；必须同时
        **清空 backing dict 的真实引用**（``shared_result_cache`` / ``panel_cache``），
        否则「实际内存仍在、账面已经没了」。

        R37-P0-037（§37.3）：unregister 失败不再 ``except: pass`` 静默吞掉——
        production 下「以为已释放、实际 governor 还记着」会造成 accounting drift，
        必须 fail-closed；research 降级 warning + telemetry。release 后做
        accounting reconciliation（declared vs actual），不一致必须告警。
        """
        import logging

        log = logging.getLogger(__name__)
        try:
            from runtime.resource_governor import global_memory_governor

            gov = global_memory_governor()
            gov.unregister_layer(self._l0_layer)
            gov.unregister_layer(self._l1_layer)
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                raise RuntimeError(
                    "ExecutionCacheSession: 无法释放 governor evict layer "
                    f"（R37-P0-037 fail-closed）——{type(exc).__name__}: {exc}"
                ) from exc
            log.warning(
                "cache session unregister governor layer 失败（research 降级）: %s", exc)
        # §108：backing refs 与 accounting 一起释放。
        if self.shared_result_cache is not None:
            self.shared_result_cache.clear()
        if self.panel_cache is not None:
            self.panel_cache.clear()
        # R37-P0-037 reconciliation：declared cache bytes vs actual retained refs。
        # 不一致必须告警；production 超阈值应 fail。
        try:
            declared = 0
            if self.stats is not None:
                declared = getattr(self.stats, "approx_bytes", None) or declared
            actual = (self.shared_result_cache is not None and
                      self._estimate_dict_bytes(self.shared_result_cache)) or 0
            if declared > 0 and actual > declared * 1.5:
                msg = (f"cache accounting drift: declared={declared} actual={actual} "
                       f"(session={self.execution_id})")
                if self.strict:
                    raise RuntimeError(f"R37-P0-037 {msg}")
                log.warning("R37-P0-037 %s", msg)
        except Exception:
            raise

    @staticmethod
    def _estimate_dict_bytes(d: dict[Any, Any]) -> int:
        try:
            from runtime.resource_governor import estimate_object_bytes

            return max(0, int(estimate_object_bytes(d)))
        except Exception:
            return 0

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
            shared_buffers=self._buffer_store,
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
