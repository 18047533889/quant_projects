"""Execution context carrying data source, caches and runtime telemetry."""
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from storage.cache import CacheManager
from storage.datasource import DataSource


@dataclass
class ExecutionContext:
    data_source: DataSource
    run_mode: str = "research"
    registry_version: int = 0
    evidence_version: str = ""
    execution_id: str = ""
    production_fallback_policy: str = "error"
    cache: CacheManager | None = None
    timestamp_col: str = "timestamp"
    instrument_col: str = "instrument"
    shared_result_cache: dict[str, Any] | None = None
    shared_long_lazy_cache: dict[str, Any] | None = None
    materialized_long_lazy: dict[str, Any] | None = None
    materialize_sql_as_long_lazy: bool = False
    panel_cache: dict[int, Any] | None = None
    template_series: Any | None = None
    # Phase 5 R16：panel-native 模板只保留 axis（MultiIndex），不再整份保留 stacked
    # Series 值——宽表 panel + 完整 stacked Series 同时存在的双份拷贝是内存放大主因。
    template_index: Any | None = None
    materialized_series: dict[str, Any] | None = None
    prefer_long_table: bool = False
    perf: Any | None = None
    query_budget: Any | None = None
    runtime_stats: dict[str, Any] | None = None
    prefer_polars_panel: bool = False
    # R36 P0-021（§104/105）：governed CSE buffer store（取代 raw dict 权威写入）。
    shared_buffers: Any = None

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = uuid4().hex
        self.run_mode = str(self.run_mode).lower()
        if self.production_fallback_policy not in {"error", "warn"}:
            raise ValueError("production_fallback_policy must be 'error' or 'warn'")
        # R20-114..118：execution-local wrapper 只挂在 ctx 上，不 setattr 到共享
        # inner（并发 context 会互相覆盖 pointer）。inner 的
        # ``_factor_engine_lqtp_wrapper`` 由 per-thread 注册表提供（见
        # ``storage.sources.data_access_source.register_logical_wrapper``）。
        self._logical_wrapper = None  # type: ignore[attr-defined]
        try:
            from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource

            if not isinstance(self.data_source, LQTPLogicalDataSource):
                inner = self.data_source
                # A new wrapper is created for every execution so SourceRef,
                # financial, benchmark and intraday memoisation cannot cross run
                # or snapshot boundaries.
                wrapper = LQTPLogicalDataSource(inner)
                setattr(wrapper, "_execution_id", self.execution_id)
                try:
                    from storage.sources.data_access_source import (
                        register_logical_wrapper,
                    )

                    register_logical_wrapper(inner, wrapper)
                except Exception:
                    pass
                self.data_source = wrapper
                self._logical_wrapper = wrapper
            else:
                # Explicitly supplied wrappers are still bound to this context;
                # execution-local caches from a previous context are cleared.
                wrapper = self.data_source
                previous = str(getattr(wrapper, "_execution_id", ""))
                if previous and previous != self.execution_id:
                    for name in tuple(vars(wrapper)):
                        if name.startswith("_intraday_") and name.endswith("_cache"):
                            delattr(wrapper, name)
                setattr(wrapper, "_execution_id", self.execution_id)
                self._logical_wrapper = wrapper
        except ImportError:
            pass
