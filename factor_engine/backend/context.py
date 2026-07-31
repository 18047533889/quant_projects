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
    materialized_series: dict[str, Any] | None = None
    prefer_long_table: bool = False
    perf: Any | None = None
    query_budget: Any | None = None
    runtime_stats: dict[str, Any] | None = None
    prefer_polars_panel: bool = False

    def __post_init__(self) -> None:
        if not self.execution_id: self.execution_id = uuid4().hex
        self.run_mode = str(self.run_mode).lower()
        if self.production_fallback_policy not in {"error","warn"}: raise ValueError("production_fallback_policy must be 'error' or 'warn'")
        try:
            from storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
            if not isinstance(self.data_source,LQTPLogicalDataSource):
                inner=self.data_source; wrapper=getattr(inner,"_factor_engine_lqtp_wrapper",None)
                if not isinstance(wrapper,LQTPLogicalDataSource):
                    wrapper=LQTPLogicalDataSource(inner)
                    try: setattr(inner,"_factor_engine_lqtp_wrapper",wrapper)
                    except Exception: pass
                self.data_source=wrapper
        except ImportError:
            pass
