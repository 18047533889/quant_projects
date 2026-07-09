"""执行期上下文：数据源、可选列缓存、MultiIndex 层级列名。"""

from dataclasses import dataclass
from typing import Any

from storage.cache import CacheManager
from storage.datasource import DataSource


@dataclass
class ExecutionContext:
    """后端从 ``data_source`` 按列名拉数；``timestamp_col``/``instrument_col`` 与数据对齐。"""

    data_source: DataSource
    cache: CacheManager | None = None
    timestamp_col: str = "timestamp"
    instrument_col: str = "instrument"
    #: 多因子 CSE 后，预计算的共享子树结果 ``sid -> Series|DataFrame``
    shared_result_cache: dict[str, Any] | None = None
    #: CSE / plan_ref：共享子树的 long-table LazyFrame ``sid -> LazyFrame(ts,inst,_v)``
    shared_long_lazy_cache: dict[str, Any] | None = None
    #: SQL partial pushdown 物化列的 long-table LazyFrame ``sid -> LazyFrame``
    materialized_long_lazy: dict[str, Any] | None = None
    #: Series id → 宽表 panel，避免同一子树重复 unstack
    panel_cache: dict[int, Any] | None = None
    #: panel-native 模式下，最终 stack 对齐用的 MultiIndex Series 模板
    template_series: Any | None = None
    #: SQL 预计算的子树结果 ``sid -> Series``（partial pushdown）
    materialized_series: dict[str, Any] | None = None
    #: 长表模式：优先保持 MultiIndex Series，减少 unstack/stack
    prefer_long_table: bool = False
    #: 可选性能提示（分块、内存上限等），见 :mod:`runtime.perf_config`
    perf: Any | None = None
    #: 可选 DuckDB 读预算（``data_access.QueryBudget``），SQL 下推 / 读路径硬限制
    query_budget: Any | None = None
    #: 运行时统计（``ExecutionCacheSession.stats``、Polars 热路径计数等）
    runtime_stats: dict[str, Any] | None = None
    #: PolarsBackend：宽表尽量保持 polars.DataFrame
    prefer_polars_panel: bool = False
