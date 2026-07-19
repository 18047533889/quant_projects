"""执行期上下文：数据源、缓存与运行时配置。

``ExecutionContext`` 贯穿所有后端执行路径，承载：
- 列级数据源（``DataSource``）与可选列缓存；
- MultiIndex 层级列名（``timestamp_col`` / ``instrument_col``）；
- CSE 共享子树、SQL 物化、panel 转换等跨子计划状态；
- 性能提示（``perf``）、查询预算（``query_budget``）与运行时遥测（``runtime_stats``）。
"""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from storage.cache import CacheManager
from storage.datasource import DataSource


@dataclass
class ExecutionContext:
    """因子引擎执行期上下文，后端从 ``data_source`` 按列名拉数。

    属性
    ----
    data_source : DataSource
        列式数据源，提供 ``load_column`` / ``scan_polars_long`` 等读接口。
    cache : CacheManager | None
        可选子计划结果缓存，按 ``plan_cache_key`` memoize。
    timestamp_col : str
        MultiIndex 时间层级列名，默认 ``"timestamp"``。
    instrument_col : str
        MultiIndex 标的层级列名，默认 ``"instrument"``。
    shared_result_cache : dict[str, Any] | None
        多因子 CSE 预计算的共享子树结果，键为 ``sid``，值为 Series/DataFrame。
    shared_long_lazy_cache : dict[str, Any] | None
        CSE / ``plan_ref`` 共享子树的 long-table LazyFrame，键为 ``sid``。
    materialized_long_lazy : dict[str, Any] | None
        SQL partial pushdown 物化列的 long-table LazyFrame，键为 ``sid``。
    materialize_sql_as_long_lazy : bool
        HybridLongBackend 标志：SQL 子树优先物化为 long LazyFrame。
    panel_cache : dict[int, Any] | None
        Series id → 宽表 panel，避免同一子树重复 unstack。
    template_series : Any | None
        panel-native 模式下，最终 stack 对齐用的 MultiIndex Series 模板。
    materialized_series : dict[str, Any] | None
        SQL 预计算的子树结果，键为 ``sid``，值为 Series（partial pushdown）。
    prefer_long_table : bool
        长表模式：优先保持 MultiIndex Series，减少 unstack/stack。
    perf : Any | None
        可选性能提示（分块、内存上限、算子后端偏好等），见 ``runtime.perf_config``。
    query_budget : Any | None
        可选 DuckDB 读预算（``data_access.QueryBudget``），SQL 下推读路径硬限制。
    runtime_stats : dict[str, Any] | None
        运行时统计（缓存命中、Polars 热路径计数、SQL 下推遥测等）。
    prefer_polars_panel : bool
        PolarsBackend 标志：宽表尽量保持 ``polars.DataFrame`` 形态。
    """

    data_source: DataSource
    run_mode: str = "research"
    registry_version: int = 0
    evidence_version: str = ""
    execution_id: str = ""
    production_fallback_policy: str = "error"
    cache: CacheManager | None = None
    timestamp_col: str = "timestamp"
    instrument_col: str = "instrument"
    #: 多因子 CSE 后，预计算的共享子树结果 ``sid -> Series|DataFrame``
    shared_result_cache: dict[str, Any] | None = None
    #: CSE / plan_ref：共享子树的 long-table LazyFrame ``sid -> LazyFrame(ts,inst,_v)``
    shared_long_lazy_cache: dict[str, Any] | None = None
    #: SQL partial pushdown 物化列的 long-table LazyFrame ``sid -> LazyFrame``
    materialized_long_lazy: dict[str, Any] | None = None
    #: HybridLongBackend：SQL 子树优先物化为 long LazyFrame
    materialize_sql_as_long_lazy: bool = False
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

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = uuid4().hex
        self.run_mode = str(self.run_mode).lower()
        if self.production_fallback_policy not in {"error", "warn"}:
            raise ValueError("production_fallback_policy must be 'error' or 'warn'")
