"""执行后端工厂：按名称构造 Pandas / Polars / SQL / Hybrid 等 Backend 实例。"""

import os

from .debug_backend import DebugBackend
from .pandas_backend import PandasBackend
from .polars_backend import PolarsBackend


def build_backend(backend_type: str):
    """根据配置字符串构建执行后端。

    参数：
        backend_type: 不区分大小写。常用值见下表。

    返回值：
        实现 ``backend.base.Backend`` 的具体实例。

    支持的后端名
    ------------
    - ``pandas`` / ``pandas_modin``：经 cleaned_operators 的 Pandas 路径
    - ``polars`` / ``polars_lazy``：Polars 宽表或 Lazy 路径
    - ``polars_long`` / ``auto_long``：长表 Polars 编译（见 polars_long_backend）
    - ``duckdb_sql`` / ``clickhouse_sql``：SQL 子树下推
    - ``auto`` / ``hybrid``：自动选 SQL + Polars/Pandas fallback
    - ``debug``：只打印计划，不读数据

    示例：
        >>> build_backend("auto")
        >>> build_backend(os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "auto"))
    """
    normalized = backend_type.strip().lower()

    if normalized == "debug":
        return DebugBackend()  # 打印计划树，不调数据源
    if normalized == "pandas":
        return PandasBackend()
    if normalized == "pandas_modin":
        os.environ.setdefault("FACTOR_ENGINE_USE_MODIN", "1")
        return PandasBackend()  # 与 pandas 同类，经 pandas_compat 走 modin.pandas
    if normalized == "polars":
        return PolarsBackend()
    if normalized in {"polars_long", "polars_native", "long_polars"}:
        from .polars_long_backend import PolarsLongBackend

        return PolarsLongBackend()
    if normalized in {"auto_long", "hybrid_long"}:
        from .hybrid_long_backend import HybridLongBackend

        return HybridLongBackend()
    if normalized == "polars_lazy":
        return PolarsBackend(use_lazy=True)  # LazyFrame 延迟计算再 collect
    if normalized in {"duckdb_sql", "sql_pushdown", "duckdb_pushdown", "sql"}:
        from .duckdb_pushdown_backend import DuckDBPushdownBackend

        return DuckDBPushdownBackend()
    if normalized in {"clickhouse_sql", "ch_sql", "clickhouse_pushdown"}:
        from .duckdb_pushdown_backend import DuckDBPushdownBackend

        # SqlBackend 按 data_source 自动选 DuckDB / ClickHouse 方言
        return DuckDBPushdownBackend()
    if normalized in {"auto", "hybrid"}:
        from .hybrid_backend import HybridBackend

        return HybridBackend()

    raise ValueError(f"Unsupported backend type: {backend_type}")
