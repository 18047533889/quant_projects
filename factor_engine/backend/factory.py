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
    - ``auto`` / ``hybrid``：SQL + Polars；数据源有 ``scan_polars_long`` 时自动走
      ``hybrid_long``（DuckDB long 物化 + 原生 Polars long），否则宽表 hybrid
    - ``debug``：只打印计划，不读数据
    - ``q_kdb`` / ``q``：Q/KDB 物理执行后端（R21-Q-FACTORY-INTEGRATION）

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
        # R40 #140: 不再改 process-global ``os.environ``（禁止 job path 污染环境）。
        # modin 选择改为 execution-context scoped —— PandasBackend 在构造时把
        # 开关写入 ``pandas_compat._MODIN_ENABLED`` ContextVar。
        return PandasBackend(use_modin_pandas=True)
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
        from .duckdb_pushdown_backend import ClickHousePushdownBackend

        return ClickHousePushdownBackend()
    if normalized in {"auto", "hybrid"}:
        from .hybrid_backend import HybridBackend

        return HybridBackend()
    if normalized in {"q_kdb", "q"}:
        from .q_backend import QBackend

        return QBackend(fallback_to_pandas=False, production_mode=True)

    raise ValueError(f"Unsupported backend type: {backend_type}")


def build_backend_execution_certificate(
    backend,
    *,
    structural_hash: str,
    bound_ops,
    backend_eligibility,
    output_shape_hash: str,
):
    """R40 #141: 构建带 backend 选择链的 ``ProductionExecutionCertificate``。

    ``build_backend("clickhouse_sql")`` 返回的 backend 已携带
    ``_requested_backend="clickhouse_sql"`` / ``_resolved_dialect="clickhouse"``
    （以及可选的 ``_datasource_identity``）。本 helper 把这些选择链字段写入
    证书，使 runtime O(1) backend 事件校验能证明实际解析方言，而不是把
    clickhouse 路由静默落到无记录的 DuckDB 路径。
    """
    from factor_engine.runtime.production_execution_certificate import ProductionExecutionCertificate

    return ProductionExecutionCertificate.build(
        structural_hash=str(structural_hash or ""),
        bound_ops=bound_ops,
        backend_eligibility=backend_eligibility,
        output_shape_hash=str(output_shape_hash or ""),
        requested_backend=str(getattr(backend, "_requested_backend", "") or ""),
        resolved_dialect=str(getattr(backend, "_resolved_dialect", "") or ""),
        datasource_identity=str(getattr(backend, "_datasource_identity", "") or ""),
    )
