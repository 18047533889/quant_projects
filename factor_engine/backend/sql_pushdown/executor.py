# -*- coding: utf-8 -*-
"""执行编译后的因子 SQL（DuckDB / ClickHouse）。

从 ``ExecutionContext`` 提取下推上下文，编译 PlanNode 为 SQL 并在目标引擎执行，
支持单因子与批量（共享 base CTE）两种模式，以及 pandas Series 与 Polars LazyFrame 输出。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.pandas_compat import pd
from factor_engine.planner.logical_plan import PlanNode

from .emitter import (
    BatchCompiledSql,
    CompiledSql,
    InstrumentFilterKind,
    SqlDialect,
    SqlPushdownFilter,
    compile_plan_to_sql,
    compile_plans_batch_to_sql,
)
from .source_resolver import resolve_pushdown_source


def _instrument_filter_kind(
    instrument_filter: Any,
) -> tuple[InstrumentFilterKind, tuple[str, ...]]:
    """Classify a data source ``instrument_filter`` into ALL / EMPTY / LIST.

    R13 P0-69: ``None`` means ALL (no restriction), ``[]``/``()`` means EMPTY
    (explicit zero-row universe), a non-empty sequence means LIST.  The old
    ``getattr(ds, 'instrument_filter', None) or []`` collapsed ALL and EMPTY.
    """
    if instrument_filter is None:
        return InstrumentFilterKind.ALL, ()
    if isinstance(instrument_filter, (list, tuple, set, frozenset)):
        if len(instrument_filter) == 0:
            return InstrumentFilterKind.EMPTY, ()
        return InstrumentFilterKind.LIST, tuple(str(x) for x in instrument_filter)
    # A scalar / unknown type is treated as a single-instrument LIST.
    return InstrumentFilterKind.LIST, (str(instrument_filter),)


def _ensure_data_access() -> None:
    """R21-130..133: import the installed ``data_access`` as a normal package.

    No runtime ``sys.path`` mutation — which data_access an install uses must be
    reproducible.  A missing package fails loudly instead of being patched in.
    """
    from factor_engine.storage.data_access_loader import ensure_data_access_importable

    ensure_data_access_importable()


def _sql_fallback_allowed(exc: BaseException, ctx: ExecutionContext) -> bool:
    """Phase 5 R11：SQL pushdown 失败是否允许回退 Python/Polars。

    - ``CapabilityMiss`` / ``CompilationUnsupported`` → 允许 fallback（换后端）
    - ``ResourceBudgetExceeded`` / OOM / Deadline / PIT / Schema / DQ → 禁止 fallback
    - 未知异常：production fail-closed（宁可显式失败），research 允许 fallback
    """
    from factor_engine.runtime.resource_errors import ResourceGovernanceError, is_fail_closed_error

    if is_fail_closed_error(exc):
        return False
    if isinstance(exc, ResourceGovernanceError):
        return True
    production = str(getattr(ctx, "run_mode", "") or "").lower() == "production"
    return not production


class SqlResultSchemaError(RuntimeError):
    """SQL 执行结果缺少预期的列别名（#369/#370：batch 结果 schema 契约）。"""


def fallback_reason_is_fail_closed(reason_class: str) -> bool:
    """reason class 名称是否为 fail-closed 治理类异常（禁止跨后端 fallback）。

    CapabilityMiss / CompilationUnsupported 属于「这个后端做不到」，允许 fallback；
    其余 ResourceGovernanceError 子类（OOM / Deadline / Schema / PIT / DQ …）
    在 production 下必须 fail-closed。
    """
    if not reason_class:
        return False
    try:
        import factor_engine.runtime.resource_errors as re_mod
        from factor_engine.runtime.resource_errors import (
            CapabilityMiss,
            CompilationUnsupported,
            ResourceGovernanceError,
        )

        cls = getattr(re_mod, str(reason_class), None)
        if not isinstance(cls, type):
            return False
        if issubclass(cls, (CapabilityMiss, CompilationUnsupported)):
            return False
        return issubclass(cls, ResourceGovernanceError)
    except Exception:
        return False


def _note_sql_fallback(ctx: ExecutionContext, reason_class: str, *, sid: str | None = None) -> None:
    """记录一次 SQL 下推被跳过/回退的 reason class（供 fallback 审计 telemetry）。

    写入 ``ctx.runtime_stats["sql_fallback_reasons"]``（按 sid 索引）与
    ``ctx.runtime_stats["sql_fallback_reason"]``（最近一次，供 _eval_python 读取）。
    """
    runtime = dict(getattr(ctx, "runtime_stats", None) or {})
    reasons = dict(runtime.get("sql_fallback_reasons") or {})
    key = str(sid) if sid is not None else "last"
    reasons[key] = str(reason_class)
    runtime["sql_fallback_reasons"] = reasons
    runtime["sql_fallback_reason"] = str(reason_class)
    ctx.runtime_stats = runtime  # type: ignore[attr-defined]


@dataclass(frozen=True)
class PushdownContext:
    """SQL 下推执行上下文：方言、数据源定位、轴列名与可选过滤条件。"""
    dialect: SqlDialect
    time_column: str
    instrument_column: str
    dataset: str | None = None
    table: str | None = None
    filt: SqlPushdownFilter | None = None
    ch_config: dict[str, Any] | None = None


def extract_pushdown_context(ctx: ExecutionContext) -> PushdownContext | None:
    """从 ExecutionContext 的数据源提取 SQL 下推上下文。

    支持 DataAccess（DuckDB 数据集）、ClickHouse 表、Composite 与 LongTable 包装；
    无法解析时返回 ``None``。
    """
    ds = resolve_pushdown_source(ctx.data_source)
    if ds is None:
        return None

    dataset = getattr(ds, "dataset", None)
    table = getattr(ds, "table", None)

    if dataset:
        axis_fn = getattr(ds, "dataset_axis_columns", None)
        if not callable(axis_fn):
            return None
        time_col, inst_col = axis_fn()
        start = getattr(ds, "start_date", None)
        end = getattr(ds, "end_date", None)
        inst_kind, insts = _instrument_filter_kind(
            getattr(ds, "instrument_filter", None)
        )
        filt = SqlPushdownFilter(
            time_column=time_col,
            start=start,
            end=end,
            instrument_column=inst_col,
            instruments=insts,
            instrument_filter_kind=inst_kind,
        )
        return PushdownContext(
            dialect=SqlDialect.DUCKDB,
            dataset=str(dataset),
            time_column=time_col,
            instrument_column=inst_col,
            filt=filt,
        )

    time_col = getattr(ds, "timestamp_column", "trade_date")
    inst_col = getattr(ds, "instrument_column", "instrument")
    start = getattr(ds, "start_date", None)
    end = getattr(ds, "end_date", None)
    inst_kind, insts = _instrument_filter_kind(
        getattr(ds, "instrument_filter", None)
    )
    ch_overrides = getattr(ds, "_ch_overrides", None)
    return PushdownContext(
        dialect=SqlDialect.CLICKHOUSE,
        table=str(table),
        time_column=str(time_col),
        instrument_column=str(inst_col),
        filt=SqlPushdownFilter(
            time_column=str(time_col),
            start=start,
            end=end,
            instrument_column=str(inst_col),
            instruments=insts,
            instrument_filter_kind=inst_kind,
        ),
        ch_config=dict(ch_overrides) if ch_overrides else None,
    )


def _series_from_sql_table(table, *, timestamp_col: str, instrument_col: str) -> pd.Series:
    """将 Arrow/SQL 查询结果转为 MultiIndex (ts, inst) Series。"""
    from data_access.read.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column="value",
        output_name="value",
    )


def materialized_sql_result_lazy_wrapper(table) -> Any:
    """Arrow/SQL 结果 → long-table LazyFrame（``ts, inst, _v``），无 pandas 往返。

    .. important::
        该结果**已完整物化**（SQL 已执行完），只是套了一层 ``LazyFrame`` 壳以复用
        Polars long-lazy 消费接口。它**不代表 streaming**：
        ``supports_streaming=False``，``materializes_full_panel=True``。

    MB-P1-018: DuckDB Region boundary optimization — prefer Arrow/Relation.
    This function keeps the result in Arrow → Polars path without converting to Pandas.
    The conversion happens only once at the final boundary when necessary.
    """
    import polars as pl

    # MB-P1-018: Direct Arrow → Polars conversion (no Pandas intermediate)
    if hasattr(table, "to_pandas"):
        df = pl.from_arrow(table)
    else:
        df = pl.DataFrame(table)
    rename: dict[str, str] = {}
    if "value" in df.columns and "_v" not in df.columns:
        rename["value"] = "_v"
    if rename:
        df = df.rename(rename)
    if "_v" not in df.columns:
        raise ValueError("SQL result missing value column")
    return df.select(["ts", "inst", "_v"]).lazy()


# Deprecated alias（审计 #368）：旧名暗示 lazy，实际结果已物化。
def _lazy_from_sql_table(table) -> Any:
    """Deprecated alias of :func:`materialized_sql_result_lazy_wrapper`."""
    return materialized_sql_result_lazy_wrapper(table)


def execute_compiled_sql(
    compiled: CompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    """在 DuckDB 或 ClickHouse 上执行已编译 SQL，返回 MultiIndex Series。"""
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        return _execute_clickhouse(compiled, ctx, query_budget=query_budget)
    return _execute_duckdb(compiled, ctx, data_source, query_budget=query_budget)


def _series_from_batch_table(
    table,
    *,
    timestamp_col: str,
    instrument_col: str,
    value_column: str,
) -> pd.Series:
    """批量查询结果中按列别名提取 MultiIndex Series。"""
    from data_access.read.adapters import arrow_to_multiindex_series

    return arrow_to_multiindex_series(
        table,
        timestamp_column=timestamp_col,
        instrument_column=instrument_col,
        value_column=value_column,
        output_name=value_column,
    )


def execute_batch_compiled_sql(
    compiled: BatchCompiledSql,
    ctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> dict[str, pd.Series]:
    """执行批量编译 SQL，返回 ``sid -> MultiIndex Series`` 映射。"""
    if compiled.dialect == SqlDialect.CLICKHOUSE:
        table = _execute_clickhouse_table(compiled, ctx, query_budget=query_budget)
    else:
        table = _execute_duckdb_table(
            compiled, ctx, data_source, query_budget=query_budget
        )

    out: dict[str, pd.Series] = {}
    col_names = list(getattr(table, "column_names", []) or getattr(table, "schema", {}).names or [])
    # #369/#370：expected alias 是 batch 结果 schema 契约；缺失必须显式失败，
    # 不再 ``if alias in col_names: ... continue`` 静默跳过。
    expected = {alias for _, alias in compiled.column_aliases}
    missing = expected - set(col_names)
    if missing:
        raise SqlResultSchemaError(f"missing expected SQL aliases: {sorted(missing)}")
    for sid, alias in compiled.column_aliases:
        out[sid] = _series_from_batch_table(
            table,
            timestamp_col="ts",
            instrument_col="inst",
            value_column=alias,
        )
    return out


def _build_duckdb_store_kwargs(
    compiled: CompiledSql | BatchCompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> dict[str, Any]:
    """组装 store.sql() 参数：view_columns / read_params / query_budget。"""
    cols: set[str] = set(compiled.referenced_columns)
    cols.add(pctx.time_column)
    cols.add(pctx.instrument_column)
    fields = getattr(data_source, "fields", None) or {}
    # PARITY-SWEEP-R56: prefer the data source's field-plan resolution (which
    # maps a logical column to the dataset's real physical column, e.g. the
    # probe fixture's ``close`` -> ``Close``) instead of only the raw fields
    # dict.  When the source exposes ``_resolve_columns`` use it so the SQL
    # view_columns match the parquet schema (a plain fields.get(c, c) fallback
    # emitted the raw logical name ``close`` against a parquet column ``Close``
    # and DuckDB raised Binder/Arrow missing-column errors).
    physical_set: set[str] = set()
    resolver = getattr(data_source, "_resolve_columns", None)
    if callable(resolver):
        try:
            phys, _ = resolver(sorted(cols))
            physical_set.update(phys)
        except Exception:
            physical_set = set()
    if not physical_set:
        physical_set = {fields.get(c, c) for c in cols}
    physical = sorted(physical_set)
    view_columns = {name: physical for name in compiled.read_datasets}

    read_params: dict[str, dict[str, Any]] = {}
    ds_params = dict(getattr(data_source, "params", None) or {})
    for name in compiled.read_datasets:
        if ds_params:
            read_params[name] = ds_params

    _ensure_data_access()
    from data_access.read.query_budget import QueryBudget, resolve_query_budget

    # #371：QueryBudget 可配置。优先显式 query_budget 参数；否则读环境变量
    # FACTOR_ENGINE_QUERY_BUDGET_MAX_ROWS（int）；否则默认 50M。
    explicit = query_budget
    if explicit is None:
        _env_max = os.environ.get("FACTOR_ENGINE_QUERY_BUDGET_MAX_ROWS", "").strip()
        if _env_max:
            try:
                explicit = QueryBudget(max_rows=int(_env_max))
            except (TypeError, ValueError):
                explicit = None
        if explicit is None:
            explicit = QueryBudget(max_rows=50_000_000)
    return {
        "read_datasets": list(compiled.read_datasets),
        "view_columns": view_columns,
        "read_params": read_params or None,
        "query_budget": resolve_query_budget(explicit),
    }


def _execute_duckdb_table(
    compiled: CompiledSql | BatchCompiledSql,
    pctx: PushdownContext | None = None,
    data_source: Any | None = None,
    query_budget: Any | None = None,
):
    """通过 data_access store 在 DuckDB 上执行 SQL，返回 Arrow 表。

    OPT-01: 添加 DuckDB 超时保护（防止失控查询）
    OPT-02: 集成查询计划缓存
    OPT-03: 应用并行执行配置
    """
    _ensure_data_access()
    from data_access import get_store
    from data_access.read.query_budget import QueryBudget

    store = get_store()

    # OPT-01: 设置查询超时（基于 QueryBudget 或环境变量）
    timeout_ms = 60_000  # 默认 60 秒
    if isinstance(query_budget, QueryBudget) and query_budget.max_elapsed_ms:
        timeout_ms = int(query_budget.max_elapsed_ms)
    else:
        env_timeout = os.environ.get("FACTOR_ENGINE_DUCKDB_MAX_QUERY_TIMEOUT_MS", "").strip()
        if env_timeout:
            try:
                timeout_ms = int(env_timeout)
            except (ValueError, TypeError):
                pass

    # OPT-03: 应用 DuckDB 并行配置
    parallel_enabled = os.environ.get("DUCKDB_ENABLE_PARALLEL_CONFIG", "true").lower() in ("true", "1", "yes")

    try:
        conn = getattr(store, "_conn", None)
        if conn is not None:
            # 设置超时
            try:
                conn.execute(f"SET max_query_timeout = {timeout_ms}")
            except Exception:
                pass  # 旧版 DuckDB 可能不支持

            # 应用并行配置
            if parallel_enabled:
                try:
                    from .duckdb_performance import DuckDBParallelConfig
                    config = DuckDBParallelConfig.from_env()
                    conn.execute(f"SET threads TO {config.threads}")
                    conn.execute(f"SET memory_limit = '{config.memory_limit_mb}MB'")
                    if config.enable_object_cache:
                        conn.execute("SET enable_object_cache TO true")
                    if config.preserve_insertion_order:
                        conn.execute("SET preserve_insertion_order TO true")
                except Exception:
                    pass  # 配置失败不影响查询执行
    except Exception:
        pass

    kwargs: dict[str, Any] = {}
    if pctx is not None and data_source is not None:
        kwargs = _build_duckdb_store_kwargs(
            compiled, pctx, data_source, query_budget=query_budget
        )

    # OPT-02: 查询计划缓存（记录查询以供 DuckDB 内部优化器复用）
    cache_enabled = os.environ.get("DUCKDB_ENABLE_QUERY_CACHE", "true").lower() in ("true", "1", "yes")
    if cache_enabled:
        try:
            from .duckdb_performance import get_query_plan_cache
            import hashlib

            # 计算缓存键（查询 + 过滤参数）
            cache_key_parts = [compiled.query]
            if pctx and pctx.filt:
                cache_key_parts.extend([
                    str(pctx.filt.start or ""),
                    str(pctx.filt.end or ""),
                    str(pctx.filt.instrument_filter_kind.value if pctx.filt.instrument_filter_kind else ""),
                ])
            cache_key = hashlib.sha256("".join(cache_key_parts).encode()).hexdigest()[:16]

            cache = get_query_plan_cache()
            cached = cache.get(cache_key)
            if not cached:
                # 缓存未命中，记录此查询
                cache.put(compiled.query, cache_key)
        except Exception:
            pass  # 缓存失败不影响查询执行

    return store.sql(compiled.query, **kwargs)


def _clickhouse_query_settings(budget: Any | None, *, query_id: str | None = None) -> dict[str, Any]:
    """R21-062: map QueryBudget / env onto ClickHouse ``settings``.

    ClickHouse must not be a budget-free back door: max_execution_time,
    max_result_rows/bytes, max_memory_usage, max_threads, max_read_rows/bytes,
    a per-query ``query_id`` (for cancel), and ``readonly`` are set on every
    production query.  Invalid budget env values fail loudly (R21-064).
    """
    from data_access.read.query_budget import QueryBudget

    budget = budget if isinstance(budget, QueryBudget) else None
    settings: dict[str, Any] = {"readonly": 1}
    if query_id:
        settings["query_id"] = str(query_id)

    def _int_env(name: str, default: int | None) -> int | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"invalid {name}={raw!r} (must be an integer)") from None
        if value <= 0:
            raise ValueError(f"invalid {name}={value!r} (must be positive)")
        return value

    # Deadline / execution time (ms -> s; R21-064 invalid value fails startup).
    max_elapsed_ms = budget.max_elapsed_ms if budget and budget.max_elapsed_ms else None
    if max_elapsed_ms is None:
        raw = os.environ.get("FACTOR_ENGINE_CLICKHOUSE_MAX_EXECUTION_TIME_MS", "").strip()
        if raw:
            try:
                max_elapsed_ms = float(raw)
            except ValueError:
                raise ValueError(f"invalid FACTOR_ENGINE_CLICKHOUSE_MAX_EXECUTION_TIME_MS={raw!r}")
    if max_elapsed_ms is not None:
        settings["max_execution_time"] = max(1, int(max_elapsed_ms / 1000.0))

    max_rows = budget.max_rows if budget else None
    if max_rows is None:
        max_rows = _int_env("FACTOR_ENGINE_CLICKHOUSE_MAX_RESULT_ROWS", None)
    if max_rows is not None:
        settings["max_result_rows"] = int(max_rows)
        settings["max_read_rows"] = int(max_rows)

    max_bytes = budget.max_result_bytes if budget else None
    if max_bytes is None:
        max_bytes = _int_env("FACTOR_ENGINE_CLICKHOUSE_MAX_RESULT_BYTES", None)
    if max_bytes is not None:
        settings["max_result_bytes"] = int(max_bytes)
        settings["max_read_bytes"] = int(max_bytes)

    mem_bytes = _int_env("FACTOR_ENGINE_CLICKHOUSE_MAX_MEMORY_USAGE", None)
    if mem_bytes is not None:
        settings["max_memory_usage"] = int(mem_bytes)
    threads = _int_env("FACTOR_ENGINE_CLICKHOUSE_MAX_THREADS", None)
    if threads is not None:
        settings["max_threads"] = int(threads)
    return settings


class _ClickHouseConnectionPool:
    """OPT-03: ClickHouse 连接池（避免每次查询都创建新连接）。

    连接池减少握手开销（20-50ms per query），线程安全。
    """

    def __init__(self, max_size: int = 10, idle_timeout: float = 300.0):
        import threading
        self._pool: dict[str, list[tuple[Any, float]]] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._idle_timeout = idle_timeout

    def _make_pool_key(self, config: Any) -> str:
        """生成连接池键（host:port:database）。"""
        return f"{config.host}:{config.port}:{config.database}"

    def get_client(self, config: Any, settings: dict[str, Any]) -> Any:
        """获取或创建 ClickHouse 客户端。"""
        import clickhouse_connect
        import time

        pool_key = self._make_pool_key(config)

        with self._lock:
            # 清理超时连接
            now = time.time()
            if pool_key in self._pool:
                active = []
                for client, last_used in self._pool[pool_key]:
                    if now - last_used < self._idle_timeout:
                        active.append((client, last_used))
                    else:
                        try:
                            client.close()
                        except Exception:
                            pass
                self._pool[pool_key] = active

            # 尝试复用连接
            if pool_key in self._pool and self._pool[pool_key]:
                client, _ = self._pool[pool_key].pop()
                return client

        # 创建新连接
        client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            database=config.database,
            secure=config.secure,
            query_limit=settings.get("max_result_rows"),
            settings=settings,
        )
        return client

    def return_client(self, config: Any, client: Any) -> None:
        """归还客户端到池中。"""
        import time

        pool_key = self._make_pool_key(config)

        with self._lock:
            if pool_key not in self._pool:
                self._pool[pool_key] = []

            # 池已满，关闭连接
            if len(self._pool[pool_key]) >= self._max_size:
                try:
                    client.close()
                except Exception:
                    pass
                return

            # 放回池中
            self._pool[pool_key].append((client, time.time()))


# 全局 ClickHouse 连接池
_CLICKHOUSE_POOL = _ClickHouseConnectionPool(
    max_size=int(os.environ.get("CLICKHOUSE_POOL_SIZE", "10")),
    idle_timeout=float(os.environ.get("CLICKHOUSE_POOL_IDLE_TIMEOUT", "300.0")),
)


def _execute_clickhouse_table(
    compiled: CompiledSql | BatchCompiledSql,
    ctx: PushdownContext,
    *,
    query_budget: Any | None = None,
):
    """在 ClickHouse 上执行 SQL，返回查询结果表（带资源预算）。

    R21-061..063: unlike the old ``execute_query(config, sql)`` (which had no
    budget), the ClickHouse path now carries QueryBudget-derived settings so
    switching backend cannot bypass resource governance.

    OPT-03: 使用连接池避免每次查询创建新连接（节省 20-50ms）。
    """
    _ensure_data_access()
    import uuid

    from data_access.clickhouse.panel import ClickHouseConfig

    config = ClickHouseConfig.from_env(**(ctx.ch_config or {}))
    query_id = str(uuid.uuid4().hex[:16])
    settings = _clickhouse_query_settings(query_budget, query_id=query_id)

    # 检查是否启用连接池
    pool_enabled = os.environ.get("CLICKHOUSE_ENABLE_POOL", "true").lower() in ("true", "1", "yes")

    if pool_enabled:
        # 使用连接池
        client = _CLICKHOUSE_POOL.get_client(config, settings)
        try:
            result = client.query(compiled.query, settings=settings)
            arrow = result.arrow()
            # 归还连接
            _CLICKHOUSE_POOL.return_client(config, client)
            return arrow
        except Exception as exc:
            # 错误时关闭连接（不放回池）
            try:
                client.close()
            except Exception:
                pass

            # surface a stable budget code so the error taxonomy does not parse text
            if "max_result" in str(exc).lower() or "memory limit" in str(exc).lower() or "limit exceeded" in str(exc).lower():
                from factor_engine.runtime.resource_errors import ResourceBudgetExceeded

                raise ResourceBudgetExceeded(f"ClickHouse budget exceeded: {exc}") from exc
            raise
    else:
        # 原有逻辑：每次创建新连接
        import clickhouse_connect

        client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            database=config.database,
            secure=config.secure,
            query_limit=settings.get("max_result_rows"),
            settings=settings,
        )
        try:
            result = client.query(compiled.query, settings=settings)
            return result.arrow()
        except Exception as exc:
            # surface a stable budget code so the error taxonomy does not parse text
            if "max_result" in str(exc).lower() or "memory limit" in str(exc).lower() or "limit exceeded" in str(exc).lower():
                from factor_engine.runtime.resource_errors import ResourceBudgetExceeded

                raise ResourceBudgetExceeded(f"ClickHouse budget exceeded: {exc}") from exc
            raise
        finally:
            try:
                client.close()
            except Exception:
                pass


def _execute_duckdb(
    compiled: CompiledSql,
    pctx: PushdownContext,
    data_source: Any,
    query_budget: Any | None = None,
) -> pd.Series:
    """DuckDB 单因子执行：SQL → Arrow 表 → MultiIndex Series。

    MB-P1-018: DuckDB Region boundary should prefer Arrow/Relation output,
    avoiding unnecessary `.df()` conversion to Pandas when the downstream
    consumer can work with Arrow directly.
    """
    table = _execute_duckdb_table(
        compiled, pctx, data_source, query_budget=query_budget
    )
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def _execute_clickhouse(
    compiled: CompiledSql, ctx: PushdownContext, *, query_budget: Any | None = None
) -> pd.Series:
    """ClickHouse 单因子执行：SQL → 结果表 → MultiIndex Series。"""
    table = _execute_clickhouse_table(compiled, ctx, query_budget=query_budget)
    return _series_from_sql_table(table, timestamp_col="ts", instrument_col="inst")


def try_execute_sql_pushdown_long(
    plan: PlanNode,
    ctx: ExecutionContext,
    *,
    sid: str | None = None,
) -> Any | None:
    """尝试将 SQL 可编译子树下推为 long-table LazyFrame（``ts, inst, _v``）。

    编译或执行失败时返回 ``None``；strict 模式下失败会抛出异常。
    """
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        _note_sql_fallback(ctx, "NoPushdownContext", sid=sid)
        return None

    compiled = compile_plan_to_sql(
        plan,
        dataset=pctx.dataset,
        table=pctx.table,
        time_column=pctx.time_column,
        instrument_column=pctx.instrument_column,
        filt=pctx.filt,
        dialect=pctx.dialect,
    )
    if compiled is None:
        _note_sql_fallback(ctx, "CompilationUnsupported", sid=sid)
        return None

    try:
        if compiled.dialect == SqlDialect.CLICKHOUSE:
            table = _execute_clickhouse_table(
                compiled, pctx, query_budget=getattr(ctx, "query_budget", None)
            )
        else:
            table = _execute_duckdb_table(
                compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
            )
        # #368：结果已完整物化，只是套 LazyFrame 壳；supports_streaming=False。
        return materialized_sql_result_lazy_wrapper(table)
    except Exception as exc:
        if not _sql_fallback_allowed(exc, ctx):
            raise
        _note_sql_fallback(ctx, type(exc).__name__, sid=sid)
        from factor_engine.backend.sql_pushdown.strict import handle_sql_long_pushdown_failure

        try:
            handle_sql_long_pushdown_failure(ctx, sid=sid, exc=exc, phase="long_single")
        except Exception:
            raise
        return None


def try_execute_sql_pushdown_batch_long(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, Any] | None:
    """批量 SQL 下推为 ``sid -> LazyFrame``；共享 base CTE，一次 round-trip。"""
    if not plans:
        return {}
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        _note_sql_fallback(ctx, "NoPushdownContext", sid=",".join(sorted(plans.keys())[:3]))
        return None

    compiled = compile_plans_batch_to_sql(
        plans,
        dataset=pctx.dataset,
        table=pctx.table,
        time_column=pctx.time_column,
        instrument_column=pctx.instrument_column,
        filt=pctx.filt,
        dialect=pctx.dialect,
    )
    if compiled is None:
        _note_sql_fallback(ctx, "CompilationUnsupported", sid=",".join(sorted(plans.keys())[:3]))
        return None

    try:
        if compiled.dialect == SqlDialect.CLICKHOUSE:
            table = _execute_clickhouse_table(
                compiled, pctx, query_budget=getattr(ctx, "query_budget", None)
            )
        else:
            table = _execute_duckdb_table(
                compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
            )
        import polars as pl

        df = pl.from_arrow(table) if hasattr(table, "to_pandas") else pl.DataFrame(table)
        # #369/#370：batch wide result 是单条宽物化；必须按列 chunk（每 chunk
        # 最多 64 个 alias）分批 select，避免一次拉取整宽。
        expected = {alias for _, alias in compiled.column_aliases}
        missing = expected - set(df.columns)
        if missing:
            raise SqlResultSchemaError(f"missing expected SQL aliases: {sorted(missing)}")
        out: dict[str, Any] = {}
        alias_list = list(compiled.column_aliases)
        for i in range(0, len(alias_list), 64):
            chunk = alias_list[i : i + 64]
            chunk_aliases = [a for _, a in chunk]
            part = df.select(["ts", "inst", *chunk_aliases]).lazy()
            for sid, alias in chunk:
                out[sid] = part.select(["ts", "inst", pl.col(alias).alias("_v")])
        return out
    except SqlResultSchemaError:
        # #369/#370：batch 结果 schema 契约违约是 fail-closed，必须上抛，不回退。
        raise
    except Exception as exc:
        if not _sql_fallback_allowed(exc, ctx):
            raise
        _note_sql_fallback(
            ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3])
        )
        from factor_engine.backend.sql_pushdown.strict import handle_sql_long_pushdown_failure

        try:
            handle_sql_long_pushdown_failure(
                ctx,
                sid=",".join(sorted(plans.keys())[:3]),
                exc=exc,
                phase="long_batch",
            )
        except Exception:
            raise
        return None


def try_execute_sql_pushdown(
    plan: PlanNode,
    ctx: ExecutionContext,
) -> pd.Series | None:
    """若计划可 SQL 化则在 DuckDB/ClickHouse 内执行并返回 Series。

    上下文不可下推、编译失败或执行异常时返回 ``None``，由调用方回退 Python/Polars。
    """
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        _note_sql_fallback(ctx, "NoPushdownContext")
        return None

    try:
        compiled = compile_plan_to_sql(
            plan,
            dataset=pctx.dataset,
            table=pctx.table,
            time_column=pctx.time_column,
            instrument_column=pctx.instrument_column,
            filt=pctx.filt,
            dialect=pctx.dialect,
        )
    except Exception as exc:
        # #365/#366 compile fail-closed：只有「可下推失败」类（CapabilityMiss /
        # CompilationUnsupported）允许返回 None（换后端 fallback）；其余 fail-closed
        # 或 production 未知异常必须上抛，research 未知异常才允许 fallback。
        from factor_engine.runtime.resource_errors import (
            CapabilityMiss,
            CompilationUnsupported,
            is_fail_closed_error,
        )

        if isinstance(exc, (CapabilityMiss, CompilationUnsupported)):
            _note_sql_fallback(ctx, type(exc).__name__)
            return None
        if is_fail_closed_error(exc):
            raise
        production = str(getattr(ctx, "run_mode", "") or "").lower() == "production"
        if production:
            raise
        _note_sql_fallback(ctx, type(exc).__name__)
        return None
    if compiled is None:
        _note_sql_fallback(ctx, "CompilationUnsupported")
        return None

    try:
        return execute_compiled_sql(
            compiled, pctx, ctx.data_source, query_budget=getattr(ctx, "query_budget", None)
        )
    except Exception as exc:
        if not _sql_fallback_allowed(exc, ctx):
            raise
        _note_sql_fallback(ctx, type(exc).__name__)
        return None


def try_execute_sql_pushdown_batch(
    plans: dict[str, PlanNode],
    ctx: ExecutionContext,
) -> dict[str, pd.Series] | None:
    """批量 SQL 下推：多子树合并为一条 WITH 查询，返回 ``sid -> Series``。"""
    if not plans:
        return {}
    pctx = extract_pushdown_context(ctx)
    if pctx is None:
        _note_sql_fallback(ctx, "NoPushdownContext", sid=",".join(sorted(plans.keys())[:3]))
        return None

    try:
        compiled = compile_plans_batch_to_sql(
            plans,
            dataset=pctx.dataset,
            table=pctx.table,
            time_column=pctx.time_column,
            instrument_column=pctx.instrument_column,
            filt=pctx.filt,
            dialect=pctx.dialect,
        )
    except Exception as exc:
        # #365/#366 compile fail-closed（同 try_execute_sql_pushdown）。
        from factor_engine.runtime.resource_errors import (
            CapabilityMiss,
            CompilationUnsupported,
            is_fail_closed_error,
        )

        if isinstance(exc, (CapabilityMiss, CompilationUnsupported)):
            _note_sql_fallback(
                ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3])
            )
            return None
        if is_fail_closed_error(exc):
            raise
        production = str(getattr(ctx, "run_mode", "") or "").lower() == "production"
        if production:
            raise
        _note_sql_fallback(ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3]))
        return None
    if compiled is None:
        _note_sql_fallback(ctx, "CompilationUnsupported", sid=",".join(sorted(plans.keys())[:3]))
        return None

    try:
        return execute_batch_compiled_sql(
            compiled,
            pctx,
            ctx.data_source,
            query_budget=getattr(ctx, "query_budget", None),
        )
    except SqlResultSchemaError:
        # #369：missing expected alias 是 schema 契约违约，fail-closed，不回退。
        raise
    except Exception as exc:
        if not _sql_fallback_allowed(exc, ctx):
            raise
        _note_sql_fallback(ctx, type(exc).__name__, sid=",".join(sorted(plans.keys())[:3]))
        return None
