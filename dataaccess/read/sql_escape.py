"""
data_access.sql_escape —— 有限的 SQL 逃生口

职责：
    1. 把用户的 SELECT 查询跑在已注册数据集上，让临时分析场景也能用 data_access 能力
    2. 保证"窄口":
       - 只读：任何 DML/DDL/COPY/ATTACH/PRAGMA 一律拒
       - read_datasets 必填：SQL 里 FROM 的表名只能是这组 registry 里的 dataset
       - 禁用 file functions：用户 SQL 里不准出现 read_parquet / read_csv / COPY 等字面量
       - 必审计：每次调用都往 JSONL 日志里记一行

非职责：
    - 不提供写 SQL（INSERT/UPDATE 等）。要改数据走 write_arrow / upsert / publish_from_staging
    - 不做 query planning / 优化 —— DuckDB 自己会做
    - 不做结果分页、流式返回 —— 一次性返回 Arrow Table；大结果靠 time_range 切

设计参考：
    docs/data_access 历史设计说明已收敛到 data_access/docs/用户使用手册.md 与 README.md。
    路径白名单、审计日志、未来的预算/限流都要靠这条收敛。所以这个模块的口子
    只开到「能覆盖 80% 的临时分析需求，但绝不让用户碰到裸 read_parquet」。

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import threading
import time
import uuid
import inspect
from collections.abc import Iterator
from typing import Any, Mapping, Sequence

import pyarrow as pa

from data_access.core import audit
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import EngineError, ValidationError
from data_access.registry.paths import PathAuthorizer
from .query_budget import (
    QueryBudget,
    apply_sql_row_limit,
    enforce_arrow_budget,
    enforce_stream_budget,
    resolve_query_budget,
    validate_sql_view_columns,
)
from data_access.registry import DatasetRegistry


logger = logging.getLogger("data_access.sql_escape")


# 禁止出现的关键字（大小写不敏感，整词匹配）。
# 只挡明显的写/DDL/文件入口。只读 SELECT 里即便用了 INSERT 作为列名也不受影响，
# 因为我们要求用 " " 引起来（DuckDB SQL 规范），正则匹配的是裸关键字。
_FORBIDDEN_PATTERNS = [
    # 写/DDL
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bMERGE\b",
    r"\bCREATE\b", r"\bDROP\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bCOPY\b", r"\bATTACH\b", r"\bDETACH\b", r"\bEXPORT\b",
    r"\bPRAGMA\b", r"\bINSTALL\b", r"\bLOAD\b", r"\bCALL\b", r"\bEXECUTE\b",
    r"\bSET\b", r"\bEXPORT\b", r"\bIMPORT\b",
    # 文件/外部表入口
    r"\bread_parquet\b", r"\bread_csv\b", r"\bread_csv_auto\b",
    r"\bread_json\b", r"\bread_ndjson\b", r"\bread_json_auto\b",
    r"\bparquet_scan\b", r"\bcsv_scan\b",
    r"\bglob\b",
]

_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)


def _make_scope_id() -> str:
    return uuid.uuid4().hex[:8]


def _build_view_map(read_datasets: Sequence[str], scope_id: str) -> dict[str, str]:
    """dataset 名 → 唯一 TEMP VIEW 名（避免并发 sql 抢 catalog）。"""
    return {name: f"__da_{scope_id}_{name}" for name in read_datasets}


def _rewrite_query_tables(query: str, view_map: dict[str, str]) -> str:
    """Replace dataset placeholders only outside SQL strings and comments."""
    missing = set(view_map)
    out: list[str] = []
    i = 0
    state = "code"
    while i < len(query):
        if state == "code":
            if query.startswith("--", i):
                state = "line_comment"
                out.append("--")
                i += 2
                continue
            if query.startswith("/*", i):
                state = "block_comment"
                out.append("/*")
                i += 2
                continue
            if query[i] == "'":
                state = "string"
                out.append(query[i])
                i += 1
                continue
            if query[i] == '"':
                state = "quoted_ident"
                out.append(query[i])
                i += 1
                continue
            if query[i] == "{" and i + 1 < len(query) and query[i + 1] == "{":
                end = query.find("}}", i + 2)
                if end >= 0:
                    name = query[i + 2:end]
                    if name in view_map:
                        out.append(view_map[name])
                        missing.discard(name)
                        i = end + 2
                        continue
            out.append(query[i])
            i += 1
            continue
        if state == "string":
            out.append(query[i])
            if query[i] == "'":
                if i + 1 < len(query) and query[i + 1] == "'":
                    out.append(query[i + 1])
                    i += 2
                    continue
                state = "code"
            i += 1
            continue
        if state == "quoted_ident":
            out.append(query[i])
            if query[i] == '"':
                if i + 1 < len(query) and query[i + 1] == '"':
                    out.append(query[i + 1])
                    i += 2
                    continue
                state = "code"
            i += 1
            continue
        if state == "line_comment":
            out.append(query[i])
            if query[i] == "\n":
                state = "code"
            i += 1
            continue
        if query.startswith("*/", i):
            out.append("*/")
            i += 2
            state = "code"
        else:
            out.append(query[i])
            i += 1

    if missing:
        raise ValidationError(
            "sql() 现在要求用 {{dataset}} 引用 read_datasets，避免裸表名重写误伤。"
            f"缺少占位符: {sorted(missing)}. "
            "示例: SELECT * FROM {{factor_lake}} WHERE datetime >= ?"
        )
    return "".join(out)


def _resolver_supports_time_range(resolve_paths) -> bool:
    try:
        signature = inspect.signature(resolve_paths)
    except (TypeError, ValueError):
        return True
    return "time_range" in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def _resolve_paths_compat(resolve_paths, ds, ds_params, time_range, *, supports_time_range):
    """Call path resolvers without masking TypeError raised by their body."""
    if supports_time_range:
        return resolve_paths(ds, ds_params, time_range=time_range)
    return resolve_paths(ds, ds_params)

def _prepare_sql_views(
    *,
    registry: DatasetRegistry,
    read_datasets: Sequence[str],
    read_params: Mapping[str, Mapping[str, Any]],
    view_columns: Mapping[str, Sequence[str]] | None,
    scope_id: str,
    build_select_sql,
    resolve_paths,
    read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
    semantic_gate=None,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """返回 (view_map, register_specs)；register_specs = [(view_name, inlined_sql), ...]。

    ``resolve_paths`` 签名兼容：
      ``(ds, params)`` 或 ``(ds, params, *, time_range=...)``
    后者用于 COS remote：按 time_range 生成 ``s3://`` 日文件列表。

    ``semantic_gate``（#P0-12）：store 注入的 ``_prepare_read_request`` 包装。
    每个 read dataset 在构建视图**之前**先过完整 semantic gate（temporal model /
    event cutoff / required filters / allowed values）——sql() 不能绕过 generic
    read() 会拒绝的 E1/E2/RAW_EVENT/X0/effective-time-only。
    """
    view_map = _build_view_map(read_datasets, scope_id)
    time_ranges = dict(read_time_ranges or {})
    register_specs: list[tuple[str, str]] = []
    supports_time_range = _resolver_supports_time_range(resolve_paths)
    for name in read_datasets:
        ds = registry.get(name)
        ds_params = dict(read_params.get(name, {}))
        time_range = time_ranges.get(name)
        cols = list(view_columns[name]) if view_columns and name in view_columns else None
        if semantic_gate is not None:
            # #P0-12 语义门禁：失败即拒绝（fail-closed），视图不会生成。
            semantic_gate(
                name,
                columns=cols,
                time_range=time_range,
                params=ds_params,
            )
        paths = _resolve_paths_compat(
            resolve_paths, ds, ds_params, time_range,
            supports_time_range=supports_time_range,
        )
        sql, sql_params = build_select_sql(
            ds=ds,
            paths=paths,
            columns=cols,
            time_range=time_range,
            instrument_filter=None,
        )
        inlined_sql = _inline_path_params(sql, sql_params)
        register_specs.append((view_map[name], inlined_sql))
    return view_map, register_specs


def run_sql(
    *,
    registry: DatasetRegistry,
    authorizer: PathAuthorizer,
    engine: DuckDBEngine,
    query: str,
    read_datasets: Sequence[str],
    read_params: Mapping[str, Mapping[str, Any]] | None = None,
    read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
    view_columns: Mapping[str, Sequence[str]] | None = None,
    params: Sequence[Any] | None = None,
    query_budget: QueryBudget | None = None,
    build_select_sql,  # store._build_select_sql，避开循环 import
    resolve_paths,     # store._prepare_dataset_read 包装，同上
    semantic_gate=None,  # #P0-12 store._prepare_read_request 包装
) -> pa.Table:
    """执行一条只读 SELECT；只能 FROM 预先声明的数据集。

    参数：
        query: 用户提供的 SELECT，FROM 的表名必须在 read_datasets 里
        read_datasets: 本次查询会用到的数据集名列表（必填非空）
        read_params: {dataset_name: {**param}}，给参数化数据集（如 factor_lake
            要传 factor_id）。缺 dataset 的参数会在注册 view 时 raise
        read_time_ranges: {dataset_name: (start, end)}，用于 COS remote 按日
            选文件，以及 view 级 time_column 过滤
        params: 用户 SQL 里的 ? 绑定参数
        build_select_sql / resolve_paths: 从 store.DataAccessStore 注入；
            两个内部方法分别负责组装"合规 SELECT"和"路径白名单校验"

    返回：
        pa.Table，用户 SQL 的执行结果

    raise：
        ValidationError: query 含禁词、read_datasets 空、read_datasets 里某个
            名字不在 registry、参数化 dataset 缺参数
        EngineError: DuckDB 执行失败（通常是 SQL 语法错 / 列不存在 / 类型不兼容）
    """
    if not query or not query.strip():
        raise ValidationError("sql 查询不能为空")
    if not read_datasets:
        raise ValidationError(
            "sql() 必须显式声明 read_datasets，不允许不指定数据集就查询。"
            "这是为了让路径白名单与审计能归到具体 dataset。"
        )

    _reject_forbidden_tokens(query)
    _reject_denied_table_functions(query)
    # #P1-final closure 23：parser 级 allowlist——恰好一个 SELECT/WITH + 函数
    # allowlist（table/file/network function 一律拒绝，未知函数 strict fail-closed）。
    _assert_single_select_statement(query)
    try:
        from data_access.read.query_budget import is_strict_semantics

        strict = is_strict_semantics()
    except Exception:
        strict = True
    _check_sql_function_allowlist(query, strict=strict)

    read_params = dict(read_params) if read_params else {}
    budget = resolve_query_budget(query_budget)
    validate_sql_view_columns(budget, read_datasets, view_columns)

    scope_id = _make_scope_id()
    view_map, register_specs = _prepare_sql_views(
        registry=registry,
        read_datasets=read_datasets,
        read_params=read_params,
        view_columns=view_columns,
        scope_id=scope_id,
        build_select_sql=build_select_sql,
        resolve_paths=resolve_paths,
        read_time_ranges=read_time_ranges,
        semantic_gate=semantic_gate,
    )
    bounded_query = apply_sql_row_limit(
        _rewrite_query_tables(query, view_map),
        budget.max_rows,
    )

    audit_fields: dict[str, Any] = {
        "datasets": list(read_datasets),
        "read_params": dict(read_params) or None,
        "query": query[:500],
        "scope_id": scope_id,
    }

    rows = 0
    err_msg: str | None = None
    ok = False
    start = time.perf_counter()
    result_table: pa.Table | None = None

    try:
        result_table = engine.execute_scoped_sql_arrow(
            register_specs,
            bounded_query,
            params,
            deadline_ms=budget.max_elapsed_ms,
        )
        rows = result_table.num_rows
        elapsed_ms = (time.perf_counter() - start) * 1000
        enforce_arrow_budget(budget, result_table, elapsed_ms=elapsed_ms)
        ok = True
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        audit.record(
            op="sql",
            dataset=",".join(read_datasets),
            ok=ok,
            rows=rows,
            elapsed_ms=elapsed_ms,
            error=err_msg,
            extra=audit_fields,
        )

    assert result_table is not None
    logger.info(
        "sql rows=%d datasets=%s elapsed_ms=%.1f",
        rows, ",".join(read_datasets),
        (time.perf_counter() - start) * 1000,
    )
    return result_table


def run_sql_stream(
    *,
    registry: DatasetRegistry,
    authorizer: PathAuthorizer,
    engine: DuckDBEngine,
    query: str,
    read_datasets: Sequence[str],
    read_params: Mapping[str, Mapping[str, Any]] | None = None,
    read_time_ranges: Mapping[str, tuple[Any, Any] | None] | None = None,
    view_columns: Mapping[str, Sequence[str]] | None = None,
    params: Sequence[Any] | None = None,
    query_budget: QueryBudget | None = None,
    batch_size: int = 100_000,
    build_select_sql,
    resolve_paths,
    semantic_gate=None,
) -> Iterator[pa.RecordBatch]:
    """流式执行只读 SELECT；语义与 ``run_sql`` 一致，按 batch 返回。"""
    if not query or not query.strip():
        raise ValidationError("sql 查询不能为空")
    if not read_datasets:
        raise ValidationError("sql_stream() 必须显式声明 read_datasets")

    _reject_forbidden_tokens(query)
    _reject_denied_table_functions(query)
    # #P1-final closure 23：parser 级 allowlist
    _assert_single_select_statement(query)
    try:
        from data_access.read.query_budget import is_strict_semantics

        strict = is_strict_semantics()
    except Exception:
        strict = True
    _check_sql_function_allowlist(query, strict=strict)

    read_params = dict(read_params) if read_params else {}
    budget = resolve_query_budget(query_budget)
    validate_sql_view_columns(budget, read_datasets, view_columns)

    scope_id = _make_scope_id()
    view_map, register_specs = _prepare_sql_views(
        registry=registry,
        read_datasets=read_datasets,
        read_params=read_params,
        view_columns=view_columns,
        scope_id=scope_id,
        build_select_sql=build_select_sql,
        resolve_paths=resolve_paths,
        read_time_ranges=read_time_ranges,
        semantic_gate=semantic_gate,
    )
    bounded_query = apply_sql_row_limit(
        _rewrite_query_tables(query, view_map),
        budget.max_rows,
    )

    audit_fields: dict[str, Any] = {
        "datasets": list(read_datasets),
        "read_params": dict(read_params) or None,
        "query": query[:500],
        "stream": True,
        "scope_id": scope_id,
    }

    total_rows = 0
    total_bytes = 0
    err_msg: str | None = None
    ok = False
    start = time.perf_counter()

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        nonlocal total_rows, total_bytes, ok, err_msg
        try:
            batch_iter = engine.execute_scoped_sql_stream(
                register_specs,
                bounded_query,
                params,
                batch_size=batch_size,
                # #P0-41 max_elapsed_ms 下推到流式执行：第一批数据永远不返回时
                # watchdog 也能取消，不再只等第一批出来才查 elapsed。
                deadline_ms=budget.max_elapsed_ms,
            )
            for batch in batch_iter:
                if batch.num_rows == 0:
                    continue
                total_rows += batch.num_rows
                total_bytes += batch.nbytes
                enforce_stream_budget(
                    budget,
                    total_rows=total_rows,
                    total_bytes=total_bytes,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
                yield batch
            ok = True
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            audit.record(
                op="sql",
                dataset=",".join(read_datasets),
                ok=ok,
                rows=total_rows,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=err_msg,
                extra=audit_fields,
            )

    return _iter_batches()


def _iter_code_chunks(query: str) -> Iterator[str]:
    """#P2-2 只产出 SQL「code」段（跳过字符串/引号标识符/注释），
    避免 ``SELECT 'read_parquet' AS note`` 这类字符串里的关键字被误伤。"""
    i = 0
    state = "code"
    chunk_start = 0
    while i < len(query):
        if state == "code":
            if query.startswith("--", i):
                yield query[chunk_start:i]
                state = "line_comment"
                i += 2
                chunk_start = i
                continue
            if query.startswith("/*", i):
                yield query[chunk_start:i]
                state = "block_comment"
                i += 2
                chunk_start = i
                continue
            if query[i] in "'\"":
                yield query[chunk_start:i]
                quote = query[i]
                state = "string" if quote == "'" else "quoted_ident"
                i += 1
                chunk_start = i
                continue
            i += 1
            continue
        if state == "string":
            if query[i] == "'":
                if i + 1 < len(query) and query[i + 1] == "'":
                    i += 2
                    continue
                state = "code"
                chunk_start = i + 1
            i += 1
            continue
        if state == "quoted_ident":
            if query[i] == '"':
                if i + 1 < len(query) and query[i + 1] == '"':
                    i += 2
                    continue
                state = "code"
                chunk_start = i + 1
            i += 1
            continue
        if state == "line_comment":
            if query[i] == "\n":
                state = "code"
                chunk_start = i + 1
            i += 1
            continue
        if state == "block_comment":
            if query.startswith("*/", i):
                state = "code"
                i += 2
                chunk_start = i
            else:
                i += 1
    if chunk_start < len(query):
        yield query[chunk_start:]


def _reject_forbidden_tokens(query: str) -> None:
    """扫用户 SQL 的 **code 段**里是否有禁用关键字。

    #P2-2 字符串/注释里的关键字不误伤（``SELECT 'read_parquet' AS note`` 合法）。

    注意：这道防线的目的是"笔误/无意绕过"——真正的恶意场景靠「只开 TEMP VIEW
    不开 read_parquet」+ 下一层 table-function deny-by-default（见
    ``_reject_denied_table_functions``）。
    """
    for chunk in _iter_code_chunks(query):
        m = _FORBIDDEN_RE.search(chunk)
        if m:
            raise ValidationError(
                f"sql() 拒绝执行：查询含禁用关键字 {m.group(0)!r}。"
                f"sql() 只允许 SELECT/WITH；写操作请走 write_arrow / upsert / "
                f"publish_from_staging；读外部文件请先在 datasets.yaml 登记"
            )


# #P0-13 deny-by-default 的 table function 名单：保留作快速预检与降级兜底。
# 主机制是 #P1-final closure 23 的 **parser 级 allowlist**（见下）——从 DuckDB
# 自身 ``duckdb_functions()`` 注册表按 function_type 分类，任何 table 型函数
# （含未来新增的 read_*/scan/glob/httpfs 等）自动拒绝，不再靠补字符串名单。
_DENIED_TABLE_FUNCTIONS = (
    "read_parquet",
    "read_parquet_auto",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "parquet_scan",
    "csv_scan",
    "glob",
    "read_blob",
    "httpfs_",
    "sqlite_scan",
    "delta_scan",
    "iceberg_scan",
    "postgres_scan",
    "mysql_scan",
    "sql_scan",
)


# #P1-final closure 23：SQL **语法关键字**（grammar forms），虽然形如 ``name(``
# 但不是函数调用——``IN (``、``FILTER (``、``OVER (``、``EXTRACT (`` 等。
# 不在 DuckDB ``duckdb_functions()`` 注册表里、但作为 SQL 语法的一部分是安全的。
_SQL_GRAMMAR_KEYWORDS = frozenset({
    # 子句关键字（后跟括号但不是函数）
    "IN", "ON", "OVER", "PARTITION", "FILTER", "GROUP", "ORDER", "HAVING",
    "WHERE", "LIMIT", "FROM", "JOIN", "VALUES", "UNION", "INTERSECT",
    "EXCEPT", "DISTINCT", "AND", "OR", "NOT", "BETWEEN", "LIKE", "ILIKE",
    "WHEN", "THEN", "ELSE", "END", "CASE", "AS", "USING", "WITH", "SELECT",
    "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS", "NATURAL",
    "RECURSIVE", "WINDOW", "ROWS", "RANGE", "GROUPS", "PRECEDING",
    "FOLLOWING", "CURRENT", "LATERAL", "OFFSET", "FETCH", "FIRST", "LAST",
    "TIES", "ONLY", "UNBOUNDED", "COLLATE", "IS", "NULL", "TRUE", "FALSE",
    "NULLS", "ASC", "DESC", "BOTH", "LEADING", "TRAILING",
    # 特殊语法形式（有括号、DuckDB 当作 grammar 而非注册函数）
    "CAST", "EXTRACT", "COALESCE", "IFNULL", "NULLIF", "REPLACE",
    "TRANSLATE", "SUBSTRING", "TRIM", "LTRIM", "RTRIM", "POSITION",
    "DATE_PART", "EXTRACT_EPOCH", "GROUPING",
})


# #P1-final closure 23：函数 allowlist 的运行时数据源——DuckDB 自身注册表
# （``duckdb_functions()``），懒加载并进程内缓存。key = 函数名小写，
# value = 该函数名出现过的 function_type 集合（scalar/aggregate/window/table/...）。
_function_registry: dict[str, set[str]] | None = None
_function_registry_lock = threading.Lock()


def _load_function_registry() -> dict[str, set[str]]:
    """从 DuckDB ``duckdb_functions()`` 加载函数类型注册表（进程内缓存）。

    失败（无 duckdb / 版本无该函数）回退空 dict——allowlist 检查退化为
    deny-list 兜底（见 ``_check_sql_function_allowlist`` 的降级分支）。
    """
    global _function_registry
    if _function_registry is not None:
        return _function_registry
    with _function_registry_lock:
        if _function_registry is not None:
            return _function_registry
        out: dict[str, set[str]] = {}
        try:
            import duckdb

            con = duckdb.connect()
            try:
                rows = con.execute(
                    "SELECT function_name, function_type FROM duckdb_functions()"
                ).fetchall()
            finally:
                con.close()
            for name, ftype in rows:
                out.setdefault(str(name).lower(), set()).add(str(ftype).lower())
        except Exception:
            out = {}
        _function_registry = out
        return out


def _find_function_calls(query: str) -> list[str]:
    """在 code 段里找出形如 ``name(`` 的函数调用名（跳过字符串/注释/标识符引号）。

    只认「非语法关键字 + 后跟左括号」的标识符。限定名 ``schema.func(...)`` 取
    最后一段。``FILTER (`` / ``IN (`` / ``OVER (`` 等语法关键字被排除。
    """
    import re as _re

    out: list[str] = []
    for chunk in _iter_code_chunks(query):
        for m in _re.finditer(
            r"(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
            chunk,
        ):
            name = m.group("fn").upper()
            if name in _SQL_GRAMMAR_KEYWORDS:
                continue
            out.append(name.lower())
    return out


def _check_sql_function_allowlist(query: str, *, strict: bool) -> None:
    """#P1-final closure 23：**parser 级函数 allowlist**。

    对每个函数调用名分类：
      - 在 deny 名单（read_*/scan/glob/httpfs 等）→ 拒绝；
      - 在 DuckDB 注册表且**只含 table 型** overload → 拒绝（table function =
        文件/网络/外部源入口，含未来新增的 parquet_metadata/sniff_csv/
        read_json_objects 等，注册表自动分类，不再补字符串名单）；
      - 在 DuckDB 注册表且含 scalar/aggregate/window overload → 放行；
      - **不在注册表**且不是 grammar 关键字 → 未知函数：strict 拒绝（可能是
        未来版本的文件函数 / 自定义 UDF，fail-closed），research 放行（deny
        名单仍挡已知危险项）。
    """
    calls = _find_function_calls(query)
    if not calls:
        return
    registry = _load_function_registry()
    for fn in calls:
        if any(fn.startswith(d) or fn == d for d in _DENIED_TABLE_FUNCTIONS):
            raise ValidationError(
                f"sql() 拒绝执行：file/network/table function {fn!r} 默认禁用。"
                "读外部文件请先在 datasets.yaml 登记数据集。"
            )
        types = registry.get(fn)
        if types is None:
            # 不在 DuckDB 注册表：可能是语法关键字或 UDF
            if fn.upper() in _SQL_GRAMMAR_KEYWORDS:
                continue
            if strict:
                raise ValidationError(
                    f"sql() 拒绝执行：函数 {fn!r} 不在 DuckDB 允许的 "
                    "scalar/aggregate/window 集合内（未知函数，strict fail-closed）。"
                    "若确为内建函数，请升级 DuckDB 或改用 registered dataset。"
                )
            continue
        # 注册表里只有 table 型 overload → 文件/网络/外部源入口
        if types and not (types & {"scalar", "aggregate", "window"}):
            raise ValidationError(
                f"sql() 拒绝执行：{fn!r} 是 table function（{sorted(types)}）——"
                "文件/网络/外部源入口默认禁用。读外部文件请先在 datasets.yaml 登记数据集。"
            )


def _reject_denied_table_functions(query: str) -> None:
    """#P0-13 对 code 段做 table-function deny-by-default 检查（快速预检）。

    识别 ``name(`` 形态（后跟括号即函数调用）；任何名单内函数直接拒绝。真正的
    防线是 #P1-final closure 23 的 ``_check_sql_function_allowlist``——注册表
    分类覆盖未来新增的 file/network/table function；本函数保留作快速失败 + 无
    DuckDB 注册表时的降级兜底。
    """
    import re as _re

    code = " ".join(_iter_code_chunks(query))
    for fn in _DENIED_TABLE_FUNCTIONS:
        if _re.search(_re.escape(fn) + r"\s*\(", code, flags=_re.IGNORECASE):
            raise ValidationError(
                f"sql() 拒绝执行：table function {fn!r} 默认禁用（deny-by-default）。"
                "读外部文件请先在 datasets.yaml 登记数据集。"
            )


def _assert_single_select_statement(query: str) -> None:
    """#P1-final closure 23：从**语法层**证明恰好一个 SELECT/WITH 语句。

    在 code 段（跳过字符串/注释/引号标识符）按顶层 ``;`` 切分（含括号深度——
    CTE/子查询里的 ``;`` 会因深度 >0 被忽略）。多个语句 / 尾随 SQL → 拒绝；
    单个语句尾随一个 ``;`` 允许。
    """
    depth = 0
    stmts: list[str] = []
    buf: list[str] = []
    for chunk in _iter_code_chunks(query):
        for ch in chunk:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            if ch == ";" and depth == 0:
                stmts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
    stmts.append("".join(buf))
    live = [s.strip() for s in stmts if s.strip()]
    if len(live) > 1:
        raise ValidationError(
            f"sql() 拒绝执行：检测到 {len(live)} 条语句，只允许单个 SELECT/WITH。"
            "写操作/多语句请拆开执行。"
        )
    if live:
        first = live[0].split(None, 1)[0].upper() if live[0] else ""
        if first not in {"SELECT", "WITH"}:
            raise ValidationError(
                f"sql() 只允许单个 SELECT/WITH，收到 {first!r} 开头的语句"
            )


def validate_sql_sandbox(query: str) -> None:
    """sql_relation（#7）等直接执行 SQL 的入口复用的沙箱校验。

    比 ``sql()`` 的 TEMP VIEW 治理弱（无法约束 FROM 表集合），但至少在
    production/strict 下挡住：
        - 空查询 / 非 SELECT/WITH 开头（多语句、DDL）；
        - 禁用关键字（INSERT/UPDATE/DELETE/COPY/ATTACH/LOAD/INSTALL/
          read_parquet/read_csv/glob 等）——只在 code 段扫（#P2-2）；
        - deny-by-default 的 table function（#P0-13）。
    """
    if not query or not query.strip():
        raise ValidationError("sql_relation 拒绝执行空查询")
    # #P1-final closure 23：语法层证明**恰好一个** SELECT/WITH（而非只看首 token）
    _assert_single_select_statement(query)
    _reject_forbidden_tokens(query)
    _reject_denied_table_functions(query)
    try:
        from data_access.read.query_budget import is_strict_semantics

        strict = is_strict_semantics()
    except Exception:
        strict = True
    # parser 级函数 allowlist：table/file/network function 一律拒绝；未知函数
    # strict fail-closed。
    _check_sql_function_allowlist(query, strict=strict)


def _query_from_tables(query: str) -> list[str]:
    """#P0-18 提取 SQL 里 FROM/JOIN 的表名（code 段，跳过字符串/注释）。

    返回表名列表（含 ``{{name}}`` 占位与引号标识符去引号）。
    """
    import re

    out: list[str] = []
    for chunk in _iter_code_chunks(query):
        for m in re.finditer(
            r"(?i)\b(?:FROM|JOIN)\s+"
            r"(?:\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}"
            r"|\"([^\"]+)\""
            r"|([A-Za-z_][A-Za-z0-9_.]*))",
            chunk,
        ):
            name = m.group(1) or m.group(2) or m.group(3)
            if name and name not in out:
                out.append(name)
    return out


def assert_sql_tables_declared(query: str, declared: Sequence[str]) -> None:
    """#P0-18 production sql_relation 的 FROM 表集合绑定。

    ``snapshot_datasets`` 不再是纯 lineage——查询 FROM/JOIN 的每一张表都必须在
    声明集合内（token 黑名单只能挡 read_parquet/COPY 这类函数，挡不住用户
    ``FROM 其它系统表`` 绕过 DataAccess）。
    """
    if not query or not declared:
        return
    table_set = set(declared)
    # 允许大小写不敏感匹配（数据集名在 registry 是 snake_case）
    lower_declared = {str(d).lower() for d in table_set}
    unknown: list[str] = []
    for t in _query_from_tables(query):
        if t.lower() in lower_declared:
            continue
        if t.lower().startswith("__scope") or t.startswith("_"):
            # run_sql 作用域视图 / 内部临时名
            continue
        unknown.append(t)
    if unknown:
        raise ValidationError(
            f"sql_relation FROM/JOIN 引用了未声明的表：{sorted(set(unknown))}。"
            "production/strict 下必须全部出现在 snapshot_datasets 里。"
        )


def assert_sql_from_scope(query: str, *, allowed: Sequence[str]) -> None:
    """#34 收官轮：RelationHandle 的 FROM/JOIN 表集合必须全部落在允许 scope 内。

    ``allowed`` 如 ``("_sub",)``。任何其它**裸表引用**——未声明 dataset / DuckDB
    系统表（information_schema / pg_catalog）/ TEMP VIEW / 已存在普通 catalog
    表——都拒绝。不能借 ``_sub`` 合法存在的同时再读一个没有声明的数据源（仅检查
    「SQL 里出现了 FROM _sub」挡不住 ``JOIN some_other_table``）。子查询里的裸表
    引用同样被扫到（不允许藏进嵌套子查询）。

    表函数调用（``FROM read_parquet(?)`` 等，名字后紧跟 ``(``）不是裸表引用——
    由独立的函数 allowlist（``_check_sql_function_allowlist``）治理。

    与 strict 无关：数据源边界是安全/治理问题，research 也不能放开（读到的数据
    不在 snapshot/lineage 里，治理与追溯同时失效）。
    """
    import re

    allowed_l = {str(a).lower() for a in allowed}
    for chunk in _iter_code_chunks(query):
        for m in re.finditer(
            r"(?i)\b(?:FROM|JOIN)\s+"
            r"(?:\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}"
            r"|\"([^\"]+)\""
            r"|([A-Za-z_][A-Za-z0-9_.]*))",
            chunk,
        ):
            name = m.group(1) or m.group(2) or m.group(3)
            if not name:
                continue
            # 函数调用：FROM read_parquet(...)——名字后紧跟 ( 的不是表引用。
            rest = chunk[m.end():].lstrip()
            if rest.startswith("("):
                continue
            tl = name.lower()
            if tl in allowed_l:
                continue
            # 下划线开头是内部临时名/子查询别名（_sub 本身已在 allowed 里）。
            # 只放行明确的内部 scope 前缀，不放行任意 _foo（避免用户借 _ 前缀绕过）。
            if tl.startswith("__scope"):
                continue
            raise ValidationError(
                f"RelationHandle SQL 引用了 scope 外的表：{name!r}。完整 SELECT 只能 "
                f"FROM/JOIN {sorted(allowed_l)}；其它表一律拒绝（避免读取未声明的数据源"
                "绕过 DataAccess 治理）。"
            )


def _inline_path_params(sql: str, params: Sequence[Any]) -> str:
    """把参数化 SQL 的 ? 替换成 DuckDB 字面量。

    只在 sql_escape 里用，因为 CREATE VIEW AS 不支持 ? 绑定。
    params 只会是从 registry 来的路径字符串（或者 list[str]），其它类型视为错。

    string → 'string'（单引号 escape）
    list[string] → ['s1', 's2']
    其它类型 → raise，保证调用面被限死
    """
    parts: list[str] = []
    param_iter = iter(params)
    i = 0
    while i < len(sql):
        c = sql[i]
        if c == "'":
            # 进入字符串字面量，原样拷贝到匹配的单引号（兼容 '' 转义）
            parts.append(c)
            i += 1
            while i < len(sql):
                parts.append(sql[i])
                if sql[i] == "'":
                    # '' 是转义，两个都属于当前字面量
                    if i + 1 < len(sql) and sql[i + 1] == "'":
                        parts.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "?":
            try:
                value = next(param_iter)
            except StopIteration as exc:
                raise EngineError(
                    f"sql inline: SQL 里的 ? 数量多于 params（{len(params)} 个）"
                ) from exc
            parts.append(_format_literal(value))
            i += 1
            continue
        parts.append(c)
        i += 1
    # 检查 params 是不是都用光了
    remaining = list(param_iter)
    if remaining:
        raise EngineError(
            f"sql inline: params 有 {len(remaining)} 个未消费；SQL 和 params 数量对不上"
        )
    return "".join(parts)


def _format_literal(value: Any) -> str:
    """把 registry 来的值渲染成 DuckDB SQL 字面量。

    支持：str / list / bool / int / float / datetime.date / datetime.datetime /
    pandas.Timestamp（end-of-day 展开会生成 Timestamp）。
    """
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, list) and all(
        isinstance(x, (str, int, float)) for x in value
    ):
        inner = ", ".join(_format_literal(x) for x in value)
        return f"[{inner}]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    if isinstance(value, _dt.datetime):
        return f"TIMESTAMP '{value.isoformat()}'"
    if isinstance(value, _dt.date):
        return f"DATE '{value.isoformat()}'"
    # pandas.Timestamp / pd.NaT 等
    try:
        ts = value.to_pydatetime()
        return f"TIMESTAMP '{ts.isoformat()}'"
    except AttributeError:
        pass
    raise EngineError(
        f"sql inline 拒绝：参数类型 {type(value).__name__} 不在允许范围。"
        f"sql_escape 只让 registry 来的路径/时间/标量 inline；其它参数走用户 SQL 自己的 ? 绑定"
    )
