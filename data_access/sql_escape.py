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
    docs/data_access/10_架构设计.md §1.4 给出的"为什么不无限制开 sql":
    路径白名单、审计日志、未来的预算/限流都要靠这条收敛。所以这个模块的口子
    只开到「能覆盖 80% 的临时分析需求，但绝不让用户碰到裸 read_parquet」。

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from typing import Any, Mapping, Sequence

import pyarrow as pa

from . import audit
from .engine import DuckDBEngine
from .exceptions import EngineError, ValidationError
from .paths import PathAuthorizer
from .query_budget import (
    QueryBudget,
    apply_sql_row_limit,
    enforce_arrow_budget,
    enforce_stream_budget,
    resolve_query_budget,
    validate_sql_view_columns,
)
from .registry import DatasetRegistry


logger = logging.getLogger("data_access.sql_escape")


# 禁止出现的关键字（大小写不敏感，整词匹配）。
# 只挡明显的写/DDL/文件入口。只读 SELECT 里即便用了 INSERT 作为列名也不受影响，
# 因为我们要求用 " " 引起来（DuckDB SQL 规范），正则匹配的是裸关键字。
_FORBIDDEN_PATTERNS = [
    # 写/DDL
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bMERGE\b",
    r"\bCREATE\b", r"\bDROP\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bCOPY\b", r"\bATTACH\b", r"\bDETACH\b", r"\bEXPORT\b",
    r"\bPRAGMA\b", r"\bINSTALL\b", r"\bLOAD\b",
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
    """只替换 ``{{dataset}}`` 占位符，避免误改字符串字面量、注释、列别名。"""
    rewritten = query
    missing: list[str] = []

    for dataset_name, view_name in view_map.items():
        token = f"{{{{{dataset_name}}}}}"
        if token in rewritten:
            rewritten = rewritten.replace(token, view_name)
        else:
            missing.append(dataset_name)

    if missing:
        raise ValidationError(
            "sql() 现在要求用 {{dataset}} 引用 read_datasets，避免裸表名重写误伤。"
            f"缺少占位符: {missing}. "
            "示例: SELECT * FROM {{factor_lake}} WHERE datetime >= ?"
        )

    return rewritten


def _prepare_sql_views(
    *,
    registry: DatasetRegistry,
    read_datasets: Sequence[str],
    read_params: Mapping[str, Mapping[str, Any]],
    view_columns: Mapping[str, Sequence[str]] | None,
    scope_id: str,
    build_select_sql,
    resolve_paths,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """返回 (view_map, register_specs)；register_specs = [(view_name, inlined_sql), ...]。"""
    view_map = _build_view_map(read_datasets, scope_id)
    register_specs: list[tuple[str, str]] = []
    for name in read_datasets:
        ds = registry.get(name)
        ds_params = read_params.get(name, {})
        paths = resolve_paths(ds, ds_params)
        cols = list(view_columns[name]) if view_columns and name in view_columns else None
        sql, sql_params = build_select_sql(
            ds=ds,
            paths=paths,
            columns=cols,
            time_range=None,
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
    view_columns: Mapping[str, Sequence[str]] | None = None,
    params: Sequence[Any] | None = None,
    query_budget: QueryBudget | None = None,
    build_select_sql,  # store._build_select_sql，避开循环 import
    resolve_paths,     # store._resolve_paths，同上
) -> pa.Table:
    """执行一条只读 SELECT；只能 FROM 预先声明的数据集。

    参数：
        query: 用户提供的 SELECT，FROM 的表名必须在 read_datasets 里
        read_datasets: 本次查询会用到的数据集名列表（必填非空）
        read_params: {dataset_name: {**param}}，给参数化数据集（如 factor_lake
            要传 factor_id）。缺 dataset 的参数会在注册 view 时 raise
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
            register_specs, bounded_query, params
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
    view_columns: Mapping[str, Sequence[str]] | None = None,
    params: Sequence[Any] | None = None,
    query_budget: QueryBudget | None = None,
    batch_size: int = 100_000,
    build_select_sql,
    resolve_paths,
) -> Iterator[pa.RecordBatch]:
    """流式执行只读 SELECT；语义与 ``run_sql`` 一致，按 batch 返回。"""
    if not query or not query.strip():
        raise ValidationError("sql 查询不能为空")
    if not read_datasets:
        raise ValidationError("sql_stream() 必须显式声明 read_datasets")

    _reject_forbidden_tokens(query)
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


def _reject_forbidden_tokens(query: str) -> None:
    """扫用户 SQL 里是否有禁用关键字。

    注意：正则只挡最常见的 DML/DDL/文件函数；真要绕（比如 `ins` + `ert`
    字符串拼接）DuckDB parser 也会接受。这道防线的目的是"笔误/无意绕过"，
    真正的恶意场景得靠「只开 TEMP VIEW 不开 read_parquet」的设计本身兜底。
    """
    m = _FORBIDDEN_RE.search(query)
    if m:
        raise ValidationError(
            f"sql() 拒绝执行：查询含禁用关键字 {m.group(0)!r}。"
            f"sql() 只允许 SELECT/WITH；写操作请走 write_arrow / upsert / "
            f"publish_from_staging；读外部文件请先在 datasets.yaml 登记"
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
    """把 registry 来的值渲染成 DuckDB SQL 字面量。只支持 str / list[str]。"""
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        inner = ", ".join(_format_literal(x) for x in value)
        return f"[{inner}]"
    raise EngineError(
        f"sql inline 拒绝：参数类型 {type(value).__name__} 不在允许范围。"
        f"sql_escape 只让 registry 来的路径字符串 inline；其它参数走用户 SQL 自己的 ? 绑定"
    )
