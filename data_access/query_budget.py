"""
data_access.query_budget —— 读路径查询预算（行数 / 字节 / 耗时 / 列与时间窗约束）

生产环境可通过 ``QUANT_PRODUCTION_MODE=1``、``DATA_ACCESS_STRICT_READ=1``
或显式传入 ``QueryBudget`` 收紧扫描面。

职责边界：
    - ``query_budget``：硬拦截（超预算抛 ValidationError）
    - ``telemetry``：软观测（慢查询 warning，不阻塞）
    - ``audit``：追责日志
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .exceptions import ValidationError

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True)
class QueryBudget:
    """单次 read/sql 扫描预算。"""

    max_rows: int | None = None
    max_result_bytes: int | None = None
    max_elapsed_ms: float | None = None
    require_columns: bool = False
    require_time_range: bool = False


@dataclass(frozen=True)
class DatasetQueryPolicy:
    """datasets.yaml 中 per-dataset 读策略（与全局 QueryBudget 合并取更严）。"""

    require_explicit_columns: bool = False
    require_time_range: bool = False
    max_rows: int | None = None
    max_result_bytes: int | None = None
    max_elapsed_ms: float | None = None


def parse_dataset_query_policy(raw: object, *, context: str) -> DatasetQueryPolicy:
    """解析 YAML ``query_policy`` 块。"""
    if raw is None:
        return DatasetQueryPolicy()
    if not isinstance(raw, dict):
        raise ValidationError(
            f"{context}: query_policy 必须是 mapping，收到 {type(raw).__name__}"
        )

    def _opt_int(key: str) -> int | None:
        val = raw.get(key)
        if val is None:
            return None
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValidationError(f"{context}: query_policy.{key} 必须是整数")
        return int(val)

    def _opt_float(key: str) -> float | None:
        val = raw.get(key)
        if val is None:
            return None
        try:
            out = float(val)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{context}: query_policy.{key} 必须是数字"
            ) from exc
        return out

    return DatasetQueryPolicy(
        require_explicit_columns=bool(raw.get("require_explicit_columns", False)),
        require_time_range=bool(raw.get("require_time_range", False)),
        max_rows=_opt_int("max_rows"),
        max_result_bytes=_opt_int("max_result_bytes"),
        max_elapsed_ms=_opt_float("max_elapsed_ms"),
    )


def _tighter_int(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _tighter_float(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def merge_dataset_policy(
    budget: QueryBudget,
    policy: DatasetQueryPolicy | None,
) -> QueryBudget:
    """将 dataset 级策略合并进有效预算（更严者优先）。"""
    if policy is None:
        return budget
    return QueryBudget(
        max_rows=_tighter_int(budget.max_rows, policy.max_rows),
        max_result_bytes=_tighter_int(budget.max_result_bytes, policy.max_result_bytes),
        max_elapsed_ms=_tighter_float(budget.max_elapsed_ms, policy.max_elapsed_ms),
        require_columns=budget.require_columns or policy.require_explicit_columns,
        require_time_range=budget.require_time_range or policy.require_time_range,
    )


def merge_dataset_policies(
    budget: QueryBudget,
    policies: Sequence[DatasetQueryPolicy],
) -> QueryBudget:
    """多 dataset sql() 场景：逐个合并，取全局最严。"""
    merged = budget
    for policy in policies:
        merged = merge_dataset_policy(merged, policy)
    return merged


def _production_mode() -> bool:
    fe = os.environ.get("FACTOR_ENGINE_RUN_MODE", "").strip().lower()
    if fe == "production":
        return True
    return os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}


def _strict_read_mode() -> bool:
    return os.environ.get("DATA_ACCESS_STRICT_READ", "").lower() in {"1", "true", "yes"}


def resolve_query_budget(budget: QueryBudget | None = None) -> QueryBudget:
    """解析有效预算；显式传入优先，否则生产/严格读模式默认更严。"""
    if budget is not None:
        return budget
    if _production_mode() or _strict_read_mode():
        return QueryBudget(
            max_rows=50_000_000,
            require_columns=True,
            require_time_range=False,
        )
    default_max_rows = os.environ.get("DATA_ACCESS_DEFAULT_MAX_ROWS")
    if default_max_rows is not None and str(default_max_rows).strip():
        try:
            max_rows = int(default_max_rows)
        except ValueError as exc:
            raise ValidationError(
                "DATA_ACCESS_DEFAULT_MAX_ROWS 必须是整数"
            ) from exc
        return QueryBudget(max_rows=max_rows)
    return QueryBudget()


def validate_query_request(
    budget: QueryBudget,
    *,
    columns: list[str] | tuple[str, ...] | None,
    time_range: tuple[object, object] | None,
) -> None:
    """读前校验：列与时间窗约束。"""
    if budget.require_columns and not columns:
        raise ValidationError(
            "生产/严格读模式要求显式指定 columns，避免宽表全扫。"
            "可设置 QUANT_PRODUCTION_MODE=0 / DATA_ACCESS_STRICT_READ=0，"
            "或传入 query_budget=QueryBudget(require_columns=False)。"
        )
    if budget.require_time_range and time_range is None:
        raise ValidationError(
            "当前 QueryBudget 要求显式 time_range。"
        )


def validate_sql_view_columns(
    budget: QueryBudget,
    read_datasets: Sequence[str],
    view_columns: Mapping[str, Sequence[str]] | None,
) -> None:
    """sql() 读前校验：TEMP VIEW 底层列约束（避免 SELECT * 全扫）。"""
    if not budget.require_columns:
        return
    if not view_columns:
        raise ValidationError(
            "生产/严格读模式要求 sql() 显式指定 view_columns，"
            "避免 TEMP VIEW 使用 SELECT * 全扫宽表。"
            "示例：view_columns={'factors': ['datetime', 'asset', 'value']}。"
        )
    missing = [name for name in read_datasets if not view_columns.get(name)]
    if missing:
        raise ValidationError(
            f"view_columns 缺少数据集 {missing!r} 的列清单；"
            "严格模式下每个 read_datasets 都必须显式列。"
        )


def enforce_result_budget(
    budget: QueryBudget,
    *,
    rows: int,
    elapsed_ms: float,
) -> None:
    """读后对行数与耗时做硬限制。"""
    if budget.max_rows is not None and rows > budget.max_rows:
        raise ValidationError(
            f"查询结果行数 {rows} 超过预算上限 {budget.max_rows}。"
            "请缩小 time_range / instrument_filter 或提高 max_rows。"
        )
    if budget.max_elapsed_ms is not None and elapsed_ms > budget.max_elapsed_ms:
        raise ValidationError(
            f"查询耗时 {elapsed_ms:.1f}ms 超过预算上限 {budget.max_elapsed_ms:.1f}ms。"
        )


def enforce_stream_budget(
    budget: QueryBudget,
    *,
    total_rows: int,
    total_bytes: int,
    elapsed_ms: float,
) -> None:
    """流式读累计行数/字节/耗时硬限制。"""
    enforce_result_budget(budget, rows=total_rows, elapsed_ms=elapsed_ms)
    if budget.max_result_bytes is not None and total_bytes > budget.max_result_bytes:
        raise ValidationError(
            f"查询结果累计字节 {total_bytes} 超过预算上限 {budget.max_result_bytes}。"
            "请缩小扫描范围、指定更少列，或提高 max_result_bytes。"
        )


def enforce_arrow_budget(
    budget: QueryBudget,
    table: pa.Table,
    *,
    elapsed_ms: float,
) -> None:
    """一次性 materialize 结果的行数/字节/耗时硬限制。"""
    enforce_result_budget(budget, rows=table.num_rows, elapsed_ms=elapsed_ms)
    if budget.max_result_bytes is not None and table.nbytes > budget.max_result_bytes:
        raise ValidationError(
            f"查询结果字节 {table.nbytes} 超过预算上限 {budget.max_result_bytes}。"
            "请缩小扫描范围、指定更少列，或提高 max_result_bytes。"
        )


def collect_polars_with_budget(
    lf: Any,
    *,
    query_budget: QueryBudget | None = None,
) -> pa.Table:
    """Polars LazyFrame collect 后强制读后预算。"""
    import time

    budget = resolve_query_budget(query_budget)
    start = time.perf_counter()
    table = lf.collect().to_arrow()
    enforce_arrow_budget(
        budget,
        table,
        elapsed_ms=(time.perf_counter() - start) * 1000,
    )
    return table


def apply_sql_row_limit(query: str, max_rows: int | None) -> str:
    """对用户 SQL 外层追加最大行数限制；仅对简单末尾 ``LIMIT N`` 做 min 合并。"""
    if max_rows is None or max_rows <= 0:
        return query.strip().rstrip(";")
    stripped = query.strip().rstrip(";")
    import re

    match = re.search(r"\bLIMIT\s+(\d+)\s*$", stripped, flags=re.IGNORECASE)
    if match:
        existing = int(match.group(1))
        bound = min(existing, int(max_rows))
        if bound == existing:
            return stripped
        return re.sub(
            r"\bLIMIT\s+\d+\s*$",
            f"LIMIT {bound}",
            stripped,
            flags=re.IGNORECASE,
        )
    bound = int(max_rows)
    return f"SELECT * FROM ({stripped}) AS __da_bounded LIMIT {bound}"
