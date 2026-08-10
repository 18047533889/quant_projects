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

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True)
class QueryBudget:
    """单次 read/sql 扫描预算（R25 §26：远程对象与扫描字节预算）。

    - ``max_scan_files``         本地/远程匹配文件数上限（旧）
    - ``max_scan_objects``       exact source objects 上限（R25 P0-010：wildcard
                                 resolved 后的真实对象数，不是 len(FileVersion)=1）
    - ``max_scan_bytes``         estimated/actual 扫描字节上限（R25 P0-009/010）
    - ``max_remote_list_objects``COS LIST 返回对象数上限（R25 §26/§72）
    - ``max_remote_requests``    远程 HEAD/LIST 请求数上限（R25 §26）
    - ``max_estimated_memory``   预估内存上限（R25 §26）
    """

    max_rows: int | None = None
    max_result_bytes: int | None = None
    max_elapsed_ms: float | None = None
    max_scan_files: int | None = None
    require_columns: bool = False
    require_time_range: bool = False
    # R25 §26：远程对象与扫描字节预算
    max_scan_objects: int | None = None
    max_scan_bytes: int | None = None
    max_remote_list_objects: int | None = None
    max_remote_requests: int | None = None
    max_estimated_memory: int | None = None

    def __post_init__(self) -> None:
        """#7 公开 ``QueryBudget`` 自身也做 invariant validation（复用
        ``DatasetQueryPolicy`` 的规则：positive int / positive finite float /
        bool reject）。

        旧代码 programmatic ``QueryBudget(max_rows=-1, max_elapsed_ms=nan)`` 绕过
        parser 校验——``elapsed > nan`` 恒 False，等价于把 deadline check 关掉；
        负数 / bool 预算也一样静默失效。现在构造即 fail-closed。
        """
        if self.max_rows is not None:
            _positive_int(self.max_rows, context="QueryBudget.max_rows")
        if self.max_result_bytes is not None:
            _positive_int(self.max_result_bytes, context="QueryBudget.max_result_bytes")
        if self.max_elapsed_ms is not None:
            _positive_finite_float(
                self.max_elapsed_ms, context="QueryBudget.max_elapsed_ms"
            )
        if self.max_scan_files is not None:
            _positive_int(self.max_scan_files, context="QueryBudget.max_scan_files")
        for key in (
            "max_scan_objects",
            "max_scan_bytes",
            "max_remote_list_objects",
            "max_remote_requests",
            "max_estimated_memory",
        ):
            val = getattr(self, key)
            if val is not None:
                _positive_int(val, context=f"QueryBudget.{key}")
        if not isinstance(self.require_columns, bool):
            raise ValidationError(
                f"QueryBudget.require_columns 必须是布尔值，收到 {self.require_columns!r}"
            )
        if not isinstance(self.require_time_range, bool):
            raise ValidationError(
                f"QueryBudget.require_time_range 必须是布尔值，"
                f"收到 {self.require_time_range!r}"
            )


@dataclass(frozen=True)
class DatasetQueryPolicy:
    """datasets.yaml 中 per-dataset 读策略（与全局 QueryBudget 合并取更严）。"""

    require_explicit_columns: bool = False
    require_time_range: bool = False
    max_rows: int | None = None
    max_result_bytes: int | None = None
    max_elapsed_ms: float | None = None
    max_scan_files: int | None = None
    # R25 §26
    max_scan_objects: int | None = None
    max_scan_bytes: int | None = None
    max_remote_list_objects: int | None = None
    max_remote_requests: int | None = None
    max_estimated_memory: int | None = None


def _strict_bool(value: object, *, context: str) -> bool:
    """Parse YAML/env booleans without accepting truthy arbitrary strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValidationError(f"{context} 必须是布尔值 true/false")


def _positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{context} 必须是正整数")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} 必须是正整数") from exc
    if out <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValidationError(f"{context} 必须是正整数")
    return out


def _positive_finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        # #P0-final closure 8：``True`` 是 bool 不是数值预算——之前 float(True)==1.0
        # 静默通过，把安全预算悄悄变成 1ms。
        raise ValidationError(f"{context} 必须是有限正数")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} 必须是有限正数") from exc
    if not math.isfinite(out) or out <= 0:
        raise ValidationError(f"{context} 必须是有限正数")
    return out


# #P0-final closure 8：QueryPolicy 是 strict typed config——unknown key 拒绝
# （``max_scan_file:`` 这种 typo 之前被静默忽略，真实 ``max_scan_files=None``，
# 安全预算直接消失）。
_QUERY_POLICY_KEYS = frozenset({
    "require_explicit_columns", "require_time_range", "max_rows",
    "max_result_bytes", "max_elapsed_ms", "max_scan_files",
    "max_scan_objects", "max_scan_bytes", "max_remote_list_objects",
    "max_remote_requests", "max_estimated_memory",
})


def parse_dataset_query_policy(raw: object, *, context: str) -> DatasetQueryPolicy:
    """解析 YAML ``query_policy`` 块。"""
    if raw is None:
        return DatasetQueryPolicy()
    if not isinstance(raw, dict):
        raise ValidationError(
            f"{context}: query_policy 必须是 mapping，收到 {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _QUERY_POLICY_KEYS)
    if unknown:
        raise ValidationError(
            f"{context}: query_policy 含未知 key {unknown}；"
            f"应为 {sorted(_QUERY_POLICY_KEYS)} 之一"
        )

    def _opt_int(key: str) -> int | None:
        val = raw.get(key)
        return None if val is None else _positive_int(val, context=f"{context}: query_policy.{key}")

    def _opt_float(key: str) -> float | None:
        val = raw.get(key)
        return None if val is None else _positive_finite_float(val, context=f"{context}: query_policy.{key}")

    return DatasetQueryPolicy(
        require_explicit_columns=_strict_bool(
            raw.get("require_explicit_columns", False),
            context=f"{context}: query_policy.require_explicit_columns",
        ),
        require_time_range=_strict_bool(
            raw.get("require_time_range", False),
            context=f"{context}: query_policy.require_time_range",
        ),
        max_rows=_opt_int("max_rows"),
        max_result_bytes=_opt_int("max_result_bytes"),
        max_elapsed_ms=_opt_float("max_elapsed_ms"),
        max_scan_files=_opt_int("max_scan_files"),
        max_scan_objects=_opt_int("max_scan_objects"),
        max_scan_bytes=_opt_int("max_scan_bytes"),
        max_remote_list_objects=_opt_int("max_remote_list_objects"),
        max_remote_requests=_opt_int("max_remote_requests"),
        max_estimated_memory=_opt_int("max_estimated_memory"),
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
        max_scan_files=_tighter_int(budget.max_scan_files, policy.max_scan_files),
        max_scan_objects=_tighter_int(budget.max_scan_objects, policy.max_scan_objects),
        max_scan_bytes=_tighter_int(budget.max_scan_bytes, policy.max_scan_bytes),
        max_remote_list_objects=_tighter_int(
            budget.max_remote_list_objects, policy.max_remote_list_objects
        ),
        max_remote_requests=_tighter_int(
            budget.max_remote_requests, policy.max_remote_requests
        ),
        max_estimated_memory=_tighter_int(
            budget.max_estimated_memory, policy.max_estimated_memory
        ),
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


def is_strict_semantics() -> bool:
    """#P0-11 唯一 fail-closed 语义开关：production OR strict_read OR automated_research。

    所有 required/allowed filter / PIT / schema / cardinality / snapshot 的
    fail-closed 判定都用它，不再有的地方只查 ``_production_mode()``、有的地方
    查 ``_production_mode() or _strict_read_mode()`` 两套漂移。

    R25 P0-020（INV-07）：``automated_research``（AlphaProbe / LLM mining 默认）
    在 PIT / calendar / semantic ambiguity / required filters / units / source
    snapshot / authorization 上**必须接近 production strict**——机器不看 warning，
    宽松语义会静默污染搜索空间。预算可以更宽，语义不允许放宽。
    """
    if _production_mode() or _strict_read_mode():
        return True
    try:
        from data_access.security.run_mode import resolve_run_mode

        return resolve_run_mode().strict_semantics
    except Exception:
        return False


# 生产/严格读模式的**最低保障线**（#P0-14）：显式传入的宽松 QueryBudget 永远不能
# 把 production 默认放得更松——生产基线是 floor，显式 budget 只能在其上收紧。
_PRODUCTION_FLOOR = QueryBudget(
    max_rows=50_000_000,
    require_columns=True,
    require_time_range=False,
)


def merge_production_floor(budget: QueryBudget) -> QueryBudget:
    """把显式 budget 与生产/严格读 floor 合并，**取更严**（fail-closed）。"""
    return QueryBudget(
        max_rows=_tighter_int(budget.max_rows, _PRODUCTION_FLOOR.max_rows),
        max_result_bytes=_tighter_int(
            budget.max_result_bytes, _PRODUCTION_FLOOR.max_result_bytes
        ),
        max_elapsed_ms=_tighter_float(
            budget.max_elapsed_ms, _PRODUCTION_FLOOR.max_elapsed_ms
        ),
        max_scan_files=_tighter_int(
            budget.max_scan_files, _PRODUCTION_FLOOR.max_scan_files
        ),
        max_scan_objects=_tighter_int(
            budget.max_scan_objects, _PRODUCTION_FLOOR.max_scan_objects
        ),
        max_scan_bytes=_tighter_int(
            budget.max_scan_bytes, _PRODUCTION_FLOOR.max_scan_bytes
        ),
        max_remote_list_objects=_tighter_int(
            budget.max_remote_list_objects, _PRODUCTION_FLOOR.max_remote_list_objects
        ),
        max_remote_requests=_tighter_int(
            budget.max_remote_requests, _PRODUCTION_FLOOR.max_remote_requests
        ),
        max_estimated_memory=_tighter_int(
            budget.max_estimated_memory, _PRODUCTION_FLOOR.max_estimated_memory
        ),
        require_columns=budget.require_columns or _PRODUCTION_FLOOR.require_columns,
        require_time_range=(
            budget.require_time_range or _PRODUCTION_FLOOR.require_time_range
        ),
    )


def resolve_query_budget(budget: QueryBudget | None = None) -> QueryBudget:
    """解析有效预算。

    #P0-14 显式传入的 budget 在 production/strict 下**不能反向放宽**默认限制：
    ``QueryBudget(max_rows=None, require_columns=False)`` 会把生产默认 50m /
    require_columns 直接取消——现在显式 budget 先与 production floor 合并取更严。
    只有非生产/非 strict 模式下才原样信任显式 budget。
    """
    strict = _production_mode() or _strict_read_mode()
    if budget is not None:
        return merge_production_floor(budget) if strict else budget
    if strict:
        return _PRODUCTION_FLOOR
    default_max_rows = os.environ.get("DATA_ACCESS_DEFAULT_MAX_ROWS")
    if default_max_rows is not None and str(default_max_rows).strip():
        try:
            max_rows = _positive_int(default_max_rows.strip(), context="DATA_ACCESS_DEFAULT_MAX_ROWS")
        except ValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "DATA_ACCESS_DEFAULT_MAX_ROWS 必须是正整数"
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


def enforce_scan_file_budget(budget: QueryBudget, *, file_count: int) -> None:
    """扫描前对匹配文件数做硬限制。"""
    if budget.max_scan_files is not None and file_count > budget.max_scan_files:
        raise ValidationError(
            f"扫描匹配 {file_count} 个文件，超过预算上限 {budget.max_scan_files}。"
            "请缩小 time_range / instrument_filter 或提高 max_scan_files。"
        )


def enforce_scan_object_budget(budget: QueryBudget, *, object_count: int) -> None:
    """R25 P0-009/010：**真实** source object 数硬限制（不是 len(FileVersion)）。

    wildcard 解析成 exact objects 后按实际数量卡——remote 10000 个 object 不能被
    「一个 wildcard URI 算 1 个 FileVersion」绕过 max_scan_files/objects。
    """
    if budget.max_scan_objects is not None and object_count > budget.max_scan_objects:
        raise ValidationError(
            f"source snapshot 解析出 {object_count} 个 exact objects，超过预算上限 "
            f"{budget.max_scan_objects}（R25 P0-010：wildcard 按真实对象数计）。"
            "请缩小 time_range / 使用更窄前缀 / 提高 max_scan_objects。"
        )
    if budget.max_scan_files is not None and object_count > budget.max_scan_files:
        raise ValidationError(
            f"source snapshot 解析出 {object_count} 个 exact objects，超过预算上限 "
            f"{budget.max_scan_files}（wildcard 展开后按真实文件数计）。"
            "请缩小 time_range / 提高 max_scan_files。"
        )


def enforce_scan_byte_budget(budget: QueryBudget, *, scan_bytes: int) -> None:
    """R25 §26/P0-009：estimated/actual 扫描字节硬限制。"""
    if budget.max_scan_bytes is not None and scan_bytes > budget.max_scan_bytes:
        raise ValidationError(
            f"source snapshot 扫描字节 {scan_bytes} 超过预算上限 {budget.max_scan_bytes}"
            "（R25 §26）。请缩小 time_range / 列选择 / 提高 max_scan_bytes。"
        )


def enforce_remote_request_budget(
    budget: QueryBudget, *, remote_requests: int
) -> None:
    """R25 §26/§72：远程 HEAD/LIST 请求数硬限制。"""
    if budget.max_remote_requests is not None and remote_requests > budget.max_remote_requests:
        raise ValidationError(
            f"远程请求数 {remote_requests} 超过预算上限 {budget.max_remote_requests}"
            "（R25 §26）。请优先使用 source manifest 或更窄前缀。"
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
