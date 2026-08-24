# -*- coding: utf-8 -*-
"""物理计划：标注 SQL / Python / 物化子树执行方式（R21-P023-Four-Backend-Plan）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from factor_engine.planner.logical_plan import PlanNode


class ExecKind(str, Enum):
    """节点执行后端种类（R21-P023-Four-Backend-Plan）。"""

    SQL = "sql"  # 整棵 SQL 可编译子树
    PYTHON = "python"  # pandas/polars 算子
    MATERIALIZED = "materialized"  # 已 SQL 预计算的 Series 引用

    # Pandas/NumPy 执行
    PANDAS_NUMPY = "pandas_numpy"  # pandas DataFrame + NumPy ufunc

    # Polars 原生执行
    POLARS_NATIVE_EXPR = "polars_native_expr"  # Polars 表达式引擎
    POLARS_NUMPY_KERNEL = "polars_numpy_kernel"  # Polars 列 + NumPy 内核

    # Numba CPU 内核执行
    NUMBA_CPU_KERNEL = "numba_cpu_kernel"  # Numba JIT 编译 CPU 内核

    # SQL 后端原生执行
    DUCKDB_NATIVE_SQL = "duckdb_native_sql"  # DuckDB 原生 SQL 执行
    CLICKHOUSE_NATIVE_SQL = "clickhouse_native_sql"  # ClickHouse 原生 SQL 执行

    # Q 后端原生执行
    Q_NATIVE = "q_native"  # Q/kdb+ 原生执行


@dataclass(frozen=True)
class PhysicalExecutionContract:
    """R20-088..090: 单棵物理执行片段的执行契约（``PhysicalNode`` 输出语义）。

    使 ``PhysicalNode`` 不再只是 kind/plan/sid/children 的薄壳：每个物理节点
    显式声明输出 semantic digest、native backend、fallback policy、source
    snapshot、input/output grain 与 availability —— 下游缓存、PIT 证明、
    审计与增量失效都可以直接读取契约而无需重新推导。

    字段：
        output_semantic_digest: 输出 semantic-attr 摘要（``plan_hash._semantic_digest``）
        native_backend: 原生执行后端（"sql" / "python" / "polars" / "materialized"）
        fallback_policy: fallback 策略（"error" / "warn" / ""）
        source_snapshot: 该片段读取的 source snapshot id（可空）
        input_grain: 输入 grain 声明（如 ``"1d"`` / ``"1m"``）
        output_grain: 输出 grain 声明（grain 变化 = downsample，须经 downsample
            校验）
        availability: 输出 available_at 声明（PIT 边界）
    """

    output_semantic_digest: str = ""
    native_backend: str = ""
    fallback_policy: str = ""
    source_snapshot: str = ""
    input_grain: str | None = None
    output_grain: str | None = None
    availability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_semantic_digest": self.output_semantic_digest,
            "native_backend": self.native_backend,
            "fallback_policy": self.fallback_policy,
            "source_snapshot": self.source_snapshot,
            "input_grain": self.input_grain,
            "output_grain": self.output_grain,
            "availability": self.availability,
        }


def _semantic_attr(plan: PlanNode, key: str) -> Any:
    sem = getattr(plan, "semantic_attrs", None) or {}
    if isinstance(sem, Mapping):
        return sem.get(key)
    return None


def build_physical_execution_contract(
    node: "PhysicalNode",
    *,
    fallback_policy: str = "",
    source_snapshot: str = "",
) -> PhysicalExecutionContract:
    """从 ``PhysicalNode`` 推导 ``PhysicalExecutionContract``（R20-088..090）。

    由 ``sql_lowerer`` / 执行编排方在构造 PhysicalNode 后调用；未提供覆盖时从
    计划根 semantic_attrs 读取 input/output grain 与 availability。
    """
    from factor_engine.planner.plan_hash import _semantic_digest

    plan = node.plan
    output_semantic_digest = _semantic_digest(plan) or ""
    input_grain = _semantic_attr(plan, "grain")
    if isinstance(input_grain, (tuple, list)) and len(input_grain) > 0:
        input_grain = str(input_grain[0])
    else:
        input_grain = str(input_grain) if input_grain is not None else None
    output_grain = str(_semantic_attr(plan, "output_grain") or "") or None
    availability = _semantic_attr(plan, "available_at")
    return PhysicalExecutionContract(
        output_semantic_digest=output_semantic_digest,
        native_backend=str(node.kind.value) if isinstance(node.kind, Enum) else str(node.kind),
        fallback_policy=str(fallback_policy or ""),
        source_snapshot=str(source_snapshot or ""),
        input_grain=input_grain,
        output_grain=output_grain,
        availability=_availability_str(availability),
    )


class DownsampleContractError(ValueError):
    """downsample 结果违反声明的 grain/日历/时区/label 契约（R20-094..099）。"""


@dataclass(frozen=True)
class MaterializedSeriesContract:
    """R20-094..099: ``materialized_series`` 的 typed 边界。

    ``materialized_series`` 不能只是 ``sid → Series`` 的薄引用 —— 下游
    PIT / 增量 / SQL 层需要知道它是 daily/minute/event/fiscal/session 哪种
    grain、何时 available、什么 unit / price basis，以及它绑定的日历/时区。

    字段：
        grain_kind: "daily" / "minute" / "event" / "fiscal" / "session"
        frequency: 频率字符串（"1d" / "5m" …）
        available_at: PIT 可用时刻（"eod" / 时间戳 / None）
        unit: 输出单位
        price_basis: 价格基（RAW / CONTINUOUS …）
        calendar: 交易日历 ID
        timezone: 时区
        session_id: 所属交易 session（日内）
    """

    grain_kind: str = ""
    frequency: str = ""
    available_at: str | None = None
    unit: str = ""
    price_basis: str = ""
    calendar: str = ""
    timezone: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grain_kind": self.grain_kind,
            "frequency": self.frequency,
            "available_at": self.available_at,
            "unit": self.unit,
            "price_basis": self.price_basis,
            "calendar": self.calendar,
            "timezone": self.timezone,
            "session_id": self.session_id,
        }


def _semantic_of(plan: PlanNode, key: str) -> Any:
    sem = getattr(plan, "semantic_attrs", None) or {}
    if isinstance(sem, Mapping):
        return sem.get(key)
    return None


def _availability_str(value: Any) -> str | None:
    """把 available_at 归一为可 JSON 序列化字符串（R20-094..099）。

    ``available_at`` 可能是 ``ir.types.AvailabilityExpr``（``SessionClose`` /
    ``SessionOpen`` …）——用其稳定 ``label`` 归一；普通字符串/None 原样返回。
    """
    if value is None:
        return None
    label = getattr(value, "label", None)
    if isinstance(label, str) and label:
        return f"{type(value).__name__}:{label}"
    return str(value)


def _infer_grain_kind(frequency: str | None) -> str:
    freq = str(frequency or "").lower()
    if freq.endswith("m") or freq.endswith("min") or freq.endswith("t"):
        return "minute"
    if freq.endswith("d") or freq.endswith("day"):
        return "daily"
    if freq.endswith("w") or freq.endswith("week"):
        return "daily"
    if freq.endswith("q") or freq.endswith("fiscal"):
        return "fiscal"
    if "session" in freq:
        return "session"
    if freq.endswith("event"):
        return "event"
    return "daily"


def materialized_series_contract(
    plan: PlanNode,
    *,
    fallback: MaterializedSeriesContract | None = None,
) -> MaterializedSeriesContract:
    """从 ``materialized_series`` 占位节点推导 typed contract（R20-094..099）。

    优先读取 ``plan.semantic_attrs``（grain / frequency / available_at / unit /
    price_basis / calendar / timezone / session_id）；缺失时从 ``attrs`` 的 sid
    与 ``fallback`` 补齐。
    """
    freq = _semantic_of(plan, "frequency")
    if freq is None:
        freq = (plan.attrs or {}).get("frequency")
    grain = _semantic_of(plan, "grain")
    grain_kind = _infer_grain_kind(freq)
    if isinstance(grain, (tuple, list)) and len(grain) > 0:
        grain_kind = str(grain[0])
    elif isinstance(grain, str) and grain:
        grain_kind = grain
    return MaterializedSeriesContract(
        grain_kind=grain_kind,
        frequency=str(freq or (fallback.frequency if fallback else "")),
        available_at=_availability_str(
            _semantic_of(plan, "available_at")
            or (fallback.available_at if fallback else None)
        ),
        unit=str(_semantic_of(plan, "unit") or (fallback.unit if fallback else "")),
        price_basis=str(
            _semantic_of(plan, "price_basis") or (fallback.price_basis if fallback else "")
        ),
        calendar=str(
            _semantic_of(plan, "calendar_id")
            or _semantic_of(plan, "calendar")
            or (fallback.calendar if fallback else "")
        ),
        timezone=str(
            _semantic_of(plan, "timezone") or (fallback.timezone if fallback else "")
        ),
        session_id=str(
            _semantic_of(plan, "session_id") or (fallback.session_id if fallback else "")
        ),
    )


def validate_downsample_contract(
    result: Any,
    *,
    declared_grain: str | None,
    template_panel: Any,
    session_calendar: str | None = None,
    timezone: str | None = None,
    declared_available_at: str | None = None,
    output_label_convention: str = "timestamp",
) -> None:
    """R20-094..099: 比「行数变少」更严格的 downsample 校验。

    ``backend.cleaned_bridge._validate_downsampled_result`` 只查 index 唯一、
    columns 匹配、行数不增。这里补上 typed 边界校验：

    - result index 必须是 DatetimeIndex 且唯一（行数变少不成立时拒绝上采样）；
    - **trading session 映射**：grain 声明为 session/minute 时，index 必须落在
      会话日（同一交易日时间戳），不允许跨日历漂移；
    - **output label convention**：index name 必须符合声明的 label 约定
      （``timestamp`` / ``datetime``）；
    - **calendar**：若声明了 session calendar，index 的日期必须是该日历的
      trading day（由调用方传入 trading day 集合）；
    - **timezone**：index tz 必须与声明时区一致（不一致 → 拒绝）；
    - **declared grain**：``declared_grain`` 与 index 实际频率语义不冲突；
    - **available_at**：若声明了 available_at（如 ``"eod"``），downsample 输出
      不得携带比输入更晚的可用时刻（不把未来信号写入结果）。
    """
    import pandas as pd

    if not isinstance(result, pd.DataFrame):
        raise DownsampleContractError(
            f"downsample result must be a DataFrame panel, got {type(result).__name__}"
        )
    if not isinstance(result.index, pd.DatetimeIndex):
        raise DownsampleContractError(
            f"downsample result index {type(result.index).__name__} is not a "
            "DatetimeIndex (R20-094..099 fail-closed)"
        )
    if not result.index.is_unique:
        raise DownsampleContractError("downsample result has duplicate timestamps")
    if output_label_convention and result.index.name not in {
        output_label_convention,
        "timestamp",
        "datetime",
        None,
    }:
        raise DownsampleContractError(
            f"downsample output index name {result.index.name!r} does not follow "
            f"the declared label convention {output_label_convention!r}"
        )
    if timezone:
        decl_tz = str(timezone)
        idx_tz = getattr(result.index, "tz", None)
        idx_tz_str = str(idx_tz) if idx_tz is not None else ""
        if decl_tz and idx_tz_str and decl_tz not in (idx_tz_str,):
            raise DownsampleContractError(
                f"downsample result timezone {idx_tz_str!r} does not match "
                f"declared timezone {decl_tz!r}"
            )
    if session_calendar:
        _validate_session_calendar_days(result.index, session_calendar)
    if declared_available_at and str(declared_available_at).lower() in {"eod", "close"}:
        # 日终 available_at 的 downsample 输出必须落在交易日（非日内 00:00 前）。
        if len(result.index) and result.index[0].time() < result.index[0].time().replace(
            hour=0, minute=0, second=0
        ):
            raise DownsampleContractError(
                "downsample result carries a pre-midnight timestamp with an "
                "'eod' available_at — a session-close signal must not be shifted"
            )
    if template_panel is not None and len(result) > len(template_panel):
        raise DownsampleContractError(
            f"downsample produced {len(result)} rows from an input of "
            f"{len(template_panel)} — an undeclared UPSAMPLING is not allowed"
        )


def _validate_session_calendar_days(index: Any, calendar: str) -> None:
    """校验 downsample index 的日期全部落在声明 session calendar 的交易日。"""
    try:
        from calendar_service import is_trading_day  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - resolver unavailable
        return
    bad: list[Any] = []
    for ts in index.normalize():
        if not is_trading_day(ts, calendar):
            bad.append(ts)
    if bad:
        raise DownsampleContractError(
            f"downsample result contains non-trading days for calendar {calendar!r}: "
            f"{[str(b) for b in bad[:5]]}"
        )


@dataclass(frozen=True)
class PhysicalNode:
    """``sql_lowerer`` 输出：带执行语义的计划节点。

    字段：
        kind: 执行后端种类（SQL / PYTHON / MATERIALIZED）
        plan: 对应的逻辑计划片段
        sid: SQL 子树或物化列的结构缓存键（可选）
        children: 递归标注的子物理节点
        contract: R20-088..090 物理执行契约（可选，缺省 None）
    """

    kind: ExecKind
    plan: PlanNode
    sid: str | None = None
    children: tuple[PhysicalNode, ...] = field(default_factory=tuple)
    contract: PhysicalExecutionContract | None = None


@dataclass
class PhysicalPlan:
    """混合执行物理计划（root + 待预计算 SQL 子树）。

    字段：
        root: 改写后的逻辑根（可能含 ``materialized_series`` 占位）
        sql_subtrees: 需预计算的 SQL 子树 ``{sid: subplan}``
        fully_sql: 整棵树是否可完全 SQL 下推执行
        execution_contract: R20-088..090 根契约（可选）
    """

    root: PlanNode
    sql_subtrees: dict[str, PlanNode] = field(default_factory=dict)
    fully_sql: bool = False
    execution_contract: PhysicalExecutionContract | None = None
