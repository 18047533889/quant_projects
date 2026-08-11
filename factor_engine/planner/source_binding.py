# -*- coding: utf-8 -*-
"""R39-P0-PERF-002/004：typed column→source binding 与 typed TimeRange。

R39-P0-PERF-002（BatchDataRequest 真正 multi-source）
    secondary SourceRef 分组不再走 ``dict[str, str]`` 承载「dataset 名又当
    market 又当 scope」的多重语义（R39 审计基线：``_source_ref_datasets`` 的
    value 是 market，随后却被塞进 ``SourceScopeId(dataset=...)`` —— 类型混用）。

    改为：扫描 PlanNode 的 ``column`` 节点，凡是 SourceRef 列就**直接**解码出
    :class:`ColumnSourceBinding`（encoded_column / dataset / field / market /
    typed ``source_scope``）。``build_batch_data_request`` 只消费这个 typed
    binding，不再间接字符串匹配。

R39-P0-PERF-004（保留 end-only / start-only 时间窗）
    统一 :class:`TimeRange`（frozen dataclass），``(None, end)`` / ``(start,
    None)`` / ``(start, end)`` 三种形态原样保留；禁止依赖
    ``time_range[0] is not None`` 决定整个 range 是否存在。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from planner.physical_factor_dag import SourceScopeId


@dataclass(frozen=True)
class ColumnSourceBinding:
    """R39-P0-PERF-002：一个 SourceRef 列的完整 typed 身份。

    ``encoded_column`` 是 PlanNode 里出现的原始列名（``__fe_source_ref_v1__...``）；
    ``dataset`` / ``field`` / ``market`` 来自解码后的 SourceRef v2 语义字段；
    ``source_scope`` 是 typed :class:`SourceScopeId` —— 分组、scan-cost、
    read-wave 的**唯一**业务身份载体。
    """

    encoded_column: str
    dataset: str
    field: str
    market: str
    source_scope: SourceScopeId

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded_column": self.encoded_column,
            "dataset": self.dataset,
            "field": self.field,
            "market": self.market,
            "source_scope": self.source_scope.to_dict(),
        }


@dataclass(frozen=True)
class TimeRange:
    """R39-P0-PERF-004：typed 时间窗，两端独立可空。

    ``(None, end)`` = 只给上界（剪枝：end 之前的 partition / row-group）；
    ``(start, None)`` = 只给下界（剪枝：start 之后的 …）；
    ``(start, end)`` = 闭窗。
    任一端为空**不是**「没有时间过滤」，不得退化成全量扫描。
    """

    start: Any = None  # Timestamp | str | None
    end: Any = None  # Timestamp | str | None

    def as_tuple(self) -> tuple[Any, Any]:
        """传给 ``estimate_scan_cost(time_range=...)`` 的形态（允许 None 端）。"""
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return {"start": _fmt_bound(self.start), "end": _fmt_bound(self.end)}

    @staticmethod
    def from_source(source: Any) -> "TimeRange | None":
        start = getattr(source, "start_date", None)
        end = getattr(source, "end_date", None)
        if start is None and end is None:
            return None
        return TimeRange(start=start, end=end)


def _fmt_bound(value: Any) -> Any:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _scope_from_ref(ref: Any, *, dataset: str, market: str) -> SourceScopeId:
    """从 SourceRef 构造 typed scope。不同 transform / 参数的同一 dataset 得到
    不同 ``params_digest``（scan-cost 与 read-wave 是独立物理需求）。"""
    parts: list[str] = []
    transform = getattr(ref, "transform", None)
    if transform:
        parts.append(f"transform:{transform}")
    for key, value in sorted((getattr(ref, "params_dict", lambda: {})()).items()):
        parts.append(f"p:{key}={value}")
    for key, value in sorted((getattr(ref, "transform_params_dict", lambda: {})()).items()):
        parts.append(f"tp:{key}={value}")
    dv = getattr(ref, "dialect_version", None)
    if dv:
        parts.append(f"dv:{dv}")
    return SourceScopeId(
        dataset=dataset,
        market=market,
        params_digest=";".join(parts),
    )


@dataclass(frozen=True)
class SourceBindingResult:
    """R39-P0-PERF-002 hard-gate 输入：发现结果 + 计数。"""

    bindings: dict[str, ColumnSourceBinding]
    source_ref_columns_seen: int = 0  # 去重后的 SourceRef 列名总数
    source_ref_columns_bound: int = 0  # 其中产生 typed binding 的数量


def discover_column_source_bindings(plans: Iterable[Any]) -> SourceBindingResult:
    """扫描计划中的 ``column`` 节点，为每个 SourceRef 列构建 typed binding。

    - 非 SourceRef 列：不产生 binding（走 anchor group）。
    - SourceRef 列解码失败 / 无 dataset：不产生 binding（hard-gate 计数）。
    - 返回 :class:`SourceBindingResult`（bindings + seen/bound 计数），绝不返回
      ``dict[str, str]`` 多语义映射。
    """
    try:
        from api.source_ref import decode_source_ref, looks_like_source_ref
    except Exception:  # pragma: no cover - import 环境缺 api 时降级为无 binding
        return SourceBindingResult(bindings={})

    bindings: dict[str, ColumnSourceBinding] = {}
    seen: set[str] = set()
    bound: set[str] = set()

    def walk(node: Any) -> None:
        op = str(getattr(node, "op", "") or "")
        if op == "column":
            name = str((getattr(node, "attrs", None) or {}).get("name") or "")
            if name and looks_like_source_ref(name):
                seen.add(name)
                if name in bindings:
                    bound.add(name)
                    return
                try:
                    ref = decode_source_ref(name)
                except Exception:  # noqa: BLE001
                    ref = None
                if ref is not None and getattr(ref, "dataset", None):
                    dataset = str(ref.dataset)
                    field = str(ref.field or "")
                    market = str(ref.market or "")
                    bindings[name] = ColumnSourceBinding(
                        encoded_column=name,
                        dataset=dataset,
                        field=field,
                        market=market,
                        source_scope=_scope_from_ref(ref, dataset=dataset, market=market),
                    )
                    bound.add(name)
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    for plan in plans:
        walk(plan)
    return SourceBindingResult(
        bindings=bindings,
        source_ref_columns_seen=len(seen),
        source_ref_columns_bound=len(bound),
    )
