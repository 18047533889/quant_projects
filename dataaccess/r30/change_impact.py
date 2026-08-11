# -*- coding: utf-8 -*-
"""R30-P0-011 (DATA_CHANGE_MIN_RECOMPUTE)：DataChangeSet → 最小重算计划。

本模块是 DA→FE 桥：DA 侧 :mod:`data_change` 产出「哪些数据变了」的
``DataChangeSet``，本模块消费它，产出「哪些因子必须重算、重算窗口多大」的
``AffectedFactor`` 列表（最小重算），并负责把 change 翻译成 FE
``factor_engine.runtime.change_impact.compute_change_impact`` 能直接吃的输入
（``to_fe_input``，懒 import，不硬依赖 FE）。

R30-P0-011 验收：**单公司一条财报修订不允许默认触发「全 A 股全历史所有因子」
重算**。``plan_minimal_recompute`` 只挑「dataset 匹配 ∧ instrument 相交 ∧ 时间窗
相交（向前扩 lookback）」的因子，窗口 = ``changed_time_range`` 向前扩 lookback，
**绝不是全历史全因子**。

FE 侧已有（不改动）：
    ``compute_change_impact(plan, field, changed_start, changed_end,
    column_identity, calendar) -> list[AffectedWindow]``
    ``ColumnIdentity(dataset, field, market, grain, price_basis,
    revision_identity)``
本模块不 import FE 任何东西；``to_fe_input`` 只产出与 FE 关键字匹配的 dict。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from data_access.r30.data_change import ChangeKind, DataChangeSet


@dataclass(frozen=True)
class AffectedFactor:
    """一个因子因数据集变化而必须重算的最小区间。"""

    factor_id: str
    dataset: str
    affected_start: str | None
    affected_end: str | None
    reason: str
    change_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "dataset": self.dataset,
            "affected_start": self.affected_start,
            "affected_end": self.affected_end,
            "reason": self.reason,
            "change_kind": self.change_kind,
        }


def _shift_back(value: Any, days: int) -> Any:
    """把时间值向前（更早方向）偏移 ``days`` 个自然日；无法解析则原样返回。"""
    if value is None or days <= 0:
        return value
    if isinstance(value, datetime):
        return (value - timedelta(days=days)).isoformat()
    if isinstance(value, date):
        return (value - timedelta(days=days)).isoformat()
    if isinstance(value, (int, float)):
        return value - int(days)
    text = str(value)[:10]
    try:
        return (date.fromisoformat(text) - timedelta(days=days)).isoformat()
    except ValueError:
        return value


def _window_overlap(
    demand: Mapping[str, Any],
    changed_start: Any,
    changed_end: Any,
) -> bool:
    """因子需求覆盖窗口与变化区间是否相交。

    demand 里可选的 ``start`` / ``end`` 表示因子的数据覆盖窗口（防御性）；两者
    都不给 → 认为因子覆盖任何区间 → 相交。
    """
    req_start = demand.get("start")
    req_end = demand.get("end")
    if req_start is None and req_end is None:
        return True
    if req_end is not None and changed_start is not None:
        try:
            if _to_scalar(req_end) < _to_scalar(changed_start):
                return False
        except (TypeError, ValueError):
            return True
    if req_start is not None and changed_end is not None:
        try:
            if _to_scalar(req_start) > _to_scalar(changed_end):
                return False
        except (TypeError, ValueError):
            return True
    return True


def _to_scalar(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def plan_minimal_recompute(
    change: DataChangeSet,
    factor_demands: Sequence[Mapping[str, Any]],
    lookback_days: int = 0,
) -> list[AffectedFactor]:
    """把一次数据变化映射成受影响的因子列表（最小重算）。

    ``factor_demands`` 每项含：
        ``factor_id``      因子标识
        ``datasets``       依赖的 dataset 元组
        ``instruments``    标的限制（None / 空 = 全市场）
        ``lookback``       该因子的回看天数（缺省用 ``lookback_days``）
        ``window_fields``  该因子消费的列（本函数不强制使用，留给 FE 桥接）

    只挑同时满足以下条件的 demand：
        1. ``change.dataset`` ∈ demand.datasets；
        2. demand 无 instrument 限制，或与 ``change.changed_instruments`` 相交
           （change 未标明标的 → 保守视为全标的）；
        3. 需求窗口与 ``change.changed_time_range`` 相交。

    受影响窗口 = ``changed_time_range`` 向前扩 ``lookback`` 天。返回列表即
    **最小重算计划**——不会全历史全因子。
    """
    if factor_demands is None:
        return []
    demands = list(factor_demands)
    changed_insts = set(change.changed_instruments or ())
    changed_start, changed_end = change.changed_time_range or (None, None)
    global_lookback = int(lookback_days or 0)
    change_kind_str = (
        change.change_kind.value
        if isinstance(change.change_kind, ChangeKind)
        else str(change.change_kind)
    )

    affected: list[AffectedFactor] = []
    for demand in demands:
        if not isinstance(demand, Mapping):
            continue
        factor_id = demand.get("factor_id")
        if not factor_id:
            continue
        datasets = demand.get("datasets") or ()
        if change.dataset not in datasets:
            continue

        demand_insts = demand.get("instruments")
        if demand_insts is None or len(demand_insts) == 0:
            inst_ok = True  # 全市场因子：数据变化理论上都可能影响
        elif not changed_insts:
            inst_ok = True  # change 未标明标的 → 保守视为全标的变化
        else:
            inst_ok = bool(set(demand_insts) & changed_insts)
        if not inst_ok:
            continue

        if not _window_overlap(demand, changed_start, changed_end):
            continue

        lookback = int(demand.get("lookback", global_lookback) or 0)
        affected.append(
            AffectedFactor(
                factor_id=str(factor_id),
                dataset=change.dataset,
                affected_start=_shift_back(changed_start, lookback),
                affected_end=changed_end,
                reason="dataset_changed",
                change_kind=change_kind_str,
            )
        )
    return affected


def _diff_days(start: Any, end: Any) -> int | None:
    s = _to_scalar(start)
    e = _to_scalar(end)
    if isinstance(s, date) and isinstance(e, date):
        return max(int((e - s).days), 0)
    if isinstance(s, (int, float)) and isinstance(e, (int, float)):
        return max(int(e - s), 0)
    return None


def recompute_budget(
    affected: Sequence[AffectedFactor],
    change: DataChangeSet,
) -> dict[str, Any]:
    """对最小重算计划做预算估算（受影响因子数 / 最小窗口 / 估算重算成本）。"""
    affected_list = list(affected)
    starts = [a.affected_start for a in affected_list if a.affected_start is not None]
    ends = [a.affected_end for a in affected_list if a.affected_end is not None]
    min_start = min(starts) if starts else None
    max_end = max(ends) if ends else None
    window_days = _diff_days(min_start, max_end) if min_start is not None and max_end is not None else None
    instrument_count = max(len(change.changed_instruments or ()), 1)
    cells = len(affected_list) * (window_days or 1) * instrument_count
    return {
        "dataset": change.dataset,
        "change_kind": (
            change.change_kind.value
            if isinstance(change.change_kind, ChangeKind)
            else str(change.change_kind)
        ),
        "affected_factors": len(affected_list),
        "min_window_start": min_start,
        "max_window_end": max_end,
        "window_days": window_days,
        "changed_instrument_count": instrument_count,
        "estimated_recompute_cells": cells,
        "estimated_recompute_cost": cells,  # 相对成本单位
    }


# ---------------------------------------------------------------------------
# FE 桥：把 DataChangeSet 翻译成 FE compute_change_impact 的关键字输入。
# ---------------------------------------------------------------------------
def to_fe_input(change: DataChangeSet, calendar: Any = None) -> dict[str, Any]:
    """把 ``DataChangeSet`` 翻译成 FE ``compute_change_impact`` 的可展开输入。

    返回 dict 的关键字与 FE ``compute_change_impact(plan, *, field,
    changed_start, changed_end, column_identity, calendar)`` 对齐：
    ``{"field", "changed_start", "changed_end", "column_identity"}``（calendar 仅
    在传入时给出）。``column_identity`` 只含非空字段，键限定在
    ``ColumnIdentity`` 的 ``(dataset, field, market, grain, price_basis,
    revision_identity)`` 内。**懒 import、不硬依赖 FE**——本函数不 import FE。
    """
    field = change.changed_columns[0] if change.changed_columns else ""
    column_identity: dict[str, str] = {"dataset": change.dataset}
    if field:
        column_identity["field"] = field
    revision_availability = change.revision_availability or {}
    revision_identity = revision_availability.get("latest_revision_date")
    if revision_identity:
        column_identity["revision_identity"] = str(revision_identity)

    changed_start, changed_end = change.changed_time_range or (None, None)
    out: dict[str, Any] = {
        "field": field,
        "changed_start": changed_start,
        "changed_end": changed_end,
        "column_identity": column_identity,
    }
    if calendar is not None:
        out["calendar"] = calendar
    return out


__all__ = [
    "AffectedFactor",
    "plan_minimal_recompute",
    "recompute_budget",
    "to_fe_input",
]
