# -*- coding: utf-8 -*-
"""R30-P0-011 + R32-P0-079..086：DataChangeSet → 最小重算计划（正确性修复）。

本模块是 DA→FE 桥：DA 侧 :mod:`data_change` 产出「哪些数据变了」的
``DataChangeSet``，本模块消费它，产出「哪些因子必须重算、重算窗口多大」的
``AffectedFactor`` 列表（最小重算），并负责把 change 翻译成 FE
``factor_engine.runtime.change_impact.compute_change_impact`` 能直接吃的输入
（``to_fe_input``，懒 import，不硬依赖 FE）。

R30-P0-011 验收：**单公司一条财报修订不允许默认触发「全 A 股全历史所有因子」
重算**。``plan_minimal_recompute`` 只挑「dataset 匹配 ∧ instrument 相交 ∧ 时间窗
相交（向前扩 lookback）」的因子，窗口 = ``changed_time_range`` 向前扩 lookback，
**绝不是全历史全因子**。

R32-P0-079..086 正确性修复（增量重算 under-invalidation bug）：
    - R32-P0-079: rolling 依赖向前传播到 OUTPUTS（MA20 input[t]→outputs[t..t+19]）
    - R32-P0-080: window=60 是 60 BARS（交易日），不是 60 calendar days
    - R32-P0-081: cross-sectional 扩散（一变全变：rank/zscore/neutralize）
    - R32-P0-082: changed_instruments 类型化（RANGE vs EXACT_SET，manifest 是统计量）
    - R32-P0-083: changed_columns 类型化（空 tuple 不再多义）
    - R32-P0-084: to_fe_input 保留完整 changed_columns（旧 bug 只取 [0]）
    - R32-P0-085: ImpactDirection（split 可 BACKWARD 重写历史）
    - R32-P0-086: universe/calendar 变化一类事件

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
from data_access.r30.change_impact_types import (
    AxisEffect,
    ColumnChange,
    ImpactDirection,
    InstrumentChange,
    InstrumentScope,
    OperatorDependencyTraits,
    TimeAxisKind,
    coerce_column_change,
    coerce_instrument_change,
)


@dataclass(frozen=True)
class AffectedFactor:
    """一个因子因数据集变化而必须重算的最小区间。

    R32-P0-081/085: 新增 axis_effect/affected_instruments/impact_direction
    字段，用于精确标记影响范围。
    """

    factor_id: str
    dataset: str
    affected_start: str | None
    affected_end: str | None
    reason: str
    change_kind: str
    axis_effect: str | None = None
    affected_instruments: tuple[str, ...] | None = None
    impact_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "dataset": self.dataset,
            "affected_start": self.affected_start,
            "affected_end": self.affected_end,
            "reason": self.reason,
            "change_kind": self.change_kind,
            "axis_effect": self.axis_effect,
            "affected_instruments": list(self.affected_instruments) if self.affected_instruments else None,
            "impact_direction": self.impact_direction,
        }


# ---------------------------------------------------------------------------
# R32-P0-079/080: 时间偏移（backward + forward, BARS vs calendar days）
# ---------------------------------------------------------------------------
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


def _shift_forward(value: Any, days: int) -> Any:
    """R32-P0-079: 向后（未来方向）偏移 ``days`` 个自然日（forward propagation）。"""
    if value is None or days <= 0:
        return value
    if isinstance(value, datetime):
        return (value + timedelta(days=days)).isoformat()
    if isinstance(value, date):
        return (value + timedelta(days=days)).isoformat()
    if isinstance(value, (int, float)):
        return value + int(days)
    text = str(value)[:10]
    try:
        return (date.fromisoformat(text) + timedelta(days=days)).isoformat()
    except ValueError:
        return value


def _shift_back_bars(value: Any, bars: int, calendar: Any) -> Any:
    """R32-P0-080: 按 TRADING_BARS 向前偏移（需 TradingCalendar）。"""
    if value is None or bars <= 0 or calendar is None:
        return value
    base_date = _to_scalar(value)
    if not isinstance(base_date, date):
        return _shift_back(value, bars)  # fallback 自然日
    # calendar.offset(base, -bars, clamp=True) → 向前 bars 个交易日
    offset_fn = getattr(calendar, "offset", None)
    if callable(offset_fn):
        result = offset_fn(base_date, -bars, clamp=True)
        return result.isoformat() if isinstance(result, date) else result
    return _shift_back(value, bars)  # fallback


def _shift_forward_bars(value: Any, bars: int, calendar: Any) -> Any:
    """R32-P0-079/080: 按 TRADING_BARS 向后偏移（forward propagation）。"""
    if value is None or bars <= 0 or calendar is None:
        return value
    base_date = _to_scalar(value)
    if not isinstance(base_date, date):
        return _shift_forward(value, bars)  # fallback 自然日
    offset_fn = getattr(calendar, "offset", None)
    if callable(offset_fn):
        result = offset_fn(base_date, bars, clamp=True)
        return result.isoformat() if isinstance(result, date) else result
    return _shift_forward(value, bars)  # fallback


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


# ---------------------------------------------------------------------------
# R32-P0-079..086: plan_minimal_recompute（综合修复）
# ---------------------------------------------------------------------------
def plan_minimal_recompute(
    change: DataChangeSet,
    factor_demands: Sequence[Mapping[str, Any]],
    lookback_days: int = 0,
    calendar: Any = None,
) -> list[AffectedFactor]:
    """把一次数据变化映射成受影响的因子列表（最小重算）。

    R32-P0-079..086 修复：
        - forward_output_horizon（rolling 向前传播）
        - TRADING_BARS（按交易日偏移，需 calendar）
        - cross-sectional 扩散（一变全变）
        - typed InstrumentScope/ColumnChange（不再多义空容器）
        - universe/calendar 变化一类事件

    ``factor_demands`` 每项含：
        ``factor_id``           因子标识
        ``datasets``            依赖的 dataset 元组
        ``instruments``         标的限制（None / 空 = 全市场）
        ``lookback``            该因子的回看天数（缺省用 ``lookback_days``）
        ``window_fields``       该因子消费的列（本函数不强制使用，留给 FE 桥接）
        ``operator_traits``     :class:`OperatorDependencyTraits`（R32-P0-079）
        ``axis_effect``         :class:`AxisEffect`（R32-P0-081）
        ``impact_direction``    :class:`ImpactDirection`（R32-P0-085）

    只挑同时满足以下条件的 demand：
        1. ``change.dataset`` ∈ demand.datasets；
        2. demand 无 instrument 限制，或与 ``change.changed_instruments`` 相交
           （change 未标明标的 → 保守视为全标的）；
        3. 需求窗口与 ``change.changed_time_range`` 相交。

    受影响窗口 = ``changed_time_range`` 向前扩 ``backward_input_horizon``，向后扩
    ``forward_output_horizon``（R32-P0-079 修复）。返回列表即**最小重算计划**
    ——不会全历史全因子。
    """
    if factor_demands is None:
        return []
    demands = list(factor_demands)
    changed_start, changed_end = change.changed_time_range or (None, None)
    global_lookback = int(lookback_days or 0)
    change_kind_str = (
        change.change_kind.value
        if isinstance(change.change_kind, ChangeKind)
        else str(change.change_kind)
    )

    # R32-P0-082: 类型化 InstrumentScope
    inst_change = coerce_instrument_change(change.changed_instruments)
    # R32-P0-083: 类型化 ColumnChange
    col_change = coerce_column_change(change.changed_columns)

    # R32-P0-086: universe/calendar 变化一类事件
    is_universe_change = change_kind_str == "universe_change"
    is_calendar_change = change_kind_str == "calendar_change"

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

        # R32-P0-081: axis_effect（elementwise/time_series/cross_section/...）
        axis_effect = demand.get("axis_effect", "elementwise")

        # R32-P0-082: instrument 相交判定（类型化）
        demand_insts = demand.get("instruments")
        if demand_insts is None or len(demand_insts) == 0:
            inst_ok = True  # 全市场因子
        elif inst_change.is_conservative():
            inst_ok = True  # change 保守全量（UNKNOWN_ALL/UNIVERSE_WIDE/...）
        elif inst_change.scope == InstrumentScope.EXACT_SET:
            inst_ok = bool(set(demand_insts) & set(inst_change.exact_instruments))
        elif inst_change.scope == InstrumentScope.RANGE:
            # manifest min/max 是统计量，保守视为可能相交（防 under-invalidation）
            inst_ok = True
        else:
            inst_ok = True  # 其他未知 scope，保守

        if not inst_ok:
            continue

        # R32-P0-086: calendar 变化影响 rolling/time_series 算子
        if is_calendar_change and axis_effect not in {"time_series", "stateful"}:
            continue  # calendar 变化只影响时间序列算子

        # R32-P0-086: universe 变化影响 cross-sectional 算子
        if is_universe_change and axis_effect not in {
            "cross_section", "group_cross_section", "global_panel"
        }:
            continue  # universe 变化只影响截面算子

        if not _window_overlap(demand, changed_start, changed_end):
            continue

        # R32-P0-079: 算子依赖特性（backward + forward horizon）
        operator_traits = demand.get("operator_traits")
        if isinstance(operator_traits, OperatorDependencyTraits):
            backward_horizon = operator_traits.backward_input_horizon
            forward_horizon = operator_traits.forward_output_horizon
            time_axis = operator_traits.time_axis
        else:
            # 向后兼容：legacy lookback 是 backward input horizon
            lookback = int(demand.get("lookback", global_lookback) or 0)
            backward_horizon = lookback
            forward_horizon = lookback  # 保守：假设 forward = backward
            time_axis = TimeAxisKind.TRADING_BARS

        # R32-P0-080: 按 BARS 偏移（有 calendar）或自然日（无 calendar）
        if time_axis == TimeAxisKind.TRADING_BARS and calendar is not None:
            aff_start = _shift_back_bars(changed_start, backward_horizon, calendar)
            aff_end = _shift_forward_bars(changed_end, forward_horizon, calendar)
        else:
            # fallback: 按自然日（保守放大）
            aff_start = _shift_back(changed_start, backward_horizon)
            aff_end = _shift_forward(changed_end, forward_horizon)

        # R32-P0-085: impact direction（split/复权可 BACKWARD）
        impact_direction = demand.get("impact_direction", ImpactDirection.FORWARD.value)

        # R32-P0-081: cross-sectional 扩散全截面（或组内）
        if axis_effect == "cross_section":
            affected_instruments = None  # 全截面
        elif axis_effect == "group_cross_section":
            # 组内扩散（需 group_key，暂不实现精确过滤，保守全量）
            affected_instruments = None
        elif inst_change.scope == InstrumentScope.EXACT_SET:
            affected_instruments = inst_change.exact_instruments
        else:
            affected_instruments = None  # 保守全量

        affected.append(
            AffectedFactor(
                factor_id=str(factor_id),
                dataset=change.dataset,
                affected_start=aff_start,
                affected_end=aff_end,
                reason="dataset_changed",
                change_kind=change_kind_str,
                axis_effect=axis_effect,
                affected_instruments=affected_instruments,
                impact_direction=impact_direction,
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

    R32 P0-083..086：保留完整 changed_columns 列表（旧实现只取 [0]，多列修订丢失）。

    返回 dict 的关键字与 FE ``compute_change_impact(plan, *, field,
    changed_start, changed_end, column_identity, calendar)`` 对齐：
    ``{"field", "changed_start", "changed_end", "column_identity", "changed_columns"}``
    （calendar 仅在传入时给出）。``column_identity`` 只含非空字段，键限定在
    ``ColumnIdentity`` 的 ``(dataset, field, market, grain, price_basis,
    revision_identity)`` 内。**懒 import、不硬依赖 FE**——本函数不 import FE。
    """
    changed_cols = list(change.changed_columns or ())
    field = changed_cols[0] if changed_cols else ""
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
        "changed_columns": changed_cols,  # R32-P0-084: 保留完整列表
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
