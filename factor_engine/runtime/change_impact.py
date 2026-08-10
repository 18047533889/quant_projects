# -*- coding: utf-8 -*-
"""R31-P1-038 (CHANGE_IMPACT_RECOMPUTE_PASS)：source-change → affected 区间传播。

从「factor watermark」升级到 Change Impact DAG（R31 §55）：输入一个变化集合
``(dataset, field, instrument, time interval, source_version)``，沿 dependency
DAG 传播出受影响的 operator intervals 与 root partitions——只重算真正受影响的数据。

    - elementwise（add/log/…）：change at T → output affected [T, T]。
    - rolling（ts_mean/ts_std/…）：change at T → affected [T, T+window-1]
      （按 ``forward_impact``）。
    - stateful（EMA/recursive）：affected [T, ∞) 直到 checkpoint/replay end。
    - cs/group：change at T 影响当天整截面 → 至少 [T, T]，跨截面算子的
      forward_impact 由 contract 提供。

``affected_root_window(root, field, changed_start, changed_end)`` 返回根因子输出上
必须重算的最小连续区间（None 终点 = 直到 checkpoint 的重放）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AffectedWindow:
    """一个算子/因子根受 source change 影响的输出区间。"""

    node_id: str
    op: str
    start: str
    end: str | None  # None = 无限延伸到 checkpoint/replay end
    unbounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "start": self.start,
            "end": self.end,
            "unbounded": self.unbounded,
        }


def _parse_date(value: Any) -> Any:
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def _own_impact(op: str, attrs: dict[str, Any]) -> int | None:
    """算子自身的 forward impact（多少个 bar 之后仍受影响）。

    缺省：rolling 算子按 window-1；stateful 视为无限（None=unbounded）；
    其余点算子 0。优先读 execution contract 的 ``forward_impact``。
    """
    try:
        from runtime.execution_contract import forward_impact

        impact = forward_impact(op, attrs)
        # contract 权威：返回 None = unbounded（递归 state 直到 checkpoint）。
        if impact is not None:
            return int(impact)
        return None
    except Exception:
        pass
    # 兜底启发：rolling window 类（contract 未知时）
    if op.startswith(("ts_", "rolling_", "ewm")) or op in {"wma", "decay_linear"}:
        window = attrs.get("window") or attrs.get("period") or attrs.get("span")
        try:
            return max(0, int(window) - 1)
        except (TypeError, ValueError):
            return 0
    if op in {"ts_ema", "wilder", "kama", "kalman", "garch", "state_machine", "episode", "psar", "supertrend"}:
        return None  # unbounded（stateful，直到 checkpoint/replay end）
    return 0


def _walk_nodes(plan: Any) -> list[tuple[str, str, dict[str, Any], tuple[str, ...]]]:
    """展平 plan 为 (node_id, op, attrs, child_ids)，含 meta op 的 id。"""
    out: list[tuple[str, str, dict[str, Any], tuple[str, ...]]] = []
    seen: set[int] = set()
    counter: list[int] = [0]

    def walk(node: Any) -> str:
        if id(node) in seen:
            return getattr(node, "node_id", None) or f"n{id(node)}"
        seen.add(id(node))
        counter[0] += 1
        my_id = getattr(node, "node_id", None) or f"n{counter[0]}"
        child_ids = tuple(walk(c) for c in (getattr(node, "inputs", ()) or ()))
        attrs = dict(getattr(node, "attrs", None) or {})
        op = str(getattr(node, "op", "") or "")
        out.append((my_id, op, attrs, child_ids))
        return my_id

    walk(plan)
    return out


def compute_change_impact(
    plan: Any,
    *,
    field: str,
    changed_start: str,
    changed_end: str | None = None,
) -> list[AffectedWindow]:
    """把 ``(field, [changed_start, changed_end])`` 沿 plan DAG 传播。

    返回每个受影响节点的最小输出区间。根（最后一个节点）的窗口即因子输出上
    必须重算的区间。
    """
    nodes = _walk_nodes(plan)
    start = _parse_date(changed_start)
    end = _parse_date(changed_end) or start
    affected: dict[str, AffectedWindow] = {}

    def _propagate(node_id: str, op: str, attrs: dict[str, Any], child_ids: tuple[str, ...]) -> None:
        # 子节点影响并集作为本节点的「输入受影响区间」。
        in_start: Any = None
        in_end: Any | None = None
        in_unbounded = False
        for cid in child_ids:
            w = affected.get(cid)
            if w is None:
                continue
            if w.start is not None and (in_start is None or w.start < in_start):
                in_start = w.start
            if w.end is None:
                in_unbounded = True
            elif in_end is None or w.end > in_end:
                in_end = w.end
        if op == "column":
            name = str(attrs.get("name") or "")
            if name == field:
                affected[node_id] = AffectedWindow(node_id, op, str(start), str(end))
            return
        if op in {"literal"}:
            return
        impact = _own_impact(op, attrs)
        if in_start is None and op != "column":
            return  # 上游无变化
        out_start = _parse_date(in_start or start)
        if in_unbounded or impact is None:
            affected[node_id] = AffectedWindow(node_id, op, str(out_start.date()), None, unbounded=True)
        else:
            out_end = _parse_date(in_end or end) + __import__("pandas").offsets.BDay(impact)
            affected[node_id] = AffectedWindow(node_id, op, str(out_start.date()), str(out_end.date()))

    # 传播：nodes 已是后序（children 先于 parent），正序迭代即 children 先算。
    for node_id, op, attrs, child_ids in nodes:
        _propagate(node_id, op, attrs, child_ids)
    return [w for w in affected.values() if w.op != "column"]


def affected_root_window(
    plan: Any,
    *,
    field: str,
    changed_start: str,
    changed_end: str | None = None,
) -> AffectedWindow | None:
    """root 输出上必须重算的窗口（Change Impact 摘要）。"""
    affected = compute_change_impact(plan, field=field, changed_start=changed_start, changed_end=changed_end)
    if not affected:
        return None
    return affected[-1]
