"""Rolling 子表达式缓存摘要：配合 CSE shared_nodes 可观测与调度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planner.logical_plan import PlanNode

ROLLING_OPS = frozenset(
    {
        "ts_mean",
        "ts_std",
        "ts_std",
        "ts_std_dev",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_delta",
        "ts_rank",
        "ts_corr",
        "ts_correlation",
        "ts_cov",
        "ts_covariance",
        "ts_decay_linear",
        "ts_delay",
    }
)


@dataclass(frozen=True)
class RollingCacheEntry:
    """单条 rolling 共享子树的摘要条目。

    字段：
        structural_id: CSE 结构键 / sid
        op: rolling 算子名
        window: 窗口长度（无法解析时为 ``None``）
        input_column: 主输入列名（无法解析时为 ``None``）
    """

    structural_id: str
    op: str
    window: int | None
    input_column: str | None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。

        返回：
            含 structural_id、op、window、input_column 的字典
        """
        return {
            "structural_id": self.structural_id,
            "op": self.op,
            "window": self.window,
            "input_column": self.input_column,
        }


def _first_col_ref(node: PlanNode) -> str | None:
    """深度优先查找子树中首个列引用名。"""
    if node.op in {"col", "column"}:
        return str(node.attrs.get("name") or node.attrs.get("column") or "")
    for child in node.inputs:
        found = _first_col_ref(child)
        if found:
            return found
    return None


def _window_from_attrs(op: str, attrs: dict[str, Any]) -> int | None:
    """从算子 attrs 解析整数窗口参数。"""
    for key in ("window", "d", "period", "n"):
        if key in attrs and attrs[key] is not None:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue
    if op in {"ts_delay", "delay", "ts_delta"}:
        for key in ("periods", "lag"):
            if key in attrs and attrs[key] is not None:
                try:
                    return int(attrs[key])
                except (TypeError, ValueError):
                    continue
    return None


def _window_from_literals(node: PlanNode) -> int | None:
    """从 positional literal child inputs 解析 window/lag 参数。

    ``ts_mean(close, 20)`` 的 20 是 ``op=="literal"`` 的 child input 而非 attrs
    （见 ``planner/lowerings/_helpers.py`` 的 ``ts_mean``/``ts_delay`` 等），因此
    ``_window_from_attrs`` 会漏掉窗口值。本函数遍历 ``node.inputs[1:]``，跳过
    column / materialized_series / plan_ref 等非 literal 子节点，取第一个
    ``op=="literal"`` 子节点作为窗口值；对 ``ts_delay``/``delay``/``ts_delta``
    该 literal 即 lag（同一解析逻辑）。

    参数：
        node: 候选 rolling 计划节点

    返回：
        整数窗口/lag；非 rolling 算子或无法解析时返回 ``None``
    """
    if node.op not in ROLLING_OPS:
        return None
    for child in node.inputs[1:]:
        if getattr(child, "op", None) != "literal":
            continue
        value = (child.attrs or {}).get("value")
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def rolling_entry_from_node(structural_id: str, node: PlanNode) -> RollingCacheEntry | None:
    """从计划节点构造 rolling 缓存摘要条目。

    参数：
        structural_id: 共享子树的结构 sid
        node: 候选 rolling 算子节点

    返回：
        ``RollingCacheEntry``；非 rolling 算子时返回 ``None``
    """
    if node.op not in ROLLING_OPS:
        return None
    return RollingCacheEntry(
        structural_id=structural_id,
        op=node.op,
        window=_window_from_attrs(node.op, node.attrs),
        input_column=_first_col_ref(node),
    )


def summarize_rolling_cache(shared_nodes: dict[str, PlanNode]) -> dict[str, Any]:
    """从 CSE ``shared_nodes`` 提取 rolling 共享项摘要。

    参数：
        shared_nodes: 结构 CSE 产出的共享子树字典

    返回：
        含 ``rolling_shared_count`` 与 ``entries`` 列表的摘要字典
    """
    entries: list[RollingCacheEntry] = []
    for sid, node in shared_nodes.items():
        entry = rolling_entry_from_node(sid, node)
        if entry is not None:
            entries.append(entry)
    return {
        "rolling_shared_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
