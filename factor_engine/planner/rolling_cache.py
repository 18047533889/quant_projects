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
        "delay",
        "ts_delay",
    }
)


@dataclass(frozen=True)
class RollingCacheEntry:
    structural_id: str
    op: str
    window: int | None
    input_column: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_id": self.structural_id,
            "op": self.op,
            "window": self.window,
            "input_column": self.input_column,
        }


def _first_col_ref(node: PlanNode) -> str | None:
    if node.op in {"col", "column"}:
        return str(node.attrs.get("name") or node.attrs.get("column") or "")
    for child in node.inputs:
        found = _first_col_ref(child)
        if found:
            return found
    return None


def _window_from_attrs(op: str, attrs: dict[str, Any]) -> int | None:
    for key in ("window", "d", "period", "n"):
        if key in attrs and attrs[key] is not None:
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue
    if op in {"delay", "ts_delay", "ts_delta"}:
        for key in ("periods", "lag"):
            if key in attrs and attrs[key] is not None:
                try:
                    return int(attrs[key])
                except (TypeError, ValueError):
                    continue
    return None


def rolling_entry_from_node(structural_id: str, node: PlanNode) -> RollingCacheEntry | None:
    if node.op not in ROLLING_OPS:
        return None
    return RollingCacheEntry(
        structural_id=structural_id,
        op=node.op,
        window=_window_from_attrs(node.op, node.attrs),
        input_column=_first_col_ref(node),
    )


def summarize_rolling_cache(shared_nodes: dict[str, PlanNode]) -> dict[str, Any]:
    """从 CSE ``shared_nodes`` 提取 rolling 共享项摘要。"""
    entries: list[RollingCacheEntry] = []
    for sid, node in shared_nodes.items():
        entry = rolling_entry_from_node(sid, node)
        if entry is not None:
            entries.append(entry)
    return {
        "rolling_shared_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
