"""逻辑计划子树的结构化哈希（与具体列数据无关），供 CSE 与执行期子树缓存键复用。"""

from __future__ import annotations

import json
from typing import Any

from planner.logical_plan import PlanNode


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in sorted(v.items())}
    return repr(v)


def structural_key(node: PlanNode, memo: dict[int, str] | None = None) -> str:
    """递归结构键；同一子树形状得到相同字符串。"""
    if memo is None:
        memo = {}
    nid = id(node)
    if nid in memo:
        return memo[nid]
    child_keys = [structural_key(c, memo) for c in node.inputs]
    payload = {
        "op": node.op,
        "attrs": {k: _jsonable(v) for k, v in sorted(node.attrs.items())},
        "in": child_keys,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    memo[nid] = s
    return s


def plan_cache_key(node: PlanNode) -> str:
    """与 :func:`structural_key` 等价；别名用于执行期缓存命名。"""
    return structural_key(node)
