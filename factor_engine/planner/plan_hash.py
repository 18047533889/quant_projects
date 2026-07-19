"""逻辑计划子树的结构化哈希（与具体列数据无关），供 CSE 与执行期子树缓存键复用。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from planner.logical_plan import PlanNode


def _jsonable(v: Any) -> Any:
    """将 attrs 值转为可 JSON 序列化的稳定表示。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in sorted(v.items())}
    raise TypeError(f"unsupported plan attribute type: {type(v).__name__}")


def structural_key(node: PlanNode, memo: dict[int, str] | None = None) -> str:
    """递归计算子树结构键；同一子树形状得到相同字符串。

    参数：
        node: 待哈希的计划节点
        memo: 可选 ``id(node) → key`` 缓存，避免重复遍历共享引用

    返回：
        稳定 JSON 字符串，编码 ``op``、排序后的 ``attrs`` 与子节点键列表
    """
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
    semantic_version = payload["attrs"].get("semantic_version", 1)
    payload["semantic_version"] = semantic_version
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(s.encode("utf-8")).hexdigest()
    memo[nid] = key
    return key


def plan_cache_key(node: PlanNode) -> str:
    """与 :func:`structural_key` 等价；别名用于执行期缓存命名。

    参数：
        node: 待缓存的子计划根节点

    返回：
        结构哈希字符串，可作为 SQL 子树 sid 或运行时缓存键
    """
    return structural_key(node)
