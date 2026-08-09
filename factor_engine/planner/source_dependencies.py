"""执行前(preflight)二级源依赖清单。

``compute_data_scope`` 只绑定 anchor source 与执行语义，不绑定计划中
``column`` 节点携带的二级 SourceRef 依赖（StockIncome/StockBalance 等）。
本模块在 plan 侧递归收集这些 SourceRef，输出规范 JSON 清单及其 sha256 前缀，
供 preflight / plan 缓存键把二级依赖纳入作用域。

只 import ``api.source_ref`` 与 ``planner.logical_plan``，避免循环依赖。
"""

from __future__ import annotations

import hashlib
import json

from api.source_ref import SourceRefSpec, decode_source_ref
from planner.logical_plan import PlanNode

# 与 api.source_ref 的 _PREFIX 保持一致；不 import 私有名以避免耦合。
_SOURCE_REF_PREFIX = "__fe_source_ref_v1__"


def _ref_canonical(spec: SourceRefSpec) -> str:
    """把 SourceRefSpec 折叠为规范 JSON 字符串（字段排序、键排序）。

    R10-P0-018: the canonical payload includes ``dialect`` /
    ``dialect_version`` — two tables with the same columns but different
    semantic dialect versions must NOT share a source-dependency hash.  The
    field's *contract* identity (PIT policy, unit, role) is keyed by
    ``(table, field)`` inside the field catalog and is covered by the field
    name itself; the per-field catalog hash is intentionally NOT duplicated
    here (no parallel contract truth source).
    """
    return json.dumps(
        {
            "table": spec.table,
            "field": spec.field,
            "params": dict(sorted(spec.params, key=lambda kv: kv[0])),
            "transform": spec.transform,
            "transform_params": dict(sorted(spec.transform_params, key=lambda kv: kv[0])),
            "dialect": spec.dialect,
            "dialect_version": spec.dialect_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_source_dependency_manifest(plan: PlanNode) -> tuple[str, ...]:
    """递归遍历计划，收集所有 SourceRef 二级依赖的规范 JSON 清单。

    仅处理 ``op == "column"`` 且 ``attrs["name"]`` 以 ``__fe_source_ref_v1__``
    开头的节点；普通列跳过。返回去重后排序的 ``tuple[str, ...]``。

    Round-8 audit #392（三态语义的 plan 侧配合）：若名字带 source-ref 前缀但
    payload 损坏（``decode_source_ref`` 抛 ``ValueError``），**不吞异常**——向
    调用方传播，调用方应把它当作不可缓存/需显式失败处理。
    """
    found: set[str] = set()

    def walk(node: PlanNode) -> None:
        if node.op == "column":
            name = node.attrs.get("name")
            if isinstance(name, str) and name.startswith(_SOURCE_REF_PREFIX):
                spec = decode_source_ref(name)
                found.add(_ref_canonical(spec))
        for child in node.inputs or ():
            walk(child)

    walk(plan)
    return tuple(sorted(found))


def source_dependency_hash(plan: PlanNode) -> str:
    """返回 source 依赖清单的 sha256 前缀哈希（与 data_scope 同为 24 位）。"""
    manifest = build_source_dependency_manifest(plan)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
