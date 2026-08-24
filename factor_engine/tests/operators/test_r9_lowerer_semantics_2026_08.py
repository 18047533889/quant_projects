# -*- coding: utf-8 -*-
"""R9-P0-002 回归测试：Lowerer/PlanNode 重建必须携带 IR 的 ``semantic_attrs``。

评审结论：Typed IR → Plan 时 ``semantic_attrs``（unit/domain/frequency/available_at
/source_vintage/flow_semantics/price_basis/universe/semantic_kind 与 mixed_* 冲突标记）
此前全部丢失，会破坏 backend type validation / optimizer rewrite validation / SQL
lowering / PIT validation / grain validation / unit validation。

本测试锁定修复后：
1. ``Lowerer.to_logical_plan`` 把 IR 的 ``semantic_attrs`` 逐键拷贝到 PlanNode；
2. 该拷贝是独立的新 dict（浅拷贝），与 plain ``attrs`` 不共享可变别名；
3. PlanNode 重建站点（``deep_copy_plan`` 等）同样透传 ``semantic_attrs``。

本文件不触发 ``load_all()``，只验证 lowerer/PlanNode 数据结构路径。
"""

from __future__ import annotations

from factor_engine.ir.nodes import IRNode
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.lowerer import Lowerer


def _semantic() -> dict:
    return {
        "unit": "price",
        "domain": "equity",
        "frequency": "1d",
        "available_at": "2024-01-01",
        "source_vintage": "v2",
        "flow_semantics": "stock",
        "price_basis": "last",
        "universe": "A-share",
        "semantic_kind": "field",
    }


def _leaf_semantic() -> dict:
    return {
        "unit": "price",
        "domain": "equity",
        "frequency": "1d",
        "available_at": "2024-01-01",
        "source_vintage": "v2",
        "flow_semantics": "stock",
        "price_basis": "last",
        "universe": "A-share",
        "semantic_kind": "field",
        "mixed_unit": ["price", "volume"],  # mixed_* 冲突标记原样保留
    }


def _make_ir() -> IRNode:
    return IRNode(
        op="add",
        inputs=[
            IRNode(
                op="column",
                attrs={"name": "close"},
                semantic_attrs=_leaf_semantic(),
            ),
            IRNode(op="literal", attrs={"value": 1.0}),
        ],
        semantic_attrs=_semantic(),
    )


def test_lowerer_carries_semantic_attrs() -> None:
    plan = Lowerer().to_logical_plan(_make_ir())
    assert dict(plan.semantic_attrs) == _semantic()
    # 叶子节点同样携带（含 mixed_* 冲突标记 verbatim）
    assert dict(plan.inputs[0].semantic_attrs) == _leaf_semantic()
    # 与 plain attrs 完全独立
    assert plan.attrs is not plan.semantic_attrs
    assert "semantic_kind" not in plan.attrs


def test_lowered_semantic_attrs_is_a_new_dict_no_alias() -> None:
    ir = _make_ir()
    plan = Lowerer().to_logical_plan(ir)
    sa_before = dict(plan.semantic_attrs)
    # 修改 IR 的 semantic_attrs 后重新下降，原 plan 不受影响（拷贝非引用）
    mutated = IRNode(
        op=ir.op,
        inputs=ir.inputs,
        attrs=dict(ir.attrs),
        semantic_attrs={**ir.semantic_attrs, "frequency": "1w"},
    )
    plan2 = Lowerer().to_logical_plan(mutated)
    assert dict(plan.semantic_attrs) == sa_before
    assert dict(plan2.semantic_attrs)["frequency"] == "1w"


def test_plain_attrs_mutation_does_not_mutate_semantic_attrs() -> None:
    plan = Lowerer().to_logical_plan(_make_ir())
    sa_before = dict(plan.semantic_attrs)
    attrs_before = dict(plan.attrs)
    # 重建节点并改写 plain attrs（模拟下游 transform），semantic_attrs 不受影响
    rebuilt = PlanNode(
        op=plan.op,
        inputs=plan.inputs,
        attrs={**plan.attrs, "window": 5},
        semantic_attrs=dict(plan.semantic_attrs),
        node_id=plan.node_id,
    )
    assert dict(rebuilt.attrs) == {**attrs_before, "window": 5}
    assert dict(rebuilt.semantic_attrs) == sa_before
    # 两个字段底层是不同 dict，互不 alias
    assert plan.attrs is not plan.semantic_attrs
    assert rebuilt.attrs is not rebuilt.semantic_attrs


def test_plan_rebuild_sites_carry_semantic_attrs() -> None:
    from factor_engine.planner.cse import deep_copy_plan

    plan = Lowerer().to_logical_plan(_make_ir())
    copied = deep_copy_plan(plan)
    assert copied is not plan
    assert dict(copied.semantic_attrs) == dict(plan.semantic_attrs)
    assert dict(copied.inputs[0].semantic_attrs) == dict(plan.inputs[0].semantic_attrs)
    # 修改拷贝的 plain attrs 不影响原 semantic_attrs
    copied2 = deep_copy_plan(copied)
    assert dict(copied2.semantic_attrs) == dict(plan.semantic_attrs)
