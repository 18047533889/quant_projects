# -*- coding: utf-8 -*-
"""R27-164/232: 列重叠不是执行依赖 —— 共享只读 close 的因子必须能同层并行。

R27-003：``ts_mean(close,5)`` 与 ``ts_std(close,20)`` 共享只读 close，但共享只读
≠ 执行依赖；恰恰更适合放同一 locality wave。旧 ``_parallel_layers`` 用「列集合
不相交」才允许同层 —— 是性能上很大的 false dependency。
"""
from __future__ import annotations

from api.columns import col
from api.factor import Factor
from planner.dependency_graph import build_factor_batch_graph


def _analysis(name: str, cols: set[str], lookback: int = 0):
    from ir.analyzer import AnalysisResult
    from ir.nodes import IRNode

    return AnalysisResult(
        ir=IRNode(op="column", attrs={"name": next(iter(cols))}),
        lookback=lookback,
        has_ts_op=lookback > 0,
        has_cs_op=False,
        referenced_columns=cols,
    )


def test_shared_readonly_column_is_not_execution_dependency():
    # A/B/C 全部共享只读 close（R27-003 示例）。
    factors = [
        Factor(name="A", expr=col("close")),
        Factor(name="B", expr=col("close")),
        Factor(name="C", expr=col("close")),
    ]
    analyses = {
        "A": _analysis("A", {"close"}, lookback=5),
        "B": _analysis("B", {"close"}, lookback=20),
        "C": _analysis("C", {"close"}, lookback=60),
    }
    graph = build_factor_batch_graph(factors, analyses)
    # 旧行为：三层（每层只有 1 个因子，因列冲突）。新行为：单层 3 因子可并行。
    layers = graph.parallel_layers
    assert len(layers) == 1, f"shared readonly close must not split layers: {layers}"
    assert set(layers[0]) == {"A", "B", "C"}
    # 列重叠仍作为 locality hint 保留（R27-164）。
    assert any({"A", "B", "C"} <= set(g) for g in graph.locality_groups)


def test_locality_groups_kept_as_hint():
    factors = [
        Factor(name="a", expr=col("close")),
        Factor(name="b", expr=col("close") + col("open")),
        Factor(name="c", expr=col("volume")),
    ]
    analyses = {
        "a": _analysis("a", {"close"}),
        "b": _analysis("b", {"close", "open"}),
        "c": _analysis("c", {"volume"}),
    }
    graph = build_factor_batch_graph(factors, analyses)
    assert graph.column_union == frozenset({"close", "open", "volume"})
    assert ["a", "b"] in graph.locality_groups or ["b", "a"] in graph.locality_groups
    # 无真实依赖 → 单层全并行（c 不再因列不同而单独串行）。
    assert len(graph.parallel_layers) == 1
    assert set(graph.parallel_layers[0]) == {"a", "b", "c"}


def test_true_factor_dependency_still_serializes():
    # 若一因子 root 真实内嵌另一因子 root（对象 identity），必须串行。
    from ir.analyzer import AnalysisResult
    from ir.nodes import IRNode

    inner = IRNode(op="column", attrs={"name": "close"})
    # B 的 IR 内嵌 A 的 IR root → B 依赖 A。
    analyses = {
        "A": AnalysisResult(ir=inner, lookback=0, has_ts_op=False, has_cs_op=False,
                            referenced_columns={"close"}),
        "B": AnalysisResult(ir=IRNode(op="neg", inputs=[inner]), lookback=0,
                            has_ts_op=False, has_cs_op=False, referenced_columns={"close"}),
    }
    factors = [Factor(name="A", expr=col("close")), Factor(name="B", expr=col("close"))]
    graph = build_factor_batch_graph(factors, analyses)
    b_node = graph.nodes["B"]
    assert "A" in b_node.dependencies, "true embedded subplan must be a dependency"
    layers = graph.parallel_layers
    # A 在 B 前一层（A 串行先于 B）。
    assert layers[0] == ["A"] and layers[1] == ["B"], f"layers={layers}"
