"""Phase 20 依赖图与分片单元测试。"""

from __future__ import annotations

from api.columns import col
from api.factor import Factor
from planner.dependency_graph import build_factor_batch_graph
from runtime.shard_materialize import shard_by_hash, shard_factor_ids
from runtime.task_queue import shard_config_paths


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


def test_batch_graph_column_overlap_groups():
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
    assert ["a", "b"] in graph.column_overlap_groups or ["b", "a"] in graph.column_overlap_groups
    assert any("c" in layer for layer in graph.parallel_layers)


def test_shard_factor_ids_stable():
    ids = [f"f{i}" for i in range(30)]
    a = shard_factor_ids(ids, shard_index=0, shard_count=3)
    b = shard_factor_ids(ids, shard_index=0, shard_count=3)
    assert a == b
    assert len(a) + len(shard_factor_ids(ids, shard_index=1, shard_count=3)) + len(
        shard_factor_ids(ids, shard_index=2, shard_count=3)
    ) == len(ids)


def test_shard_config_paths_delegates(tmp_path):
    paths = [tmp_path / f"c{i}.yaml" for i in range(9)]
    for p in paths:
        p.write_text("x: 1\n", encoding="utf-8")
    direct = shard_by_hash(
        paths,
        shard_index=1,
        shard_count=3,
        key_fn=lambda p: str(p.resolve()),
    )
    wrapped = shard_config_paths(paths, shard_index=1, shard_count=3)
    assert [str(p) for p in wrapped] == [str(p) for p in direct]
