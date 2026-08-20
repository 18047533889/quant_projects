# -*- coding: utf-8 -*-
"""R39 §7 —— Native Fusion：PERF-021 size model / PERF-022 binary-split +
NegativeFusionCache / PERF-023 locality grouping。

测试聚焦：
    (a) size model —— 更多 AST 节点 / window 表达式 → predicted compile_ms
        单调上升、chosen block size 单调不增（向更小方向）；
    (b) binary-split —— 4-root group 中单个失败根被隔离，其余 3 根仍融合
        （stub ``execute_multi_roots`` 只对指定根失败）；
    (c) NegativeFusionCacheKey —— 已知坏 (backend, plan_family_hash) 第二次
        直接跳过 multi-root（不再付失败成本）；
    (d) locality —— 同 window 不同 source 的 roots 比同 source 不同 window
        的 roots 得分更高（locality ordering 函数）。
"""
from __future__ import annotations

import pytest

from planner.fusion_size_model import (
    FusionSizeFeatures,
    estimate_fusion_cost,
)
from planner.logical_plan import PlanNode
from planner.negative_fusion_cache import (
    NEGATIVE_FUSION_CACHE,
    NegativeFusionCacheKey,
)
from planner.physical_factor_dag import PhysicalFactorTask
from runtime.task_resource_contract import TaskResourceContract

from planner.native_fusion import (
    NativeFusionGroup,
    adaptive_fusion_block_size,
    execute_fusion_group,
    extract_fusion_size_features,
    locality_score,
    plan_native_fusion_groups,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _task(
    task_id: str,
    node: PlanNode,
    *,
    backend: str = "duckdb_sql",
    source_scope: str = "s1",
    snap: str = "snap1",
    exec_scope: str = '{"market":"A"}',
    out_bytes: int = 100,
) -> PhysicalFactorTask:
    return PhysicalFactorTask(
        task_id=task_id,
        op=getattr(node, "op", "?"),
        task_type="ROOT",
        preferred_backend=backend,
        source_scope=source_scope,
        source_snapshot_id=snap,
        execution_scope=exec_scope,
        node_ref=node,
        resource_contract=TaskResourceContract(
            output_bytes=out_bytes, peak_memory_bytes=128
        ),
    )


def _win_node(task_id: str, window: int = 5, col: str = "close") -> PlanNode:
    """一个 ts_mean(window) 计划，node_id 带 task_id（供 stub 识别失败根）。"""
    return PlanNode(
        "ts_mean",
        [PlanNode("column", attrs={"name": col})],
        attrs={"window": window},
        node_id=task_id,
    )


class _Ctx:
    def __init__(self) -> None:
        self.runtime_stats = {}


def _per_root(task):
    return f"per:{task.task_id}"


def _stub_failing_backend(fail_ids, counter=None):
    class _Stub:
        def __init__(self):
            self.calls: list[tuple] = []

        def execute_multi_roots(self, nodes, ctx):
            if counter is not None:
                counter["n"] += 1
            self.calls.append(tuple(getattr(n, "node_id", None) for n in nodes))
            failing = [
                getattr(n, "node_id", None)
                for n in nodes
                if getattr(n, "node_id", None) in set(fail_ids)
            ]
            if failing:
                raise RuntimeError(f"multi-root fails for {failing}")
            return {
                getattr(n, "node_id", None): f"val:{getattr(n, 'node_id', None)}"
                for n in nodes
            }

    return _Stub()


# ---------------------------------------------------------------------------
# (a) PERF-021 size model
# ---------------------------------------------------------------------------


def _chain(depth: int, window: int) -> PlanNode:
    n: PlanNode = PlanNode("column", attrs={"name": "close"})
    for _ in range(depth):
        n = PlanNode("ts_mean", [n], attrs={"window": window})
    return n


def test_extract_features_counts_ast_and_windows():
    simple = PlanNode(
        "add",
        [
            PlanNode("column", attrs={"name": "close"}),
            PlanNode("column", attrs={"name": "volume"}),
        ],
    )
    complex_ = _chain(3, 20)  # 3 层 window chain = 4 个 AST 节点
    f_simple = extract_fusion_size_features([_task("s", simple)])
    f_complex = extract_fusion_size_features([_task("c", complex_)])
    assert f_simple.ast_node_count == 3
    assert f_simple.window_expression_count == 0
    assert f_simple.projection_count == 2  # close + volume 两个 source column
    assert f_complex.window_expression_count == 3
    assert f_simple.ast_node_count < f_complex.ast_node_count  # 3 < 4
    assert f_complex.projection_count >= 1


def test_size_model_from_plans_monotonic():
    """更多 AST 节点 / window → compile_ms 更高、chosen block 不增。"""
    simple_plan = PlanNode(
        "add",
        [
            PlanNode("column", attrs={"name": "close"}),
            PlanNode("column", attrs={"name": "volume"}),
        ],
    )
    simple_tasks = [_task(f"s{i}", simple_plan) for i in range(200)]

    def chain(depth: int, window: int) -> PlanNode:
        n: PlanNode = PlanNode("column", attrs={"name": "close"})
        for _ in range(depth):
            n = PlanNode("ts_mean", [n], attrs={"window": window})
        return n

    complex_tasks = [_task(f"c{i}", chain(10, 20)) for i in range(200)]

    f_simple = extract_fusion_size_features(simple_tasks)
    f_complex = extract_fusion_size_features(complex_tasks)
    assert f_simple.ast_node_count < f_complex.ast_node_count
    assert f_simple.window_expression_count < f_complex.window_expression_count
    # predicted compile_ms 单调上升
    assert estimate_fusion_cost(f_simple).compile_ms < estimate_fusion_cost(
        f_complex
    ).compile_ms
    # chosen block 向更小方向（单调不增）
    b_simple = adaptive_fusion_block_size(root_count=200, features=f_simple)
    b_complex = adaptive_fusion_block_size(root_count=200, features=f_complex)
    assert b_complex <= b_simple
    assert b_simple in (32, 64, 128, 256)
    assert b_complex in (32, 64, 128, 256)


def test_size_model_compile_and_block_monotonic_series():
    """按 feature 序列：compile_ms 严格上升、block 单调不增。"""
    prev_cost = None
    prev_block = None
    for ast in (50, 150, 400, 1000, 2500):
        f = FusionSizeFeatures(
            sql_chars=ast * 20,
            ast_node_count=ast,
            window_expression_count=max(1, ast // 20),
            projection_count=20,
            estimated_intermediate_bytes=2 * 1024**3,
            output_bytes=1024**3,
        )
        cost = estimate_fusion_cost(f).compile_ms
        block = adaptive_fusion_block_size(root_count=200, features=f)
        if prev_cost is not None:
            assert cost > prev_cost, "compile_ms 必须随 AST 规模严格上升"
            assert block <= prev_block, "chosen block 必须单调不增"
        prev_cost, prev_block = cost, block
    # 低复杂度选大 block、高复杂度选小 block（方向正确）
    low_block = adaptive_fusion_block_size(
        root_count=200,
        features=FusionSizeFeatures(
            ast_node_count=50, sql_chars=1000, window_expression_count=2,
            projection_count=5, estimated_intermediate_bytes=1024**3,
            output_bytes=512 * 1024**2,
        ),
    )
    high_block = adaptive_fusion_block_size(
        root_count=200,
        features=FusionSizeFeatures(
            ast_node_count=5000, sql_chars=100000, window_expression_count=500,
            projection_count=50, estimated_intermediate_bytes=8 * 1024**3,
            output_bytes=4 * 1024**3,
        ),
    )
    assert high_block <= low_block


def test_legacy_signature_preserved():
    """旧签名（无 features）仍然工作，返回 32..256。"""
    assert 32 <= adaptive_fusion_block_size(
        root_count=100, expression_complexity=1.0, estimated_output_bytes=1024**3
    ) <= 256
    assert 32 <= adaptive_fusion_block_size(
        root_count=10, expression_complexity=1.0,
        estimated_output_bytes=64 * 1024**2,
    ) <= 256


# ---------------------------------------------------------------------------
# (b) PERF-022 binary-split recovery
# ---------------------------------------------------------------------------


def test_binary_split_isolates_failing_root():
    """4-root group 中根 r3 失败 → 只隔离 r3，其余 3 根仍融合。"""
    NEGATIVE_FUSION_CACHE.clear()
    roots = [_task(f"r{i}", _win_node(f"r{i}", window=5)) for i in range(4)]
    task_by_id = {t.task_id: t for t in roots}
    group = NativeFusionGroup(
        group_id=1, backend="stub", source_scope="s1",
        roots=tuple(t.task_id for t in roots),
    )
    backend = _stub_failing_backend(fail_ids={"r3"})
    ctx = _Ctx()
    results = execute_fusion_group(
        group, backend=backend, task_by_id=task_by_id, ctx=ctx,
        execute_root=_per_root,
    )
    # r0/r1/r2 融合产出 val:rX；r3 走 per-root fallback。
    assert results["r0"] == "val:r0"
    assert results["r1"] == "val:r1"
    assert results["r2"] == "val:r2"
    assert results["r3"] == "per:r3"
    # 调用序列：[r0..r3]=4 失败 → [r0,r1]=2 成功 → [r2,r3]=2 失败 →
    # [r2]=1 成功 → [r3]=1 失败回退。
    assert [len(c) for c in backend.calls] == [4, 2, 2, 1, 1]
    stats = ctx.runtime_stats
    assert stats.get("native_fusion_binary_split", 0) == 2
    assert stats.get("native_fusion_executed", 0) == 2  # [r0,r1] 与 [r2]
    assert stats.get("native_fusion_fallback", 0) == 1  # r3
    assert stats.get("native_fusion_planned", 0) == 1


def test_binary_split_all_success_is_single_fusion():
    """无失败根 → 整个 group 一次融合，不拆。"""
    NEGATIVE_FUSION_CACHE.clear()
    roots = [_task(f"q{i}", _win_node(f"q{i}", window=7)) for i in range(4)]
    task_by_id = {t.task_id: t for t in roots}
    group = NativeFusionGroup(
        group_id=2, backend="stub", source_scope="s1",
        roots=tuple(t.task_id for t in roots),
    )
    backend = _stub_failing_backend(fail_ids=set())
    ctx = _Ctx()
    results = execute_fusion_group(
        group, backend=backend, task_by_id=task_by_id, ctx=ctx,
        execute_root=_per_root,
    )
    assert results == {f"q{i}": f"val:q{i}" for i in range(4)}
    assert [len(c) for c in backend.calls] == [4]
    assert ctx.runtime_stats.get("native_fusion_binary_split", 0) == 0
    assert ctx.runtime_stats.get("native_fusion_executed", 0) == 1


# ---------------------------------------------------------------------------
# (c) PERF-022 NegativeFusionCache
# ---------------------------------------------------------------------------


def test_negative_cache_key_equality_and_hash():
    key1 = NegativeFusionCacheKey(
        backend="duckdb_sql", backend_version="0.10.2", plan_family_hash="abc123",
        parameter_shape=("p",), source_shape=(("s1", "snap1", ("close",)),),
    )
    key2 = NegativeFusionCacheKey(
        backend="duckdb_sql", backend_version="0.10.2", plan_family_hash="abc123",
        parameter_shape=("p",), source_shape=(("s1", "snap1", ("close",)),),
    )
    key3 = NegativeFusionCacheKey(
        backend="duckdb_sql", backend_version="0.10.2", plan_family_hash="different",
        parameter_shape=("p",), source_shape=(("s1", "snap1", ("close",)),),
    )
    assert key1 == key2
    assert key1 != key3
    NEGATIVE_FUSION_CACHE.clear()
    assert not NEGATIVE_FUSION_CACHE.is_blocked(key1)
    NEGATIVE_FUSION_CACHE.mark_blocked(key1)
    assert NEGATIVE_FUSION_CACHE.is_blocked(key2)  # 相等键 → blocked
    assert not NEGATIVE_FUSION_CACHE.is_blocked(key3)  # 不同 family → 不 block
    NEGATIVE_FUSION_CACHE.clear()


def test_negative_cache_skips_bad_combination_on_second_attempt():
    """已知坏 (backend, plan_family_hash) 第二次直接跳过 multi-root。"""
    NEGATIVE_FUSION_CACHE.clear()
    # 两个根用不同 window，保证单根负缓存键彼此不同（避免结构相同提前命中）。
    roots = [
        _task("n0", _win_node("n0", window=5)),
        _task("n1", _win_node("n1", window=10)),
    ]
    task_by_id = {t.task_id: t for t in roots}
    group = NativeFusionGroup(
        group_id=3, backend="stub", source_scope="s1",
        roots=tuple(t.task_id for t in roots),
    )
    counter = {"n": 0}
    backend = _stub_failing_backend(fail_ids={"n0", "n1"}, counter=counter)

    res1 = execute_fusion_group(
        group, backend=backend, task_by_id=task_by_id, ctx=_Ctx(),
        execute_root=_per_root,
    )
    first_calls = counter["n"]
    # [n0,n1] → 失败拆 [n0]、[n1] 都失败 → 3 次 multi 尝试。
    assert first_calls == 3
    assert res1 == {"n0": "per:n0", "n1": "per:n1"}

    # 第二次：整组已负缓存 → 一个 multi 都不打，直接 per-root。
    res2 = execute_fusion_group(
        group, backend=backend, task_by_id=task_by_id, ctx=_Ctx(),
        execute_root=_per_root,
    )
    assert counter["n"] == first_calls, "已知坏组合不能再次付失败成本"
    assert res2 == {"n0": "per:n0", "n1": "per:n1"}
    NEGATIVE_FUSION_CACHE.clear()


# ---------------------------------------------------------------------------
# (d) PERF-023 locality ordering
# ---------------------------------------------------------------------------


def test_locality_window_beats_source():
    """同 window 不同 source > 同 source 不同 window。"""
    a = _task("A", _win_node("A", window=5, col="close"))
    b = _task("B", _win_node("B", window=5, col="volume"))
    c = _task("C", _win_node("C", window=20, col="close"))
    same_window = locality_score([a, b])
    same_source = locality_score([a, c])
    assert same_window > same_source


def test_locality_score_range_and_deterministic():
    a = _task("A", _win_node("A", window=5, col="close"))
    b = _task("B", _win_node("B", window=5, col="volume"))
    assert locality_score([a, b]) == locality_score([a, b])
    assert locality_score([a]) == 0.0
    # 完全一致 → 分数最高；无共同特征 → 分数低。
    identical = locality_score([a, _task("D", _win_node("D", window=5, col="close"))])
    no_common = locality_score(
        [a, _task("E", _win_node("E", window=30, col="open"))]
    )
    assert identical > no_common


def test_plan_native_fusion_groups_preserves_primary_grouping():
    """主分组键 (backend, source_scope, execution_scope) 不变 + locality 排序。"""
    tasks = [
        _task("A", _win_node("A", window=5, col="close"), backend="duckdb_sql"),
        _task("B", _win_node("B", window=5, col="volume"), backend="duckdb_sql"),
        # 不同 execution_scope → 不能与 A/B 同组
        _task("C", _win_node("C", window=5), backend="duckdb_sql",
              exec_scope='{"market":"B"}'),
    ]
    capability = {"duckdb_sql": True}
    groups = plan_native_fusion_groups(tasks, backend_capability=capability)
    by_scope_groups = [g for g in groups if g.source_scope == "s1"]
    assert len(by_scope_groups) == 1
    assert set(by_scope_groups[0].roots) == {"A", "B"}
    assert all(g.source_scope == "s1" for g in by_scope_groups)


def test_uncertified_backend_not_planned():
    """无 multi-root capability 的 backend 不生成 fusion 组（诚实）。"""
    tasks = [
        _task("A", _win_node("A", window=5), backend="pandas_numpy"),
        _task("B", _win_node("B", window=5), backend="pandas_numpy"),
    ]
    groups = plan_native_fusion_groups(
        tasks, backend_capability={"pandas_numpy": False}
    )
    assert groups == []
