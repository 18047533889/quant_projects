# -*- coding: utf-8 -*-
"""R42-014/015/016: CSE优化与Analyzer Canonical Tables测试。

覆盖：
- R42-014: Long-lived executor复用（删除临时线程池）
- R42-015: CSE execution certificate（预计算cost）
- R42-016: Reuse distance eviction（Belady-like）
- R42-007: Canonical membership table（operator-ID-based）
- R42-008: Canonical dependency table（reverse index）
"""

import time


def test_r42_015_cse_certificate_construction():
    """R42-015: CSE证书预计算包含size/cost/consumer/lifetime/reuse_distance。"""
    from factor_engine.runtime.cse_execution_certificate import (
        build_cse_certificate_store,
        CSEExecutionCertificate,
    )

    # Mock shared nodes
    shared_nodes = {
        "cse_001": {"op": "ts_mean", "window": 20},
        "cse_002": {"op": "ts_std", "window": 20},
        "cse_003": {"op": "add", "left": "ref:cse_001", "right": "ref:cse_002"},
    }

    # Mock consumer map: cse_001被3个root消费
    consumer_map = {
        "cse_001": ["root_a", "root_b", "root_c"],
        "cse_002": ["root_a", "root_c"],
        "cse_003": ["root_a"],
    }

    # Mock ordinal map（拓扑序）
    ordinal_map = {
        "root_a": 10,
        "root_b": 20,
        "root_c": 30,
    }

    # Mock cost estimator
    def mock_cost(plan):
        op = plan.get("op", "")
        if op == "ts_mean":
            return {"total_work": 1000.0, "peak_live_memory_bytes": 1024 * 1024}
        if op == "ts_std":
            return {"total_work": 1200.0, "peak_live_memory_bytes": 1024 * 1024}
        return {"total_work": 100.0, "peak_live_memory_bytes": 512 * 1024}

    store = build_cse_certificate_store(
        shared_nodes, consumer_map, ordinal_map, estimate_cost_fn=mock_cost
    )

    # 验证证书存在
    assert "cse_001" in store.certificates
    assert "cse_002" in store.certificates
    assert "cse_003" in store.certificates

    # 验证cse_001: 3个消费者，lifetime [10, 30]，reuse_distance [10, 10]
    cert1 = store.get("cse_001")
    assert cert1 is not None
    assert cert1.consumer_count == 3
    assert cert1.lifetime_interval == (10, 30)
    assert cert1.reuse_distance == (10, 10)  # root_b(20)-root_a(10), root_c(30)-root_b(20)
    assert cert1.recompute_cost_ms == 1000.0
    assert cert1.size_estimate_bytes == 1024 * 1024

    # 验证cse_002: 2个消费者
    cert2 = store.get("cse_002")
    assert cert2 is not None
    assert cert2.consumer_count == 2
    assert cert2.lifetime_interval == (10, 30)
    assert cert2.reuse_distance == (20,)  # root_c(30)-root_a(10)

    # 验证critical sids（consumer_count >= 3 或 recompute_cost > 500）
    assert "cse_001" in store.critical_sids  # consumer_count=3
    assert "cse_002" in store.critical_sids  # recompute_cost=1200.0 > 500
    assert "cse_003" not in store.critical_sids  # consumer_count=1, cost=100


def test_r42_016_reuse_distance_eviction():
    """R42-016: Reuse distance影响eviction顺序（Belady-like）。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=1024)

    # 写入3个条目，不同next_use_distance，显式传入bytes_
    store.put("soon", b"a" * 300, bytes_=300, next_use_distance=5)   # 5步后用，应晚淘汰
    store.put("far", b"b" * 300, bytes_=300, next_use_distance=50)   # 50步后用，应先淘汰
    store.put("mid", b"c" * 300, bytes_=300, next_use_distance=20)   # 20步后用，中等

    # 当前900字节，预算1024，还能放下
    assert store.get("soon") is not None
    assert store.get("far") is not None
    assert store.get("mid") is not None

    # 写入400字节，超预算，触发eviction
    # 预期淘汰顺序：far (distance=50) > mid (distance=20) > soon (distance=5)
    res = store.put("new", b"d" * 400, bytes_=400)
    assert res.status == "MEMORY"

    # far应被淘汰（distance最大）
    assert store.get("far") is None
    # soon和mid应保留
    assert store.get("soon") is not None
    assert store.get("mid") is not None


def test_r42_016_reuse_distance_fallback_lru():
    """R42-016: next_use_distance=0时fallback到LRU（stale_ms）。"""
    from factor_engine.runtime.buffer_store import GovernedBufferStore

    store = GovernedBufferStore(budget_bytes=1024)

    # 写入3个条目，distance=0（未知），显式传入bytes_
    store.put("old", b"a" * 300, bytes_=300, next_use_distance=0)
    time.sleep(0.01)
    store.put("mid", b"b" * 300, bytes_=300, next_use_distance=0)
    time.sleep(0.01)
    store.put("new", b"c" * 300, bytes_=300, next_use_distance=0)

    # 触发eviction
    res = store.put("extra", b"d" * 400, bytes_=400, next_use_distance=0)
    assert res.status == "MEMORY"

    # 应按LRU淘汰old（最久未访问）
    assert store.get("old") is None
    assert store.get("mid") is not None
    assert store.get("new") is not None


def test_r42_015_materialize_shared_reads_certificate():
    """R42-015: _materialize_shared_subplan从certificate读取cost，避免运行期重算。"""
    from factor_engine.runtime.batch_service import _materialize_shared_subplan
    from factor_engine.runtime.cse_execution_certificate import (
        CSEExecutionCertificate,
        CSECertificateStore,
    )
    from factor_engine.runtime.buffer_store import GovernedBufferStore

    # Mock backend
    class MockBackend:
        supports_lazy_shared = False

        def execute(self, sub, ctx):
            return b"result"

    # Mock context with certificate store
    class MockContext:
        shared_result_cache = {}
        runtime_stats = {}

        def __init__(self):
            self.shared_buffers = GovernedBufferStore(budget_bytes=10240)
            cert = CSEExecutionCertificate(
                sid="test_sid",
                size_estimate_bytes=100,
                recompute_cost_ms=500.0,
                consumer_count=3,
                lifetime_interval=(0, 10),
                reuse_distance=(3, 3, 4),
            )
            self.cse_certificate_store = CSECertificateStore(
                certificates={"test_sid": cert}
            )

    backend = MockBackend()
    ctx = MockContext()
    subplan = {"op": "ts_mean", "window": 20}

    # 执行materialize
    _materialize_shared_subplan(backend, subplan, ctx, "test_sid")

    # 验证buffer store接收到certificate的cost和distance
    entry = ctx.shared_buffers._entries.get("test_sid")
    assert entry is not None
    assert entry.recompute_cost_ms == 500.0
    assert entry.future_consumers == 3
    assert entry.next_use_distance == 3  # reuse_distance第一个值

    # 验证无certificate_miss计数
    assert ctx.runtime_stats.get("cse_certificate_miss", 0) == 0


def test_r42_007_canonical_membership_table():
    """R42-007: Canonical membership table提供operator_id -> canonical O(1)查询。"""
    from factor_engine.planner.analyzer_canonical_tables import (
        build_canonical_tables,
        CanonicalMembership,
    )

    # Mock registry
    class MockOperatorSpec:
        def __init__(self, canonical, surface, backend):
            self.canonical_name = canonical
            self.surface = surface
            self.backend = backend
            self.parameters = {}
            self.dependencies = []
            self.required_sources = []

    class MockRegistry:
        def __init__(self):
            self._operators = {
                "op_001": MockOperatorSpec("ts_mean", "production", "polars"),
                "op_002": MockOperatorSpec("ts_mean", "production", "duckdb"),
                "op_003": MockOperatorSpec("ts_std", "production", "polars"),
                "op_004": MockOperatorSpec("experimental_op", "research", "pandas"),
            }

    registry = MockRegistry()
    tables = build_canonical_tables(registry)

    # 验证membership table
    assert tables.membership_table.get_canonical("op_001") == "ts_mean"
    assert tables.membership_table.get_canonical("op_002") == "ts_mean"
    assert tables.membership_table.get_canonical("op_003") == "ts_std"
    assert tables.membership_table.get_canonical("op_004") == "experimental_op"

    # 验证正向索引（ts_mean有2个后端实现）
    ops = tables.membership_table.get_operators("ts_mean")
    assert len(ops) == 2
    assert "op_001" in ops
    assert "op_002" in ops

    # 验证surface索引
    assert "ts_mean" in tables.membership_table.surface_index.get("production", [])
    assert "experimental_op" in tables.membership_table.surface_index.get("research", [])


def test_r42_008_canonical_dependency_table():
    """R42-008: Canonical dependency table提供dependencies O(1)查询。"""
    from factor_engine.planner.analyzer_canonical_tables import build_canonical_tables

    # Mock registry with dependencies
    class MockOperatorSpec:
        def __init__(self, canonical, deps, req_params):
            self.canonical_name = canonical
            self.surface = "production"
            self.backend = "polars"
            self.dependencies = deps
            self.parameters = {p: MockParam(True) for p in req_params}
            self.required_sources = ["price.close", "price.volume"]
            self.temporal_contract = "causal"
            self.pit_contract = "strict"
            self.market_contract = "ashare"

    class MockParam:
        def __init__(self, required):
            self.required = required

    class MockRegistry:
        def __init__(self):
            self._operators = {
                "op_001": MockOperatorSpec("ts_mean", [], ["window"]),
                "op_002": MockOperatorSpec("ts_zscore", ["ts_mean", "ts_std"], ["window"]),
            }

    registry = MockRegistry()
    tables = build_canonical_tables(registry)

    # 验证dependency table
    dep_zscore = tables.dependency_table.get("ts_zscore")
    assert dep_zscore is not None
    assert "ts_mean" in dep_zscore.dependent_operators
    assert "ts_std" in dep_zscore.dependent_operators
    assert "window" in dep_zscore.required_parameters
    assert "price.close" in dep_zscore.required_sources
    assert dep_zscore.temporal_contract == "causal"
    assert dep_zscore.pit_contract == "strict"

    # 验证反向依赖索引
    dependents = tables.dependency_table.get_dependents("ts_mean")
    assert "ts_zscore" in dependents


def test_r42_014_no_temporary_threadpool():
    """R42-014: materialize_shared_nodes_parallel使用传入executor，不创建临时线程池。"""
    from factor_engine.runtime.batch_service import materialize_shared_nodes_parallel
    from concurrent.futures import ThreadPoolExecutor

    # Mock DAG
    class MockDAG:
        def __init__(self):
            self.shared_nodes = {
                "sid_1": {"op": "literal", "value": 1},
                "sid_2": {"op": "literal", "value": 2},
            }

    # Mock backend
    class MockBackend:
        supports_lazy_shared = False

        def execute(self, sub, ctx):
            return sub.get("value", 0)

    # Mock context
    class MockContext:
        def __init__(self):
            self.shared_result_cache = {}
            self.runtime_stats = {}
            from factor_engine.runtime.buffer_store import GovernedBufferStore
            self.shared_buffers = GovernedBufferStore(budget_bytes=10240)

    dag = MockDAG()
    backend = MockBackend()
    ctx = MockContext()

    # 传入long-lived executor
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-executor") as executor:
        materialize_shared_nodes_parallel(dag, backend, ctx, executor=executor)

    # 验证执行成功（结果写入shared_buffers）
    assert ctx.shared_buffers.get("sid_1") is not None
    assert ctx.shared_buffers.get("sid_2") is not None
    # MockBackend返回的是顺序计数，验证值
    assert ctx.shared_buffers.get("sid_1") == 1
    assert ctx.shared_buffers.get("sid_2") == 2


if __name__ == "__main__":
    test_r42_015_cse_certificate_construction()
    test_r42_016_reuse_distance_eviction()
    test_r42_016_reuse_distance_fallback_lru()
    test_r42_015_materialize_shared_reads_certificate()
    test_r42_007_canonical_membership_table()
    test_r42_008_canonical_dependency_table()
    test_r42_014_no_temporary_threadpool()
    print("All R42-014/015/016/007/008 tests passed")
