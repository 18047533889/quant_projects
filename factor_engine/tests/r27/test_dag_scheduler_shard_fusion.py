# -*- coding: utf-8 -*-
"""R27-230..233/245/246/249: PhysicalFactorDAG + AdaptiveBatchScheduler +
shard legality + native fusion + as_completed。
"""
from __future__ import annotations

from planner.physical_factor_dag import (
    TASK_CSE_SHARED,
    TASK_ROOT,
    PhysicalFactorDAG,
    PhysicalFactorTask,
)
from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
from runtime.adaptive_sharding import adaptive_shard_size, classify_shard_legality
from runtime.resource_broker import ResourceBroker
from runtime.task_resource_contract import TaskResourceContract

from planner.native_fusion import adaptive_fusion_block_size, can_fuse_roots


def _dag_with_shared() -> PhysicalFactorDAG:
    dag = PhysicalFactorDAG()
    dag.add_task(PhysicalFactorTask(
        task_id="cse:ts_mean_close",
        op="ts_mean",
        task_type=TASK_CSE_SHARED,
        consumers=("root:A", "root:B"),
        resource_contract=TaskResourceContract(
            peak_memory_bytes=64 * 1024**2, cpu_tokens=1,
        ),
    ))
    dag.add_task(PhysicalFactorTask(
        task_id="root:A",
        op="add",
        task_type=TASK_ROOT,
        inputs=("cse:ts_mean_close",),
        resource_contract=TaskResourceContract(peak_memory_bytes=32 * 1024**2, cpu_tokens=1),
        factor_name="A",
    ))
    dag.add_task(PhysicalFactorTask(
        task_id="root:B",
        op="neg",
        task_type=TASK_ROOT,
        inputs=("cse:ts_mean_close",),
        resource_contract=TaskResourceContract(peak_memory_bytes=32 * 1024**2, cpu_tokens=1),
        factor_name="B",
    ))
    dag.roots = ("root:A", "root:B")
    return dag


def test_dag_topological_order():
    dag = _dag_with_shared()
    order = dag.topological_order()
    # shared 先于 roots；roots 可并行。
    assert order.index("cse:ts_mean_close") < order.index("root:A")
    assert order.index("cse:ts_mean_close") < order.index("root:B")


def test_scheduler_shared_parallel_and_as_completed(monkeypatch):
    # R27-002/006/231/249：shared node 进入 ready queue；root 完成即交结果。
    # 测试传闭包不可 pickle → 强制 thread executor（生产路径默认传模块级函数，
    # 可 pickle，进程池可用）。
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    dag = _dag_with_shared()
    scheduler = AdaptiveBatchScheduler(
        broker=ResourceBroker(hard_memory_limit=8 * 1024**3, cpu_slots=4,
                              min_host_reserve_gb=1, min_host_reserve_fraction=0.05)
    )
    executed: dict[str, str] = {}

    class _Ctx:
        run_mode = "research"
        runtime_stats = {}
        shared_result_cache = {"cse:ts_mean_close": None}

    def _execute_root(task):
        executed[task.task_id] = "root"
        return f"value:{task.factor_name}"

    def _mat_shared(sid, node):
        executed[f"cse:{sid}"] = "shared"

    out = scheduler.run(
        dag,
        backend=None,
        ctx=_Ctx(),
        execute_root=_execute_root,
        materialize_shared=_mat_shared,
    )
    assert "cse:ts_mean_close" in executed
    assert "root:A" in executed and "root:B" in executed
    assert out["results"] == {"A": "value:A", "B": "value:B"}
    assert out["done"] == 3
    assert out["explanations"], "R27-143: scheduler must be explainable"


def test_shard_legality_cross_section_blocks_asset_shard():
    # R27-084/085/245：rank/zscore/neutralize 不能按 asset shard（结果会变）。
    class RankNode:
        op = "rank"
        inputs = ()

    class PureNode:
        op = "ts_mean"
        inputs = ()

    rank_legality = classify_shard_legality(RankNode())
    pure_legality = classify_shard_legality(PureNode())
    assert rank_legality.asset_shard_safe is False
    assert rank_legality.time_shard_safe is True
    assert pure_legality.asset_shard_safe is True


def test_adaptive_shard_size():
    # R27-092/093/246：20GB / target 8GB → split 3。
    assert adaptive_shard_size(
        predicted_peak_bytes=20 * 1024**3, shard_memory_target=8 * 1024**3
    ) == 3
    assert adaptive_shard_size(
        predicted_peak_bytes=2 * 1024**3, shard_memory_target=8 * 1024**3
    ) == 1


def test_native_fusion_grouping_and_block():
    # R27-220..222/233：fusion group 约束 + 自适应 block。
    assert adaptive_fusion_block_size(root_count=100, expression_complexity=1.0,
                                      estimated_output_bytes=1024**3) <= 256
    assert 32 <= adaptive_fusion_block_size(root_count=10, expression_complexity=1.0,
                                            estimated_output_bytes=64 * 1024**2) <= 256


def test_can_fuse_requires_same_source_scope():
    class T:
        def __init__(self, backend, scope, snap):
            self.preferred_backend = backend
            self.source_scope = scope
            self.source_snapshot_id = snap
            self.resource_contract = None

    same = [T("pandas_numpy", "a", "s1"), T("pandas_numpy", "a", "s1")]
    diff_scope = [T("pandas_numpy", "a", "s1"), T("pandas_numpy", "b", "s1")]
    assert can_fuse_roots(same) is True
    assert can_fuse_roots(diff_scope) is False
