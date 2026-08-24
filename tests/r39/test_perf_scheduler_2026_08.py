# -*- coding: utf-8 -*-
"""R39-PERF-016/018/019：scheduler 性能整改 —— PlanExecutionCertificate +
MicroBatchTask + event-driven wait。

- PERF-016：``plan()`` 阶段构建 :class:`PlanExecutionCertificate`，
  ``task_execution_certificate`` O(1) 读取；``_priority_score`` 从证书读 cost，
  不再每 ready task 重走整棵 DAG。
- PERF-018：:class:`MicroBatchTask` —— 16–128 个低成本 root 合并成一个 Future
  串行执行（per-root 错误隔离）；指标 future_count / factor_count /
  micro_batch_task_count / micro_batch_root_count。
- PERF-019：``scheduler_wait_polling_count == 0``（happy path 不依赖周期 polling，
  ``wait(FIRST_COMPLETED)`` 事件驱动）。

使用轻量合成任务；不跑完整 engine。小型 RAM —— 本文件只做 focused 测试。
"""
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# 环境兼容：storage 包正被并发 cluster 临时编辑（materializer.py 语法错误）。
# 本文件只依赖 runtime/planner 轻量路径；若 real storage 不可导入则预置一个
# 最小 fake ``storage.time_window``，使 ``runtime/__init__.py`` 的 source-window
# contract 安装不被 broken storage 阻断。storage 健康时完全走 real 路径。
# ---------------------------------------------------------------------------


def _ensure_runtime_importable() -> None:
    if "storage" in sys.modules:
        return
    try:
        import factor_engine.storage  # noqa: F401
        import factor_engine.storage.time_window  # noqa: F401

        return  # real storage works; no seeding needed
    except (SyntaxError, ModuleNotFoundError, ImportError):
        pass
    _tw = types.ModuleType("factor_engine.storage.time_window")
    _tw.narrow_data_source_for_window = lambda **kw: None
    _sp = types.ModuleType("storage")
    _sp.time_window = _tw
    sys.modules["storage"] = _sp
    sys.modules["factor_engine.storage.time_window"] = _tw


_ensure_runtime_importable()

import pytest  # noqa: E402

from factor_engine.planner.physical_factor_dag import (  # noqa: E402
    TASK_ROOT,
    PhysicalFactorDAG,
    PhysicalFactorTask,
)
from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler  # noqa: E402
from factor_engine.runtime.resource_broker import ResourceBroker  # noqa: E402
from factor_engine.runtime.task_resource_contract import TaskResourceContract  # noqa: E402

_SOURCE_SCOPE = "dataset:mock::snapshot:s1::market:us"
_EXEC_SCOPE = '{"frequency":"1d","universe_id":"ALL","market":"us"}'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_scheduler(*, cpu_slots: int = 32) -> AdaptiveBatchScheduler:
    return AdaptiveBatchScheduler(
        broker=ResourceBroker(
            hard_memory_limit=8 * 1024**3,
            cpu_slots=cpu_slots,
            min_host_reserve_gb=1,
            min_host_reserve_fraction=0.05,
        )
    )


def _cheap_root(i: int, *, factor_name: str | None = None) -> PhysicalFactorTask:
    name = factor_name if factor_name is not None else f"F{i}"
    return PhysicalFactorTask(
        task_id=f"root:{name}",
        op="add",
        task_type=TASK_ROOT,
        inputs=(),
        consumers=(),
        source_scope=_SOURCE_SCOPE,
        execution_scope=_EXEC_SCOPE,
        preferred_backend="pandas_numpy",
        backend_candidates=("pandas_numpy",),
        resource_contract=TaskResourceContract(
            predicted_elapsed_ms=1.0,
            cpu_tokens=1,
            io_tokens=0,
            peak_memory_bytes=1024 * 1024,
            output_bytes=1024 * 1024,
            backend_threads=1,
        ),
        factor_name=name,
        executable=True,
    )


def _bare_dag(n: int) -> PhysicalFactorDAG:
    dag = PhysicalFactorDAG()
    for i in range(n):
        dag.add_task(_cheap_root(i))
    dag.roots = tuple(f"root:F{i}" for i in range(n))
    return dag


class _Ctx:
    run_mode = "research"
    runtime_stats = {}
    shared_result_cache = {}


# ---------------------------------------------------------------------------
# PERF-016：PlanExecutionCertificate
# ---------------------------------------------------------------------------


def test_certificate_data_layer_fields(tmp_path):
    """证书数据层：build_plan_execution_certificate 各字段来自 task+plan_cost。"""
    from factor_engine.runtime.plan_execution_certificate import (
        PlanExecutionCertificate,
        build_plan_execution_certificate,
    )

    node = types.SimpleNamespace(
        root=types.SimpleNamespace(
            op="add",
            inputs=(types.SimpleNamespace(op="column", inputs=()),),
            shape=(10, 2),
        )
    )
    task = PhysicalFactorTask(
        task_id="root:F0",
        op="add",
        task_type=TASK_ROOT,
        node_ref=node,
        backend_candidates=("pandas_numpy", "polars"),
        time_range=("2024-01-01", "2024-12-31"),
        preferred_backend="pandas_numpy",
    )
    plan_cost = {
        "total_work": 42.0,
        "peak_live_memory_bytes": 8000,
        "node_count": 3,
        "expensive_ops": ["add"],
    }
    cert = build_plan_execution_certificate(task, plan_cost)
    assert isinstance(cert, PlanExecutionCertificate)
    assert cert.cost == plan_cost
    assert cert.total_work() == 42.0
    assert cert.peak_memory_bytes() == 8000
    assert cert.estimated_memory == 8000
    assert cert.backend_eligibility == ("pandas_numpy", "polars")
    assert cert.history == ("2024-01-01", "2024-12-31")
    assert cert.output_shape == (10, 2)
    assert "add" in cert.bound_ops and "column" in cert.bound_ops


def test_plan_builds_execution_certificate_o1(monkeypatch):
    """plan() 阶段为每个 task 构建证书，task_execution_certificate O(1) 读取，
    且证书 cost 与 plan.cost_by_task 完全一致。

    用 monkeypatch 替换 ``lower_batch_dag`` 返回合成 physical DAG，避免依赖
    storage/backend 全链路（storage 正被并发 cluster 临时编辑）。
    """
    import factor_engine.planner.physical_lowerer as pl
    from factor_engine.planner.dag import DAGPlan, FactorPlan
    from factor_engine.planner.logical_plan import PlanNode

    def fake_lower(
        dag, analyses=None, ctx=None, rows=None, instruments=0, scan_cost_map=None
    ):
        phys = PhysicalFactorDAG()
        for i, fp in enumerate(dag.roots):
            node = getattr(fp, "root", None)
            rid = f"root:{fp.factor_name}"
            phys.add_task(PhysicalFactorTask(
                task_id=rid,
                op=str(getattr(node, "op", "add")),
                task_type=TASK_ROOT,
                inputs=(),
                consumers=(),
                source_scope=_SOURCE_SCOPE,
                execution_scope="exec",
                backend_candidates=("pandas_numpy",),
                preferred_backend="pandas_numpy",
                estimated_cost={
                    "total_work": 10.0 + i,
                    "peak_live_memory_bytes": 2000 + i,
                    "node_count": 3,
                },
                factor_name=fp.factor_name,
                node_ref=node,
                executable=True,
            ))
        phys.roots = tuple(f"root:{fp.factor_name}" for fp in dag.roots)
        return phys

    monkeypatch.setattr(pl, "lower_batch_dag", fake_lower)

    roots = [
        FactorPlan(
            factor_name=f"F{i}",
            root=PlanNode(
                op="add",
                inputs=(PlanNode(op="column", attrs={"name": "close"}),),
            ),
        )
        for i in range(3)
    ]
    dagplan = DAGPlan(roots=roots)
    sched = _make_scheduler()
    plan = sched.plan(dagplan, ctx=None, enable_cse=False)

    assert len(sched._certificates) == len(plan.physical_dag.tasks)
    for tid in plan.physical_dag.tasks:
        cert = sched.task_execution_certificate(tid)
        assert cert is not None, tid
        # PERF-016：证书 cost 就是 plan 阶段算好的 cost（不重新遍历 DAG）。
        assert cert.cost == plan.cost_by_task[tid], tid
        assert cert.backend_eligibility == ("pandas_numpy",)
        assert cert.bound_ops, tid
    # O(1) 读取：返回同一个缓存对象。
    tid0 = list(plan.physical_dag.tasks)[0]
    assert sched.task_execution_certificate(tid0) is sched._certificates[tid0]


def test_priority_score_consumes_certificate_cost(monkeypatch):
    """_priority_score 优先读证书 cost（不重走 DAG），并回退 contract。"""
    dag = PhysicalFactorDAG()
    dag.add_task(_cheap_root(0))
    dag.add_task(_cheap_root(1))
    dag.roots = ("root:F0", "root:F1")
    sched = _make_scheduler()
    # 预置证书：cost 与 task 的 contract 不同 → 应读证书。
    from factor_engine.runtime.plan_execution_certificate import build_plan_execution_certificate

    task = dag.tasks["root:F0"]
    sched._certificates["root:F0"] = build_plan_execution_certificate(
        task, {"total_work": 999.0, "peak_live_memory_bytes": 123456}
    )
    score = sched._priority_score(dag, task)
    # memory term 来自证书 peak_memory_bytes = 123456 bytes → 约 0.0001 GiB 罚分。
    # 证书 total_work=999 进入 critical path cost map。
    assert sched._cost_ms_cache is not None
    assert sched._cost_ms_cache["root:F0"] == 999.0
    assert score > 0  # critical/reuse/fanout 仍主导


def test_priority_scores_share_one_critical_path_pass(monkeypatch):
    """A wide ready queue computes critical paths once, not once per root."""
    dag = _bare_dag(64)
    sched = _make_scheduler()
    calls = 0

    def fail_if_recomputed(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("per-task critical-path traversal regressed")

    monkeypatch.setattr(dag, "critical_path_remaining_ms", fail_if_recomputed)
    for task in dag.tasks.values():
        sched._priority_score(dag, task)
    assert calls == 0
    assert sched._critical_path_cache is not None




def test_micro_batch_coalesces_cheap_roots(monkeypatch):
    """20 个低成本 root → micro_batch_task_count < 20，且 20 个结果全部正确。"""
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    sched = _make_scheduler()
    dag = _bare_dag(20)

    def _execute_root(task):
        return f"v:{task.factor_name}"

    out = sched.run(dag, backend=None, ctx=_Ctx(), execute_root=_execute_root)
    stats = out["scheduler_stats"]
    assert stats["micro_batch_task_count"] < 20
    assert stats["micro_batch_task_count"] >= 1
    assert stats["micro_batch_root_count"] == 20
    assert stats["future_count"] < 20
    assert len(out["results"]) == 20
    for i in range(20):
        assert out["results"][f"F{i}"] == f"v:F{i}"


def test_dispatch_micro_batch_per_root_error_isolation():
    """dispatch_micro_batch 单元：一个 root 失败，其余 19 个仍正常返回。"""
    from factor_engine.runtime.micro_batch_task import dispatch_micro_batch

    task_by_id = {f"root:F{i}": _cheap_root(i) for i in range(20)}

    def _execute_root(task):
        if task.factor_name == "F3":
            raise RuntimeError("custom_factor_failure")
        return f"v:{task.factor_name}"

    roots = tuple(f"root:F{i}" for i in range(20))
    results, failures = dispatch_micro_batch(roots, task_by_id, _execute_root)
    assert set(failures) == {"root:F3"}
    assert len(results) == 19
    for i in range(20):
        if i == 3:
            continue
        assert results[f"root:F{i}"] == f"v:F{i}"


def test_micro_batch_error_isolation_via_scheduler(monkeypatch):
    """调度器级错误隔离：F3 失败走 retry→FAILED_FATAL 抛出；其余 19 个 root 在
    抛错前已提交且结果正确（失败不阻断同批其它 root 执行）。"""
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    sched = _make_scheduler()
    captured: dict[str, str] = {}

    def _execute_root(task):
        if task.factor_name == "F3":
            raise RuntimeError("custom_factor_failure")
        return f"v:{task.factor_name}"

    with pytest.raises(RuntimeError, match="custom_factor_failure"):
        sched.run(
            _bare_dag(20),
            backend=None,
            ctx=_Ctx(),
            execute_root=_execute_root,
            result_handler=lambda n, r: captured.__setitem__(n, r),
        )
    # 其余 19 个 root 执行并返回正确结果（F3 失败不吞掉同批其它 root）。
    assert len(captured) == 19
    for i in range(20):
        if i == 3:
            continue
        assert captured[f"F{i}"] == f"v:F{i}"


def test_future_per_factor_ratio_drops(monkeypatch):
    """开启 micro-batch 后 future_count/factor_count 显著下降 vs 不开启。"""
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    dag = _bare_dag(20)

    # with micro-batch
    sched = _make_scheduler()
    out_on = sched.run(
        dag, backend=None, ctx=_Ctx(), execute_root=lambda t: f"v:{t.factor_name}"
    )
    on = out_on["scheduler_stats"]

    # without micro-batch（把 MIN_ROOTS 顶到无穷 → 不形成 micro-batch）
    import factor_engine.runtime.adaptive_batch_scheduler as abs_mod

    monkeypatch.setattr(abs_mod, "_MICRO_BATCH_MIN_ROOTS", 10**9)
    sched_off = _make_scheduler()
    out_off = sched_off.run(
        dag, backend=None, ctx=_Ctx(), execute_root=lambda t: f"v:{t.factor_name}"
    )
    off = out_off["scheduler_stats"]

    assert on["micro_batch_task_count"] >= 1
    assert off["micro_batch_task_count"] == 0
    assert on["future_count"] < off["future_count"]
    assert on["future_per_factor"] < off["future_per_factor"]
    assert off["future_count"] >= 20  # 逐 root future


def test_micro_batch_membership_threshold():
    """低于 MICRO_BATCH_MIN_ROOTS 的 cheap root 不形成 micro-batch。"""
    sched = _make_scheduler()
    dag = _bare_dag(3)
    ready = [(sched._priority_score(dag, t), tid) for tid, t in dag.tasks.items()]
    batches = sched._plan_micro_batches(ready, dag, {}, set())
    assert batches == []


# ---------------------------------------------------------------------------
# PERF-019：event-driven wait（无周期 polling）
# ---------------------------------------------------------------------------


def test_scheduler_wait_polling_count_zero_happy_path(monkeypatch):
    """happy path：scheduler_wait_polling_count == 0（wait 事件驱动，无周期轮询）。"""
    monkeypatch.setenv("FACTOR_ENGINE_HYBRID_FORCE", "thread")
    sched = _make_scheduler()
    out = sched.run(
        _bare_dag(20), backend=None, ctx=_Ctx(), execute_root=lambda t: f"v:{t.factor_name}"
    )
    stats = out["scheduler_stats"]
    assert stats["scheduler_wait_polling_count"] == 0
    assert stats["factor_count"] == 20
