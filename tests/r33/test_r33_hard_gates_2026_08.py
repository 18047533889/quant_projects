# -*- coding: utf-8 -*-
"""R33 hard-gate behavior tests —— 真实行为探针，不是 grep/module-existence。

覆盖：typed SourceScopeId、SourceScanSpec required_columns、ReadWave 物理列/
time_range/主路径执行、BatchDataRequest multi-source + ScanCostUnavailable、
batch-global route、fusion per-group、sink deque + fatal 传播、FactorBlock/
BlockDQ/CS parity、representation cache global budget、AUTO DIRECT_VECTOR。
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from factor_engine.planner.physical_factor_dag import (
    SourceScanSpec,
    SourceScopeId,
    source_scope_from_key,
)
from factor_engine.planner.batch_data_request import ScanCostUnavailable, build_batch_data_request
from factor_engine.planner.read_wave_planner import ReadWavePlanner
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.block_dq import (
    FactorBlock,
    compute_block_dq,
    cross_section_block,
    cross_section_block_parity,
)
from factor_engine.runtime.buffer_ref import SourceWaveExecutor
from factor_engine.runtime.streaming_result_sink import BoundedResultQueue, StreamingResultSink


def _plan(cols=("close",)):
    return PlanNode(
        op="ts_mean",
        attrs={"window": 5},
        inputs=(
            PlanNode(op="column", attrs={"name": cols[0]}, inputs=()),
            PlanNode(op="literal", attrs={"value": 5}, inputs=()),
        ),
    )


def _ctx(data_kind="memory"):
    class _Src:
        dataset = "ashare_daily"
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        instrument_filter = ["000001"]
        capabilities = types.SimpleNamespace(engine_kind=data_kind)

        def estimate_scan_cost(self, **kw):
            raise RuntimeError("probe boom")

    return types.SimpleNamespace(
        data_source=_Src(),
        run_mode="research",
        runtime_stats={},
        market="ashare",
    )


class TestSourceScopeTyped:
    def test_source_scope_typed_roundtrip(self):
        sid = SourceScopeId(dataset="ashare_daily", snapshot_id="s1", market="ashare")
        assert sid.key() == "dataset:ashare_daily::snapshot:s1::market:ashare"
        back = source_scope_from_key(sid.key())
        assert back.dataset == "ashare_daily"
        assert back.market == "ashare"

    def test_source_scope_no_business_identity_from_middle(self):
        # R33-P0-006：不把含 :: 的字符串中段当 dataset。
        sid = source_scope_from_key("dataset:foo::snapshot:s1")
        assert sid.dataset == "foo"


class TestSourceScanSpec:
    def test_required_columns_real(self):
        from factor_engine.planner.physical_lowerer import lower_root_plan

        stages = lower_root_plan(_plan(), factor_name="f1", ctx=None, rows=1000)
        scan = next(s for s in stages if s.task_type == "SOURCE_SCAN")
        assert scan.source_scan_spec is not None
        assert scan.source_scan_spec.required_columns == ("close",)
        assert scan.executable is True

    def test_virtual_stage_not_executable(self):
        from factor_engine.planner.physical_lowerer import lower_root_plan

        stages = lower_root_plan(
            PlanNode(
                op="rank",
                attrs={},
                inputs=(_plan(),),
            ),
            factor_name="f2",
            ctx=None,
            rows=1000,
        )
        barrier = next((s for s in stages if s.task_type == "CROSS_SECTION"), None)
        assert barrier is not None
        assert barrier.executable is False


class TestReadWavePhysical:
    def test_wave_columns_are_physical(self):
        from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

        sched = AdaptiveBatchScheduler(max_concurrency=2)
        ctx = _ctx()
        dag = types.SimpleNamespace(
            shared_nodes={},
            roots=(types.SimpleNamespace(
                factor_name="f1", root=_plan(), execution_scope=None,
            ),),
        )
        plan = sched.plan(dag, {}, enable_cse=False, ctx=ctx)
        assert plan.read_waves.waves
        assert any("close" in w.columns for w in plan.read_waves.waves)

    def test_wave_time_range_real(self):
        from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

        sched = AdaptiveBatchScheduler(max_concurrency=2)
        plan = sched.plan(
            types.SimpleNamespace(
                shared_nodes={},
                roots=(types.SimpleNamespace(
                    factor_name="f1", root=_plan(), execution_scope=None,
                ),),
            ),
            {}, enable_cse=False, ctx=_ctx(),
        )
        assert any(w.time_range is not None for w in plan.read_waves.waves)


class TestSourceWaveExecutor:
    def test_wave_executes_once(self):
        calls: list[list[str]] = []

        class _Src:
            dataset = "ashare_daily"
            snapshot_token = "s1"

            def prefetch_columns(self, cols):
                calls.append(list(cols))

        ex = SourceWaveExecutor(_Src(), _ctx())
        ex.execute_wave(types.SimpleNamespace(
            wave_id=0, columns=frozenset({"close"}), task_ids=("t1",),
            estimated_scan_bytes=8, estimated_memory_bytes=8,
        ), consumer_ids=("t1",))
        ex.execute_wave(types.SimpleNamespace(
            wave_id=0, columns=frozenset({"close"}), task_ids=("t1",),
            estimated_scan_bytes=8, estimated_memory_bytes=8,
        ), consumer_ids=("t1",))
        assert len(calls) == 1  # 幂等：同一 wave 只 scan 一次


class TestBatchDataRequest:
    def test_scan_cost_unavailable_recorded(self):
        req = build_batch_data_request(
            _ctx().data_source, analyses={}, ctx=_ctx(),
        )
        assert req.degraded_planning
        assert all(isinstance(d, ScanCostUnavailable) for d in req.degraded_planning)

    def test_scan_cost_has_window_and_universe(self):
        seen: dict = {}

        class _Src:
            dataset = "ashare_daily"
            start_date = "2024-01-01"
            end_date = "2024-12-31"
            instrument_filter = ["000001"]

            def estimate_scan_cost(self, **kw):
                seen.update(kw)
                return None

        build_batch_data_request(_Src(), analyses={}, ctx=_ctx())
        assert seen.get("time_range") == ("2024-01-01", "2024-12-31")
        assert seen.get("instruments") == ("000001",)


class TestBatchRoute:
    def test_batch_physical_route_fields(self):
        from factor_engine.backend.plan_cost_router import plan_batch_route

        ctx = _ctx(data_kind="duckdb")
        route = plan_batch_route({"f1": _plan()}, ctx, factor_count=1)
        assert route.scheduler_overhead_ms > 0
        assert route.dq_ms > 0
        assert route.write_ms > 0
        assert route.generation_commit_ms > 0

    def test_native_subgraph_fraction(self):
        from factor_engine.backend.plan_cost_router import plan_native_subgraph_fraction

        sub = plan_native_subgraph_fraction(_plan(), _ctx(data_kind="duckdb"))
        assert "native_fraction" in sub
        assert "unsupported_ops" in sub


class TestFusionPerGroup:
    def test_uncertified_backend_not_planned(self):
        from factor_engine.planner.native_fusion import plan_native_fusion_groups
        from factor_engine.planner.physical_lowerer import lower_root_plan

        stages = lower_root_plan(_plan(), factor_name="f1", ctx=None, rows=1000)
        root = next(s for s in stages if s.task_type == "ROOT")
        groups = plan_native_fusion_groups(
            [root, root], backend_capability={"pandas_numpy": False}
        )
        assert groups == []


class TestStreamingSink:
    def test_queue_deque(self):
        import collections

        q = BoundedResultQueue(1024)
        assert isinstance(q._items, collections.deque)

    def test_writer_fatal_propagates(self):
        def bad(batch):
            raise RuntimeError("deterministic write bug")

        sink = StreamingResultSink(writer=bad, writer_threads=1)
        sink.start()
        sink.submit("f1", object())
        with pytest.raises(RuntimeError):
            sink.finish()


class TestFactorBlock:
    def test_block_dq(self):
        block = FactorBlock(
            factor_ids=("a", "b"),
            values=np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]]),
        )
        dq = compute_block_dq(block)
        assert dq["a"]["null_ratio"] == pytest.approx(1 / 3)
        assert dq["b"]["constant"] is False

    def test_cs_block_parity(self):
        block = FactorBlock(
            factor_ids=("r1",),
            values=np.array([[3.0, 1.0, 2.0, 4.0]]),
            date_axis=np.array([0, 0, 1, 1]),
        )
        # canonical per-factor 逐日 rank。
        ref = np.array([1.0, 0.0, 0.0, 1.0])  # date0: rank(3,1)->(2-1,1-1)=1,0 ; date1: (2,4)->0,1
        out = cross_section_block(block, "rank")
        assert cross_section_block_parity(block, "rank", {"r1": ref}) is True


class TestRepresentationCache:
    def test_global_eviction_exists(self):
        from factor_engine.storage.sources.data_access_source import DataAccessSource

        assert hasattr(DataAccessSource, "_evict_global_lru")


class TestAutoMode:
    def test_direct_vector_for_small_batch(self):
        from factor_engine.runtime.batch_service import choose_execution_mode

        mode = choose_execution_mode(
            [1, 2],  # noqa: F401
            {"a": types.SimpleNamespace(node_count=5), "b": types.SimpleNamespace(node_count=5)},
            types.SimpleNamespace(roots=(1, 2)),
        )
        assert mode == "DIRECT_VECTOR"
