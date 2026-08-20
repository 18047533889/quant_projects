# -*- coding: utf-8 -*-
"""R39 §4 ReadWave 物理 footprint optimizer —— PERF-005..012 行为探针。

覆盖：
    (a) PERF-005 source/consumer 分离 + 无零列 request；
    (b) PERF-006 footprint 覆盖固定 fallback；
    (c) PERF-007 marginal 评分倾向共享列候选 + bounded local search；
    (d) PERF-008 cost-based superset coalescing（20d/60d/120d 仅成本证明有收益才合并）；
    (e) PERF-009 ScopeCompatibility + instrument/universe 指标；
    (f) PERF-010 baseline/physical_union/saved 字节记账；
    (g) PERF-011 SourceRepresentation + preferred_representation 默认；
    (h) PERF-012 typed recovery 决策表 + FAILED_WAVE_TO_N_ROOT_SCANS hard gate。
"""
from __future__ import annotations

import types

import pytest

from planner.projected_column_footprint import ProjectedColumnFootprint
from planner.read_wave_planner import (
    ReadWave,
    ReadWavePlan,
    ReadWavePlanner,
    ScopeCompatibility,
    WaveCostModel,
    build_waves_from_dag,
    count_zero_column_waves,
    non_source_tasks_in_wave_source_tasks,
    scope_compatibility,
)
from planner.source_representation import (
    SourceRepresentation,
    consumer_backend_mask,
    preferred_representation_for_mask,
    representation_for_backend,
)
from planner.wave_recovery import (
    WaveRecoveryCategory,
    WaveRecoveryPlan,
    classify_wave_failure,
    hard_gate_failed_wave_to_n_root_scans,
    recover_wave_failure,
    would_fallback_to_n_root_scans,
)
from planner.physical_factor_dag import SourceScanSpec


def _task(
    tid,
    task_type,
    *,
    inputs=(),
    source_scope="dataset:d::snapshot:snap",
    backend="pandas_numpy",
    cols=(),
    time_range=("2024-01-01", "2024-12-31"),
    instrument_scope=None,
    universe_id=None,
):
    spec = None
    if task_type == "SOURCE_SCAN":
        spec = SourceScanSpec(
            dataset="d",
            required_columns=tuple(cols),
            time_range=time_range,
            instrument_scope=instrument_scope,
            universe_id=universe_id,
        )
    return types.SimpleNamespace(
        task_id=tid,
        task_type=task_type,
        inputs=tuple(inputs),
        consumers=(),
        source_scope=source_scope,
        source_snapshot_id="snap",
        source_scan_spec=spec,
        required_columns=tuple(cols),
        time_range=time_range,
        preferred_backend=backend,
        instrument_scope=instrument_scope,
        universe_id=universe_id,
    )


def _dag_with_consumers():
    return types.SimpleNamespace(
        tasks={
            "scan1": _task("scan1", "SOURCE_SCAN", cols=["close", "open"]),
            "scan2": _task("scan2", "SOURCE_SCAN", cols=["close", "volume"]),
            "cse1": _task("cse1", "CSE_SHARED", inputs=("scan1",)),
            "root1": _task(
                "root1", "ROOT", inputs=("cse1",), backend="duckdb_engine"
            ),
            "root2": _task("root2", "ROOT", inputs=("scan2",)),
        }
    )


# ---------------------------------------------------------------------------
# (a) PERF-005
# ---------------------------------------------------------------------------


class TestPerf005SourceConsumerSeparation:
    def test_consumers_are_not_scan_requests(self):
        dag = _dag_with_consumers()
        plan = build_waves_from_dag(dag, rows_estimate=1000)
        assert plan.waves
        for w in plan.waves:
            assert w.columns, "zero-column read wave must not exist"
            assert w.source_tasks, "wave must have at least one source task"
            # source_tasks 只含 SOURCE_SCAN。
            for sid in w.source_tasks:
                assert dag.tasks[sid].task_type == "SOURCE_SCAN"
            # task_ids = source + consumer 并集。
            assert set(w.task_ids) == set(w.source_tasks) | set(w.consumer_tasks)

        sources = set().union(*(set(w.source_tasks) for w in plan.waves))
        consumers = set().union(*(set(w.consumer_tasks) for w in plan.waves))
        assert {"scan1", "scan2"} <= sources
        assert {"cse1", "root1", "root2"} <= consumers

    def test_hard_gates_zero(self):
        dag = _dag_with_consumers()
        plan = build_waves_from_dag(dag, rows_estimate=1000)
        assert count_zero_column_waves(plan) == 0  # ZERO_COLUMN_READ_WAVE
        assert non_source_tasks_in_wave_source_tasks(dag, plan) == 0

    def test_read_wave_legacy_task_ids_backcompat(self):
        # 外部只给 task_ids → 视为 source_tasks。
        w = ReadWave(
            wave_id=0, source_scope="s", dataset="d", snapshot_id="snap",
            columns=frozenset({"c"}), time_range=None, task_ids=("old",),
        )
        assert w.source_tasks == ("old",)
        assert w.task_ids == ("old",)

    def test_read_wave_new_construction(self):
        w = ReadWave(
            wave_id=0, source_scope="s", dataset="d", snapshot_id="snap",
            columns=frozenset({"c"}), time_range=None,
            source_tasks=("b", "a"), consumer_tasks=("d", "c"),
        )
        assert w.source_tasks == ("a", "b")
        assert w.consumer_tasks == ("c", "d")
        assert set(w.task_ids) == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# (b) PERF-006
# ---------------------------------------------------------------------------


class TestPerf006ProjectedFootprint:
    def test_footprint_overrides_fixed_fallback(self):
        fp = {
            "close": ProjectedColumnFootprint("close", decoded_bytes=10_000_000),
            "open": ProjectedColumnFootprint("open", decoded_bytes=2_000_000),
        }
        p = ReadWavePlanner(rows_estimate=500_000)
        p.register_scan_task(
            "s1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-01-01", "2024-12-31"), columns=["close", "open"],
            column_footprints=fp,
        )
        plan = p.plan()
        assert len(plan.waves) == 1
        w = plan.waves[0]
        # 12M（decoded）≠ 固定 2 列 × 500k × 8B = 8M。
        assert w.estimated_memory_bytes == 12_000_000
        assert w.decoded_resident_bytes == 12_000_000

    def test_footprint_source_in_build_waves_from_dag(self):
        class _Foot:
            def column_footprints(self, dataset, snapshot, columns):
                return {
                    c: ProjectedColumnFootprint(c, decoded_bytes=1_000_000)
                    for c in columns
                }

        dag = types.SimpleNamespace(
            tasks={
                "s1": _task("s1", "SOURCE_SCAN", cols=["close", "open"]),
                "s2": _task("s2", "SOURCE_SCAN", cols=["close"]),
            }
        )
        plan = build_waves_from_dag(
            dag, rows_estimate=500_000, footprint_source=_Foot()
        )
        w = plan.waves[0]
        # union cols {close, open} × 1M decoded each = 2M（不是 2×500k×8=8M）。
        assert w.estimated_memory_bytes == 2_000_000

    def test_fallback_keeps_500k_8b_when_no_footprint(self):
        p = ReadWavePlanner(rows_estimate=500_000)
        p.register_scan_task(
            "s1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-01-01", "2024-12-31"), columns=["close", "open"],
        )
        plan = p.plan()
        assert plan.waves[0].estimated_memory_bytes == 2 * 500_000 * 8


# ---------------------------------------------------------------------------
# (c) PERF-007
# ---------------------------------------------------------------------------


class TestPerf007MarginalScoring:
    def test_prefers_shared_column_candidate(self):
        p = ReadWavePlanner(rows_estimate=1000)
        p.register_scan_task(
            "r1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a", "b"],
        )
        p.register_scan_task(
            "r2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a", "c"],  # 与 r1 共享 a
        )
        p.register_scan_task(
            "r3", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["x", "y"],  # 与 r1 无共享
        )
        r1, r2, r3 = p._requests[0], p._requests[1], p._requests[2]
        idx = p._best_marginal([r2, r3], set(r1.columns), [r1])
        assert idx == 0, f"expected shared-column candidate r2, got idx={idx}"

    def test_marginal_denominator_is_incremental(self):
        # incremental_live_bytes = union - current，不是整个 union。
        p = ReadWavePlanner(rows_estimate=1000)
        p.register_scan_task(
            "r1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a", "b", "c", "d", "e"],
        )
        p.register_scan_task(
            "r2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a"],
        )
        r1, r2 = p._requests[0], p._requests[1]
        current_mem = p._wave_memory_bytes(set(r1.columns), [r1])
        union_mem = p._wave_memory_bytes(set(r1.columns) | set(r2.columns), [r1, r2])
        incremental = max(0, union_mem - current_mem)
        # r2 全在 r1 里 → 边际 0。
        assert incremental == 0

    def test_bounded_local_search_merges_subset_waves(self):
        # 同 scope / 同 time_range：r2 是 r1 列子集 → 贪心可能分波，local search 合并。
        p = ReadWavePlanner(rows_estimate=1000, wave_memory_budget=4000 * 8)
        # 预算只能装 1 列 → 贪心每波 1 个 request；r2 列是 r1 子集。
        p.register_scan_task(
            "r1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a"],
        )
        p.register_scan_task(
            "r2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["a"],
        )
        plan = p.plan()
        # local search 把 r2 并进 r1（子集 → 不增加内存），1 个 wave。
        assert len(plan.waves) == 1
        assert set(plan.waves[0].source_tasks) == {"r1", "r2"}


# ---------------------------------------------------------------------------
# (d) PERF-008
# ---------------------------------------------------------------------------


class TestPerf008SupersetCoalescing:
    def _register_20_60_120(self, p):
        p.register_scan_task(
            "w20", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-12-12", "2024-12-31"), columns=["close"],
        )
        p.register_scan_task(
            "w60", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-11-02", "2024-12-31"), columns=["close"],
        )
        p.register_scan_task(
            "w120", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-09-03", "2024-12-31"), columns=["close"],
        )

    def test_merge_when_cost_justifies(self):
        p = ReadWavePlanner(rows_estimate=500_000)  # 默认 cost model → 合并有收益
        self._register_20_60_120(p)
        plan = p.plan()
        assert len(plan.waves) == 1, f"expected 1 coalesced wave, got {len(plan.waves)}"
        w = plan.waves[0]
        assert w.superset_coalesce is True
        assert w.time_range == ("2024-09-03", "2024-12-31")
        assert set(w.source_tasks) == {"w20", "w60", "w120"}

    def test_no_merge_when_cost_not_justified(self):
        cm = WaveCostModel(min_superset_benefit_ratio=0.001)  # 极难合并
        p = ReadWavePlanner(rows_estimate=500_000, cost_model=cm)
        self._register_20_60_120(p)
        plan = p.plan()
        assert len(plan.waves) == 3, f"expected 3 separate waves, got {len(plan.waves)}"
        assert all(not w.superset_coalesce for w in plan.waves)

    def test_merge_respects_memory_budget(self):
        # 合并后 union 内存超预算 → 不合并。每 request 不同列（union 内存会随
        # 合并增长）：预算 8M = 2 列；合并 3 列超预算 → 停在 2 个 wave。
        p = ReadWavePlanner(rows_estimate=500_000, wave_memory_budget=8_000_000)
        p.register_scan_task(
            "w20", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-12-12", "2024-12-31"), columns=["a"],
        )
        p.register_scan_task(
            "w60", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-11-02", "2024-12-31"), columns=["b"],
        )
        p.register_scan_task(
            "w120", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-09-03", "2024-12-31"), columns=["c"],
        )
        plan = p.plan()
        for w in plan.waves:
            assert w.estimated_memory_bytes <= p.wave_memory_budget
        # 预算只允许 2 列合并；第 3 波（另 1 列）留下 → 2 个 wave。
        assert len(plan.waves) == 2
        assert any(w.superset_coalesce for w in plan.waves)


# ---------------------------------------------------------------------------
# (e) PERF-009
# ---------------------------------------------------------------------------


class TestPerf009ScopeCompatibility:
    def test_compatibility_matrix(self):
        a = types.SimpleNamespace(instrument_scope=("a", "b"), universe_id="u1")
        b = types.SimpleNamespace(instrument_scope=("a", "b"), universe_id="u1")
        assert scope_compatibility(a, b) is ScopeCompatibility.EXACT

        c = types.SimpleNamespace(instrument_scope=("b", "c"), universe_id="u1")
        assert scope_compatibility(a, c) is ScopeCompatibility.UNION_COMPATIBLE

        d = types.SimpleNamespace(instrument_scope=("x", "y"), universe_id="u1")
        assert scope_compatibility(a, d) is ScopeCompatibility.INCOMPATIBLE

        e = types.SimpleNamespace(instrument_scope=("a", "b"), universe_id="u2")
        assert scope_compatibility(a, e) is ScopeCompatibility.INCOMPATIBLE

        full = types.SimpleNamespace(instrument_scope=(), universe_id="")
        assert scope_compatibility(full, a) is ScopeCompatibility.UNION_COMPATIBLE

    def test_csi300_not_merged_into_full_a_by_default(self):
        # CSI300（有名字 universe + 股票列表）不并入全 A（空 universe），除非成本证明。
        p = ReadWavePlanner(rows_estimate=500_000)
        p.register_scan_task(
            "csi", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-12-12", "2024-12-31"), columns=["close"],
            instrument_scope=("000001", "000002"), universe_id="csi300",
        )
        p.register_scan_task(
            "full", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-09-03", "2024-12-31"), columns=["close"],
            instrument_scope=(), universe_id="",
        )
        plan = p.plan()
        assert len(plan.waves) == 2, (
            "CSI300 不得并入全 A（成本模型未证明收益）"
        )

    def test_scope_merge_metrics_accounting(self):
        p = ReadWavePlanner(rows_estimate=1000)
        # 重叠 instrument scope + 同 universe → 合并可行，且报告 union 指标。
        p.register_scan_task(
            "r1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-12-12", "2024-12-31"), columns=["close"],
            instrument_scope=("000001", "000002"), universe_id="A",
        )
        p.register_scan_task(
            "r2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-09-03", "2024-12-31"), columns=["close"],
            instrument_scope=("000001", "000003"), universe_id="A",
        )
        plan = p.plan()
        assert len(plan.waves) == 1
        w = plan.waves[0]
        assert w.superset_coalesce is True
        # 额外 instrument 1 个（000002 或 000003 之一）→ 1 × rows×8。
        assert w.instrument_scope_union_extra_bytes == 1 * 1000 * 8
        assert plan.instrument_scope_union_extra_bytes == 1 * 1000 * 8
        assert plan.universe_scope_union_extra_rows == 0

    def test_universe_extra_rows_metric(self):
        p = ReadWavePlanner(rows_estimate=1000)
        w1 = ReadWave(
            wave_id=0, source_scope="s", dataset="d", snapshot_id="snap",
            columns=frozenset({"c"}), time_range=None,
            universe_id="csi300", instrument_scope=("000001",),
        )
        w2 = ReadWave(
            wave_id=1, source_scope="s", dataset="d", snapshot_id="snap",
            columns=frozenset({"c"}), time_range=None,
            universe_id="", instrument_scope=(),
        )
        _extra_bytes, extra_rows = p._scope_merge_metrics(w1, w2)
        assert extra_rows == abs(1000 - 250)  # rows('')=1000, rows('csi300')=250


# ---------------------------------------------------------------------------
# (f) PERF-010
# ---------------------------------------------------------------------------


class TestPerf010ByteAccounting:
    def test_baseline_gt_physical_union_saved_diff(self):
        p = ReadWavePlanner(rows_estimate=500_000)
        p.register_scan_task(
            "s1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-01-01", "2024-12-31"), columns=["close", "open"],
        )
        p.register_scan_task(
            "s2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=("2024-01-01", "2024-12-31"), columns=["close", "volume"],
        )
        plan = p.plan()
        w = plan.waves[0]
        assert w.baseline_duplicate_scan_bytes == 2 * (2 * 500_000 * 8)
        assert w.physical_union_scan_bytes == 3 * 500_000 * 8
        assert w.baseline_duplicate_scan_bytes > w.physical_union_scan_bytes
        assert w.saved_scan_bytes == (
            w.baseline_duplicate_scan_bytes - w.physical_union_scan_bytes
        )
        assert w.decoded_resident_bytes == w.estimated_memory_bytes
        # plan 级汇总一致。
        assert plan.baseline_duplicate_scan_bytes == w.baseline_duplicate_scan_bytes
        assert plan.physical_union_scan_bytes == w.physical_union_scan_bytes
        assert plan.saved_scan_bytes == w.saved_scan_bytes

    def test_physical_union_uses_compressed_footprints(self):
        fp = {
            "close": ProjectedColumnFootprint(
                "close", decoded_bytes=10_000_000, compressed_scan_bytes=3_000_000
            ),
            "open": ProjectedColumnFootprint(
                "open", decoded_bytes=2_000_000, compressed_scan_bytes=2_000_000
            ),
            "volume": ProjectedColumnFootprint(
                "volume", decoded_bytes=4_000_000, compressed_scan_bytes=4_000_000
            ),
        }
        p = ReadWavePlanner(rows_estimate=500_000)
        p.register_scan_task(
            "s1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["close", "open"], column_footprints=fp,
        )
        p.register_scan_task(
            "s2", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["close", "volume"], column_footprints=fp,
        )
        plan = p.plan()
        w = plan.waves[0]
        assert w.physical_union_scan_bytes == 3_000_000 + 2_000_000 + 4_000_000


# ---------------------------------------------------------------------------
# (g) PERF-011
# ---------------------------------------------------------------------------


class TestPerf011SourceRepresentation:
    def test_enum_values(self):
        assert SourceRepresentation.DUCKDB_RELATION.value == "duckdb_relation"
        assert SourceRepresentation.ARROW_BATCH_BLOCK.value == "arrow_batch_block"
        assert SourceRepresentation.POLARS_LAZY.value == "polars_lazy"
        assert SourceRepresentation.NUMPY_COLUMN_BLOCK.value == "numpy_column_block"
        assert SourceRepresentation.PANDAS_COLUMNS.value == "pandas_columns"

    def test_backend_mask_and_preferred(self):
        assert consumer_backend_mask(["duckdb_engine"]) == (
            SourceRepresentation.DUCKDB_RELATION.bit
        )
        assert preferred_representation_for_mask(
            SourceRepresentation.DUCKDB_RELATION.bit
        ) is SourceRepresentation.DUCKDB_RELATION
        # 混合 / 未知 → pandas fallback。
        assert preferred_representation_for_mask(
            SourceRepresentation.DUCKDB_RELATION.bit | SourceRepresentation.POLARS_LAZY.bit
        ) is SourceRepresentation.PANDAS_COLUMNS
        assert preferred_representation_for_mask(0) is SourceRepresentation.PANDAS_COLUMNS

    def test_representation_for_backend(self):
        assert representation_for_backend("pandas_numpy") is SourceRepresentation.PANDAS_COLUMNS
        assert representation_for_backend("duckdb_polars") is SourceRepresentation.DUCKDB_RELATION

    def test_wave_default_representation_pandas(self):
        p = ReadWavePlanner(rows_estimate=1000)
        p.register_scan_task(
            "s1", dataset="d", source_scope="ss", snapshot_id="snap",
            time_range=None, columns=["close"],
        )
        plan = p.plan()
        assert plan.waves[0].preferred_representation == "pandas_columns"
        assert plan.waves[0].consumer_backend_mask == 0

    def test_duckdb_consumer_sets_representation(self):
        dag = types.SimpleNamespace(
            tasks={
                "s1": _task("s1", "SOURCE_SCAN", cols=["close"]),
                "r1": _task(
                    "r1", "ROOT", inputs=("s1",), backend="duckdb_engine"
                ),
            }
        )
        plan = build_waves_from_dag(dag, rows_estimate=1000)
        w = plan.waves[0]
        assert w.consumer_tasks == ("r1",)
        assert w.preferred_representation == "duckdb_relation"
        assert w.consumer_backend_mask == SourceRepresentation.DUCKDB_RELATION.bit


class TestSourceWaveExecutorRepresentation:
    def test_executor_emits_preferred_representation(self):
        from runtime.buffer_ref import SourceWaveExecutor

        class _Src:
            dataset = "ashare_daily"
            snapshot_token = "s1"

            def prefetch_columns(self, cols):
                pass

        ex = SourceWaveExecutor(_Src(), ctx=None)
        ref = ex.execute_wave(
            types.SimpleNamespace(
                wave_id=0, columns=frozenset({"close"}), task_ids=("t1",),
                estimated_scan_bytes=8, estimated_memory_bytes=8,
                preferred_representation="duckdb_relation",
            ),
            consumer_ids=("t1",),
        )
        assert ref is not None
        assert ref.representation == "duckdb_relation"
        assert ref.location == "source.native"

    def test_executor_default_remains_pandas(self):
        from runtime.buffer_ref import SourceWaveExecutor

        class _Src:
            dataset = "ashare_daily"
            snapshot_token = "s1"

            def prefetch_columns(self, cols):
                pass

        ex = SourceWaveExecutor(_Src(), ctx=None)
        ref = ex.execute_wave(
            types.SimpleNamespace(
                wave_id=1, columns=frozenset({"close"}), task_ids=("t1",),
                estimated_scan_bytes=8, estimated_memory_bytes=8,
            ),
            consumer_ids=("t1",),
        )
        assert ref.representation == "pandas_columns"
        assert ref.location == "source.column_cache"


# ---------------------------------------------------------------------------
# (h) PERF-012
# ---------------------------------------------------------------------------


class TestPerf012TypedRecovery:
    def test_decision_table(self):
        cases = [
            (RuntimeError("wave oversized, exceeds budget"), WaveRecoveryCategory.OVERSIZED, "split_2_way", 0, 2),
            (RuntimeError("duckdb representation unsupported"), WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED, "downgrade_representation", 0, 1),
            (OSError("transient connection reset"), WaveRecoveryCategory.TRANSIENT_IO, "bounded_retry", 3, 1),
            (MemoryError("memory pressure exceeded"), WaveRecoveryCategory.MEMORY_PRESSURE, "shrink", 0, 2),
            (RuntimeError("permanent source schema mismatch"), WaveRecoveryCategory.PERMANENT_SOURCE_ERROR, "abort", 0, 1),
        ]
        for exc, cat, action, retry, split in cases:
            plan = recover_wave_failure(exc, wave=None)
            assert plan.category is cat, f"{exc}: {plan}"
            assert plan.action == action
            assert plan.retry_limit == retry
            assert plan.split_factor == split
            assert isinstance(plan, WaveRecoveryPlan)

    def test_classify_wave_failure(self):
        assert classify_wave_failure(RuntimeError("too large")) is WaveRecoveryCategory.OVERSIZED
        assert classify_wave_failure(RuntimeError("backend not supported")) is (
            WaveRecoveryCategory.REPRESENTATION_UNSUPPORTED
        )
        assert classify_wave_failure(ConnectionError("timeout")) is WaveRecoveryCategory.TRANSIENT_IO
        assert classify_wave_failure(MemoryError("oom")) is WaveRecoveryCategory.MEMORY_PRESSURE
        assert classify_wave_failure(ValueError("bad data")) is (
            WaveRecoveryCategory.PERMANENT_SOURCE_ERROR
        )

    def test_hard_gate_failed_wave_to_n_root_scans(self):
        # 有 typed recovery plan → 不 fallback → hard gate 0。
        for exc in (
            RuntimeError("wave oversized"),
            RuntimeError("representation unsupported"),
            OSError("transient io"),
            MemoryError("pressure"),
        ):
            assert would_fallback_to_n_root_scans(exc, wave=None) is False
            assert hard_gate_failed_wave_to_n_root_scans(exc, wave=None) == 0
        # 只有 PERMANENT_SOURCE_ERROR → abort → fallback。
        assert would_fallback_to_n_root_scans(RuntimeError("permanent error"), None) is True
        assert hard_gate_failed_wave_to_n_root_scans(RuntimeError("permanent error"), None) == 1
