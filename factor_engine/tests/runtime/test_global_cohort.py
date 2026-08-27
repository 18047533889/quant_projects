# -*- coding: utf-8 -*-
"""P78-P7/P8: GlobalFactorManifest + GlobalSubexpressionIndex + ExecutionCohort。

覆盖（任务指定）：
    (a) 100k 因子 → manifest 只持指纹（紧凑 dict，非 DAG；对象数 << 全量 IR）。
    (b) cohort 分组：相同 key 的因子同组；不同 key 分离。
    (c) 动态 cohort 大小：小预算 → 小 cohort；大预算 → 大 cohort（单调）。
    (d) CSE 决策：高复用 + 低内存租金 → MATERIALIZE_MEMORY；低复用 + 高租金 → RECOMPUTE。
    (e) last-consumer free：最后一个消费者完成后物化节点被释放（refcount 0 → freed）。
    (f) 跨 cohort 复用：两个不同 cohort 共享的子表达式物化一次并复用（不重算）。

设计要点：
    - 复用既有框架（AdaptiveBatchScheduler / ResourceBroker / CSECacheOptimizer /
      SpillStore / PhysicalFactorDAG），本测试只验证新增的 manifest + index +
      cohort 层。
    - 不触碰 mining/ 与 cleaned_operators/。
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import pytest

from factor_engine.runtime.execution_cohort import (
    ExecutionCohortExecutor,
    ExecutionCohortKey,
    cohort_key_for,
    cohort_size_for,
    group_manifest_into_cohorts,
)
from factor_engine.runtime.global_factor_manifest import (
    FactorFingerprint,
    GlobalFactorManifest,
    build_fingerprint,
)
from factor_engine.runtime.global_subexpression_index import (
    MATERIALIZE_MEMORY,
    MATERIALIZE_SPILL,
    RECOMPUTE,
    GlobalSubexpressionIndex,
)


# ---------------------------------------------------------------------------
# (a) 100k manifest 紧凑（anti-mega-DAG）
# ---------------------------------------------------------------------------


def _build_manifest(n: int) -> GlobalFactorManifest:
    m = GlobalFactorManifest()
    for i in range(n):
        fp = build_fingerprint(
            factor_name=f"f{i:05d}",
            subexpression_semantic_hash=f"sub:{i % 100}",
            consumer_count=1 + (i % 5),
            input_fields=("open", "close", "volume"),
            lookback=20,
            execution_axis="panel",
            backend_affinity="polars",
            estimated_cost=50.0,
        )
        m.register(f"f{i:05d}", fp)
    return m


def test_manifest_100k_is_compact_not_dag():
    manifest = _build_manifest(100_000)
    assert len(manifest) == 100_000
    # 紧凑：manifest 字节 << 全量 IR 字节。
    assert manifest.is_compact()
    assert manifest.estimate_manifest_bytes() < manifest.equivalent_full_ir_bytes()
    # 反 mega-DAG：manifest 是扁平 dict，不是 DAG。
    assert manifest.is_dag() is False
    # 指纹是 frozen dataclass（无子图引用），可安全常驻 100k 个。
    fp = manifest.get("f00000")
    assert isinstance(fp, FactorFingerprint)
    assert fp.factor_semantic_hash
    assert fp.subexpression_semantic_hash == "sub:0"


def test_manifest_compact_ratio():
    # 100k 因子：manifest 应远小于全量 IR（至少 5 倍）。
    manifest = _build_manifest(100_000)
    ratio = manifest.equivalent_full_ir_bytes() / max(1, manifest.estimate_manifest_bytes())
    assert ratio >= 5.0


# ---------------------------------------------------------------------------
# (b) cohort 分组
# ---------------------------------------------------------------------------


def test_cohort_grouping_same_key_together_different_separate():
    manifest = GlobalFactorManifest()
    # 两个因子同 market/universe/frequency/source/field_family → 同组。
    manifest.register(
        "fA",
        build_fingerprint(
            factor_name="fA", subexpression_semantic_hash="s1", consumer_count=2,
            lookback=20, execution_axis="panel", backend_affinity="polars",
        ),
    )
    manifest.register(
        "fB",
        build_fingerprint(
            factor_name="fB", subexpression_semantic_hash="s2", consumer_count=2,
            lookback=20, execution_axis="panel", backend_affinity="polars",
        ),
    )
    # 不同 backend_affinity → 不同组（分组函数从指纹读取该维度）。
    manifest.register(
        "fC",
        build_fingerprint(
            factor_name="fC", subexpression_semantic_hash="s3", consumer_count=2,
            lookback=20, execution_axis="panel", backend_affinity="duckdb",
        ),
    )
    groups = group_manifest_into_cohorts(
        manifest, market="A", universe="u1", frequency="daily",
        source_scope="ashare", required_field_family="price",
    )
    key_ab = cohort_key_for(
        manifest.get("fA"), market="A", universe="u1", frequency="daily",
        source_scope="ashare", required_field_family="price",
    )
    key_c = cohort_key_for(
        manifest.get("fC"), market="A", universe="u1", frequency="daily",
        source_scope="ashare", required_field_family="price",
    )
    # 不同 backend_affinity → 不同 cohort key。
    assert key_ab != key_c
    # 相同 key 的因子同组。
    assert "fA" in groups[key_ab]
    assert "fB" in groups[key_ab]
    # 不同 key 的因子分离（fC 在 duckdb 组，不在 polars 组）。
    assert "fC" in groups[key_c]
    assert "fC" not in groups[key_ab]


def test_cohort_key_8_dimensions():
    k = ExecutionCohortKey(
        market="A", universe="u1", frequency="daily", source_scope="ashare",
        required_field_family="price", lookback_bucket="20-60",
        execution_axis="panel", backend_affinity="polars",
    )
    assert len(k.to_tuple()) == 8
    # 任一维度不同 → key 不同。
    k2 = ExecutionCohortKey(
        market="A", universe="u1", frequency="daily", source_scope="ashare",
        required_field_family="price", lookback_bucket="20-60",
        execution_axis="panel", backend_affinity="duckdb",
    )
    assert k != k2


# ---------------------------------------------------------------------------
# (c) 动态 cohort 大小（单调）
# ---------------------------------------------------------------------------


def test_cohort_size_scales_with_budget_monotonic():
    per = 100.0
    small = cohort_size_for(1_000, per)
    large = cohort_size_for(1_000_000, per)
    assert small < large
    # 单调：预算越大 → cohort 越大。
    sizes = [cohort_size_for(b, per) for b in (100, 1_000, 10_000, 100_000, 1_000_000)]
    assert sizes == sorted(sizes)
    # 下限：至少 1。
    assert cohort_size_for(1, per) >= 1
    # 精确：budget // per_factor。
    assert cohort_size_for(10_000, 100.0) == 100


def test_cohort_size_not_hardcoded_1000():
    # 小预算 → 远小于 1000；大预算 → 可大于 1000。
    per = 100.0
    assert cohort_size_for(1_000, per) < 1000
    assert cohort_size_for(1_000_000, per) > 1000


# ---------------------------------------------------------------------------
# (d) CSE 决策（cost-aware，非「缓存每一个重复」）
# ---------------------------------------------------------------------------


def test_cse_decision_high_reuse_low_rent_materialize_memory():
    idx = GlobalSubexpressionIndex()
    idx.register("sub:hot", recompute_cost=100.0, materialized_bytes=1_000, reuse_count=10)
    decision = idx.decide("sub:hot")
    assert decision.action == MATERIALIZE_MEMORY


def test_cse_decision_low_reuse_high_rent_recompute():
    idx = GlobalSubexpressionIndex()
    # 低复用（2 次）+ 高内存租金（大 materialized_bytes × 高 rent）。
    idx.register("sub:cold", recompute_cost=1.0, materialized_bytes=10**9, reuse_count=2)
    decision = idx.decide("sub:cold")
    assert decision.action == RECOMPUTE


def test_cse_decision_single_consumer_recompute():
    idx = GlobalSubexpressionIndex()
    idx.register("sub:single", recompute_cost=100.0, materialized_bytes=1_000, reuse_count=1)
    assert idx.decide("sub:single").action == RECOMPUTE


def test_cse_decision_spill_when_memory_rent_too_high_but_spill_ok():
    # 高复用 + 高内存租金（内存态不划算）但 spill 后划算 → MATERIALIZE_SPILL。
    idx = GlobalSubexpressionIndex(
        memory_rent_per_byte=1e-3,  # 高内存租金
        spill_cost_per_byte=1e-9,    # 低 spill 成本
        transfer_cost_per_byte=1e-9,
    )
    idx.register("sub:spill", recompute_cost=100.0, materialized_bytes=10**6, reuse_count=10)
    decision = idx.decide("sub:spill")
    assert decision.action == MATERIALIZE_SPILL


# ---------------------------------------------------------------------------
# (e) last-consumer free（refcount 0 → freed）
# ---------------------------------------------------------------------------


def test_last_consumer_free():
    idx = GlobalSubexpressionIndex()
    idx.register("sub:x", recompute_cost=10.0, materialized_bytes=1_000, reuse_count=2)
    idx.materialize("sub:x", "value", action=MATERIALIZE_MEMORY)
    assert idx.is_materialized("sub:x")
    # 两个消费者。
    idx.add_consumer("sub:x")
    idx.add_consumer("sub:x")
    assert idx.refcount("sub:x") == 2
    # 第一个消费者完成 → 仍物化。
    assert idx.release_consumer("sub:x") is False
    assert idx.is_materialized("sub:x")
    # 最后一个消费者完成 → refcount 0 → 释放。
    assert idx.release_consumer("sub:x") is True
    assert idx.is_materialized("sub:x") is False
    assert idx.get_materialized("sub:x") is None


# ---------------------------------------------------------------------------
# (f) 跨 cohort 复用（物化一次，不重算）
# ---------------------------------------------------------------------------


def test_cross_cohort_reuse_materialized_once():
    idx = GlobalSubexpressionIndex()
    # 子表达式被两个不同 cohort 的因子共享（reuse_count=2）。
    idx.register("sub:shared", recompute_cost=50.0, materialized_bytes=1_000, reuse_count=2)
    decision = idx.decide("sub:shared")
    assert decision.action == MATERIALIZE_MEMORY

    # 物化一次。
    idx.materialize("sub:shared", "materialized_value", action=MATERIALIZE_MEMORY)
    assert idx.is_materialized("sub:shared")

    # 两个 cohort 各一个消费者。
    idx.add_consumer("sub:shared")  # cohort A
    idx.add_consumer("sub:shared")  # cohort B

    # cohort A 消费：命中物化值（不重算）。
    assert idx.get_materialized("sub:shared") == "materialized_value"
    # cohort B 消费：仍命中同一物化值（不重算）。
    assert idx.get_materialized("sub:shared") == "materialized_value"

    # 两个消费者都完成 → 释放。
    idx.release_consumer("sub:shared")
    idx.release_consumer("sub:shared")
    assert idx.is_materialized("sub:shared") is False


def test_cross_cohort_reuse_telemetry():
    idx = GlobalSubexpressionIndex()
    idx.register("sub:shared", recompute_cost=50.0, materialized_bytes=1_000, reuse_count=2)
    idx.materialize("sub:shared", "v", action=MATERIALIZE_MEMORY)
    idx.record_reuse_saved("sub:shared")
    idx.record_recompute("sub:other")
    tele = idx.telemetry()
    assert tele.materialize_memory_count == 1
    assert tele.recompute_count == 1
    assert tele.reuse_saved_ms > 0
    assert tele.peak_cache_bytes >= 1_000


# ---------------------------------------------------------------------------
# 集成：ExecutionCohortExecutor 消费 manifest + index
# ---------------------------------------------------------------------------


def test_cohort_executor_plans_and_executes():
    manifest = _build_manifest(50)
    idx = GlobalSubexpressionIndex()
    executor = ExecutionCohortExecutor(manifest=manifest, subexpression_index=idx)

    def _build_dag(names, key):
        # 真实路径会编译成 PhysicalFactorDAG；这里返回轻量占位（测试只验证编排）。
        return {"names": names, "key": key}

    def _run_cohort(dag, names, key):
        return {n: f"result:{n}" for n in names}

    results = executor.execute(
        build_dag=_build_dag,
        run_cohort=_run_cohort,
        market="A", universe="u1", frequency="daily",
        source_scope="ashare", required_field_family="price",
    )
    assert len(results) >= 1
    total = sum(len(r.factor_names) for r in results)
    assert total == 50
    # 每个因子都执行了。
    all_names = [n for r in results for n in r.factor_names]
    assert len(all_names) == 50
    assert len(set(all_names)) == 50
