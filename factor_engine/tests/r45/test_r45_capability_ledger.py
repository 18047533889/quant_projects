# -*- coding: utf-8 -*-
"""R45-IncrementalCapabilityLedger: 认证账本 + 扩展 capability matrix 测试。

覆盖 :mod:`runtime.incremental_contract` 的 R45 扩展：
  1. ``incremental_capability_matrix_extended`` 覆盖权威 registry 全量。
  2. ``build_incremental_certification_ledger`` 认证分布；同 canonical 不既为
     TRUE_INCREMENTAL 又为 NOT_CERTIFIED。
  3. 经 parity 证明的 9 个状态算子为 TRUE_INCREMENTAL（至少不是 FULL_REPLAY_ONLY）。
  4. ``lifecycle_incremental_eligible`` 聚合计数诚实（不超过权威总数，且与人工判定
     一致）；严格硬门 ``production_usable`` 是 lifecycle 资格的真子集（R50）。
  5. 维度探测不虚报：当前 head 无 duckdb / q / numba 后端时对应维度全 False。
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.incremental_contract import (
    IncrementalCertificationLevel,
    IncrementalMode,
    build_incremental_certification_ledger,
    classify_certification_level,
    incremental_capability_matrix_extended,
)

load_all()


# ---------------------------------------------------------------------------
# 1. 权威总数与矩阵完整性
# ---------------------------------------------------------------------------
def test_extended_matrix_covers_full_registry_count() -> None:
    """扩展矩阵对 OperatorRegistry._catalog 权威全集每 canonical 恰好一行。"""
    canonicals = sorted(OperatorRegistry._catalog)
    rows = incremental_capability_matrix_extended()
    assert len(rows) == len(canonicals) == len({r["canonical"] for r in rows})
    # 矩阵 canonical 集合与权威 registry 完全一致（无缺漏 / 无多算）。
    assert {r["canonical"] for r in rows} == set(canonicals)


def test_ledger_operator_total_equals_authoritative_count() -> None:
    """账本 operator_total == 权威 registry 计数（不是硬编码 466）。"""
    ledger = build_incremental_certification_ledger()
    assert ledger.operator_total == len(OperatorRegistry._catalog)
    assert ledger.operator_total == 1624


# ---------------------------------------------------------------------------
# 2. 认证分布一致性
# ---------------------------------------------------------------------------
def test_ledger_no_canonical_is_both_true_and_not_certified() -> None:
    """任何 canonical 不会既列于 TRUE_INCREMENTAL 又列于 NOT_CERTIFIED。"""
    ledger = build_incremental_certification_ledger()
    true_set = set(ledger.true_incremental_canonicals)
    not_set = set(ledger.not_certified_canonicals)
    assert not (true_set & not_set)
    # 且两者并集覆盖全部 canonical（无遗漏）。
    assert true_set | not_set == set(OperatorRegistry._catalog)


def test_ledger_distribution_partitions_every_row() -> None:
    """distribution 计数总和 == operator_total，且仅含合法 5 级。"""
    ledger = build_incremental_certification_ledger()
    assert sum(ledger.distribution.values()) == ledger.operator_total
    assert set(ledger.distribution) <= {
        level.value for level in IncrementalCertificationLevel
    }


# ---------------------------------------------------------------------------
# 3. parity 证明的状态算子认证
# ---------------------------------------------------------------------------
def test_parity_proven_stateful_canonicals_are_true_incremental() -> None:
    """parity 证明的 9 个状态算子全部为 TRUE_INCREMENTAL（不是 FULL_REPLAY_ONLY）。"""
    from factor_engine.runtime.incremental_parity import SEGMENTED_CANONICALS

    ledger = build_incremental_certification_ledger()
    true_set = set(ledger.true_incremental_canonicals)
    for canon in SEGMENTED_CANONICALS:
        assert canon in true_set, f"{canon} 应经 parity 认证为 TRUE_INCREMENTAL"
        level = classify_certification_level(canon)
        assert level is IncrementalCertificationLevel.TRUE_INCREMENTAL
        assert level is not IncrementalCertificationLevel.FULL_REPLAY_ONLY


def test_unproven_canonical_is_not_certified() -> None:
    """未证明的 canonical 必须 NOT_CERTIFIED（诚实，不虚报认证）。"""
    for canon in ("ts_mean", "log", "ts_delta"):
        level = classify_certification_level(canon)
        assert level is IncrementalCertificationLevel.NOT_CERTIFIED


# ---------------------------------------------------------------------------
# 4. 生命周期资格 + 生产可用硬门诚实性
# ---------------------------------------------------------------------------
def test_live_count_is_honest_subset_of_total() -> None:
    """lifecycle 资格数 ≤ operator_total，且每个资格 canonical 维度证据齐全。"""
    ledger = build_incremental_certification_ledger()
    rows = incremental_capability_matrix_extended()
    assert 0 <= ledger.lifecycle_incremental_eligible <= ledger.operator_total
    # 行级聚合与账本一致。
    assert sum(1 for r in rows if r["lifecycle_incremental_eligible"]) == ledger.lifecycle_incremental_eligible
    # 每个资格 canonical：lifecycle=production & prod_certified & not hidden & 非 FULL_REPLAY。
    for r in rows:
        if r["lifecycle_incremental_eligible"]:
            assert r["incremental_mode"] != IncrementalMode.FULL_REPLAY.value
            cat = OperatorRegistry._catalog[r["canonical"]]
            assert cat.get("lifecycle_status") == "production"
            assert cat.get("production_certified") is True
            assert not cat.get("hidden_from_default_mining")


def test_production_usable_is_strict_subset_of_lifecycle_eligible() -> None:
    """严格硬门 production_usable 是 lifecycle 资格的真子集（R50 诚实性）。"""
    ledger = build_incremental_certification_ledger()
    rows = incremental_capability_matrix_extended()
    assert 0 <= ledger.production_usable <= ledger.lifecycle_incremental_eligible
    assert sum(1 for r in rows if r["production_usable"]) == ledger.production_usable
    # 每个 production_usable canonical 必须满足全部硬门条件。
    for r in rows:
        if r["production_usable"]:
            assert r["lifecycle_incremental_eligible"]
            assert r["ashare_source_ready"]
            assert r["pit"]
            assert r["pandas"] or r["polars"]
            assert r["certification_level"] == IncrementalCertificationLevel.TRUE_INCREMENTAL.value
            assert r["e2e"]


def test_live_production_usable_alias_matches_lifecycle_eligible() -> None:
    """向后兼容别名 live_production_usable == lifecycle_incremental_eligible。"""
    ledger = build_incremental_certification_ledger()
    rows = incremental_capability_matrix_extended()
    assert ledger.live_production_usable == ledger.lifecycle_incremental_eligible
    for r in rows:
        assert r["live_production_usable"] == r["lifecycle_incremental_eligible"]


# ---------------------------------------------------------------------------
# 5. 维度诚实性
# ---------------------------------------------------------------------------
def test_unproven_dimensions_reported_false_not_false_true() -> None:
    """当前 head 无 duckdb / q / numba 后端时，对应维度对每个 canonical 均为 False。"""
    rows = incremental_capability_matrix_extended()
    for r in rows:
        assert r["duckdb"] is False, r["canonical"]
        assert r["q"] is False, r["canonical"]
        assert r["numba"] is False, r["canonical"]


def test_extended_matrix_carries_all_r45_dimension_fields() -> None:
    """扩展矩阵每行包含 R45 新增维度字段。"""
    rows = incremental_capability_matrix_extended(canonicals=["ts_mean", "ts_ema", "log"])
    expected = {
        "certification_level", "direct_mining", "ashare_source_ready", "pit",
        "pandas", "polars", "duckdb", "q", "numba", "incremental", "e2e",
        "lifecycle_incremental_eligible", "production_usable", "backend_status",
        "live_production_usable",
    }
    for r in rows:
        assert expected <= set(r), r["canonical"]
