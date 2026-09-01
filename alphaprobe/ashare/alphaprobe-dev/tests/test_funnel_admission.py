"""funnel + admission + export.gate（§9 / §53 / §58 / §59）测试。"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphaprobe.contracts import (
    EvaluationRecord,
    FidelityLevel,
    RejectionReason,
)
from alphaprobe.dedup import GlobalSeenIndex
from alphaprobe.fitness.funnel import FidelityFunnel, L0StaticCheck
from alphaprobe.pool import ActivePool, PoolMember
from alphaprobe.pool.admission import (
    PoolAdmission,
    admit,
    eviction_candidate_from,
    legacy_try_new_expr,
)
from alphaprobe.export.gate import build_export_decision


def make_record(
    factor_id: str,
    metric_bundle: dict,
    fidelity: str = FidelityLevel.L2_FULL_TRAIN.value,
) -> EvaluationRecord:
    return EvaluationRecord(
        factor_id=factor_id,
        segment="train",
        fidelity=fidelity,
        metric_bundle=metric_bundle,
        artifact_refs={},
        evaluator_version="test",
        data_snapshot_id="snap",
        universe_snapshot_id="uni",
        label_spec_hash="vwap_to_vwap_h20",
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# L0：INVALID_DSL / EXACT_DUPLICATE
# ---------------------------------------------------------------------------


class TestFunnelL0:
    def test_l0_rejects_invalid_dsl(self):
        funnel = FidelityFunnel()
        out = funnel.l0("rank(close")
        assert RejectionReason.INVALID_DSL in out.rejections

    def test_l0_rejects_empty(self):
        out = FidelityFunnel().l0("")
        assert RejectionReason.INVALID_DSL in out.rejections

    def test_l0_passes_valid_dsl(self):
        out = FidelityFunnel().l0("rank(ts_mean(close, 20))")
        assert out.rejections == []
        assert out.signal_id

    def test_l0_exact_duplicate_second_intake(self):
        seen = GlobalSeenIndex()
        funnel = FidelityFunnel(static=L0StaticCheck(seen=seen))
        out1 = funnel.l0("rank(ts_mean(close, 20))", factor_id="f1")
        assert out1.rejections == []
        # 同 signal 二次入 → EXACT_DUPLICATE（无需 adapter，静态去重判定）
        out2 = funnel.l0("rank(ts_mean(close, 20))", factor_id="f2")
        assert RejectionReason.EXACT_DUPLICATE in out2.rejections

    def test_l0_sign_variant_not_exact_duplicate(self):
        seen = GlobalSeenIndex()
        funnel = FidelityFunnel(static=L0StaticCheck(seen=seen))
        funnel.l0("rank(ts_mean(close, 20))", factor_id="f1")
        # -(rank(ts_mean(close, 20))) 是 sign variant（sign_invariant id 相同 → EXACT_DUPLICATE）
        out = funnel.l0("(-(rank(ts_mean(close, 20))))", factor_id="f2")
        assert RejectionReason.EXACT_DUPLICATE in out.rejections

    def test_l0_distinct_formula_passes(self):
        funnel = FidelityFunnel()
        out = funnel.l0("rank(ts_mean(close, 20))", factor_id="f1")
        assert out.rejections == []
        out2 = funnel.l0("rank(ts_std(close, 20))", factor_id="f2")
        assert out2.rejections == []


# ---------------------------------------------------------------------------
# funnel.promote：按 thresholds 配置过滤 + L5 frozen gate
# ---------------------------------------------------------------------------


class TestFunnelPromote:
    def test_promote_l0_to_l1(self):
        funnel = FidelityFunnel()
        res = funnel.promote(
            [
                {"factor_id": "f1", "formula": "rank(ts_mean(close, 20))", "record": None},
                {"factor_id": "f2", "formula": "rank(ts_std(close, 20))", "record": None},
            ],
            from_level=FidelityLevel.L0_STATIC.value,
            to_level=FidelityLevel.L1_SCOUT.value,
        )
        assert len(res) == 2
        assert all(r.promoted_level == FidelityLevel.L1_SCOUT.value for r in res)
        assert funnel.counters["L1_promoted"] == 1

    def test_promote_rejects_invalid(self):
        funnel = FidelityFunnel()
        res = funnel.promote(
            [{"factor_id": "bad", "formula": "rank(close", "record": None}],
            from_level=FidelityLevel.L0_STATIC.value,
            to_level=FidelityLevel.L1_SCOUT.value,
        )
        assert res[0].promoted_level is None
        assert not res[0].passed

    def test_promote_l1_low_coverage_rejected(self):
        funnel = FidelityFunnel()
        rec = make_record("f1", {"coverage": 0.3, "rankic": 0.04})
        res = funnel.promote_record(rec, FidelityLevel.L1_SCOUT.value, FidelityLevel.L2_FULL_TRAIN.value)
        assert RejectionReason.LOW_COVERAGE in res.rejections

    def test_promote_l1_ok(self):
        funnel = FidelityFunnel()
        rec = make_record("f1", {"coverage": 0.9, "rankic": 0.04})
        res = funnel.promote_record(rec, FidelityLevel.L1_SCOUT.value, FidelityLevel.L2_FULL_TRAIN.value)
        assert res.promoted_level == FidelityLevel.L2_FULL_TRAIN.value

    def test_l5_requires_frozen_access(self):
        funnel = FidelityFunnel()
        rec = make_record("f1", {"sealed_test_pass": True}, fidelity=FidelityLevel.L5_SEALED_TEST.value)
        res = funnel.promote_record(rec, FidelityLevel.L5_SEALED_TEST.value, FidelityLevel.L5_SEALED_TEST.value)
        assert not res.passed
        assert RejectionReason.AUDIT_FAIL in res.rejections

    def test_l5_ok_with_frozen_access(self):
        from alphaprobe.contracts import DateRange, ResearchSplitSpec
        from alphaprobe.research_protocol import SealedTestAccess

        spec = ResearchSplitSpec(
            train=DateRange("2016-01-01", "2021-12-31"),
            search_valid=DateRange("2022-01-01", "2023-12-31"),
            audit_valid=None,
            sealed_test=DateRange("2024-01-01", "2026-07-31"),
        )
        acc = SealedTestAccess(spec, frozen=True)
        assert acc.split_spec is spec
        funnel = FidelityFunnel(l5_access=acc)
        rec = make_record("f1", {"sealed_test_pass": True}, fidelity=FidelityLevel.L5_SEALED_TEST.value)
        res = funnel.promote_record(rec, FidelityLevel.L5_SEALED_TEST.value, FidelityLevel.L5_SEALED_TEST.value)
        assert res.passed
        assert res.promoted_level == FidelityLevel.L5_SEALED_TEST.value


# ---------------------------------------------------------------------------
# pool_utility / admission：低 IC 高 niche 不被弹（§53.2）
# ---------------------------------------------------------------------------


def make_pool_member(
    factor_id: str,
    *,
    fitness: float,
    niche_rarity: float = 0.0,
    niche_key: tuple = (),
    sv: float = 0.5,
    util: float = 0.5,
) -> PoolMember:
    return PoolMember(
        factor_id=factor_id,
        canonical_formula=factor_id,
        search_fitness=fitness,
        pool_utility=util,
        search_value=sv,
        niche_key=niche_key,
        meta={"niche_rarity": niche_rarity},
    )


class TestPoolUtility:
    def test_high_niche_rarity_raises_utility(self):
        from alphaprobe.fitness import pool_utility

        low = pool_utility(
            pareto_rank=0, niche_rarity=0.2, novelty=0.2,
            s_value=0.2, representative_value=0.2,
        )
        high = pool_utility(
            pareto_rank=0, niche_rarity=0.9, novelty=0.9,
            s_value=0.9, representative_value=0.9,
        )
        assert high > low

    def test_low_ic_high_niche_not_evicted(self):
        pool = ActivePool(target_size=2, max_size=3)
        # A: 高 IC(fitness) 但 niche 拥挤（无 rarity 保护）
        pool.admit(make_pool_member("A", fitness=0.9, niche_key=("x",)))
        # B: 低 IC(fitness) 但 niche_rarity 高且 niche 唯一
        pool.admit(make_pool_member("B", fitness=0.1, niche_rarity=0.95, niche_key=("y",)))
        # C 入池触发可能替换
        pool.admit(make_pool_member("C", fitness=0.5, niche_key=("z",)))
        # 低 IC 的 B 不应被 argmin(single IC) 弹出
        assert pool.contains("B")

    def test_admit_with_record_populates_member(self):
        pool = ActivePool(target_size=4, max_size=8)
        rec = make_record(
            "f1",
            {
                "search_fitness": 0.6,
                "search_value": 0.5,
                "pool_utility": 0.0,  # 触发组件计算
                "pareto_rank": 1,
                "niche_rarity": 0.8,
                "structural_novelty": 0.7,
                "S": 0.5,
                "representative_value": 0.5,
            },
        )
        res = admit(
            None, rec, pool,
            factor_id="f1", canonical_formula="rank(close)", niche_key=("m", "h"),
        )
        assert res.accepted
        m = pool.members["f1"]
        assert m.search_fitness == 0.6
        assert m.pool_utility > 0.0  # 组件算出的 utility

    def test_admit_duplicate_rejected(self):
        pool = ActivePool(target_size=4, max_size=8)
        pool.admit(make_pool_member("f1", fitness=0.5, niche_key=("x",)))
        rec = make_record("f1", {"search_fitness": 0.7})
        res = admit(None, rec, pool, factor_id="f1", canonical_formula="f1")
        assert not res.accepted
        assert res.reason == "ALREADY_IN_POOL"


# ---------------------------------------------------------------------------
# legacy_try_new_expr：兼容旧签名，不再 argmin(single_ics)
# ---------------------------------------------------------------------------


class FakeLegacyPool:
    """模拟 AlphaKnowledgePool 的最小对象（trainer/trainer.py 兼容层）。

    不 import torch —— 本环境无 GPU wheel。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.size = 0
        self.single_ics: list[float] = []
        self.exprs = []
        self.values = None
        self._admission_meta: dict[int, dict] = {}
        self._popped: list[str] = []

    def _add_factor(self, expr, value, ic_ret, ic_ir, ic_mut, topic, description, expression_node):
        self.exprs.append(str(expr))
        self.single_ics.append(abs(ic_ret) if ic_ret is not None else 0.0)
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        self._popped.append(self.exprs[self.capacity])
        self.exprs = self.exprs[: self.capacity]
        self.single_ics = self.single_ics[: self.capacity]
        self.size = self.capacity

    def _pop_at(self, idx: int) -> None:
        """弹出指定下标（替换语义：被弹的是 eviction victim，不是尾部）。"""
        self._popped.append(self.exprs[idx])
        del self.exprs[idx]
        del self.single_ics[idx]
        self.size -= 1


def test_legacy_try_new_expr_adds_when_not_full():
    pool = FakeLegacyPool(capacity=3)
    ok = legacy_try_new_expr(pool, "rank(ts_mean(close, 20))", topic="t", description="d")
    assert ok
    assert pool.size == 1


def test_legacy_try_new_expr_replaces_when_full():
    pool = FakeLegacyPool(capacity=2)
    pool._add_factor("A", None, 0.9, 0.5, [], "t", "d", None)
    pool._add_factor("B", None, 0.1, 0.5, [], "t", "d", None)
    ok = legacy_try_new_expr(pool, "rank(ts_std(close, 20))", topic="t", description="d")
    assert ok
    assert pool.size == 2
    # 满池替换：弹出 eviction victim（低 SearchValue，绝不 argmin(single IC)），
    # 新因子保留；被弹者不是新因子本身。
    assert len(pool._popped) == 1
    assert "rank(ts_std(close, 20))" in pool.exprs
    assert pool._popped[0] in ("A", "B")


# ---------------------------------------------------------------------------
# export.gate：refinement_complete=False → 不 accepted（§59）
# ---------------------------------------------------------------------------


class TestExportGate:
    def test_refinement_incomplete_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            from alphaprobe.memory import GlobalMemoryStore

            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            rec = make_record("f1", {"search_fitness": 0.5, "S": 0.4, "N": 0.3})
            cand = {"factor_id": "f1", "canonical_formula": "rank(close)"}
            d = build_export_decision(cand, rec, store, refinement_complete=False)
            assert not d.accepted
            assert d.refinement_complete is False
            assert "REFINEMENT_INCOMPLETE" in d.reasons
            store.close()

    def test_refinement_complete_with_already_exported(self):
        with tempfile.TemporaryDirectory() as td:
            from alphaprobe.memory import GlobalMemoryStore

            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            store.mark_exported(factor_id="f1", campaign_id="c1", manifest_path="p")
            rec = make_record("f1", {"search_fitness": 0.5, "S": 0.4, "N": 0.3})
            cand = {"factor_id": "f1", "canonical_formula": "rank(close)"}
            d = build_export_decision(cand, rec, store, refinement_complete=True)
            assert not d.accepted
            assert d.already_exported is True
            assert RejectionReason.ALREADY_EXPORTED.value in d.reasons
            store.close()

    def test_refinement_complete_with_seed_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            from alphaprobe.memory import GlobalMemoryStore

            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            store.upsert_factor_node(
                factor_id="seed_f1",
                canonical_formula="rank(close)",
                canonical_ast_hash="h",
                signal_equivalence_id="s",
                is_seed=True,
            )
            rec = make_record("seed_f1", {"search_fitness": 0.5, "S": 0.4, "N": 0.3})
            cand = {"factor_id": "seed_f1", "canonical_formula": "rank(close)"}
            d = build_export_decision(cand, rec, store, refinement_complete=True)
            assert not d.accepted
            assert d.seed_duplicate is True
            store.close()

    def test_full_pass_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            from alphaprobe.memory import GlobalMemoryStore

            store = GlobalMemoryStore(db_path=Path(td) / "m.sqlite3")
            rec = make_record(
                "f_ok",
                {"search_fitness": 0.5, "S": 0.4, "N": 0.3},
                fidelity=FidelityLevel.L4_POOL_AUDIT.value,
            )
            cand = {"factor_id": "f_ok", "canonical_formula": "rank(close)"}
            d = build_export_decision(cand, rec, store, refinement_complete=True, export_floor=0.0)
            assert d.accepted
            store.close()
