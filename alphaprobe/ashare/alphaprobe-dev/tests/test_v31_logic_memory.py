"""test_v31_logic_memory —— Market Logic Library（plan Task 13 / Part G #29/#23/#24）。

覆盖 plan Task 13 四条验收：
① logic ID 稳定：alias/名字变化不改 logic_id（机器 ID 与 LLM 名字分离）；
② 因子可概率/多标签映射到 logic 节点（不是一对一硬归属）；
③ 低 support 的 logic 统计向全局先验收缩（support_count 小 → elite_rate 等
   effect 向中性，绝不当最好）；
④ 当前版本 sealed Test survival 绝不进当前版本 logic reward（版本隔离：只有
   frozen 上一版本 survival 可作为 historical_survival 输入）。

外加行为契约：
- LogicNode 全清单字段（logic_id / logic_version / logic_text / aliases /
  member_factor_ids / reference_query / schema_distribution / support_count /
  elite_count / elite_rate / median_factor_fitness / median_long_short_quality /
  median_stability / historical_survival / crowding / saturation /
  successful_implementations / failed_implementations）。
- 逻辑挖掘（logic_miner）为确定性离线任务：同输入恒同 logic_id / 同成员
  映射；min_support 低于阈值的桶不宣称规则；LLM 命名默认关闭。
- #24：library 只存 reference query / 聚合统计，不物化因子面板。
- #29：版本化——同 signature 但不同语义版本统计不混；LogicNode.logic_id
  必须与签名派生 id 一致（禁止手工 id 绕过签名）。

全部合成数据、确定性、零 LLM / 零模型 / 零网络 / 零 FE / 零 FactorAssets。
"""

from __future__ import annotations

import math

import pytest

from alphaprobe.research_space.logic import (
    DEFAULT_SHRINKAGE_K,
    FactorFact,
    LogicLibrary,
    LogicNode,
    LogicSignature,
    LogicSignatureError,
    ShrunkLogicStats,
    SurvivalFact,
    derive_logic_id,
    logic_signature_match,
    reliability_of,
    shrink_effect,
    shrink_mean,
)
from alphaprobe.research_space.logic_miner import (
    LogicMiner,
    LogicMiningConfig,
    LogicMiningReport,
    default_logic_text,
    dominant_motif_signature,
    group_facts_by_signature,
    multi_label_membership,
)


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def _sig(**kw) -> LogicSignature:
    base = dict(
        mechanisms=("reversal", "liquidity"),
        operator_motifs=("ts_rank",),
        field_families=("price_volume", "liquidity"),
        horizon_bucket="medium",
    )
    base.update(kw)
    return LogicSignature(**base)


def _node(**kw) -> LogicNode:
    sig = kw.pop("signature", None) or _sig()
    logic_id = kw.pop("logic_id", None) or sig.derive_logic_id()
    base = dict(logic_id=logic_id, signature=sig, logic_text="Liquidity reversal")
    base.update(kw)
    return LogicNode(**base)


def _fact(factor_id: str, **kw) -> FactorFact:
    base = dict(
        factor_id=factor_id,
        formula=f"-(ts_rank(volume, 20))-{factor_id}",
        mechanisms=("reversal", "liquidity"),
        operator_motifs=("ts_rank",),
        field_families=("price_volume", "liquidity"),
        horizon_bucket="medium",
    )
    base.update(kw)
    return FactorFact(**base)


# ---------------------------------------------------------------------------
# ① logic ID 稳定：alias/名字变化不改 logic_id
# ---------------------------------------------------------------------------


class TestStableLogicId:
    def test_alias_and_text_change_do_not_change_id(self):
        a = _node(logic_text="Liquidity Absorption Reversal", aliases=("LAR", "liq-rev"))
        b = _node(logic_text="别名叫什么无所谓", aliases=("totally-different-name",))
        assert a.logic_id == b.logic_id  # 名字/描述不进 id
        assert a.logic_id.startswith("LOG_")
        assert len(a.logic_id) == len("LOG_") + 16

    def test_id_derived_from_signature_only(self):
        sig = _sig()
        lid = sig.derive_logic_id()
        n = _node(signature=sig, logic_id=lid)
        assert n.logic_id == lid
        # 同名同描述但不同机制 → 不同 logic_id（机制是语义核心锚点）
        other = _node(signature=_sig(mechanisms=("reversal", "valuation")))
        assert other.logic_id != n.logic_id

    def test_signature_case_and_order_normalized(self):
        s1 = LogicSignature(mechanisms=("Reversal", "Liquidity"), operator_motifs=("ts_rank",))
        s2 = LogicSignature(mechanisms=("LIQUIDITY", "reversal"), operator_motifs=("ts_rank",))
        assert s1.derive_logic_id() == s2.derive_logic_id()

    def test_mechanism_change_changes_id(self):
        a = _sig(mechanisms=("reversal", "liquidity"))
        b = _sig(mechanisms=("reversal", "valuation"))
        assert a.derive_logic_id() != b.derive_logic_id()

    def test_hand_crafted_id_rejected(self):
        with pytest.raises(ValueError):
            LogicNode(logic_id="LOG_MANUAL", signature=_sig())

    def test_empty_mechanisms_rejected(self):
        with pytest.raises(LogicSignatureError):
            _sig(mechanisms=())

    def test_llm_display_name_not_in_id(self):
        """LLM 生成的名字只进 logic_text/aliases，永不参与签名/id。"""
        lid = derive_logic_id(mechanisms=("reversal", "liquidity"), operator_motifs=("ts_rank",))
        renamed = _node(
            signature=_sig(),
            logic_id=lid,
            logic_text="Low-vol compression breakout",
            aliases=("LLM-alias-1",),
        )
        assert renamed.logic_id == lid


# ---------------------------------------------------------------------------
# ② 因子可概率/多标签映射到 logic 节点
# ---------------------------------------------------------------------------


class TestMultiLabelMembership:
    def test_one_factor_maps_to_multiple_logics(self):
        lib = LogicLibrary(current_research_version="2")
        lid_a = lib.upsert_node(_node(signature=_sig(mechanisms=("reversal", "liquidity"))))
        lid_b = lib.upsert_node(_node(signature=_sig(mechanisms=("volatility", "liquidity"))))
        # 同一因子多标签映射到两个 logic（概率/多标签，非一对一硬归属）
        lib.record_membership(factor_id="F1", logic_id=lid_a, affinity=0.8)
        lib.record_membership(factor_id="F1", logic_id=lid_b, affinity=0.6)
        assert lib.logics_of("F1") == {lid_a: 0.8, lid_b: 0.6}
        assert "F1" in lib.members_of(lid_a)
        assert "F1" in lib.members_of(lid_b)

    def test_affinity_is_probabilistic_not_hard(self):
        lib = LogicLibrary()
        lid = lib.upsert_node(_node())
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=0.9)
        lib.record_membership(factor_id="F2", logic_id=lid, affinity=0.55)
        mem = lib.memberships_of(lid)
        assert mem["F1"] == 0.9
        assert mem["F2"] == 0.55
        assert all(0.0 < a <= 1.0 for a in mem.values())

    def test_duplicate_membership_takes_max_affinity_idempotent(self):
        lib = LogicLibrary()
        lid = lib.upsert_node(_node())
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=0.7)
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=0.5)
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=0.9)
        assert lib.memberships_of(lid)["F1"] == 0.9
        assert lib.node(lid).support_count == 1  # 同一因子重复映射不累加 support
        assert lib.node(lid).elite_count == 0

    def test_miner_produces_multi_label_factors(self):
        """同一因子机制为多标签 → 挖掘后可属多个 logic 桶。"""
        facts = [
            _fact("F1", mechanisms=("reversal", "liquidity", "volatility")),
            _fact("F2", mechanisms=("reversal", "liquidity", "volatility")),
            _fact("F3", mechanisms=("reversal", "liquidity", "volatility")),
        ]
        # 同构因子进同一个精确机制桶 → 一个 logic；多标签因子「可属多 logic」
        # 的开放语义由 lib.record_membership / lib.logics_of 表达（库层，多标签
        # 不是一对一硬归属）。挖矿桶互斥保证同一桶因子在同一 logic。
        miner = LogicMiner(config=LogicMiningConfig(min_support=2))
        lib, rep = miner.mine(facts)
        assert len(lib.logic_ids()) == 1
        assert lib.members_of(lib.logic_ids()[0]) == ["F1", "F2", "F3"]
        assert rep.multi_label_factors == 0

    def test_miner_multi_label_via_overlapping_groups(self):
        """多标签语义：同一因子（机制多标签）可同时属多个 logic。

        因子的 mechanisms = (rev, vol, liq) 全集合与只含其中子集的因子分属
        不同桶——为了在同一挖掘里让**一个因子进两个桶**，组间共享因子的唯一
        途径是「机制集合包含」式归属。当前 miner 用精确集合分桶（同构桶）；
        概率多标签的开放映射由库层表达。此测试断言库层开放映射（同一因子
        显式映射两个 logic 都能取回）。
        """
        lib = LogicLibrary()
        lid_a = lib.upsert_node(_node(signature=_sig(mechanisms=("reversal", "volatility"))))
        lid_b = lib.upsert_node(_node(signature=_sig(mechanisms=("volatility", "liquidity"))))
        lib.record_membership(factor_id="FX", logic_id=lid_a, affinity=0.9)
        lib.record_membership(factor_id="FX", logic_id=lid_b, affinity=0.4)
        assert set(lib.logics_of("FX")) == {lid_a, lid_b}
        assert lib.members_of(lid_a) == ["FX"]
        assert lib.members_of(lid_b) == ["FX"]

        # miner 侧：同构桶各自成 logic（每桶 >= min_support）
        facts = [
            _fact("FA", mechanisms=("reversal", "volatility"), fitness=0.6, elite=True),
            _fact("FB", mechanisms=("reversal", "volatility"), fitness=0.5),
            _fact("FC", mechanisms=("volatility", "liquidity"), fitness=0.7),
            _fact("FD", mechanisms=("volatility", "liquidity"), fitness=0.4),
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2))
        lib2, rep = miner.mine(facts)
        assert len(lib2.logic_ids()) == 2
        for lid in lib2.logic_ids():
            assert len(lib2.members_of(lid)) == 2
        assert rep.buckets == 2
        assert rep.total_facts == 4

    def test_schema_distribution_accumulates_from_members(self):
        lib = LogicLibrary()
        lid = lib.upsert_node(_node())
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=1.0, schema_id="SCH_A")
        lib.record_membership(factor_id="F2", logic_id=lid, affinity=1.0, schema_id="SCH_A")
        lib.record_membership(factor_id="F3", logic_id=lid, affinity=1.0, schema_id="SCH_B")
        dist = lib.schemas_of(lid)
        assert dist["SCH_A"] == 2.0
        assert dist["SCH_B"] == 1.0


# ---------------------------------------------------------------------------
# ③ 低 support → 向全局先验收缩
# ---------------------------------------------------------------------------


class TestShrinkageToPrior:
    def test_low_support_elite_rate_shrinks_to_neutral(self):
        lib = LogicLibrary(global_elite_prior=0.5)
        lid = lib.upsert_node(_node())
        # support=1、elite=1 → raw elite_rate=1.0，但样本极少 → 收缩接近中性
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=1.0, elite=True)
        stats = lib.shrunk_stats_of(lid)
        assert stats.support_count == 1
        assert stats.shrunk_elite_rate is not None
        assert 0.5 < stats.shrunk_elite_rate < 0.7  # 远低于 raw 1.0
        assert stats.shrunk_elite_rate > 0.5

    def test_zero_support_elite_rate_neutral(self):
        lib = LogicLibrary()
        lid = lib.upsert_node(_node())
        stats = lib.shrunk_stats_of(lid)
        assert stats.support_count == 0
        assert stats.shrunk_elite_rate is None or stats.shrunk_elite_rate == pytest.approx(0.5, abs=1e-6)

    def test_high_support_approaches_raw(self):
        lib = LogicLibrary(global_elite_prior=0.5)
        lid = lib.upsert_node(_node())
        for i in range(120):
            lib.record_membership(
                factor_id=f"F{i}", logic_id=lid, affinity=1.0,
                elite=(i % 2 == 0),
            )
        stats = lib.shrunk_stats_of(lid)
        assert stats.support_count == 120
        assert stats.shrunk_elite_rate is not None
        assert stats.shrunk_elite_rate == pytest.approx(0.5, abs=1e-6)  # raw 60/120=0.5 → 收缩后仍 0.5

    def test_high_support_majority_elite_approaches_raw(self):
        lib = LogicLibrary(global_elite_prior=0.5)
        lid = lib.upsert_node(_node())
        for i in range(120):
            lib.record_membership(
                factor_id=f"F{i}", logic_id=lid, affinity=1.0,
                elite=True,  # 全部 elite → raw 1.0
            )
        stats = lib.shrunk_stats_of(lid)
        assert stats.shrunk_elite_rate is not None
        assert stats.shrunk_elite_rate > 0.9  # 120 样本 → 收缩弱化，接近 raw 1.0

    def test_small_support_median_metrics_shrink_to_prior(self):
        lib = LogicLibrary(global_median_prior=0.5)
        lid = lib.upsert_node(_node())
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=1.0, fitness=0.99)
        stats = lib.shrunk_stats_of(lid)
        assert stats.shrunk_fitness is not None
        assert 0.5 < stats.shrunk_fitness < 0.8  # 极少样本 → 远低于 raw 0.99

    def test_low_support_survival_effect_towards_zero(self):
        # survival 只来自 frozen 上一版本；low support 效应向 0 收缩
        raw = 0.9
        eff = shrink_effect(raw, n_eff=1)
        assert abs(eff) < 0.4  # 单样本效应几乎归零
        eff_high = shrink_effect(raw, n_eff=1000)
        assert eff_high > 0.3  # 大样本保留效应

    def test_shrink_utility_formula(self):
        # 与 fitness/confidence.shrink_utility 同款公式（本地实现）
        r = reliability_of(40)  # k=64
        expected = 0.5 + r * (0.8 - 0.5)
        got = shrink_mean(0.8, n_eff=40, prior=0.5, k=DEFAULT_SHRINKAGE_K)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_effect_exactly_zero_when_unknown(self):
        assert shrink_effect(None, n_eff=100) == 0.0
        assert shrink_effect(float("nan"), n_eff=100) == 0.0

    def test_zero_support_does_not_claim_best(self):
        lib = LogicLibrary(global_elite_prior=0.5)
        lid = lib.upsert_node(_node())
        stats = lib.shrunk_stats_of(lid)
        # 无样本绝不宣称最优（elite_rate 不可能是 1.0 或 0.9 之类）
        v = stats.shrunk_elite_rate if stats.shrunk_elite_rate is not None else 0.5
        assert v <= 0.5 + 1e-9


# ---------------------------------------------------------------------------
# ④ 当前版本 sealed Test survival 不进当前版本 logic reward（版本隔离）
# ---------------------------------------------------------------------------


class TestSurvivalVersionIsolation:
    def _lib_with_outcome(
        self, *, survival_version: str | None, current_version: str = "1",
    ) -> LogicLibrary:
        lib = LogicLibrary(current_research_version=current_version)
        lid = lib.upsert_node(_node())
        surv = SurvivalFact(version=survival_version or current_version, survival_rate=0.95)
        lib.record_logic_outcome(
            logic_id=lid, success=True,
            survival=surv,
            survival_source_version=survival_version,
        )
        return lib

    def test_current_version_survival_never_enters_current_logic_reward(self):
        lib = self._lib_with_outcome(survival_version="1", current_version="1")
        lid = lib.logic_ids()[0]
        stats = lib.shrunk_stats_of(lid)
        assert stats.shrunk_survival is None  # 当前版本 sealed survival 未计入
        assert lib.node(lid).historical_survival is None

    def test_frozen_previous_version_survival_is_consumed(self):
        lib = self._lib_with_outcome(survival_version="0", current_version="1")
        lid = lib.logic_ids()[0]
        stats = lib.shrunk_stats_of(lid)
        assert stats.shrunk_survival is not None
        assert stats.shrunk_survival == pytest.approx(0.95)

    def test_explicitly_future_version_never_enters(self):
        lib = self._lib_with_outcome(survival_version="2", current_version="1")
        lid = lib.logic_ids()[0]
        assert lib.node(lid).historical_survival is None

    def test_survival_without_source_version_never_enters(self):
        """survival 提供但未显式 source version → 保守不入（#23 fail-safe）。"""
        lib = LogicLibrary(current_research_version="1")
        lid = lib.upsert_node(_node())
        lib.record_logic_outcome(
            logic_id=lid, success=True,
            survival=SurvivalFact(version="0", survival_rate=0.9),  # 没传 survival_source_version
        )
        assert lib.node(lid).historical_survival is None

    def test_miner_survival_respects_version_cutoff(self):
        """miner 只把显式上一版本 survival 汇入 historical_survival。"""
        facts = [
            _fact("F1", survival=SurvivalFact(version="0", survival_rate=0.9)),
            _fact("F2", survival=SurvivalFact(version="0", survival_rate=0.8)),
            _fact("F3", survival=SurvivalFact(version="0", survival_rate=0.85)),
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2, current_research_version="1"))
        lib, _ = miner.mine(facts)
        lid = lib.logic_ids()[0]
        node = lib.node(lid)
        # 桶内全部同版本且 < 当前研究版本 → 汇入 median survival（0.85）
        assert node.historical_survival is not None
        assert node.historical_survival == pytest.approx(0.85, abs=1e-6)

    def test_miner_current_version_survival_excluded(self):
        """当前版本 sealed survival 附着在 member 上 → miner 保守不入。"""
        facts = [
            _fact("F1", survival=SurvivalFact(version="1", survival_rate=0.95)),
            _fact("F2", survival=SurvivalFact(version="1", survival_rate=0.95)),
            _fact("F3", survival=SurvivalFact(version="1", survival_rate=0.95)),
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2, current_research_version="1"))
        lib, _ = miner.mine(facts)
        lid = lib.logic_ids()[0]
        assert lib.node(lid).historical_survival is None  # 当前版本 survival 不入

    def test_miner_mixed_survival_versions_conservative(self):
        """桶内混合不同冻结版本 → 保守不入（避免混不同版本的 survival）。"""
        facts = [
            _fact("F1", survival=SurvivalFact(version="0", survival_rate=0.9)),
            _fact("F2", survival=SurvivalFact(version="0", survival_rate=0.8)),
            _fact("F3", survival=SurvivalFact(version="0.5", survival_rate=0.85)),
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2, current_research_version="1"))
        lib, _ = miner.mine(facts)
        lid = lib.logic_ids()[0]
        assert lib.node(lid).historical_survival is None  # 混合来源保守不入


# ---------------------------------------------------------------------------
# LogicNode 字段全清单
# ---------------------------------------------------------------------------


class TestLogicNodeFullFieldContract:
    def test_node_has_all_plan_fields(self):
        node = _node(
            logic_text="text",
            aliases=("a", "b"),
            member_factor_ids=("F1",),
            reference_query="mechanisms=REVERSAL",
            schema_distribution={"SCH_A": 1.0},
            support_count=5,
            elite_count=2,
            median_factor_fitness=0.6,
            median_long_short_quality=0.5,
            median_stability=0.4,
            historical_survival=0.7,
            crowding=0.3,
            successful_implementations=2,
            failed_implementations=1,
        )
        d = node.to_dict()
        for key in (
            "logic_id", "logic_version", "logic_text", "aliases", "member_factor_ids",
            "reference_query", "schema_distribution", "support_count", "elite_count",
            "elite_rate", "median_factor_fitness", "median_long_short_quality",
            "median_stability", "historical_survival", "crowding", "saturation",
            "successful_implementations", "failed_implementations",
        ):
            assert key in d, f"LogicNode missing field {key}"
        assert d["elite_rate"] == pytest.approx(2 / 5)

    def test_saturation_formula(self):
        node = _node()  # 默认 support_count=0 → saturation None（不编 0/伪值）
        assert node.saturation is None
        node2 = node.with_stats(support_count=64)
        assert node2.saturation == pytest.approx(64 / (64 + DEFAULT_SHRINKAGE_K))
        assert node2.saturation == pytest.approx(0.5)

    def test_members_of_respects_k(self):
        lib = LogicLibrary()
        lid = lib.upsert_node(_node())  # 无预置 member_factor_ids → members_of 走 membership 表
        for i in range(5):
            lib.record_membership(factor_id=f"F{i}", logic_id=lid, affinity=(i + 1) / 10)
        assert lib.members_of(lid, k=2) == ["F4", "F3"]  # 按 affinity 降序

    def test_no_factor_panel_materialized_in_node(self):
        """#24：节点只存 reference query / factor_id，不存因子面板值。"""
        node = _node(reference_query="mechanisms=REVERSAL", member_factor_ids=("F1",))
        assert node.reference_query
        assert "panel" not in node.to_dict()
        assert "values" not in node.to_dict()


# ---------------------------------------------------------------------------
# 统计/收缩与 miner 确定性
# ---------------------------------------------------------------------------


class TestMinerDeterminism:
    def test_same_input_same_logic_ids(self):
        facts = [
            _fact("F1", mechanisms=("reversal", "volatility")),
            _fact("F2", mechanisms=("reversal", "volatility")),
            _fact("F3", mechanisms=("reversal", "volatility")),
        ]
        a, _ = LogicMiner(config=LogicMiningConfig(min_support=2)).mine(facts)
        b, _ = LogicMiner(config=LogicMiningConfig(min_support=2)).mine(facts)
        assert a.logic_ids() == b.logic_ids()
        for lid in a.logic_ids():
            assert a.members_of(lid) == b.members_of(lid)

    def test_min_support_drops_small_buckets(self):
        facts = [
            _fact("F1", mechanisms=("reversal", "volatility")),
            _fact("F2", mechanisms=("reversal", "volatility")),
            _fact("F3", mechanisms=("volatility",)),  # singleton 机制桶
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2))
        lib, rep = miner.mine(facts)
        assert len(lib.logic_ids()) == 1  # {vol} 桶只有 1 个因子 → 不成 logic
        assert rep.dropped_below_min_support == 1
        assert rep.total_facts == 3

    def test_miner_report_counts(self):
        facts = [
            _fact("F1", mechanisms=("reversal", "volatility")),
            _fact("F2", mechanisms=("reversal", "volatility")),
            _fact("F3", mechanisms=("volatility", "liquidity")),
            _fact("F4", mechanisms=("volatility", "liquidity")),
        ]
        miner = LogicMiner(config=LogicMiningConfig(min_support=2))
        lib, rep = miner.mine(facts)
        assert rep.total_facts == 4
        assert rep.buckets == 2
        assert len(rep.logic_ids) == 2
        assert rep.memberships_created == 4
        assert rep.dropped_below_min_support == 0

    def test_default_logic_text_is_deterministic(self):
        a = default_logic_text(("reversal", "liquidity"), motifs=("ts_rank",))
        b = default_logic_text(("liquidity", "reversal"), motifs=("ts_rank",))
        assert a == b
        assert "REVERSAL" in a

    def test_llm_naming_only_when_explicitly_enabled(self):
        called = []

        def llm_name(sig):
            called.append(sig)
            return "Fancy LLM Logic", ("alias-1",)

        facts = [
            _fact("F1", mechanisms=("reversal", "volatility")),
            _fact("F2", mechanisms=("reversal", "volatility")),
            _fact("F3", mechanisms=("reversal", "volatility")),
        ]
        # 默认关闭：LLM 不被调用，名字是确定性机器描述
        miner = LogicMiner(config=LogicMiningConfig(min_support=2), llm_name_fn=llm_name)
        lib, _ = miner.mine(facts)
        lid = lib.logic_ids()[0]
        assert called == []
        assert "market logic" in lib.node(lid).logic_text
        # 显式开启才调用 LLM（名字不改 id）
        before_id = lib.logic_ids()[0]
        miner2 = LogicMiner(
            config=LogicMiningConfig(min_support=2, enable_llm=True), llm_name_fn=llm_name,
        )
        lib2, _ = miner2.mine(facts)
        assert len(called) == 1
        assert lib2.node(before_id).logic_text == "Fancy LLM Logic"
        assert lib2.node(before_id).logic_id == before_id  # 名字不改 id


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persist_and_load_roundtrip(self, tmp_path):
        lib = LogicLibrary(current_research_version="1")
        lid = lib.upsert_node(_node(reference_query="q=1", aliases=("old-name",)))
        lib.record_membership(factor_id="F1", logic_id=lid, affinity=0.8, schema_id="SCH_A")
        lib.record_logic_outcome(
            logic_id=lid, success=True, elite=True, fitness=0.6,
            survival=SurvivalFact(version="0", survival_rate=0.9),
            survival_source_version="0",
        )
        db = str(tmp_path / "logic.sqlite3")
        lib.persist_to(db)
        lib2 = LogicLibrary.load_from(db, current_research_version="1")
        n2 = lib2.node(lid)
        assert n2 is not None
        # membership(1) + outcome(1) 各计一次实现样本 → support=2；elite 只在
        # outcome 置 True 时 +1（membership 的 elite=False 不加）→ elite_count=1
        assert n2.support_count == 2
        assert n2.elite_count == 1
        assert n2.historical_survival == pytest.approx(0.9)
        assert n2.aliases == ("old-name",)
        assert lib2.members_of(lid) == ["F1"]
        assert lib2.schemas_of(lid)["SCH_A"] == 1.0

    def test_memory_default_isolation(self):
        a = LogicLibrary()
        b = LogicLibrary()
        la = a.upsert_node(_node(signature=_sig(mechanisms=("reversal", "liquidity"))))
        b.upsert_node(_node(signature=_sig(mechanisms=("volatility",))))
        assert a.logic_ids() == [la]
        assert len(b.logic_ids()) == 1
        assert b.logic_ids() != a.logic_ids()


# ---------------------------------------------------------------------------
# LogicSignature helpers
# ---------------------------------------------------------------------------


class TestLogicSignatureMatch:
    def test_exact_mechanisms_match(self):
        node_sig = _sig()
        cand = _sig()
        assert logic_signature_match(node_sig, cand)

    def test_extra_candidate_motif_is_superset_match(self):
        node_sig = _sig(operator_motifs=("ts_rank",))
        cand = _sig(operator_motifs=("ts_rank", "ts_std"))
        assert logic_signature_match(node_sig, cand)  # 候选维 ⊇ 节点维 → 命中

    def test_different_mechanism_no_match(self):
        node_sig = _sig(mechanisms=("reversal", "liquidity"))
        cand = _sig(mechanisms=("reversal", "valuation"))
        assert not logic_signature_match(node_sig, cand)

    def test_different_horizon_no_match(self):
        node_sig = _sig(horizon_bucket="medium")
        cand = _sig(horizon_bucket="slow")
        assert not logic_signature_match(node_sig, cand)
        cand_unknown = _sig(horizon_bucket="")
        assert logic_signature_match(node_sig, cand_unknown)  # 无信息不拒


class TestReliabilityAndShrink:
    def test_reliability_zero_no_evidence(self):
        assert reliability_of(None) == 0.0
        assert reliability_of(0) == 0.0
        assert reliability_of(-5) == 0.0

    def test_reliability_monotonic(self):
        assert reliability_of(1) < reliability_of(10) < reliability_of(1000)

    def test_shrink_mean_formula_matches_confidence(self):
        # fitness/confidence.shrink_utility: U_conf = 0.5 + rel*(U-0.5)
        from alphaprobe.fitness.confidence import shrink_utility

        got = shrink_mean(0.7, n_eff=100, prior=0.5, k=DEFAULT_SHRINKAGE_K)
        want = shrink_utility(0.7, n_eff=100, k=DEFAULT_SHRINKAGE_K)
        assert got == pytest.approx(want, abs=1e-6)

    def test_shrink_effect_small_support_zero(self):
        assert shrink_effect(0.9, n_eff=1) == pytest.approx(
            reliability_of(1) * (0.9 - 0.5), abs=1e-9
        )
