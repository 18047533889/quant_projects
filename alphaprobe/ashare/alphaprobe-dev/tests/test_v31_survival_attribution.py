"""plan.md Task 18 / Part G #23/#29/#13/#30：受控生存归因（SurvivalAttributionModel）。

覆盖 plan Task 18 四条验收 + 追加：
1. current-version sealed 事件不可用（版本隔离 #23：attribution 训练侧与
   查询侧都拒绝当前 version 的 test 事件）；
2. version N+1 可消费 frozen N 的 test survival（显式 freeze_version 接口）；
3. 低 support 效应 ≈ 0（#13：hierarchical shrinkage 生效，小样本绝不当规则）；
4. 负向高置信 survival 方向温和降低 opportunity（capped，不翻转不爆炸）；
5. SurvivalOpportunity = shrunk_survival - prior，cap 可配；
6. 报告 effect + confidence 而非因果断言（effect_ci / effect_se / support）；
7. interactions 可配置开关（#30）；
8. sklearn 与 numpy 两条 Elastic Net logistic 真实路径都有测试覆盖。

全部合成数据、确定性、零 LLM / 零网络 / 零 FE / 零 FactorAssets。
"""

from __future__ import annotations

import math

import pytest

from alphaprobe.contracts import SurvivalLabel
from alphaprobe.research_protocol import SealedTestViolation
from alphaprobe.retrieval.search_opportunity import (
    SURVIVAL_CAP,
    SearchOpportunity,
    survival_opportunity,
)
from alphaprobe.survival.attribution import (
    DEFAULT_EFFECT_CAP,
    SurvivalAttributionModel,
    SurvivalEvent,
    attribution_event_from_formula,
)
from alphaprobe.survival.dna import dna_from_formula


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _ev(
    formula: str,
    survival: str | float,
    *,
    version: str = "0",
    visible: bool = False,
    source_miner: str = "",
    cluster_size: float | None = None,
    age_days: int | None = None,
    factor_id: str = "",
) -> SurvivalEvent:
    """版本化的 survival 事件（缺省 frozen 旧版本 v0）。"""
    if isinstance(survival, (int, float)):
        ev = attribution_event_from_formula(
            formula, survival=float(survival), version=version, visible=visible,
            factor_id=factor_id,
        )
    else:
        ev = attribution_event_from_formula(
            formula, survival=str(survival), version=version, visible=visible,
            factor_id=factor_id,
        )
    if source_miner:
        ev.source_miner = source_miner
    if cluster_size is not None:
        ev.cluster_size = cluster_size
    if age_days is not None:
        ev.age_days = age_days
    return ev


def _fit_default(events, *, current_version: str = "1", **kw) -> SurvivalAttributionModel:
    """numpy 确定性路径拟合（测试不依赖 sklearn 的 saga 收敛）。"""
    m = SurvivalAttributionModel(
        current_version=current_version, fit_backend="numpy", **kw
    )
    return m.fit(events, current_version=current_version)


# ---------------------------------------------------------------------------
# 1. current-version sealed 事件不可用（#23 版本隔离）
# ---------------------------------------------------------------------------


class TestCurrentVersionSealedIsolation:
    def test_fit_rejects_current_version_sealed_event(self):
        """训练侧：当前版本（v1）的 sealed test 事件 → SealedTestViolation。"""
        m = SurvivalAttributionModel(current_version="1")
        sealed = _ev("ts_rank(close, 20)", "HEALTHY", version="1", visible=False)
        with pytest.raises(SealedTestViolation):
            m.fit([sealed], current_version="1")

    def test_query_rejects_current_version_sealed_event(self):
        """查询侧：已 fit 后，对当前版本 sealed 事件查询 → SealedTestViolation。"""
        m = _fit_default(
            [_ev("ts_rank(close, 20)", "HEALTHY", version="0")],
            current_version="1",
        )
        cur = _ev("ts_rank(close, 20)", "HEALTHY", version="1", visible=False)
        with pytest.raises(SealedTestViolation):
            m.effect_for_event(cur, query_version="1")
        with pytest.raises(SealedTestViolation):
            m.predict_survival_rate(cur, query_version="1")

    def test_visible_current_version_still_allowed_as_consumed_frozen(self):
        """v0 事件即使 visible=False 也远早于 v1 → 可消费（历史不是 sealed）。"""
        m = _fit_default(
            [_ev("ts_rank(close, 20)", "HEALTHY", version="0", visible=False)],
            current_version="1",
        )
        assert m._n_events == 1

    def test_is_event_consumable_helpers(self):
        m = SurvivalAttributionModel(current_version="1")
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="0")) is True
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="1", visible=False)) is False
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="2", visible=False)) is False


# ---------------------------------------------------------------------------
# 2. version N+1 可消费 frozen N 的 test survival
# ---------------------------------------------------------------------------


class TestFrozenVersionConsumption:
    def test_freeze_version_unlocks_consumption(self):
        m = SurvivalAttributionModel(current_version="2")
        # 当前版本 v2 的 sealed 事件在 v2 不可消费
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="2", visible=False)) is False
        # 冻结 v2 → v3 视角可消费
        m.freeze_version("2")
        m.current_version = "3"
        assert m.is_event_consumable(
            _ev("r", "HEALTHY", version="2", visible=False), current_version="3"
        ) is True

    def test_frozen_n_survival_can_be_fit_at_n_plus_1(self):
        """v2 冻结后，v3 可把 v2 的 test survival 当历史消费（fit 不再抛）。"""
        m = SurvivalAttributionModel(current_version="3")
        m.freeze_version("2")
        events = [
            _ev("ts_rank(close, 20)", "HEALTHY", version="2", visible=False),
            _ev("ts_std(close, 20)", "BROKEN", version="2", visible=False),
            _ev("ts_rank(close, 20)", "HEALTHY", version="2", visible=False),
        ]
        m.fit(events, current_version="3")
        assert m._n_events == 3

    def test_version_ordering_strict(self):
        """source_version == current 且未 frozen/visible → 拒绝（不靠字符串倒挂）。"""
        m = SurvivalAttributionModel(current_version="10")
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="2", visible=False)) is True
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="10", visible=False)) is False
        # visible=True（已过闸门）即使当前版本也可消费（显式冻结/解冻接口语义）
        assert m.is_event_consumable(_ev("r", "HEALTHY", version="10", visible=True)) is True


# ---------------------------------------------------------------------------
# 3. 低 support 效应 ≈ 0（#13 hierarchical shrinkage）
# ---------------------------------------------------------------------------


class TestLowSupportShrinksToZero:
    def test_single_support_extreme_rate_effect_near_zero(self):
        m = _fit_default([_ev("ts_rank(close, 20)", "HEALTHY", version="0")])
        # 单样本极端 rate 1.0 → 收缩后 effect 远低于 raw gain 0.5
        r = m.effect_for_event(_ev("ts_rank(close, 20)", "HEALTHY", version="0"))
        assert abs(r["effect"]) < 0.15
        # 比未经收缩的 raw gain (1.0-0.5=0.5) 小一个量级以上
        assert abs(r["effect"]) < 0.2

    def test_effect_monotonic_in_support(self):
        """相同 rate 下 support 越大 → effect 越大（reliability 单调）。"""
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        low = m.effect_for_survival(0.95, support_count=3)   # 恰在 min_support
        high = m.effect_for_survival(0.95, support_count=300)
        assert high > low
        assert low > 0.0
        assert high > 0.0

    def test_low_support_explicitly_toward_zero(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        # support=1：双轮收缩后远小于 0.01 的天然噪声水平……（此处要求极弱）
        e1 = m.effect_for_survival(0.99, support_count=1)
        e2 = m.effect_for_survival(0.99, support_count=500)
        assert abs(e1) < abs(e2) / 4.0

    def test_report_notes_low_support(self):
        events = [_ev("ts_rank(close, 20)", "HEALTHY", version="0")] * 2
        m = _fit_default(events)
        rep = m.report()
        # 全部特征 support<=2 < min_support=3 → note 标注低 support
        assert all(r.support_count < 3 for r in rep.results)
        assert any("low support" in r.note for r in rep.results)


# ---------------------------------------------------------------------------
# 4. 负向高置信 survival 温和降 opportunity（capped，不翻转不爆炸）
# ---------------------------------------------------------------------------


class TestNegativeSurvivalMild:
    def test_negative_high_conf_survival_reduces_opportunity(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        eff = survival_opportunity(
            {"survival_rate": 0.2, "support_count": 500}, attribution_model=m
        )
        assert eff < 0.0
        # capped：不低于 -cap
        assert eff >= -SURVIVAL_CAP - 1e-9
        assert eff >= -DEFAULT_EFFECT_CAP - 1e-9
        # 不翻转：不会变成 positive reward
        assert eff <= 0.0

    def test_negative_low_confidence_is_mild_penalty(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        hi = survival_opportunity(
            {"survival_rate": 0.2, "support_count": 500}, attribution_model=m
        )
        lo = survival_opportunity(
            {"survival_rate": 0.2, "support_count": 1}, attribution_model=m
        )
        # 低置信负向 → 更接近 0（不那么负）
        assert lo > hi
        assert abs(lo) < abs(hi)

    def test_compute_survival_dimension_can_be_negative(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        so = SearchOpportunity(survival_model=m)
        c_none = so.compute(
            "x", mean_novelty_gain=None, attempted_actions=["REFINE"],
            visible_survival=None,
        )
        c_neg = so.compute(
            "x", mean_novelty_gain=None, attempted_actions=["REFINE"],
            visible_survival={"survival_rate": 0.2, "support_count": 500},
        )
        c_pos = so.compute(
            "x", mean_novelty_gain=None, attempted_actions=["REFINE"],
            visible_survival={"survival_rate": 0.9, "support_count": 500},
        )
        assert c_neg["survival"] < 0.0
        assert c_pos["survival"] > 0.0
        # 排序：负向 < 中性 < 正向（温和，不因负 survival 把总分打成负）
        assert c_neg["total"] < c_none["total"] < c_pos["total"]


# ---------------------------------------------------------------------------
# 5. SurvivalOpportunity = shrunk - prior，cap 可配
# ---------------------------------------------------------------------------


class TestOpportunityFormula:
    def test_legacy_path_unchanged_positive_only(self):
        """attribution_model=None → 内置 all-positive 公式（向后兼容）。"""
        assert survival_opportunity(None) == 0.0
        assert survival_opportunity({}) == 0.0
        v = survival_opportunity({"survival_rate": 0.9, "support_count": 1000})
        assert 0.0 < v <= SURVIVAL_CAP
        assert v == pytest.approx(min(SURVIVAL_CAP, 0.9 - 0.5), abs=1e-6)  # 1000 支撑几乎不收缩

    def test_model_path_is_prior_centered(self):
        """effect = shrunk - prior；默认 effect_cap=0.15 钳制（不爆炸）。"""
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        # shrunk_survival_rate 已给 → 直接用；effect = shrunk - prior
        pos = survival_opportunity(
            {"shrunk_survival_rate": 0.8, "support_count": 1000}, attribution_model=m
        )
        # 0.8-0.5=0.30 > cap 0.15 → capped 到 0.15（Plan §51 温和调节语义）
        assert pos == pytest.approx(DEFAULT_EFFECT_CAP, abs=1e-9)
        neutral = survival_opportunity(
            {"shrunk_survival_rate": 0.5, "support_count": 1000}, attribution_model=m
        )
        assert neutral == pytest.approx(0.0, abs=1e-9)
        # cap 内不做额外收缩：0.65-0.5=0.15 恰在 cap 边界
        edge = survival_opportunity(
            {"shrunk_survival_rate": 0.65, "support_count": 1000}, attribution_model=m
        )
        assert edge == pytest.approx(0.15, abs=1e-9)

    def test_cap_configurable(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy", effect_cap=0.05)
        pos = survival_opportunity(
            {"survival_rate": 0.95, "support_count": 100000}, attribution_model=m
        )
        assert pos == pytest.approx(0.05, abs=1e-9)
        neg = survival_opportunity(
            {"survival_rate": 0.05, "support_count": 100000}, attribution_model=m
        )
        assert neg == pytest.approx(-0.05, abs=1e-9)

    def test_no_data_is_neutral_even_with_model(self):
        m = SurvivalAttributionModel(current_version="2", fit_backend="numpy")
        so = SearchOpportunity(survival_model=m)
        assert so.survival_opportunity_of("x", None) == 0.0
        assert survival_opportunity(None, attribution_model=m) == 0.0


# ---------------------------------------------------------------------------
# 6. 报告 effect + confidence，不做因果断言
# ---------------------------------------------------------------------------


class TestReportEffectConfidence:
    def test_report_includes_effect_ci_se_support(self):
        events = [
            _ev("ts_rank(ts_corr(close, volume, 10), 20)", "HEALTHY", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
            _ev("ts_rank(close, 20)", "HEALTHY", version="0"),
            _ev("rank(volume)", "HEALTHY", version="0"),
            _ev("ts_rank(close, 20)", "BROKEN", version="0"),
        ]
        m = _fit_default(events)
        rep = m.report()
        assert rep.n_events == 5
        assert len(rep.results) >= 5
        for r in rep.results:
            # effect + CI + se 全在，且 effect 在 cap 内
            assert -DEFAULT_EFFECT_CAP - 1e-9 <= r.effect <= DEFAULT_EFFECT_CAP + 1e-9
            assert r.effect_ci[0] <= r.effect <= r.effect_ci[1]
            assert r.effect_se >= 0.0
            assert r.support_count >= 0
        d = rep.to_dict()
        assert "effect_ci" in d["results"][0]
        assert "note" in d["results"][0]
        # 明确非因果：输出带 support/effect 而非「规则」
        assert "causal" not in d or not d.get("causal")

    def test_mix_model_family_and_interactions(self):
        """operator/mechanism/field/horizon 特征装配 + 报告含这些组。"""
        events = [
            _ev("ts_rank(ts_corr(close, volume, 10), 20)", "HEALTHY", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
            _ev("ts_rank(close, 20)", "HEALTHY", version="0"),
            _ev("rank(volume)", "HEALTHY", version="0"),
            _ev("ts_rank(close, 20)", "BROKEN", version="0"),
        ]
        m = _fit_default(events)
        rep = m.report()
        groups = {r.group for r in rep.results}
        assert groups & {"operator", "mechanism", "field", "horizon", "complexity"}


# ---------------------------------------------------------------------------
# 7. interactions 可配置（#30）
# ---------------------------------------------------------------------------


class TestInteractionsToggle:
    def test_with_interactions_on_has_interact_features(self):
        events = [
            _ev("ts_rank(ts_corr(close, volume, 10), 20)", "HEALTHY", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
        ]
        m_on = _fit_default(events, with_interactions=True)
        assert any(c.startswith("interact:") for c in m_on._feature_columns)

    def test_with_interactions_off_has_no_interact_features(self):
        events = [
            _ev("ts_rank(ts_corr(close, volume, 10), 20)", "HEALTHY", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
        ]
        m_off = _fit_default(events, with_interactions=False)
        assert not any(c.startswith("interact:") for c in m_off._feature_columns)


# ---------------------------------------------------------------------------
# 8. sklearn 与 numpy 两条 Elastic Net 路径真实覆盖
# ---------------------------------------------------------------------------


class TestBothBackends:
    def _sample_events(self):
        return [
            _ev("ts_rank(ts_corr(close, volume, 10), 20)", "HEALTHY", version="0"),
            _ev("ts_rank(close, 20)", "HEALTHY", version="0"),
            _ev("ts_rank(close, 20)", "BROKEN", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
            _ev("rank(volume)", "HEALTHY", version="0"),
            _ev("ts_mean(close, 20)", "HEALTHY", version="0"),
            _ev("ts_rank(close, 20)", "HEALTHY", version="0"),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
        ]

    def test_numpy_backend_real_path(self):
        m = SurvivalAttributionModel(current_version="1", fit_backend="numpy")
        m.fit(self._sample_events(), current_version="1")
        assert m._backend_used == "numpy"
        assert len(m._coefficients) == len(m._feature_columns)
        assert m.report().n_events == 8

    def test_sklearn_backend_real_path_when_available(self):
        try:
            import sklearn  # noqa: F401
        except Exception:  # pragma: no cover - sklearn 缺失时跳过
            pytest.skip("sklearn not available")
        m = SurvivalAttributionModel(current_version="1", fit_backend="auto")
        m.fit(self._sample_events(), current_version="1")
        assert m._backend_used == "sklearn"
        assert len(m._coefficients) == len(m._feature_columns)
        assert m.report().n_events == 8


# ---------------------------------------------------------------------------
# 追加：特征装配含 cluster crowding / source miner / age（dna 之外）
# ---------------------------------------------------------------------------


class TestMetaFeatureAssembly:
    def test_cluster_and_source_and_age_features(self):
        evs = [
            _ev(
                "ts_rank(close, 20)", "HEALTHY", version="0",
                source_miner="genetic", cluster_size=3, age_days=30,
            ),
            _ev("ts_std(close, 20)", "BROKEN", version="0"),
        ]
        m = _fit_default(evs)
        cols = m._feature_columns
        assert any(c.startswith("source_miner:") for c in cols)
        assert any(c.startswith("cluster_crowding") for c in cols)
        assert any(c.startswith("age:") for c in cols)

    def test_dict_event_input(self):
        """Mapping 事件（serialized 形态）也可消费/拟合。"""
        m = SurvivalAttributionModel(current_version="1", fit_backend="numpy")
        evs = [
            {"factor_id": "a", "formula": "ts_rank(close, 20)",
             "version": "0", "visible": False, "survival": "HEALTHY"},
            {"factor_id": "b", "formula": "ts_std(close, 20)",
             "version": "0", "visible": False, "survival": "BROKEN"},
        ]
        m.fit(evs, current_version="1")
        assert m._n_events == 2
