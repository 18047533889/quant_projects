"""FactorFitness V2.1 —— confidence shrinkage / metric missing policy / soft floors。

plan.md Task 8（FactorFitness V2.1）+ Part F1 + Part G #19/#20/#21 + Part J3。

覆盖：
1. MetricRequirement（REQUIRED / OPTIONAL / DIAGNOSTIC）：
   - REQUIRED 在给定 fidelity 缺失 → 不能 promote（L3 不可过）。
   - OPTIONAL 缺失 → 该维度 utility 0.5（不是 0 也不是 1）。
   - DIAGNOSTIC 缺失 → 无 score 影响。
2. Confidence shrinkage：
   - reliability = sqrt(n_eff/(n_eff+k))；U_conf = 0.5 + reliability*(U-0.5)。
   - 同 Sharpe 但更大有效样本 → 更高置信调整后 L。
   - 无有效样本数 → 中性 / 走 required-policy，绝不当最好。
3. Soft quality floors：
   - Fitness = clip(Core × gP × gL × gS × gR - penalties, 0, 1)；guard 平滑可配置
     （非隐藏硬门）；很强 P 无法完全抵消灾难性 L；填充良好正常用例下 V2 原排序保持。
4. k 与 guard 阈值全部可配置（Part J3 中性默认值）。
"""

from __future__ import annotations

import pytest

from alphaprobe.fitness import MetricCalibrator
from alphaprobe.fitness.contracts import (
    ComplexityInfo,
    EvaluationBundle,
    FactorFitnessResult,
)
from alphaprobe.fitness.factor_fitness import (
    DEFAULT_V2_WEIGHTS,
    V21SoftFloorConfig,
    V2FactorWeights,
    factor_fitness_v2,
    factor_fitness_v2_1,
)


# ---------------------------------------------------------------------------
# calibrator 工具（与 test_factor_fitness_v2.py 同套路：seed + freeze）
# ---------------------------------------------------------------------------


def _make_calibrator() -> MetricCalibrator:
    c = MetricCalibrator(min_warmup_n=10)
    seeds = {
        # P
        "rankic_valid": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
        "median_subperiod_rankic": [0.003, 0.008, 0.012, 0.02, 0.03, 0.04],
        "hac_tstat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        # Q
        "decile_monotonicity": [0.1, 0.3, 0.5, 0.7, 0.9],
        "isotonic_fit_quality": [0.1, 0.3, 0.5, 0.7, 0.95],
        "top10_excess": [0.001, 0.002, 0.005, 0.01],
        "d10_minus_d1": [0.005, 0.01, 0.02, 0.04],
        "top_tail_quality": [0.3, 0.5, 0.7, 0.9, 1.0],
        # L
        "net_sharpe": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "calmar_ratio": [0.2, 0.5, 0.8, 1.2, 1.6, 2.0],
        "sortino_ratio": [0.6, 1.0, 1.5, 2.0, 2.5],
        "net_annualized_ls_return": [0.05, 0.1, 0.15, 0.2, 0.3],
        "d10_long_only_active_return": [0.02, 0.04, 0.06, 0.1, 0.15],
        "max_drawdown": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        "max_dd_duration": [5, 10, 20, 40, 80, 160],
        "tuw": [5, 10, 20, 40, 80, 160],
        "q20_rolling_sharpe": [0.3, 0.6, 1.0, 1.5, 2.0],
        "positive_month_ratio": [0.4, 0.5, 0.6, 0.7, 0.8],
        # S
        "rankicir": [0.1, 0.3, 0.5, 0.8, 1.2],
        "q20_rolling_rankic": [0.005, 0.01, 0.02, 0.03],
        "positive_subperiod_ratio": [0.4, 0.55, 0.7, 0.85],
        "train_valid_retention": [0.3, 0.5, 0.7, 0.9],
        "worst_subperiod_rankic": [-0.01, 0.0, 0.005, 0.01, 0.02],
        "ic_decay": [0.1, 0.3, 0.5, 0.7],
        # N
        "residual_rankic": [0.002, 0.005, 0.01, 0.02],
        "structural_novelty": [0.2, 0.4, 0.6, 0.8],
        "mean_top5_abs_corr": [0.2, 0.4, 0.6, 0.8],
        # R
        "coverage": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "nan_inf_ratio": [0.0, 0.05, 0.1, 0.2, 0.4],
        "untradeable_ratio": [0.0, 0.05, 0.1, 0.2, 0.4],
        "winsor_sensitivity": [0.0, 0.02, 0.05, 0.1, 0.2],
        "numerical_stability": [0.5, 0.7, 0.9, 1.0],
        # Fragility
        "contribution_concentration": [0.0, 0.1, 0.3, 0.5],
        "coverage_instability": [0.0, 0.05, 0.1, 0.2],
        "denominator_risk": [0.0, 0.05, 0.1, 0.2],
        "extreme_value_dependence": [0.0, 0.05, 0.1, 0.2],
        # cost
        "cost_1x": [0.001, 0.002, 0.005, 0.01, 0.02],
        "turnover": [0.1, 0.2, 0.5, 1.0, 2.0],
    }
    c.seed_prior(seeds)
    c.freeze("v21_test")
    return c


def _bundle(**kw) -> EvaluationBundle:
    return EvaluationBundle(dict(kw))


def _good_bundle(**over) -> EvaluationBundle:
    d = dict(
        rankic_valid=0.03,
        median_subperiod_rankic=0.02,
        hac_tstat=4.0,
        decile_monotonicity=0.9,
        isotonic_fit_quality=0.9,
        top10_excess=0.008,
        d10_minus_d1=0.03,
        top_tail_quality=0.9,
        group_returns=[float(i) for i in range(1, 11)],
        net_sharpe=2.0,
        calmar_ratio=1.2,
        sortino_ratio=1.8,
        net_annualized_ls_return=0.15,
        d10_long_only_active_return=0.08,
        max_drawdown=0.12,
        max_dd_duration=20,
        tuw=30,
        q20_rolling_sharpe=1.2,
        positive_month_ratio=0.7,
        rankicir=0.8,
        q20_rolling_rankic=0.02,
        positive_subperiod_ratio=0.75,
        train_valid_retention=0.85,
        worst_subperiod_rankic=0.005,
        ic_decay=0.3,
        residual_rankic=0.01,
        structural_novelty=0.5,
        mean_top5_abs_corr=0.4,
        coverage=0.9,
        nan_inf_ratio=0.02,
        untradeable_ratio=0.05,
        winsor_sensitivity=0.05,
        numerical_stability=0.9,
        contribution_concentration=0.1,
        coverage_instability=0.05,
        denominator_risk=0.05,
        extreme_value_dependence=0.05,
        cost_1x=0.005,
        turnover=0.5,
    )
    d.update(over)
    return _bundle(**d)


# ---------------------------------------------------------------------------
# 8.1 MetricRequirement + missing-policy
# ---------------------------------------------------------------------------


class TestMetricRequirement:
    def test_members(self):
        from alphaprobe.fitness.contracts import MetricRequirement

        assert {m.name for m in MetricRequirement} == {
            "REQUIRED",
            "OPTIONAL",
            "DIAGNOSTIC",
        }
        assert MetricRequirement.REQUIRED.value == "required"
        assert MetricRequirement.OPTIONAL.value == "optional"
        assert MetricRequirement.DIAGNOSTIC.value == "diagnostic"

    def test_optional_missing_utility_is_05(self):
        """OPTIONAL 缺失 → 该维度 utility 0.5（不是 0 也不是 1）。"""
        from alphaprobe.fitness.contracts import MetricRequirement
        from alphaprobe.fitness.missing_policy import optional_fallback_utility

        u = optional_fallback_utility(
            MetricRequirement.OPTIONAL,
            key="whatever",
            calibrator=_make_calibrator(),
        )
        assert u == pytest.approx(0.5)
        assert u != 0.0 and u != 1.0

    def test_diagnostic_missing_no_impact(self):
        from alphaprobe.fitness.contracts import MetricRequirement
        from alphaprobe.fitness.missing_policy import requirement_fallback_utility

        # DIAGNOSTIC 缺失时不给分数影响：fallback 为 None → 由外层原样跳过
        assert requirement_fallback_utility(MetricRequirement.DIAGNOSTIC) is None
        # REQUIRED 缺失 → None（是否 promote 由 gate 层用 missing_policy 判定）
        assert requirement_fallback_utility(MetricRequirement.REQUIRED) is None

    def test_required_missing_cannot_pass_l3(self):
        """REQUIRED long-short metric 缺失 → L3 不可过（funnel gate 拒绝）。"""
        from alphaprobe.contracts import EvaluationRecord, FidelityLevel, RejectionReason
        from alphaprobe.fitness.contracts import (
            MetricRequirement,
            MetricRequirementPolicy,
        )
        from alphaprobe.fitness.funnel import FidelityFunnel

        funnel = FidelityFunnel()
        rec = EvaluationRecord(
            factor_id="f1",
            segment="search_valid",
            fidelity=FidelityLevel.L3_SEARCH_VALID.value,
            metric_bundle=_good_bundle(net_sharpe=None).raw(),
            artifact_refs={},
            evaluator_version="t",
            data_snapshot_id="s",
            universe_snapshot_id="u",
            label_spec_hash="h",
            created_at=None,
        )
        reasons = funnel.gate(
            "L3_search_valid",
            rec,
            metric_requirements=MetricRequirementPolicy(
                {MetricRequirement.REQUIRED: ("net_sharpe",)}
            ),
        )
        assert reasons  # REQUIRED 缺失 → 拒绝（不能 promote）

    def test_optional_missing_does_not_block_l3(self):
        from alphaprobe.contracts import EvaluationRecord, FidelityLevel
        from alphaprobe.fitness.contracts import (
            MetricRequirement,
            MetricRequirementPolicy,
        )
        from alphaprobe.fitness.funnel import FidelityFunnel

        funnel = FidelityFunnel()
        rec = EvaluationRecord(
            factor_id="f1",
            segment="search_valid",
            fidelity=FidelityLevel.L3_SEARCH_VALID.value,
            metric_bundle=_good_bundle(net_sharpe=None).raw(),
            artifact_refs={},
            evaluator_version="t",
            data_snapshot_id="s",
            universe_snapshot_id="u",
            label_spec_hash="h",
            created_at=None,
        )
        reasons = funnel.gate(
            "L3_search_valid",
            rec,
            metric_requirements=MetricRequirementPolicy(
                {MetricRequirement.OPTIONAL: ("net_sharpe",)}
            ),
        )
        assert not reasons  # OPTIONAL 缺失不阻断 promote


# ---------------------------------------------------------------------------
# 8.2 Confidence shrinkage
# ---------------------------------------------------------------------------


class TestConfidenceShrinkage:
    def test_reliability_formula(self):
        from alphaprobe.fitness.confidence import reliability, shrink_utility

        # reliability = sqrt(n_eff/(n_eff+k))
        assert reliability(n_eff=0, k=64.0) == pytest.approx(0.0)
        assert reliability(n_eff=64, k=64.0) == pytest.approx((0.5) ** 0.5)
        r_big = reliability(n_eff=10000, k=64.0)
        assert 0.99 < r_big <= 1.0

        # U_conf = 0.5 + reliability * (U - 0.5)
        assert shrink_utility(u=0.9, reliability=1.0) == pytest.approx(0.9)
        assert shrink_utility(u=0.9, reliability=0.0) == pytest.approx(0.5)
        # 高于 0.5 的 U 被向下收缩、低于 0.5 的 U 被向上收缩
        assert shrink_utility(u=0.9, reliability=0.5) == pytest.approx(0.7)
        assert shrink_utility(u=0.1, reliability=0.5) == pytest.approx(0.3)

    def test_same_sharpe_larger_neff_higher_confidence_L(self):
        """同 Sharpe（同 raw utility），更大有效样本 → 更高置信调整后 L。"""
        c = _make_calibrator()
        low_n = factor_fitness_v2_1(
            _good_bundle(), c, dimension_n_eff={"L": 5}, k=64.0,
            apply_confidence_shrinkage=True,
        )
        high_n = factor_fitness_v2_1(
            _good_bundle(), c, dimension_n_eff={"L": 5000}, k=64.0,
            apply_confidence_shrinkage=True,
        )
        assert high_n.L > low_n.L
        assert high_n.L > 0.5 > low_n.L or high_n.L > low_n.L

    def test_no_effective_sample_is_neutral_not_best(self):
        """无有效样本数 → 中性（>0.5 的 U 回缩到接近 0.5），绝不当最好。"""
        c = _make_calibrator()
        no_n = factor_fitness_v2_1(
            _good_bundle(), c, dimension_n_eff={"L": 0}, k=64.0,
            apply_confidence_shrinkage=True,
        )
        full_n = factor_fitness_v2_1(
            _good_bundle(), c, dimension_n_eff={"L": 2000}, k=64.0,
            apply_confidence_shrinkage=True,
        )
        # 无有效样本的 L 被回缩到中性附近，不会超过有充分样本的同 raw L
        assert no_n.L < full_n.L
        assert no_n.L == pytest.approx(0.5, abs=0.06)


# ---------------------------------------------------------------------------
# 8.3 Soft quality floors
# ---------------------------------------------------------------------------


class TestSoftQualityFloors:
    def test_very_strong_P_cannot_fully_offset_catastrophic_L(self):
        """高 P 无法完全抵消灾难性 L：bad-L factor 不能压过 all-good factor。"""
        c = _make_calibrator()
        all_good = factor_fitness_v2_1(
            _good_bundle(), c, soft_floor=V21SoftFloorConfig(l_floor=0.20),
            apply_soft_floors=True,
        )
        # 灾难性 L：极差 long-short（net_sharpe 触底 + MDD 巨大）但 P 拉满
        cat_l = factor_fitness_v2_1(
            _good_bundle(
                rankic_valid=0.05,
                median_subperiod_rankic=0.04,
                hac_tstat=6.0,
                net_sharpe=0.05,
                calmar_ratio=0.01,
                sortino_ratio=0.1,
                net_annualized_ls_return=-0.05,
                d10_long_only_active_return=0.0,
                max_drawdown=0.6,
                max_dd_duration=400,
                tuw=300,
                q20_rolling_sharpe=0.1,
                positive_month_ratio=0.2,
            ),
            c,
            soft_floor=V21SoftFloorConfig(l_floor=0.20),
            apply_soft_floors=True,
        )
        assert cat_l.L < 0.2
        assert all_good.fitness > cat_l.fitness

    def test_guard_floor_is_smooth_not_hard_gate(self):
        """Soft floor 平滑可配置：L 极差时 gL 连续下降而非一刀切拒绝。"""
        from alphaprobe.fitness.factor_fitness import guard_g

        # guard_g 在 floor 之上为 1.0；floor 以下平滑连续降到 0；0 处 g=0
        assert guard_g(0.6, floor=0.25) == pytest.approx(1.0)
        below = guard_g(0.2, floor=0.25)  # 0 < u < floor
        assert 0.0 < below < 1.0
        assert guard_g(0.24, floor=0.25) > guard_g(0.2, floor=0.25)  # 平滑单调
        assert guard_g(0.0, floor=0.25) == pytest.approx(0.0)
        # 可配置：更低的 floor → 同样 bad 分惩罚更轻
        assert guard_g(0.2, floor=0.1) > guard_g(0.2, floor=0.4)
        # floor=0 → guard 关闭（恒 1，与 V2 一致）
        assert guard_g(0.0, floor=0.0) == pytest.approx(1.0)

    def test_v2_ranking_retained_when_well_populated(self):
        """正常填充良好用例下 V2 原排序保持（同 rankic、better LS → fitness 更高）。"""
        c = _make_calibrator()
        weak = factor_fitness_v2_1(
            _good_bundle(net_sharpe=1.0, calmar_ratio=0.4, max_drawdown=0.4), c
        )
        strong = factor_fitness_v2_1(
            _good_bundle(net_sharpe=2.5, calmar_ratio=1.8, max_drawdown=0.08), c
        )
        assert strong.fitness > weak.fitness

    def test_v21_matches_v2_when_guards_pass(self):
        """全维度高分 → 各 guard = 1.0，V2.1 == V2（无 hidden gate 改变语义）。"""
        c = _make_calibrator()
        v2 = factor_fitness_v2(_good_bundle(), c)
        v21 = factor_fitness_v2_1(_good_bundle(), c)
        assert v21.fitness == pytest.approx(v2.fitness)
        assert v21.fitness > 0.4

    def test_penalties_still_subtracted_after_product(self):
        """Fitness = clip(Core × gP×gL×gS×gR − penalties, 0, 1)。

        注：V2 cost_penalty 语义为 cap × U(cost)，U 的 lower_is_better 单调方向由
        calibrator 分布决定（V2 既有语义，非 Task 8 范围）；这里只锚定结构：
        fitness == clip(core − 三惩罚)（默认无 guard 时），且 penalties 从乘后
        core 扣减、不是叠加在 core 上。
        """
        c = _make_calibrator()
        v2 = factor_fitness_v2_1(_good_bundle(), c)
        # 无 guard（默认 floor=0）→ fitness == clip(core - 三惩罚)
        assert v2.fitness == pytest.approx(
            v2.core - v2.cost_penalty - v2.complexity_penalty - v2.fragility_penalty
        )
        # 惩罚为正时 fitness 严格低于 core（乘后扣减可观测）
        assert v2.cost_penalty >= 0.0 and v2.fragility_penalty >= 0.0
        assert v2.fitness <= v2.core + 1e-9
        # 灾难性 cost 用例下 fitness 也能被压到 0 以下再 clip 到 ≥0（不炸）
        bad = factor_fitness_v2_1(_good_bundle(cost_1x=0.02, turnover=2.0), c)
        assert 0.0 <= bad.fitness <= 1.0


# ---------------------------------------------------------------------------
# 默认配置可配置性 / 向后兼容（Part J3 中性默认值）
# ---------------------------------------------------------------------------


class TestV21DefaultsAndConfig:
    def test_default_config_has_neutral_k_and_floors(self):
        from alphaprobe.fitness.factor_fitness import (
            V21SoftFloorConfig,
            _v21_default_soft_floors,
        )
        from alphaprobe.fitness.confidence import DEFAULT_SHRINKAGE_K

        cfg = V21SoftFloorConfig()
        assert cfg.guard_floor.P == 0.0 and cfg.guard_floor.L == 0.0
        assert _v21_default_soft_floors() == cfg  # 默认全 0 floor（V2 逐字一致）
        # k 中性默认值合理（非 0、非巨量），且可被覆盖
        assert DEFAULT_SHRINKAGE_K > 0.0
        custom = factor_fitness_v2_1(
            _good_bundle(), _make_calibrator(), k=8.0,
            soft_floor=V21SoftFloorConfig(l_floor=0.1),
        )
        assert 0.0 <= custom.fitness <= 1.0

    def test_v2_weights_still_work_with_v21(self):
        """V2.1 沿用 V2FactorWeights；显式权重对象被正确消费。"""
        c = _make_calibrator()
        w = V2FactorWeights(P=0.1, Q=0.1, L=0.5, S=0.1, N=0.1, R=0.1)
        r = factor_fitness_v2_1(_good_bundle(), c, weights=w)
        assert 0.0 <= r.fitness <= 1.0
        assert isinstance(r, FactorFitnessResult)

    def test_dimension_n_eff_optional(self):
        """不传 dimension_n_eff → 默认全 None → 无收缩（与 V2 逐字相同）。"""
        c = _make_calibrator()
        a = factor_fitness_v2_1(_good_bundle(), c)
        b = factor_fitness_v2(_good_bundle(), c)
        assert a.L == pytest.approx(b.L)
        assert a.fitness == pytest.approx(b.fitness)


# ---------------------------------------------------------------------------
# 实现内部 helper 直接单测（monkeypatch 友好）
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_effective_n_from_metric(self):
        """有效样本可来自 metric_bundle 的 n 键或显式 dimension_n_eff。"""
        from alphaprobe.fitness.factor_fitness import _effective_n_for

        # 显式 dict 优先
        assert _effective_n_for({"L": 77}, key="L", metric_n_keys=("n_L",)) == 77
        # 无显式 → bundle 里读 n 键
        b = _good_bundle(n_ls_days=120)
        assert _effective_n_for(None, key="L", metric_n_keys=("n_ls_days",), bundle=b) == 120
        # 都没有 → None（不编造样本量）
        assert _effective_n_for(None, key="L", metric_n_keys=("n_ls_days",)) is None
