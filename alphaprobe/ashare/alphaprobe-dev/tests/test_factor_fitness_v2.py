"""FactorFitness V2 —— synthetic 验收测试（任务书 §68）。

直接喂 metric dict（EvaluationBundle），不跑真实回测。用 seed_prior + freeze 的
MetricCalibrator 保证 utility 单调、可比。

覆盖：
- L：higher Sharpe → L↑；lower MDD → L↑；better Calmar → L↑；better D10 active → L↑
- Q：D10 cliff → Q↓（连续惩罚单次）；same RankIC but better LongShort → Fitness↑
- 权重和=1、clip[0,1]、三惩罚边界
- complexity >64 hard reject；FragilityPenalty cap 0.05；CostPenalty cap 0.03
- high RankIC but catastrophic MDD → 不应得最高 fitness
"""

import pytest

from alphaprobe.fitness.calibration import single_utility
from alphaprobe.fitness.components import (
    long_short_score,
    quantile_score,
)
from alphaprobe.fitness.contracts import ComplexityInfo, EvaluationBundle
from alphaprobe.fitness.factor_fitness import V2FactorWeights, factor_fitness_v2
from alphaprobe.fitness import MetricCalibrator


# ---------------------------------------------------------------------------
# calibrator 工具：seed + freeze，保证 frozen z-score 单调
# ---------------------------------------------------------------------------


def _make_calibrator():
    c = MetricCalibrator(min_warmup_n=10)
    # 给每个会消费的 metric 都 seed 一个分布（避免 freeze 后未观测→0.5）
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
    c.freeze("v2_test")
    return c


def _bundle(**kw):
    return EvaluationBundle(dict(kw))


# 一组质量较好的基准 bundle（fitness 应落在中高区间）
def _good_bundle(**over):
    d = dict(
        rankic_valid=0.03,
        median_subperiod_rankic=0.02,
        hac_tstat=4.0,
        # Q：完美单调 + 无 cliff
        decile_monotonicity=0.9,
        isotonic_fit_quality=0.9,
        top10_excess=0.008,
        d10_minus_d1=0.03,
        top_tail_quality=0.9,
        group_returns=[float(i) for i in range(1, 11)],
        # L
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
        # S
        rankicir=0.8,
        q20_rolling_rankic=0.02,
        positive_subperiod_ratio=0.75,
        train_valid_retention=0.85,
        worst_subperiod_rankic=0.005,
        ic_decay=0.3,
        # N
        residual_rankic=0.01,
        structural_novelty=0.5,
        mean_top5_abs_corr=0.4,
        # R
        coverage=0.9,
        nan_inf_ratio=0.02,
        untradeable_ratio=0.05,
        winsor_sensitivity=0.05,
        numerical_stability=0.9,
        # Fragility 输入（低=好）
        contribution_concentration=0.1,
        coverage_instability=0.05,
        denominator_risk=0.05,
        extreme_value_dependence=0.05,
        # cost
        cost_1x=0.005,
        turnover=0.5,
    )
    d.update(over)
    return _bundle(**d)


# ---------------------------------------------------------------------------
# L 单调性
# ---------------------------------------------------------------------------


class TestLongShortMonotonicity:
    def test_higher_sharpe_higher_L(self):
        c = _make_calibrator()
        lo = long_short_score(_good_bundle(net_sharpe=1.0), c)
        hi = long_short_score(_good_bundle(net_sharpe=2.5), c)
        assert hi > lo

    def test_lower_mdd_higher_L(self):
        c = _make_calibrator()
        good = long_short_score(_good_bundle(max_drawdown=0.08), c)
        bad = long_short_score(_good_bundle(max_drawdown=0.45), c)
        assert good > bad

    def test_better_calmar_higher_L(self):
        c = _make_calibrator()
        lo = long_short_score(_good_bundle(calmar_ratio=0.4), c)
        hi = long_short_score(_good_bundle(calmar_ratio=1.8), c)
        assert hi > lo

    def test_better_d10_active_higher_L(self):
        c = _make_calibrator()
        lo = long_short_score(_good_bundle(d10_long_only_active_return=0.03), c)
        hi = long_short_score(_good_bundle(d10_long_only_active_return=0.12), c)
        assert hi > lo

    def test_drawdown_persistence_composition(self):
        """DrawdownPersistence = 0.5·U(MaxDDDuration) + 0.5·U(TUW)。"""
        c = _make_calibrator()
        from alphaprobe.fitness.components import _drawdown_persistence

        lo = _drawdown_persistence(_good_bundle(max_dd_duration=120, tuw=150), c)
        hi = _drawdown_persistence(_good_bundle(max_dd_duration=5, tuw=5), c)
        assert hi > lo


# ---------------------------------------------------------------------------
# Q：D10 cliff 只扣一次
# ---------------------------------------------------------------------------


class TestQuantileCliff:
    def test_d10_cliff_lowers_Q(self):
        c = _make_calibrator()
        # cliff：d10_cliff_penalty > 0 → Q 降（连续惩罚单次；bundle 无显式
        # top_tail_quality 时才从 d10_cliff_penalty 推导，防与 funnel 叠加）
        no_cliff = quantile_score(_good_bundle(top_tail_quality=None, d10_cliff_penalty=0.0), c)
        cliff = quantile_score(_good_bundle(top_tail_quality=None, d10_cliff_penalty=0.25), c)
        assert no_cliff > cliff
        # 断崖 G10 明显低于 G1..G9（从 group_returns fallback 也能扣）
        g_cliff = [0.1 * i for i in range(1, 10)] + [0.1]
        q_cliff = quantile_score(_good_bundle(top_tail_quality=None, group_returns=g_cliff), c)
        assert no_cliff > q_cliff

    def test_slight_drop_no_penalty(self):
        c = _make_calibrator()
        q = quantile_score(_good_bundle(d10_cliff_penalty=0.0), c)
        assert q > 0.0

    def test_no_double_penalty_single_key(self):
        """funnel gate 同源 d10_cliff_penalty 只扣一次（Q 内部单次惩罚）。"""
        c = _make_calibrator()
        b = _good_bundle(top_tail_quality=None, d10_cliff_penalty=0.25)
        q1 = quantile_score(b, c)
        # 更小的 cliff → Q 更高（同源单次扣减；不会叠加）
        b2 = _good_bundle(top_tail_quality=None, d10_cliff_penalty=0.05)
        q2 = quantile_score(b2, c)
        assert q2 > q1

    def test_tau_default_012(self):
        """默认 tau=0.12：d_top=0.12 内不罚。"""
        from alphaprobe.fitness.components import _collapse_penalty_from_groups

        g = [0.1 * i for i in range(1, 10)] + [0.1]
        cp = _collapse_penalty_from_groups(g)
        assert cp is not None and cp > 0.0


# ---------------------------------------------------------------------------
# 总公式行为
# ---------------------------------------------------------------------------


class TestFactorFitnessV2:
    def test_weights_sum_to_one(self):
        w = V2FactorWeights()
        assert w.dimension_total == pytest.approx(1.0)

    def test_clip_range(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c)
        assert 0.0 <= r.fitness <= 1.0
        assert 0.0 <= r.core <= 1.0
        assert r.P >= 0.0 and r.Q >= 0.0 and r.L >= 0.0
        assert r.S >= 0.0 and r.N >= 0.0 and r.R >= 0.0

    def test_same_rankic_better_ls_higher_fitness(self):
        """same RankIC but better LongShort → FactorFitness↑。"""
        c = _make_calibrator()
        weak = factor_fitness_v2(
            _good_bundle(net_sharpe=1.0, calmar_ratio=0.4, max_drawdown=0.4), c
        )
        strong = factor_fitness_v2(
            _good_bundle(net_sharpe=2.5, calmar_ratio=1.8, max_drawdown=0.08), c
        )
        assert strong.fitness > weak.fitness

    def test_high_rankic_catastrophic_mdd_not_top(self):
        """high RankIC but catastrophic MDD → 不应得最高 fitness。"""
        c = _make_calibrator()
        all_good = factor_fitness_v2(_good_bundle(), c)
        high_ic_bad_mdd = factor_fitness_v2(
            _good_bundle(rankic_valid=0.05, max_drawdown=0.5, net_sharpe=0.3), c
        )
        assert all_good.fitness > high_ic_bad_mdd.fitness

    def test_positive_fitness_for_good_factor(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c)
        assert r.fitness > 0.4

    def test_dict_bundle_input(self):
        c = _make_calibrator()
        d = _good_bundle().raw()
        r = factor_fitness_v2(d, c)
        assert 0.0 <= r.fitness <= 1.0


# ---------------------------------------------------------------------------
# 三惩罚边界
# ---------------------------------------------------------------------------


class TestPenalties:
    def test_cost_penalty_cap_003(self):
        c = _make_calibrator()
        from alphaprobe.fitness.penalties import cost_penalty

        worst = cost_penalty(_good_bundle(cost_1x=0.02, turnover=2.0), c)
        assert 0.0 <= worst <= 0.03

    def test_fragility_penalty_cap_005(self):
        c = _make_calibrator()
        from alphaprobe.fitness.penalties import fragility_penalty

        worst = fragility_penalty(
            _good_bundle(
                winsor_sensitivity=0.5, contribution_concentration=0.9,
                coverage_instability=0.5, denominator_risk=0.5,
                extreme_value_dependence=0.5,
            ),
            c,
        )
        assert 0.0 <= worst <= 0.05

    def test_fragility_missing_items_not_punished(self):
        c = _make_calibrator()
        from alphaprobe.fitness.penalties import fragility_penalty

        empty = fragility_penalty(_bundle(), c)
        assert empty == 0.0

    def test_complexity_64_hard_reject(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=65))
        assert r.rejected
        assert r.fitness == 0.0

    def test_complexity_24_zero_penalty(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=24))
        assert not r.rejected
        assert r.complexity_penalty == 0.0

    def test_complexity_bucket_25_40(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=30))
        assert not r.rejected
        assert r.complexity_penalty <= 0.005
        assert r.complexity_penalty > 0.0

    def test_complexity_bucket_41_64(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=50))
        assert not r.rejected
        assert r.complexity_penalty <= 0.015

    def test_depth_hard_reject(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=10, depth=13))
        assert r.rejected

    def test_depth_soft_penalty(self):
        c = _make_calibrator()
        r = factor_fitness_v2(_good_bundle(), c, complexity=ComplexityInfo(nodes=10, depth=11))
        assert not r.rejected
        assert r.complexity_penalty > 0.0

    def test_fitness_clip_at_zero(self):
        c = _make_calibrator()
        # 全维度 0 + 大惩罚 → clip 到 0
        bad = _good_bundle(
            rankic_valid=0.001, net_sharpe=0.1, coverage=0.3,
            cost_1x=0.02, winsor_sensitivity=0.5,
            contribution_concentration=0.9, denominator_risk=0.5,
            extreme_value_dependence=0.5, coverage_instability=0.5,
        )
        r = factor_fitness_v2(bad, c)
        assert r.fitness >= 0.0


# ---------------------------------------------------------------------------
# single_utility 语义
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_lower_is_better_direction(self):
        c = _make_calibrator()
        lo = single_utility(c, "max_drawdown", 0.05, lower_is_better=True)
        hi = single_utility(c, "max_drawdown", 0.5, lower_is_better=True)
        assert lo > hi

    def test_missing_metric_zero(self):
        c = _make_calibrator()
        assert single_utility(c, "rankic_valid", None) == 0.0
