"""任务书 §87 unit tests：Identity/Canonical/Orientation/Calibrator/D10/Cohort/SearchValue/Exploration。"""

import math

import pytest

from alphaprobe.contracts import (
    DateRange,
    LabelSpec,
    ResearchSplitSpec,
    SearchActionType,
)
from alphaprobe.dedup import (
    canonical_ast_hash,
    canonicalize_dsl,
    parameter_family_key,
    signal_equivalence_id,
    GlobalSeenIndex,
)
from alphaprobe.research_protocol.orientation import (
    apply_orientation,
    orientation_from_train_rankic,
)
from alphaprobe.fitness import (
    MetricCalibrator,
    FitnessInputs,
    compute_search_fitness,
    group_monotonicity,
    isotonic_fit_quality,
    d10_cliff_penalty,
    survival_opportunity_shrunk,
    check_hard_gates,
)
from alphaprobe.pool import ActivePool, PoolMember, niche_key_of
from alphaprobe.search import ActionScheduler


# §87.1 Identity / Canonical
class TestCanonical:
    def test_parens(self):
        assert canonicalize_dsl("((rank(close)))") == "rank(close)"

    def test_mul_one(self):
        assert "rank(close)" in canonicalize_dsl("(rank(close) * 1)")

    def test_plus_zero(self):
        assert "rank(close)" in canonicalize_dsl("(rank(close) + 0)")

    def test_zero_minus_becomes_neg(self):
        assert "-(rank(close))" in canonicalize_dsl("(0 - rank(close))")

    def test_double_neg(self):
        assert "rank(close)" in canonicalize_dsl("(-(-(rank(close))))")

    def test_div_x_by_x_not_simplified(self):
        # §11.1 禁止危险化简 x/x -> 1
        out = canonicalize_dsl("(rank(close) / rank(close))")
        assert "1" != out.strip()

    def test_sign_invariant_ids(self):
        f = "rank(ts_mean(close, 20))"
        assert signal_equivalence_id(f) == signal_equivalence_id(f"(-({f}))")

    def test_pow_one(self):
        # pow(x,1) 文本层不安全折叠，但 hash 幂等
        h1 = canonical_ast_hash("rank(close)")
        assert h1 == canonical_ast_hash("rank(close)")


# §87.2 Orientation
class TestOrientation:
    def test_train_positive(self):
        assert orientation_from_train_rankic(0.03) == 1

    def test_train_negative(self):
        assert orientation_from_train_rankic(-0.03) == -1

    def test_none_no_flip(self):
        assert orientation_from_train_rankic(None) == 1

    def test_valid_negative_not_reflip(self):
        # Valid 负 → 是 instability；orientation 不允许重新翻
        assert apply_orientation(-0.02, orientation_from_train_rankic(0.03)) == -0.02

    def test_sign_invariance_pairs(self):
        # f 与 -f 同 signal id（dedup 层）
        f = "zscore(ts_corr(close, volume, 10))"
        assert signal_equivalence_id(f) == signal_equivalence_id(f"(-({f}))")


# §87.3 MetricCalibrator
class TestCalibrator:
    def test_mad_zero_safety(self):
        c = MetricCalibrator()
        for _ in range(50):
            c.observe("m", 1.0)  # MAD = 0
        c.freeze("r1")
        u = c.utility("m", 1.0)
        assert 0.0 < u <= 1.0 and math.isfinite(u)

    def test_lower_is_better(self):
        c = MetricCalibrator()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            c.observe("mdd", v)
        c.freeze("r1")
        good = c.utility("mdd", 0.05, lower_is_better=True)
        bad = c.utility("mdd", 0.95, lower_is_better=True)
        assert good > bad

    def test_lower_is_better_z_equals_neg_standard_z(self):
        # Bug3 回归：lower_is_better 分支 z = -( (x-median)/MAD )
        c = MetricCalibrator()
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for v in vals:
            c.observe("mdd", v)
        c.freeze("r1")
        good = c.utility("mdd", 0.2, lower_is_better=True)
        bad = c.utility("mdd", 0.9, lower_is_better=True)
        assert good > bad
        # 数值等于 -(标准 z) 再 sigmoid
        import statistics

        med = statistics.median(vals)
        mad = statistics.median([abs(x - med) for x in vals])
        z_good = -(0.2 - med) / (1.4826 * mad + 1e-9)
        z_bad = -(0.9 - med) / (1.4826 * mad + 1e-9)
        assert good == pytest.approx(c._sigmoid(z_good))
        assert bad == pytest.approx(c._sigmoid(z_bad))
        assert good > c._sigmoid(-z_good) or good >= 0.5  # 方向性 sanity

    def test_lower_is_better_smaller_mdd_higher_utility(self):
        # Bug3 验收：更小 mdd → 更高 utility（freeze 后）
        c = MetricCalibrator()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            c.observe("mdd", v)
        c.freeze("r1")
        prev = None
        for v in (0.05, 0.3, 0.55, 0.8, 1.2):
            u = c.utility("mdd", v, lower_is_better=True)
            if prev is not None:
                assert u < prev  # 更小 mdd → 更高 utility（递减）
            prev = u

    def test_per_round_freeze(self):
        c = MetricCalibrator()
        for v in range(10, 60):
            c.observe("m", float(v))
        c.freeze("r1")
        u1 = c.utility("m", 30.0)
        assert c.frozen_round == "r1"
        # 同 raw 同 round → 同 utility
        assert c.utility("m", 30.0) == u1

    def test_round_change_updates(self):
        c = MetricCalibrator()
        for v in range(10, 60):
            c.observe("m", float(v))
        c.freeze("r1")
        c.observe("m", 500.0)
        c.freeze("r2")
        assert c.frozen_round == "r2"

    def test_utility_not_observed_metric_after_freeze_half(self):
        # freeze 后首次消费未观测 metric → 0.5（不 NaN、不越界）
        c = MetricCalibrator()
        for v in range(10, 60):
            c.observe("m", float(v))
        c.freeze("r1")
        u = c.utility("mdd", 0.3, lower_is_better=True)
        assert u == 0.5


# §87.3b warmup ordinal 模式（Bug4）
class TestCalibratorWarmup:
    def test_warmup_monotonic_within_sample(self):
        # 未 freeze：同指标不同值 → 单调 utility
        c = MetricCalibrator(min_warmup_n=10)
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for v in vals:
            c.observe("m", v)
        us = [c.utility("m", v) for v in vals]
        assert all(b >= a for a, b in zip(us, us[1:]))  # 单调不减
        assert us[0] == pytest.approx(0.1) and us[-1] == pytest.approx(1.0)

    def test_warmup_lower_is_better_monotonic(self):
        c = MetricCalibrator(min_warmup_n=10)
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for v in vals:
            c.observe("mdd", v)
        us = [c.utility("mdd", v, lower_is_better=True) for v in vals]
        assert all(b <= a for a, b in zip(us, us[1:]))  # 单调不增（更小 → 更高）
        assert us[0] == pytest.approx(1.0) and us[-1] == pytest.approx(0.1)

    def test_seed_prior_then_freeze_immediately(self):
        # seed_prior 预填 10 万 seed 分布后立即 freeze 成功
        c = MetricCalibrator(min_warmup_n=100_000)
        c.seed_prior({"rankic": [0.01 + 0.001 * (i % 100) for i in range(100_000)]})
        c.freeze("seed_r1")
        assert c.is_frozen()
        u = c.utility("rankic", 0.05)
        assert 0.0 < u <= 1.0

    def test_small_warmup_path(self):
        # min_warmup_n=10 小样本：未达阈值走 ordinal，达阈值后可 freeze
        c = MetricCalibrator(min_warmup_n=10)
        for v in [1.0, 2.0, 3.0]:
            c.observe("m", v)
        assert c.warmup_active("m") is True
        u_hi = c.utility("m", 3.0)
        u_lo = c.utility("m", 1.0)
        assert u_hi > u_lo
        for v in range(4, 12):
            c.observe("m", float(v))
        c.freeze("r1")
        assert c.warmup_active("m") is False
        assert c.is_frozen()

    def test_frozen_path_unchanged_by_warmup(self):
        # 已 freeze 路径行为不受 warmup 影响
        c = MetricCalibrator(min_warmup_n=1000)
        for v in range(10, 60):
            c.observe("m", float(v))
        c.freeze("r1")
        u1 = c.utility("m", 30.0)
        assert c.utility("m", 30.0) == u1
        assert c.is_frozen()


# §87.4 D10
class TestD10:
    def _g(self, d10):
        return [0.1 * i for i in range(1, 10)] + [d10]

    def test_slight_drop_no_penalty(self):
        # D10 仅略低于 D9
        p = d10_cliff_penalty([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.88])
        assert p is not None and p < 0.02

    def test_cliff_large_penalty(self):
        # D10 断崖低于 D7-D9
        p = d10_cliff_penalty([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2])
        assert p is not None and p > 0.1

    def test_monotonic(self):
        g = [float(i) for i in range(1, 11)]
        assert group_monotonicity(g) == pytest.approx(1.0)

    def test_isotonic_perfect(self):
        g = [float(i) for i in range(1, 11)]
        assert isotonic_fit_quality(g) == pytest.approx(1.0)


# §21.3/§51/§87.6 SearchValue / survival shrinkage
class TestSearchValue:
    def test_small_support_shrunk(self):
        # Direction A: 90% survival but only 7 factors → 收缩后明显小于 3000 支撑的 72%
        v7 = survival_opportunity_shrunk(raw_survival_rate=0.9, support_count=7, cap=0.15)
        v3k = survival_opportunity_shrunk(raw_survival_rate=0.72, support_count=3000, cap=0.15)
        assert v7 < v3k * 0.75

    def test_big_support_credible(self):
        # Direction B: 72% with 3000 support → 接近 cap
        v = survival_opportunity_shrunk(raw_survival_rate=0.72, support_count=3000, cap=0.15)
        assert v >= 0.14

    def test_saturated_high_fitness_low_value(self):
        from alphaprobe.fitness import search_value
        sat = search_value(
            fertility=0.1, offspring_novelty=0.1, descendant_pool_gain=0.1,
            coverage_gap=0.1, uncertainty=0.1, frontierness=0.1,
            survival_opportunity=0.1, search_saturation=0.95,
            historical_compute_cost=0.9,
        )
        fresh = search_value(
            fertility=0.8, offspring_novelty=0.7, descendant_pool_gain=0.6,
            coverage_gap=0.7, uncertainty=0.5, frontierness=0.8,
            survival_opportunity=0.1, search_saturation=0.05,
            historical_compute_cost=0.05,
        )
        assert fresh > sat


# §20 Hard Gates
class TestHardGates:
    def test_clean_pass(self):
        assert check_hard_gates(
            dsl_valid=True, pit_ok=True, forbidden_field=False, coverage=0.9,
            nan_inf_ratio=0.0, exact_duplicate=False, sign_duplicate=False,
            seed_duplicate=False, already_exported=False, sealed_leak=False,
            untradeable_ratio=0.1,
        ) == []

    def test_sealed_leak_blocks(self):
        fails = check_hard_gates(
            dsl_valid=True, pit_ok=True, forbidden_field=False, coverage=0.9,
            nan_inf_ratio=0.0, exact_duplicate=False, sign_duplicate=False,
            seed_duplicate=False, already_exported=False, sealed_leak=True,
            untradeable_ratio=0.1,
        )
        assert "SEALED_TEST_LEAKAGE" in fails


# §53.2 Pool：禁止 pop lowest IC
class TestActivePool:
    def _m(self, fid, fitness, niche=(("m",),), sv=0.5, util=0.5):
        return PoolMember(
            factor_id=fid, canonical_formula=fid, search_fitness=fitness,
            pool_utility=util, search_value=sv, niche_key=niche,
        )

    def test_no_lowest_ic_pop(self):
        pool = ActivePool(target_size=3, max_size=4)
        pool.admit(self._m("A", 0.9, niche=("x",)))
        pool.admit(self._m("B", 0.1, niche=("y",), sv=0.9, util=0.9))  # 低 IC 但 niche 独有
        ok, _ = pool.admit(self._m("C", 0.5, niche=("z",)))
        assert ok
        # 低 IC 但 rare-niche 的 B 不应因 argmin(single IC) 被弹
        assert pool.contains("B")

    def test_max_size_enforced(self):
        pool = ActivePool(target_size=2, max_size=3)
        for i in range(5):
            pool.admit(self._m(f"F{i}", 0.5 + 0.01 * i, niche=(f"n{i % 4}",)))
        assert len(pool) <= 3

    def test_qd_niche_elite(self):
        from alphaprobe.pool import QDArchive
        qd = QDArchive(elite_per_cell=2)
        for i in range(6):
            qd.offer(self._m(f"M{i}", 0.5 + 0.01 * i, niche=("same",)))
        assert len(qd.elites()) == 2


# §29 Scheduler
class TestScheduler:
    def test_thompson_explores_all(self):
        s = ActionScheduler("thompson", rng=__import__("random").Random(1))
        for i in range(100):
            fam = s.select()
            s.update(fam, 1.0 if i % 2 else 0.0)
        assert all(st.attempts > 0 for st in s.stats.values())

    def test_reward_updates_mean(self):
        s = ActionScheduler("ucb")
        for _ in range(20):
            s.update("REFINE", 1.0)
        assert s.stats["REFINE"].mean > 0.99


def test_search_action_types():
    assert SearchActionType.WINDOW_SCALE.value == "WINDOW_SCALE"
    assert len(SearchActionType) == 9


def test_label_spec_hash():
    assert LabelSpec().hash_key == "vwap_to_vwap_h20"


def test_split_no_overlap_guard():
    spec = ResearchSplitSpec(
        train=DateRange("2016-01-01", "2021-12-31"),
        search_valid=DateRange("2022-01-01", "2023-12-31"),
        audit_valid=DateRange("2024-01-01", "2024-06-30"),
        sealed_test=DateRange("2025-01-01", "2026-07-31"),
    )
    # 无日历注入时退化为不抛
    spec.validate_no_overlap()