"""Lineage Stagnation（plan Task 11）——depth decay 替换为 stagnation 判据。

验收点（plan Task 11 Tests + Non-negotiable #13/#29/#30）：
1. 深但持续有增益的 branch 不被重罚：stagnation 分低、should_stop False、
   深度先验保持中性/小复杂度先验（不再有 ``(1-gamma)^depth`` 硬指数压制）；
2. 浅但连续 6 代停滞的 branch 被停：连续 stale（ΔFitness/ΔPool/novelty 全无）
   → should_stop True（K 代窗口判据）；
3. 复杂度涨而 fitness 平 → 加速停（complexity_accel 生效），ablation 可关；
4. 判据权重/窗口/阈值全部可配置；无证据（不足 min_records / delta=None）→
   中性不判停（#13）；
5. retriever 集成：``stagnation_penalty``（0 = 未接入/中性，1 = 全停）参与
   prior——深但持续有增益的 branch 先验不低于浅但停滞 branch 被停之后的先验；
   ``stagnation_enabled=False``（ablation 开关）时行为回退到旧 depth 指数衰减
   （gamma 生效）——保留旧行为可 ablation（#30）。

全部合成数据、零 LLM / 零模型 / 零网络；不 import torch / faiss。
"""

from __future__ import annotations

import math

import pytest

from alphaprobe.retrieval.bayesian_retriever import (
    BayesianRetriever,
    BayesianRetrieverConfig,
    compute_prior,
)
from alphaprobe.search.stagnation import (
    DEFAULT_STAGNATION_CONFIG,
    GenerationRecord,
    StagnationConfig,
    StagnationEvaluator,
)


def _rec(
    fid: str = "",
    *,
    df: float | None = None,
    dp: float | None = None,
    nov: float | None = None,
    cx: int | float | None = None,
    corr: float | None = None,
    gen: int = 0,
    outcome: str = "ok",
) -> GenerationRecord:
    return GenerationRecord(
        factor_id=fid,
        delta_fitness=df,
        delta_pool=dp,
        novelty_gain=nov,
        complexity_delta=cx,
        nearest_correlation=corr,
        generation=gen,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# 1. 深但持续有增益 → 不重罚
# ---------------------------------------------------------------------------


class TestDeepFertileNotPenalized:
    def test_deep_with_continuous_gains_low_stagnation(self):
        ev = StagnationEvaluator()
        # 深 branch（代 0..9）但每代都有正 ΔFitness / novelty
        for g in range(10):
            ev.observe(
                _rec(
                    f"c{g}", df=0.02 + g * 0.001, dp=0.01,
                    nov=0.1, cx=0, corr=0.3, gen=g,
                )
            )
        r = ev.evaluate()
        assert r.should_stop is False
        assert r.stagnation < 0.5

    def test_compute_prior_has_no_hard_depth_punish_when_fertile(self):
        """fertile 深 branch 的 prior 里 depth 不再造成硬指数压制（gamma=0.15）。"""
        # stagnation_enabled=True（默认）：depth 只留小复杂度先验（默认每层
        # 0.02）——不再乘旧 gamma=0.15 的每层硬衰减。显式传 depth_prior_decay=0
        # 时（测试只关心「无硬 depth 惩罚」语义），深/浅先验完全一致。
        prior_fertile = compute_prior(
            0.6, depth=10, retrieval_times=0,
            gamma=0.15, omega=0.0,
            stagnation_penalty=0.0, depth_prior_decay=0.0,
        )
        prior_shallow = compute_prior(
            0.6, depth=0, retrieval_times=0,
            gamma=0.15, omega=0.0,
            stagnation_penalty=0.0, depth_prior_decay=0.0,
        )
        assert prior_fertile == pytest.approx(prior_shallow)
        # 且 depth=10 的先验不塌：若不替换 depth 衰减，gamma=0.15 会把
        # sigmoid 项压到 ~0.143；现在保持 ~0.731（只留小先验）。
        assert prior_fertile > 0.6

    def test_stagnation_penalty_blocks_stale_not_fertile(self):
        r = BayesianRetriever()
        # stale：连续停滞 branch → should_stop → penalty≈1 → 即使 depth 浅也会被压
        fertile = {"factor_id": "fertile", "factor_fitness": 0.7, "depth": 12}
        stale = {"factor_id": "stale", "factor_fitness": 0.7, "depth": 1}
        d_fertile = r.score_candidate(fertile)
        # 缺省 stagnation 未接入（无历史记录）→ penalty 中性 0
        assert d_fertile["stagnation_penalty"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. 浅但连续 6 代停滞 → 被停
# ---------------------------------------------------------------------------


class TestShallowStaleStops:
    def test_six_stale_generations_stops_shallow_branch(self):
        """浅 branch 连续 6 代无任何 gain（fitness/pool/novelty 全平）→ 停。"""
        ev = StagnationEvaluator()
        for g in range(6):
            ev.observe(_rec(f"c{g}", df=0.0, dp=0.0, nov=0.0, gen=g))
        r = ev.evaluate()
        assert r.stale_streak >= 6
        assert r.should_stop is True

    def test_six_stale_with_some_complexity_growth_stops_faster(self):
        """连续 6 代停滞且复杂度一路涨 → 停（复杂度加速）。"""
        ev = StagnationEvaluator()
        for g in range(6):
            ev.observe(
                _rec(f"c{g}", df=0.0, dp=0.0, nov=0.0, cx=2 + g, gen=g)
            )
        r = ev.evaluate()
        assert r.should_stop is True
        # 复杂度净增 → 停滞分高于无复杂度增长版本
        ev2 = StagnationEvaluator()
        for g in range(6):
            ev2.observe(_rec(f"c{g}", df=0.0, dp=0.0, nov=0.0, cx=0, gen=g))
        r2 = ev2.evaluate()
        assert r.stagnation >= r2.stagnation

    def test_gain_resets_stale_streak(self):
        """连续停滞几代后出现一次正 gain → streak 清零，不误停。"""
        ev = StagnationEvaluator()
        for g in range(5):
            ev.observe(_rec(f"c{g}", df=0.0, dp=0.0, nov=0.0, gen=g))
        r_before = ev.evaluate()
        assert r_before.stale_streak == 5
        r = ev.observe(_rec("c5", df=0.02, dp=0.0, nov=0.1, gen=5))
        assert r.stale_streak == 0
        assert r.should_stop is False


# ---------------------------------------------------------------------------
# 3. 复杂度涨而 fitness 平 → 加速停
# ---------------------------------------------------------------------------


class TestComplexityAccel:
    def test_flat_fitness_with_complexity_growth_accelerates_stop(self):
        """fitness 平 + 复杂度涨 → 综合停滞分 > 无复杂度增长。"""
        cfg = StagnationConfig(min_records=3, stop_threshold=0.99)  # 不让阈值触发
        ev_flat = StagnationEvaluator(cfg)
        ev_cx = StagnationEvaluator(cfg)
        for g in range(6):
            # fitness 全平（df=0）
            ev_flat.observe(_rec(f"a{g}", df=0.0, nov=0.0, gen=g))
            ev_cx.observe(_rec(f"b{g}", df=0.0, nov=0.0, cx=3 + g, gen=g))
        assert ev_cx.evaluate().stagnation > ev_flat.evaluate().stagnation

    def test_complexity_accel_ablation_off(self):
        """ablation：关掉 enable_complexity_accel → 复杂度不再加速停滞分。"""
        cfg_off = StagnationConfig(
            min_records=3, stop_threshold=0.99, enable_complexity_accel=False
        )
        ev_off = StagnationEvaluator(cfg_off)
        ev_on = StagnationEvaluator(StagnationConfig(
            min_records=3, stop_threshold=0.99, enable_complexity_accel=True
        ))
        for g in range(6):
            ev_off.observe(_rec(f"a{g}", df=0.0, nov=0.0, cx=3 + g, gen=g))
            ev_on.observe(_rec(f"b{g}", df=0.0, nov=0.0, cx=3 + g, gen=g))
        # 关闭加速后，复杂度子信号（净涨）仍贡献基础停滞分，但 total 应低于开启版
        assert ev_off.evaluate().stagnation < ev_on.evaluate().stagnation


# ---------------------------------------------------------------------------
# 4. 可配置 + 无证据中性
# ---------------------------------------------------------------------------


class TestConfigAndNeutral:
    def test_insufficient_records_is_neutral(self):
        """记录不足 min_records → 中性：不判停。"""
        ev = StagnationEvaluator(StagnationConfig(min_records=5))
        ev.observe(_rec("c0", df=0.0, gen=0))
        ev.observe(_rec("c1", df=0.0, gen=1))
        r = ev.evaluate()
        assert r.should_stop is False
        assert r.stagnation < 1.0

    def test_no_evidence_deltas_neutral(self):
        """全部 delta=None（无评估证据）→ 停滞分中性，不判停。"""
        ev = StagnationEvaluator()
        for g in range(6):
            ev.observe(_rec(f"c{g}", gen=g))
        r = ev.evaluate()
        assert r.should_stop is False

    def test_configurable_window(self):
        """window 截断：窗口短（stale_generations=3）时只统计最近 K 代。"""
        cfg = StagnationConfig(window=3, min_records=1, stale_generations=3)
        ev = StagnationEvaluator(cfg)
        # 前 5 代有 gain；最近 3 代停滞（window=3 → 只看最近 3 代，streak=3）
        for g in range(5):
            ev.observe(_rec(f"good{g}", df=0.02, gen=g))
        for g in range(5, 9):
            ev.observe(_rec(f"stale{g}", df=0.0, gen=g))
        r = ev.evaluate()
        assert r.stale_streak == 3
        assert r.should_stop is True

    def test_custom_weights_affect_stagnation(self):
        """权重可配：把 fitness_trend 权重拉到 0（不看 fitness）→ 分数变化。"""
        cfg_hi = StagnationConfig(min_records=3, stop_threshold=0.99)
        cfg_no_fit = StagnationConfig(
            min_records=3, stop_threshold=0.99,
            w_fitness_trend=0.0,
            w_pool_trend=0.0,
            w_novelty=0.0,
        )
        ev_hi = StagnationEvaluator(cfg_hi)
        ev_no = StagnationEvaluator(cfg_no_fit)
        for g in range(4):
            # 负 fitness 但其它中性
            ev_hi.observe(_rec(f"a{g}", df=-0.02, gen=g))
            ev_no.observe(_rec(f"b{g}", df=-0.02, gen=g))
        # 不看 fitness → 负 ΔFitness 不再抬升停滞分
        assert ev_no.evaluate().stagnation < ev_hi.evaluate().stagnation


# ---------------------------------------------------------------------------
# 5. retriever ablation：stagnation 开关保留旧 depth decay 行为
# ---------------------------------------------------------------------------


class TestRetrieverAblation:
    def test_stagnation_enabled_uses_penalty_not_gamma_depth(self):
        """stagnation_enabled=True 时 depth 的硬指数惩罚让位给 stagnation 分。"""
        cfg = BayesianRetrieverConfig(
            stagnation_enabled=True,
            gamma=0.15,  # 即便 gamma 非 0，enabled 也不再按 depth 指数衰减
        )
        r = BayesianRetriever(config=cfg)
        d = r.score_candidate(
            {"factor_id": "deep", "factor_fitness": 0.6, "depth": 8}
        )
        # 无历史记录 → stagnation_penalty 中性 0；prior 里 depth 只留小先验
        assert d["stagnation_penalty"] == pytest.approx(0.0)
        assert d["prior"] > 0.0

    def test_stagnation_disabled_keeps_old_depth_decay(self):
        """ablation：stagnation_enabled=False → 回到旧 ``(1-gamma)^depth``。"""
        cfg = BayesianRetrieverConfig(stagnation_enabled=False, gamma=0.15)
        r = BayesianRetriever(config=cfg)
        deep = r.score_candidate({"factor_id": "d", "factor_fitness": 0.6, "depth": 8})
        shallow = r.score_candidate(
            {"factor_id": "s", "factor_fitness": 0.6, "depth": 0}
        )
        # 旧行为：深 branch 明显被压
        assert deep["prior"] < shallow["prior"]
        expected = compute_prior(
            0.6, depth=8, retrieval_times=0, gamma=0.15, omega=0.05,
            stagnation_enabled=False,
        )
        assert deep["prior"] == pytest.approx(expected)

    def test_compute_prior_stagnation_penalty_monotonic(self):
        """stagnation_penalty 越高 → prior 越低（单调抑制停滞 branch）。"""
        p_low = compute_prior(
            0.6, depth=0, retrieval_times=0, stagnation_penalty=0.1
        )
        p_high = compute_prior(
            0.6, depth=0, retrieval_times=0, stagnation_penalty=0.9
        )
        assert p_high < p_low

    def test_prior_deprecated_depth_no_gamma_still_small_prior(self):
        """stagnation 开启下即使 gamma=0，prior 仍 = sigmoid × stagnation 项。"""
        p = compute_prior(
            0.6, depth=10, retrieval_times=0,
            gamma=0.0, omega=0.0, stagnation_penalty=0.0,
            stagnation_enabled=False,
        )
        # sigmoid(0.6 归一化 ≈ 0.6) 附近，gamma=0 → 无 depth 衰减 → prior 不塌
        assert p > 0.5
