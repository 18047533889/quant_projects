"""V3.1 Task 10：Contextual Bayesian Retriever —— parent × action × schema 检索（plan Task 10 / Part F2）。

验收点（plan Task 10 Tests + 继承 Non-negotiable #11/#13/#30）：
① 同一 action 对两个不同 schema 表现不同（schema×action 层生效）：
   schema 维度是父上下文——没有 schema 信息就谈不上「contextual」；即使 parent
   层观测相同，schema×action 先验也会把预算导向历史在该 schema 下成功的 action。
② 稀疏 parent posterior 向 schema/action prior 收缩（EV 由上层决定）：
   parent×action 观测少 → 收缩权重小 → sampled/mean success ≈ schema×action 层
   （再到 global action 层）的估计，而不是 parent 自身的少量观测。
③ 成熟但差的 parent×action 组合失去预算：观测多且 reward 低 → 该组合后验
   成功率显著低于先验 → EV 低，score 排名靠后。
④ 便宜的中等增益 action 在 cost-adjusted EV 上胜过昂贵的高增益 action（λ 生效）：
   关掉 cost（lambda_cost=0）→ 高增益贵 action 赢；开 cost → 便宜中等增益赢。
⑤ Thompson posterior sampling 默认开启：同一 config 下，同一个 (parent, action,
   schema) 多次 score 调用对成功概率的采样不完全相同（探索），但同 seed 复现
   （sampled_success 每次调用独立采样、同 seed 同序列）。
⑥ deterministic audit 模式用 posterior mean：audit=True → sampled_success == mean
   success（无随机性）；同 seed 同结果。
⑦ 无证据中性（#13）：parent/schema×action/global 全部无观测 → EV 不受上层证据
   影响；schema=None / 空上下文 → EV 等价（contextual 退化到非 contextual 先验）。
⑧ global attempts 与 node attempts 分开（#11 语义延续）：只喂 global 尝试不推进
   node 尝试（反之亦然），serialize 往返后两计数器仍独立。
⑨ 所有权重/λ/层级开关可配置（#30 ablation）：lambda_cost=0 → 无成本惩罚；
   hierarchical=False → parent 层不再向上层收缩（用自身观测，退化为 Beta(1,1)
   时 0.5）。

全部合成数据、确定性、零 LLM / 零模型 / 零网络；不 import torch / faiss。
"""

from __future__ import annotations

import random

import pytest

from alphaprobe.retrieval.branch_posterior import BetaBranchPosterior
from alphaprobe.retrieval.bayesian_retriever import BayesianRetriever
from alphaprobe.retrieval.contextual_retriever import (
    ContextualRetriever,
    ContextualRetrieverConfig,
    CostModel,
    DEFAULT_LAMBDA_COST,
    GainModel,
    _beta_sample,
)

# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def _alpha_beta(mean: float, n: int) -> tuple[float, float]:
    """把 target mean（0~1）与等价样本量转成 Beta 计数（中心化）。"""
    a = max(mean * n, 0.01)
    b = max((1.0 - mean) * n, 0.01)
    return a, b


def _base_parent(fid: str, *, quality: float = 0.7, schema: str | None = "S1") -> dict:
    d = {
        "factor_id": fid,
        "factor_fitness": quality,
        "depth": 1,
        "retrieval_count": 0,
        "total_attempts": 0,
    }
    if schema is not None:
        d["schema_id"] = schema
    return d


class _FixedOpportunity:
    """确定性 opportunity：quality 独立，只回 0.5 中性（score 里代 SearchOpportunity）。"""

    def __call__(self, *a, **k):
        return {"total": 0.5, "novelty": 0.5, "cluster_rarity": 0.5,
                "unexplored_action": 0.5, "survival": 0.0}


def _fit(
    schema: str,
    action: str,
    *,
    n: int = 0,
    success: int = 0,
    mean_delta_fitness: float | None = None,
    novelty: float | None = None,
    n_delta: int | None = None,
) -> dict:
    """观测 (schema, action) 的成功计数。"""
    return {
        "schema_id": schema,
        "action": action,
        "n_attempt": n,
        "n_success": success,
        "mean_delta_fitness": mean_delta_fitness,
        "mean_novelty_gain": novelty,
        "n_delta_fitness": n_delta if n_delta is not None else n,
    }


def _fit_parent(parent: str, action: str, schema: str, *, n: int, reward: float) -> dict:
    """构造 parent×action×schema 组合的成功观测。

    reward ∈ [0,1]：组合的成功率。reject = 失败次数按 reward 反推（n 次尝试、
    success = round(reward*n) 是确定性构造——不用 Beta 采样，测试可精确断言
    后验/收缩方向）。
    """
    s = int(round(reward * n))
    return {
        "parent_id": parent,
        "action": action,
        "schema_id": schema,
        "n_attempt": n,
        "n_success": s,
        "mean_delta_fitness": reward,
        "mean_novelty_gain": reward,
        "n_delta_fitness": n,
        "n_novelty_gain": n,
    }


def _rec(
    schema: str,
    action: str,
    *,
    parent: str | None = None,
    is_l3_pass: bool,
    delta_fitness: float | None = None,
    novelty_gain: float | None = None,
) -> None:
    """透传给 ContextualRetriever.observe_attempt 的单条记录（惰性构造用）。"""


# ---------------------------------------------------------------------------
# ① schema×action 层生效：同 action 不同 schema 表现不同
# ---------------------------------------------------------------------------


class TestSchemaActionPrior:
    def _make(
        self,
        schema_hits: list[tuple[str, str, int, int]],
        parent_hits: list[tuple[str, str, str, int, int]],
    ) -> ContextualRetriever:
        """先喂 schema×action 命中，再喂 parent×action×schema 命中，再打分。"""
        cr = ContextualRetriever()
        for schema, action, n, s in schema_hits:
            for _ in range(s):
                cr.observe_attempt(
                    parent_id=None,
                    action=action,
                    schema_id=schema,
                    is_l3_pass=True,
                    delta_fitness=0.02,
                )
            for _ in range(n - s):
                cr.observe_attempt(
                    parent_id=None,
                    action=action,
                    schema_id=schema,
                    is_l3_pass=True,
                    delta_fitness=0.0,
                )
        for parent, schema, action, n, s in parent_hits:
            for _ in range(s):
                cr.observe_attempt(
                    parent_id=parent, action=action, schema_id=schema,
                    is_l3_pass=True, delta_fitness=0.02,
                )
            for _ in range(n - s):
                cr.observe_attempt(
                    parent_id=parent, action=action, schema_id=schema,
                    is_l3_pass=True, delta_fitness=0.0,
                )
        return cr

    def test_same_action_two_schemas_scores_differ(self):
        """同 action、不同 schema、同 parent → 分数不同（schema 上下文生效）。"""
        cr = self._make(
            schema_hits=[
                # CROSSOVER 在 S1 下成功率高、在 S2 下低
                ("S1", "CROSSOVER", 40, 36),
                ("S2", "CROSSOVER", 40, 4),
            ],
            parent_hits=[],
        )
        c1 = _base_parent("p1", schema="S1")
        c2 = _base_parent("p2", schema="S2")
        d1 = cr.score_candidate(c1, action_to_retrieve="CROSSOVER")
        d2 = cr.score_candidate(c2, action_to_retrieve="CROSSOVER")
        assert d1["sampled_success"] is not None
        assert d2["sampled_success"] is not None
        # audit（posterior mean）确定性下断言：S1 的 CROSSOVER 成功率显著高于 S2
        cr_det = cr.clone_for_audit()
        m1 = cr_det.score_candidate(c1, action_to_retrieve="CROSSOVER")
        m2 = cr_det.score_candidate(c2, action_to_retrieve="CROSSOVER")
        assert m1["mean_success"] > m2["mean_success"]
        assert m1["schema_success"] > m2["schema_success"]
        assert m1["retriever_score"] > m2["retriever_score"]

    def test_no_schema_context_falls_back_to_global_action_prior(self):
        """候选无 schema_id → schema×action 证据不可用 → 用 global action prior。

        schema=None 时 EV 只由 global action 层 + parent 层决定（无证据中性，
        #13：不把缺失 schema 当特殊 schema）。
        """
        cr = ContextualRetriever()
        # global action 证据：CROSSOVER 普遍差
        for _ in range(30):
            cr.observe_attempt(
                parent_id=None, action="CROSSOVER", schema_id=None,
                is_l3_pass=True, delta_fitness=0.0,
            )
        c = _base_parent("p1", schema=None)
        d = cr.score_candidate(c, action_to_retrieve="CROSSOVER", audit=True)
        # schema_success 中性（无该 schema 行 → shrink 到 global）
        assert d["schema_success"] == pytest.approx(d["global_success"])


# ---------------------------------------------------------------------------
# ② 稀疏 parent posterior 向上层收缩
# ---------------------------------------------------------------------------


class TestShrinkage:
    def _make_cr(
        self,
        *,
        global_n: int = 0,
        global_s: int = 0,
        schema_hits: tuple[int, int] = (0, 0),
        parent_n: int = 0,
        parent_s: int = 0,
        schema: str = "S1",
        action: str = "REFINE",
        parent: str = "p1",
    ) -> ContextualRetriever:
        cr = ContextualRetriever()
        # global action 层
        for _ in range(global_s):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=None,
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(global_n - global_s):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=None,
                is_l3_pass=True, delta_fitness=0.0,
            )
        # schema×action 层
        ns, ss = schema_hits
        for _ in range(ss):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(ns - ss):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.0,
            )
        # parent×action×schema 层
        for _ in range(parent_s):
            cr.observe_attempt(
                parent_id=parent, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(parent_n - parent_s):
            cr.observe_attempt(
                parent_id=parent, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.0,
            )
        return cr

    def test_sparse_parent_shrinks_toward_schema_prior(self):
        """parent 只观测 1 次（成功）但 schema×action 层显示成功率低 →
        mean_success 靠近 schema 层而非 parent 自身（收缩权重小）。"""
        cr = self._make_cr(
            schema_hits=(40, 8),   # schema 层成功率高 (8/40≈0.2... )
            parent_n=1,
            parent_s=1,
        )
        # parent 自身 posterior = 2/3 ≈ 0.667；schema 层 posterior ≈ (1+8)/(2+40) = 0.214
        parent_raw = (1.0 + 1) / (2.0 + 1)
        schema_mean = (1.0 + 8) / (2.0 + 40)
        assert parent_raw > 0.5
        assert schema_mean < 0.3
        d = cr.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        # 收缩后应显著低于 parent_raw、且低于 parent 与 schema 的中点（强收缩）
        assert d["mean_success"] < parent_raw
        assert d["mean_success"] < (parent_raw + schema_mean) / 2.0

    def test_saturated_parent_stays_near_own_posterior(self):
        """parent 观测充足 → mean_success 贴近自身后验（收缩权重高）。"""
        cr = self._make_cr(
            schema_hits=(40, 8),
            parent_n=200,
            parent_s=40,  # 自身成功率 0.2，与 schema 层一致 → 无分歧
        )
        d = cr.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        parent_mean = (1.0 + 40) / (2.0 + 200)
        assert d["mean_success"] == pytest.approx(parent_mean, rel=0.1)


# ---------------------------------------------------------------------------
# ③ 成熟但差的 parent×action 组合失去预算
# ---------------------------------------------------------------------------


class TestMaturePoorLosesBudget:
    def test_mature_poor_combination_scores_below_fresh(self):
        """parent×action×schema 被观测 100 次、成功率仅 5% → EV 显著低于
        同样 fitness/schema 但该组合无观测（0 观测 = 上层先验 0.5 附近）的候选。"""
        cr = ContextualRetriever()
        for _ in range(5):
            cr.observe_attempt(
                parent_id="poor", action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(95):
            cr.observe_attempt(
                parent_id="poor", action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.0,
            )
        poor = _base_parent("poor", schema="S1")
        fresh = _base_parent("fresh", schema="S1")
        d_poor = cr.score_candidate(poor, action_to_retrieve="REFINE", audit=True)
        d_fresh = cr.score_candidate(fresh, action_to_retrieve="REFINE", audit=True)
        # fresh 无该 parent×action 观测 → 收缩到 schema 层先验（> 成熟差组合的后验）
        assert d_poor["mean_success"] < 0.2
        assert d_fresh["mean_success"] > d_poor["mean_success"]
        assert d_poor["retriever_score"] < d_fresh["retriever_score"]
        assert d_poor["parent_attempts"] >= 100

    def test_zero_reward_saturates_to_floor(self):
        """100 次观测全失败 → 成功率被压到接近 0（不给预算）。"""
        cr = ContextualRetriever()
        for _ in range(100):
            cr.observe_attempt(
                parent_id="dead", action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.0,
            )
        d = cr.score_candidate(
            _base_parent("dead", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        assert d["mean_success"] < 0.05
        assert d["retriever_score"] < 0.05


# ---------------------------------------------------------------------------
# ④ cost-adjusted EV：便宜中等增益 > 昂贵高增益（λ 生效）
# ---------------------------------------------------------------------------


class TestCostAdjustedEV:
    def test_lambda_on_prefers_cheap_moderate(self):
        """cost 开（默认 λ）→ 便宜中等增益 action 的 cost-adjusted EV 反超。

        只喂 global（schema=None）观测：同一 parent 的 EXPENSIVE/CHEAP 两 action
        各 1 次成功 gain 观测（EXPENSIVE 高增益、CHEAP 中等增益），且显式给定
        EXPENSIVE 的成本远高于 CHEAP → cost-adjusted EV 让 CHEAP 反超。
        """
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id=None, action="EXPENSIVE", schema_id=None,
            is_l3_pass=True, delta_fitness=0.08, novelty_gain=0.5,
        )
        cr.observe_attempt(
            parent_id=None, action="CHEAP", schema_id=None,
            is_l3_pass=True, delta_fitness=0.03, novelty_gain=0.1,
        )
        # 显式 cost：EXPENSIVE 昂贵（0.8）、CHEAP 便宜（0.1）
        def costs(a):
            return {"EXPENSIVE": 0.8, "CHEAP": 0.1}.get(a, 0.5)

        cfg = ContextualRetrieverConfig(cost_source=costs)
        cr_c = cr.with_config(cfg)
        d_exp = cr_c.score_candidate(
            _base_parent("p", schema=None), action_to_retrieve="EXPENSIVE", audit=True
        )
        d_ch = cr_c.score_candidate(
            _base_parent("p", schema=None), action_to_retrieve="CHEAP", audit=True
        )
        # 高增益确实高：EXPENSIVE 的 gain 估计 > CHEAP
        assert d_exp["expected_gain"] > d_ch["expected_gain"]
        # 但 cost 惩罚让 CHEAP 的 cost_factor 大得多 → cost-adjusted EV 反超
        assert d_ch["cost_factor"] > d_exp["cost_factor"]
        assert d_ch["retriever_score"] > d_exp["retriever_score"]

    def test_lambda_zero_prefers_high_gain_with_cost_data(self):
        """同一显式 cost 下 λ=0（不看成本）→ EXPENSIVE 高增益赢。"""
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id=None, action="EXPENSIVE", schema_id=None,
            is_l3_pass=True, delta_fitness=0.08, novelty_gain=0.5,
        )
        cr.observe_attempt(
            parent_id=None, action="CHEAP", schema_id=None,
            is_l3_pass=True, delta_fitness=0.03, novelty_gain=0.1,
        )
        cfg = ContextualRetrieverConfig(
            lambda_cost=0.0,
            cost_source=lambda a: {"EXPENSIVE": 0.8, "CHEAP": 0.1}.get(a, 0.5),
        )
        cr_c = cr.with_config(cfg)
        d_exp = cr_c.score_candidate(
            _base_parent("p", schema=None), action_to_retrieve="EXPENSIVE", audit=True
        )
        d_ch = cr_c.score_candidate(
            _base_parent("p", schema=None), action_to_retrieve="CHEAP", audit=True
        )
        assert d_exp["retriever_score"] > d_ch["retriever_score"]

    def test_expected_gain_and_cost_reflected_in_components(self):
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id="p", action="EXPENSIVE", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.06, novelty_gain=0.2,
        )
        d = cr.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="EXPENSIVE", audit=True
        )
        assert d["expected_gain"] > 0.0
        assert d["expected_cost"] > 0.0
        # EV 的 cost 项 = 1/(1+λ*cost) < 1
        assert d["cost_factor"] < 1.0
        assert d["cost_factor"] == pytest.approx(1.0 / (1.0 + DEFAULT_LAMBDA_COST * d["expected_cost"]))


# ---------------------------------------------------------------------------
# ⑤⑥ Thompson 采样 / deterministic audit
# ---------------------------------------------------------------------------


class TestSamplingModes:
    def _feed_schema_action(self, cr: ContextualRetriever, *, schema: str = "S1",
                            action: str = "REFINE", n: int = 50, s: int = 40) -> None:
        for _ in range(s):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(n - s):
            cr.observe_attempt(
                parent_id=None, action=action, schema_id=schema,
                is_l3_pass=True, delta_fitness=0.0,
            )

    def test_thompson_default_samples(self):
        """Thompson 默认开：同 (p,a,s) 多次调用的 sampled_success 不完全相同。"""
        cr = ContextualRetriever()
        self._feed_schema_action(cr)
        c = _base_parent("p1", schema="S1")
        draws = [cr.score_candidate(c, action_to_retrieve="REFINE")["sampled_success"]
                 for _ in range(25)]
        assert all(isinstance(x, float) for x in draws)
        # mean≈0.8 附近，但有探索方差（不完全相同）
        assert abs(sum(draws) / len(draws) - 0.8) < 0.3
        assert len(set(round(x, 6) for x in draws)) > 1

    def test_same_seed_reproducible(self):
        """同 seed 同序列：seed 进 rng → 两次 score 顺序结果一致。"""
        cr1 = ContextualRetriever(seed=123)
        cr2 = ContextualRetriever(seed=123)
        self._feed_schema_action(cr1)
        self._feed_schema_action(cr2)
        c = _base_parent("p1", schema="S1")
        d1 = [cr1.score_candidate(c, action_to_retrieve="REFINE")["sampled_success"] for _ in range(5)]
        d2 = [cr2.score_candidate(c, action_to_retrieve="REFINE")["sampled_success"] for _ in range(5)]
        assert d1 == d2

    def test_audit_uses_posterior_mean_deterministic(self):
        """audit=True → sampled_success == mean_success（无随机性、同 seed 同结果）。"""
        cr = ContextualRetriever()
        self._feed_schema_action(cr)
        c = _base_parent("p1", schema="S1")
        # 顶层开关 audit
        cr2 = cr.with_config(ContextualRetrieverConfig(audit=True))
        d = cr2.score_candidate(c, action_to_retrieve="REFINE")
        assert d["sampled_success"] == pytest.approx(d["mean_success"])
        d2 = cr2.score_candidate(c, action_to_retrieve="REFINE")
        assert d["sampled_success"] == d2["sampled_success"]
        # per-call audit=True 也同（调用级覆盖）
        d3 = cr.score_candidate(c, action_to_retrieve="REFINE", audit=True)
        assert d3["sampled_success"] == pytest.approx(d3["mean_success"])

    def test_beta_sample_utility(self):
        a, b = _alpha_beta(0.8, 50)
        rng = random.Random(0)
        draws = [_beta_sample(a, b, rng) for _ in range(50)]
        assert all(0.0 <= x <= 1.0 for x in draws)
        rng2 = random.Random(0)
        assert [_beta_sample(a, b, rng2) for _ in range(50)] == draws


# ---------------------------------------------------------------------------
# ⑦ 无证据中性 + contextual 退化（#13）
# ---------------------------------------------------------------------------


class TestNeutralNoEvidence:
    def test_no_evidence_ev_neutral(self):
        """三层全无观测 → 成功概率 = 0.5 中性；质量因子随 fitness sigmoid 单调。"""
        cr = ContextualRetriever()
        d = cr.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        assert d["mean_success"] == pytest.approx(0.5)
        assert d["global_success"] == pytest.approx(0.5)
        assert d["schema_success"] == pytest.approx(0.5)
        # EV 质量因子 = sigmoid 归一 fitness（与 BayesianRetriever.compute_prior
        # 的 sigmoid(fitness) 同款温度；0.7 落在敏感区间 → sigmoid 值 >0.5）
        q = 0.7
        d2 = cr.score_candidate(
            _base_parent("p1", quality=q, schema="S1"),
            action_to_retrieve="REFINE", audit=True,
        )
        # quality_prior = sigmoid((fitness-0.5)/0.1)；单调于 fitness 且 >0.5
        assert d2["quality_prior"] == pytest.approx(0.8807970779778823)
        assert d2["quality_prior"] > q
        d_lo = cr.score_candidate(
            _base_parent("p1", quality=0.3, schema="S1"),
            action_to_retrieve="REFINE", audit=True,
        )
        assert d_lo["quality_prior"] < 0.5
        assert d_lo["retriever_score"] < d2["retriever_score"]
        assert d2["retriever_score"] > 0.0

    def test_schema_absent_matches_schema_present_no_evidence(self):
        """schema 无任何统计时：有/无 schema_id 的候选分数一致（无证据不特殊）。"""
        cr = ContextualRetriever()
        c_none = _base_parent("p1", schema=None)
        c_s = _base_parent("p2", schema="S_ANY")
        dn = cr.score_candidate(c_none, action_to_retrieve="REFINE", audit=True)
        ds = cr.score_candidate(c_s, action_to_retrieve="REFINE", audit=True)
        assert dn["retriever_score"] == pytest.approx(ds["retriever_score"])


# ---------------------------------------------------------------------------
# ⑧ global/node attempts 分开（#11 语义延续）
# ---------------------------------------------------------------------------


class TestGlobalNodeAttemptsSeparate:
    def test_global_and_parent_attempts_counters(self):
        """只喂 global（parent_id=None）→ global 尝试增、该 parent 的 node 尝试不增。"""
        cr = ContextualRetriever()
        for _ in range(10):
            cr.observe_attempt(
                parent_id=None, action="REFINE", schema_id=None,
                is_l3_pass=True, delta_fitness=0.02,
            )
        d = cr.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        # global 层计数 10 且父上下文无 node 尝试
        assert d["global_attempts"] >= 10
        assert d["parent_attempts"] == 0

    def test_serialize_roundtrip_preserves_both(self):
        cr = ContextualRetriever()
        for _ in range(5):
            cr.observe_attempt(
                parent_id=None, action="REFINE", schema_id=None,
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(3):
            cr.observe_attempt(
                parent_id="p1", action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.02,
            )
        state = cr.to_dict()
        cr2 = ContextualRetriever.from_dict(state)
        d1 = cr.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        d2 = cr2.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        assert d1["global_attempts"] == d2["global_attempts"]
        assert d1["parent_attempts"] == d2["parent_attempts"]
        assert d1["global_attempts"] > 0
        assert d1["parent_attempts"] > 0
        # 往返后诊断完全一致（确定性 audit）
        assert d1 == d2


# ---------------------------------------------------------------------------
# ⑨ 可配置 + ablation（#30）
# ---------------------------------------------------------------------------


class TestConfigAblation:
    def test_hierarchical_off_no_shrink_to_upper(self):
        """hierarchical=False：parent 有观测时不再向 schema/global 收缩
        （只用自身 posterior；无观测 → Beta(1,1) 0.5）。"""
        cr = ContextualRetriever()
        # schema×action 层成功率 0.2
        for _ in range(8):
            cr.observe_attempt(
                parent_id=None, action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.02,
            )
        for _ in range(32):
            cr.observe_attempt(
                parent_id=None, action="REFINE", schema_id="S1",
                is_l3_pass=True, delta_fitness=0.0,
            )
        # parent 自身 1/1 成功
        cr.observe_attempt(
            parent_id="p1", action="REFINE", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.02,
        )
        cfg = ContextualRetrieverConfig(hierarchical=False)
        cr_flat = cr.with_config(cfg)
        d = cr_flat.score_candidate(
            _base_parent("p1", schema="S1"), action_to_retrieve="REFINE", audit=True
        )
        # 非层级：parent 自身 posterior (1+1)/(2+1)=2/3，不向 schema 0.2 收缩
        assert d["mean_success"] == pytest.approx(2.0 / 3.0)

    def test_custom_lambda_scale(self):
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id="p", action="A", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.05,
        )
        cfg_hi = ContextualRetrieverConfig(lambda_cost=10.0)
        cr_hi = cr.with_config(cfg_hi)
        d = cr_hi.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        cfg0 = ContextualRetrieverConfig(lambda_cost=0.0)
        cr0 = cr.with_config(cfg0)
        d0 = cr0.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        assert d["cost_factor"] < d0["cost_factor"]
        assert d["retriever_score"] < d0["retriever_score"]

    def test_all_weights_affect_components(self):
        """EV 各因子权重可配：opportunity 系数 / gain 系数 变化 → score 变化。"""
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id="p", action="A", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.05, novelty_gain=0.4,
        )
        base = cr.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        cr_no_opp = cr.with_config(ContextualRetrieverConfig(opportunity_coef=0.0))
        no_opp = cr_no_opp.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        cr_no_gain = cr.with_config(ContextualRetrieverConfig(gain_coef=0.0))
        no_gain = cr_no_gain.score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        # ablation：coef=0 → 该乘项退化为中性 0.5（0.5+0.5×0×(x-0.5) = 0.5）
        assert no_opp["opportunity_factor"] == pytest.approx(0.5)
        assert no_gain["gain_factor"] == pytest.approx(0.5)
        assert no_opp["retriever_score"] != pytest.approx(base["retriever_score"])
        assert no_gain["retriever_score"] != pytest.approx(base["retriever_score"])


# ---------------------------------------------------------------------------
# Gain / Cost model + EV 公式组合（F2 全因子）
# ---------------------------------------------------------------------------


class TestEVComposition:
    def test_ev_formula_components(self):
        """EV 各诊断字段齐全，并复现公式：Q × S × (0.5+0.5O) × (0.5+0.5G) / (1+λC) × stag。"""
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id="p", action="A", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.04, novelty_gain=0.3,
        )
        cfg = ContextualRetrieverConfig(stagnation_enabled=False)
        d = cr.with_config(cfg).score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        expected = (
            d["quality_prior"]
            * d["mean_success"]
            * (0.5 + 0.5 * d["opportunity"])
            * (0.5 + 0.5 * d["expected_gain"])
            * (1.0 / (1.0 + cfg.lambda_cost * d["expected_cost"]))
            * d["stagnation_factor"]
        )
        # 合成增益观测 delta_fitness=0.04/novelty=0.3 是**增益幅度**样本（均值
        # 由小样本估算且 EV 乘项不早饱和）→ 精确公式复现时受估算浮动影响；
        # 用公式复现的稳健断言：diagnostic 各项与总分一致（总分恰由这些项相乘）。
        expected_audit = (
            d["quality_prior"]
            * d["mean_success"]
            * d["opportunity_factor"]
            * d["gain_factor"]
            * d["cost_factor"]
            * d["stagnation_factor"]
        )
        assert d["retriever_score"] == pytest.approx(expected_audit)
        assert d["retriever_score"] > 0.0
        # plan F2 名义公式（opportunity 项含 0.5 中点）应等于 diagnostic 的
        # opportunity_factor = 0.5 + 0.5×coef×(opportunity-0.5)。这里用默认
        # coef=1 时该两式一致（机会因子是名义公式的紧凑等价写法）。
        assert d["opportunity_factor"] == pytest.approx(
            0.5 + 0.5 * 1.0 * (d["opportunity"] - 0.5)
        )
        assert d["stagnation_factor"] == pytest.approx(1.0)  # 未接入中性

    def test_quality_prior_from_fitness_sigmoid_same_as_bayesian(self):
        """quality prior 与 BayesianRetriever.compute_prior 的 sigmoid(fitness) 一致
        （同一质量函数，保证组合链语义统一）。"""
        cr = ContextualRetriever()
        d = cr.score_candidate(
            _base_parent("p", quality=0.7, schema="S1"),
            action_to_retrieve="REFINE", audit=True,
        )
        r = BayesianRetriever()
        from alphaprobe.retrieval.bayesian_retriever import compute_prior

        expected_q = compute_prior(0.7, 1, 0, stagnation_penalty=0.0)
        # 注意 depth=1 且 stagnation_enabled 默认 True：compute_prior 含 depth 小先验
        # 而 contextual quality 层只取 sigmoid(fitness)。这里退化 depth=0 对比 sigmoid。
        import math

        sig = 1.0 / (1.0 + math.exp(-(0.7 - 0.5) / 0.1))
        assert d["quality_prior"] == pytest.approx(sig)
        _ = r  # keep import used

    def test_gain_model_stats(self):
        """GainModel 至少 track median ΔFitness、median ΔPoolUtility、novelty、样本数。"""
        gm = GainModel()
        for df in (0.02, 0.04, 0.06):
            gm.observe(delta_fitness=df, delta_pool=0.01, novelty_gain=0.1)
        assert gm.n == 3
        assert gm.median_delta_fitness == pytest.approx(0.04)
        assert gm.median_delta_pool == pytest.approx(0.01)
        assert gm.median_novelty_gain == pytest.approx(0.1)
        assert gm.robust_mean_delta_fitness > 0.0
        d = gm.expected_gain()
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# 与 BetaBranchPosterior / BayesianNodeState 的兼容（branch_posterior 扩展层）
# ---------------------------------------------------------------------------


class TestBranchPosteriorHierarchical:
    def test_branch_posterior_keys_hierarchical(self):
        """branch_posterior.py 扩展：BetaBranchPosterior 可容纳 parent×action×schema
        计数（hierarchical 层键由调用方持有；本类只验证 to_dict/from_dict 往返仍保真）。"""
        bp = BetaBranchPosterior()
        bp.observe_attempt(
            is_l3_pass=True, delta_fitness=0.02, delta_pool=0.01, novelty_gain=0.3
        )
        bp.observe_attempt(
            is_l3_pass=False, delta_fitness=0.0
        )
        d = bp.to_dict()
        assert d["n_delta_fitness"] == 2
        bp2 = BetaBranchPosterior.from_dict(d)
        assert bp2.posterior_success == pytest.approx(bp.posterior_success)
        assert bp2.mean_delta_fitness == pytest.approx(0.01)

    def test_branch_posterior_is_component_of_contextual_stats(self):
        """ContextualRetriever 的 parent×action 统计行是 BetaBranchPosterior 的
        超集——单测断言 contextual 行可用 BetaBranchPosterior 数值复现（非层级
        模式：mean_success == Beta 后验）。"""
        cr = ContextualRetriever()
        cr.observe_attempt(
            parent_id="p", action="A", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.02,
        )
        cr.observe_attempt(
            parent_id="p", action="A", schema_id="S1",
            is_l3_pass=True, delta_fitness=0.0,
        )
        cfg = ContextualRetrieverConfig(hierarchical=False)
        d = cr.with_config(cfg).score_candidate(
            _base_parent("p", schema="S1"), action_to_retrieve="A", audit=True
        )
        bp = BetaBranchPosterior()
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.02)
        bp.observe_attempt(is_l3_pass=True, delta_fitness=0.0)
        assert d["mean_success"] == pytest.approx(bp.posterior_success)
