"""plan Task 17：Surrogate + EVI multi-fidelity promotion 测试。

覆盖 plan Task 17 Tests + Non-negotiable #23 / #30：
1. 无已训练 surrogate → 确定性 fallback（回到现有 funnel 阈值语义，不瞎猜）；
2. 训练数据来源约束：#23 sealed Test 标签绝不可作为 surrogate 目标
   （version / segment 标记校验，违反即 fail-closed
   :class:`SealedTestLabelError`）；
3. 高 EVI / 低成本 candidate 在 promotion 建议序上胜过低概率 / 高成本
   candidate（cost 进 EV 而非仅日志）；
4. 等增益下贵 action 的 cost-adjusted EV 更低；
5. 可注入 sklearn 兼容模型（不含 sklearn：用确定性双参数概率 stub），
   训练后 P(L3) 反映历史（高频成功 action 概率上升）；
6. EVI/cost 全开关（#30）：enabled=False / use_model=False →
   recommend 恒确定性 fallback；
7. 模型缺 predict_proba → fail-closed ModelNotProbabilisticError（不静默近似）。

全部合成数据、零 LLM / 零模型训练 / 零网络；不 import torch / faiss /
lightgbm / xgboost / sklearn。
"""

from __future__ import annotations

import pytest

from alphaprobe.surrogate.features import (
    DefaultFeatureExtractor,
    ExtractionContext,
    SealedTestLabelError,
)
from alphaprobe.surrogate.promotion import (
    MIN_HISTORY_FOR_FIT,
    ModelNotProbabilisticError,
    PromotionSurrogate,
    PromotionSurrogateConfig,
    PromotionSuggestion,
    ThresholdPredictor,
)


def ctx(
    factor_id: str = "",
    *,
    formula: str = "rank(ts_mean(close, 5))",
    passed: bool | None = None,
    cost: float | None = None,
    qe_seconds: float | None = None,
    fe_seconds: float | None = None,
    success: float | None = None,
    parent_fitness: float | None = None,
    param_family: str = "",
    cluster: str = "",
    schema: str = "",
    logic: str = "",
    mb: dict | None = None,
) -> ExtractionContext:
    metric = dict(mb or {})
    if passed is not None and "passed" not in metric:
        metric["passed"] = passed
    return ExtractionContext(
        factor_id=factor_id,
        formula=formula,
        metric_bundle=metric,
        estimated_cost=cost,
        estimated_qe_seconds=qe_seconds,
        estimated_fe_seconds=fe_seconds,
        action_success_rate=success,
        parent_fitness=parent_fitness,
        parameter_family_id=param_family,
        cluster_id=cluster,
        schema_id=schema,
        logic_id=logic,
    )


class DeterministicTwoParamModel:
    """确定性双参数概率 stub（等价 sklearn 兼容 shape，零外部依赖）。

    P(L3) = 0.2 + 0.8 × sigmoid(margin)；margin = 2×(feature[19]-0.5)。
    feature[19] = parent_fitness。fit 只记录 y（不训练权重），predict_proba
    返回 [[1-p, p]]。这足以验证「训练后概率反映历史成功样本」而无需任何
    真实 ML 库。
    """

    def __init__(self) -> None:
        self.fitted = False
        self.n_pos = 0
        self.n_neg = 0

    def fit(self, X, y):
        self.fitted = True
        ys = list(y)
        self.n_pos = sum(1 for v in ys if int(v) > 0.5)
        self.n_neg = len(ys) - self.n_pos
        return self

    def predict_proba(self, X):
        import math

        rows = []
        for row in X:
            parent_fit = float(row[19]) if len(row) > 19 else 0.0
            # 历史成功率高 → 正向偏置（后验移动可观测）
            bias = 0.3 if self.n_pos >= self.n_neg else 0.0
            margin = 2.0 * (parent_fit - 0.5) + bias
            p = 1.0 / (1.0 + math.exp(-margin))
            p = min(max(p, 1e-3), 1.0 - 1e-3)
            rows.append([1.0 - p, p])
        return rows


class FakeNonProbabilisticModel:
    """只 fit 不 predict_proba（fail-closed 测试用）。"""

    def fit(self, X, y):
        self.fitted = True
        return self


def _make_surrogate(**cfg) -> PromotionSurrogate:
    return PromotionSurrogate(
        extractor=DefaultFeatureExtractor(),
        config=PromotionSurrogateConfig(**cfg),
    )


# ---------------------------------------------------------------------------
# 1. 无已训练 surrogate → 确定性 fallback
# ---------------------------------------------------------------------------


class TestDeterministicFallback:
    def test_no_model_no_history_uses_threshold_fallback(self):
        s = _make_surrogate()
        assert not s.is_trained
        assert s.n_observations == 0
        # 无模型 → source 必须是 deterministic_fallback，绝不 source='model'
        rec = s.recommend([ctx("a")])
        assert rec[0].source == "deterministic_fallback"

    def test_fallback_is_threshold_semantics_not_guess(self):
        """P(L3) 落在 [0.05, 0.6] 保守带内且随 L0 证据单调移动。"""
        s = _make_surrogate()
        low = s.predict_p_l3(ctx("rejected", passed=False))
        high = s.predict_p_l3(ctx("clean", passed=True))
        assert high > low
        assert 0.05 <= low <= 0.6
        assert 0.05 <= high <= 0.6

    def test_fallback_respects_success_history_and_cost_caps(self):
        s = _make_surrogate()
        p_rich = s.predict_p_l3(ctx("r", success=0.9))
        p_poor = s.predict_p_l3(ctx("p", success=0.1))
        assert p_rich > p_poor
        # 极端历史成功也不允许逃出 [0.05, 0.6]（不瞎猜）
        assert 0.05 <= p_rich <= 0.6

    def test_fallback_recommend_orders_static_clean_first(self):
        s = _make_surrogate()
        clean = ctx("clean", passed=True)
        dirty = ctx("dirty", passed=False)
        rec = s.recommend([dirty, clean])
        assert [x.factor_id for x in rec] == ["clean", "dirty"]
        assert rec[0].evi >= rec[1].evi

    def test_empty_candidates_returns_empty(self):
        s = _make_surrogate()
        assert s.recommend([]) == []

    def test_predictor_is_deterministic(self):
        s = _make_surrogate()
        c = ctx("a")
        assert s.predict_p_l3(c) == pytest.approx(s.predict_p_l3(c))


# ---------------------------------------------------------------------------
# 2. #23 sealed Test 零读取：训练目标禁止 sealed/held-out
# ---------------------------------------------------------------------------


class TestSealedTestZeroRead:
    @pytest.mark.parametrize(
        "label",
        [
            {"segment": "test", "l3_status": "pass"},
            {"segment": "sealed", "l3_pass": 1},
            {"segment": "held_out", "l3_status": "ok"},
            {"segment": "hidden", "l3_pass": True},
            {"version": "sealed", "l3_status": "pass"},
            {"version": "frozen_test", "l3_pass": 1},
            {"split": "oof", "l3_status": "pass"},
            {"segment": "L5_sealed_test", "l3_pass": 1},
        ],
    )
    def test_sealed_label_raises_fail_closed(self, label):
        s = _make_surrogate()
        with pytest.raises(SealedTestLabelError):
            s.observe(ctx("c"), label)

    def test_research_caliber_segments_are_allowed(self):
        s = _make_surrogate()
        for seg in ("train", "validation", "valid", "research", "val", ""):
            s.observe(ctx(f"c-{seg}"), {"segment": seg, "l3_status": "pass"})
        assert s.n_observations == 6

    def test_unknown_segment_also_fails_closed(self):
        s = _make_surrogate()
        with pytest.raises(SealedTestLabelError):
            s.observe(ctx("c"), {"segment": "foresight", "l3_status": "pass"})

    def test_label_frozen_flag_trains_nothing_sealed(self):
        """L5（frozen Test）结果不可进入 surrogate 训练。"""
        s = _make_surrogate()
        with pytest.raises(SealedTestLabelError):
            s.observe(ctx("c"), {"segment": "L5_sealed_test", "l3_status": "pass"})
        assert s.n_observations == 0

    def test_context_sealed_marker_raises_too(self):
        s = _make_surrogate()
        c = ExtractionContext(
            factor_id="c",
            formula="x",
            parent_ids=["sealed-bucket"],
            metric_bundle={"passed": True},
        )
        with pytest.raises(SealedTestLabelError):
            s.observe(c, {"segment": "train", "version": "sealed", "l3_status": "pass"})


# ---------------------------------------------------------------------------
# 3. EVI promotion 序：高 EVI/低成本胜过低概率/高成本
# ---------------------------------------------------------------------------


class TestEVIOrdering:
    def _make(self) -> PromotionSurrogate:
        # 禁用增益缩放，EVI = p × gain / cost，便于精确断言序
        return _make_surrogate(enabled=True, use_model=True, gain_weight=1.0)

    def test_high_evi_low_cost_beats_low_prob_high_cost(self):
        s = self._make()
        high_evi = ctx(
            "high-evi",
            passed=True,           # P ≈ 0.1（fallback 上限 0.6，这里静态干净）
            cost=0.05,             # 近零成本
            qe_seconds=10.0,
            success=0.8,           # 历史 action 高频成功 → P 更高
        )
        low_prob_high_cost = ctx(
            "low-prob-high-cost",
            passed=False,          # P 低
            cost=0.95,             # 昂贵
            qe_seconds=7200.0,
            success=0.1,
        )
        rec = s.recommend([low_prob_high_cost, high_evi])
        assert rec[0].factor_id == "high-evi"
        # EVI = P×gain/cost 单调成立（用计算值自检）
        assert rec[0].evi > rec[1].evi

    def test_equal_gain_cheap_action_higher_cost_adjusted_ev(self):
        """等增益下贵 action 的 cost-adjusted EV 更低（cost 进 EV）。"""
        s = self._make()
        gains = {"cheap": 0.5, "pricey": 0.5}
        cheap = ctx("cheap", passed=True, cost=0.1)
        pricey = ctx("pricey", passed=True, cost=0.9)
        rec = s.recommend([pricey, cheap], gains=gains)
        assert rec[0].factor_id == "cheap"
        # 显式：EV_cheap ≈ 0.5×P/c_cheap；EV_pricey ≈ 0.5×P/c_pricey（同 P 更贵更低）
        assert cheap.estimated_cost < pricey.estimated_cost
        assert rec[0].evi > rec[1].evi
        # 同一增益下成本 9 倍 → EVI 约 9 倍（等 P、等 gain）
        p_c = s.predict_p_l3(cheap)
        p_p = s.predict_p_l3(pricey)
        # passed 全同 → 同 fallback 概率；EVI 比 ≈ cost 反比
        assert p_c == pytest.approx(p_p)
        assert rec[0].evi / rec[1].evi == pytest.approx(0.9 / 0.1, rel=1e-3)

    def test_low_cost_wins_over_equal_prob_higher_cost(self):
        s = self._make()
        # 相同静态 + 相同 cost… 但 qe_seconds 不同 → fallback 成本估算不同
        cheap = ctx("cheap", passed=True, cost=None, qe_seconds=5.0)
        spendy = ctx("spendy", passed=True, cost=None, qe_seconds=7200.0)
        rec = s.recommend([spendy, cheap])
        assert rec[0].factor_id == "cheap"
        assert rec[0].expected_cost < rec[1].expected_cost

    def test_evi_formula_shape(self):
        s = _make_surrogate()
        c = ctx("a", passed=True, cost=0.2)
        p = s.predict_p_l3(c)
        cost = s.expected_cost_of(c)
        rec = s.recommend([c])[0]
        # metric_bundle passed → fallback gain=0.6；增益缩放 gain_weight=1.0
        # → EVI = p × 0.6 / cost
        assert rec.evi == pytest.approx(p * 0.6 / cost)
        # 同 candidate 重算必得同 EVI（确定性）
        again = s.recommend([c])[0]
        assert again.evi == pytest.approx(rec.evi)


# ---------------------------------------------------------------------------
# 4. 训练后 surrogate：可注入模型 + 概率反映历史
# ---------------------------------------------------------------------------


class FakeSkLearnLogistic:
    """极简 sklearn LogisticRegression 替代（仅 numpy，确定性解）。

    P(L3) = sigmoid(w·x + b)，w 为历史中 parent_fitness 正例 vs 负例的
    均值差方向。fit 后预测概率随正例比例单调。够验证「训练数据进模型」。
    """

    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = 0.0
        self.classes_ = [0, 1]

    def fit(self, X, y):
        import numpy as np

        arr = np.asarray(X, dtype="float64")
        ys = np.asarray([int(v) for v in y], dtype="float64")
        pos = arr[ys > 0.5]
        neg = arr[ys <= 0.5]
        self.intercept_ = 0.0
        self.coef_ = np.zeros(arr.shape[1])
        if len(pos) and len(neg):
            # 特征 19 = parent_fitness：正例 fitness 高 → 正系数
            delta = pos[:, 19].mean() - neg[:, 19].mean()
            self.coef_[19] = 2.0 if delta > 0 else -2.0
        elif len(pos):
            self.coef_[19] = 2.0
        else:
            self.coef_[19] = -2.0
        return self

    def predict_proba(self, X):
        import math

        import numpy as np

        arr = np.asarray(X, dtype="float64")
        margin = arr @ self.coef_ + self.intercept_
        out = []
        for m in margin:
            p = 1.0 / (1.0 + math.exp(-float(m)))
            p = min(max(p, 1e-3), 1.0 - 1e-3)
            out.append([1.0 - p, p])
        return out


class TestTrainedSurrogate:
    def test_no_model_no_forced_fit_returns_false(self):
        s = _make_surrogate()
        assert s.fit() is False

    def test_fit_requires_min_history(self):
        s = _make_surrogate(use_model=True)
        s.model = DeterministicTwoParamModel()
        for i in range(MIN_HISTORY_FOR_FIT - 1):
            s.observe(ctx(f"c{i}", passed=i % 2 == 0), {"segment": "train", "l3_status": "pass"})
        assert s.fit() is False
        assert not s.is_trained
        # 补一条到门槛 → fit 成功
        s.observe(ctx("c-last", passed=True), {"segment": "train", "l3_status": "pass"})
        assert s.fit() is True
        assert s.is_trained

    def test_model_missing_predict_proba_fails_closed(self):
        s = _make_surrogate(use_model=True)
        s.model = FakeNonProbabilisticModel()
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(ctx(f"c{i}", passed=True), {"segment": "train", "l3_status": "pass"})
        with pytest.raises(ModelNotProbabilisticError):
            s.fit()

    def test_trained_source_is_model(self):
        s = _make_surrogate(use_model=True)
        s.model = DeterministicTwoParamModel()
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(
                ctx(f"c{i}", passed=True, parent_fitness=0.7),
                {"segment": "train", "l3_status": "pass"},
            )
        assert s.fit() is True
        assert s.is_trained
        rec = s.recommend([ctx("new", passed=True, parent_fitness=0.7)])
        assert rec[0].source == "model"

    def test_model_probability_reflects_history(self):
        """正例居多的历史 → 确定性 fallback 先验向观测移动（历史进先验）。"""
        s = _make_surrogate(use_model=False)  # 关闭模型，只验证观测→先验
        assert s.fit() is False  # 无模型 → 不 fit，但先验已同步
        cand = ctx("cand", passed=True, parent_fitness=0.8)
        # 观测前
        before_p = s.fallback.predict_p_l3(s.extractor.extract(cand))
        # 全部正例历史（研究 segment）
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(
                ctx(f"c{i}", passed=True, parent_fitness=0.8),
                {"segment": "train", "l3_status": "pass"},
            )
        s.fit()
        after_p = s.fallback.predict_p_l3(s.extractor.extract(cand))
        assert 0.05 <= before_p <= 0.6
        assert 0.05 <= after_p <= 0.6
        assert after_p > before_p

    def test_reset_clears_history(self):
        s = _make_surrogate(use_model=True)
        s.model = DeterministicTwoParamModel()
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(ctx(f"c{i}", passed=True), {"segment": "train", "l3_status": "pass"})
        s.fit()
        assert s.is_trained
        s.reset()
        assert s.n_observations == 0
        assert not s.is_trained

    def test_state_and_dict_roundtrip_keys(self):
        s = _make_surrogate()
        st = s.state()
        assert st["n_observations"] == 0
        assert "trained" in st and "model_fitted" in st
        d = s.to_dict()
        assert d["state"]["enabled"] is True
        assert d["config"]["gain_weight"] == 1.0

    def test_config_enabled_false_keeps_deterministic(self):
        """#30 开关：enabled=False → fit/recommend 恒确定性 fallback。"""
        s = _make_surrogate(enabled=False, use_model=True)
        s.model = DeterministicTwoParamModel()
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(ctx(f"c{i}", passed=True), {"segment": "train", "l3_status": "pass"})
        assert s.fit() is False
        assert not s.is_trained
        rec = s.recommend([ctx("a", passed=True)])
        assert rec[0].source == "deterministic_fallback"

    def test_config_use_model_false_keeps_deterministic(self):
        s = _make_surrogate(enabled=True, use_model=False)
        s.model = DeterministicTwoParamModel()
        for i in range(MIN_HISTORY_FOR_FIT):
            s.observe(ctx(f"c{i}", passed=True), {"segment": "train", "l3_status": "pass"})
        assert s.fit() is False
        assert not s.is_trained


# ---------------------------------------------------------------------------
# 5. 特征提取确定性 & 宽度契约
# ---------------------------------------------------------------------------


class TestFeatureExtractor:
    def test_feature_width_and_names(self):
        fe = DefaultFeatureExtractor()
        f = fe.extract(ExtractionContext(formula="rank(ts_mean(close, 5))", metric_bundle={}))
        assert len(f.as_row()) == len(f.names) == 24
        assert f.names[0] == "operator_count"
        assert f.names[-1] == "estimated_cost"
        assert len(set(f.names)) == 24

    def test_features_are_deterministic(self):
        fe = DefaultFeatureExtractor()
        a = fe.extract(ctx("x", mb={"coverage": 0.7}))
        b = fe.extract(ctx("x", mb={"coverage": 0.7}))
        assert a.as_row() == b.as_row()

    def test_complexity_features_from_formula(self):
        fe = DefaultFeatureExtractor()
        f = fe.extract(ExtractionContext(formula="rank(ts_mean(close, 5) / ts_std(volume, 20))"))
        d = dict(zip(f.names, f.as_row()))
        assert d["operator_count"] >= 2
        assert d["complexity"] > 0
        assert d["lookback"] >= 5

    def test_domain_features(self):
        fe = DefaultFeatureExtractor()
        f = fe.extract(
            ExtractionContext(
                formula="",
                field_set=["open", "close", "volume"],
                data_fields=["open", "close", "volume"],
            )
        )
        d = dict(zip(f.names, f.as_row()))
        # known_fields 未注入 → field_domain_known 仅当存在已知价格/量/估值域成员
        assert d["data_domain_price"] > 0.5  # open/close ∈ price
        assert d["data_domain_volume"] > 0.0  # volume ∈ volume
        assert d["field_domain_known"] == 1.0  # price/volume 域成员本身即已知域
        assert d["data_domain_price"] + d["data_domain_volume"] > 0.5

    def test_known_fields_injection(self):
        fe = DefaultFeatureExtractor(known_fields=["close", "volume"])
        f = fe.extract(
            ExtractionContext(
                formula="", field_set=["close", "pe_ttm"], data_fields=["close", "pe_ttm"]
            )
        )
        d = dict(zip(f.names, f.as_row()))
        assert d["field_domain_known"] == 1.0
        assert d["data_domain_price"] > 0.0
        assert d["data_domain_valuation"] > 0.0

    def test_metric_bundle_consumed_verbatim_not_recomputed(self):
        """#7：只消费现成数值；complexity 数值直接来自 bundle 不改算。"""
        fe = DefaultFeatureExtractor()
        f = fe.extract(ExtractionContext(formula="rank(x)", complexity=5))
        d = dict(zip(f.names, f.as_row()))
        assert d["complexity"] == 5.0

    def test_extractor_protocol_runtime_checkable(self):
        from alphaprobe.surrogate.features import FeatureExtractor

        assert isinstance(DefaultFeatureExtractor(), FeatureExtractor)

    def test_context_duck_extract_ctx(self):
        """candidate 可暴露 extract_ctx()（FutureWrapper 集成形态）。"""

        class Wrapper:
            def __init__(self, c: ExtractionContext) -> None:
                self._c = c

            def extract_ctx(self) -> ExtractionContext:
                return self._c

        s = _make_surrogate()
        rec = s.recommend([Wrapper(ctx("wrapped", passed=True))])
        assert rec[0].factor_id == "wrapped"

    def test_recommend_rejects_garbage_candidate(self):
        s = _make_surrogate()
        with pytest.raises(TypeError):
            s.recommend([12345])


class TestPromotionSuggestion:
    def test_to_dict_fields(self):
        s = PromotionSuggestion(
            factor_id="a",
            formula="x",
            p_l3_pass=0.3,
            expected_fitness_gain=0.5,
            expected_novelty_gain=0.2,
            expected_pool_gain=0.1,
            expected_cost=0.4,
            evi=0.375,
            source="model",
        )
        d = s.to_dict()
        assert d["factor_id"] == "a"
        assert d["evi"] == 0.375
        assert d["source"] == "model"
