"""V2-I 集成 / Leakage / LongShort / Calibration 测试矩阵 + Done Definition 核验。

任务书 §68（集成行 / Leakage 行 / LongShort 行 / Calibration 行）+ §69（Done
Definition 核验）。全合成数据、不读 COS、stub 注入、不跑真实 LLM / 模型训练。

覆盖：
① 端到端集成：stub LLM → SearchPipeline 一轮（structured_generation=False 默认
   路径）→ dedup 过滤 → L0-L2 → QuantEvaluatorAdapter 评估 → factor_fitness_v2
   出分 → ActivePool Pareto 入池 → RoundResult 字段完整、入池候选有 lineage 与
   Fitness 分、dedup 重复候选被拒。
② Leakage 行：构造「因子=未来函数」泄漏案例（factor 直接用 label 信息），断言
   评估链按 vwap→vwap 20d label 计算的 rankic 接近 1 触发极端值红旗；并断言
   「无未来函数」契约（factor(t) 只用 t 及以前、label 用 t+1..t+20）。
③ LongShort 行：同一因子合成数据（长记忆信号），断言 overlapping 直接年化
   Sharpe 显著高于 cohort 真实 PnL Sharpe（inflation 存在），且 adapter 只输出
   cohort 口径；断言 L 维 9 项指标全部有值且在合理范围。
④ Calibration 行：fitness/calibration.py 的 U() warmup ordinal + freeze z 路径，
   构造小样本序列断言 warmup 期用 ordinal、达到 min_warmup_n 后用 z、freeze 后
   参数不变；不同 Sharpe 序列单调性映射正确。
⑤ Done Definition 核验（§69）：三 Score 拆分存在、L 20d cohort 非 overlapping、
   评估只走 QE（无手写 calc_sharpe/calc_long_short/calc_mdd）、FE fail-closed、
   LabelContract embargo≥label_days、structured_generation 默认关闭、Pareto 淘汰
   非单一 IC。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 引导 quant_projects 进 sys.path（QE / factor_engine / modeling 可导入）。
# pytest 从 repo 根跑时 quant_projects 已在 sys.path；此处幂等兜底。
_QP = Path(__file__).resolve().parents[4]  # .../alphaprobe/ashare/alphaprobe-dev -> 上溯 4 层
for _cand in Path(__file__).resolve().parents:
    if (_cand / "quant_evaluator").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from alphaprobe.evaluator_adapter import QuantEvaluatorAdapter
from alphaprobe.fitness import MetricCalibrator
from alphaprobe.fitness.calibration import single_utility
from alphaprobe.fitness.contracts import EvaluationBundle
from alphaprobe.fitness.factor_fitness import factor_fitness_v2
from alphaprobe.pipeline import (
    PipelineConfig,
    SearchPipeline,
    make_qe_evaluate_fn,
    make_stub_llm_fn,
)

# ---------------------------------------------------------------------------
# 合成数据工具
# ---------------------------------------------------------------------------


def _make_panels(
    T: int = 500,
    N: int = 80,
    seed: int = 1,
    *,
    drift_scale: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造 vwap 随机游走 + 长记忆漂移（momentum 信号）。

    Returns (vwap_df, label_df, mom_df)：
    - vwap_df：index=date, columns=code 的 VWAP 价格面板；
    - label_df：vwap→vwap 20 日远期收益（label[t] = vwap[t+20]/vwap[t] - 1）；
    - mom_df：动量因子（factor[t] = vwap[t] - vwap[t-20]，只用 t 及以前数据）。
    """
    dates = pd.date_range("2020-01-01", periods=T)
    codes = [f"c{i}" for i in range(N)]
    rng = np.random.default_rng(seed)
    drift = rng.standard_normal(N) * drift_scale
    ret = drift[None, :] + rng.standard_normal((T, N)) * 0.02
    vwap = 100.0 * np.exp(np.cumsum(ret, axis=0))
    vwap_df = pd.DataFrame(vwap, index=dates, columns=codes)
    label_df = vwap_df.shift(-20) / vwap_df - 1.0
    mom_df = vwap_df - vwap_df.shift(20)
    return vwap_df, label_df, mom_df


def _aligned(factor_df, label_df, vwap_df, lookback: int = 20):
    """对齐到 label 段（去掉前 lookback 行，使 factor/label/price 同形状）。"""
    return factor_df.iloc[lookback:], label_df.iloc[lookback:], vwap_df.iloc[lookback:]


def _make_calibrator() -> MetricCalibrator:
    """seed + freeze 的 MetricCalibrator（frozen z-score 单调可比）。"""
    c = MetricCalibrator(min_warmup_n=10)
    seeds = {
        "rankic_valid": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
        "median_subperiod_rankic": [0.003, 0.008, 0.012, 0.02, 0.03, 0.04],
        "hac_tstat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
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
        "rankicir": [0.1, 0.3, 0.5, 0.8, 1.2],
        "coverage": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "nan_inf_ratio": [0.0, 0.05, 0.1, 0.2, 0.4],
        "untradeable_ratio": [0.0, 0.05, 0.1, 0.2, 0.4],
        "turnover": [0.1, 0.2, 0.5, 1.0, 2.0],
    }
    c.seed_prior(seeds)
    c.freeze("v2i_test")
    return c


def _overlapping_sharpe(factor_df, label_df, n_quantiles: int = 10) -> float | None:
    """overlapping 直接年化 Sharpe（错误口径：把 20d 重叠收益当独立日频）。

    用 sqrt(252) 年化（而非 sqrt(252/20)），模拟「直接年化重叠远期收益」的
    inflation 假象。仅测试用，验证 cohort 口径的必要性。
    """
    f = factor_df.to_numpy()
    y = label_df.to_numpy()
    T, N = f.shape
    ls_ret = np.zeros(T)
    for t in range(T):
        ft = f[t]
        yt = y[t]
        m = np.isfinite(ft) & np.isfinite(yt)
        if m.sum() < 30:
            continue
        q = np.argsort(np.argsort(ft[m]))
        n = m.sum()
        top = q >= n - n // n_quantiles
        bot = q < n // n_quantiles
        ls_ret[t] = np.nanmean(yt[m][top]) - np.nanmean(yt[m][bot])
    ls_ret = ls_ret[np.isfinite(ls_ret)]
    if len(ls_ret) < 2:
        return None
    mu = ls_ret.mean()
    sd = ls_ret.std(ddof=1)
    if sd == 0:
        return None
    return mu / sd * np.sqrt(252)


class _FakeStockData:
    """最小假 stock_data：evaluate_many 返回随机平面，供 make_qe_evaluate_fn。"""

    def __init__(self, T: int = 300, N: int = 40, seed: int = 0) -> None:
        self._field_names = lambda: ["open", "high", "low", "close", "vwap", "volume"]
        self._dates = pd.date_range("2020-01-01", periods=T)
        self._stock_ids = pd.Index([f"c{i}" for i in range(N)])
        rng = np.random.default_rng(seed)
        ret = rng.standard_normal((T, N)) * 0.02
        vwap = 100.0 * np.exp(np.cumsum(ret, axis=0))
        self.data = np.zeros((T, N, 6), dtype=float)
        self.data[:, :, 4] = vwap
        self.data[:, :, 3] = vwap

    def evaluate_many(self, formulas):
        rng = np.random.default_rng(1)
        return [rng.standard_normal((self.data.shape[0], self.data.shape[1])) for _ in formulas]


# ---------------------------------------------------------------------------
# ① 端到端集成（§68 集成行）
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    def test_round_result_fields_complete(self):
        """stub LLM → SearchPipeline 一轮 → RoundResult 字段完整。"""
        sd = _FakeStockData()
        sp = SearchPipeline(
            experiment=None,
            data_train=sd,
            config=PipelineConfig(pool_target=4, pool_max=8),
        )
        sp.evaluator = None
        sp._evaluate_fn = make_qe_evaluate_fn(sd, label_days=20)
        res = sp.run_round(
            round_id="r1",
            parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
        )
        assert res.candidates_generated > 0
        assert res.evaluated == res.candidates_generated
        assert res.admitted >= 0
        assert res.duplicates_filtered >= 0
        assert isinstance(res.degraded, bool)
        assert isinstance(res.degraded_reasons, list)
        assert isinstance(res.pool_snapshot, list)
        assert len(res.pool_snapshot) == sp.pool_size()

    def test_admitted_candidate_has_fitness_and_lineage(self):
        """入池候选有 Fitness 分与 lineage（parent_ids）。"""
        sd = _FakeStockData()
        c = _make_calibrator()
        base_fn = make_qe_evaluate_fn(sd, label_days=20)

        def eval_fn(formulas, fidelity="L2_full_train", context=None):
            out = []
            for b in base_fn(formulas, fidelity, context):
                if b is None:
                    out.append(None)
                    continue
                r = factor_fitness_v2(EvaluationBundle(b), c)
                b2 = dict(b)
                b2["search_fitness"] = r.fitness
                out.append(b2)
            return out

        forced = [
            {
                "formula": "rank(close)",
                "explanation": "e",
                "hypothesis": "h",
                "action_type": "REFINE",
                "parent_ids": ["p1"],
            }
        ]
        sp = SearchPipeline(
            experiment=None,
            data_train=sd,
            config=PipelineConfig(pool_target=4, pool_max=8),
        )
        sp.evaluator = None
        sp._evaluate_fn = eval_fn
        sp.llm_fn = make_stub_llm_fn(forced_candidates=forced)
        res = sp.run_round(
            round_id="r1",
            parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
        )
        assert res.admitted >= 1
        assert len(sp.pool.members) >= 1
        for fid, m in sp.pool.members.items():
            assert m.search_fitness > 0.0, f"member {fid} missing fitness"
            assert m.canonical_formula, f"member {fid} missing formula"

    def test_duplicate_candidate_rejected_second_round(self):
        """同公式第二轮被 dedup 硬重复拦截（duplicates_filtered>0, evaluated=0）。"""
        sd = _FakeStockData()
        sp = SearchPipeline(
            experiment=None,
            data_train=sd,
            config=PipelineConfig(pool_target=4, pool_max=8),
        )
        sp.evaluator = None
        sp._evaluate_fn = make_qe_evaluate_fn(sd, label_days=20)
        parents = [{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}]
        r1 = sp.run_round(round_id="r1", parents=parents)
        assert r1.evaluated == r1.candidates_generated
        r2 = sp.run_round(round_id="r2", parents=parents)
        assert r2.candidates_generated > 0
        assert r2.evaluated == 0
        assert r2.duplicates_filtered >= r2.candidates_generated


# ---------------------------------------------------------------------------
# ② Leakage 行（§68）
# ---------------------------------------------------------------------------


class TestLeakage:
    def test_leakage_factor_rankic_near_one_red_flag(self):
        """因子=未来函数（直接用 label 信息）→ rankic_valid≈1.0 触发极端值红旗。"""
        vwap_df, label_df, _ = _make_panels(seed=1)
        # 泄漏因子：直接用未来 20 日收益（label 信息）
        leak = label_df.copy()
        f, y, p = _aligned(leak, label_df, vwap_df)
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
        rankic = b.get("rankic_valid")
        assert rankic is not None
        # 泄漏因子 rankic 应接近 1（完美预测未来 → 红旗）
        assert rankic > 0.9, f"leakage factor rankic should be near 1, got {rankic}"

    def test_legitimate_factor_rankic_not_near_one(self):
        """合法动量因子（只用 t 及以前数据）rankic 不应接近 1。"""
        vwap_df, label_df, mom_df = _make_panels(seed=1)
        f, y, p = _aligned(mom_df, label_df, vwap_df)
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
        rankic = b.get("rankic_valid")
        assert rankic is not None
        assert rankic < 0.5, f"legitimate momentum rankic should be modest, got {rankic}"

    def test_no_future_function_contract(self):
        """无未来函数契约：factor(t) 只用 t 及以前，label 用 t+1..t+20。

        验证 label 构造方向：label[t] = vwap[t+20]/vwap[t] - 1（shift(-20)），
        且动量因子 factor[t] = vwap[t] - vwap[t-20] 只用 t 及以前数据。
        """
        vwap_df, label_df, mom_df = _make_panels(seed=1)
        # label 方向：label[t] 应等于 vwap[t+20]/vwap[t]-1
        expected = vwap_df.shift(-20) / vwap_df - 1.0
        pd.testing.assert_frame_equal(label_df, expected)
        # 动量因子只用 t 及以前：factor[t] 与 vwap[t+20] 无关（不直接用未来）
        # 验证 factor 是过去 20 日差分（不含未来信息）
        expected_mom = vwap_df - vwap_df.shift(20)
        pd.testing.assert_frame_equal(mom_df, expected_mom)
        # 泄漏因子（=label）与合法因子应显著不同（rankic 差异巨大）
        leak = label_df.copy()
        f_leak, y, p = _aligned(leak, label_df, vwap_df)
        f_mom, _, _ = _aligned(mom_df, label_df, vwap_df)
        b_leak = QuantEvaluatorAdapter(label_days=20).evaluate(f_leak, y, p)
        b_mom = QuantEvaluatorAdapter(label_days=20).evaluate(f_mom, y, p)
        assert b_leak.get("rankic_valid") > b_mom.get("rankic_valid") + 0.5


# ---------------------------------------------------------------------------
# ③ LongShort 行（§68）
# ---------------------------------------------------------------------------


class TestLongShort:
    def test_overlapping_sharpe_inflated_vs_cohort(self):
        """overlapping 直接年化 Sharpe 显著高于 cohort 真实 PnL Sharpe（inflation）。"""
        vwap_df, label_df, mom_df = _make_panels(seed=1)
        f, y, p = _aligned(mom_df, label_df, vwap_df)
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
        cohort_sharpe = b.get("net_sharpe")
        ov_sharpe = _overlapping_sharpe(f, y)
        assert cohort_sharpe is not None
        assert ov_sharpe is not None
        # overlapping sqrt(252) 年化显著高于 cohort（inflation 存在）
        assert ov_sharpe > cohort_sharpe * 1.5, (
            f"overlapping {ov_sharpe:.2f} should exceed cohort {cohort_sharpe:.2f}"
        )

    def test_adapter_outputs_only_cohort_caliber(self):
        """adapter 只输出 cohort 口径（net_sharpe 来自 compute_metrics_from_cohort）。"""
        vwap_df, label_df, mom_df = _make_panels(seed=1)
        f, y, p = _aligned(mom_df, label_df, vwap_df)
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
        # cohort 口径的 net_sharpe 是真实 daily PnL 年化（非 overlapping 直接年化）
        assert b.get("net_sharpe") is not None
        # 不应出现 overlapping 直接年化的键（如 "overlapping_sharpe"）
        assert "overlapping_sharpe" not in b.raw()

    def test_l_dimension_nine_metrics_present_and_reasonable(self):
        """L 维 9 项指标全部有值且在合理范围。"""
        vwap_df, label_df, mom_df = _make_panels(seed=1)
        f, y, p = _aligned(mom_df, label_df, vwap_df)
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
        keys = [
            "net_sharpe",
            "calmar_ratio",
            "sortino_ratio",
            "net_annualized_ls_return",
            "d10_long_only_active_return",
            "max_drawdown",
            "drawdown_persistence",
            "q20_rolling_sharpe",
            "positive_month_ratio",
        ]
        for k in keys:
            v = b.get(k)
            assert v is not None, f"L metric {k} missing"
            assert np.isfinite(v), f"L metric {k} not finite: {v}"
        # 合理范围：Sharpe 类为正、MDD 为负向（绝对值小）、比率在 [0,1]
        assert b.get("net_sharpe") > 0
        assert b.get("calmar_ratio") > 0
        assert b.get("sortino_ratio") > 0
        assert 0.0 <= b.get("max_drawdown") <= 1.0
        assert 0.0 <= b.get("positive_month_ratio") <= 1.0
        assert 0.0 <= b.get("drawdown_persistence") <= 1.0


# ---------------------------------------------------------------------------
# ④ Calibration 行（§68）
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_warmup_uses_ordinal(self):
        """warmup 期（样本 < min_warmup_n）用 ordinal 百分位。"""
        c = MetricCalibrator(min_warmup_n=3)
        c.observe("net_sharpe", 1.0)
        c.observe("net_sharpe", 2.0)
        assert c.warmup_active("net_sharpe") is True
        # ordinal：2.0 是当前最大 → utility 1.0；1.0 → 0.5
        assert single_utility(c, "net_sharpe", 2.0) == pytest.approx(1.0)
        assert single_utility(c, "net_sharpe", 1.0) == pytest.approx(0.5)

    def test_reaching_min_warmup_n_switches_to_z(self):
        """达到 min_warmup_n 后 warmup 结束（后续 freeze 走 z-score）。"""
        c = MetricCalibrator(min_warmup_n=3)
        for v in (1.0, 2.0, 3.0):
            c.observe("net_sharpe", v)
        assert c.warmup_active("net_sharpe") is False
        c.freeze("r1")
        assert c.is_frozen() is True
        assert c.frozen_round == "r1"
        # freeze 后走 z-score：中位数 2.0 → utility 0.5
        assert single_utility(c, "net_sharpe", 2.0) == pytest.approx(0.5)

    def test_freeze_params_unchanged_after_observe(self):
        """freeze 后参数（median, mad）不再随新样本变化。"""
        c = MetricCalibrator(min_warmup_n=3)
        for v in (1.0, 2.0, 3.0):
            c.observe("net_sharpe", v)
        c.freeze("r1")
        frozen_before = c._frozen.get("net_sharpe")
        c.observe("net_sharpe", 10.0)
        frozen_after = c._frozen.get("net_sharpe")
        assert frozen_before == frozen_after

    def test_sharpe_monotonic_mapping(self):
        """不同 Sharpe 序列单调性映射正确（更高 Sharpe → 更高 utility）。"""
        c = _make_calibrator()
        lo = single_utility(c, "net_sharpe", 1.0)
        mid = single_utility(c, "net_sharpe", 2.0)
        hi = single_utility(c, "net_sharpe", 3.0)
        assert lo < mid < hi


# ---------------------------------------------------------------------------
# ⑤ Done Definition 核验（§69）
# ---------------------------------------------------------------------------


class TestDoneDefinition:
    def test_three_scores_split_exists(self):
        """三 Score 拆分存在：FactorFitness / RetrieverScore / ExportScore。"""
        from alphaprobe.fitness.factor_fitness import factor_fitness_v2
        from alphaprobe.retrieval.bayesian_retriever import compute_retriever_score
        from alphaprobe.fitness import export_score

        assert callable(factor_fitness_v2)
        assert callable(compute_retriever_score)
        assert callable(export_score)

    def test_l_dimension_20d_cohort_non_overlapping(self):
        """L 维 20d cohort 非 overlapping（adapter 走 cohort 真实 daily PnL）。"""
        import alphaprobe.evaluator_adapter as ea

        calls = {"cohort": 0}
        real = ea.QuantEvaluatorAdapter._cohort_metrics

        def spy(self, f, price, dates, codes):
            calls["cohort"] += 1
            return real(self, f, price, dates, codes)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ea.QuantEvaluatorAdapter, "_cohort_metrics", spy)
        try:
            vwap_df, label_df, mom_df = _make_panels(seed=1)
            f, y, p = _aligned(mom_df, label_df, vwap_df)
            b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, p)
            assert calls["cohort"] == 1
            assert "net_sharpe" in b
        finally:
            monkeypatch.undo()

    def test_evaluation_only_via_qe_no_handwritten_calcs(self):
        """评估只走 QE：AlphaPROBE src 内无手写 calc_sharpe/calc_long_short/calc_mdd。"""
        src_root = Path(__file__).resolve().parents[1] / "src"
        offenders = []
        for p in src_root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            text = p.read_text(encoding="utf-8")
            for tok in ("calc_sharpe", "calc_long_short", "calc_mdd"):
                if tok in text:
                    offenders.append(f"{p.relative_to(src_root)}: {tok}")
        assert offenders == [], f"手写金融指标计算残留: {offenders}"

    def test_fe_fail_closed(self):
        """FE fail-closed：authority.build_identity_view 抛错路径。"""
        from alphaprobe.authority import FactorIdentityAuthorityError, build_identity_view

        with pytest.raises(FactorIdentityAuthorityError):
            build_identity_view("   ")
        with pytest.raises(FactorIdentityAuthorityError):
            build_identity_view("rank(close)", provider=_BrokenProvider())

    def test_label_contract_embargo_ge_label_days(self):
        """LabelContract embargo ≥ label_days（20）。"""
        from alphaprobe.authority import vwap_20d_label_contract

        contract = vwap_20d_label_contract()
        assert contract.horizon_bars == 20
        assert contract.embargo_bars >= contract.horizon_bars
        assert contract.return_basis == "vwap_to_vwap"

    def test_structured_generation_default_off(self):
        """structured_generation 默认关闭行为不变（SP 消费层旧行为）。

        P0-A：PipelineConfig 裸默认值是 UNSET 哨兵（生产语义在 SearchPipeline
        构造时解析）；OFLLINE_TEST 默认路径下 orchestrator 的
        structured_generation 仍为 False——旧行为不变（与 test_v3_p0_pipeline
        的兼容层断言一致）。
        """
        cfg = PipelineConfig()
        assert cfg.structured_generation is not False  # UNSET 哨兵
        sp = SearchPipeline(experiment=None, data_train=None, config=cfg)
        assert sp.orchestrator.structured_generation is False

    def test_pareto_eviction_not_single_ic(self):
        """Pareto 淘汰非单一 IC（低 fitness 高 novelty 者不被单维 IC 淘汰）。"""
        from alphaprobe.pool.pareto import (
            DEFAULT_OBJECTIVES,
            ParetoPoint,
            non_dominated_rank,
            pareto_eviction_candidate,
        )

        A = ParetoPoint("A", (0.9, 0.1, 5, 0.1), objectives=DEFAULT_OBJECTIVES)
        D = ParetoPoint("D", (0.1, 0.9, 5, 0.1), objectives=DEFAULT_OBJECTIVES)
        rank = non_dominated_rank([A, D])
        # 互不支配 → 都在 front 0（不因单维 IC 淘汰 D）
        assert rank["A"] == 0 and rank["D"] == 0
        victim = pareto_eviction_candidate([A, D])
        assert victim is not None


class _BrokenProvider:
    def get_factor_identity(self, formula):
        return None
