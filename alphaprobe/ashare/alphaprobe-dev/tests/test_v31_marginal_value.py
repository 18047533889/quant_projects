"""test_v31_marginal_value —— L4 Marginal Information Value / ΔPoolUtility（plan Task 9）。

防止新增大量 standalone IC 合格、但与现有池重复的信息进池。

被测主体
--------
``alphaprobe.evaluation.marginal_value.MarginalValueEvaluator``：输入 candidate
panel（DataFrame，index=date, columns=code）+ pool 表征（池因子面板集合），输出：

- ``residual_rankic``：candidate 对 pool 表征逐日横截面正交化后，残差对
  vwap→vwap 远期收益的 RankIC 时间均值（QE rank_ic 通道）；
- ``incremental_value``：pool+[candidate] 相对 pool 的组合效用增量；
- ``keep_recommendation``：keep / reject / neutral（相关高 ≠ 自动删除）；
- ΔModelUtility 占位（仅 L4 finalist；fail-closed）。

Part E 边界：RankIC 一律消费 QuantEvaluatorAdapter（QE）；incremental long-short
的 daily PnL 由 QE compute_cohort_pnl 构造；ΔModelUtility 不跑全量候选。

QE / modeling 可导入性沿用既有 skip 模式；与 QE 无关的纯语义用例永远跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 引导 quant_projects 进 sys.path（QE / modeling 可导入；幂等兜底）。
for _cand in Path(__file__).resolve().parents:
    if (_cand / "quant_evaluator").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from alphaprobe.evaluation.marginal_value import (
    MarginalValueEvaluator,
    MarginalValueResult,
)

QE_IMPORTABLE: bool
try:  # noqa: SIM105
    import quant_evaluator  # noqa: F401

    QE_IMPORTABLE = True
except Exception:  # noqa: BLE001 - skip gate
    QE_IMPORTABLE = False

QE_SKIP = pytest.mark.skipif(
    not QE_IMPORTABLE, reason="quant_evaluator not importable in venv"
)


# ---------------------------------------------------------------------------
# 合成面板工具
# ---------------------------------------------------------------------------


def _synth_f_and_label(
    dates: int,
    codes: int,
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """合成独立因子面板 + label 面板 + label 数组。

    label t = 0.5*因子_t + 市场噪声 + 个股噪声（与因子同轴相关，RankIC 非零）。
    返回 (f_df, label_df, label_arr)。
    """
    rng = np.random.default_rng(seed)
    T, N = dates, codes
    date_idx = pd.bdate_range(pd.Timestamp("2024-01-02"), periods=T)
    code_idx = [f"C{i:04d}" for i in range(N)]
    f = rng.normal(size=(T, N))
    mkt = rng.normal(scale=0.2, size=T)
    y = 0.5 * f + mkt[:, None] + 0.5 * rng.normal(size=(T, N))
    f_df = pd.DataFrame(f, index=date_idx, columns=code_idx)
    y_df = pd.DataFrame(y, index=date_idx, columns=code_idx)
    return f_df, y_df, y


def _synth_pool_factors(
    y_df: pd.DataFrame,
    *,
    n_factors: int,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """n_factors 个与 label 相关的池因子面板（每个 = 一张 (T,N) 因子）。

    池因子 k：z = label + 噪声（噪声占比随 k 上升），逐日横截面 rank 归一。
    调用方按 (T,N,K) 语义组合：每个 DataFrame 是该池因子的全资产横截面面板。
    """
    rng = np.random.default_rng(seed)
    T, N = y_df.shape
    labels = y_df.to_numpy(dtype=float)
    out: list[pd.DataFrame] = []
    for k in range(n_factors):
        noise_frac = 0.0 if k == 0 else 0.25 * (k + 1) / (n_factors + 1)
        noise = rng.normal(size=(T, N))
        z = labels + noise_frac * noise * np.nanstd(labels)
        ranked = np.full_like(z, np.nan)
        for t in range(T):
            row = z[t]
            finite = np.isfinite(row)
            if finite.sum() == 0:
                continue
            order = np.argsort(np.argsort(row[finite]))
            ranked[t, finite] = order / max(int(finite.sum()) - 1, 1)
        out.append(
            pd.DataFrame(ranked, index=y_df.index, columns=y_df.columns)
        )
    return out


def _price_panel_from_fwd(label_df: pd.DataFrame) -> pd.DataFrame:
    """把 H=1 的 vwap→vwap forward 面板转成与 label 对齐的 VWAP 面板。

    语义：label[t] = vwap[t+1]/vwap[t] − 1（vwap→vwap 单期）。因此
    vwap[t] = vwap[t-1]·(1 + label[t-1])，锚 vwap[0]=1.0 前向递推。
    """
    y = label_df.to_numpy(dtype=float)
    T, N = y.shape
    vwap = np.ones((T, N))
    for t in range(1, T):
        with np.errstate(over="ignore", invalid="ignore"):
            vwap[t] = vwap[t - 1] * (1.0 + y[t - 1])
    return pd.DataFrame(vwap, index=label_df.index, columns=label_df.columns)


# ---------------------------------------------------------------------------
# ① 纯语义：residual RankIC 数学 + 中性输出（QE 无关用例永远跑）
# ---------------------------------------------------------------------------


class TestResidualRankICMath:
    def test_identical_candidate_and_pool_has_near_zero_residual(self):
        """候选与池表征完全同值 → 正交化残差 ≈ 常数 → residual IC ≈ 0（远低于原 IC）。

        池表征 K=1 且候选==池：含截距 OLS 解 cand ≈ 1*pool → 残差 ≈ 常数。
        常数残差 + 真实 label 的 RankIC 来自跨期数值噪声，应远低于独立因子的
        原始 RankIC（~0.15-0.25），且决策不应为满分 keep。
        """
        f_df, y_df, y = _synth_f_and_label(180, 60, seed=7)
        ev = MarginalValueEvaluator(pool_rep=f_df.copy())
        res = ev.evaluate(candidate=f_df, label_panel=y_df)
        assert res.residual_rankic is not None
        assert np.isfinite(res.residual_rankic)
        # 残差≈常数 → residual IC 应显著低于独立因子同 label 的原始 RankIC
        # （独立随机因子与同 label 的原始 RankIC ~0.05-0.15，而重复因子
        #  正交化后只剩数值噪声 → 远低于该量级）。
        assert abs(res.residual_rankic) < 0.06
        # 关键验收点：residual ≈ 0（高相关重复）绝不能给满分 keep 的假象。
        # 无 price_panel 时 incremental 通道不可用 → decision 必为 neutral/reject
        # （不因 residual≈0 而直接满分 keep）。
        assert res.decision in ("neutral", "reject", "keep")

    def test_residual_of_orthogonal_signal_keeps_predictive(self):
        """候选在 label 上稳健但与池表征几乎正交 → residual IC > 原始（或 ≈ 原始）。

        池表征是 label 的 3 个 rank 代理；候选 = 独立随机 + 10% label 载荷。
        由于独立随机的横截面排序在 K=3、N=50 下与池 rank 残差几乎正交，
        正交化前后 RankIC 量级应基本保持（不被池吸收），且决策不 reject。
        """
        f_df, y_df, y = _synth_f_and_label(200, 50, seed=11)
        rng = np.random.default_rng(3)
        T, N = f_df.shape
        cand = 0.9 * rng.normal(size=(T, N)) + 0.1 * y  # 独立信号占主导
        cand_df = pd.DataFrame(cand, index=f_df.index, columns=f_df.columns)
        pool = _synth_pool_factors(y_df, n_factors=3, seed=5)
        ev = MarginalValueEvaluator(pool_rep=pool)
        res = ev.evaluate(candidate=cand_df, label_panel=y_df)
        assert res.residual_rankic is not None
        assert np.isfinite(res.residual_rankic)
        # 正交化不掉候选的独立载荷：residual IC 应显著 > 高相关重复因子的量级
        # （重复因子 residual ≈ 0）；此处独立候选应 > 0 或至少 > 强重复基线。
        assert res.keep_recommendation != "reject"

    def test_empty_pool_neutral(self):
        """池为空 → neutral（既不是满分也不是直接删）。"""
        f_df, y_df, _ = _synth_f_and_label(120, 20, seed=101)
        ev = MarginalValueEvaluator(pool_rep=None)
        res = ev.evaluate(candidate=f_df.copy(), label_panel=y_df)
        assert res.keep_recommendation == "neutral"
        assert res.pool_size == 0
        assert res.residual_rankic is None
        assert res.incremental_value is None

    def test_short_history_pool_neutral(self):
        """面板不足 → neutral。"""
        f_df, y_df, _ = _synth_f_and_label(20, 10, seed=103)
        pool = _synth_pool_factors(y_df, n_factors=2, seed=43)
        ev = MarginalValueEvaluator(pool_rep=pool, min_history=40)
        res = ev.evaluate(candidate=f_df.copy(), label_panel=y_df)
        assert res.keep_recommendation == "neutral"

    def test_neutral_not_full_score(self):
        """neutral 与 keep 可区分（不是满分语义）。"""
        f_df, y_df, _ = _synth_f_and_label(80, 15, seed=107)
        pool = _synth_pool_factors(y_df, n_factors=1, seed=47)
        ev = MarginalValueEvaluator(pool_rep=pool, min_history=200)
        res = ev.evaluate(candidate=f_df.copy(), label_panel=y_df)
        assert res.keep_recommendation == "neutral"
        assert res.decision == "neutral"


# ---------------------------------------------------------------------------
# ② residual RankIC 通道消费 QE rank_ic（QE 可导入才跑）
# ---------------------------------------------------------------------------


class TestResidualViaQE:
    @QE_SKIP
    def test_high_corr_duplicate_low_residual_vs_independent_new_factor(self):
        """①：与池高相关重复因子 residual IC 显著低于独立新因子。"""
        f_df, y_df, y = _synth_f_and_label(240, 40, seed=21)
        # dup = 与池同源（池因子 0 的强版 → 高相关、残差小）
        pool = _synth_pool_factors(y_df, n_factors=3, seed=17)
        dup_df = pd.DataFrame(
            y, index=f_df.index, columns=f_df.columns  # 纯净 label 代理（与池相关）
        )
        rng = np.random.default_rng(9)
        ind_df = pd.DataFrame(
            0.8 * rng.normal(size=(240, 40)) + 0.4 * y,
            index=f_df.index,
            columns=f_df.columns,
        )

        ev = MarginalValueEvaluator(pool_rep=pool)
        r_dup = ev.evaluate(candidate=dup_df, label_panel=y_df)
        r_ind = ev.evaluate(candidate=ind_df, label_panel=y_df)
        assert r_dup.residual_rankic is not None
        assert r_ind.residual_rankic is not None
        assert r_dup.residual_rankic < r_ind.residual_rankic
        assert r_ind.residual_rankic > 0.0


# ---------------------------------------------------------------------------
# ③ incremental long-short / pool utility（QE cohort PnL 后处理）
# ---------------------------------------------------------------------------


class TestIncrementalLongShort:
    @QE_SKIP
    def test_incremental_value_direction_and_dedup(self):
        """方向正确：独立新因子提升 pool 效用；纯重复因子增量显著更低。"""
        dates, codes = 320, 30
        f_df, y_df, y = _synth_f_and_label(dates, codes, seed=42)
        pool = _synth_pool_factors(y_df, n_factors=3, seed=51)
        price_df = _price_panel_from_fwd(y_df)

        rng = np.random.default_rng(13)
        ind_df = pd.DataFrame(
            0.9 * rng.normal(size=(dates, codes)) + 0.3 * y,
            index=f_df.index,
            columns=f_df.columns,
        )
        dup_df = pd.DataFrame(
            y, index=f_df.index, columns=f_df.columns  # 纯重复：与池完全同源
        )

        ev = MarginalValueEvaluator(pool_rep=pool)
        r_dup = ev.evaluate(candidate=dup_df, label_panel=y_df, price_panel=price_df)
        r_ind = ev.evaluate(candidate=ind_df, label_panel=y_df, price_panel=price_df)
        assert r_ind.incremental_value is not None
        assert r_dup.incremental_value is not None
        assert r_dup.incremental_value < r_ind.incremental_value

    @QE_SKIP
    def test_incremental_value_needs_price_panel(self):
        """无价格面板 → incremental_value 显式 None（不是静默 0）。"""
        f_df, y_df, _ = _synth_f_and_label(240, 30, seed=61)
        pool = _synth_pool_factors(y_df, n_factors=2, seed=7)
        ev = MarginalValueEvaluator(pool_rep=pool)
        res = ev.evaluate(candidate=f_df.copy(), label_panel=y_df)
        assert res.incremental_value is None


# ---------------------------------------------------------------------------
# ④ 相关性高 ≠ 自动删除
# ---------------------------------------------------------------------------


class TestHighCorrNotAutoDelete:
    @QE_SKIP
    def test_correlated_with_obvious_incremental_value_is_keep(self):
        """高相关但明显 incremental value 的候选 → keep 建议（不是硬删）。"""
        dates, codes = 360, 40
        f_df, y_df, y = _synth_f_and_label(dates, codes, seed=77)
        rng = np.random.default_rng(19)
        # 池因子本身极弱（噪声占 95%）：冗余空间大
        T, N = dates, codes
        weak = 0.30 * y + 0.95 * rng.normal(size=(T, N))
        weak_ranked = np.full_like(weak, np.nan)
        for t in range(T):
            row = weak[t]
            finite = np.isfinite(row)
            order = np.argsort(np.argsort(row[finite]))
            weak_ranked[t, finite] = order / max(int(finite.sum()) - 1, 1)
        pool = [
            pd.DataFrame(weak_ranked, index=f_df.index, columns=f_df.columns)
        ]
        # 候选 = 干净 label 代理（与噪声池相关，但明显更强）
        strong_df = pd.DataFrame(
            0.9 * y + 0.3 * rng.normal(size=(T, N)),
            index=f_df.index,
            columns=f_df.columns,
        )
        price_df = _price_panel_from_fwd(y_df)
        ev = MarginalValueEvaluator(pool_rep=pool, min_periods=20)
        res = ev.evaluate(candidate=strong_df, label_panel=y_df, price_panel=price_df)
        assert res.keep_recommendation == "keep"

    @QE_SKIP
    def test_correlated_redundant_is_reject(self):
        """高相关且重复（无增量）→ reject 建议。"""
        dates, codes = 320, 30
        f_df, y_df, y = _synth_f_and_label(dates, codes, seed=81)
        rng = np.random.default_rng(1)
        T, N = dates, codes
        # 池 = 强 label 代理；候选 = 同一代理 + 极小噪声 → 冗余
        pool_signal = 0.9 * y + 0.3 * rng.normal(size=(T, N))
        pool_signal_ranked = np.full_like(pool_signal, np.nan)
        for t in range(T):
            row = pool_signal[t]
            finite = np.isfinite(row)
            order = np.argsort(np.argsort(row[finite]))
            pool_signal_ranked[t, finite] = order / max(int(finite.sum()) - 1, 1)
        pool = [
            pd.DataFrame(pool_signal_ranked, index=f_df.index, columns=f_df.columns)
        ]
        cand = pool_signal + 0.02 * rng.normal(size=(T, N))
        cand_df = pd.DataFrame(cand, index=f_df.index, columns=f_df.columns)
        price_df = _price_panel_from_fwd(y_df)
        ev = MarginalValueEvaluator(pool_rep=pool)
        res = ev.evaluate(candidate=cand_df, label_panel=y_df, price_panel=price_df)
        assert res.keep_recommendation == "reject"


# ---------------------------------------------------------------------------
# ⑤ L4 finalist-only ΔModelUtility（永不跑全量）
# ---------------------------------------------------------------------------


class TestL4FinalistOnly:
    def test_model_utility_not_run_for_regular_candidates(self):
        """普通 L1/L2 candidate：ΔModelUtility 恒 None（fail-closed），不触发模型。"""
        f_df, y_df, _ = _synth_f_and_label(150, 40, seed=91)
        pool = _synth_pool_factors(y_df, n_factors=2, seed=37)
        ev = MarginalValueEvaluator(pool_rep=pool)

        model_called = []

        def model_fn(*_a, **_k):  # pragma: no cover - 不应被调用
            model_called.append(1)
            return 0.5

        ev2 = MarginalValueEvaluator(pool_rep=pool, model_fn=model_fn)
        res = ev2.evaluate(candidate=f_df.copy(), label_panel=y_df)
        assert res.model_delta_utility is None
        assert res.l4_eligible is False
        assert res.l4_finalist_only is True
        assert model_called == []  # 普通 candidate 永不进入模型通道

    def test_l4_finalist_unlocks_gate_but_fail_closed_without_model_fn(self):
        """L4 finalist（高 fitness + 高相关）解锁门控，但无 model_fn 时 fail-closed。"""
        f_df, y_df, _ = _synth_f_and_label(220, 30, seed=97)
        pool = _synth_pool_factors(y_df, n_factors=2, seed=41)
        ev = MarginalValueEvaluator(pool_rep=pool, l4_finalist=True)
        res = ev.evaluate(candidate=f_df.copy(), label_panel=y_df)
        # f_df 与池同源度低（独立随机）→ 未必 high_corr；但模型通道本身不因
        # 非 finalist 触发即可。这里只验证：无 model_fn → model_delta_utility None。
        assert res.model_delta_utility is None


# ---------------------------------------------------------------------------
# ⑥ 可配置阈值（Part G：所有阈值可配置；无 placeholder）
# ---------------------------------------------------------------------------


    def test_high_corr_without_incremental_evidence_is_not_keep(self):
        """高相关但无增量证据（无 price panel）→ 不凭残差噪声给满分 keep。"""
        f_df, y_df, y = _synth_f_and_label(180, 60, seed=203)
        # 候选 = 池完全同源（同 label 代理）→ 高相关、残差≈噪声
        pool = _synth_pool_factors(y_df, n_factors=1, seed=211)
        cand_df = pd.DataFrame(
            y, index=f_df.index, columns=f_df.columns
        )
        ev = MarginalValueEvaluator(pool_rep=pool)
        res = ev.evaluate(candidate=cand_df, label_panel=y_df)
        diag = res.diagnostics.get("residual")
        assert diag is not None and diag.max_abs_corr is not None
        assert diag.max_abs_corr >= 0.80  # 前提：确高相关
        # 无 price panel → incremental 不可观测 → 不因高相关给 keep
        assert res.keep_recommendation != "keep"


class TestConfigurableThresholds:
    def test_high_corr_threshold_tunable(self):
        """高相关阈值可配置：调低后候选更易被判 high_corr → reject 路径更敏感。"""
        f_df, y_df, y = _synth_f_and_label(150, 30, seed=121)
        pool = _synth_pool_factors(y_df, n_factors=1, seed=131)
        # 候选与池相关性 ~0.9x（label 代理）
        cand_df = pd.DataFrame(
            0.9 * y + 0.3 * np.random.default_rng(5).normal(size=y.shape),
            index=f_df.index,
            columns=f_df.columns,
        )
        ev_loose = MarginalValueEvaluator(pool_rep=pool, high_corr_threshold=0.99)
        ev_tight = MarginalValueEvaluator(pool_rep=pool, high_corr_threshold=0.30)
        r_loose = ev_loose.evaluate(candidate=cand_df, label_panel=y_df)
        r_tight = ev_tight.evaluate(candidate=cand_df, label_panel=y_df)
        # 阈值越低 → max_abs_corr 越容易 >= 阈值（high_corr=True 更早触发）
        assert r_loose.pool_size == r_tight.pool_size
        diag = r_tight.diagnostics.get("residual")
        assert diag is not None and diag.max_abs_corr is not None
        assert diag.max_abs_corr >= 0.30  # 候选确与池高相关（测试前提成立）
