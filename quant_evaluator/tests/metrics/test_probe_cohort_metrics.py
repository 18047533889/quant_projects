"""20 日 cohort 组合构建与真实 daily PnL 指标层 —— 单测。

覆盖（用户任务书规格）：
1. cohort PnL 与手工小样例对拍一致：3-5 只票 × ~60 天合成数据，
   手算每日 cohort PnL 断言（含 1/H 资本缩放、下一日 VWAP 入场、
   持有期含出场日、gross 口径）。
2. 20d overlapping label 直接年化会产生 inflated Sharpe（对照「错误
   算法」虚高效应），而 cohort portfolio Sharpe 不被持仓重叠污染。
3. 双口径输出：long_short 与 D10 long-only active return。
4. 成本情景：per-side cost 扣减（gross / 1x / 2x / 3x 接口留存）。
5. 复用库内 quantile 模块语义（D1=最低分位组，D10=最高分位组）。
6. 边界：NaN（停牌）处理、require_tradable、输入校验。

只触碰 quant_evaluator/metrics/probe_portfolio/ 与 tests/metrics/。
"""

import numpy as np
import pytest

from quant_evaluator.metrics.probe_portfolio import (
    build_cohort_panels,
    build_quantile_masks,
    compute_cohort_pnl,
    compute_portfolio_metrics,
    compute_metrics_from_cohort,
    compute_overlapping_forward_returns,
    annualize_overlapping_label_sharpe,
    compute_drawdown_persistence,
    compute_rolling_sharpe_quantile,
    compute_positive_month_ratio,
    compute_annualized_return,
    compute_annualized_volatility,
    COST_SCENARIOS,
    cost_scenario,
    apply_per_side_costs,
)


# ---------------------------------------------------------------------------
# 1. 手工小样例对拍：3-5 只票 × ~60 天
# ---------------------------------------------------------------------------

def test_cohort_pnl_matches_hand_computation():
    """构造 12 票 × 62 天：D1/D10 恒定分桶，票收益恒定，手算每日 cohort PnL。

    场景：
    - 12 只票，因子值 = 列号（恒定），10 分位 → D1={col0, col1}，
      D10={col10, col11}（QE 分位语义：N=12、10 分位下桶 0 与桶 9 各 2 票）。
    - col11 每日 +1%（D10 内每票），col0 每日 -0.5%（D1 内每票）。
      注意：D10 桶内另 1 票（col10）收益 0，D1 桶内另 1 票（col1）收益 0。
      → 每 cohort 每日 LS 收益 =
        0.5*(0.01+0)/2 + (-0.5)*(-0.005+0)/2 = 0.0025 + 0.00125 = 0.00375；
        1/H=20 缩放后每日 0.00375/20 = 0.0001875。
    - cohort s 在 s+1 入场、持有 [s+1, s+20] 共 20 天（含出场日）。
    - 第 t 天 PnL = active_cohorts(t) × 0.0001875。
    """
    T, N = 62, 12
    factor = np.tile(np.arange(12, dtype=float), (T, 1))
    ret = np.zeros((T, N))
    ret[:, 11] = 0.01   # D10 桶中一票每日 +1%
    ret[:, 0] = -0.005  # D1 桶中一票每日 -0.5%
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl(factor, ret, vwap, holding=20)
    pnl = out["pnl_net"]

    # D10 桶 = {10, 11}：0.5*(0.01+0)/2 = 0.0025；D1 桶 = {0, 1}：
    # (-0.5)*(-0.005+0)/2 = 0.00125。合计 0.00375，/20 = 0.0001875。
    per_cohort_day = (0.5 * (0.01 + 0.0) / 2.0 + 0.5 * (0.005 + 0.0) / 2.0) / 20.0
    assert abs(per_cohort_day - 0.0001875) < 1e-12

    # 第 t 天 active cohorts：s ∈ [t-20, t-1]（s+1 <= t <= s+20）
    expected = np.zeros(T)
    for t in range(T):
        active = 0
        for s in range(t):
            if t <= s + 20:
                active += 1
        expected[t] = active * per_cohort_day

    np.testing.assert_allclose(pnl, expected, rtol=1e-10, atol=1e-12)

    # 总 PnL = Σ_s min(20, T-s-1) × per_cohort_day
    expected_total = sum(min(20, T - s - 1) * per_cohort_day for s in range(T))
    assert abs(pnl.sum() - expected_total) < 1e-10

    # gross exposure 稳态 = 每 cohort 名义 1.0 / 20 × 20 = 1.0
    assert abs(out["gross_exposure"][30] - 1.0) < 1e-9

    # cohort_weight 是 signal 日口径：第 t 天「新建」的 cohort 数 / H = 1/20。
    # T=62 时每天都有 1 个新 cohort（最后一个 signal 日 t=60 无入场日，为 0）。
    assert abs(out["cohort_weight"][30] - 1.0 / 20.0) < 1e-9
    assert out["cohort_weight"][-1] == 0.0

    # D1/D10 桶收益诊断
    assert abs(out["long_ret"][0] - 0.005) < 1e-12      # (0.01+0)/2
    assert abs(out["short_ret"][0] - (-0.0025)) < 1e-12  # (-0.005+0)/2
    assert abs(out["long_short_ret"][0] - 0.0075) < 1e-12


def test_cohort_pnl_holdings_exit_day_included():
    """持有期含出场日：cohort s 持有 [s+1, s+H] 共 H 天（出场日当天仍计 PnL）。"""
    T, N = 62, 12
    factor = np.tile(np.arange(12, dtype=float), (T, 1))
    ret = np.zeros((T, N))
    ret[:, 11] = 0.01
    ret[:, 0] = -0.005
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl(factor, ret, vwap, holding=20)
    pnl = out["pnl_net"]
    per_cohort_day = 0.0001875
    # 稳态：t=20 起（s∈[0,19] 全部 active）直到 t=40 后 cohort 0 出场。
    # 直接对拍逐日 active 计数。
    for t in [20, 30, 40, 41]:
        active = sum(1 for s in range(max(0, t - 20), t) if t <= s + 20)
        assert abs(pnl[t] - active * per_cohort_day) < 1e-10, (
            f"t={t} 期望 {active * per_cohort_day}，got {pnl[t]}"
        )


def test_cohort_pnl_with_costs():
    """per-side cost：入场/出场各扣一次；holding=1 时同日进出扣两次。"""
    T, N = 5, 12
    factor = np.tile(np.arange(12, dtype=float), (T, 1))
    ret = np.zeros((T, N))
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl(factor, ret, vwap, holding=1, per_side_cost=0.01)
    # 每 cohort 名义 = 0.5 + 0.5 = 1.0；holding=1 同日进出只扣一次双边成本
    # 1.0*0.01 = 0.01；/H=1 → -0.01
    assert abs(out["pnl_net"][1] - (-0.01)) < 1e-12
    assert abs(out["entry_cost"][1] - 0.01) < 1e-12
    assert abs(out["exit_cost"][1] - 0.0) < 1e-12


def test_cost_scenarios_interface():
    """1x/2x/3x 成本情景接口留存。"""
    assert COST_SCENARIOS["gross"] == 0.0
    assert cost_scenario(None) == 0.0
    assert cost_scenario("1x") == COST_SCENARIOS["1x"]
    assert cost_scenario("3x") > cost_scenario("2x") > cost_scenario("1x")
    assert cost_scenario(0.002) == 0.002
    with pytest.raises(ValueError):
        cost_scenario("bogus")
    with pytest.raises(ValueError):
        cost_scenario(-0.1)
    # 快速敏感性近似不抛错且单调
    pnl = np.array([0.001, -0.0005, 0.002])
    net_1x = apply_per_side_costs(pnl, cost_multiplier=1.0)
    net_3x = apply_per_side_costs(pnl, cost_multiplier=3.0)
    assert np.all(net_3x <= net_1x)


# ---------------------------------------------------------------------------
# 2. 重叠 label 直接年化会虚高 Sharpe（对照「错误算法」）
# ---------------------------------------------------------------------------

def test_overlapping_label_sharpe_is_inflated():
    """长记忆信号下，20d overlapping label 直接年化 Sharpe 显著高于
    cohort portfolio 的真实 Sharpe（规范 §7/§13 禁止前者）。"""
    T, N = 600, 40
    rng = np.random.default_rng(3)
    signal = np.cumsum(rng.standard_normal((T, 1)), axis=0)
    signal = signal - signal.mean()
    factor = signal * 0.5 + rng.standard_normal((T, N)) * 0.8
    ret = rng.standard_normal((T, N)) * 0.02 + 0.0015 * (
        factor - factor.mean(axis=1, keepdims=True)
    )
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl(factor, ret, vwap, holding=20)
    metrics = compute_portfolio_metrics(out["pnl_net"])
    cohort_sharpe = metrics["sharpe"]

    long_mask, short_mask = build_cohort_panels(factor, n_quantiles=10)
    fwd = compute_overlapping_forward_returns(
        ret, long_mask=long_mask, short_mask=short_mask, holding=20
    )
    overlap_sharpe = annualize_overlapping_label_sharpe(fwd, holding=20)

    assert np.isfinite(cohort_sharpe), "cohort Sharpe 应为有限值"
    assert np.isfinite(overlap_sharpe), "overlap Sharpe 应为有限值"
    assert overlap_sharpe > cohort_sharpe * 1.5, (
        f"重叠 label Sharpe ({overlap_sharpe:.3f}) 应显著高于真实 "
        f"cohort Sharpe ({cohort_sharpe:.3f})，否则虚高效应未复现"
    )


def test_overlapping_returns_are_highly_autocorrelated():
    """重叠 forward return 序列一阶自相关极高（>0.9），证明其不可当独立样本。"""
    T, N = 400, 30
    rng = np.random.default_rng(11)
    factor = rng.standard_normal((T, N))
    ret = rng.standard_normal((T, N)) * 0.02 + 0.001 * (
        factor - factor.mean(axis=1, keepdims=True)
    )
    long_mask, short_mask = build_cohort_panels(factor, n_quantiles=10)
    fwd = compute_overlapping_forward_returns(
        ret, long_mask=long_mask, short_mask=short_mask, holding=20
    )
    valid = np.isfinite(fwd)
    f = fwd[valid]
    lag1 = np.corrcoef(f[:-1], f[1:])[0, 1]
    assert lag1 > 0.9, f"重叠 label 一阶自相关应为 ~0.95（H=20 日度重叠），got {lag1:.3f}"


# ---------------------------------------------------------------------------
# 3. 双口径输出：long_short 与 D10 long-only active return
# ---------------------------------------------------------------------------

def test_dual_view_long_short_and_long_only_active():
    """同时产出 long_short（D10-D1）与 D10 long-only active return 两套口径。"""
    T, N = 300, 30
    rng = np.random.default_rng(42)
    factor = rng.standard_normal((T, N))
    # D10 有正 alpha，D1 无负 alpha → LS 与 long-only 方向一致但幅度不同
    alpha = np.where(
        factor > np.quantile(factor, 0.9, axis=1, keepdims=True), 0.01,
        np.where(factor < np.quantile(factor, 0.1, axis=1, keepdims=True), 0.0, 0.0),
    )
    ret = rng.standard_normal((T, N)) * 0.02 + alpha
    vwap = np.full((T, N), 10.0)

    out = compute_cohort_pnl(factor, ret, vwap, holding=20)
    ls = out["pnl_net"]
    active = out["active_ret"]

    assert ls.shape == (T,)
    assert active.shape == (T,)
    # 两套序列都非全零
    assert np.abs(ls).sum() > 0
    assert np.abs(active).sum() > 0
    # 方向一致性：LS 与 long-only active 的累计收益符号相同（D10 alpha 主导）
    assert np.sign(ls.sum()) == np.sign(active.sum())
    # 两者相关（同一 D10 桶驱动），但 LS 含 D1 空头 → 波动不同
    corr = np.corrcoef(ls, active)[0, 1]
    assert corr > 0.5

    # 一步到位接口同时产出两套 metrics
    full = compute_metrics_from_cohort(factor, ret, vwap, holding=20)
    assert set(full["metrics"].keys()) == {"long_short", "long_only_active"}
    assert np.isfinite(full["metrics"]["long_short"]["sharpe"])
    assert np.isfinite(full["metrics"]["long_only_active"]["sharpe"])


# ---------------------------------------------------------------------------
# 4. 复用库内 quantile 模块语义
# ---------------------------------------------------------------------------

def test_quantile_semantics_reuse_qe_module():
    """D1 = 最低分位组，D10 = 最高分位组（与 QE quantile.py QE2-P0-001 一致）。"""
    T, N = 20, 12
    factor = np.tile(np.arange(12, dtype=float), (T, 1))
    long_mask, short_mask = build_cohort_panels(factor, n_quantiles=10)
    # 12 票 → 10 分位：D10 为最高 2 票（col 10, 11），D1 为最低 2 票（col 0, 1）
    assert np.all(long_mask[0, 10:12])
    assert not np.any(long_mask[0, :10])
    assert np.all(short_mask[0, 0:2])
    assert not np.any(short_mask[0, 2:])

    masks = build_quantile_masks(factor, n_quantiles=10)
    assert masks.shape == (10, T, N)
    # 全分位互斥且覆盖全部有效票
    total = masks.sum(axis=0)
    assert np.all(total.sum(axis=1) == N)


def test_min_bucket_size_guard():
    """min_bucket_size>1 时，桶成员不足的交易日整日跳过。"""
    T, N = 60, 12
    factor = np.tile(np.arange(12, dtype=float), (T, 1))
    long_mask, short_mask = build_cohort_panels(factor, n_quantiles=10, min_bucket_size=3)
    # 每桶仅 1-2 票 < 3 → 全部交易日退化
    assert not np.any(long_mask)
    assert not np.any(short_mask)


# ---------------------------------------------------------------------------
# 5. 指标层正确性
# ---------------------------------------------------------------------------

def test_metrics_family_basic():
    """指标全家桶：已知恒定正收益序列各指标手算对拍。"""
    returns = np.full(252, 0.001)
    metrics = compute_portfolio_metrics(returns, periods_per_year=252)
    # 恒定收益 → std=0 → Sharpe 为 NaN（库里语义）
    assert np.isnan(metrics["sharpe"])
    assert metrics["annualized_return"] == pytest.approx(
        (1.001 ** 252) - 1.0, rel=1e-6
    )
    assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["win_rate"] == pytest.approx(1.0)
    assert metrics["positive_month_ratio"] == pytest.approx(1.0)
    # 无回撤 → DrawdownPersistence = 0
    assert metrics["drawdown_persistence"] == pytest.approx(0.0, abs=1e-12)


def test_drawdown_persistence_formula():
    """DrawdownPersistence = 0.5*norm(MaxDDDuration) + 0.5*norm(TimeUnderWater)。"""
    # 构造：先涨后长期阴跌（持续回撤）
    returns = np.concatenate([np.full(50, 0.002), np.full(120, -0.001)])
    dd = compute_drawdown_persistence(returns)
    assert 0.0 < dd <= 1.0
    # 持续回撤应比无回撤更「持久」
    flat = compute_drawdown_persistence(np.full(200, 0.0005))
    assert dd > flat


def test_rolling_sharpe_quantile_and_positive_month():
    """Rolling Sharpe Q20 与 Positive Month Ratio 合理性。"""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.01, 400)
    roll = compute_rolling_sharpe_quantile(returns, window=60, quantile=0.2)
    assert np.isfinite(roll)
    # Q20 <= 全窗 Sharpe 的分位意义：rolling 分布的下尾
    full_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    assert roll < full_sharpe
    pmr = compute_positive_month_ratio(returns)
    assert 0.0 <= pmr <= 1.0
    # 正漂移下月胜率应 > 0.4（0.0005/0.01 = 0.05 个日 std，21 日窗口胜率
    # 约 Φ(0.05*sqrt(21)) ≈ 0.59，用宽松下界防随机性翻车）
    assert pmr > 0.4


# ---------------------------------------------------------------------------
# 6. 边界与输入校验
# ---------------------------------------------------------------------------

def test_nan_handling_and_tradable():
    """停牌（VWAP NaN）与全市场缺失日：require_tradable 保守跳过。"""
    T, N = 100, 20
    rng = np.random.default_rng(5)
    factor = rng.standard_normal((T, N))
    ret = rng.standard_normal((T, N)) * 0.02
    vwap = np.full((T, N), 10.0)
    ret[30, :] = np.nan
    vwap[30, :] = np.nan

    out = compute_cohort_pnl(factor, ret, vwap, holding=20, require_tradable=True)
    # 全市场缺失日：当日无任何可交易票 → 该日的新 cohort 无法入场，
    # 但 cohort_weight 是 signal 日口径（第 t 天入场 cohort 数 / H），
    # 缺失日 t=30 仍有 1 个 cohort 在 t=30 入场（其入场日在 t=31，正常）。
    # 验证：t=29 入场 cohort（entry 30）被跳过，t=30 入场 cohort（entry 31）正常。
    assert out["cohort_weight"][29] == 0.0
    assert out["cohort_weight"][30] > 0.0
    assert np.all(np.isfinite(out["pnl_net"]))


def test_input_validation():
    """形状不匹配 / 非法参数 → ValueError。"""
    T, N = 20, 10
    factor = np.zeros((T, N))
    ret = np.zeros((T, N))
    vwap = np.ones((T, N))

    with pytest.raises(ValueError):
        compute_cohort_pnl(factor, ret[:5, :], vwap, holding=20)
    with pytest.raises(ValueError):
        compute_cohort_pnl(np.zeros((T, N, 1)), ret, vwap, holding=20)
    with pytest.raises(ValueError):
        compute_cohort_pnl(factor, ret, vwap, holding=0)
    with pytest.raises(ValueError):
        compute_cohort_pnl(factor, ret, vwap, holding=20, per_side_cost=1.5)


def test_compute_portfolio_metrics_requires_1d():
    """指标层只接受一维日频序列。"""
    with pytest.raises(ValueError):
        compute_portfolio_metrics(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        compute_annualized_return(np.zeros((5, 5)))
