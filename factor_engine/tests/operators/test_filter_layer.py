# -*- coding: utf-8 -*-
"""Filter Layer测试套件（2026-08-12信号滤波层专项）。

测试 ts_hampel_filter_causal 的：
  - 注册与元数据
  - 严格因果性（只使用过去窗口）
  - clip vs median 替换策略
  - 保留小跳变、压制大毛刺
  - scale_floor 防除零
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def registry():
    """加载完整算子注册表。"""
    # Import only what's needed for filter layer tests
    import factor_engine.cleaned_operators.filter_despike
    import factor_engine.cleaned_operators.filter_smooth
    import factor_engine.cleaned_operators.filter_hysteresis
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry


def test_hampel_filter_causal_registration(registry):
    """验证 ts_hampel_filter_causal 已正确注册。"""
    assert "ts_hampel_filter_causal" in registry.list_canonical()
    backends = registry.backends_for("ts_hampel_filter_causal")
    assert "pandas_numpy" in backends

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    assert op.metadata.name == "ts_hampel_filter_causal"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "despike"
    assert contract.causal is True
    assert contract.uses_current_observation is True
    assert contract.stateful is False


def test_hampel_uses_past_window_only(registry):
    """验证严格因果性：当前点不参与自己的阈值估计。"""
    # 构造信号：前面平稳，t=5处有极大毛刺
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "A": [1.0, 1.1, 0.9, 1.05, 0.95, 100.0, 1.0, 1.1, 0.9, 1.05],
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, n_sigma=3.0, replacement="clip")

    # t=5 (index=5): 过去4个点 [1.05, 0.95, 1.1, 0.9]，中位数 ~1.0
    # MAD = 1.4826 * median(|x - 1.0|) ≈ 1.4826 * 0.075 ≈ 0.11
    # 阈值 = 3 * 0.11 ≈ 0.33
    # 毛刺 100.0 偏离 ~99，远超阈值，应被clip
    assert result.loc[dates[5], "A"] < 10.0, "毛刺应被压制"
    assert result.loc[dates[5], "A"] > 0.0, "clip应保留方向"

    # 前面和后面的正常点应保持不变
    assert abs(result.loc[dates[0], "A"] - 1.0) < 0.01
    assert abs(result.loc[dates[6], "A"] - 1.0) < 0.01


def test_confidence_weighted_ema_adapts_alpha(registry):
    """验证置信度调节EMA：高置信快速更新，低置信平滑。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    # 构造信号和置信度
    data = pd.DataFrame({
        "signal": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        "confidence": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }, index=dates)

    op = registry.get("state_confidence_weighted_ema", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(
        data[["signal"]],
        data[["confidence"]].set_axis(["signal"], axis=1),
        alpha_min=0.05,
        alpha_max=0.5
    )

    # 低置信期（t=0-2）：alpha=alpha_min=0.05，更新慢
    # 高置信期（t=3-5）：alpha=alpha_max=0.5，更新快
    # 低置信期（t=6-9）：alpha=alpha_min=0.05，更新慢

    # t=3时从低置信转高置信，之后几步应快速跟上
    val_before_high_conf = result.loc[dates[2], "signal"]
    val_after_high_conf = result.loc[dates[5], "signal"]

    # 高置信期应显著拉近原始信号
    assert val_after_high_conf > val_before_high_conf + 0.2, "高置信期应快速更新"

    # t=6转回低置信，后续更新应变慢
    val_t6 = result.loc[dates[6], "signal"]
    val_t9 = result.loc[dates[9], "signal"]
    gap_t9 = abs(1.9 - val_t9)

    # 低置信期滞后应明显
    assert gap_t9 > 0.1, "低置信期应保持平滑"


def test_uncertainty_deadband_threshold(registry):
    """验证不确定度阈值滤波器：变化未超过uncertainty时不更新。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    # 构造信号和不确定度
    data = pd.DataFrame({
        "signal": [10.0, 10.5, 11.0, 11.2, 11.3, 15.0, 15.1, 15.2, 15.0, 14.8],
        "uncertainty": [0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5],
    }, index=dates)

    op = registry.get("state_uncertainty_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(
        data[["signal"]],
        data[["uncertainty"]].set_axis(["signal"], axis=1),
        k_sigma=1.0
    )

    # t=0: 初始化为10.0
    assert abs(result.loc[dates[0], "signal"] - 10.0) < 0.01

    # t=1: 变化0.5，uncertainty=0.2，threshold=0.2*1.0=0.2，0.5>0.2，应更新
    assert abs(result.loc[dates[1], "signal"] - 10.5) < 0.01

    # t=3: 从11.0到11.2，变化0.2=0.2，边界情况不更新，保持11.0
    assert abs(result.loc[dates[3], "signal"] - 11.0) < 0.01

    # t=4: 从11.0到11.3，变化0.3>0.2，应更新
    assert abs(result.loc[dates[4], "signal"] - 11.3) < 0.01

    # t=5: 从11.3到15.0，变化3.7>>0.2，应更新
    assert abs(result.loc[dates[5], "signal"] - 15.0) < 0.01

    # t=6-9: uncertainty=0.5，threshold=0.5，小幅波动应被过滤
    # t=6: 15.1-15.0=0.1<0.5，保持15.0
    assert abs(result.loc[dates[6], "signal"] - 15.0) < 0.01

    # t=7: 15.2-15.0=0.2<0.5，保持15.0
    assert abs(result.loc[dates[7], "signal"] - 15.0) < 0.01


def test_hampel_clip_vs_median_replacement(registry):
    """验证 clip 和 median 两种替换策略。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "A": [10.0, 11.0, 9.0, 10.5, 9.5, 50.0, 10.0, 11.0],
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata

    # clip模式：保留方向，压制幅度
    result_clip = op.calculate(data, window=5, n_sigma=3.0, replacement="clip")
    spike_clip = result_clip.loc[dates[5], "A"]
    assert spike_clip > 10.0, "clip应保留上升方向"
    assert spike_clip < 50.0, "clip应压制幅度"

    # median模式：直接替换为中位数
    result_median = op.calculate(data, window=5, n_sigma=3.0, replacement="median")
    spike_median = result_median.loc[dates[5], "A"]
    assert 9.0 < spike_median < 11.0, "median应替换为窗口中位数"


def test_hampel_preserves_small_jumps_clips_spikes(registry):
    """验证保留真实小跳变，压制异常大毛刺。"""
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    # Use data with natural variance so MAD > 0
    data = pd.DataFrame({
        "A": [
            10.0, 10.2, 9.8, 10.1,    # 平稳期（有自然波动）
            9.9, 11.5,                 # t=5: 小跳变
            10.0, 10.1, 9.9,           # 恢复
            50.0,                      # t=9: 大毛刺
            10.0, 10.2,                # 恢复
        ],
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, n_sigma=3.0, replacement="clip")

    # 独立 Hampel oracle: median=10.0, MAD=0.1, threshold=3*1.4826*0.1.
    expected_small_clip = 10.0 + 3.0 * 1.4826 * 0.1
    assert result.loc[dates[5], "A"] == pytest.approx(expected_small_clip)

    # 大毛刺应被压制
    assert result.loc[dates[9], "A"] < 20.0, "大毛刺应被clip"
    assert result.loc[dates[9], "A"] > 10.0, "clip后应仍高于中位数"


def test_hampel_scale_floor_prevents_division_by_zero(registry):
    """验证 scale_floor 防止MAD为零时的除零错误。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    # 前面完全平稳（MAD=0），然后出现跳变
    data = pd.DataFrame({
        "A": [10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    # 即使MAD=0，scale_floor也能防止除零
    result = op.calculate(data, window=5, n_sigma=3.0, replacement="clip", scale_floor=1e-10)

    # t=5: 过去窗口全是10.0，MAD=0 → 使用scale_floor
    # 阈值 = 3 * 1e-10（极小），所以 15.0 会被视为毛刺并clip
    # 但由于阈值极小，clip后仍接近10.0
    assert np.isfinite(result.loc[dates[5], "A"]), "不应出现NaN"
    assert result.loc[dates[5], "A"] >= 10.0, "结果应有效"


def test_hampel_handles_nan_gracefully(registry):
    """验证对NaN的正确处理。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "A": [10.0, np.nan, 10.0, 10.0, 10.0, 50.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, n_sigma=3.0, replacement="clip")

    # NaN输入应输出NaN
    assert pd.isna(result.loc[dates[1], "A"])

    # 有NaN的窗口：只使用有效值估计
    # t=5: 窗口包含[nan, 10, 10, 10]，使用3个有效值
    # 有限历史值完全平坦，契约的 zero-scale policy 是 bypass，而不是让数值
    # epsilon 决定经济跳变是否显著。
    assert result.loc[dates[5], "A"] == 50.0


def test_l1_turnover_prox_registration(registry):
    """验证 state_l1_turnover_prox 已正确注册。"""
    assert "state_l1_turnover_prox" in registry.list_canonical()
    backends = registry.backends_for("state_l1_turnover_prox")
    assert "pandas_numpy" in backends

    op = registry.get("state_l1_turnover_prox", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    meta = op.metadata
    assert op.metadata.name == "state_l1_turnover_prox"
    assert meta.category == "signal_filter"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "rate_limit"
    assert contract.causal is True
    assert contract.stateful is True


def test_l1_turnover_prox_soft_threshold(registry):
    """验证 L1 proximal operator 的 soft-threshold 行为。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # 构造信号：初始值10，然后小幅波动
    data = pd.DataFrame({
        "A": [10.0, 10.5, 9.8, 10.2, 9.9, 11.5, 8.5, 10.1, 10.3, 9.7],
    }, index=dates)

    op = registry.get("state_l1_turnover_prox", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, lambda_turnover=1.0)

    # t=0: 初始化为10.0
    assert abs(result.loc[dates[0], "A"] - 10.0) < 0.01

    # t=1: delta=0.5 < 1.0，应保持在10.0（no trade）
    assert abs(result.loc[dates[1], "A"] - 10.0) < 0.01

    # t=2: delta=9.8-10.0=-0.2 < 1.0，应保持在10.0
    assert abs(result.loc[dates[2], "A"] - 10.0) < 0.01

    # t=5: x=11.5, y_prev约10.0, delta=1.5 > 1.0
    # y_new = 10.0 + (1.5 - 1.0) = 10.5
    assert abs(result.loc[dates[5], "A"] - 10.5) < 0.1

    # t=6: x=8.5, y_prev约10.5, delta=-2.0 < -1.0
    # y_new = 10.5 + (-2.0 + 1.0) = 9.5
    assert abs(result.loc[dates[6], "A"] - 9.5) < 0.1


def test_l1_turnover_prox_lambda_effect(registry):
    """验证 lambda_turnover 参数对 no-trade band 的影响。"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "A": [10.0, 11.5, 12.5, 9.0, 10.5, 11.0],
    }, index=dates)

    op = registry.get("state_l1_turnover_prox", backend="pandas_numpy")

    meta = op.metadata

    # 小 lambda: 窄 band，更多交易
    result_small = op.calculate(data, lambda_turnover=0.5)
    # 大 lambda: 宽 band，更少交易
    result_large = op.calculate(data, lambda_turnover=2.0)

    # t=1: x=11.5, y_prev=10.0, delta=1.5
    # lambda=0.5: 1.5 > 0.5，会更新 → y=10.0+(1.5-0.5)=11.0
    # lambda=2.0: 1.5 < 2.0，不更新 → y=10.0
    assert result_small.loc[dates[1], "A"] > 10.5
    assert abs(result_large.loc[dates[1], "A"] - 10.0) < 0.01


def test_l2_partial_adjustment_registration(registry):
    """验证 state_l2_partial_adjustment 已正确注册。"""
    assert "state_l2_partial_adjustment" in registry.list_canonical()
    backends = registry.backends_for("state_l2_partial_adjustment")
    assert "pandas_numpy" in backends

    op = registry.get("state_l2_partial_adjustment", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    meta = op.metadata
    assert op.metadata.name == "state_l2_partial_adjustment"
    assert meta.category == "signal_filter"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "rate_limit"
    assert contract.causal is True
    assert contract.stateful is True


def test_l2_partial_adjustment_smooth(registry):
    """验证 L2 partial adjustment 的平滑调整行为。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    # 信号从10跳到20，观察调整过程
    data = pd.DataFrame({
        "A": [10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    }, index=dates)

    op = registry.get("state_l2_partial_adjustment", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, lambda_smooth=1.0)

    # t=0,1: 初始化
    assert abs(result.loc[dates[0], "A"] - 10.0) < 0.01
    assert abs(result.loc[dates[1], "A"] - 10.0) < 0.01

    # t=2: x=20, y_prev=10, lambda=1.0
    # y = (20 + 1.0*10) / (1+1.0) = 30/2 = 15.0
    assert abs(result.loc[dates[2], "A"] - 15.0) < 0.01

    # t=3: x=20, y_prev=15
    # y = (20 + 1.0*15) / 2 = 35/2 = 17.5
    assert abs(result.loc[dates[3], "A"] - 17.5) < 0.01

    # 应该逐步收敛到20，但每步只调整一半差距
    assert result.loc[dates[4], "A"] < 20.0
    assert result.loc[dates[5], "A"] < 20.0
    # 最终应接近20
    assert abs(result.loc[dates[7], "A"] - 20.0) < 0.5


def test_l2_partial_adjustment_lambda_effect(registry):
    """验证 lambda_smooth 参数对调整速度的影响。"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "A": [10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    }, index=dates)

    op = registry.get("state_l2_partial_adjustment", backend="pandas_numpy")

    meta = op.metadata

    # lambda_smooth 小: 快速调整
    result_fast = op.calculate(data, lambda_smooth=0.5)
    # lambda_smooth 大: 慢速调整
    result_slow = op.calculate(data, lambda_smooth=5.0)

    # t=1: x=20, y_prev=10
    # lambda=0.5: y = (20 + 0.5*10) / 1.5 = 25/1.5 ≈ 16.67
    # lambda=5.0: y = (20 + 5.0*10) / 6.0 = 70/6 ≈ 11.67
    assert result_fast.loc[dates[1], "A"] > 15.0
    assert result_slow.loc[dates[1], "A"] < 13.0

    # t=3: fast应更接近20
    assert result_fast.loc[dates[3], "A"] > result_slow.loc[dates[3], "A"]


def test_l1_l2_handles_nan(registry):
    """验证两个算子对NaN的正确处理。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "A": [10.0, 12.0, np.nan, 15.0, np.nan, 18.0, 20.0, 22.0],
    }, index=dates)

    op_l1 = registry.get("state_l1_turnover_prox", backend="pandas_numpy")
    op_l2 = registry.get("state_l2_partial_adjustment", backend="pandas_numpy")

    result_l1 = op_l1.calculate(data, lambda_turnover=1.0)
    result_l2 = op_l2.calculate(data, lambda_smooth=1.0)

    # NaN输入应输出NaN
    assert pd.isna(result_l1.loc[dates[2], "A"])
    assert pd.isna(result_l2.loc[dates[2], "A"])
    assert pd.isna(result_l1.loc[dates[4], "A"])
    assert pd.isna(result_l2.loc[dates[4], "A"])

    # NaN后重新初始化
    assert np.isfinite(result_l1.loc[dates[3], "A"])
    assert np.isfinite(result_l2.loc[dates[3], "A"])
    assert np.isfinite(result_l1.loc[dates[5], "A"])
    assert np.isfinite(result_l2.loc[dates[5], "A"])



def test_hampel_warmup_period(registry):
    """验证预热期行为：窗口不足时直接输出原值。"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "A": [100.0, 200.0, 1.0, 1.0, 1.0, 1.0],  # 前两个是毛刺
    }, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, n_sigma=3.0, replacement="clip")

    # t=0,1,2,3: 窗口不足4个历史点，应直接输出原值
    assert result.loc[dates[0], "A"] == 100.0, "预热期应保留原值"
    assert result.loc[dates[1], "A"] == 200.0, "预热期应保留原值"
    assert result.loc[dates[2], "A"] == 1.0
    assert result.loc[dates[3], "A"] == 1.0

    # t=4: 有4个历史点[100, 200, 1, 1]，可以检测
    # t=5: 有4个历史点[200, 1, 1, 1]，可以检测
    assert np.isfinite(result.loc[dates[4], "A"])
    assert np.isfinite(result.loc[dates[5], "A"])


def test_hampel_parameter_validation(registry):
    """验证参数校验。"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({"A": [1.0] * 6}, index=dates)

    op = registry.get("ts_hampel_filter_causal", backend="pandas_numpy")

    meta = op.metadata

    # window < 2 应报错
    with pytest.raises(ValueError, match="window"):
        op.calculate(data, window=1)

    # n_sigma <= 0 应报错
    with pytest.raises(ValueError, match=r"(?:ts_hampel_filter_causal\.n_sigma \[runtime\]: n_sigma must be >= 1\.0|ts_hampel_filter_causal requires n_sigma > 0, got 0\.0)"):
        op.calculate(data, window=5, n_sigma=0.0)

    # scale_floor <= 0 应报错
    with pytest.raises(ValueError, match="scale_floor > 0"):
        op.calculate(data, window=5, scale_floor=0.0)

    # 无效的 replacement 应报错
    with pytest.raises(ValueError, match="replacement must be"):
        op.calculate(data, window=5, replacement="invalid")


# ============================================================================
# Hysteresis and Turnover Control Tests (2026-08-12 P0)
# ============================================================================


def test_adaptive_deadband_registration(registry):
    """验证 state_adaptive_deadband 已正确注册。"""
    assert "state_adaptive_deadband" in registry.list_canonical()
    backends = registry.backends_for("state_adaptive_deadband")
    assert "pandas_numpy" in backends


def test_adaptive_deadband_dimensionless(registry):
    """验证自适应死区对无量纲信号的换手减少效果。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    # 构造带噪声的信号：基础趋势 + 小幅抖动
    np.random.seed(42)
    base = np.linspace(0, 1, 30)
    noise = 0.02 * np.random.randn(30)
    data = pd.DataFrame({"A": base + noise}, index=dates)

    op = registry.get("state_adaptive_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, band_mult=1.5, scale_window=10, scale_method="mad_delta")

    # 验证输出有效且平滑
    assert result.notna().all().all()
    # 自适应死区应减少小幅抖动，输出更平滑
    original_changes = np.abs(data["A"].diff()).sum()
    filtered_changes = np.abs(result["A"].diff()).sum()
    assert filtered_changes < original_changes, "死区应减少换手"


def test_adaptive_deadband_scale_methods(registry):
    """验证 mad_delta 和 std_delta 两种尺度估计方法。"""
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    np.random.seed(123)
    data = pd.DataFrame({"A": np.cumsum(np.random.randn(25) * 0.1)}, index=dates)

    op = registry.get("state_adaptive_deadband", backend="pandas_numpy")

    meta = op.metadata

    # MAD方法（鲁棒）
    result_mad = op.calculate(data, band_mult=1.0, scale_window=10, scale_method="mad_delta")
    # STD方法（敏感）
    result_std = op.calculate(data, band_mult=1.0, scale_window=10, scale_method="std_delta")

    # 两者都应有效且平滑原信号
    assert result_mad.notna().all().all()
    assert result_std.notna().all().all()


def test_rank_deadband_cross_sectional_stability(registry):
    """验证排名空间死区的横截面稳定性。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    # 三只股票：A稳定高位，B稳定中位，C稳定低位
    np.random.seed(456)
    data = pd.DataFrame({
        "A": 10.0 + 0.2 * np.random.randn(20),  # 高位小抖动
        "B": 5.0 + 0.2 * np.random.randn(20),   # 中位小抖动
        "C": 1.0 + 0.2 * np.random.randn(20),   # 低位小抖动
    }, index=dates)

    op = registry.get("state_rank_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, band_pct=0.15, group=None)

    # 验证输出有效
    assert result.notna().all().all()

    # 排名相对稳定时，输出应保持不变（减少换手）
    # A应保持接近初始高值，C应保持接近初始低值
    original_changes = (data != data.shift()).sum().sum()
    filtered_changes = (result != result.shift()).sum().sum()
    # 由于排名稳定，死区应显著减少更新次数
    assert filtered_changes < original_changes * 0.8


def test_rank_deadband_with_group(registry):
    """验证分组排名死区。"""
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    data = pd.DataFrame({
        "A": np.linspace(1, 3, 15),
        "B": np.linspace(2, 4, 15),
        "C": np.linspace(0.5, 2.5, 15),
    }, index=dates)
    # 分组：A和B一组，C单独一组
    group = pd.DataFrame({
        "A": [1] * 15,
        "B": [1] * 15,
        "C": [2] * 15,
    }, index=dates)

    op = registry.get("state_rank_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, band_pct=0.1, group=group)

    # 验证输出有效
    assert result.notna().all().all()
    # 分组内排名变化缓慢时，应减少更新


def test_quantile_hysteresis_different_thresholds(registry):
    """验证分位数迟滞的双阈值状态机。"""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    # 构造信号：三只股票，一只逐渐上升到顶部然后回落
    data = pd.DataFrame({
        "A": [0.5] * 10 + list(np.linspace(0.5, 1.5, 15)) + [1.0] * 10 + list(np.linspace(1.0, 0.3, 5)),
        "B": [0.8] * 40,  # 中位稳定
        "C": [0.2] * 40,  # 低位稳定
    }, index=dates)

    op = registry.get("state_quantile_hysteresis", backend="pandas_numpy")

    meta = op.metadata
    # enter_quantile=0.9, exit_quantile=0.6
    result = op.calculate(data, enter_quantile=0.9, exit_quantile=0.6, group=None)

    # 验证输出是0/1二值
    assert result.isin([0.0, 1.0, np.nan]).all().all()

    # A在上升到高分位时应进入状态1
    # 找到A的最大值附近（应该是1）
    a_series = result["A"]
    max_idx = data["A"].idxmax()
    # A在峰值附近应为1
    assert a_series.loc[max_idx] == 1.0

    # B starts as the top-ranked name, enters, and correctly remains latched while
    # its rank stays above the 0.6 exit threshold.
    assert (result["B"] == 1.0).all()
    assert (result["C"] == 0.0).sum() > 35


def test_quantile_hysteresis_state_persistence(registry):
    """验证迟滞状态的持续性（减少抖动）。"""
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    # 构造横截面：一个股票在边界附近震荡
    data = pd.DataFrame({
        "A": [0.95, 0.88, 0.92, 0.87, 0.93, 0.89, 0.91, 0.88, 0.94, 0.87,
              0.92, 0.89, 0.90, 0.88, 0.91, 0.89, 0.90, 0.88, 0.92, 0.87,
              0.91, 0.89, 0.90, 0.88, 0.91],
        "B": [0.5] * 25,   # 低位
        "C": [0.3] * 25,   # 更低位
    }, index=dates)

    op = registry.get("state_quantile_hysteresis", backend="pandas_numpy")

    meta = op.metadata
    # 窄阈值带：enter=0.9, exit=0.85，A在此范围震荡
    result = op.calculate(data, enter_quantile=0.9, exit_quantile=0.85, group=None)

    # 验证状态持续性：一旦进入1，不会因小幅波动立即退出
    a_series = result["A"]
    # 状态转换次数应少于信号波动次数
    state_changes = (a_series.diff() != 0).sum()
    # A值在边界震荡多次，但状态应相对稳定
    assert state_changes < 15, "迟滞应减少状态抖动"


def test_hysteresis_parameter_validation(registry):
    """验证迟滞算子的参数校验。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": np.linspace(0, 1, 10)}, index=dates)

    # state_adaptive_deadband
    op_adaptive = registry.get("state_adaptive_deadband", backend="pandas_numpy")
    with pytest.raises(ValueError, match="scale_window"):
        op_adaptive.calculate(data, scale_window=1)
    with pytest.raises(ValueError, match=r"state_adaptive_deadband\.band_mult \[runtime\]: band_mult must be >= 0\.0"):
        op_adaptive.calculate(data, band_mult=-1.0)

    # state_rank_deadband
    op_rank = registry.get("state_rank_deadband", backend="pandas_numpy")
    with pytest.raises(ValueError, match=r"state_rank_deadband\.band_pct \[runtime\]: band_pct must be <= 1\.0"):
        op_rank.calculate(data, band_pct=1.5)

    # state_quantile_hysteresis
    op_quant = registry.get("state_quantile_hysteresis", backend="pandas_numpy")
    with pytest.raises(ValueError, match=r"state_quantile_hysteresis\.enter_quantile \[runtime\]: enter_quantile must be <= 1\.0"):
        op_quant.calculate(data, enter_quantile=1.5)
    with pytest.raises(ValueError, match="exit_quantile <= enter_quantile"):
        op_quant.calculate(data, enter_quantile=0.7, exit_quantile=0.9)



# ============================================================================
# ts_super_smoother tests (two-pole IIR low-pass filter)
# ============================================================================

def test_super_smoother_registration_contract_metadata(registry):
    """验证 ts_super_smoother 已正确注册。"""
    assert "ts_super_smoother" in registry.list_canonical()
    backends = registry.backends_for("ts_super_smoother")
    assert "pandas_numpy" in backends

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_super_smoother"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "low_pass"
    assert contract.causal is True
    assert contract.stateful is True
    # No serializer/restore kernel exists yet; the runtime requires full replay.
    assert contract.checkpointable is False
    assert contract.time_shard_safe is False


def test_super_smoother_stronger_attenuation_than_ema_noisy_trend(registry):
    """验证 super smoother 的衰减强于单极 EMA。

    两极滤波器在截止频率以上的衰减速度是单极的两倍（-40dB/decade vs -20dB/decade）。
    用高频噪声信号验证：super smoother 的输出方差应小于 EMA。
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")

    # Use a pure, deterministic high-frequency component so trend variance does
    # not contaminate the attenuation oracle.
    t = np.arange(200)
    signal = np.sin(2.0 * np.pi * t / 4.0)

    data = pd.DataFrame({"A": signal}, index=dates)

    # Super smoother
    op_ss = registry.get("ts_super_smoother", backend="pandas_numpy")
    result_ss = op_ss.calculate(data, period=20)

    # 简单 EMA (作为对照)
    ema = data["A"].ewm(span=20, adjust=False).mean()

    # Compare steady-state RMS gain at the same input frequency.
    warmup = 40
    var_ss = result_ss.iloc[warmup:, 0].var()
    var_ema = ema.iloc[warmup:].var()

    # Super smoother 的方差应显著小于 EMA（更强的噪声抑制）
    assert var_ss < var_ema * 0.6, "Super smoother 应有更强的高频衰减"


def test_super_smoother_stateful_needs_two_lags_increasing_series(registry):
    """验证 super smoother 需要两个历史状态初始化。

    前两个有限观测用于初始化 y_0 和 y_1，从第三个观测开始递归。
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, period=10)

    # 前两个观测应直接输出（初始化）
    assert result.iloc[0, 0] == 1.0, "y_0 = x_0"
    assert result.iloc[1, 0] == 2.0, "y_1 = x_1"

    # 从第三个点开始递归滤波
    assert result.iloc[2, 0] != 3.0, "y_2 应是滤波后的值"
    assert 2.0 < result.iloc[2, 0] < 3.5, "y_2 应在合理范围"

    # 滤波器应平滑递增趋势
    for i in range(3, 9):
        assert result.iloc[i, 0] > result.iloc[i-1, 0], "平滑趋势应单调递增"


def test_super_smoother_step_response_no_overshoot_large_step(registry):
    """验证 Butterworth-like 滤波器的阶跃响应无过冲。

    标准 Butterworth 设计的特点是通带平坦、无振铃（maximally flat）。
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # 阶跃信号：前50个点为0，后50个点为10
    signal = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, period=10)

    # 阶跃后的响应应单调上升到目标值，无过冲
    step_idx = 50
    response = result.iloc[step_idx:step_idx+30, 0].values

    # A two-pole Butterworth-like response can ring slightly; pin a bounded
    # response and convergence instead of the mathematically false monotonicity
    # claim.
    assert response.max() <= 10.5, "阶跃响应不得有超过5%的显著过冲"
    assert response[-1] == pytest.approx(10.0, abs=0.01)

    # 最终应收敛到目标值附近
    assert response[-1] > 8.0, "应接近目标值10.0"


def test_super_smoother_handles_nan_correctly(registry):
    """验证对 NaN 的处理：输出 NaN，状态保持冻结。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, period=5)

    # NaN 位置应输出 NaN
    assert pd.isna(result.iloc[3, 0])

    # NaN 前后的观测应正常滤波
    assert np.isfinite(result.iloc[2, 0])
    assert np.isfinite(result.iloc[4, 0])


def test_super_smoother_parameter_validation(registry):
    """验证参数校验。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": [1.0] * 10}, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata

    # period < 3 应报错（两极滤波器不稳定）
    with pytest.raises(ValueError, match="period >= 3"):
        op.calculate(data, period=2)


def test_super_smoother_certified_periods(registry):
    """验证认证参数值可以正常运行。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    signal = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata

    # 认证参数: [5, 10, 20, 40]
    for period in [5, 10, 20, 40]:
        result = op.calculate(data, period=period)
        assert result.shape == data.shape
        assert result.notna().sum().iloc[0] >= 2, f"period={period} 应有有效输出"

    with pytest.raises((TypeError, ValueError), match="undeclared keyword.*clip_sigma"):
        op.calculate(data, clip_sigma=0)

    # warmup_window <= 0 应报错
    with pytest.raises((TypeError, ValueError), match="undeclared keyword.*warmup_window"):
        op.calculate(data, warmup_window=0)

    # scale_floor <= 0 应报错
    with pytest.raises((TypeError, ValueError), match="undeclared keyword.*scale_floor"):
        op.calculate(data, scale_floor=0)


# ============================================================================
# ts_kama Tests (Kaufman Adaptive Moving Average)
# ============================================================================


def test_kama_registration(registry):
    """验证 ts_kama 已正确注册。"""
    assert "ts_kama" in registry.list_canonical()
    backends = registry.backends_for("ts_kama")
    assert "pandas_numpy" in backends

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_kama"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "adaptive_low_pass"
    assert contract.causal is True
    assert contract.stateful is True
    # KAMA restoration needs both recursive output and ER input-ring history.
    assert contract.checkpointable is False


def test_kama_high_er_follows_fast(registry):
    """验证高效率比（强趋势）下KAMA快速响应。"""
    # 构造干净上升趋势（高效率）
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    trend = np.linspace(100, 150, 50)  # 完美线性趋势
    data = pd.DataFrame({"A": trend}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=10, fast_period=2, slow_period=30)

    # 预热后，KAMA应紧跟价格（ER ≈ 1 → SC ≈ fast_alpha² ≈ 0.44）
    warmup = 10
    assert result.isna().sum().sum() == warmup

    # 检查KAMA与趋势的滞后较小
    valid_idx = result["A"].notna()
    lag = (data.loc[valid_idx, "A"] - result.loc[valid_idx, "A"]).abs()
    assert lag.mean() < 5.0, "强趋势中KAMA应紧跟价格"


def test_kama_low_er_smooths_heavy(registry):
    """验证低效率比（震荡市）下KAMA强力平滑。"""
    # 构造震荡横盘市场（低效率）
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    noise = np.random.RandomState(42).randn(50) * 2
    choppy = 100 + noise  # 均值回归噪声
    data = pd.DataFrame({"A": choppy}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=10, fast_period=2, slow_period=30)

    # KAMA输出应比输入更平滑
    valid_idx = result["A"].notna()
    input_vol = data.loc[valid_idx, "A"].diff().std()
    output_vol = result.loc[valid_idx, "A"].diff().std()

    assert output_vol < input_vol, "震荡市中KAMA应强力平滑"


def test_kama_efficiency_ratio_bounded_zero_one(registry):
    """验证效率比保持在[0,1]区间，递归稳定。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    # 混合场景：趋势 + 噪声
    trend = np.linspace(100, 120, 100)
    noise = np.random.RandomState(123).randn(100) * 3
    price = trend + noise
    data = pd.DataFrame({"A": price}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=10, fast_period=2, slow_period=30)

    # KAMA应在预热后产生有限输出
    assert result["A"].notna().sum() > 0
    assert np.all(np.isfinite(result["A"].dropna()))

    # 验证无爆炸行为（ER有界 → SC有界 → 递归稳定）
    valid = result["A"].dropna()
    assert valid.min() > 80, "KAMA不应坍缩"
    assert valid.max() < 140, "KAMA不应爆炸"


def test_kama_stateful_recursive(registry):
    """验证状态依赖：y_t依赖y_{t-1}，非单纯滚动窗口。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    price = np.concatenate([
        np.full(15, 100.0),  # 稳定期
        np.full(15, 110.0),  # 跳变
    ])
    data = pd.DataFrame({"A": price}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=5, fast_period=2, slow_period=20)

    # 跳变后KAMA应逐步调整（非瞬时）
    valid = result["A"].dropna()
    assert len(valid) > 0

    # 末值应接近新水平（110）但可能未完全收敛
    assert valid.iloc[-1] > 100, "KAMA应在跳变后调整"
    assert valid.iloc[-1] <= 110, "KAMA不应过冲"


def test_kama_missing_data_break_rewarm(registry):
    """验证NaN输入使递归状态失效，需重新预热er_window+1期。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    price = np.full(30, 100.0)
    price[15:18] = np.nan  # 数据缺口
    data = pd.DataFrame({"A": price}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=5, fast_period=2, slow_period=20)

    # 缺口前：初始预热后应有有效KAMA（5+1=6期）
    assert result.loc[dates[6], "A"] == 100.0

    # 缺口期间：NaN
    assert result.loc[dates[15:18], "A"].isna().all()

    # 缺口后：需再次预热6期
    # 索引18-23是缺口后6期
    assert result.loc[dates[18:23], "A"].isna().all()
    # 索引24应有有效输出（18+6=24）
    assert pd.notna(result.loc[dates[24], "A"])


def test_kama_fast_must_be_less_than_slow(registry):
    """验证关系约束：fast_period < slow_period。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"A": np.arange(20, dtype=float) + 100}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata

    with pytest.raises(ValueError, match="fast_period must be < slow_period"):
        op.calculate(data, er_window=10, fast_period=30, slow_period=20)

    with pytest.raises(ValueError, match="fast_period must be < slow_period"):
        op.calculate(data, er_window=10, fast_period=20, slow_period=20)


def test_kama_er_window_minimum(registry):
    """验证er_window >= 2。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"A": np.arange(20, dtype=float) + 100}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata

    with pytest.raises(ValueError, match="er_window must be >= 2"):
        op.calculate(data, er_window=1, fast_period=2, slow_period=20)


def test_kama_period_minimum(registry):
    """验证fast_period和slow_period >= 1。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"A": np.arange(20, dtype=float) + 100}, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata

    with pytest.raises(ValueError, match="fast_period must be >= 1"):
        op.calculate(data, er_window=10, fast_period=0, slow_period=20)

    with pytest.raises(ValueError, match="slow_period must be >= 1"):
        op.calculate(data, er_window=10, fast_period=2, slow_period=0)


def test_kama_multi_column_independent(registry):
    """验证多列面板独立处理。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    data = pd.DataFrame({
        "A": np.linspace(100, 120, 30),  # 上升趋势
        "B": np.linspace(100, 80, 30),   # 下降趋势
    }, index=dates)

    op = registry.get("ts_kama", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, er_window=10, fast_period=2, slow_period=30)

    # 两列预热后均应有有效输出
    assert result["A"].notna().sum() > 0
    assert result["B"].notna().sum() > 0

    # 趋势方向应保持
    valid_a = result["A"].dropna()
    valid_b = result["B"].dropna()

    assert valid_a.iloc[-1] > valid_a.iloc[0], "列A应上升"
    assert valid_b.iloc[-1] < valid_b.iloc[0], "列B应下降"


# ============================================================================
# state_adaptive_slew_limit Tests (2026-08-12 P0)
# ============================================================================


def test_adaptive_slew_limit_registration(registry):
    """验证 state_adaptive_slew_limit 已正确注册。"""
    assert "state_adaptive_slew_limit" in registry.list_canonical()
    backends = registry.backends_for("state_adaptive_slew_limit")
    assert "pandas_numpy" in backends

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "state_adaptive_slew_limit"
    assert meta.category == "signal_filter"


def test_adaptive_slew_limits_single_bar_change(registry):
    """验证slew limiter限制大单步变化为自适应scale。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    tickers = ["A"]

    # 构造信号：前期平稳，t=25处有大跳变
    values = np.ones(30) * 100.0
    values[25] = 200.0  # +100跳变

    data = pd.DataFrame(values, index=dates, columns=tickers)

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, slew_mult=1.0, scale_window=20, scale_method="mad_delta")

    # 跳变前，delta近似0，scale近似0，输出跟踪输入
    assert np.allclose(result.iloc[:24, 0].values, 100.0, atol=1e-6)

    # 跳变处（t=25），delta被scale_t限制（因过去delta~0，scale小）
    # 输出不应跳到200
    spike_out = result.iloc[25, 0]
    assert spike_out < 200.0, "跳变应被slew限制"
    assert spike_out > 100.0, "输出应增加但被限幅"


def test_adaptive_slew_dimensionless_scale_aware(registry):
    """验证slew limiter适应信号波动率（MAD vs STD）。"""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tickers = ["B"]

    # 带中等波动的信号
    np.random.seed(42)
    x_vals = 100.0 + np.cumsum(np.random.randn(50) * 2.0)
    data = pd.DataFrame(x_vals, index=dates, columns=tickers)

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata

    # MAD方法
    result_mad = op.calculate(data, slew_mult=1.5, scale_window=10, scale_method="mad_delta")
    # STD方法
    result_std = op.calculate(data, slew_mult=1.5, scale_window=10, scale_method="std_delta")

    # 预热后均应有有效输出
    assert result_mad.iloc[15:].notna().all().all()
    assert result_std.iloc[15:].notna().all().all()

    # 两者应不同（MAD vs STD scale不同）
    assert not np.allclose(result_mad.iloc[20:].values, result_std.iloc[20:].values, atol=1e-8)


def test_adaptive_slew_stateful_recursive(registry):
    """验证slew limiter状态递归：y_t依赖y_{t-1}。"""
    dates = pd.date_range("2020-01-01", periods=25, freq="D")
    tickers = ["C"]

    # 信号在t=15从0跳到10
    x_vals = np.concatenate([np.zeros(15), np.ones(10) * 10.0])
    data = pd.DataFrame(x_vals, index=dates, columns=tickers)

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, slew_mult=0.5, scale_window=10, scale_method="mad_delta")

    # 跳变前（t<15）：delta~0，输出跟踪输入
    assert np.allclose(result.iloc[:14, 0].values, 0.0, atol=1e-6)

    # 跳变后：输出应逐步接近10，而非瞬时跳变
    # （因slew_mult=0.5且过去delta~0，limit很小）
    assert result.iloc[15, 0] < 10.0, "跳变应被slew限制"

    # 输出应最终接近10（递归更新）
    final_val = result.iloc[-1, 0]
    assert final_val > result.iloc[15, 0], "输出应持续增加"


def test_adaptive_slew_warmup_period(registry):
    """验证预热期（前scale_window+1行）直接通过。"""
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    tickers = ["D"]

    x_vals = np.arange(15, dtype=float)
    data = pd.DataFrame(x_vals, index=dates, columns=tickers)

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, slew_mult=1.0, scale_window=10, scale_method="mad_delta")

    # 前11行（0..10）应直接通过（预热）
    assert np.allclose(result.iloc[:11, 0].values, x_vals[:11], atol=1e-8)


def test_adaptive_slew_invalid_params(registry):
    """验证无效参数抛出ValueError。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame([[1.0]] * 10, index=dates, columns=["E"])

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata

    # slew_mult <= 0
    with pytest.raises(ValueError, match="slew_mult > 0"):
        op.calculate(data, slew_mult=0.0, scale_window=5, scale_method="mad_delta")

    # scale_window < 2
    with pytest.raises(ValueError, match="scale_window"):
        op.calculate(data, slew_mult=1.0, scale_window=1, scale_method="mad_delta")

    # 无效scale_method
    with pytest.raises(ValueError, match=r"state_adaptive_slew_limit\.scale_method \[runtime\]: scale_method='invalid' is not an allowed choice"):
        op.calculate(data, slew_mult=1.0, scale_window=5, scale_method="invalid")


def test_adaptive_slew_nan_handling(registry):
    """验证NaN输入：传播NaN，重入时重置状态。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    tickers = ["F"]

    x_vals = np.ones(20) * 5.0
    x_vals[10:12] = np.nan  # 缺口
    data = pd.DataFrame(x_vals, index=dates, columns=tickers)

    op = registry.get("state_adaptive_slew_limit", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, slew_mult=1.0, scale_window=5, scale_method="mad_delta")

    # NaN行产生NaN输出
    assert np.isnan(result.iloc[10, 0])
    assert np.isnan(result.iloc[11, 0])

    # NaN缺口后，状态在t=12重新初始化
    assert np.isfinite(result.iloc[12, 0])


# ============================================================================
# ts_causal_local_linear_smoother Tests (2026-08-12 P0)
# ============================================================================


def test_local_linear_smoother_registration(registry):
    """验证 ts_causal_local_linear_smoother 已正确注册。"""
    assert "ts_causal_local_linear_smoother" in registry.list_canonical()
    backends = registry.backends_for("ts_causal_local_linear_smoother")
    assert "pandas_numpy" in backends

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_causal_local_linear_smoother"
    assert meta.category == "signal_filter"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "low_pass"
    assert contract.causal is True
    assert contract.stateful is False
    assert contract.time_shard_safe is True


def test_local_linear_lower_lag_on_trend(registry):
    """验证局部线性平滑器在趋势上的滞后小于SMA。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # 构造线性上升趋势
    trend = np.linspace(100, 150, 100)
    data = pd.DataFrame({"A": trend}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=20, min_periods=10)

    # 简单移动平均作为对照
    sma = data["A"].rolling(window=20, min_periods=10).mean()

    # 在预热后，局部线性应更接近当前值（滞后更小）
    warmup = 20
    local_linear_vals = result.iloc[warmup:, 0].values
    sma_vals = sma.iloc[warmup:].values
    true_vals = trend[warmup:]

    # 计算滞后（与真实值的距离）
    local_linear_lag = np.abs(true_vals - local_linear_vals).mean()
    sma_lag = np.abs(true_vals - sma_vals).mean()

    assert local_linear_lag < sma_lag, \
        f"局部线性滞后 ({local_linear_lag:.2f}) 应小于 SMA ({sma_lag:.2f})"


def test_local_linear_trailing_only(registry):
    """验证严格使用trailing window（one-sided）。"""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    # 前半段平稳，后半段有跳变
    signal = np.concatenate([np.full(25, 100.0), np.full(25, 150.0)])
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=10, min_periods=5)

    # t=24 (跳变前)，trailing window只包含[100, 100, ..., 100]
    # 输出应接近100
    pre_jump_idx = 24
    assert 99.0 < result.iloc[pre_jump_idx, 0] < 101.0, "跳变前应保持在100附近"

    # t=34 (跳变后10期)，trailing window是[150, 150, ..., 150]
    # 输出应接近150
    post_jump_idx = 34
    assert result.iloc[post_jump_idx, 0] > 140.0, "跳变后应收敛到新水平"


def test_local_linear_endpoint_value(registry):
    """验证输出是endpoint fitted value而非窗口均值。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    # 构造有斜率的序列
    slope_signal = np.arange(30, dtype=float)
    data = pd.DataFrame({"A": slope_signal}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=10, min_periods=10)

    # 对于完美线性趋势，局部线性拟合应完美重现
    # y = a + b*x，对于 x=[0,1,...,9] 和 y=[t-9,...,t]
    # endpoint (x=9) 的拟合值应该接近 y[9] = t
    valid_idx = result.notna().values[:, 0]
    assert valid_idx.sum() > 0

    # 从t=9开始有有效输出（window=10, min_periods=10）
    t = 9
    fitted_val = result.iloc[t, 0]
    true_val = slope_signal[t]
    # 对于完美线性，拟合应该完美
    assert abs(fitted_val - true_val) < 0.01, \
        f"完美线性趋势拟合误差应极小，实际 {abs(fitted_val - true_val)}"


def test_local_linear_handles_nan(registry):
    """验证对NaN的正确处理。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    signal = np.arange(30, dtype=float)
    signal[15:18] = np.nan  # 缺口
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=10, min_periods=5)

    # NaN不应阻塞后续计算（只使用有效值）
    # t=20: trailing window包含[nan, nan, nan, 18, 19, 20, ...]
    # 应使用有效值拟合
    assert np.isfinite(result.iloc[20, 0])

    # 预热期前应为NaN
    assert result.iloc[:4, 0].isna().all()


def test_local_linear_parameter_validation(registry):
    """验证参数校验。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame({"A": np.arange(20, dtype=float)}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata

    # window < 2 应报错
    with pytest.raises(ValueError, match="window"):
        op.calculate(data, window=1)

    # min_periods < 2 应报错
    with pytest.raises(ValueError, match=r"ts_causal_local_linear_smoother\.min_periods \[runtime\]: min_periods must be >= 2"):
        op.calculate(data, window=10, min_periods=1)

    # min_periods > window 应报错
    with pytest.raises(ValueError, match="ts_causal_local_linear_smoother: min_periods must not exceed window"):
        op.calculate(data, window=10, min_periods=20)


def test_local_linear_certified_windows(registry):
    """验证认证窗口值可以正常运行。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(42)
    signal = np.cumsum(np.random.randn(100)) + 100
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata

    # 认证参数: [10, 20, 40]
    for window in [10, 20, 40]:
        result = op.calculate(data, window=window, min_periods=window)
        assert result.shape == data.shape
        valid_count = result.notna().sum().iloc[0]
        assert valid_count >= 100 - window, f"window={window} 应有足够有效输出"


def test_local_linear_smooth_noisy_signal(registry):
    """验证对带噪声信号的平滑效果。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(123)
    # 趋势 + 噪声
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 5, 100)
    signal = trend + noise
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_causal_local_linear_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=20, min_periods=10)

    # 输出应比输入更平滑
    valid_idx = result.notna().values[:, 0]
    input_vol = np.diff(signal[valid_idx]).std()
    output_vol = np.diff(result.loc[result.notna().values[:, 0], "A"].values).std()

    assert output_vol < input_vol, "平滑后波动应减小"



# ============================================================================
# ts_butterworth_lowpass_causal Tests (2026-08-12 P0)
# ============================================================================


def test_butterworth_lowpass_registration(registry):
    """验证 ts_butterworth_lowpass_causal 已正确注册。"""
    assert "ts_butterworth_lowpass_causal" in registry.list_canonical()
    backends = registry.backends_for("ts_butterworth_lowpass_causal")
    assert "pandas_numpy" in backends

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    assert op.metadata.name == "ts_butterworth_lowpass_causal"
    assert op.metadata.category == "signal_filter"
    assert hasattr(op, "filter_contract")
    contract = op.filter_contract
    assert contract.role.value == "low_pass"
    assert contract.causal is True
    assert contract.stateful is True
    # No serialized SOS-state adapter exists: execution must replay the full
    # prefix rather than claim checkpoint/resume support.
    assert contract.checkpointable is False
    assert contract.time_shard_safe is False
    assert contract.lag_class == "variable"


def test_butterworth_stronger_attenuation_than_sma(registry):
    """验证 Butterworth 滤波器对高频噪声的衰减强于 SMA。

    Butterworth 是 IIR 滤波器，在截止频率以上有陡峭的衰减 (-40dB/decade
    for order=2)，应比简单移动平均更有效地抑制噪声。
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")

    # A deterministic three-bar sinusoid avoids an SMA comb-filter zero and
    # measures both filters at one explicitly high frequency.
    t = np.arange(200)
    signal = np.sin(2.0 * np.pi * t / 3.0)

    data = pd.DataFrame({"A": signal}, index=dates)

    # Butterworth filter
    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")
    meta = op.metadata
    result_butter = op.calculate(data, cutoff_period=20, order=2)

    # Simple moving average
    sma = data["A"].rolling(window=20, min_periods=1).mean()

    # 计算稳定后的方差（跳过预热期）
    warmup = 40
    var_butter = result_butter.iloc[warmup:, 0].var()
    var_sma = sma.iloc[warmup:].var()

    # Butterworth 应有更低的方差（更强的噪声抑制）
    assert var_butter < var_sma * 0.1, "Butterworth 应比 SMA 有更强的噪声衰减"


def test_butterworth_causal_forward_only(registry):
    """验证严格因果性：只使用过去和当前观测，禁止 filtfilt。

    阶跃响应应匹配声明的数字 Butterworth SOS 递推且无超前响应。
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # 阶跃信号：前50个点为0，后50个点为10
    signal = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, cutoff_period=10, order=2)

    # 阶跃前（t<50）：输出应接近0
    assert result.iloc[40, 0] < 1.0, "阶跃前输出应接近0"

    # 阶跃时刻（t=50）：输出不应超前知道跳变
    # 因果滤波器会有延迟
    assert result.iloc[50, 0] < 10.0, "阶跃时刻输出应滞后"

    # A causal Butterworth step response may overshoot; compare against the
    # declared fs=1.0 SOS recurrence instead of inventing monotonicity.
    from scipy import signal as scipy_signal
    sos = scipy_signal.butter(2, 1.0 / 10.0, fs=1.0, btype="low", output="sos")
    expected, _ = scipy_signal.sosfilt(
        sos, signal, zi=scipy_signal.sosfilt_zi(sos) * signal[0]
    )
    np.testing.assert_allclose(result.iloc[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert result.iloc[-1, 0] == pytest.approx(10.0, abs=1e-8)
    prefix = op.calculate(data.iloc[:60], cutoff_period=10, order=2)
    pd.testing.assert_frame_equal(result.iloc[:60], prefix)


def test_butterworth_stateful_recursive(registry):
    """验证状态递归：输出依赖历史状态，非简单滚动窗口。

    IIR 滤波器维护内部状态（SOS 的 zi），每个新输入更新状态。
    """
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    # 简单上升趋势
    data = pd.DataFrame({
        "A": np.linspace(0, 10, 50),
    }, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, cutoff_period=10, order=2)

    # 滤波器应跟随趋势但有滞后
    assert result.loc[dates[-1], "A"] < data.loc[dates[-1], "A"], "滤波器应滞后于输入"
    assert result.loc[dates[-1], "A"] > data.loc[dates[0], "A"], "滤波器应跟随趋势"

    # 对于上升序列，输出应单调或接近单调
    diff = result["A"].diff()[1:]
    # 允许少量违反（数值误差或初始瞬态）
    assert (diff >= -1e-10).sum() > 45, "上升输入应产生上升输出"


def test_butterworth_certified_parameters(registry):
    """验证认证参数值可以正常运行。

    cutoff_period: [10, 20, 40, 60]
    order: [2, 3, 4]
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    signal = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.RandomState(42).normal(0, 0.1, 100)
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata

    # 认证的 cutoff_period 值
    for cutoff in [10, 20, 40, 60]:
        result = op.calculate(data, cutoff_period=cutoff, order=2)
        assert result.shape == data.shape
        assert result.notna().sum().iloc[0] >= 50, f"cutoff_period={cutoff} 应有有效输出"

    # 认证的 order 值
    for ord_val in [2, 3, 4]:
        result = op.calculate(data, cutoff_period=20, order=ord_val)
        assert result.shape == data.shape
        assert result.notna().sum().iloc[0] >= 50, f"order={ord_val} 应有有效输出"


def test_butterworth_higher_order_steeper_rolloff(registry):
    """验证更高阶数产生更陡峭的频率响应。

    高阶 Butterworth 在截止频率以上衰减更快。
    """
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")

    # 低频趋势 + 高频噪声
    t = np.arange(200)
    trend = 10.0
    high_freq_noise = 2.0 * np.sin(2 * np.pi * t / 5)  # 5天周期高频分量
    signal = trend + high_freq_noise

    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata

    result_order2 = op.calculate(data, cutoff_period=20, order=2)
    result_order4 = op.calculate(data, cutoff_period=20, order=4)

    # 跳过预热期
    warmup = 40
    var_order2 = result_order2.iloc[warmup:, 0].var()
    var_order4 = result_order4.iloc[warmup:, 0].var()

    # 高阶滤波器应有更低的方差（更强的衰减）
    assert var_order4 < var_order2, "order=4 应比 order=2 有更强的高频衰减"


def test_butterworth_handles_nan_correctly(registry):
    """验证对 NaN 的处理：输出 NaN，状态保持冻结。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    }, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, cutoff_period=10, order=2)

    # NaN 位置应输出 NaN
    assert pd.isna(result.iloc[3, 0])

    # NaN 前后的观测应正常滤波
    assert np.isfinite(result.iloc[2, 0])
    assert np.isfinite(result.iloc[4, 0])


def test_butterworth_parameter_validation(registry):
    """验证参数校验。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": [1.0] * 10}, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata

    # cutoff_period < 3 应报错
    with pytest.raises(ValueError, match="cutoff_period"):
        op.calculate(data, cutoff_period=2, order=2)

    # order < 1 应报错
    with pytest.raises(ValueError, match="order must be >= 1"):
        op.calculate(data, cutoff_period=20, order=0)

    # order > 10 应报错（数值稳定性）
    with pytest.raises(ValueError, match="order must be <= 10"):
        op.calculate(data, cutoff_period=20, order=11)


def test_butterworth_multi_column_independent(registry):
    """验证多列面板独立处理。"""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "A": np.linspace(100, 120, 50),  # 上升趋势
        "B": np.linspace(100, 80, 50),   # 下降趋势
        "C": np.full(50, 100.0),         # 常数
    }, index=dates)

    op = registry.get("ts_butterworth_lowpass_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, cutoff_period=10, order=2)

    # 所有列应有有效输出
    assert result["A"].notna().all()
    assert result["B"].notna().all()
    assert result["C"].notna().all()

    # 趋势方向应保持
    assert result["A"].iloc[-1] > result["A"].iloc[0], "列A应上升"
    assert result["B"].iloc[-1] < result["B"].iloc[0], "列B应下降"
    # 常数应保持相对稳定
    assert 95.0 < result["C"].iloc[-1] < 105.0, "列C应接近常数"

# ============================================================================
# Median Despike Family Tests (2026-08-12 P0)
# ============================================================================


def test_median3_causal_registration(registry):
    """验证 ts_median3_causal 已正确注册。"""
    assert "ts_median3_causal" in registry.list_canonical()
    backends = registry.backends_for("ts_median3_causal")
    assert "pandas_numpy" in backends

    op = registry.get("ts_median3_causal", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_median3_causal"
    assert meta.category == "signal_filter"


def test_median3_suppresses_isolated_spike(registry):
    """验证三点中位数抑制孤立 spike。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # 构造孤立 spike：周围都是 10，中间一个点是 100
    data = pd.DataFrame({
        "A": [10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_median3_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data)

    # 前两个点：窗口不足，直接输出
    assert result.loc[dates[0], "A"] == 10.0
    assert result.loc[dates[1], "A"] == 10.0

    # t=2（spike）：median(10, 10, 100) = 10
    assert result.loc[dates[2], "A"] == 10.0, "孤立 spike 应被抑制为中位数"

    # t=3：median(10, 100, 10) = 10
    assert result.loc[dates[3], "A"] == 10.0

    # t=4：median(100, 10, 10) = 10
    assert result.loc[dates[4], "A"] == 10.0

    # 后续正常点保持 10
    assert result.loc[dates[5], "A"] == 10.0


def test_median3_preserves_sustained_jump(registry):
    """验证三点中位数保留持续跳变（非孤立 spike）。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # 持续跳变：前面 10，中间连续三个 20，后面恢复 10
    data = pd.DataFrame({
        "A": [10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_median3_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data)

    # t=2: median(10, 10, 20) = 10
    assert result.loc[dates[2], "A"] == 10.0

    # t=3: median(10, 20, 20) = 20（跳变开始显现）
    assert result.loc[dates[3], "A"] == 20.0, "持续跳变应保留"

    # t=4: median(20, 20, 20) = 20
    assert result.loc[dates[4], "A"] == 20.0

    # t=5: median(20, 20, 10) = 20
    assert result.loc[dates[5], "A"] == 20.0


def test_median3_handles_nan(registry):
    """验证对 NaN 的处理。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "A": [10.0, np.nan, 10.0, 100.0, 10.0, 10.0, np.nan, 10.0],
    }, index=dates)

    op = registry.get("ts_median3_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data)

    # NaN 输入应输出 NaN
    assert pd.isna(result.loc[dates[1], "A"])

    # t=2: 窗口包含 NaN，输出 NaN
    assert pd.isna(result.loc[dates[2], "A"])

    # t=3: median(nan, 10, 100) → NaN（有 NaN）
    assert pd.isna(result.loc[dates[3], "A"])

    # t=4: median(10, 100, 10) = 10（全有限）
    assert result.loc[dates[4], "A"] == 10.0


def test_median3_warmup_period(registry):
    """验证前两行预热期直接输出。"""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "A": [100.0, 200.0, 10.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_median3_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data)

    # 前两行直接输出（即使是异常值）
    assert result.loc[dates[0], "A"] == 100.0
    assert result.loc[dates[1], "A"] == 200.0

    # t=2: median(100, 200, 10) = 100
    assert result.loc[dates[2], "A"] == 100.0


def test_rolling_median_causal_registration(registry):
    """验证 ts_rolling_median_causal 已正确注册。"""
    assert "ts_rolling_median_causal" in registry.list_canonical()
    backends = registry.backends_for("ts_rolling_median_causal")
    assert "pandas_numpy" in backends

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_rolling_median_causal"
    assert meta.category == "signal_filter"


def test_rolling_median_more_robust_than_mean(registry):
    """验证滚动中位数比均值更抗 outlier。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    # 构造信号：平稳 + 孤立 spike
    values = [10.0] * 5 + [100.0] + [10.0] * 14
    data = pd.DataFrame({"A": values}, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata
    result_median = op.calculate(data, window=5, min_periods=3)

    # 简单滚动均值
    result_mean = data["A"].rolling(window=5, min_periods=3).mean()

    # t=5（spike）：median 应接近 10，mean 会被拉高
    spike_idx = 5
    median_at_spike = result_median.loc[dates[spike_idx], "A"]
    mean_at_spike = result_mean.iloc[spike_idx]

    assert median_at_spike < 30.0, "中位数应抗 outlier"
    assert mean_at_spike == pytest.approx(28.0), "五点均值 oracle 应为 (4*10+100)/5"

    # t=6（spike 后一期）：窗口包含 spike，median 仍更稳健
    median_after = result_median.loc[dates[spike_idx + 1], "A"]
    mean_after = result_mean.iloc[spike_idx + 1]

    assert median_after < mean_after, "中位数应比均值更接近正常水平"


def test_rolling_median_short_windows(registry):
    """验证短窗口（3/5/7）工作正常。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    np.random.seed(42)
    signal = 10.0 + np.random.randn(20) * 0.5
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata

    # 认证窗口：3, 5, 7
    for window in [3, 5, 7]:
        result = op.calculate(data, window=window, min_periods=window)
        # 预热后应有有效输出
        valid_count = result["A"].notna().sum()
        expected_valid = len(dates) - window + 1
        assert valid_count == expected_valid, f"window={window} 应有 {expected_valid} 个有效值"


def test_rolling_median_min_periods(registry):
    """验证 min_periods 参数控制预热期。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": list(range(10))}, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata

    # window=5, min_periods=3
    result = op.calculate(data, window=5, min_periods=3)

    # 前 2 行（<3 个观测）应为 NaN
    assert pd.isna(result.loc[dates[0], "A"])
    assert pd.isna(result.loc[dates[1], "A"])

    # t=2（3 个观测）开始有效
    assert pd.notna(result.loc[dates[2], "A"])


def test_rolling_median_handles_nan_in_window(registry):
    """验证窗口内 NaN 的处理：只使用有限值计算。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "A": [10.0, np.nan, 10.0, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, min_periods=3)

    # t=4: 窗口 [10, nan, 10, nan, 10]，有 3 个有限值，满足 min_periods
    assert pd.notna(result.loc[dates[4], "A"])
    assert result.loc[dates[4], "A"] == 10.0, "应只使用有限值计算中位数"


def test_rolling_median_parameter_validation(registry):
    """验证参数校验。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({"A": [1.0] * 10}, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata

    # window < 2 应报错
    with pytest.raises(ValueError, match="window"):
        op.calculate(data, window=1)

    # min_periods < 1 应报错
    with pytest.raises(ValueError, match=r"ts_rolling_median_causal\.min_periods \[runtime\]: min_periods must be >= 1"):
        op.calculate(data, window=5, min_periods=0)

    # min_periods > window 应报错
    with pytest.raises(ValueError, match="ts_rolling_median_causal: min_periods must not exceed window"):
        op.calculate(data, window=3, min_periods=5)


def test_rolling_median_multi_column_independent(registry):
    """验证多列独立处理。"""
    dates = pd.date_range("2020-01-01", periods=15, freq="D")
    data = pd.DataFrame({
        "A": [10.0] * 5 + [100.0] + [10.0] * 9,  # t=5 有 spike
        "B": [5.0] * 15,  # 平稳
        "C": [20.0] * 10 + [200.0] + [20.0] * 4,  # t=10 有 spike
    }, index=dates)

    op = registry.get("ts_rolling_median_causal", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, window=5, min_periods=3)

    # A 列的 spike 不应影响 B 列
    assert result.loc[dates[5], "B"] == 5.0, "B 列应不受 A 列 spike 影响"

    # C 列的 spike 不应影响 A 列
    assert result.loc[dates[10], "A"] == 10.0, "A 列应不受 C 列 spike 影响"


def test_cost_aware_deadband_proportional_to_cost(registry):
    """验证成本感知deadband：band与cost_proxy成正比。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    # 构造信号和成本代理（如spread/Amihud）
    signal = pd.DataFrame({
        "A": [10.0, 10.3, 10.6, 10.8, 11.1, 11.5, 12.0, 12.3, 12.6, 13.0],
    }, index=dates)
    cost = pd.DataFrame({
        "A": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
    }, index=dates)

    op = registry.get("state_cost_aware_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(signal, cost, cost_mult=2.0)

    # 注意：死区比较的基准是"上次更新后的输出"，不是上一期输入。
    # t=0: 初始化为10.0
    assert abs(result.loc[dates[0], "A"] - 10.0) < 0.01

    # 低成本期 cost=0.1 → band=0.2
    # t=1: |10.3-10.0|=0.3 > 0.2 → 更新
    assert abs(result.loc[dates[1], "A"] - 10.3) < 0.01
    # t=2: |10.6-10.3|=0.3 > 0.2 → 更新
    assert abs(result.loc[dates[2], "A"] - 10.6) < 0.01
    # t=3: |10.8-10.6|=0.2 不大于 band → 保持 10.6（边界不更新）
    assert abs(result.loc[dates[3], "A"] - 10.6) < 0.01
    # t=4: |11.1-10.6|=0.5 > 0.2 → 更新
    assert abs(result.loc[dates[4], "A"] - 11.1) < 0.01

    # 高成本期 cost=0.5 → band=1.0，同等幅度的变化被抑制
    # t=5: |11.5-11.1|=0.4 < 1.0 → 保持 11.1
    assert abs(result.loc[dates[5], "A"] - 11.1) < 0.01
    # t=6: |12.0-11.1|=0.9 < 1.0 → 仍保持 11.1
    assert abs(result.loc[dates[6], "A"] - 11.1) < 0.01
    # t=7: |12.3-11.1|=1.2 > 1.0 → 更新
    assert abs(result.loc[dates[7], "A"] - 12.3) < 0.01
    # t=8/t=9: 相对 12.3 的变化 0.3 / 0.7 均 < 1.0 → 保持
    assert abs(result.loc[dates[8], "A"] - 12.3) < 0.01
    assert abs(result.loc[dates[9], "A"] - 12.3) < 0.01

    # 正比性：band = cost_mult * cost_proxy，cost_mult 越大更新次数越少
    def _updates(mult: float) -> int:
        out = op.calculate(signal, cost, cost_mult=mult)["A"].to_numpy()
        return int(np.count_nonzero(np.abs(np.diff(out)) > 1e-9))

    assert _updates(0.0) > _updates(2.0) > _updates(5.0), "band 应随 cost_mult 单调变宽"


def test_cost_aware_slew_inversely_proportional(registry):
    """验证成本感知slew：允许变化率与cost成反比。"""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")

    # 构造信号和成本（高成本 → 低流动性 → 慢速更新）
    signal = pd.DataFrame({
        "A": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    }, index=dates)
    cost = pd.DataFrame({
        "A": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0],
    }, index=dates)

    op = registry.get("state_cost_aware_slew", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(signal, cost, slew_mult=3.0)

    # t=0: 初始化为10.0
    assert abs(result.loc[dates[0], "A"] - 10.0) < 0.01

    # Bounded dimensional formula: limit = slew_mult / (1 + cost).
    low_cost_limit = 3.0 / 1.1
    assert result.loc[dates[1], "A"] == pytest.approx(10.0 + low_cost_limit)
    assert result.loc[dates[2], "A"] == pytest.approx(10.0 + 2.0 * low_cost_limit)

    # cost=1 gives a 1.5-unit limit.
    assert result.loc[dates[3], "A"] == pytest.approx(10.0 + 2.0 * low_cost_limit + 1.5)

    # t=4: 从23到30跳7.0，应限制为23+3=26
    assert result.loc[dates[4], "A"] == pytest.approx(10.0 + 2.0 * low_cost_limit + 3.0)

    # t=5: 从26到35跳9.0，应限制为26+3=29
    assert result.loc[dates[5], "A"] == pytest.approx(10.0 + 2.0 * low_cost_limit + 4.5)

    # 验证高成本期间输出变化缓慢（slew limiting生效）
    high_cost_change = result.loc[dates[5], "A"] - result.loc[dates[3], "A"]
    assert high_cost_change < 7.0, "高成本期应限制变化速率"


def test_cost_aware_deadband_registration(registry):
    """验证cost-aware deadband正确注册。"""
    assert "state_cost_aware_deadband" in registry.list_canonical()
    backends = registry.backends_for("state_cost_aware_deadband")
    assert "pandas_numpy" in backends

    op = registry.get("state_cost_aware_deadband", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    assert op.metadata.name == "state_cost_aware_deadband"
    assert op.metadata.category == "signal_filter"


def test_cost_aware_slew_registration(registry):
    """验证cost-aware slew正确注册。"""
    assert "state_cost_aware_slew" in registry.list_canonical()
    backends = registry.backends_for("state_cost_aware_slew")
    assert "pandas_numpy" in backends

    op = registry.get("state_cost_aware_slew", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    assert op.metadata.name == "state_cost_aware_slew"
    assert op.metadata.category == "signal_filter"


def test_cost_aware_deadband_zero_cost_handling(registry):
    """验证cost=0或NaN时的处理。"""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    signal = pd.DataFrame({
        "A": [10.0, 10.5, 11.0, 11.5, 12.0],
    }, index=dates)
    cost = pd.DataFrame({
        "A": [0.0, 0.0, np.nan, 0.1, 0.1],
    }, index=dates)

    op = registry.get("state_cost_aware_deadband", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(signal, cost, cost_mult=2.0)

    # Invalid or non-positive cost fails closed; it must not silently become an
    # unconstrained trading instruction.
    assert pd.isna(result.loc[dates[1], "A"])
    assert pd.isna(result.loc[dates[2], "A"])


def test_cost_aware_slew_handles_invalid_cost(registry):
    """验证cost无效（0/负数/NaN）时slew的容错。"""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    signal = pd.DataFrame({
        "A": [10.0, 20.0, 30.0, 40.0, 50.0],
    }, index=dates)
    cost = pd.DataFrame({
        "A": [0.0, -0.1, np.nan, 0.01, 0.01],
    }, index=dates)

    op = registry.get("state_cost_aware_slew", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(signal, cost, slew_mult=1.0)

    # Negative and missing costs fail closed and preserve the last valid state.
    assert pd.isna(result.loc[dates[1], "A"])
    assert pd.isna(result.loc[dates[2], "A"])


# ============================================================================
# Two-Pole IIR Low-Pass Filter Tests (ts_super_smoother, 2026-08-12 P0)
# ============================================================================


def test_super_smoother_registration_category(registry):
    """验证 ts_super_smoother 已正确注册。"""
    assert "ts_super_smoother" in registry.list_canonical()
    backends = registry.backends_for("ts_super_smoother")
    assert "pandas_numpy" in backends

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    assert op.metadata.name == "ts_super_smoother"
    assert meta.category == "signal_filter"


def test_super_smoother_stronger_attenuation_than_ema_sinusoidal(registry):
    """验证 two-pole filter 比 EMA 有更强的噪声抑制。"""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    # 构造信号：低频趋势 + 高频噪声
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t) + 0.3 * np.sin(10*t)  # 慢波 + 快波

    data = pd.DataFrame({"A": signal}, index=dates)

    op_super = registry.get("ts_super_smoother", backend="pandas_numpy")
    op_ema = registry.get("ts_ema", backend="pandas_numpy")

    result_super = op_super.calculate(data, period=20)
    result_ema = op_ema.calculate(data, span=20)

    # 后半段稳定后，super_smoother 应该更平滑（方差更小）
    tail = slice(50, None)
    var_super = result_super.iloc[tail, 0].var()
    var_ema = result_ema.iloc[tail, 0].var()

    assert var_super < var_ema * 0.95, "Super smoother 应比 EMA 更平滑"


def test_super_smoother_stateful_needs_two_lags_spike_series(registry):
    """验证 two-pole filter 需要两个历史状态。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame({
        "A": [10.0, 10.0, 10.0, 10.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    }, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, period=5)

    # 前两个点作为初始化：y_0 = x_0, y_1 = x_1
    assert result.loc[dates[0], "A"] == 10.0
    assert result.loc[dates[1], "A"] == 10.0

    # 从第三个点开始递归计算
    assert np.isfinite(result.loc[dates[2], "A"])

    # 毛刺处：two-pole 应有更强的衰减
    spike_response = result.loc[dates[4], "A"]
    assert spike_response < 30.0, "Two-pole 应强力衰减毛刺"
    assert spike_response > 10.0, "应有响应"


def test_super_smoother_step_response_no_overshoot_unit_step(registry):
    """验证 Butterworth 特性：阶跃响应无显著过冲。"""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    # 阶跃信号：前20个点=0，后30个点=1
    signal = np.concatenate([np.zeros(20), np.ones(30)])
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_super_smoother", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, period=10)

    # 阶跃后的响应应接近单调上升（允许极小的 Butterworth ripple）
    step_region = result.loc[dates[20]:dates[35], "A"].values
    diffs = np.diff(step_region)

    # Butterworth 可能有极小的 ripple（~1-2%），不是显著过冲
    max_negative_diff = np.abs(diffs[diffs < 0].sum()) if np.any(diffs < 0) else 0
    assert max_negative_diff < 0.05, "不应有显著过冲或振荡"

    # 最终应收敛到 1.0
    final_value = result.loc[dates[-1], "A"]
    assert abs(final_value - 1.0) < 0.01, "最终应收敛到目标值"


# ============================================================================
# ts_robust_ema tests (innovation-layer clipping)
# ============================================================================

def test_robust_ema_registration(registry):
    """验证 ts_robust_ema 已正确注册。"""
    assert "ts_robust_ema" in registry.list_canonical()
    backends = registry.backends_for("ts_robust_ema")
    assert "pandas_numpy" in backends

    op = registry.get("ts_robust_ema", backend="pandas_numpy")

    meta = op.metadata
    assert op is not None
    meta = op.metadata
    assert op.metadata.name == "ts_robust_ema"
    assert "signal_filter" in (meta.tags or []) or meta.category == "signal_filter"
    assert any("stateful" in str(t) for t in (meta.tags or []))


def test_robust_ema_clips_large_innovations(registry):
    """验证 innovation 层截断：大毛刺被clip，不传染到状态。"""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # 平稳序列中插入极大毛刺
    data = pd.DataFrame({
        "A": [1.0, 1.1, 0.9, 1.05, 0.95, 100.0, 1.0, 1.1, 0.9, 1.05],
    }, index=dates)

    op = registry.get("ts_robust_ema", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, span=10, clip_sigma=3.0, warmup_window=4)

    # 毛刺处（index=5）：innovation = 100 - y_prev ≈ 99
    # 应被clip到合理范围，不让状态跳到高位
    spike_value = result.iloc[5, 0]
    assert spike_value < 10.0, "毛刺应被clip，状态不应跳变"
    assert spike_value > 0.5, "clip后应保留方向"

    # 毛刺后迅速恢复
    assert result.iloc[6, 0] < 5.0, "毛刺后应迅速衰减"
    assert abs(result.iloc[-1, 0] - 1.0) < 0.5, "最终应收敛回稳定水平"


def test_robust_ema_stateful_recursive(registry):
    """验证递归状态依赖：y_t = y_{t-1} + alpha * e_t*。"""
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    # 逐步上升的序列
    data = pd.DataFrame({
        "A": np.linspace(1.0, 2.0, 20),
    }, index=dates)

    op = registry.get("ts_robust_ema", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, span=10, clip_sigma=3.0)

    # 输出应是平滑的递归序列
    assert result.notna().all().all(), "输出应无NaN"

    # 初始化：y_0 = x_0
    assert abs(result.iloc[0, 0] - 1.0) < 1e-6, "初始值应等于x_0"

    # 递归更新应平滑追踪输入
    assert result.iloc[10, 0] > result.iloc[0, 0], "应追踪上升趋势"
    assert result.iloc[-1, 0] < data.iloc[-1, 0], "EMA应滞后于输入"


def test_robust_ema_converges_to_stable_level(registry):
    """验证收敛性：对常数序列，EMA应收敛到该常数。"""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    # 常数序列
    data = pd.DataFrame({
        "A": np.full(50, 5.0),
    }, index=dates)

    op = registry.get("ts_robust_ema", backend="pandas_numpy")

    meta = op.metadata
    result = op.calculate(data, span=10, clip_sigma=3.0)

    # 经过足够时间（~3*span），应收敛到5.0
    final_value = result.iloc[-1, 0]
    assert abs(final_value - 5.0) < 0.01, "应收敛到常数水平"


def test_robust_ema_more_robust_than_standard_ema(registry):
    """验证 robust EMA 相比标准 EMA 对异常值的鲁棒性。"""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    # 平稳序列 + 一个大异常值
    signal = np.ones(30) * 10.0
    signal[15] = 100.0  # 大异常值
    data = pd.DataFrame({"A": signal}, index=dates)

    op = registry.get("ts_robust_ema", backend="pandas_numpy")

    meta = op.metadata
    result_robust = op.calculate(data, span=10, clip_sigma=3.0, warmup_window=10)

    # 标准 EMA（使用 pandas ewm）
    result_standard = data.ewm(span=10, adjust=False).mean()

    # 在异常值之后（index=20），robust EMA 应更接近真实水平 10.0
    robust_at_20 = result_robust.iloc[20, 0]
    standard_at_20 = result_standard.iloc[20, 0]

    print(f"Robust EMA at t=20: {robust_at_20:.2f}, Standard EMA: {standard_at_20:.2f}")

    # Robust EMA 受异常值影响更小
    assert abs(robust_at_20 - 10.0) < abs(standard_at_20 - 10.0), "Robust EMA应更接近真实水平"

    # 最终两者都应收敛回 10.0，但 robust 应更快
    robust_final = abs(result_robust.iloc[-1, 0] - 10.0)
    standard_final = abs(result_standard.iloc[-1, 0] - 10.0)
    assert robust_final <= standard_final, "Robust EMA应更快恢复"
