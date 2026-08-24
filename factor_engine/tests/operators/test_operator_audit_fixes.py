# -*- coding: utf-8 -*-
"""Semantic regression tests for the 2026-08 operator-correctness audit fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _frame(rows: int, cols: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(0.0, index=idx, columns=[f"C{i}" for i in range(cols)])


def _values(rows: int, data: list) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(np.asarray(data, dtype=float).reshape(rows, -1), index=idx, columns=[f"C{i}" for i in range(np.asarray(data).ndim - 1 if np.asarray(data).ndim > 1 else 1)])


def _op(name):
    return OperatorRegistry.get(name)


# ---------------------------------------------------------------------------
# 涨跌停
# ---------------------------------------------------------------------------


def test_ashare_limit_one_price_all_ohlc_at_limit() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    one = lambda v: pd.DataFrame([v] * 3, index=idx, columns=["A"])
    op = _op("ashare_limit_one_price")
    # 一字涨停：OHLC 全部 = 10 = 上限
    out = op.calculate(one(10.0), one(10.0), one(10.0), one(10.0), one(10.0), one(9.0), "up", 0.005)
    assert out["A"].tolist() == [1.0, 1.0, 1.0]
    # 非一字：high 偏离上限 -> 0
    out2 = op.calculate(one(10.0), one(10.5), one(10.0), one(10.0), one(10.0), one(9.0), "up", 0.005)
    assert out2["A"].tolist() == [0.0, 0.0, 0.0]


def test_ashare_limit_failed_is_same_day_break() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    hi = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=["A"])
    cl = pd.DataFrame([9.9, 10.0, 9.8], index=idx, columns=["A"])
    ul = pd.DataFrame([10.0, 10.0, 10.0], index=idx, columns=["A"])
    out = _op("ashare_limit_failed").calculate(hi, cl, ul, 0.005)
    assert out["A"].tolist() == [1.0, 0.0, 1.0]


def test_ashare_limit_up_touch_uses_high_not_close() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    hi = pd.DataFrame([10.0, 9.5], index=idx, columns=["A"])
    cl = pd.DataFrame([10.0, 10.0], index=idx, columns=["A"])
    ul = pd.DataFrame([10.0, 10.0], index=idx, columns=["A"])
    out = _op("ashare_limit_up_touch").calculate(hi, ul, 0.005)
    # 第2日 close 在涨停但 high 未触及 -> 0
    assert out["A"].tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# 时序风险
# ---------------------------------------------------------------------------


def test_ts_current_drawdown_duration_respects_window() -> None:
    # 价格 10 -> 9 -> 8 -> 9 -> 10（新高），window 影响连续回撤
    idx = pd.date_range("2024-01-01", periods=5)
    px = pd.DataFrame([10.0, 9.0, 8.0, 9.0, 10.0], index=idx, columns=["A"])
    out20 = _op("ts_current_drawdown_duration").calculate(px, window=20)
    # window=20（全历史）: row1=1, row2=2, row3=3（均低于全历史峰值10）, row4=新高=0
    assert out20["A"].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]
    out3 = _op("ts_current_drawdown_duration").calculate(px, window=3)
    # window=3: row3 的窗口运行峰值=max(8,9)=9，等于当前值 -> 回撤中断=0
    assert out3["A"].tolist() == [0.0, 1.0, 2.0, 0.0, 0.0]
    # window 必须改变结果（修复前忽略 window 时两者相同）
    assert out20["A"].iloc[3] != out3["A"].iloc[3]


def test_ts_price_delay_reacts_to_benchmark_lag() -> None:
    rng = np.random.default_rng(0)
    n = 80
    bench = rng.standard_normal(n)
    # stock = 今日基准 + 昨日基准（滞后成分）→ delay 应 > 0
    stock = bench + np.concatenate([[0.0], bench[:-1]]) + rng.standard_normal(n) * 0.1
    idx = pd.date_range("2024-01-01", periods=n)
    sr = pd.DataFrame(stock, index=idx, columns=["A"])
    br = pd.DataFrame(bench, index=idx, columns=["A"])
    out = _op("ts_price_delay").calculate(sr, br, window=40, max_lag=3, min_periods=6)
    # 有滞后成分，delay 应在 (0,1)；末尾行应有限
    assert np.isfinite(out["A"].iloc[-1])
    assert 0.0 <= out["A"].iloc[-1] <= 1.0
    assert out["A"].iloc[-1] > 0.0


# ---------------------------------------------------------------------------
# 日期 / 事件
# ---------------------------------------------------------------------------


def test_calendar_day_diff_handles_datetime_frames() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    d1 = pd.DataFrame(pd.to_datetime(["2024-01-01", "2024-01-02"]), index=idx, columns=["A"])
    d2 = pd.DataFrame(pd.to_datetime(["2024-01-05", "2024-01-02"]), index=idx, columns=["A"])
    out = _op("calendar_day_diff").calculate(d1, d2)
    assert out["A"].tolist() == [4.0, 0.0]


def test_fin_announcement_lag_datetime() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    period = pd.DataFrame(pd.to_datetime(["2024-01-01", "2024-01-02"]), index=idx, columns=["A"])
    pub = pd.DataFrame(pd.to_datetime(["2024-04-01", "2024-04-02"]), index=idx, columns=["A"])
    out = _op("fin_announcement_lag").calculate(period, pub)
    assert out["A"].tolist() == [91.0, 91.0]


def test_event_decay_asof_preserves_sign_and_magnitude() -> None:
    idx = pd.date_range("2024-01-01", periods=4)
    ev = pd.DataFrame([0.0, 2.0, 0.0, 0.0], index=idx, columns=["A"])
    out = _op("event_decay_asof").calculate(ev, half_life=10.0)
    # 事件幅度 2.0 应保留（而非置 1）
    assert out["A"].iloc[1] == pytest.approx(2.0)
    # 负事件应得负衰减
    ev_neg = pd.DataFrame([0.0, -3.0, 0.0, 0.0], index=idx, columns=["A"])
    out_neg = _op("event_decay_asof").calculate(ev_neg, half_life=10.0)
    assert out_neg["A"].iloc[2] < 0.0


def test_event_decay_asof_rejects_small_half_life() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    ev = pd.DataFrame([0.0, 1.0], index=idx, columns=["A"])
    with pytest.raises(ValueError, match="half_life"):
        _op("event_decay_asof").calculate(ev, half_life=0.5)


# ---------------------------------------------------------------------------
# 条件算子
# ---------------------------------------------------------------------------


def test_conditional_ops_exclude_nan_x() -> None:
    idx = pd.date_range("2024-01-01", periods=6)
    x = pd.DataFrame([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=idx, columns=["A"])
    cond = pd.DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx, columns=["A"])
    # ts_min_if 窗口=6：NaN 的 x 不入选 -> min=1
    out = _op("ts_min_if").calculate(x, cond, window=6, min_periods=1)
    assert out["A"].iloc[-1] == pytest.approx(1.0)


def test_ts_regression_resid_if_current_sample_semantics() -> None:
    idx = pd.date_range("2024-01-01", periods=8)
    x = pd.DataFrame(np.arange(8, dtype=float), index=idx, columns=["A"])
    y = pd.DataFrame(2.0 * np.arange(8, dtype=float) + 0.5, index=idx, columns=["A"])
    cond = pd.DataFrame([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], index=idx, columns=["A"])
    out = _op("ts_regression_resid_if").calculate(y, x, cond, window=8, min_periods=3)
    # 行5、6 条件为假 -> NaN
    assert np.isnan(out["A"].iloc[5])
    assert np.isnan(out["A"].iloc[6])
    # 行7 条件为真 -> 用当前样本残差（y=2x+0.5 完美拟合 -> 0）
    assert out["A"].iloc[7] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 状态 / 指数
# ---------------------------------------------------------------------------


def test_ts_transition_count_nan_breaks_sequence() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    cond = pd.DataFrame([1.0, 0.0, np.nan, 1.0, 1.0], index=idx, columns=["A"])
    out = _op("ts_transition_count").calculate(cond, window=5)
    # 行2 NaN -> NaN；行3 从缺失后重新开始，不因 True->NaN->True 计两次
    assert np.isnan(out["A"].iloc[2])
    # 行3：窗口内有限序列 [1,0,(nan)] -> 只有 1->0 一次切换
    assert out["A"].iloc[3] == pytest.approx(1.0)


def test_index_membership_age_stops_after_exit() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    member = pd.DataFrame([0.0, 1.0, 1.0, 0.0, 1.0], index=idx, columns=["A"])
    out = _op("index_membership_age").calculate(member)
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == pytest.approx(0.0)
    assert out["A"].iloc[2] == pytest.approx(1.0)
    assert np.isnan(out["A"].iloc[3])  # 剔除后不再累计
    assert out["A"].iloc[4] == pytest.approx(0.0)  # 重新纳入


def test_index_member_distinguishes_zero_from_unknown() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    member = pd.DataFrame([1.0, 0.0, np.nan], index=idx, columns=["A"])
    out = _op("index_member").calculate(member)
    assert out["A"].tolist()[0] == pytest.approx(1.0)
    assert out["A"].tolist()[1] == pytest.approx(0.0)
    assert np.isnan(out["A"].tolist()[2])


# ---------------------------------------------------------------------------
# 关系权重
# ---------------------------------------------------------------------------


def test_relation_rank_weighted_sum_denominator_counts_valid_only() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    s1 = pd.DataFrame([[0.4]], index=idx, columns=["A"])
    s2 = pd.DataFrame([[0.3]], index=idx, columns=["A"])
    s3 = pd.DataFrame([[np.nan]], index=idx, columns=["A"])  # 无效名次
    out = _op("relation_rank_weighted_sum").calculate(s1, s2, s3)
    expected = (0.4 / 1.0 + 0.3 / 2.0) / (1.0 + 0.5)
    assert out["A"].iloc[0] == pytest.approx(expected)


def test_relation_peer_weighted_mean_ex_self_excludes_nan_value() -> None:
    idx = pd.date_range("2024-01-01", periods=1)
    value = pd.DataFrame([[1.0, np.nan, 3.0]], index=idx, columns=["A", "B", "C"])
    weight = pd.DataFrame([[1.0, 1.0, 1.0]], index=idx, columns=["A", "B", "C"])
    group = pd.DataFrame([["X", "X", "X"]], index=idx, columns=["A", "B", "C"])
    out = _op("relation_peer_weighted_mean_ex_self").calculate(value, weight, group)
    # 组内有效 value 只有 A=1, C=3；B 的 value 为 NaN 不计入权重
    # A 的 peer = 3.0；C 的 peer = 1.0；B 的 value NaN -> NaN
    assert out["A"].iloc[0] == pytest.approx(3.0)
    assert out["C"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(out["B"].iloc[0])
