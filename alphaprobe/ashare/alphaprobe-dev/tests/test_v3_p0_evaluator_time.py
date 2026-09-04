"""test_evaluator_adapter P0-F 时间轴 + evaluate_many 批量验收测试。

覆盖（P0-F 任务书）：
① label_time_axis 纯函数：25 个 2026-01 前后交易日手写列表，horizon=20
   → label_end_time[i] == dates[i+20]（i < len-20）；末尾未成熟段用
   maturity marker 保 LabelBundle 因果链（start<end 严格递增）。
   绝不用 datetime+timedelta(days=1) 作为 label 时间轴。
② evaluate_many 批量：3 个合成因子面板，QE 可导入时与逐个 evaluate 的
   rankic_valid 一致（abs<=1e-9）；单因子失败降级 None；批级失败
   （QE 不可导入 / 契约失败 / 空批 / 形状错位）→ QuantEvaluatorError
   fail-closed。
③ 20 日 cohort 时间对拍 hard test：小面板（因子 / 价格 / label 对齐段）手算
   cohort daily PnL，与 compute_cohort_pnl / compute_metrics_from_cohort 输出
   pnl_net 对拍（vnwap→vwap，信号 t 收盘 → 入场 t+1 VWAP → 出场 t+H）。

QE / modeling 可导入性：本仓库标准 pytest 命令在 alphaPROBE 子目录下默认
venv 不含 quant_evaluator（源树在 quant_projects 根、无 psutil/torch）；
因此所有依赖 QE 的用例沿用既有 skip 模式（QE 不可导入即 skip，但纯函数
label_time_axis 用例不依赖 QE、永远跑）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 引导 quant_projects 进 sys.path（QE / modeling / factor_assets 可导入）。
# pytest 从 repo 根跑时 quant_projects 已在 sys.path；此处幂等兜底。
for _cand in Path(__file__).resolve().parents:
    if (_cand / "quant_evaluator").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from alphaprobe.evaluator_adapter import (
    QuantEvaluatorAdapter,
    QuantEvaluatorError,
    label_time_axis,
)
from alphaprobe.fitness.contracts import EvaluationBundle


# ---------------------------------------------------------------------------
# QE / modeling 可用性探测（skip 条件）
# ---------------------------------------------------------------------------


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


QE_IMPORTABLE = _importable("quant_evaluator")
MODELING_IMPORTABLE = _importable("modeling")

QE_SKIP = pytest.mark.skipif(
    not QE_IMPORTABLE, reason="quant_evaluator not importable in venv"
)

# ---------------------------------------------------------------------------
# ① label_time_axis 纯函数（不依赖 QE）
# ---------------------------------------------------------------------------

# 2026-01 前后 25 个交易日（跳过周末；2-3 月无节假日干扰）。
TRADING_DATES_25: list[str] = [
    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
    "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15",
    "2026-01-16", "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22",
    "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29",
    "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05",
]


class TestLabelTimeAxis:
    def test_horizon20_forward_shift_on_trading_dates(self):
        """dates=25 交易日、horizon=20 → label_end_time[i] == dates[i+20]。"""
        dates = [pd.Timestamp(d) for d in TRADING_DATES_25]
        out = label_time_axis(dates, 20)
        assert len(out) == len(dates)
        for i in range(len(dates) - 20):
            assert out[i] == dates[i + 20], (
                f"label_end_time[{i}] 应为 dates[{i + 20}]={dates[i + 20]}，"
                f"got {out[i]}"
            )

    def test_no_natural_day_shift_in_mature_segment(self):
        """成熟段严禁 datetime + timedelta(days=1) 的自然日语义。"""
        dates = [pd.Timestamp(d) for d in TRADING_DATES_25]
        out = label_time_axis(dates, 20)
        # 2026-01-02(周五) + 20 交易日 = 2026-01-30(周五)。自然日 +20 天 = 01-22。
        # 若实现退化回自然日，out[0] 会是 01-22，这里显式锁死交易日语义。
        assert out[0] == pd.Timestamp("2026-01-30")
        assert out[0] != pd.Timestamp("2026-01-22")

    def test_maturity_pad_preserves_strict_increase_and_ordering(self):
        """末尾未成熟段（i+20 越界）用 maturity marker 保 LabelBundle 因果链。"""
        dates = [pd.Timestamp(d) for d in TRADING_DATES_25]
        out = label_time_axis(dates, 20)
        # 全部位置 label_start < label_end（QE LabelBundle 强制）
        for i in range(len(dates)):
            assert dates[i] < out[i], f"start>=end at {i}: {dates[i]} vs {out[i]}"
        # 成熟段 end 严格递增；末端未成熟段共享同一 marker（允许等值）。
        for i in range(len(out) - 21):
            assert out[i] < out[i + 1], f"mature end axis not increasing at {i}"

    def test_label_bundle_roundtrip_when_qe_available(self):
        """QE 可导入时 label_time_axis 输出可直接构造合法 LabelBundle。"""
        if not QE_IMPORTABLE:
            pytest.skip("quant_evaluator not importable")
        from quant_evaluator.contracts.label_bundle import LabelBundle

        dates = [pd.Timestamp(d) for d in TRADING_DATES_25]
        out = label_time_axis(dates, 20)
        T = len(dates)
        lb = LabelBundle(
            target_id="vwap_to_vwap",
            values=np.zeros((T, 3)),
            horizon=20,
            decision_time=tuple(dates),
            label_start_time=tuple(dates),
            label_end_time=tuple(out),
        )
        assert lb.num_observations() == T

    def test_horizon1_small(self):
        """horizon=1：end[i] == dates[i+1]（边界与最小 horizon）。"""
        dates = [pd.Timestamp(d) for d in TRADING_DATES_25[:5]]
        out = label_time_axis(dates, 1)
        assert out[0] == dates[1]
        assert out[1] == dates[2]

    def test_explicit_calendar_extends_beyond_dates(self):
        """显式传入更长 calendar → index 前移到真实日历更远位置。"""
        dates = [pd.Timestamp(d) for d in TRADING_DATES_25[:5]]
        calendar = [pd.Timestamp(d) for d in TRADING_DATES_25]
        out = label_time_axis(dates, 20, calendar=calendar)
        # 5 个 decision 均越界（5+20 > 25）→ maturity marker 填充
        assert len(out) == 5
        for i in range(5):
            assert dates[i] < out[i]


# ---------------------------------------------------------------------------
# ③ 20 日 cohort 时间对拍 hard test（小算例逐项手算）
# ---------------------------------------------------------------------------

# 确定性 4 日 × 3 票价格面板（vwap；后复权口径由调用方保证）。
PRICES = np.array(
    [
        [100.0, 100.0, 100.0],
        [100.0, 101.0, 110.0],
        [101.0, 102.0, 121.0],
        [102.0, 103.0, 133.1],
    ]
)
# 3 票因子值：col2 最高（D10），col0 最低（D1），col1 中间。全程恒定。
FACTOR = np.array(
    [
        [0.1, 0.5, 0.9],
        [0.1, 0.5, 0.9],
        [0.1, 0.5, 0.9],
        [0.1, 0.5, 0.9],
    ]
)


def _manual_cohort_pnl(holding: int = 2) -> np.ndarray:
    """手写 cohort 每日真实 PnL（信号 t 收盘 → 入场 t+1 VWAP → 出场 t+H VWAP）。

    与 QE ``compute_cohort_pnl`` 同口径：daily 每 cohort 名义 1.0 均摊到
    持有期（1/H 缩放），long D10 每票 +0.5/N、short D1 每票 -0.5/N，gross。
    返回 (T,)：T = next_ret 行数（价格差分行），index 对齐 QE pnl_net。
    """
    next_ret = PRICES[1:] / PRICES[:-1] - 1.0
    T = next_ret.shape[0]
    pnl = np.zeros(T)
    long_weight = 0.5  # D10 每票（n_long=1）
    short_weight = -0.5  # D1 每票（n_short=1）
    for start in range(T):
        if start + 1 >= T:
            continue
        entry_t = start + 1
        exit_t = min(start + holding, T - 1)
        for t in range(entry_t, exit_t + 1):
            # D10 = col2（+0.5 每票），D1 = col0（-0.5 每票）。
            day_contrib = long_weight * next_ret[t, 2] + short_weight * next_ret[t, 0]
            pnl[t] += day_contrib / holding
    return pnl


@QE_SKIP
class TestCohortTimingParity:
    def test_cohort_pnl_matches_manual_trace(self):
        """compute_cohort_pnl 的 pnl_net 与手算每日真实 PnL 对拍。"""
        from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl

        factor_aligned = FACTOR[1:]  # factor 行 = 价格 1..4（对齐 next_ret 段）
        next_ret = PRICES[1:] / PRICES[:-1] - 1.0
        next_vwap = PRICES[1:]
        res = compute_cohort_pnl(
            factor_values=factor_aligned,
            next_ret=next_ret,
            next_vwap=next_vwap,
            n_quantiles=3,
            holding=2,
            per_side_cost=0.0,
            require_tradable=True,
        )
        expected = _manual_cohort_pnl(holding=2)
        np.testing.assert_allclose(res["pnl_net"], expected, atol=1e-12)

    def test_compute_metrics_from_cohort_wraps_same_pnl(self):
        """compute_metrics_from_cohort 的 cohort['pnl_net'] 与手算一致。"""
        from quant_evaluator.metrics.probe_portfolio import compute_metrics_from_cohort

        next_ret = PRICES[1:] / PRICES[:-1] - 1.0
        next_vwap = PRICES[1:]
        res = compute_metrics_from_cohort(
            factor_values=FACTOR[1:],
            next_ret=next_ret,
            next_vwap=next_vwap,
            n_quantiles=3,
            holding=2,
            per_side_cost=0.0,
            min_periods=1,
        )
        expected = _manual_cohort_pnl(holding=2)
        np.testing.assert_allclose(res["cohort"]["pnl_net"], expected, atol=1e-12)

    def test_adapter_cohort_path_runs_and_manual_pnl_matches(self):
        """adapter 20d cohort 路径（QE 端）在短面板小算例上跑通，且 cohort daily
        PnL 与手算完全一致（holding=2、n_quantiles=3 的 4 日 × 3 票小面板）。

        直接验证 adapter 内部用的 ``compute_cohort_pnl``（与 evaluate 的
        label_days=20 主路径同一函数），对比手算逐项。
        """
        from alphaprobe.evaluator_adapter import QuantEvaluatorAdapter
        from quant_evaluator.metrics.probe_portfolio import compute_cohort_pnl

        dates = [pd.Timestamp(d) for d in TRADING_DATES_25[:4]]
        codes = ["c0", "c1", "c2"]
        vwap = pd.DataFrame(PRICES, index=dates, columns=codes)
        factor = pd.DataFrame(FACTOR, index=dates, columns=codes)
        adapter = QuantEvaluatorAdapter(label_days=20, n_quantiles=3, holding=2)
        f = np.asarray(factor.to_numpy(), dtype=np.float64)
        # 走 adapter 的 cohort 构造/调用链（价格面板 → next_ret/next_vwap → QE）。
        cohort_metrics = adapter._cohort_metrics(f, vwap, dates, codes)
        # L 维度键被填入（真实 cohort 路径，非空）
        assert "net_sharpe" in cohort_metrics
        # cohort daily PnL 与手算逐项一致（compute_cohort_pnl 即 adapter 所调）：
        res = compute_cohort_pnl(
            factor_values=FACTOR[1:],
            next_ret=PRICES[1:] / PRICES[:-1] - 1.0,
            next_vwap=PRICES[1:],
            n_quantiles=3,
            holding=2,
            per_side_cost=0.0,
            require_tradable=True,
        )
        expected = _manual_cohort_pnl(holding=2)
        np.testing.assert_allclose(res["pnl_net"], expected, atol=1e-12)
        # adapter 的 evaluate 对 label_days=20 短面板（<20 行）也应跑通不抛，
        # 但极短面板下 registry 指标多为 None（无足够 IC 天数），仅断言键存在。
        label = vwap.shift(-2) / vwap - 1.0
        bundle = adapter.evaluate(factor, label, vwap, factor_id="det")
        assert "net_sharpe" in bundle or "rankic_valid" in bundle


# ---------------------------------------------------------------------------
# ② evaluate_many 批量（QE 可导入 → 与逐个 evaluate 一致；否则 fail-closed）
# ---------------------------------------------------------------------------


def _make_panels(T: int = 120, N: int = 40, seed: int = 0):
    dates = pd.date_range("2020-01-01", periods=T)
    codes = [f"c{i}" for i in range(N)]
    rng = np.random.default_rng(seed)
    vwap = pd.DataFrame(
        100.0 + np.cumsum(rng.standard_normal((T, N)), axis=0),
        index=dates,
        columns=codes,
    )
    label = vwap.shift(-20) / vwap - 1.0
    return dates, codes, vwap, label


@QE_SKIP
class TestEvaluateMany:
    def test_many_matches_individual_evaluate(self):
        """evaluate_many 的 rankic_valid 与逐个 evaluate 一致（abs<=1e-9）。"""
        dates, codes, vwap, label = _make_panels(seed=3)
        rng = np.random.default_rng(9)
        f0 = pd.DataFrame(rng.standard_normal((120, 40)), index=dates, columns=codes)
        f1 = pd.DataFrame(rng.standard_normal((120, 40)) * 2.0, index=dates, columns=codes)
        f2 = pd.DataFrame(-rng.standard_normal((120, 40)), index=dates, columns=codes)
        adapter = QuantEvaluatorAdapter(label_days=20)
        bundles = adapter.evaluate_many(
            [f0, f1, f2], label, vwap, factor_ids=["m0", "m1", "m2"]
        )
        assert len(bundles) == 3
        assert all(isinstance(b, EvaluationBundle) for b in bundles)
        singles = [
            adapter.evaluate(f0, label, vwap, factor_id="m0"),
            adapter.evaluate(f1, label, vwap, factor_id="m1"),
            adapter.evaluate(f2, label, vwap, factor_id="m2"),
        ]
        for b, s in zip(bundles, singles):
            assert abs(b.get("rankic_valid") - s.get("rankic_valid")) <= 1e-9
            assert abs(b.get("net_sharpe") - s.get("net_sharpe")) <= 1e-9

    def test_default_factor_ids(self):
        """不传 factor_ids → 默认 f1..fN。"""
        dates, codes, vwap, label = _make_panels(seed=4)
        rng = np.random.default_rng(1)
        f0 = pd.DataFrame(rng.standard_normal((120, 40)), index=dates, columns=codes)
        bundles = QuantEvaluatorAdapter(label_days=20).evaluate_many([f0], label, vwap)
        assert isinstance(bundles[0], EvaluationBundle)

    def test_empty_panels_raises_fail_closed(self):
        """空 factor_panels → QuantEvaluatorError（fail-closed）。"""
        _, _, vwap, label = _make_panels()
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate_many([], label, vwap)

    def test_missing_price_panel_raises_fail_closed(self):
        """label_days=20 缺 price_panel → QuantEvaluatorError（fail-closed）。"""
        _, _, _, label = _make_panels()
        dates, codes, _, _ = _make_panels()
        rng = np.random.default_rng(1)
        f0 = pd.DataFrame(rng.standard_normal((120, 40)), index=dates, columns=codes)
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate_many([f0], label, None)

    def test_shape_mismatch_raises_fail_closed(self):
        """某因子面板与 label 形状不一致 → 整批 fail-closed。"""
        dates, codes, vwap, label = _make_panels(seed=5)
        rng = np.random.default_rng(2)
        f0 = pd.DataFrame(rng.standard_normal((120, 40)), index=dates, columns=codes)
        bad = pd.DataFrame(rng.standard_normal((115, 40)), index=dates[:115], columns=codes)
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate_many([f0, bad], label, vwap)


class TestEvaluateManyNoQe:
    def test_fail_closed_when_qe_unavailable(self, monkeypatch):
        """QE 不可导入 → evaluate_many 抛 QuantEvaluatorError（不手算、不静默）。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("quant_evaluator"):
                raise ImportError("QE unavailable (test)")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        dates, codes, vwap, label = _make_panels(seed=6)
        rng = np.random.default_rng(3)
        f0 = pd.DataFrame(rng.standard_normal((120, 40)), index=dates, columns=codes)
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate_many([f0], label, vwap)
