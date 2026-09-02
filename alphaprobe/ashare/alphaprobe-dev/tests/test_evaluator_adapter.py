"""V2-E QuantEvaluatorAdapter 统一评估层接入 —— 合成数据验收测试。

覆盖（任务书 §31-§32）：
① adapter 输出 EvaluationBundle 字段齐全且与 fitness 白名单一致
② 20d label 走 cohort 路径（LS 指标来自 compute_metrics_from_cohort 而非 overlapping）
③ QE fail-closed 抛错
④ cost_scenario 传递
⑤ 与 pipeline 集成冒烟（stub 因子 → adapter → factor_fitness_v2 出分）

全合成数据，不读真实 COS。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaprobe.evaluator_adapter import (
    COST_SCENARIOS,
    QuantEvaluatorAdapter,
    QuantEvaluatorError,
)
from alphaprobe.fitness.contracts import EvaluationBundle, QE_METRIC_KEYS
from alphaprobe.fitness.factor_fitness import factor_fitness_v2
from alphaprobe.fitness import MetricCalibrator


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_panels(T: int = 120, N: int = 40, seed: int = 0):
    dates = pd.date_range("2020-01-01", periods=T)
    codes = [f"c{i}" for i in range(N)]
    rng = np.random.default_rng(seed)
    factor = pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=codes)
    vwap = pd.DataFrame(
        100.0 + np.cumsum(rng.standard_normal((T, N)), axis=0),
        index=dates,
        columns=codes,
    )
    label = vwap.shift(-20) / vwap - 1.0
    return factor, label, vwap


def _make_calibrator():
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
    c.freeze("v2e_test")
    return c


# ---------------------------------------------------------------------------
# ① EvaluationBundle 字段齐全且与 fitness 白名单一致
# ---------------------------------------------------------------------------


class TestBundleFields:
    def test_output_is_evaluation_bundle(self):
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        assert isinstance(b, EvaluationBundle)

    def test_l_dimension_keys_present(self):
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        for key in (
            "net_sharpe",
            "calmar_ratio",
            "sortino_ratio",
            "net_annualized_ls_return",
            "d10_long_only_active_return",
            "max_drawdown",
            "drawdown_persistence",
            "q20_rolling_sharpe",
            "positive_month_ratio",
        ):
            assert key in b, f"missing L key {key}"

    def test_p_s_r_keys_present(self):
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        for key in ("rankic_valid", "rankicir", "hac_tstat", "coverage", "turnover"):
            assert key in b, f"missing key {key}"

    def test_all_keys_in_fitness_whitelist(self):
        """adapter 产出的键必须全部落在 fitness QE_METRIC_KEYS 白名单内。"""
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        for key in b.raw():
            assert key in QE_METRIC_KEYS, f"key {key!r} not in fitness whitelist"

    def test_ls_metrics_are_finite_or_none(self):
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        for key in (
            "net_sharpe",
            "calmar_ratio",
            "sortino_ratio",
            "max_drawdown",
            "drawdown_persistence",
            "q20_rolling_sharpe",
            "positive_month_ratio",
        ):
            val = b.get(key)
            assert val is None or np.isfinite(val), f"{key} not finite: {val}"


# ---------------------------------------------------------------------------
# ② 20d label 走 cohort 路径（LS 指标来自 compute_metrics_from_cohort）
# ---------------------------------------------------------------------------


class TestCohortPath:
    def test_ls_metrics_come_from_cohort_not_overlapping(self, monkeypatch):
        """断言 net_sharpe 等来自 compute_metrics_from_cohort 而非 overlapping 年化。"""
        import alphaprobe.evaluator_adapter as ea

        calls = {"cohort": 0}
        real = ea.QuantEvaluatorAdapter._cohort_metrics

        def spy(self, f, price, dates, codes):
            calls["cohort"] += 1
            return real(self, f, price, dates, codes)

        monkeypatch.setattr(ea.QuantEvaluatorAdapter, "_cohort_metrics", spy)
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        assert calls["cohort"] == 1
        # cohort 路径产出真实 daily PnL 指标（net_sharpe 非 None 或 NaN 均可）
        assert "net_sharpe" in b

    def test_non_20d_label_skips_cohort(self, monkeypatch):
        """label_days != 20 时不走 cohort（无 price_panel 也不抛）。"""
        import alphaprobe.evaluator_adapter as ea

        calls = {"cohort": 0}
        real = ea.QuantEvaluatorAdapter._cohort_metrics

        def spy(self, f, price, dates, codes):
            calls["cohort"] += 1
            return real(self, f, price, dates, codes)

        monkeypatch.setattr(ea.QuantEvaluatorAdapter, "_cohort_metrics", spy)
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=5).evaluate(f, y, v)
        assert calls["cohort"] == 0
        assert "net_sharpe" not in b

    def test_20d_requires_price_panel(self):
        """label_days=20 但缺 price_panel → fail-closed 抛错。"""
        f, y, _ = _make_panels()
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate(f, y, None)


# ---------------------------------------------------------------------------
# ③ QE fail-closed 抛错
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_qe_unimportable_raises(self, monkeypatch):
        """QE 不可导入 → 抛 QuantEvaluatorError（不是静默 0 分）。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("quant_evaluator"):
                raise ImportError("QE unavailable (test)")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        f, y, v = _make_panels()
        with pytest.raises(QuantEvaluatorError):
            QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)

    def test_bad_cost_scenario_raises(self):
        """非法成本情景 → 抛错（fail-closed）。"""
        f, y, v = _make_panels()
        with pytest.raises(Exception):
            QuantEvaluatorAdapter(label_days=20, cost_scenario="bogus").evaluate(f, y, v)


# ---------------------------------------------------------------------------
# ④ cost_scenario 传递
# ---------------------------------------------------------------------------


class TestCostScenario:
    def test_cost_scenario_forwarded_to_cohort(self, monkeypatch):
        """cost_scenario 透传进 compute_metrics_from_cohort 的 per_side_cost。"""
        import quant_evaluator.metrics.probe_portfolio as pp

        seen = {}

        def fake_cost_scenario(name):
            seen["name"] = name
            return 0.0012 if name == "1x" else 0.0

        monkeypatch.setattr(pp, "cost_scenario", fake_cost_scenario)
        f, y, v = _make_panels()
        QuantEvaluatorAdapter(label_days=20, cost_scenario="1x").evaluate(f, y, v)
        assert seen.get("name") == "1x"

    def test_default_gross(self):
        assert COST_SCENARIOS[0] == "gross"
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        assert "net_sharpe" in b


# ---------------------------------------------------------------------------
# ⑤ 与 pipeline 集成冒烟（stub 因子 → adapter → factor_fitness_v2 出分）
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_adapter_bundle_feeds_factor_fitness_v2(self):
        f, y, v = _make_panels()
        b = QuantEvaluatorAdapter(label_days=20).evaluate(f, y, v)
        c = _make_calibrator()
        r = factor_fitness_v2(b, c)
        assert 0.0 <= r.fitness <= 1.0
        assert 0.0 <= r.L <= 1.0
        assert 0.0 <= r.P <= 1.0

    def test_make_qe_evaluate_fn_produces_bundles(self):
        """pipeline.make_qe_evaluate_fn 用假 stock_data 产出 adapter bundle。"""
        from alphaprobe.pipeline import make_qe_evaluate_fn

        class FakeStockData:
            def __init__(self):
                self._field_names = lambda: ["open", "high", "low", "close", "vwap", "volume"]
                self._dates = pd.date_range("2020-01-01", periods=120)
                self._stock_ids = pd.Index([f"c{i}" for i in range(40)])
                self.data = np.zeros((120, 40, 6), dtype=float)
                for t in range(120):
                    self.data[t, :, 4] = 100.0 + 0.1 * t + np.random.default_rng(t).standard_normal(40)
                    self.data[t, :, 3] = 100.0 + 0.1 * t

            def evaluate_many(self, formulas):
                return [np.random.default_rng(i).standard_normal((120, 40)) for i in range(len(formulas))]

        fn = make_qe_evaluate_fn(FakeStockData(), label_days=20)
        out = fn(["rank(close)", "ts_mean(close, 20)"], "L2_full_train", None)
        assert len(out) == 2
        for b in out:
            assert b is not None
            assert "net_sharpe" in b
            assert "rankic_valid" in b

    def test_make_qe_evaluate_fn_none_data_all_none(self):
        from alphaprobe.pipeline import make_qe_evaluate_fn

        fn = make_qe_evaluate_fn(None)
        out = fn(["rank(close)"], "L2_full_train", None)
        assert len(out) == 1
        assert out[0] is None
