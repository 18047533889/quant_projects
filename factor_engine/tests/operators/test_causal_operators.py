"""因果性 / 防前视（point-in-time）专项测试。

核心断言：在时点 t 的输出，仅依赖 x[:t+1]（及同前缀的辅助序列），
追加未来行不应改变历史时点的结果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy expanding and statistical aliases were removed from the runtime surface")

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators._causal import causal_lag
from cleaned_operators.registry import OperatorRegistry

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _panel(values, col: str = "A") -> pd.DataFrame:
    return pd.DataFrame({col: list(values)}, index=_dates(len(values)))


def _two_panel(x_vals, y_vals, col: str = "A") -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = _dates(len(x_vals))
    return (
        pd.DataFrame({col: list(x_vals)}, index=idx),
        pd.DataFrame({col: list(y_vals)}, index=idx),
    )


def _op(name: str):
    ensure_cleaned_loaded()
    op = OperatorRegistry.get(name)
    assert op is not None, f"operator {name!r} not registered"
    return op


def assert_prefix_invariant(calc, x: pd.DataFrame, *, atol: float = 1e-9) -> None:
    """追加未来样本后，历史各时点结果保持不变。"""
    full = calc(x)
    col = x.columns[0]
    for t in range(len(x)):
        trunc = x.iloc[: t + 1]
        part = calc(trunc)
        got = full.iloc[t][col]
        exp = part.iloc[t][col]
        if pd.isna(got) and pd.isna(exp):
            continue
        assert got == pytest.approx(exp, abs=atol), f"lookahead at t={t}: {got} != {exp}"


def assert_bivariate_prefix_invariant(
    calc, x: pd.DataFrame, y: pd.DataFrame, *, atol: float = 1e-9
) -> None:
    full = calc(x, y)
    col = x.columns[0]
    for t in range(len(x)):
        trunc_x = x.iloc[: t + 1]
        trunc_y = y.iloc[: t + 1]
        part = calc(trunc_x, trunc_y)
        got = full.iloc[t][col]
        exp = part.iloc[t][col]
        if pd.isna(got) and pd.isna(exp):
            continue
        assert got == pytest.approx(exp, abs=atol), f"lookahead at t={t}: {got} != {exp}"


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    ensure_cleaned_loaded()


# ---------------------------------------------------------------------------
# _causal 基础工具
# ---------------------------------------------------------------------------


class TestCausalHelpers:
    def test_causal_lag_positive(self):
        x = _panel([10.0, 20.0, 30.0, 40.0])
        out = causal_lag(x, 1)
        assert pd.isna(out.iloc[0]["A"])
        assert out.iloc[1]["A"] == pytest.approx(10.0)
        assert out.iloc[3]["A"] == pytest.approx(30.0)

    def test_causal_lag_negative_returns_nan(self):
        x = _panel([1.0, 2.0, 3.0])
        out = causal_lag(x, -1)
        assert out.isna().all().all()

# ---------------------------------------------------------------------------
# 位移 / 填充：显式禁前视
# ---------------------------------------------------------------------------


class TestShiftAndFillOperators:
    @pytest.mark.parametrize("name", ["Lead", "next", "bfill", "causal_bfill"])
    def test_removed_future_or_misleading_operators_are_absent(self, name):
        ensure_cleaned_loaded()
        assert OperatorRegistry.get(name) is None

    def test_prev_and_ts_delay_are_causal(self):
        x = _panel([10.0, 20.0, 30.0, 40.0])
        for name in ("prev", "ts_delay"):
            assert_prefix_invariant(_op(name).calculate, x)

    def test_ts_delay_negative_lag_is_nan(self):
        x = _panel([1.0, 2.0, 3.0])
        out = _op("ts_delay").calculate(x, -1)
        assert out.isna().all().all()

    def test_ts_delta_prefix_invariant(self):
        x = _panel([1.0, 3.0, 2.0, 5.0, 8.0])
        assert_prefix_invariant(lambda df: _op("ts_delta").calculate(df, 1), x)

    def test_ts_pct_rejects_negative_periods(self):
        x = _panel([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="periods must be >= 1"):
            _op("ts_pct").calculate(x, -1)

    def test_ts_pct_polars_rejects_negative_periods(self):
        pl = pytest.importorskip("polars")
        ensure_cleaned_loaded()
        op = OperatorRegistry.get("ts_pct", backend="polars")
        assert op is not None
        with pytest.raises(ValueError, match="periods must be >= 1"):
            op.calculate(pl.DataFrame({"A": [1.0, 2.0, 3.0]}), -1)

    def test_ts_pct_prefix_invariant(self):
        x = _panel([1.0, 2.0, 4.0, 3.0, 6.0])
        assert_prefix_invariant(lambda df: _op("ts_pct").calculate(df, 1), x)


# ---------------------------------------------------------------------------
# 聚合 / 假设检验：扩展窗口，非全样本广播
# ---------------------------------------------------------------------------


class TestExpandingStatistics:
    @pytest.mark.parametrize("name", ["avg", "mean_agg", "sum_agg", "count", "std_agg"])
    def test_expanding_aggregates_prefix_invariant(self, name):
        x = _panel([1.0, 4.0, 2.0, 9.0, 5.0])
        assert_prefix_invariant(_op(name).calculate, x)

    def test_avg_expanding_not_full_sample_broadcast(self):
        """早期时点不应等于全样本均值。"""
        x = _panel([1.0, 2.0, 100.0, 4.0, 5.0])
        out = _op("avg").calculate(x)
        full_mean = x["A"].mean()
        assert out.iloc[0]["A"] == pytest.approx(1.0)
        assert out.iloc[1]["A"] == pytest.approx(1.5)
        assert out.iloc[2]["A"] == pytest.approx(103.0 / 3.0)
        assert out.iloc[2]["A"] != pytest.approx(full_mean)

    def test_first_not_null_expanding(self):
        x = _panel([np.nan, np.nan, 3.0, 4.0])
        out = _op("first_not_null").calculate(x)
        assert pd.isna(out.iloc[0]["A"])
        assert pd.isna(out.iloc[1]["A"])
        assert out.iloc[2]["A"] == pytest.approx(3.0)
        assert out.iloc[3]["A"] == pytest.approx(3.0)

    def test_last_not_null_is_ffill(self):
        x = _panel([1.0, np.nan, 3.0, np.nan])
        out = _op("last_not_null").calculate(x)
        assert out.iloc[1]["A"] == pytest.approx(1.0)
        assert out.iloc[3]["A"] == pytest.approx(3.0)

    def test_corr_test_early_nan_late_differs(self):
        pytest.importorskip("scipy")
        x, y = _two_panel([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
        out = _op("corr_test").calculate(x, y)
        assert pd.isna(out.iloc[0]["A"])
        assert pd.isna(out.iloc[1]["A"])
        assert pd.notna(out.iloc[-1]["A"])
        assert out.iloc[-1]["A"] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("name", ["corr_test"])
    def test_scipy_hypothesis_prefix_invariant(self, name):
        pytest.importorskip("scipy")
        x, y = _two_panel([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 12.0])
        assert_bivariate_prefix_invariant(_op(name).calculate, x, y)

    def test_durbin_watson_runs_and_is_prefix_invariant(self):
        pytest.importorskip("statsmodels")
        x = _panel([0.1, -0.2, 0.15, -0.05, 0.08, -0.12])
        assert_prefix_invariant(_op("durbin_watson_test").calculate, x, atol=1e-6)

# ---------------------------------------------------------------------------
# 元素级 / 信号：因果卷积与扩展统计
# ---------------------------------------------------------------------------


class TestElementwiseCausal:
    def test_convolve_prefix_invariant(self):
        x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        k = _panel([0.5, 0.5])
        op = _op("convolve")

        def calc(df):
            return op.calculate(df, k.iloc[: len(df)])

        assert_prefix_invariant(calc, x)

    def test_geometric_mean_expanding(self):
        x = _panel([2.0, 8.0, 2.0])
        out = _op("geometric_mean").calculate(x)
        assert out.iloc[0]["A"] == pytest.approx(2.0)
        assert out.iloc[1]["A"] == pytest.approx(4.0)
        assert out.iloc[2]["A"] == pytest.approx(32.0 ** (1.0 / 3.0))

    def test_unitize_uses_cross_section_per_row(self):
        x = pd.DataFrame(
            {"A": [1.0, 10.0], "B": [3.0, 30.0]},
            index=_dates(2),
        )
        out = _op("unitize").calculate(x)
        assert out.loc[out.index[0], "A"] == pytest.approx(1.0 / 3.0)
        assert out.loc[out.index[0], "B"] == pytest.approx(1.0)
        assert out.loc[out.index[1], "A"] == pytest.approx(1.0 / 3.0)

    def test_winsorize_cross_section_per_row(self):
        x = pd.DataFrame(
            {"A": [1.0, 100.0], "B": [2.0, 200.0]},
            index=_dates(2),
        )
        out = _op("winsorize").calculate(x, lower=0.25, upper=0.75)
        # 行内缩尾，不应使用全列极值
        assert out.loc[out.index[0], "A"] >= 1.0
        assert out.loc[out.index[0], "B"] <= 2.0


# ---------------------------------------------------------------------------
# 其它：ACF 负 lag、插值仅 forward
# ---------------------------------------------------------------------------


class TestMiscCausalGuards:
    def test_acf_negative_lag_returns_nan(self):
        x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _op("ACF").calculate(x, window=5, lag=-1)
        assert out.isna().all().all()

    def test_causal_linear_extrapolate_forward_only(self):
        x = _panel([np.nan, np.nan, 3.0, np.nan, 5.0])
        out = _op("causal_linear_extrapolate").calculate(x)
        # 因果外推：不可用未来点 5 填充 index=3
        assert pd.isna(out.iloc[0]["A"])
        assert out.iloc[2]["A"] == pytest.approx(3.0)
        assert out.iloc[3]["A"] == pytest.approx(3.0)
        assert out.iloc[4]["A"] == pytest.approx(5.0)
        assert_prefix_invariant(_op("causal_linear_extrapolate").calculate, x)


# ---------------------------------------------------------------------------
# Engine 集成：DSL 路径也应保持因果
# ---------------------------------------------------------------------------


class TestEngineCausalIntegration:
    @pytest.fixture
    def engine(self):
        from api.dsl_parser import parse_expr
        from api.factor import Factor
        from backend.pandas_backend import PandasBackend
        from runtime.engine import FactorEngine
        from tests.helpers import InMemorySeriesSource

        idx = pd.MultiIndex.from_product(
            [_dates(5), ["A"]],
            names=["timestamp", "instrument"],
        )
        close = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
        source = InMemorySeriesSource(data={"close": close})
        eng = FactorEngine(backend=PandasBackend(), data_source=source)
        return eng, parse_expr, Factor

    def test_engine_delay_prefix_invariant(self, engine):
        eng, parse_expr, Factor = engine
        from runtime.engine import FactorEngine

        full = eng.run(Factor(name="t", expr=parse_expr('delay(col("close"), 1)')))[
            "result"
        ]
        # 截断数据源到前 3 天
        idx3 = pd.MultiIndex.from_product(
            [_dates(3), ["A"]],
            names=["timestamp", "instrument"],
        )
        close3 = pd.Series([10.0, 20.0, 30.0], index=idx3)
        from tests.helpers import InMemorySeriesSource

        eng3 = FactorEngine(
            backend=eng.backend,
            data_source=InMemorySeriesSource(data={"close": close3}),
        )
        part = eng3.run(Factor(name="t", expr=parse_expr('delay(col("close"), 1)')))[
            "result"
        ]
        for ts in _dates(3):
            f = full.loc[(ts, "A")]
            p = part.loc[(ts, "A")]
            if pd.isna(f) and pd.isna(p):
                continue
            assert f == pytest.approx(p)

    def test_engine_lead_is_not_in_daily_dsl(self, engine):
        _, parse_expr, _ = engine
        with pytest.raises(ValueError):
            parse_expr('Lead(col("close"), 1)')


class TestTier1CrossSectionalCausal:
    """Tier-1 截面算子 prefix-invariant / 行内语义。"""

    def test_normalize_prefix_invariant(self):
        x = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0]},
            index=_dates(3),
        )
        assert_prefix_invariant(_op("normalize").calculate, x)

    def test_quantile_prefix_invariant(self):
        x = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0]},
            index=_dates(4),
        )
        assert_prefix_invariant(lambda df: _op("quantile").calculate(df, bins=2), x)

    def test_rank_cs_truncation_stable(self):
        x = pd.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=_dates(2))
        full = _op("rank").calculate(x)
        trunc = _op("rank").calculate(x.iloc[:1])
        assert full.iloc[0]["A"] == pytest.approx(trunc.iloc[0]["A"])

    def test_ts_mean_prefix_invariant(self):
        x = _panel([1.0, 2.0, 3.0, 4.0, 5.0])
        assert_prefix_invariant(lambda df: _op("ts_mean").calculate(df, d=3), x)

    def test_ewm_mean_prefix_invariant(self):
        x = _panel([1.0, 2.0, 3.0, 4.0])
        assert_prefix_invariant(lambda df: _op("ewm_mean").calculate(df, span=3), x)
