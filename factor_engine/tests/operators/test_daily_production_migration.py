# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 daily-production migration (review §11).

Covers:
- B1  group_ts_decay_linear real time-decay golden; honest group_decay_linear alias.
- B2  group fallback_policy (nan / global / keep_original) on whole-column-NaN group.
- B3  relation transition: NaN must not create phantom entries/exits (pair-valid).
- B4  event_decay_asof: NaN before first valid event; carry vs break policies.
- B5  event return split: compounded vs arithmetic sum golden cases.
- B6  ts_regression_forecast_error out-of-sample golden.
- B7  cs_bucket_fixed / cs_bucket_historical.
-     daily migration: migrated factor ops classify daily; recursive/excluded stay
      extended; zero unclassified canonicals; six-gate certification fields present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

import cleaned_operators.operator_surface as S  # noqa: E402


def _dates(n: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


# ---------------------------------------------------------------------------
# B1 group_ts_decay_linear (real time decay)
# ---------------------------------------------------------------------------

def test_group_ts_decay_linear_matches_hand_computed_wma() -> None:
    dates = _dates(4)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    g = pd.DataFrame({"A": "g1"}, index=dates, dtype=object)
    out = OperatorRegistry.get("group_ts_decay_linear").calculate(x, g, window=3, normalize=False)
    # trailing window [2,3,4], linear weights [1,2,3] normalized -> (2*1+3*2+4*3)/6 = 20/6
    assert out["A"].iloc[-1] == pytest.approx(20.0 / 6.0, abs=1e-9)


def test_group_ts_decay_linear_unknown_group_is_nan() -> None:
    dates = _dates(4)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    g = pd.DataFrame({"A": [np.nan, "g1", "g1", "g1"]}, index=dates, dtype=object)
    out = OperatorRegistry.get("group_ts_decay_linear").calculate(x, g, window=3)
    assert np.isnan(out["A"].iloc[0])  # 单股分组未知 → NaN
    assert np.isfinite(out["A"].iloc[-1])


def test_group_decay_linear_honest_rank_weighted_alias() -> None:
    assert OperatorRegistry._aliases["group_rank_linear_weighted_value"] == "group_decay_linear"


# ---------------------------------------------------------------------------
# B2 group fallback_policy
# ---------------------------------------------------------------------------

def test_group_fallback_nan_global_keep_original() -> None:
    dates = _dates(3)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=dates)
    g = pd.DataFrame({"A": ["g1", np.nan, np.nan], "B": ["g1", np.nan, np.nan]}, index=dates, dtype=object)
    mean = OperatorRegistry.get("group_mean")

    nan_row = mean.calculate(x, g, fallback_policy="nan").iloc[-1]
    assert np.isnan(nan_row["A"]) and np.isnan(nan_row["B"])

    global_row = mean.calculate(x, g, fallback_policy="global").iloc[-1]
    assert global_row["A"] == pytest.approx((3.0 + 6.0) / 2.0)  # 全截面均值

    keep_row = mean.calculate(x, g, fallback_policy="keep_original").iloc[-1]
    assert keep_row["A"] == pytest.approx(3.0)


def test_group_demean_fallback_default_nan() -> None:
    dates = _dates(2)
    x = pd.DataFrame({"A": [1.0, 2.0], "B": [10.0, 20.0]}, index=dates)
    g = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    out = OperatorRegistry.get("group_neutralize").calculate(x, g)
    assert np.isnan(out.iloc[-1]).all()
    out_global = OperatorRegistry.get("group_neutralize").calculate(x, g, fallback_policy="global")
    assert np.isfinite(out_global.iloc[-1]).all()


# ---------------------------------------------------------------------------
# B3 relation transition: NaN must not create phantom entries/exits
# ---------------------------------------------------------------------------

def test_relation_transition_nan_breaks_sequence() -> None:
    from cleaned_operators.relation.ops import RelationEntryCount, RelationExitCount

    mem = pd.DataFrame({"A": [1.0, np.nan, 1.0, 0.0]}, index=_dates(4))
    # 唯一已知转换是 row2->row3 的 1→0（退出）；NaN 两侧都不计数。
    assert RelationExitCount()._calculate_series(mem, 4)["A"].iloc[-1] == pytest.approx(1.0)
    assert RelationEntryCount()._calculate_series(mem, 4)["A"].iloc[-1] == pytest.approx(0.0)
    # legacy missing_policy="false" 把 NaN 当非成员 → NaN->1 计为一次进入。
    assert RelationEntryCount()._calculate_series(mem, 4, missing_policy="false")["A"].iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B4 event_decay_asof
# ---------------------------------------------------------------------------

def test_event_decay_asof_nan_before_first_valid() -> None:
    from cleaned_operators.state_event import EventDecayAsOf

    dates = _dates(4)
    ev = pd.DataFrame({"A": [np.nan, 1.0, 0.0, 0.0]}, index=dates)
    out = EventDecayAsOf()._calculate_series(ev, half_life=1.0)
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == pytest.approx(1.0)
    assert out["A"].iloc[2] == pytest.approx(0.5)


def test_event_decay_asof_break_resets() -> None:
    from cleaned_operators.state_event import EventDecayAsOf

    dates = _dates(5)
    ev = pd.DataFrame({"A": [1.0, np.nan, 1.0, 0.0, 0.0]}, index=dates)
    out = EventDecayAsOf()._calculate_series(ev, half_life=1.0, missing_policy="break")
    vals = out["A"].to_numpy()
    assert vals[0] == pytest.approx(1.0)
    assert np.isnan(vals[1])
    assert vals[2] == pytest.approx(1.0)  # 打断后重启
    assert vals[3] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# B5 event return split
# ---------------------------------------------------------------------------

def test_event_return_split_goldens() -> None:
    dates = _dates(5)
    ret = pd.DataFrame({"A": [0.01] * 5}, index=dates)
    ev = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=dates)
    compounded = OperatorRegistry.get("event_return_since_last").calculate(ret, ev, window=5, event_effective_lag=0)
    assert compounded["A"].iloc[-1] == pytest.approx(1.01 ** 4 - 1.0, abs=1e-9)
    arith = OperatorRegistry.get("event_arithmetic_return_sum").calculate(ret, ev, window=5, event_effective_lag=0)
    assert arith["A"].iloc[-1] == pytest.approx(0.04, abs=1e-9)
    logsum = OperatorRegistry.get("event_log_return_sum").calculate(ret, ev, window=5, event_effective_lag=0)
    assert logsum["A"].iloc[-1] == pytest.approx(4 * np.log1p(0.01), abs=1e-9)
    cnt = OperatorRegistry.get("event_active_count").calculate(ret, ev, window=5, event_effective_lag=0)
    assert cnt["A"].iloc[-1] == pytest.approx(4.0)
    # event_compounded_return 是复利口径别名
    alias = OperatorRegistry.get("event_compounded_return").calculate(ret, ev, window=5, event_effective_lag=0)
    pd.testing.assert_frame_equal(alias, compounded, check_dtype=False)


# ---------------------------------------------------------------------------
# B6 ts_regression_forecast_error out-of-sample
# ---------------------------------------------------------------------------

def test_regression_forecast_error_is_out_of_sample() -> None:
    dates = _dates(6)
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)
    fe = OperatorRegistry.get("ts_regression_forecast_error").calculate(y, x, window=4, min_periods=3)
    assert abs(fe["A"].iloc[-1]) < 1e-9  # y=x → 预测误差 0
    # 样本内残差与 forecast_error 是不同算子
    resid = OperatorRegistry.get("ts_regression_in_sample_resid").calculate(y, x, window=4, min_periods=3)
    assert OperatorRegistry.get("ts_regression_forecast_error") is not None
    assert np.isfinite(resid["A"].iloc[-1])


# ---------------------------------------------------------------------------
# B7 cs_bucket_fixed / cs_bucket_historical
# ---------------------------------------------------------------------------

def test_cs_bucket_fixed_uses_breaks() -> None:
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.nan]})
    out = OperatorRegistry.get("cs_bucket_fixed").calculate(x, [1.5, 2.5])
    assert out["A"].tolist()[:3] == [1.0, 2.0, 3.0]
    assert np.isnan(out["A"].iloc[3])


def test_cs_bucket_historical_uses_past_quantiles() -> None:
    dates = _dates(5)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    out = OperatorRegistry.get("cs_bucket_historical").calculate(x, window=4, quantiles=(0.5,), min_periods=3)
    assert np.isnan(out["A"].iloc[2])  # 历史样本不足
    assert out["A"].iloc[-1] == pytest.approx(2.0)  # 5 > 历史中位数 2.5


# ---------------------------------------------------------------------------
# Daily migration surface verification
# ---------------------------------------------------------------------------

def test_migrated_factor_ops_classify_daily() -> None:
    for name in (
        "ts_argmax", "ts_argmin", "ts_regression_forecast_error", "ts_time_slope",
        "cs_bucket", "cs_bucket_fixed", "fin_yoy", "fin_ttm_quarterly",
        "group_ts_decay_linear", "cdl_doji", "pattern_double_top", "amihud_illiquidity",
        # second-round promotion: experimental factor operators now daily
        "cs_robust_resid", "ts_transition_count", "suspension_frequency",
        "listing_age", "index_reconstitution_churn", "intra_realized_variance",
    ):
        assert S.classify_canonical(name) == "daily", name


def test_third_round_promoted_ops_classify_daily() -> None:
    # 2026-08 第三轮:递归/状态族(经 full-replay 生产路径)、holder ID 匹配重写、
    # relation 逐对有效、market-model PIT 认证、intraday_volatility 日频化。
    for name in (
        "KAMA", "Supertrend", "SupertrendDirection", "PSAR", "ts_ema",
        "RSI_WILDER", "ATR_WILDER", "ADX", "MACD_line", "MACD_signal",
        "MACD_hist", "expanding_rank", "hump_decay", "ts_sma_cn", "trade_when",
        "ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr",
        "group_decay_linear", "rank_corr", "ts_sum_decay", "ts_max_buildup",
        "ts_moment", "ts_poly2_coeff", "ts_poly2_resid", "digital_count",
        "lqtp_historical_cvar",
        "holder_weighted_churn", "holder_entry_share", "holder_exit_share",
        "holder_net_entry_share", "holder_rank_stability",
        "relation_entry_count", "relation_exit_count", "relation_weighted_change",
        "multi_index_entry_intensity",
        "tail_beta", "residual_momentum_capm", "coskewness_to_market",
        "idio_vol", "idio_skew", "intraday_volatility",
    ):
        assert S.classify_canonical(name) == "daily", name


def test_intentionally_off_daily_ops_stay_extended() -> None:
    # 迁移桩(fin_ttm / rolling_beta_to_market)、逐日广播的非因子工具
    # (relation_distinct_count / relation_overlap_ratio)、真 session-aware
    # (intraday_vwap_deviation)有意保留在 extended,不进入 daily DSL。
    for name in (
        "fin_ttm", "rolling_beta_to_market",
        "relation_distinct_count", "relation_overlap_ratio",
        "intraday_vwap_deviation",
    ):
        assert S.classify_canonical(name) != "daily", name


def test_no_unclassified_canonicals() -> None:
    assert S.unclassified_canonicals(OperatorRegistry.list_canonical()) == ()


def test_daily_dsl_exposes_migrated_ops_and_hides_off_daily() -> None:
    from api.operator_registry import build_dsl_allowlist

    pub = build_dsl_allowlist()
    assert "ts_regression_forecast_error" in pub
    assert "cs_bucket_fixed" in pub
    # 第三轮提升:递归/状态族与重写算子进入 daily DSL。
    assert "KAMA" in pub
    assert "MACD_line" in pub
    assert "holder_weighted_churn" in pub
    assert "multi_index_entry_intensity" in pub
    # 迁移桩 / 非因子工具 / session-aware 保持隐藏。
    assert "fin_ttm" not in pub
    assert "relation_distinct_count" not in pub
    assert "intraday_vwap_deviation" not in pub


# ---------------------------------------------------------------------------
# Six-gate certification structure
# ---------------------------------------------------------------------------

def test_six_gate_certification_fields_present() -> None:
    from cleaned_operators.semantic_certification import OperatorCertification, operator_certification_for

    for name in ("ts_mean", "ts_argmax"):
        catalog = OperatorRegistry._catalog[name]
        for field in ("implementation_certified", "semantic_certified", "temporal_certified",
                      "source_contract_certified", "edge_case_passed", "backend_passed"):
            assert field in catalog, field
        cert = operator_certification_for(name, catalog)
        assert isinstance(cert, OperatorCertification)
        assert cert.production_certified == all(
            (cert.implementation_passed, cert.semantic_passed, cert.temporal_passed,
             cert.source_pit_passed, cert.edge_case_passed, cert.backend_passed)
        )
