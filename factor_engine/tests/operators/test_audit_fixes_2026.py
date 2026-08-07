# -*- coding: utf-8
"""Regression tests for the 2026 audit remediation pass.

Covers:
  S2   production promotion gate (fail-closed for experimental/research ops)
  S4.5 negative-lag rejection (FutureReferenceError)
  S5.3 fillna unknown-method rejection
  S6.1 out-of-sample conditional regression residual
  S9   index/listing/suspension unknown-state semantics
  S10  financial PIT edge cases (zero-variance z-score, constant R2, fiscal year)
  S11  valuation scale-homogenised dispersion + true WLS
  S15.3 explicit pct_change fill_method
  S17.3 duplicate policy key removal
  S19  source-safe operator building blocks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from backend.operator_errors import FutureReferenceError
from cleaned_operators.registry import OperatorRegistry


# --------------------------------------------------------------------------
# S2: production promotion gate
# --------------------------------------------------------------------------

def test_s2_experimental_registered_ops_not_promoted():
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.semantic_certification import should_fail_closed

    targets = factor_production_targets()
    # Model-family / research operators must be fail-closed and non-production.
    # The causal reworked variants (ts_ar_prior_forecast, ts_*_forecast_error)
    # are promoted replacements (audit §9.3-§9.5); the diagnostic in-sample
    # variants and un-reworked model families stay experimental.
    for name in ("ts_garch_vol_surprise", "ts_ar_forecast",
                 "ts_huber_regression_resid", "ts_expectile_regression_resid"):
        assert should_fail_closed(name), name
        catalog = OperatorRegistry._catalog.get(name, {})
        assert str(catalog.get("status")) == "experimental", name
        assert catalog.get("pit_safe") is False, name
        assert catalog.get("production_certified") is False, name


def test_s2_isolated_ops_not_promoted():
    from cleaned_operators.semantic_certification import should_fail_closed

    # fin_ttm / nan_to_num remain isolated (legacy stub / NaN sink conflation).
    for name in ("fin_ttm", "nan_to_num"):
        assert should_fail_closed(name), name
        # Registered operators get the experimental lifecycle; unclassified ones
        # are simply absent from the catalog (never promoted either way).
        catalog = OperatorRegistry._catalog.get(name)
        if catalog is None:
            continue
        assert str(catalog.get("status")) == "experimental", name
        assert catalog.get("pit_safe") is False, name


def test_s2_reworked_isolated_ops_are_promoted():
    # 2026-08 第三轮:relation_weighted_change / holder_weighted_churn 完成
    # 语义重写(ID 匹配),已从隔离清单解除并升到 daily。
    from cleaned_operators.semantic_certification import should_fail_closed
    from cleaned_operators.operator_surface import classify_canonical

    for name in ("relation_weighted_change", "holder_weighted_churn",
                 "relation_entry_count", "multi_index_entry_intensity"):
        assert not should_fail_closed(name), name
        assert classify_canonical(name) == "daily", name


def test_s2_reworked_unknown_state_ops_are_promoted():
    # index_reconstitution_churn / listing_age / suspension_frequency completed the
    # unknown-state rework (S9) and are now on the daily surface, not fail-closed.
    from cleaned_operators.semantic_certification import should_fail_closed
    from cleaned_operators.operator_surface import classify_canonical

    for name in ("index_reconstitution_churn", "listing_age", "suspension_frequency"):
        assert not should_fail_closed(name), name
        assert classify_canonical(name) == "daily", name


def test_s2_legitimate_production_surface_kept():
    from cleaned_operators.production_hardening import factor_production_targets

    targets = factor_production_targets()
    for name in ("ts_mean", "ts_argmax", "ts_time_slope", "ts_regression_resid",
                 "ts_topk_mean", "MACD_line", "RSI_WILDER"):
        assert name in targets, name
        catalog = OperatorRegistry._catalog.get(name, {})
        # Six-gate certification is the single production authority (review
        # §2.3/§2.4): status may only be "production" when evidence certifies
        # it.  Without current evidence the operator fails closed to
        # experimental even though it remains a reviewed production target.
        if catalog.get("production_certified") is True:
            assert str(catalog.get("status")) == "production", name
            assert catalog.get("pit_safe") is True, name
        else:
            assert str(catalog.get("status")) == "experimental", name
            assert catalog.get("pit_safe") is False, name


def test_s2_four_certificates_attached():
    catalog = OperatorRegistry._catalog["ts_mean"]
    for field in ("implementation_certified", "semantic_certified",
                  "temporal_certified", "source_contract_certified"):
        assert field in catalog, field


# --------------------------------------------------------------------------
# S4.5: negative lag must raise, never silently return all-NaN
# --------------------------------------------------------------------------

def test_s4_5_negative_lag_raises():
    from cleaned_operators.common.statistics import ACF, autocorr

    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(FutureReferenceError):
        ACF()._calculate_series(frame, window=3, lag=-1)
    with pytest.raises(FutureReferenceError):
        autocorr()._calculate_series(frame, lag=-1)


def test_s4_5_causal_lag_raises():
    from cleaned_operators._causal import causal_lag

    frame = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(FutureReferenceError):
        causal_lag(frame, -1)


# --------------------------------------------------------------------------
# S5.3: fillna unknown method must fail loudly
# --------------------------------------------------------------------------

def test_s5_3_fillna_unknown_method_raises():
    from cleaned_operators.common.data_cleaning import FillNA

    frame = pd.DataFrame({"A": [1.0, np.nan]})
    with pytest.raises(ValueError):
        FillNA()._calculate_series(frame, "definitely_not_a_method")


# --------------------------------------------------------------------------
# S6.1: conditional regression residual must be out-of-sample
# --------------------------------------------------------------------------

def test_s6_1_conditional_resid_excludes_current_row():
    from cleaned_operators.conditional_ext import TsRegressionResidIf

    index = pd.date_range("2024-01-01", periods=5)
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    cond = pd.DataFrame({"A": [1.0] * 5}, index=index)
    out = TsRegressionResidIf()._calculate_series(y, x, cond, window=5, min_periods=3)
    # Fit on rows 0..3 (y = x), predict x4 = 5 -> residual 5 - 5 = 0.
    assert out["A"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# S9: index/listing/suspension unknown-state semantics
# --------------------------------------------------------------------------

def test_s9_listing_age_pre_listing_is_nan():
    listing = pd.DataFrame(
        {"A": [np.nan, np.nan, pd.Timestamp("2024-01-03")]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    out = OperatorRegistry.get("listing_age").calculate(listing)
    assert np.isnan(out["A"].iloc[0])
    assert np.isnan(out["A"].iloc[1])
    assert out["A"].iloc[2] == 0.0  # listing day is age 0, not NaN


def test_s9_suspension_frequency_uses_known_denominator():
    sus = pd.DataFrame(
        {"A": [1.0, np.nan, 1.0, 0.0]},
        index=pd.date_range("2024-01-01", periods=4),
    )
    out = OperatorRegistry.get("suspension_frequency").calculate(sus, 4)
    # Known days: rows 0,2,3 = 3 days; suspended known: rows 0,2 = 2 -> 2/3.
    assert out["A"].iloc[-1] == pytest.approx(2.0 / 3.0)


def test_s9_reconstitution_churn_ignores_unknown_boundaries():
    # NaN -> 1 must NOT be counted as an entry (unknown treated as non-member).
    mem = pd.DataFrame(
        {"A": [1.0, np.nan, 1.0, 0.0]},
        index=pd.date_range("2024-01-01", periods=4),
    )
    out = OperatorRegistry.get("index_reconstitution_churn").calculate(mem, 4)
    # Only the 1 -> 0 exit at row 3 (both sides known) counts.
    assert out["A"].iloc[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# S10: financial PIT edge cases
# --------------------------------------------------------------------------

def test_s10_zscore_history_zero_variance_is_nan():
    from cleaned_operators.fundamental.transforms_v2 import fin_zscore_history

    index = pd.date_range("2024-01-01", periods=4)
    x = pd.DataFrame({"A": [5.0, 5.0, 5.0, 5.0]}, index=index)
    pid = pd.DataFrame(
        {"A": [pd.Timestamp(d) for d in ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")]},
        index=index,
    )
    out = fin_zscore_history(x, pid, 4)
    assert np.isnan(out["A"].iloc[-1])


def test_s10_trend_r2_constant_series_is_nan():
    from cleaned_operators.fundamental.transforms_v2 import _trend_stat

    index = pd.date_range("2024-01-01", periods=4)
    x = pd.DataFrame({"A": [5.0, 5.0, 5.0, 5.0]}, index=index)
    pid = pd.DataFrame(
        {"A": [pd.Timestamp(d) for d in ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")]},
        index=index,
    )
    out = _trend_stat(x, pid, 4, "r2")
    assert np.isnan(out["A"].iloc[-1])


def test_s10_quarter_from_cumulative_requires_same_fiscal_year():
    from cleaned_operators.fundamental.flow_semantics_v2 import fin_quarter_from_cumulative

    index = pd.date_range("2024-01-01", periods=2)
    cum = pd.DataFrame({"A": [100.0, 300.0]}, index=index)
    pid = pd.DataFrame(
        {"A": [pd.Timestamp("2024-03-31"), pd.Timestamp("2025-06-30")]},
        index=index,
    )
    quarter = pd.DataFrame({"A": [1, 2]}, index=index)
    out = fin_quarter_from_cumulative(cum, pid, quarter)
    assert out["A"].iloc[0] == pytest.approx(100.0)
    assert np.isnan(out["A"].iloc[1])  # Q2 2025 must not subtract Q1 2024


# --------------------------------------------------------------------------
# S11: valuation scale homogenisation + WLS
# --------------------------------------------------------------------------

def test_s11_cashflow_disagreement_homogenises_scale():
    from cleaned_operators.valuation.ops_v2 import _valuation_cashflow_disagreement

    index = pd.date_range("2024-01-01", periods=6)
    pe = pd.DataFrame({"A": [10.0, 20.0, 15.0, 30.0, 8.0, 12.0]}, index=index)
    ocf = pd.DataFrame({"A": [0.05, 0.02, 0.08, 0.01, 0.03, 0.04]}, index=index)
    ey = pd.DataFrame({"A": [0.10, 0.05, 0.12, 0.04, 0.06, 0.08]}, index=index)
    out = _valuation_cashflow_disagreement(pe, pe * 2, pe * 3, ocf, ey)
    assert out.shape == pe.shape
    assert np.isfinite(out.to_numpy()).all()


def test_s11_quality_mismatch_uses_wls():
    from cleaned_operators.valuation.ops_v2 import _valuation_quality_mismatch

    # A real cross-section needs several instruments per row.
    index = pd.date_range("2024-01-01", periods=3)
    cols = [f"S{i}" for i in range(10)]
    valuation = pd.DataFrame(np.linspace(1, 10, 30).reshape(3, 10), index=index, columns=cols)
    quality = pd.DataFrame(np.tile(np.linspace(1, 10, 10), (3, 1)), index=index, columns=cols)
    weight = pd.DataFrame(np.full((3, 10), 2.0), index=index, columns=cols)
    out = _valuation_quality_mismatch(valuation, quality, weight)
    assert np.isfinite(out.to_numpy()).all()


# --------------------------------------------------------------------------
# S15.3: explicit pct_change fill_method
# --------------------------------------------------------------------------

def test_s15_3_pct_change_explicit_no_fill():
    frame = pd.DataFrame({"A": [1.0, np.nan, 4.0, 8.0]})
    # pandas 2.x still defaults to the deprecated fill_method='pad'; the explicit
    # fill_method=None form must NOT forward-fill the missing prior value.
    explicit = frame.pct_change(fill_method=None)
    assert np.isnan(explicit["A"].iloc[1])
    assert np.isnan(explicit["A"].iloc[2])
    # volume_volatility must not forward-fill a missing prior volume either.
    from cleaned_operators.price_volume.liquidity_v2 import volume_volatility

    out = volume_volatility(frame, 2)
    assert np.isnan(out["A"].iloc[1])


# --------------------------------------------------------------------------
# S17.3: duplicate policy key removed
# --------------------------------------------------------------------------

def test_s17_3_real_turnover_rate_policy_single():
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    policy = _EXPLICIT_POLICIES.get("real_turnover_rate")
    assert policy is not None
    # The surviving entry must be elementwise (volume / float_shares daily rate),
    # not a duplicated ts-scope that silently overrode it.
    assert policy.get("scope") == "elementwise"


# --------------------------------------------------------------------------
# S19: source-safe building blocks
# --------------------------------------------------------------------------

def test_s19_validity_and_staleness():
    index = pd.date_range("2024-01-01", periods=5)
    x = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=index)
    assert OperatorRegistry.get("ts_valid_count").calculate(x, 5, 1)["A"].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert OperatorRegistry.get("ts_coverage_ratio").calculate(x, 5, 1)["A"].tolist() == [0.2, 0.2, 0.4, 0.4, 0.6]
    assert OperatorRegistry.get("ts_staleness").calculate(x, 5)["A"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_s19_ffill_limited_and_log_domain():
    index = pd.date_range("2024-01-01", periods=5)
    x = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=index)
    out = OperatorRegistry.get("ts_ffill_limited").calculate(x, 1)
    assert out["A"].tolist() == [1.0, 1.0, 3.0, 3.0, 5.0]
    lg = OperatorRegistry.get("log_positive_or_nan").calculate(
        pd.DataFrame({"A": [1.0, -1.0, 0.0, 2.0, np.nan]}, index=index)
    )
    assert lg["A"].tolist()[0] == pytest.approx(0.0)
    assert np.isnan(lg["A"].tolist()[1])
    assert np.isnan(lg["A"].tolist()[2])


def test_s19_unambiguous_extreme_position():
    index = pd.date_range("2024-01-01", periods=6)
    x = pd.DataFrame({"A": [1.0, 3.0, 3.0, 2.0, 0.0, 0.0]}, index=index)
    age = OperatorRegistry.get("ts_argmax_age").calculate(x, 4, 1)["A"].tolist()
    # Window at last row = [3,2,0,0]; most recent max at age 3 (row 2).
    assert age[-1] == pytest.approx(3.0)
    idx_from_oldest = OperatorRegistry.get("ts_argmax_index_from_oldest").calculate(x, 4, 1)["A"].tolist()
    assert idx_from_oldest[-1] == pytest.approx(0.0)


def test_s19_cs_impute_explicit_broadcast():
    x = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [3.0, 5.0], "C": [np.nan, 4.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    out = OperatorRegistry.get("cs_impute_mean").calculate(x)
    # Row 0 mean of finite = (1 + 3)/2 = 2 -> A=1, B=3, C=2.
    assert out["C"].iloc[0] == pytest.approx(2.0)
