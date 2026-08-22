# -*- coding: utf-8 -*-
"""Round-3 audit tests for the ts_model family (dynamic regression / Kalman /
GARCH / HAR).

Covers audit round-3 items 31-34:
* 31 — the dynamic-regression "current coefficient" slot: the legacy in-sample
  coeffs are explicitly documented in-sample; the causal ``*_prior`` slot trains
  on <= t-1 (adds the missing ``ts_quantile_regression_coeff_prior``).
* 32 — Kalman scale/gap: missing observations accumulate ``K*Q`` covariance and
  never re-estimate the noise scale on gap edges; non-finite noise scale fails
  closed.
* 33 — GARCH / HAR-from-return reject raw price-level inputs (typed return
  semantics + runtime detection).
* 34 — HAR variance-forecast vs volatility-forecast canonicals are distinct and
  honestly named (``ts_har_rv_next_vol_forecast`` takes sqrt; the new
  ``ts_har_rv_next_var_forecast`` keeps the RV scale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Direct module imports register the operators under test without depending on
# the full ``load_all`` closure (the shared tree is concurrently edited by other
# audit fixers; these three modules are the owning, file-disjoint units).
import cleaned_operators.ts_model.dynamic_regression  # noqa: F401
import cleaned_operators.ts_model.state_space  # noqa: F401
import cleaned_operators.ts_model.volatility  # noqa: F401
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.ts_model.state_space import _kalman_level


def _dates(n: int = 120) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def _ret_panel(n: int = 120, seed: int = 0, scale: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, 2)) * scale, index=_dates(n), columns=["A", "B"])


def _price_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A clearly non-stationary raw price level (random walk, positive, grows)."""
    rng = np.random.default_rng(seed)
    walk = 100.0 + np.cumsum(rng.standard_normal(n))
    level = np.abs(walk) + 1.0
    arr = np.column_stack([level, level])
    return pd.DataFrame(arr, index=_dates(n), columns=["A", "B"])


def _op(name: str):
    op = OperatorRegistry.get(name)
    assert op is not None, name
    return op


# ---------------------------------------------------------------------------
# Item 31 — dynamic regression coefficient slot timing
# ---------------------------------------------------------------------------

def test_prior_quantile_coeff_is_causal_unaffected_by_current_shock() -> None:
    """The causal slot must train on <= t-1: a huge current-row shock must not
    move the reported coefficient (a genuine out-of-sample 'current slot')."""
    x = _ret_panel(90, seed=1, scale=1.0)
    rng = np.random.default_rng(2)
    y = 1.5 * x + 0.3 + rng.standard_normal(x.shape) * 0.01
    base = _op("ts_quantile_regression_coeff_prior").calculate(y, x, window=70, q=0.5, min_periods=10)
    y2 = y.copy()
    y2.iloc[-1, 0] = 99.0
    shocked = _op("ts_quantile_regression_coeff_prior").calculate(y2, x, window=70, q=0.5, min_periods=10)
    assert base["A"].iloc[-1] == pytest.approx(shocked["A"].iloc[-1], abs=1e-9)


def test_in_sample_quantile_coeff_does_use_current_observation() -> None:
    """The legacy in-sample coeff genuinely includes the current observation —
    that is exactly why it is documented in-sample and hidden from mining."""
    x = _ret_panel(90, seed=3, scale=1.0)
    rng = np.random.default_rng(4)
    y = 1.5 * x + 0.3 + rng.standard_normal(x.shape) * 0.01
    base = _op("ts_quantile_regression_coeff").calculate(y, x, window=70, q=0.5, min_periods=10)
    y2 = y.copy()
    y2.iloc[-1, 0] = 99.0
    shocked = _op("ts_quantile_regression_coeff").calculate(y2, x, window=70, q=0.5, min_periods=10)
    assert abs(base["A"].iloc[-1] - shocked["A"].iloc[-1]) > 1e-6


def test_quantile_coeff_prior_recovers_slope_on_symmetric_data() -> None:
    """q=0.5 pinball (median) regression on symmetric noise recovers the slope,
    and the causal variant is shape-preserving / registered."""
    x = _ret_panel(90, seed=5, scale=1.0)
    rng = np.random.default_rng(6)
    y = 1.5 * x + 0.3 + rng.standard_normal(x.shape) * 0.01
    out = _op("ts_quantile_regression_coeff_prior").calculate(y, x, window=70, q=0.5, min_periods=10)
    assert out.shape == x.shape
    assert out["A"].iloc[-1] == pytest.approx(1.5, abs=0.4)


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("ts_multi_regression_coeff", "ts_multi_regression_coeff_prior"),
        ("ts_huber_regression_coeff", "ts_huber_regression_coeff_prior"),
        ("ts_ridge_regression_coeff", "ts_ridge_regression_coeff_prior"),
        ("ts_expectile_regression_coeff", "ts_expectile_regression_coeff_prior"),
        ("ts_quantile_regression_coeff", None),
    ],
)
def test_in_sample_coeff_metadata_is_explicitly_in_sample(name: str, replacement) -> None:
    """Legacy coeff operators are documented in-sample at the operator level and
    carry the diagnostic_only tag; the causal *_prior replacement is named."""
    op = _op(name)
    assert "in-sample" in op.metadata.description, name
    assert "diagnostic_only" in op.metadata.tags, name
    if replacement is not None:
        assert replacement in op.metadata.description, name


def test_legacy_coeff_canonicals_stamped_in_sample_and_hidden() -> None:
    """load-time catalog stamps (semantic_certification) mark the in-sample
    coefficient canonicals diagnostic and hidden from default mining."""
    from cleaned_operators.semantic_certification import stamp_compatibility_metadata

    stamp_compatibility_metadata()
    for canon in ("ts_multi_regression_coeff", "ts_huber_regression_coeff",
                  "ts_ridge_regression_coeff", "ts_expectile_regression_coeff",
                  "ts_quantile_regression_coeff"):
        entry = OperatorRegistry._catalog.get(canon, {})
        assert entry.get("in_sample") is True, canon
        assert entry.get("hidden_from_default_mining") is True, canon
    assert OperatorRegistry._catalog["ts_multi_regression_coeff"]["preferred_replacements"] == [
        "ts_multi_regression_coeff_prior"
    ]


# ---------------------------------------------------------------------------
# Item 32 — Kalman scale / gap handling
# ---------------------------------------------------------------------------

def test_kalman_gap_covariance_grows_by_q_per_missing_row() -> None:
    """A K-row gap must accumulate K*Q of extra uncertainty (predict-only), not
    keep the stale pre-gap covariance as if observations were contiguous."""
    arr = np.array([1.0, 2.0, np.nan, np.nan, np.nan, 3.0, 4.0])
    p = _kalman_level(arr, 0.1, 1.0, "uncertainty")
    # p before the gap (t=1) vs each missing row (t=2,3,4)
    assert p[2] == pytest.approx(p[1] + 0.1, rel=1e-12)
    assert p[3] == pytest.approx(p[2] + 0.1, rel=1e-12)
    assert p[4] == pytest.approx(p[3] + 0.1, rel=1e-12)


def test_kalman_gap_level_is_last_known_level() -> None:
    """During a gap the filtered level is the best-known state (propagated), and
    the first observation after the gap is a genuine Kalman update — the scale
    is NOT re-estimated from the gap edges."""
    arr = np.array([1.0, np.nan, np.nan, 2.0])
    mu = _kalman_level(arr, 0.1, 1.0, "level")
    assert mu[1] == pytest.approx(1.0, rel=1e-12)
    assert mu[2] == pytest.approx(1.0, rel=1e-12)
    # t=3: update with p_pred = (r + 2q) + q = 1.3
    p_pred = 1.0 + 3 * 0.1
    k = p_pred / (p_pred + 1.0)
    assert mu[3] == pytest.approx(1.0 + k * (2.0 - 1.0), rel=1e-12)


def test_kalman_innovation_after_gap_uses_grown_scale() -> None:
    """The standardised innovation after a gap must be scaled by the grown
    innovation variance (gap uncertainty included), not the pre-gap scale."""
    arr = np.array([1.0, np.nan, np.nan, np.nan, 2.0])
    z = _kalman_level(arr, 0.1, 1.0, "innovation_z")
    # pre-gap r=1.0; after 3 missing rows p_prev = 1.0 + 3*0.1 = 1.3;
    # p_pred = 1.3 + 0.1 = 1.4 -> innovation var = p_pred + r = 2.4
    manual = (2.0 - 1.0) / np.sqrt(1.4 + 1.0)
    assert z[4] == pytest.approx(manual, rel=1e-9)
    # and it must differ from the contiguous (no-gap) scale denominator.
    contiguous = _kalman_level(np.array([1.0, 2.0]), 0.1, 1.0, "innovation_z")
    assert z[4] != pytest.approx(contiguous[1], rel=1e-9)


def test_kalman_leading_gap_and_all_missing_stay_nan() -> None:
    """Fail closed on unknown scale: before the first finite observation the
    filtered value is NaN, and an all-missing series never fabricates a value."""
    mu = _kalman_level(np.array([np.nan, np.nan, np.nan]), 0.1, 1.0, "level")
    assert np.all(np.isnan(mu))
    # leading NaNs then a real observation: still NaN until the observation.
    mu2 = _kalman_level(np.array([np.nan, np.nan, 5.0]), 0.1, 1.0, "level")
    assert np.isnan(mu2[0]) and np.isnan(mu2[1])
    assert mu2[2] == pytest.approx(5.0, rel=1e-12)


@pytest.mark.parametrize("q,r", [(float("inf"), 1.0), (0.1, float("inf")),
                                 (float("nan"), 1.0), (0.1, float("nan"))])
def test_kalman_nonfinite_noise_scale_fails_closed(q: float, r: float) -> None:
    """A non-finite noise scale is an *unknown* scale and must be rejected up
    front, never silently emitted as a misleading filtered value."""
    with pytest.raises(ValueError):
        _kalman_level(np.array([1.0, 2.0, 3.0]), q, r, "level")
    from cleaned_operators.ts_model.state_space import _kalman_trend_slope, _kalman_beta

    with pytest.raises(ValueError):
        _kalman_trend_slope(np.array([1.0, 2.0, 3.0]), q, q, r)
    with pytest.raises(ValueError):
        _kalman_beta(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), q, r, "beta")


# ---------------------------------------------------------------------------
# Item 33 — GARCH typed input semantics / raw-price rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ts_garch_next_vol_forecast", "ts_garch_vol_surprise", "ts_garch_persistence",
    "ts_garch_standardized_shock", "ts_gjr_garch_vol_forecast", "ts_gjr_leverage",
    "ts_har_from_return_next_vol", "ts_har_from_return_forecast_error_z",
])
def test_return_typed_ops_reject_raw_price_level(name: str) -> None:
    """GARCH / HAR-from-return require a stationary return series; a clearly
    non-stationary raw price level must be rejected at runtime."""
    price = _price_panel(120)
    with pytest.raises(ValueError, match="return"):
        _op(name).calculate(price, window=60)


@pytest.mark.parametrize("name", [
    "ts_garch_next_vol_forecast", "ts_garch_vol_surprise", "ts_garch_persistence",
    "ts_garch_standardized_shock", "ts_gjr_garch_vol_forecast", "ts_gjr_leverage",
])
def test_return_typed_ops_declare_return_input_units(name: str) -> None:
    metadata = _op(name).metadata
    assert metadata.input_units.get("x") == "return", name


def test_garch_accepts_returns_and_produces_finite_output() -> None:
    ret = _ret_panel(180, seed=7)
    out = _op("ts_garch_persistence").calculate(ret, window=120)["A"]
    valid = out.dropna()
    assert valid.size > 0
    assert np.isfinite(valid.to_numpy()).all()
    assert (valid.between(0.0, 1.0)).all()


def test_garch_shock_deterministic_and_finite() -> None:
    ret = _ret_panel(200, seed=8)
    a = _op("ts_garch_standardized_shock").calculate(ret, window=150)["A"]
    b = _op("ts_garch_standardized_shock").calculate(ret, window=150)["A"]
    assert np.allclose(a.fillna(0.0).to_numpy(), b.fillna(0.0).to_numpy(), equal_nan=True)
    assert a.dropna().size > 0
    assert np.isfinite(a.dropna().to_numpy()).all()


def test_har_rv_typed_ops_accept_rv_panel_not_rejected() -> None:
    """The *_rv canonicals take a realized-variance panel (positive, mean
    reverting) — they are NOT return-typed and must not be rejected."""
    rng = np.random.default_rng(9)
    rv = pd.DataFrame(np.abs(rng.standard_normal(90)) * 1e-4 + 1e-4, index=_dates(90), columns=["A"])
    out = _op("ts_har_rv_next_vol_forecast").calculate(rv, window=80)["A"]
    assert out.dropna().size > 0


# ---------------------------------------------------------------------------
# Item 34 — HAR variance-forecast vs volatility-forecast naming
# ---------------------------------------------------------------------------

def test_har_rv_next_forecast_is_compat_alias_to_vol_forecast() -> None:
    """The old name claimed a realized-variance forecast but the kernel returns
    sqrt(RV) — it must now resolve to the honestly-named volatility canonical
    and must NOT be a duplicate canonical (mining never double-searches)."""
    assert OperatorRegistry.resolve_canonical("ts_har_rv_next_forecast") == "ts_har_rv_next_vol_forecast"
    assert "ts_har_rv_next_forecast" not in OperatorRegistry.list_canonical()
    assert _op("ts_har_rv_next_vol_forecast") is not None


def test_har_var_forecast_equals_vol_forecast_squared() -> None:
    """The two forecast canonicals are the SAME HAR model — one returns the RV
    (variance) forecast, the other its square root (volatility)."""
    rng = np.random.default_rng(10)
    rv = pd.DataFrame(np.abs(rng.standard_normal(90)) * 1e-4 + 1e-4, index=_dates(90), columns=["A"])
    vol = _op("ts_har_rv_next_vol_forecast").calculate(rv, window=80)["A"]
    var = _op("ts_har_rv_next_var_forecast").calculate(rv, window=80)["A"]
    mask = vol.notna() & var.notna()
    assert mask.sum() > 0
    assert np.allclose(var[mask].to_numpy(), vol[mask].to_numpy() ** 2, rtol=1e-12)


def test_har_forecast_output_units_honest() -> None:
    assert _op("ts_har_rv_next_vol_forecast").metadata.output_unit == "volatility"
    assert _op("ts_har_rv_next_var_forecast").metadata.output_unit == "variance"
    assert "波动率" in _op("ts_har_rv_next_vol_forecast").metadata.description
    assert "方差" in _op("ts_har_rv_next_var_forecast").metadata.description


def test_legacy_har_rv_forecast_duplicate_describes_volatility() -> None:
    """The kept deprecated duplicate (``ts_har_rv_forecast``) returns the same
    sqrt kernel and is now honestly described as a volatility forecast."""
    desc = _op("ts_har_rv_forecast").metadata.description
    assert "波动率" in desc
    # Same kernel as the vol-forecast canonical: equal output on identical input.
    rng = np.random.default_rng(11)
    rv = pd.DataFrame(np.abs(rng.standard_normal(90)) * 1e-4 + 1e-4, index=_dates(90), columns=["A"])
    a = _op("ts_har_rv_forecast").calculate(rv, window=80)["A"]
    b = _op("ts_har_rv_next_vol_forecast").calculate(rv, window=80)["A"]
    pd.testing.assert_series_equal(a, b, check_dtype=False)


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------

def test_new_and_renamed_canonicals_registered_and_surface_consistent() -> None:
    from cleaned_operators.operator_surface import classify_canonical

    for name in ("ts_quantile_regression_coeff_prior",
                 "ts_har_rv_next_vol_forecast", "ts_har_rv_next_var_forecast"):
        assert classify_canonical(name) in {"daily", "extended", "research"}, name
