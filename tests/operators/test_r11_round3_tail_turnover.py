# -*- coding: utf-8 -*-
"""R11 round-3 audit fixes: weighted-tail naming/semantics, stratified ties,
energy-distance U-stat, joint-shift asof tagging, copula null gating, and
turnover-survival fail-closed + diagnostics.

Covers items 113-117 and 139-147 for the files owned by this fixer:
``cleaned_operators/weighted_tail.py``,
``cleaned_operators/polars_chip_tail.py``,
``cleaned_operators/turnover_survival.py``,
``cleaned_operators/distribution_break.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
import factor_engine.cleaned_operators.distribution_break as _db
import factor_engine.cleaned_operators.turnover_survival as _ts
import factor_engine.cleaned_operators.weighted_tail as _wt

ensure_cleaned_loaded()


def _pn(canonical: str):
    return OperatorRegistry.get(canonical, "pandas_numpy")


def _panels(n: int = 80, cols: int = 2, seed: int = 7):
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (n, cols))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 10.0, index=dates, columns=assets)
    turnover = pd.DataFrame(
        np.clip(rng.lognormal(-1.2, 0.6, (n, cols)), 0.0, 0.6),
        index=dates, columns=assets,
    )
    volume = pd.DataFrame(rng.lognormal(0, 0.3, (n, cols)) * 1e5, index=dates, columns=assets)
    weight = volume.div(volume.sum(axis=1), axis=0)
    returns = close.pct_change().fillna(0.0)
    return {"close": close, "turnover": turnover, "volume": volume,
            "weight": weight, "returns": returns}


# ---------------------------------------------------------------------------
# 113 / 114: weighted semivariance naming split + downside-deviation unit
# ---------------------------------------------------------------------------

def test_weighted_semivariance_split_and_alias():
    """ts_weighted_semivariance is the no-sqrt true semivariance; the sqrt
    formula is ts_weighted_downside_deviation; the back-compat alias resolves."""
    panels = _panels()
    x, wgt = panels["returns"], panels["weight"]
    sv = _pn("ts_weighted_semivariance").calculate(x, wgt, window=20)
    dd = _pn("ts_weighted_downside_deviation").calculate(x, wgt, window=20)
    both = np.isfinite(sv.to_numpy()) & np.isfinite(dd.to_numpy())
    assert both.any(), "expected finite downside-deviation rows"
    # downside_deviation^2 == semivariance wherever both are finite.
    np.testing.assert_allclose(
        dd.to_numpy()[both] ** 2, sv.to_numpy()[both], rtol=1e-12, atol=1e-12
    )
    assert np.nanmin(sv.to_numpy()) >= 0.0
    # semivariance is a squared quantity: NOT the sqrt'd value.
    assert np.nanmax(np.abs(sv.to_numpy() - dd.to_numpy())) > 0.0
    # back-compat alias for the historical sqrt behaviour.
    assert OperatorRegistry.resolve_canonical("ts_weighted_semivariance_sqrt") == "ts_weighted_downside_deviation"
    # the new canonical lives on the extended-only surface (a runtime mutator;
    # daily promotion is the coordinator's partition contract).
    import factor_engine.cleaned_operators.operator_surface as _surface_mod
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    assert "ts_weighted_downside_deviation" in frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    assert classify_canonical("ts_weighted_downside_deviation") in ("daily", "extended")


def test_new_canonicals_registered_on_extended_surface():
    import factor_engine.cleaned_operators.operator_surface as _surface_mod

    live = frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    for c in ("ts_weighted_downside_deviation", "ts_turnover_old_mass",
              "ts_turnover_cost_entropy_vol_scaled"):
        assert _pn(c) is not None, f"{c}: missing pandas_numpy runtime"
        assert c in live, f"{c}: not on extended-only surface"


def test_downside_deviation_unit_and_stratified_unit():
    """R3-114: sqrt'd downside deviation unit = same_as:x (not a fixed ratio);
    R3-117: stratified mean spread unit = same_as:target."""
    dd = _pn("ts_weighted_downside_deviation").metadata
    assert "unit:same_as:x" in [str(t) for t in dd.tags]
    assert dd.output_unit == "same_as:x"
    assert not ("unit:ratio" in [str(t) for t in dd.tags])
    sm = _pn("ts_stratified_mean_spread").metadata
    assert "unit:same_as:target" in [str(t) for t in sm.tags]
    assert sm.output_unit == "same_as:target"
    # the true semivariance keeps the square -> unit(x)^2.
    sv = _pn("ts_weighted_semivariance").metadata
    assert "unit:unit(x)^2" in [str(t) for t in sv.tags]


def test_weighted_semivariance_polars_parity():
    pl = pytest.importorskip("polars")
    panels = _panels()
    x, wgt = panels["returns"], panels["weight"]

    def to_pl(df):
        return pl.from_pandas(df.reset_index()).rename({"index": "date"})

    for c in ("ts_weighted_semivariance", "ts_weighted_downside_deviation"):
        pd_out = _pn(c).calculate(x, wgt, window=20)
        pl_out = OperatorRegistry.get(c, "polars").calculate(to_pl(x), to_pl(wgt), window=20)
        pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(
            index=pd_out.index, columns=pd_out.columns)
        both = ~(pd_out.isna() & pl_df.isna())
        diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
        assert diff.size == 0 or np.nanmax(diff) < 1e-9, f"{c}: polars parity"


# ---------------------------------------------------------------------------
# 115: weighted empirical quantile is an ECDF inverse (no interpolation)
# ---------------------------------------------------------------------------

def test_weighted_quantile_ecdf_inverse_no_interpolation():
    v = np.array([10.0, 20.0])
    w = np.array([0.4, 0.6])
    # q=0.5 must be the OBSERVED 20, never a made-up 14.7 (linear interpolation
    # between 10@0.4 and 20@0.6 would fabricate 14.7).
    assert _wt._weighted_quantile(v, w, 0.5) == 20.0
    # endpoint clamp: q <= cdf[0] maps to the observed 10.
    assert _wt._weighted_quantile(v, w, 0.4) == 10.0
    assert _wt._weighted_quantile(v, w, 0.1) == 10.0
    # weighted ES lower q=0.5: 10 fully + 1/6 of the 20-tie -> weighted mean 12.
    assert np.isclose(_wt._weighted_es_tail(v, w, 0.5, "lower", 1), 12.0)


def test_weighted_es_polars_parity():
    pl = pytest.importorskip("polars")
    panels = _panels()
    x, wgt = panels["returns"], panels["weight"]

    def to_pl(df):
        return pl.from_pandas(df.reset_index()).rename({"index": "date"})

    pd_out = _pn("ts_weighted_expected_shortfall").calculate(x, wgt, window=60, quantile=0.2, min_tail_count=2)
    pl_out = OperatorRegistry.get("ts_weighted_expected_shortfall", "polars").calculate(
        to_pl(x), to_pl(wgt), window=60, quantile=0.2, min_tail_count=2)
    pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(index=pd_out.index, columns=pd_out.columns)
    both = ~(pd_out.isna() & pl_df.isna())
    diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
    assert diff.size == 0 or np.nanmax(diff) < 1e-9


# ---------------------------------------------------------------------------
# 116: stratified mean sorter boundary ties are fractional
# ---------------------------------------------------------------------------

def test_stratified_mean_spread_fractional_ties():
    # sorter [1,2,2,2,3], q=0.4 -> k=2: top = 1 + 1/3*(4+3+2) over 2 = 2.0,
    # bottom = 5 + 1/3*(4+3+2) over 2 = 4.0, spread = -2.0.
    xs = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ss = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    top = _wt._stratified_stratum_mean(xs, ss, 2, top=True)
    bot = _wt._stratified_stratum_mean(xs, ss, 2, top=False)
    assert np.isclose(top, 2.0)
    assert np.isclose(bot, 4.0)

    # End-to-end determinism + permutation-invariance of the tied group order.
    dates = pd.bdate_range("2024-01-01", periods=10)
    target = pd.DataFrame({"A": [5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0]}, index=dates)
    sorter = pd.DataFrame({"A": [1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 2.0, 3.0]}, index=dates)
    out = _pn("ts_stratified_mean_spread").calculate(target, sorter, window=5, quantile=0.4, min_periods=5)
    last = float(out.iloc[-1, 0])
    assert np.isclose(last, -2.0)


# ---------------------------------------------------------------------------
# 139 / 140 / 141: distribution-break fixes
# ---------------------------------------------------------------------------

def test_energy_distance_is_u_stat():
    # X=[(0,0),(1,0)], Y=[(10,0),(11,0)].  U-stat excludes the diagonal.
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y = np.array([[10.0, 0.0], [11.0, 0.0]])
    ed = _db._energy_distance(X, Y)
    # cross-mean = 10; U-stat within-X = within-Y = 1 (m*(m-1) denominator);
    # U-stat ed = 2*10 - 1 - 1 = 18.
    # The V-stat would instead use m*m denominators: 2*10 - 0.5 - 0.5 = 19.
    assert np.isclose(ed, 18.0, rtol=1e-12, atol=1e-12)


def test_joint_energy_shift_asof_previous_observation():
    for c in ("ts_joint_energy_shift", "ts_energy_break_score"):
        tags = [str(t) for t in _pn(c).metadata.tags]
        assert "asof_previous_observation" in tags, f"{c}: missing asof tag"


def test_copula_nuisance_params_gated():
    meta = _pn("ts_copula_central_asymmetry").metadata
    specs = meta.param_specs
    assert "window" in specs and "grid" in specs
    for name in ("window", "grid"):
        assert specs[name].searchable is False, f"{name}: must not be searchable"
        assert specs[name].param_role.name == "ESTIMATOR_RESOLUTION"
    # the raw-statistic null-dependence is documented in the docstring.
    assert "null" in meta.description.lower() or "null" in meta.description.lower()
    # smoke run: still computes
    panels = _panels(60)
    out = _pn("ts_copula_central_asymmetry").calculate(
        panels["returns"], panels["volume"], window=40, grid=8)
    assert out.index.equals(panels["returns"].index)


# ---------------------------------------------------------------------------
# 142 / 143: turnover-survival fail-closed
# ---------------------------------------------------------------------------

def test_negative_turnover_fails_closed():
    dates = pd.bdate_range("2024-01-01", periods=50)
    price = pd.DataFrame(np.tile(np.linspace(10.0, 20.0, 50)[:, None], (1, 2)),
                         index=dates, columns=["A", "B"])
    turnover = pd.DataFrame(0.30, index=dates, columns=["A", "B"])
    turnover.iloc[30, 1] = -0.2  # data error on a lag row of col B
    out = _pn("ts_turnover_reference_price").calculate(price, turnover, window=20)
    # col A (clean, high turnover) is finite in the mature window; col B is NaN
    # from t=31 onward because the negative-turnover lag poisons its window.
    assert np.isfinite(out["A"].iloc[40])
    assert np.isnan(out["B"].iloc[40])


def test_missing_price_with_positive_turnover_fails_closed():
    dates = pd.bdate_range("2024-01-01", periods=50)
    price = pd.DataFrame(np.tile(np.linspace(10.0, 20.0, 50)[:, None], (1, 2)),
                         index=dates, columns=["A", "B"])
    turnover = pd.DataFrame(0.30, index=dates, columns=["A", "B"])
    price.iloc[30, 0] = np.nan      # unknown price that day
    turnover.iloc[30, 0] = 0.30     # ... but 30% churn happened that day
    out = _pn("ts_turnover_reference_price").calculate(price, turnover, window=20)
    assert np.isfinite(out["B"].iloc[40])
    # col A must NOT renormalise the remaining mass to 100%: whole distribution NaN.
    assert np.isnan(out["A"].iloc[40])


# ---------------------------------------------------------------------------
# 144 / 145 / 146 / 147: old-mass diagnostic, model-parameter doc, cost
# quantile ECDF, volatility-scaled entropy
# ---------------------------------------------------------------------------

def test_old_mass_diagnostic_and_tightened_threshold():
    dates = pd.bdate_range("2024-01-01", periods=50)
    price = pd.DataFrame(np.tile(np.linspace(10.0, 20.0, 50)[:, None], (1, 2)),
                         index=dates, columns=["A", "B"])
    turnover = pd.DataFrame(0.02, index=dates, columns=["A", "B"])  # low churn
    om = _pn("ts_turnover_old_mass").calculate(price, turnover, window=40)
    # M_old = exp(-0.02*40) = 0.449... exposed as a diagnostic even though the
    # production full-cost gate is tripped.
    assert np.isclose(float(om.iloc[-1, 0]), np.exp(-0.02 * 40), rtol=1e-9)
    ref = _pn("ts_turnover_reference_price").calculate(price, turnover, window=40)
    # tightened threshold (0.10): 44.9% old mass >> 10% -> fail closed.
    assert np.isnan(float(ref.iloc[-1, 0]))


def test_max_old_mass_is_versioned_model_parameter():
    assert _ts._MAX_OLD_MASS == 0.10
    doc = (_ts.__doc__ or "").lower()
    assert "model parameters" in doc and "_max_old_mass" in doc


def test_cost_quantile_is_observed_value():
    # Weighted cost quantile must return an observed chip cost, never an
    # interpolated one (R3-146).  Prices {10,20} with weights {0.4,0.6}: the
    # q=0.5 quantile is the OBSERVED 20 (ECDF inverse), not a made-up 14.7.
    cdf = np.cumsum(np.array([0.4, 0.6]))
    values = np.array([10.0, 20.0])
    q = _ts._weighted_quantile(values, cdf, (0.5,))
    assert q[0] == 20.0
    # q=0.4 clamps to the observed 10.
    assert _ts._weighted_quantile(values, cdf, (0.4,))[0] == 10.0


def test_volatility_scaled_cost_entropy():
    dates = pd.bdate_range("2024-01-01", periods=80)
    # a high-volatility price path: wide log-cost dispersion
    rng = np.random.default_rng(11)
    r = rng.normal(0, 0.02, 80)
    price = pd.DataFrame(np.exp(np.cumsum(r)) * 10.0, index=dates, columns=["A"])
    turnover = pd.DataFrame(0.30, index=dates, columns=["A"])
    ce = _pn("ts_turnover_cost_entropy").calculate(price, turnover, window=40)
    cev = _pn("ts_turnover_cost_entropy_vol_scaled").calculate(price, turnover, window=40)
    assert np.isfinite(float(cev.iloc[-1, 0]))
    assert 0.0 <= float(cev.iloc[-1, 0]) <= 1.0
    # the two measures are distinct (vol-scaled is a different shape summary).
    assert abs(float(cev.iloc[-1, 0]) - float(ce.iloc[-1, 0])) > 1e-9
    # polars parity
    pl = pytest.importorskip("polars")

    def to_pl(df):
        return pl.from_pandas(df.reset_index()).rename({"index": "date"})

    pd_out = _pn("ts_turnover_cost_entropy_vol_scaled").calculate(price, turnover, window=40)
    pl_out = OperatorRegistry.get("ts_turnover_cost_entropy_vol_scaled", "polars").calculate(
        to_pl(price), to_pl(turnover), window=40)
    pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(
        index=pd_out.index, columns=pd_out.columns)
    both = ~(pd_out.isna() & pl_df.isna())
    diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
    assert diff.size == 0 or np.nanmax(diff) < 1e-9


def test_old_mass_polars_parity():
    pl = pytest.importorskip("polars")
    panels = _panels()

    def to_pl(df):
        return pl.from_pandas(df.reset_index()).rename({"index": "date"})

    price, turnover = panels["close"], panels["turnover"]
    pd_out = _pn("ts_turnover_old_mass").calculate(price, turnover, window=40)
    pl_out = OperatorRegistry.get("ts_turnover_old_mass", "polars").calculate(
        to_pl(price), to_pl(turnover), window=40)
    pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(
        index=pd_out.index, columns=pd_out.columns)
    both = ~(pd_out.isna() & pl_df.isna())
    diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
    assert diff.size == 0 or np.nanmax(diff) < 1e-9
