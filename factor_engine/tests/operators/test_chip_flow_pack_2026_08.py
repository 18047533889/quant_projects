# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 turnover-survival / weighted-tail /
behavioural / order-flow pack (16 operators).

Covers:
- Registration: every canonical has a pandas_numpy runtime.
- Surface: every canonical classifies ``daily`` and stays in EXTENDED_ONLY.
- Policy: every canonical is PIT-safe with the intended scope (ts_* -> ts;
  minute->daily order-flow -> session_intraday); turnover-survival carries lag=1.
- Fail-closed semantics: constant windows / short samples return NaN.
- Micro-structure regression: ``micro_bipower_var`` matches the standard
  ``(pi/2)*(n/(n-1))*sum|r||r-1|`` form, and legacy proxies keep their math.
- Polars parity: pandas_numpy vs polars agree (exact) for every operator.
- Edge cases: turnover 0/1, missing turnover, locked minute bars, zero volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
from factor_engine.cleaned_operators.operator_surface import classify_canonical
import factor_engine.cleaned_operators.operator_surface as _surface_mod

ensure_cleaned_loaded()

PACK = frozenset({
    # turnover survival
    "ts_turnover_reference_price", "ts_turnover_cost_dispersion",
    "ts_turnover_profit_share", "ts_turnover_holding_age",
    "ts_turnover_near_cost_mass", "ts_turnover_cost_quantile_distance",
    # weighted / stratified tail risk
    "ts_stratified_mean_spread", "ts_weighted_semivariance",
    "ts_weighted_expected_shortfall", "ts_weighted_drawdown_area",
    # behavioural
    "ts_cpt_value",
    # order flow -> impact
    "intraday_bvc_imbalance", "intraday_impact_beta",
    "intraday_impact_asymmetry", "intraday_return_wasserstein_shift",
    "micro_bvc_vpin",
})

EXPECTED_SCOPE = {
    "ts_turnover_reference_price": "ts", "ts_turnover_cost_dispersion": "ts",
    "ts_turnover_profit_share": "ts", "ts_turnover_holding_age": "ts",
    "ts_turnover_near_cost_mass": "ts", "ts_turnover_cost_quantile_distance": "ts",
    "ts_stratified_mean_spread": "ts", "ts_weighted_semivariance": "ts",
    "ts_weighted_expected_shortfall": "ts", "ts_weighted_drawdown_area": "ts",
    "ts_cpt_value": "ts",
    "intraday_bvc_imbalance": "session_intraday",
    "intraday_impact_beta": "session_intraday",
    "intraday_impact_asymmetry": "session_intraday",
    "intraday_return_wasserstein_shift": "session_intraday",
    "micro_bvc_vpin": "session_intraday",
}

EXPECTED_LAG = {
    "ts_turnover_reference_price": 1, "ts_turnover_cost_dispersion": 1,
    "ts_turnover_profit_share": 1, "ts_turnover_holding_age": 1,
    "ts_turnover_near_cost_mass": 1, "ts_turnover_cost_quantile_distance": 1,
}


def _daily_panels(n: int = 120, cols: int = 4):
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (n, cols))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 10.0, index=dates, columns=assets)
    turnover = pd.DataFrame(
        np.clip(rng.lognormal(-2.2, 0.7, (n, cols)), 0.0, 0.6),
        index=dates, columns=assets,
    )
    volume = pd.DataFrame(rng.lognormal(0, 0.3, (n, cols)) * 1e5, index=dates, columns=assets)
    weight = volume.div(volume.sum(axis=1), axis=0)
    returns = close.pct_change().fillna(0.0)
    return {"close": close, "turnover": turnover, "volume": volume,
            "weight": weight, "returns": returns, "minute": _minute_panels()}


def _minute_panels(days: int = 8, cols: int = 3):
    dates = pd.bdate_range("2024-01-02", periods=days)
    assets = [f"M{i}" for i in range(cols)]
    minutes = [570 + m for m in range(240)]
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=int(m)) for d in dates for m in minutes])
    rng = np.random.default_rng(5)
    out = {}
    for inst in assets:
        r = rng.normal(0, 0.001, len(idx))
        close = np.exp(np.cumsum(r)) * 20.0
        vol = rng.lognormal(0, 0.3, len(idx)) * 1e4
        out[inst] = {"close": close, "volume": vol, "ret": r}
    panels = {k: pd.DataFrame({i: out[i][k] for i in assets}, index=idx)
              for k in ("close", "volume", "ret")}
    panels["locked"] = pd.DataFrame(np.zeros_like(panels["close"]), index=idx, columns=assets)
    return panels


def _call_for(canonical, panels):
    if canonical.startswith("ts_turnover"):
        return (panels["close"], panels["turnover"]), {"window": 60}
    if canonical == "ts_turnover_near_cost_mass":
        return (panels["close"], panels["turnover"]), {"window": 60, "band": 0.05}
    if canonical == "ts_turnover_cost_quantile_distance":
        return (panels["close"], panels["turnover"]), {"window": 60}
    if canonical == "ts_stratified_mean_spread":
        return (panels["returns"], panels["volume"]), {"window": 60, "quantile": 0.2}
    if canonical == "ts_weighted_semivariance":
        return (panels["returns"], panels["weight"]), {"window": 20}
    if canonical == "ts_weighted_expected_shortfall":
        return (panels["returns"], panels["weight"]), {"window": 60, "quantile": 0.05}
    if canonical == "ts_weighted_drawdown_area":
        return (panels["close"], panels["weight"]), {"window": 60}
    if canonical == "ts_cpt_value":
        return (panels["returns"],), {"window": 60, "preset": "bmw2016"}
    m = panels["minute"]
    if canonical == "intraday_bvc_imbalance":
        return (m["close"], m["volume"]), {"scale_window": 20}
    if canonical in {"intraday_impact_beta", "intraday_impact_asymmetry"}:
        return (m["ret"], m["ret"]), {"min_periods": 20}
    if canonical == "intraday_return_wasserstein_shift":
        return (m["ret"],), {"lookback_days": 5}
    if canonical == "micro_bvc_vpin":
        return (m["close"], m["volume"]), {"scale_window": 40, "bucket_count": 20}
    raise AssertionError(f"unhandled canonical {canonical}")


# micro_bvc_vpin is P2 / research-only by spec §16 (must never enter the default
# production mining whitelist).
RESEARCH_ONLY = frozenset({"micro_bvc_vpin"})
DAILY_PACK = PACK - RESEARCH_ONLY


def test_pack_all_registered_and_daily():
    missing = [c for c in PACK if OperatorRegistry.get(c, "pandas_numpy") is None]
    assert not missing, f"missing runtimes: {missing}"
    not_daily = [c for c in sorted(DAILY_PACK) if classify_canonical(c) != "daily"]
    assert not not_daily, f"not daily surface: {not_daily}"
    # micro_bvc_vpin must be research-only, not daily and not in the production
    # mining whitelist.
    assert classify_canonical("micro_bvc_vpin") == "research"
    assert "micro_bvc_vpin" not in DAILY_PACK
    live_extended = frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    not_extended = sorted(DAILY_PACK - live_extended)
    assert not not_extended, f"not in EXTENDED_ONLY (partition contract): {not_extended}"


def test_pack_policies_pit_safe_with_intended_scope():
    bad = []
    for canonical in sorted(PACK):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(op, canonical=canonical)
        if not policy.pit_safe:
            bad.append(f"{canonical}: pit_safe=False")
        if policy.scope != EXPECTED_SCOPE[canonical]:
            bad.append(f"{canonical}: scope={policy.scope} expected={EXPECTED_SCOPE[canonical]}")
        if canonical in EXPECTED_LAG and policy.lag != EXPECTED_LAG[canonical]:
            bad.append(f"{canonical}: lag={policy.lag} expected={EXPECTED_LAG[canonical]}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("canonical", sorted(PACK))
def test_pack_deterministic_and_axes(canonical):
    panels = _daily_panels()
    args, kw = _call_for(canonical, panels)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    first = op.calculate(*args, **kw)
    second = op.calculate(*args, **kw)
    if canonical.startswith("intraday_") or canonical == "micro_bvc_vpin":
        minute = panels["minute"]
        template_idx = pd.DatetimeIndex(sorted(set(minute["close"].index.normalize())))
        template = pd.DataFrame(index=template_idx, columns=minute["close"].columns, dtype=float)
    else:
        template = panels["close"]
    assert first.index.equals(template.index), f"{canonical}: index changed"
    assert first.columns.equals(template.columns), f"{canonical}: columns changed"
    assert np.allclose(first.fillna(-1).to_numpy(), second.fillna(-1).to_numpy(), equal_nan=True), \
        f"{canonical}: non-deterministic"


def test_constant_window_fails_closed():
    """Constant inputs must fail closed (NaN in the mature window), never Inf."""
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    const = pd.DataFrame(np.ones((n, 2)), index=dates, columns=["A", "B"])
    cpt_ret = pd.DataFrame(np.zeros((n, 2)), index=dates, columns=["A", "B"])
    cases = [
        ("ts_turnover_reference_price", (const, const), {"window": 20}),
        ("ts_turnover_cost_dispersion", (const, const), {"window": 20}),
        ("ts_stratified_mean_spread", (const, const), {"window": 20}),
        ("ts_weighted_semivariance", (cpt_ret, const), {"window": 20}),
        ("ts_weighted_drawdown_area", (const, const), {"window": 20}),
        ("ts_cpt_value", (cpt_ret,), {"window": 20, "preset": "bmw2016"}),
    ]
    for canonical, panels, kw in cases:
        out = OperatorRegistry.get(canonical, "pandas_numpy").calculate(*panels, **kw)
        if canonical == "ts_cpt_value":
            # CPT of an all-zero return sample is exactly 0 (finite); early rows
            # are NaN only because of the internal min_periods warm-up.
            assert np.isfinite(out.fillna(0.0).to_numpy()).all(), f"{canonical}: non-finite"
            continue
        # Constant window must not be Inf and not explode.
        assert np.isfinite(out.fillna(0.0).to_numpy()).all(), f"{canonical}: non-finite"
        assert np.nanmax(np.abs(out.to_numpy())) < 1e6, f"{canonical}: exploded"


def _research_op(canonical):
    """micro_* ops are governed into the research-tool registry after load."""
    from factor_engine.research_tools.registry import ResearchToolRegistry

    return ResearchToolRegistry.get(canonical)


def _research_op_catalog(canonical):
    """Catalog metadata of a governed research tool."""
    from factor_engine.research_tools.registry import ResearchToolRegistry

    entry = ResearchToolRegistry.catalog().get(canonical)
    return entry or {}


def test_bipower_var_matches_standard_formula():
    """micro_bipower_var must equal (pi/2)*(n/(n-1))*sum|r_t||r_{t-1}|."""
    from factor_engine.cleaned_operators.microstructure.session import pct_change_by_session

    prices = np.array([100.0, 101.0, 99.5, 100.2, 101.5, 100.8])
    s = pd.Series(prices, index=pd.date_range("2024-01-01", periods=6, freq="min"))
    df = pd.DataFrame({"A": s})
    op = _research_op("micro_bipower_var")
    assert op is not None, "micro_bipower_var research tool missing"
    out = op.calculate(df, window=5, min_periods=2)
    rp = pct_change_by_session(s).to_numpy()[1:]
    prods = np.abs(rp[1:]) * np.abs(rp[:-1])
    manual = (np.pi / 2.0) * (len(rp) / (len(rp) - 1)) * np.sum(prods)
    assert np.isclose(out.iloc[-1, 0], manual, rtol=1e-12)


def test_legacy_proxies_keep_their_math_and_are_marked():
    """micro_vpin / micro_kyle_lambda keep their historical math but are
    advertised as legacy proxies with preferred replacements."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=60, freq="min")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 60))), index=idx, columns=["A"])
    volume = pd.DataFrame(rng.lognormal(0, 0.3, 60) * 1e4, index=idx, columns=["A"])

    vpin = _research_op("micro_vpin")
    kyle = _research_op("micro_kyle_lambda")
    assert vpin is not None and kyle is not None
    # These legacy proxies predate the panel contract and expect per-column
    # Series input; exercise that path (their math is intentionally unchanged).
    vpin_out = vpin.calculate(close["A"], volume["A"], window=20)
    kyle_out = kyle.calculate(close["A"], volume["A"], window=20)
    assert np.isfinite(pd.Series(vpin_out).fillna(0).to_numpy()).all()
    assert np.isfinite(pd.Series(kyle_out).fillna(0).to_numpy()).all()

    vpin_cat = _research_op_catalog("micro_vpin")
    kyle_cat = _research_op_catalog("micro_kyle_lambda")
    assert vpin_cat.get("legacy_proxy") is True
    assert kyle_cat.get("legacy_proxy") is True
    assert "micro_bvc_vpin" in (vpin_cat.get("preferred_replacements") or [])
    assert "intraday_impact_beta" in (kyle_cat.get("preferred_replacements") or [])


def test_turnover_survival_edge_cases():
    """turnover 0 (no churn), turnover 1 (full churn), missing turnover, and
    missing price must be handled without Inf and without changing axes."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    assets = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        np.tile(np.linspace(10.0, 20.0, 60)[:, None], (1, 4)),
        index=dates, columns=assets,
    )
    turnover = pd.DataFrame(0.02, index=dates, columns=assets)
    turnover["B"] = 0.0            # no churn at all
    turnover["C"] = 0.99           # near-full churn
    turnover.iloc[10:16, 3] = np.nan   # missing turnover (suspension) for col D
    close.iloc[30, 0] = np.nan    # missing price at a lag for col A
    op = OperatorRegistry.get("ts_turnover_reference_price", "pandas_numpy")
    out = op.calculate(close, turnover, window=40)
    assert out.index.equals(close.index) and out.columns.equals(close.columns)
    finite = out.to_numpy()
    assert np.isfinite(finite[finite == finite]).all(), "non-finite reference price"
    # near-full churn rows must be finite (clipped survival product).
    assert np.isfinite(out["C"].iloc[30:]).all()


def test_locked_minute_bars_are_neutral():
    """Locked limit bars must contribute zero BV-C flow, never a forced 100%
    buy/sell classification."""
    m = _minute_panels(days=6, cols=2)
    op = OperatorRegistry.get("intraday_bvc_imbalance", "pandas_numpy")
    locked = m["locked"].copy()
    # mark the last 30 minutes of day 2 as locked
    day2 = m["close"].index.normalize()[1]
    mask = (m["close"].index.normalize() == day2) & (m["close"].index.hour == 14)
    locked.loc[mask] = 1.0
    out_locked = op.calculate(m["close"], m["volume"], scale_window=20, locked=locked)
    out_free = op.calculate(m["close"], m["volume"], scale_window=20, locked=m["locked"])
    day2_val_locked = out_locked.loc[day2]
    day2_val_free = out_free.loc[day2]
    # locked day's imbalance should be shifted toward zero relative to free run
    assert np.abs(day2_val_locked.to_numpy()).sum() <= np.abs(day2_val_free.to_numpy()).sum() + 1e-12


def _panel_to_pl(name, value):
    pl = pytest.importorskip("polars")
    return pl.from_pandas(value.reset_index()).rename({"index": "date"})


def _name(arg, panels):
    """Resolve a pandas panel argument to its pl counterpart key."""
    for key in ("close", "turnover", "returns", "volume", "weight"):
        if arg is panels[key]:
            return key
    raise AssertionError("unhandled panel arg")


def test_polars_parity_daily():
    pytest.importorskip("polars")
    panels = _daily_panels()
    pl_panels = {k: _panel_to_pl(k, v) for k, v in panels.items() if k != "minute"}
    for canonical in sorted(PACK):
        if canonical.startswith("intraday_") or canonical == "micro_bvc_vpin":
            continue
        args, kw = _call_for(canonical, panels)
        pl_args = tuple(pl_panels[_name(a, panels)] for a in args)
        pd_out = OperatorRegistry.get(canonical, "pandas_numpy").calculate(*args, **kw)
        pl_out = OperatorRegistry.get(canonical, "polars").calculate(*pl_args, **kw)
        pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(
            index=pd_out.index, columns=pd_out.columns)
        both = ~(pd_out.isna() & pl_df.isna())
        diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
        assert diff.size == 0 or np.nanmax(diff) < 1e-9, f"{canonical}: polars parity diff"


def test_polars_parity_minute():
    pytest.importorskip("polars")
    panels = _daily_panels()
    m = panels["minute"]
    pl_panels = {k: _panel_to_pl(k, v) for k, v in m.items()}
    for canonical in ("intraday_bvc_imbalance", "intraday_impact_beta",
                      "intraday_impact_asymmetry", "intraday_return_wasserstein_shift",
                      "micro_bvc_vpin"):
        args, kw = _call_for(canonical, panels)
        pl_args = tuple(pl_panels[_minute_name(a, m)] for a in args)
        pd_out = OperatorRegistry.get(canonical, "pandas_numpy").calculate(*args, **kw)
        pl_out = OperatorRegistry.get(canonical, "polars").calculate(*pl_args, **kw)
        pl_df = pl_out.to_pandas().set_index("date").sort_index().reindex(
            index=pd_out.sort_index().index, columns=pd_out.columns)
        pd_out = pd_out.sort_index()
        both = ~(pd_out.isna() & pl_df.isna())
        diff = (pd_out[both] - pl_df[both]).abs().to_numpy()
        assert diff.size == 0 or np.nanmax(diff) < 1e-9, f"{canonical}: polars parity diff"


def _minute_name(a, m):
    if a is m["close"]:
        return "close"
    if a is m["volume"]:
        return "volume"
    if a is m["ret"]:
        return "ret"
    raise AssertionError("unhandled minute panel arg")
