# -*- coding: utf-8 -*-
"""R11 round-3 file-disjoint fixes: composition CompositionSchema (P0-74..77)
and activity / volume-clock correctness (P0-O #78..#82).

P0-N composition
    74  CompositionFamily is too broad — EPS has a different unit.
    75  MarketCap / Turnover must NOT compose as compositional parts.
    76  Volume / Amount are not composition parts (shares vs currency).
    77  OHLC is not strictly compositional — four price statistics, not parts.
        Composition operators now accept ONLY a CompositionSchema (same units,
        positive, same economic whole, explicit PartId) and gate misclassified
        known parts (eps / market_cap / turnover / volume / amount / raw OHLC)
        with a clear ValueError at the call boundary.

P0-O activity & volume clock
    78  activity_clock._align must raise (SameAxisError) instead of silently
        reindexing a date-shifted / different-stock panel.
    79  new ``ts_activity_clock_lagged_value_prior`` excludes the current bar,
        so a high-activity shock day never produces lagged_value == x_t (a
        manufactured zero momentum); the current-inclusive variant stays.
    80  volume clock: a zero-activity bar must carry the SAME price as the
        previous observable bar, else the day is data-inconsistent -> NaN.
    81  volume clock: the Q=0 grid point is anchored at the session open
        (optional ``open`` panel), never an interpolation artifact.
    82  ``buckets`` is estimator resolution on the fixed grid {8, 16, 32},
        non-searchable, default 16.

NOTE: ``load_all()`` is currently blocked by a PRE-EXISTING canonical/alias
collision in ``cleaned_operators/polars_geometry_math.py``
(``ts_partial_distance_correlation`` is an alias from ``dependence_ext`` but is
still listed in ``_GEOMETRY_MATH_CANONICALS``).  This test file therefore
registers ONLY the owned modules directly — they are self-contained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.closure import SameAxisError
from factor_engine.cleaned_operators.registry import OperatorRegistry

_COMPOSITION_3PART = (
    "composition_clr_component",
    "composition_entropy",
    "composition_normalized_entropy",
)
_COMPOSITION_2PART = (
    "composition_aitchison_distance",
    "composition_ilr_balance",
    "composition_js_divergence",
)

# Known misclassified parts that must be gated (P0-74/75/76/77).
_MISCLASSIFIED = ("eps", "market_cap", "turnover", "volume", "amount", "close")


@pytest.fixture(scope="module")
def _loaded():
    # Register the owned modules directly (see module docstring re: load_all).
    import factor_engine.cleaned_operators.activity_clock  # noqa: F401
    import factor_engine.cleaned_operators.composition  # noqa: F401
    import factor_engine.cleaned_operators.update_clock  # noqa: F401
    import factor_engine.cleaned_operators.volume_clock  # noqa: F401


def _named_part(name: str, n: int = 8) -> pd.DataFrame:
    """A single-column panel whose ``.name`` attribute is the explicit PartId.

    The CompositionSchema gate keys off the part field name (PartId).  A panel
    carries an explicit PartId via its ``.name`` attribute while its physical
    column stays aligned with the other parts (same column, different PartId),
    which is exactly how the operator boundary sees a named composition part.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(np.linspace(1.0, float(n), n)[:, None], index=idx, columns=["v"])
    df.name = name
    return df


def _minute_price(days: int = 2, per_day: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=570 + m) for d in dates for m in range(per_day)]
    )
    return pd.DataFrame(
        np.tile(np.linspace(10.0, 12.0, per_day), days)[:, None],
        index=idx,
        columns=["S0"],
    )


# ---------------------------------------------------------------------------
# P0-74..77 — CompositionSchema gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("canonical", _COMPOSITION_3PART)
@pytest.mark.parametrize("bad", _MISCLASSIFIED)
def test_composition_gates_misclassified_part(_loaded, canonical, bad):
    """Any known non-composable field (EPS/market-cap/turnover/volume/OHLC)
    is rejected with a clear ValueError, not silently folded into a CLR."""
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    with pytest.raises(ValueError, match="not a valid composition part"):
        op.calculate(_named_part("revenue"), _named_part(bad), _named_part("equity"))


@pytest.mark.parametrize("canonical", _COMPOSITION_2PART)
def test_two_composition_gates_misclassified_part(_loaded, canonical):
    """Two-composition operators run the SAME gate before pairing parts."""
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    x = [_named_part("revenue"), _named_part("net_income"), _named_part("eps")]
    y = [_named_part("revenue"), _named_part("net_income"), _named_part("total_assets")]
    with pytest.raises(ValueError, match="not a valid composition part"):
        op.calculate(*x, *y)


def test_composition_ohlc_not_a_part_whole(_loaded):
    """Raw OHLC are four price statistics, not four positive parts of a total."""
    op = OperatorRegistry.get("composition_clr_component", "pandas_numpy")
    with pytest.raises(ValueError, match="OHLC"):
        op.calculate(_named_part("open"), _named_part("high"), _named_part("close"))


def test_composition_eps_different_unit(_loaded):
    """P0-74: EPS has a per-share unit and is NOT part of the money-flow whole."""
    op = OperatorRegistry.get("composition_entropy", "pandas_numpy")
    with pytest.raises(ValueError, match="per-share"):
        op.calculate(
            _named_part("share_of_volume_bucket_1"),
            _named_part("share_of_volume_bucket_2"),
            _named_part("eps"),
        )


def test_composition_market_cap_turnover_not_parts(_loaded):
    """P0-75: market valuation / turnover are whole-firm quantities, not parts."""
    op = OperatorRegistry.get("composition_entropy", "pandas_numpy")
    with pytest.raises(ValueError, match="market valuation"):
        op.calculate(
            _named_part("share_of_volume_bucket_1"),
            _named_part("market_cap"),
            _named_part("share_of_volume_bucket_2"),
        )
    with pytest.raises(ValueError, match="turnover"):
        op.calculate(
            _named_part("share_of_volume_bucket_1"),
            _named_part("turnover"),
            _named_part("share_of_volume_bucket_2"),
        )


def test_composition_volume_amount_not_parts(_loaded):
    """P0-76: raw volume (shares) and amount (currency) are not composable."""
    op = OperatorRegistry.get("composition_entropy", "pandas_numpy")
    with pytest.raises(ValueError, match="share-of-total"):
        op.calculate(_named_part("volume"), _named_part("amount"), _named_part("close"))


def test_composition_schema_financial_statement_whole_is_finite(_loaded):
    """R26-096..099: ``revenue`` / ``net_income`` / ``total_assets`` are NOT
    mutually-exclusive parts of one whole (Revenue vs NetIncome are different
    economic flows; ``Assets = Liabilities + Equity`` makes Assets the WHOLE, so
    listing it as a part double-counts).  The composition gate now REJECTS them
    as known non-composable parts — the old "financial_statement" auto-schema
    was an invalid part-whole and is removed."""
    from factor_engine.cleaned_operators.composition import _same_composition

    with pytest.raises(ValueError, match="not a mutually-exclusive part|double-count"):
        _same_composition(
            [_named_part("revenue"), _named_part("net_income"), _named_part("total_assets")]
        )


def test_composition_schema_unknown_part_fields_pass(_loaded):
    """Unknown (caller-supplied CompositionSchema) part fields pass the gate —
    e.g. volume-by-bucket shares / holder shares, which ARE composable."""
    from factor_engine.cleaned_operators.composition import _same_composition

    parts = [
        _named_part("share_of_volume_bucket_1"),
        _named_part("share_of_volume_bucket_2"),
        _named_part("share_of_volume_bucket_3"),
    ]
    _same_composition(parts)  # no raise
    parts2 = [
        _named_part("holder_share_1"),
        _named_part("holder_share_2"),
        _named_part("holder_share_3"),
    ]
    _same_composition(parts2)  # no raise


def test_composition_family_mismatch_still_rejected(_loaded):
    """P0-69 preserved: explicit composition_id of a different family is a clear
    error (tested at the gate level — composition_id is an internal keyword).
    R26-096..099: revenue/net_income/total_assets are now KNOWN non-composable
    parts, so they are rejected at the gate regardless of composition_id."""
    from factor_engine.cleaned_operators.composition import _same_composition

    parts = [
        _named_part("revenue"),
        _named_part("net_income"),
        _named_part("total_assets"),
    ]
    with pytest.raises(ValueError, match="not a mutually-exclusive part|double-count"):
        _same_composition(parts, composition_id="holder_share")


# ---------------------------------------------------------------------------
# P0-O #78 — activity_clock._align fails closed
# ---------------------------------------------------------------------------
def test_activity_clock_align_raises_on_date_shift(_loaded):
    from factor_engine.cleaned_operators.activity_clock import _align

    a = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
    b = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 2])  # date shifted one slot
    with pytest.raises(SameAxisError):
        _align(a, b)


def test_activity_clock_align_raises_on_stock_column_change(_loaded):
    from factor_engine.cleaned_operators.activity_clock import _align

    a = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
    b = pd.DataFrame({"B": [1.0, 2.0]}, index=[0, 1])  # different stock column
    with pytest.raises(SameAxisError):
        _align(a, b)


def test_activity_clock_operator_rejects_misaligned_panels(_loaded):
    """The public operator route also fails closed (no silent reindex)."""
    idx = list(range(30))
    x = pd.DataFrame({"A": np.arange(30, dtype=float)}, index=idx)
    act_shift = pd.DataFrame({"A": np.ones(30)}, index=[i + 1 for i in idx])
    op = OperatorRegistry.get("ts_activity_clock_lagged_value", "pandas_numpy")
    with pytest.raises(ValueError):
        op.calculate(x, act_shift, budget=1.0, scale_window=10, max_lookback=20)


def test_activity_clock_align_passes_on_exact_match(_loaded):
    from factor_engine.cleaned_operators.activity_clock import _align

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    a = pd.DataFrame(np.ones((5, 2)), index=idx, columns=["A", "B"])
    b = pd.DataFrame(np.ones((5, 2)) * 2.0, index=idx, columns=["A", "B"])
    out = _align(a, b)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# P0-O #79 — ts_activity_clock_lagged_value_prior (excludes current)
# ---------------------------------------------------------------------------
def _shock_panels(n: int = 30, shock_row: int = 20):
    idx = list(range(n))
    x = pd.DataFrame({"A": np.arange(n, dtype=float)}, index=idx)
    act = pd.DataFrame({"A": np.ones(n)}, index=idx)
    act.iloc[shock_row, 0] = 1000.0  # a single high-activity bar
    return x, act


def test_activity_clock_prior_registered_and_aliased(_loaded):
    op = OperatorRegistry.get("ts_activity_clock_lagged_value_prior", "pandas_numpy")
    assert op is not None
    alias_op = OperatorRegistry.get("activity_clock_lagged_value_prior", "pandas_numpy")
    assert alias_op is not None


def test_activity_clock_prior_no_zero_momentum_on_shock_day(_loaded):
    """The shock day must NOT read momentum zero (lagged_value != x_t)."""
    x, act = _shock_panels()
    cur = OperatorRegistry.get("ts_activity_clock_lagged_value", "pandas_numpy").calculate(
        x, act, budget=1.0, scale_window=20, max_lookback=60
    )
    pri = OperatorRegistry.get("ts_activity_clock_lagged_value_prior", "pandas_numpy").calculate(
        x, act, budget=1.0, scale_window=20, max_lookback=60
    )
    # current-inclusive: a single bar consumes the whole budget -> k*=0 -> x_t
    assert cur.iloc[20, 0] == pytest.approx(20.0)
    assert (20.0 - cur.iloc[20, 0]) == pytest.approx(0.0)
    # prior-only: strictly prior -> k*=1 -> x_{t-1} -> momentum = 1, not 0
    assert pri.iloc[20, 0] == pytest.approx(19.0)
    assert (20.0 - pri.iloc[20, 0]) == pytest.approx(1.0)


def test_activity_clock_prior_equals_current_inclusive_false(_loaded):
    x, act = _shock_panels()
    pri = OperatorRegistry.get("ts_activity_clock_lagged_value_prior", "pandas_numpy").calculate(
        x, act, budget=1.0, scale_window=20, max_lookback=60
    )
    cur_false = OperatorRegistry.get(
        "ts_activity_clock_lagged_value", "pandas_numpy"
    ).calculate(x, act, budget=1.0, scale_window=20, max_lookback=60, include_current=False)
    assert np.allclose(pri.to_numpy(), cur_false.to_numpy(), equal_nan=True)


def test_activity_clock_age_keeps_current_inclusive_contract(_loaded):
    """ts_activity_clock_age still defaults to the current-inclusive kernel
    (backward compatible): on the flat reference it reports age 4."""
    idx = list(range(60))
    act = pd.DataFrame({"A": np.ones(60)}, index=idx)
    age = OperatorRegistry.get("ts_activity_clock_age", "pandas_numpy").calculate(
        act, budget=5.0, scale_window=10, max_lookback=60
    )
    assert age["A"].iloc[-1] == 4.0


# ---------------------------------------------------------------------------
# P0-O #80 — volume clock: zero-activity bar price consistency
# ---------------------------------------------------------------------------
def test_volume_clock_zero_activity_price_mismatch_fails_day(_loaded):
    """A zero-activity bar whose price differs from the previous observable
    price would reconnect two different prices when deleted -> fail the day."""
    per_day = 30
    price = _minute_price(days=2, per_day=per_day)
    activity = pd.DataFrame(
        np.ones(2 * per_day)[:, None], index=price.index, columns=["S0"]
    )
    # Day 2 minute 15: zero activity, price INCONSISTENT with minute 14.
    activity.iloc[per_day + 15, 0] = 0.0
    price.iloc[per_day + 15, 0] = 11.5  # minute 14 is ~10.97
    op = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy")
    out = op.calculate(price, activity, 16)
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-01"), "S0"])
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "S0"])


def test_volume_clock_zero_activity_consistent_price_stays_finite(_loaded):
    """A zero-activity bar carrying the SAME price as the previous observable
    is a valid 'no-trade' minute and does not corrupt the day."""
    per_day = 30
    price = _minute_price(days=2, per_day=per_day)
    activity = pd.DataFrame(
        np.ones(2 * per_day)[:, None], index=price.index, columns=["S0"]
    )
    activity.iloc[per_day + 15, 0] = 0.0
    price.iloc[per_day + 15, 0] = price.iloc[per_day + 14, 0]  # consistent
    op = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy")
    out = op.calculate(price, activity, 16)
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-01"), "S0"])
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-02"), "S0"])


def test_volume_clock_roughness_also_fails_zero_price_mismatch(_loaded):
    per_day = 30
    price = _minute_price(days=2, per_day=per_day)
    activity = pd.DataFrame(
        np.ones(2 * per_day)[:, None], index=price.index, columns=["S0"]
    )
    activity.iloc[per_day + 15, 0] = 0.0
    price.iloc[per_day + 15, 0] = 11.5
    op = OperatorRegistry.get("intraday_volume_clock_roughness", "pandas_numpy")
    out = op.calculate(price, activity, 16)
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-01"), "S0"])
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "S0"])


# ---------------------------------------------------------------------------
# P0-O #81 — volume clock: explicit session-open anchor at Q=0
# ---------------------------------------------------------------------------
def test_volume_clock_path_q0_uses_session_open(_loaded):
    from factor_engine.cleaned_operators.volume_clock import _volume_clock_log_path

    price = np.linspace(10.0, 12.0, 30)
    activity = np.ones(30)
    grid, path = _volume_clock_log_path(price, activity, 16, np.full(30, 9.5))
    assert grid[0] == 0.0
    assert np.isclose(path[0], np.log(9.5))


def test_volume_clock_path_q0_fallback_first_bar(_loaded):
    from factor_engine.cleaned_operators.volume_clock import _volume_clock_log_path

    price = np.linspace(10.0, 12.0, 30)
    activity = np.ones(30)
    _, path = _volume_clock_log_path(price, activity, 16, None)
    assert np.isclose(path[0], np.log(price[0]))


def test_volume_clock_operator_accepts_open_panel(_loaded):
    per_day = 30
    price = _minute_price(days=2, per_day=per_day)
    activity = pd.DataFrame(
        np.ones(2 * per_day)[:, None], index=price.index, columns=["S0"]
    )
    open_px = pd.DataFrame(
        np.full((2 * per_day, 1), 9.5), index=price.index, columns=["S0"]
    )
    op = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy")
    out = op.calculate(price, activity, 16, open=open_px)
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-01"), "S0"])


# ---------------------------------------------------------------------------
# P0-O #82 — buckets is estimator resolution, fixed grid {8, 16, 32}
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "canonical", ("intraday_volume_clock_path_efficiency", "intraday_volume_clock_roughness")
)
def test_volume_clock_buckets_estimator_resolution(_loaded, canonical):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    spec = op.metadata.param_specs["buckets"]
    assert spec.choices == (8, 16, 32)
    assert spec.default == 16
    assert spec.searchable is False
    assert spec.param_role is ParamRole.ESTIMATOR_RESOLUTION


@pytest.mark.parametrize("buckets", (8, 16, 32))
def test_volume_clock_buckets_grid_accepted(_loaded, buckets):
    price = _minute_price(days=1, per_day=30)
    activity = pd.DataFrame(np.ones(30)[:, None], index=price.index, columns=["S0"])
    op = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy")
    out = op.calculate(price, activity, buckets)
    assert out.shape == (1, 1)


def test_volume_clock_buckets_off_grid_rejected(_loaded):
    price = _minute_price(days=1, per_day=30)
    activity = pd.DataFrame(np.ones(30)[:, None], index=price.index, columns=["S0"])
    op = OperatorRegistry.get("intraday_volume_clock_path_efficiency", "pandas_numpy")
    with pytest.raises(ValueError, match="not an allowed choice"):
        op.calculate(price, activity, 4)


# ---------------------------------------------------------------------------
# Regression: owned-module canonical surfaces
# ---------------------------------------------------------------------------
def test_new_canonicals_on_extended_surface(_loaded):
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    cls = classify_canonical("ts_activity_clock_lagged_value_prior")
    assert cls in ("extended", "daily")
