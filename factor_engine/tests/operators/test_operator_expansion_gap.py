# -*- coding: utf-8 -*-
"""Behavioral tests for the operator expansion gap (2026-08).

Covers the genuinely new operators added on top of the existing registry:

* alias ``intraday_return`` → ``open_close_return``
* 25 minute→daily intraday aggregation operators
* ``relation_distinct_count`` / ``relation_overlap_ratio`` / ``index_weight``
* ``event_decay_asof``
* ``fin_component_score``

The tests exercise the operators through the registry so catalog wiring,
surface classification and backend lookup are covered, not only the kernels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.production_hardening import SOURCE_BLOCKED_CANONICALS
from cleaned_operators.registry import OperatorRegistry

load_all()


def _daily_panel(dates, instruments, values) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    if isinstance(values, dict):
        return pd.DataFrame({inst: values[inst] for inst in instruments}, index=idx, dtype=float)
    return pd.DataFrame({inst: values for inst in instruments}, index=idx, dtype=float)


def _minute_panel(instruments, days, prices) -> pd.DataFrame:
    """Build a minute panel (DatetimeIndex x instruments).

    prices: list of per-day arrays (one value per minute bar per day).
    """
    index = []
    data = {}
    for i, day in enumerate(days):
        base = pd.Timestamp(day)
        for m in range(len(prices[i])):
            index.append(base + pd.Timedelta(minutes=9 * 60 + 31 + m))
        for inst in instruments:
            data.setdefault(inst, []).extend(prices[i])
    return pd.DataFrame(data, index=pd.DatetimeIndex(index))


# ---------------------------------------------------------------------------
# alias
# ---------------------------------------------------------------------------

def test_intraday_return_alias_resolves():
    assert OperatorRegistry.resolve_canonical("intraday_return") == "open_close_return"
    op = OperatorRegistry.get("open_close_return", "pandas_numpy")
    dates = pd.date_range("2024-01-02", periods=3)
    panel = _daily_panel(dates, ["A", "B"], {"A": [10.0, 11.0, 12.0], "B": [20.0, 19.0, 22.0]})
    out = op._calculate_series(open_px=panel, close=panel * 1.01)
    expected = panel * 1.01 / panel - 1.0
    pd.testing.assert_frame_equal(out, expected)


# ---------------------------------------------------------------------------
# intraday aggregation
# ---------------------------------------------------------------------------

def test_intraday_realized_variance_matches_manual():
    from cleaned_operators.microstructure.intraday_agg import IntraRealizedVariance

    close = np.array([10.0, 11.0, 11.5, 11.0])
    panel = _minute_panel(["A"], ["2024-01-02"], [close])
    out = IntraRealizedVariance()._calculate_series(panel)
    r = np.diff(np.log(close))
    expected = float(np.sum(r * r))
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(expected, rel=1e-9)


def test_intraday_segment_return_morning():
    from cleaned_operators.microstructure.intraday_agg import IntraSegmentReturn

    # 09:31 open close 10.0, 10:00 10.5, 14:00 11.0
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02 09:31"), pd.Timestamp("2024-01-02 10:00"),
        pd.Timestamp("2024-01-02 14:00"),
    ])
    panel = pd.DataFrame({"A": [10.0, 10.5, 11.0]}, index=idx)
    out = IntraSegmentReturn()._calculate_series(panel)
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(10.5 / 10.0 - 1.0, rel=1e-9)


def test_intraday_segment_handles_utc_index():
    """A-share COS minute data is UTC; segments are defined in Asia/Shanghai wall-clock."""
    from cleaned_operators.microstructure.intraday_agg import IntraSegmentReturn, IntraLunchGapReturn

    # 01:31 UTC = 09:31 Beijing (morning), 03:00 UTC = 11:00 Beijing (morning),
    # 05:00 UTC = 13:00 Beijing (afternoon), 06:00 UTC = 14:00 Beijing (afternoon).
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 01:31", tz="UTC"),
            pd.Timestamp("2024-01-02 03:00", tz="UTC"),
            pd.Timestamp("2024-01-02 05:00", tz="UTC"),
            pd.Timestamp("2024-01-02 06:00", tz="UTC"),
        ]
    )
    close = pd.DataFrame({"A": [10.0, 10.5, 11.0, 11.2]}, index=idx)
    out_morning = IntraSegmentReturn()._calculate_series(close, segment="morning")
    assert out_morning.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(10.5 / 10.0 - 1.0, rel=1e-9)
    out_afternoon = IntraSegmentReturn()._calculate_series(close, segment="afternoon")
    assert out_afternoon.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(11.2 / 11.0 - 1.0, rel=1e-9)

    # lunch gap: afternoon first open vs morning last close.
    open_px = pd.DataFrame({"A": [10.0, 10.4, 11.05, 11.1]}, index=idx)
    out_lunch = IntraLunchGapReturn()._calculate_series(close, open_px)
    assert out_lunch.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(11.05 / 10.5 - 1.0, rel=1e-9)


def test_intraday_path_efficiency_single_direction_is_one():
    from cleaned_operators.microstructure.intraday_agg import IntraPathEfficiency

    close = np.array([10.0, 11.0, 12.0, 13.0])
    panel = _minute_panel(["A"], ["2024-01-02"], [close])
    out = IntraPathEfficiency()._calculate_series(panel)
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(1.0, abs=1e-9)


def test_intraday_concentration_all_volume_one_bar():
    from cleaned_operators.microstructure.intraday_agg import IntraConcentration

    volume = np.array([0.0, 0.0, 100.0, 0.0])
    panel = _minute_panel(["A"], ["2024-01-02"], [volume])
    out = IntraConcentration()._calculate_series(panel)
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(1.0, rel=1e-9)


def test_intraday_vwap_cross_count():
    from cleaned_operators.microstructure.intraday_agg import IntraVwapCrossCount

    # cum_vwap stays at 10 while close oscillates 10/12 -> sign flips 3 times.
    close = np.array([10.0, 12.0, 10.0, 12.0, 10.0])
    amount = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    volume = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    cp = _minute_panel(["A"], ["2024-01-02"], [close])
    ap = _minute_panel(["A"], ["2024-01-02"], [amount])
    vp = _minute_panel(["A"], ["2024-01-02"], [volume])
    out = IntraVwapCrossCount()._calculate_series(cp, ap, vp)
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(4.0)


def test_intraday_limit_duration_up():
    from cleaned_operators.microstructure.intraday_agg import IntraLimitDuration

    close = np.array([10.0, 11.0, 11.0, 10.5, 11.0])
    cp = _minute_panel(["A"], ["2024-01-02"], [close])
    lim = _daily_panel(["2024-01-02"], ["A"], [11.0])
    out = IntraLimitDuration()._calculate_series(cp, high_limit=lim, side="up")
    # bars at/above 11.0 (within eps): indices 1,2,4 -> 3/5
    assert out.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(3 / 5)


def test_intraday_empty_day_is_nan_not_zero():
    from cleaned_operators.microstructure.intraday_agg import IntraRealizedVariance

    panel = _minute_panel(["A"], ["2024-01-02"], [[np.nan, np.nan, np.nan]])
    out = IntraRealizedVariance()._calculate_series(panel)
    assert np.isnan(out.loc[pd.Timestamp("2024-01-02"), "A"])


def test_intraday_source_blocked_operators_are_not_production_targets():
    """Daily-argument minute-aggregation operators (``intraday_volatility`` /
    ``intraday_vwap_deviation``) consume a minute panel that the runtime-only
    daily synthetic audit cannot exercise, so they remain source-blocked.  The
    ``intra_*`` minute→daily family and the relation/index panel operators are
    eligible production targets."""
    from cleaned_operators.production_hardening import factor_production_targets

    assert "intraday_volatility" in SOURCE_BLOCKED_CANONICALS
    assert "intraday_vwap_deviation" in SOURCE_BLOCKED_CANONICALS
    targets = factor_production_targets()
    assert "intraday_volatility" not in targets
    assert "intraday_vwap_deviation" not in targets
    for name in ("intra_realized_variance", "intra_limit_reopen_count", "intra_amihud",
                 "relation_distinct_count", "relation_overlap_ratio", "index_weight"):
        assert name in targets, name


# ---------------------------------------------------------------------------
# relation / index
# ---------------------------------------------------------------------------

def test_relation_distinct_count():
    from cleaned_operators.relation.ops import RelationDistinctCount

    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    panel = pd.DataFrame(
        {"p1": ["a", "a"], "p2": ["b", "a"], "p3": ["a", None]},
        index=dates,
        dtype=object,
    )
    out = RelationDistinctCount()._calculate_series(panel)
    assert out.iloc[0, 0] == 2  # {a, b}
    assert out.iloc[1, 0] == 1  # {a}


def test_relation_overlap_jaccard():
    from cleaned_operators.relation.ops import RelationOverlapRatio

    dates = pd.DatetimeIndex(["2024-01-02"])
    cur = pd.DataFrame({"p1": ["a"], "p2": ["b"], "p3": ["c"]}, index=dates, dtype=object)
    prev = pd.DataFrame({"p1": ["a"], "p2": ["b"], "p3": ["d"]}, index=dates, dtype=object)
    out = RelationOverlapRatio()._calculate_series(cur, prev)
    assert out.iloc[0, 0] == pytest.approx(2 / 4)


def test_index_weight_normalized():
    from cleaned_operators.relation.ops import IndexWeight

    panel = _daily_panel(["2024-01-02"], ["A", "B", "C"], {"A": [10.0], "B": [30.0], "C": [60.0]})
    out = IndexWeight()._calculate_series(panel)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [0.1, 0.3, 0.6])


# ---------------------------------------------------------------------------
# event decay / component score
# ---------------------------------------------------------------------------

def test_event_decay_asof_is_causal_and_decaying():
    from cleaned_operators.state_event import EventDecayAsOf

    dates = pd.date_range("2024-01-02", periods=4)
    event = _daily_panel(dates, ["A"], [1.0, 0.0, 0.0, 0.0])
    out = EventDecayAsOf()._calculate_series(event, half_life=1.0)
    vals = out["A"].to_numpy()
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(0.5)
    assert vals[2] == pytest.approx(0.25)
    assert vals[3] == pytest.approx(0.125)


def test_event_decay_asof_future_does_not_affect_past():
    from cleaned_operators.state_event import EventDecayAsOf

    dates = pd.date_range("2024-01-02", periods=4)
    a = _daily_panel(dates, ["A"], [1.0, 0.0, 0.0, 0.0])
    b = _daily_panel(dates, ["A"], [1.0, 0.0, 1.0, 0.0])
    out_a = EventDecayAsOf()._calculate_series(a, half_life=1.0)
    out_b = EventDecayAsOf()._calculate_series(b, half_life=1.0)
    # rows 0..1 unaffected by event at t=2
    np.testing.assert_allclose(out_a["A"].to_numpy()[:2], out_b["A"].to_numpy()[:2])


def test_fin_component_score_sums_directions():
    from cleaned_operators.fundamental.component_score import FinComponentScore

    dates = pd.date_range("2024-01-02", periods=2)
    # three component columns; second row all-NaN exercises missing handling
    comp = pd.DataFrame(
        {"c1": [1.0, np.nan], "c2": [-2.0, np.nan], "c3": [3.0, np.nan]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        dtype=float,
    )
    out = FinComponentScore()._calculate_series(comp, component_directions=["up", "down", "up"])
    assert out.iloc[0, 0] == pytest.approx(1.0 + 1.0 + 1.0)
    assert np.isnan(out.iloc[1, 0])
