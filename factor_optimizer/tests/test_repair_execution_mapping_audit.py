"""Executable repair mapping audit with independent numeric oracles."""
import math

import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.preprocessing import IneligibleSmoothingRepair, compile_smoothing_repair


def _plan(family, parameters, *, scale=8.0):
    return compile_smoothing_repair(family, parameters, natural_time_scale=scale,
                                    training_context_ref="train:audit-fixed")


def _panel(a_values, b_values):
    dates = pd.date_range("2026-01-01", periods=len(a_values))
    rows = []
    for date, a, b in zip(dates, a_values, b_values):
        rows.extend((("A", date, a), ("B", date, b)))
    return pd.DataFrame(rows, columns=["asset_id", "date", "value"])


def _asset(result, panel, asset):
    return result[panel["asset_id"].eq(asset)].reset_index(drop=True)


@pytest.mark.parametrize(("method", "expected_a"), [
    ("SMA", [np.nan] * 8 + [8.0, 10.0, 12.0]),
    ("EWMA", [np.nan, 1.0, 1.1659919135906576, 1.4841990830832286,
              1.941988257675288, 2.527774695302193, 3.2309351406511837,
              4.041728025648462, 4.951220292983205, 5.951220292983205,
              7.0342162497785345]),
    ("IIR", [np.nan, 1.0, 1.1659919135906576, 1.4841990830832286,
             1.941988257675288, 2.527774695302193, 3.2309351406511837,
             4.041728025648462, 4.951220292983205, 5.951220292983205,
             7.0342162497785345]),
    ("KAMA", [np.nan] * 9 + [143.0 / 9.0, 1399.0 / 81.0]),
    ("Kalman", [np.nan, 1.0, 1.0294936438747737, 1.115947212729022,
                1.283363021442562, 1.5512516063033104, 1.9340825691755732,
                2.4412041649198484, 3.0771580510897243, 3.8422782783149807,
                4.73345364408469]),
])
def test_each_smoothing_method_matches_hand_derived_lagged_values(method, expected_a):
    """Catches current-bar use, wrong recurrence, and cross-asset state bleed."""
    panel = _panel(list(range(1, 22, 2)), [100.] * 11)
    plan = _plan("CAUSAL_SMOOTHING", {"method": method, "natural_time_scale_relative": 1.0})
    result = plan.execute(panel, allow_research=True)
    np.testing.assert_allclose(_asset(result, panel, "A"), expected_a, rtol=1e-12, equal_nan=True)
    finite_b = _asset(result, panel, "B").dropna()
    assert len(finite_b) > 0
    np.testing.assert_allclose(finite_b, 100.0, rtol=0, atol=2e-14)


@pytest.mark.parametrize("method", ["SMA", "EWMA", "IIR", "KAMA", "Kalman"])
def test_each_smoothing_method_is_prefix_causal_asset_isolated_and_missing_safe(method):
    """Catches future leakage, grouping loss, and silent missing-value filling."""
    panel = _panel([1., 2., 4., 8., 16., np.nan, 32., 64., 128., 256., 512., 1024.], [10.] * 12)
    plan = _plan("CAUSAL_SMOOTHING", {"method": method, "natural_time_scale_relative": 1.0})
    baseline = plan.execute(panel, allow_research=True)
    changed = panel.copy()
    future_a = changed["date"].ge(pd.Timestamp("2026-01-10")) & changed["asset_id"].eq("A")
    changed.loc[future_a, "value"] = 1e9
    altered = plan.execute(changed, allow_research=True)
    cutoff = panel["date"].le(pd.Timestamp("2026-01-10"))
    np.testing.assert_allclose(baseline[cutoff], altered[cutoff], equal_nan=True)
    b_only = panel[panel["asset_id"].eq("B")].reset_index(drop=True)
    np.testing.assert_allclose(_asset(baseline, panel, "B"),
                               plan.execute(b_only, allow_research=True), equal_nan=True)
    missing_lag_output = _asset(baseline, panel, "A").iloc[6]
    if method == "EWMA":
        assert np.isfinite(missing_lag_output)  # pandas EWM ignores missing observations.
    else:
        assert math.isnan(missing_lag_output)


def test_decay_modes_have_independent_numeric_oracles_and_reject_zero_gain():
    panel = _panel([1., 3., 5., 7.], [100.] * 4)
    absolute = _plan("DECAY_REFINEMENT", {"decay": 0.75, "half_life_relative": False}, scale=20)
    np.testing.assert_allclose(_asset(absolute.execute(panel, allow_research=True), panel, "A"),
                               [np.nan, 1.0, 1.5, 2.375], equal_nan=True)
    relative = _plan("DECAY_REFINEMENT", {"decay": 0.5, "half_life_relative": True}, scale=6)
    np.testing.assert_allclose(_asset(relative.execute(panel, allow_research=True), panel, "A"),
                               [np.nan, 1.0, 1.4125989480318006, 2.152677898136905], equal_nan=True)
    with pytest.raises(IneligibleSmoothingRepair, match="alpha"):
        _plan("DECAY_REFINEMENT", {"decay": 0.0, "half_life_relative": False})
