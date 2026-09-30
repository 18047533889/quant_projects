"""Executable repair mapping audit with independent numeric oracles."""
import datetime
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


def test_value_repair_sma_uses_fe_long_windows_and_matches_fp_oracle():
    from factor_optimizer.adapters.repair_execution import compile_value_repair
    from factor_preprocess.transforms.smoothing import trailing_sma

    rows = [
        ("A", pd.Timestamp("2026-01-01"), 1.0),
        ("B", pd.Timestamp("2026-01-01"), 10.0),
        ("A", pd.Timestamp("2026-01-03"), 3.0),
        ("B", pd.Timestamp("2026-01-03"), 20.0),
        ("A", pd.Timestamp("2026-01-07"), 7.0),
        ("B", pd.Timestamp("2026-01-08"), np.inf),
        ("A", pd.Timestamp("2026-01-09"), np.nan),
        ("B", pd.Timestamp("2026-01-10"), 40.0),
        ("A", pd.Timestamp("2026-01-12"), 9.0),
        ("B", pd.Timestamp("2026-01-13"), 50.0),
        ("B", pd.Timestamp("2026-01-15"), 60.0),
        ("B", pd.Timestamp("2026-01-17"), 70.0),
    ]
    frame = pd.DataFrame(rows, columns=["asset_id", "date", "value"])
    frame["value"] = pd.array(frame["value"], dtype="Float64")
    frame.index = [4, 4, 2, 2, 9, 9, 1, 1, 8, 8, 3, 3]
    plan = compile_value_repair(
        "CAUSAL_SMOOTHING", {"method": "SMA", "natural_time_scale_relative": 0.5},
        natural_time_scale=6.0, training_context_ref="train:fe-sma-audit",
    )
    assert plan.transform == "trailing_sma"
    actual = plan.execute(frame, allow_research=True)

    indexed = frame.assign(_position=np.arange(len(frame)))
    indexed = indexed.sort_values(["asset_id", "date"], kind="stable").reset_index(drop=True)
    reference = trailing_sma(indexed, window=3, min_periods=3)
    oracle = np.full(len(frame), np.nan)
    oracle[indexed["_position"].to_numpy()] = reference.to_numpy()
    np.testing.assert_allclose(actual.to_numpy(), oracle, equal_nan=True)
    assert actual.index.equals(frame.index)
    assert actual.iloc[6] == pytest.approx(11.0 / 3.0)
    assert math.isnan(actual.iloc[10])
    assert actual.iloc[11] == pytest.approx(50.0)

    changed = frame.copy()
    future = changed["date"].gt(pd.Timestamp("2026-01-15"))
    changed.loc[future, "value"] = 1e12
    altered = plan.execute(changed, allow_research=True)
    prefix = frame["date"].le(pd.Timestamp("2026-01-15"))
    np.testing.assert_allclose(actual[prefix], altered[prefix], equal_nan=True)

    reordered = frame.sort_values(["date", "asset_id"], ascending=[True, False], kind="stable")
    reordered_result = plan.execute(reordered, allow_research=True)
    by_key = dict(zip(zip(reordered["date"], reordered["asset_id"]), reordered_result))
    keys = list(zip(frame["date"], frame["asset_id"]))
    np.testing.assert_allclose([by_key[key] for key in keys], actual, equal_nan=True)

    empty = plan.execute(frame.iloc[:0], allow_research=True)
    assert empty.empty and empty.index.equals(frame.iloc[:0].index)

    unsorted = frame.iloc[[2, 0, 1, *range(3, len(frame))]]
    with pytest.raises(ValueError, match="per-asset dates must be monotone"):
        plan.execute(unsorted, allow_research=True)


@pytest.mark.parametrize("date_kind", ["timezone", "python_date", "string", "integer"])
def test_fe_sma_preserves_supported_date_key_types(date_kind):
    from factor_optimizer.adapters.fe_smoothing import execute_lagged_sma
    from factor_preprocess.transforms.smoothing import trailing_sma

    if date_kind == "timezone":
        dates = pd.date_range("2026-01-01", periods=6, tz="Asia/Shanghai")
    elif date_kind == "python_date":
        dates = pd.Series([datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(6)], dtype=object)
    elif date_kind == "string":
        dates = pd.Series([f"d-{i:02d}" for i in range(6)], dtype=object)
    else:
        dates = pd.Series(range(6), dtype="int64")
    frame = pd.DataFrame({
        "asset_id": ["A", "B"] * 6,
        "date": np.repeat(np.asarray(dates), 2),
        "value": np.column_stack((np.arange(6.0), 100.0 + np.arange(6.0))).reshape(-1),
    })
    frame.index = np.tile(np.arange(3), 4)
    actual = execute_lagged_sma(frame, window=2)
    reference = trailing_sma(frame, window=2, min_periods=2)
    np.testing.assert_allclose(actual.to_numpy(), reference.to_numpy(), equal_nan=True)
    assert actual.index.equals(frame.index)
