"""Regression oracles for long-panel repair dispatch on sparse, interleaved axes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair


def _plan(family, parameters):
    return compile_value_repair(
        family, parameters, natural_time_scale=8.0,
        training_context_ref="train:repair-method-oracle-oct03",
    )


def _interleaved_sparse_panel():
    # Each asset's dates remain increasing, while the combined long frame is
    # interleaved and its calendar spacing is irregular.
    dates = pd.to_datetime([
        "2026-01-02", "2026-01-05", "2026-01-09", "2026-01-14",
        "2026-01-20", "2026-01-27", "2026-02-05", "2026-02-15",
        "2026-02-28", "2026-03-14", "2026-03-31",
    ])
    a_values = np.arange(1.0, 22.0, 2.0)
    b_values = np.full(len(dates), 100.0)
    rows = []
    for i in range(len(dates)):
        rows.append(("A", dates[i], a_values[i]))
        if i in {0, 2, 5, 8, 10}:
            rows.append(("B", dates[i], b_values[i]))
    frame = pd.DataFrame(rows, columns=["asset_id", "date", "value"])
    frame.index = [i % 4 for i in range(len(frame))]
    return frame


@pytest.mark.parametrize(("method", "expected_a"), [
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
def test_non_sma_smoothing_oracles_survive_sparse_interleaved_rows_and_future_changes(
        method, expected_a):
    frame = _interleaved_sparse_panel()
    plan = _plan("CAUSAL_SMOOTHING", {
        "method": method, "natural_time_scale_relative": 1.0,
    })
    actual = plan.execute(frame, allow_research=True)
    a_rows = frame["asset_id"].eq("A")
    b_rows = frame["asset_id"].eq("B")
    np.testing.assert_allclose(actual[a_rows], expected_a, rtol=1e-12, equal_nan=True)
    if method == "KAMA":
        assert actual[b_rows].isna().all()  # five observations are shorter than ER=8.
    else:
        np.testing.assert_allclose(actual[b_rows].dropna(), 100.0, rtol=0, atol=2e-14)
    assert actual.index.equals(frame.index)

    # Alter only late observations for A. Earlier A results and every B
    # result must remain identical.
    cutoff = frame.loc[a_rows, "date"].iloc[7]
    changed = frame.copy()
    future_a = changed["asset_id"].eq("A") & changed["date"].gt(cutoff)
    changed.loc[future_a, "value"] = -1e9
    altered = plan.execute(changed, allow_research=True)
    prefix_a = a_rows & frame["date"].le(cutoff)
    np.testing.assert_allclose(actual[prefix_a], altered[prefix_a], equal_nan=True)
    np.testing.assert_allclose(actual[b_rows], altered[b_rows], equal_nan=True)


@pytest.mark.parametrize(("family", "sign"), [
    ("U_SHAPE_REPAIR", 1.0),
    ("INVERTED_U_REPAIR", -1.0),
])
def test_rank_shape_resets_at_sparse_date_boundaries_with_hand_calculated_ranks(family, sign):
    d0, d1 = pd.Timestamp("2026-01-02"), pd.Timestamp("2026-03-14")
    frame = pd.DataFrame({
        "asset_id": ["E", "A", "C", "B", "A", "E", "D", "C"],
        "date": [d1, d0, d1, d0, d1, d0, d0, d0],
        "value": [15.0, 10.0, 5.0, 20.0, 5.0, np.nan, 20.0, np.inf],
    }, index=[8, 2, 8, 1, 2, 4, 4, 1])
    plan = _plan(family, {
        "center": 0.25, "power": 2.0, "asymmetry": False,
    })
    actual = plan.execute(frame, allow_research=True)

    # d0 finite values [10,20,20] have percentile ranks [0,.75,.75].
    # d1 finite values [5,5,15] have ranks [.25,.25,1]. Nonfinite values stay missing.
    reference = pd.Series(
        [sign * 0.5625, sign * 0.0625, sign * 0.0, sign * 0.25,
         sign * 0.0, np.nan, sign * 0.25, np.nan],
        index=frame.index, name="value",
    )
    pd.testing.assert_series_equal(actual, reference)
