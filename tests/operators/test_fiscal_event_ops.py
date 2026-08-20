# -*- coding: utf-8 -*-
"""Strict fiscal-event semantics and PIT regression tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


from cleaned_operators.fiscal_event_ops import (
    FiscalEventView,
    pd_fiscal_autocorr,
    pd_fiscal_perpetual_inventory,
    pd_fiscal_reversal_ratio,
    pd_fiscal_sign_consistency,
    pd_fiscal_true_streak,
    pd_fin_seasonal_percentile,
    pd_fin_seasonal_zscore,
    pd_date_diff_days,
    pd_row_sum_skipna,
)
from storage.sources.relation_metrics import relation_entropy, relation_jaccard


def _panel(values, periods):
    index = pd.date_range("2025-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=index, dtype=float), pd.DataFrame({"A": periods}, index=index)


def test_distinct_daily_rows_do_not_add_fiscal_observations():
    x, period = _panel([1, 1, 2, 2, 3, 3], ["2024Q1", "2024Q1", "2024Q2", "2024Q2", "2024Q3", "2024Q3"])
    view = FiscalEventView.from_panel(x, period)
    assert list(view.events[-1][0]) == [2024 * 4, 2024 * 4 + 1, 2024 * 4 + 2]


def test_revision_policy_changes_only_from_visible_revision_row():
    x, period = _panel([10, 10, 12, 15], ["2024Q1", "2024Q1", "2024Q2", "2024Q1"])
    latest = FiscalEventView.from_panel(x, period, revision_policy="latest_available")
    first = FiscalEventView.from_panel(x, period, revision_policy="first_available")
    assert latest.events[2][0][2024 * 4] == 10
    assert latest.events[3][0][2024 * 4] == 15
    assert first.events[3][0][2024 * 4] == 10


def test_future_mutation_does_not_change_prefix():
    x, period = _panel([1, 2, 3, 4, 5], ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"])
    before = pd_fiscal_perpetual_inventory(x, period, warmup_periods=1)
    changed = x.copy(); changed.iloc[-1, 0] = 5000
    after = pd_fiscal_perpetual_inventory(changed, period, warmup_periods=1)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_row_sum_skipna_finite_and_min_count():
    index = pd.date_range("2025-01-01", periods=3)
    a = pd.DataFrame({"A": [1.0, np.nan, np.inf]}, index=index)
    b = pd.DataFrame({"A": [2.0, 3.0, -np.inf]}, index=index)
    result = pd_row_sum_skipna(a, b, 4.0, min_count=2)
    np.testing.assert_allclose(result["A"].to_numpy(), [7.0, 7.0, np.nan], equal_nan=True)


def test_autocorr_and_reversal_are_event_based():
    x, period = _panel([1, 2, 3, 4], ["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    autocorr = pd_fiscal_autocorr(x, period, periods=4, lag=1, min_pairs=3)
    assert autocorr.iloc[-1, 0] == 1.0
    y, _ = _panel([1, -2, 3, -4], ["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    reversal = pd_fiscal_reversal_ratio(y, period, periods=4, min_pairs=3)
    assert reversal.iloc[-1, 0] == 1.0


def test_row_sum_skipna_has_real_sql_lowering():
    from backend.sql_pushdown.emitter import compile_plan_to_sql
    from planner.logical_plan import PlanNode
    plan = PlanNode(
        "row_sum_skipna",
        [PlanNode("column", [], {"name": "a"}), PlanNode("column", [], {"name": "b"})],
        {"min_count": 2},
    )
    compiled = compile_plan_to_sql(plan, dataset="panel", time_column="ts", instrument_column="inst")
    assert "isnan" in compiled.query.lower()
    assert "min_count" not in compiled.query.lower()


def test_sign_consistency_does_not_difference_signal_again():
    x, period = _panel([1, 1, -1, 1], ["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    result = pd_fiscal_sign_consistency(x, period, periods=4, min_periods=3)
    assert result.iloc[-1, 0] == 3 / 4


def test_sign_consistency_uses_full_requested_window():
    x, period = _panel([1, 1, -1, 1], ["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    result = pd_fiscal_sign_consistency(x, period, periods=4, min_periods=3)
    assert result.iloc[-1, 0] == 3 / 4


def test_relation_snapshot_metrics_preserve_all_group_keys():
    left = pd.DataFrame({
        "instrument": ["A", "A", "A", "B"],
        "snapshot_id": [1, 1, 2, 1],
        "entity_id": ["x", "x", "z", "q"],
    })
    right = pd.DataFrame({
        "instrument": ["A", "A", "B"],
        "snapshot_id": [1, 2, 1],
        "entity_id": ["x", "y", "r"],
    })
    result = relation_jaccard(left, right).set_index(["instrument", "snapshot_id"])
    assert result.loc[("A", 1), "relation_jaccard"] == 1.0
    assert result.loc[("A", 2), "relation_jaccard"] == 0.0
    assert result.loc[("B", 1), "relation_jaccard"] == 0.0


def test_relation_entropy_rejects_negative_and_handles_zero_weight():
    with np.testing.assert_raises(ValueError):
        relation_entropy(pd.Series(["x"]), pd.Series([-1.0]), pd.Series([1]))
    result = relation_entropy(pd.Series(["x", "y"]), pd.Series([0.0, 0.0]), pd.Series([1, 1]))
    assert np.isnan(result.loc[0, "relation_entropy"])


def test_relation_snapshot_metrics_boundaries():
    left = pd.DataFrame({"instrument": ["A", "A"], "snapshot_id": [1, 1], "entity_id": ["x", "y"]})
    right = pd.DataFrame({"instrument": ["A", "A"], "snapshot_id": [1, 1], "entity_id": ["y", "z"]})
    result = relation_jaccard(left, right)
    assert result.loc[0, "relation_jaccard"] == 1 / 3
    entropy = relation_entropy(pd.Series(["x", "x", "y"]), pd.Series([1.0, 1.0, 2.0]), pd.Series([1, 1, 1]))
    assert entropy.loc[0, "relation_entropy"] == 1.0
