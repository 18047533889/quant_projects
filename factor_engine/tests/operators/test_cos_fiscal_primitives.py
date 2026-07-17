# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

@pytest.fixture(scope="module", autouse=True)
def _load():
    load_all()

def _pd(name):
    op = OperatorRegistry.get(name, backend="pandas_numpy")
    assert op is not None
    return op

def test_period_lag_uses_exact_fiscal_ordinal_not_first_seen_order():
    values = pd.DataFrame({"A": [1.0, 3.0, 2.0, 4.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q3", "2023Q2", "2023Q4"]})
    result = _pd("period_lag").calculate(values, periods, 1)
    assert np.isnan(result.iloc[0, 0]) and np.isnan(result.iloc[1, 0])
    assert result.iloc[2, 0] == 1.0 and result.iloc[3, 0] == 3.0

def test_period_lag_revision_policy_is_explicit():
    values = pd.DataFrame({"A": [1.0, 2.0, 2.5, 3.0]})
    periods = pd.DataFrame({"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3"]})
    latest = _pd("period_lag").calculate(values, periods, 1, "latest_available")
    first = _pd("period_lag").calculate(values, periods, 1, "first_available")
    assert latest.iloc[-1, 0] == 2.5 and first.iloc[-1, 0] == 2.0

def test_true_range_first_row_has_explicit_high_low_fallback():
    high = pd.DataFrame({"A": [12.0, 13.0]})
    low = pd.DataFrame({"A": [10.0, 11.0]})
    close = pd.DataFrame({"A": [11.0, 12.0]})
    result = _pd("true_range").calculate(high, low, close)
    assert result.iloc[0, 0] == 2.0 and result.iloc[1, 0] == 2.0

def test_legacy_top_n_aliases_use_explicit_topk_with_compatible_default():
    assert OperatorRegistry._aliases["ts_top_n_avg"] == "ts_topk_mean"
    assert OperatorRegistry._aliases["ts_bottom_n_sum"] == "ts_bottomk_sum"
    values = pd.DataFrame({"A": [1.0, 2.0, 10.0, 4.0]})
    legacy = OperatorRegistry.get("ts_top_n_avg", backend="pandas_numpy").calculate(values, 3)
    assert legacy.iloc[2, 0] == pytest.approx(13.0 / 3.0)
    assert legacy.iloc[3, 0] == pytest.approx(16.0 / 3.0)
    assert _pd("ts_topk_mean").calculate(values, 3, 2).iloc[-1, 0] == 7.0

def test_weighted_cross_section_and_revision_primitives():
    x = pd.DataFrame([[1.0, 3.0, 9.0]], columns=list("ABC"))
    w = pd.DataFrame([[1.0, 3.0, 0.0]], columns=list("ABC"))
    mean = _pd("cs_weighted_mean").calculate(x, w)
    assert mean.iloc[0, 0] == pytest.approx(2.5)
    assert mean.iloc[0, 1] == pytest.approx(2.5)
    assert np.isnan(mean.iloc[0, 2])
    values = pd.DataFrame({"A": [10.0, 12.0, 15.0]})
    periods = pd.DataFrame({"A": ["2024Q1", "2024Q1", "2024Q2"]})
    revisions = pd.DataFrame({"A": [1, 2, 1]})
    delta = _pd("revision_delta").calculate(values, periods, revisions)
    assert delta.iloc[1, 0] == 2.0 and np.isnan(delta.iloc[2, 0])

def test_new_polars_backends_are_real_registrations():
    for name in ("fundamental_staleness", "revision_delta", "period_stability", "cs_weighted_mean", "cs_weighted_zscore", "group_weighted_mean", "ts_topk_mean", "ts_bottomk_std", "safe_div_null"):
        assert "pandas_numpy" in OperatorRegistry.backends_for(name)
        assert "polars" in OperatorRegistry.backends_for(name)
        source = OperatorRegistry.catalog()[name]["backend_meta"]["polars"]["source"]
        assert "bridge" not in source.lower() and source != "daily_panel_polars"
