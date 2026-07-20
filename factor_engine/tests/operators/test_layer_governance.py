# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.layer_governance import formula_field_names
from cleaned_operators.overhaul.base import PandasFunctionOperator
from cleaned_operators.registry import OperatorRegistry
from factor_recipes.registry import FactorRecipeRegistry
from pit_contract import pit_asof_join, validate_fundamental_events
from research_tools.registry import ResearchToolRegistry


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_fields_are_disjoint_from_operator_canonicals() -> None:
    assert not (formula_field_names() & set(OperatorRegistry.list_canonical()))
    assert "vwap" in formula_field_names()
    assert OperatorRegistry.get("vwap") is None
    assert "vwap" not in OperatorRegistry._aliases
    assert FactorRecipeRegistry.get("vwap") is None
    assert ResearchToolRegistry.get("vwap") is None


def test_old_fundamental_row_operators_are_deleted() -> None:
    for name in ("ttm", "quarter", "yoy", "avg2"):
        assert OperatorRegistry.get(name) is None
        assert name not in OperatorRegistry._aliases


def test_research_tools_are_physically_outside_operator_registry() -> None:
    for name in ("kpss_test", "fft", "pca", "expanding_mean", "fillna", "row_avg"):
        assert OperatorRegistry.get(name) is None
        assert name not in OperatorRegistry.list_canonical()
    assert "expanding_mean" in ResearchToolRegistry.list_canonical()
    assert "fillna" in ResearchToolRegistry.list_canonical()


def test_simple_composites_are_recipes_not_operators() -> None:
    mapping = {
        "MOM": "momentum",
        "ROC": "rate_of_change",
        "BollingerUpper": "bollinger_upper",
        "StochasticK": "stochastic_k",
        "open_gap": "overnight_gap",
        "close_gap": "intraday_return",
        "current_ratio": "current_ratio",
    }
    for canonical, recipe in mapping.items():
        assert OperatorRegistry.get(canonical) is None
        assert FactorRecipeRegistry.get(recipe) is not None


def test_only_fused_technical_composites_remain() -> None:
    for name in ("MACD_line", "MACD_signal", "MACD_hist", "RSI_WILDER", "ATR_WILDER", "ADX"):
        assert OperatorRegistry.get(name) is not None
    for name in ("KAMA", "OBV", "DPO", "TRIX", "ADXR", "ATR", "RSI"):
        assert OperatorRegistry.get(name) is None


def test_cross_sectional_dialect_is_canonicalized() -> None:
    for canonical, legacy in (
        ("cs_count", "c_count"),
        ("cs_mean", "c_mean"),
        ("cs_std", "c_std"),
        ("cs_sum", "c_sum"),
    ):
        assert OperatorRegistry.get(canonical) is not None
        assert OperatorRegistry._aliases.get(legacy) == canonical


def test_strict_cleaning_primitives() -> None:
    x = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, 5.0]})
    result = OperatorRegistry.get("ffill_limit").calculate(x, 2)
    assert result["A"].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    assert np.isnan(result["A"].iloc[3])

    panel = pd.DataFrame({"A": [1.0, np.nan], "B": [3.0, 5.0]})
    mean_filled = OperatorRegistry.get("cs_fill_mean").calculate(panel)
    median_filled = OperatorRegistry.get("cs_fill_median").calculate(panel)
    assert mean_filled.loc[1, "A"] == 5.0
    assert median_filled.loc[1, "A"] == 5.0


def test_period_primitives_require_consecutive_quarters() -> None:
    period = pd.DataFrame({"A": ["2023Q1", "2023Q1", "2023Q2", "2023Q2", "2023Q4", "2023Q4", "2024Q1", "2024Q1"]})
    value = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0, 40.0, 40.0, 50.0, 50.0]})

    change = OperatorRegistry.get("period_change").calculate(value, period, 1, "absolute", True)
    assert change["A"].iloc[2] == 10.0
    assert np.isnan(change["A"].iloc[4])  # Q3 is missing.
    assert change["A"].iloc[6] == 10.0

    ttm = OperatorRegistry.get("ttm_from_quarterly").calculate(value, period, 4, True)
    assert np.isnan(ttm["A"].iloc[-1])


def test_period_average_and_cagr() -> None:
    periods = []
    values = []
    value = 100.0
    for ordinal in range(13):
        year = 2021 + ordinal // 4
        quarter = ordinal % 4 + 1
        periods.append(f"{year}Q{quarter}")
        values.append(value)
        value *= 1.21 ** 0.25
    period = pd.DataFrame({"A": periods})
    data = pd.DataFrame({"A": values})

    average = OperatorRegistry.get("period_average").calculate(data, period, 2, True)
    assert average["A"].iloc[-1] == pytest.approx((values[-2] + values[-1]) / 2)
    cagr = OperatorRegistry.get("period_cagr").calculate(data, period, 12, 4, "strict", True)
    assert cagr["A"].iloc[-1] == pytest.approx(0.21, rel=1e-6)


def test_pit_contract_uses_available_at_and_staleness() -> None:
    events = pd.DataFrame({
        "instrument": ["A", "A"],
        "period_end": ["2024-03-31", "2024-03-31"],
        "available_at": ["2024-05-01T20:00:00Z", "2024-06-01T20:00:00Z"],
        "revision_id": [1, 2],
        "revenue": [100.0, 110.0],
    })
    validate_fundamental_events(events)
    decisions = pd.DataFrame({
        "instrument": ["A", "A", "A"],
        "decision_timestamp": ["2024-04-30T20:00:00Z", "2024-05-15T20:00:00Z", "2024-06-15T20:00:00Z"],
    })
    joined = pit_asof_join(decisions, events, max_age_days=180)
    assert pd.isna(joined.loc[0, "revenue"])
    assert joined.loc[1, "revenue"] == 100.0
    assert joined.loc[2, "revenue"] == 110.0


def test_registry_rejects_implicit_duplicate_after_bootstrap() -> None:
    name = "__layer_governance_test_operator__"
    op = PandasFunctionOperator(name, "internal", ["x"], "test", lambda x, **_: x)
    OperatorRegistry.thaw_for_bootstrap()
    try:
        OperatorRegistry.register(op, canonical=name, source="test")
        with pytest.raises(ValueError, match="duplicate operator registration"):
            OperatorRegistry.register(op, canonical=name, source="test")
        OperatorRegistry.register(
            op,
            canonical=name,
            source="test_v2",
            replace=True,
            replacement_reason="test explicit replacement",
            semantic_version="2.0",
        )
        assert OperatorRegistry._replacement_history[-1]["replacement_reason"] == "test explicit replacement"
    finally:
        if name in OperatorRegistry._operators:
            OperatorRegistry.unregister(name)
        if OperatorRegistry.lifecycle() == "building":
            OperatorRegistry.finalize()
        OperatorRegistry.freeze()
