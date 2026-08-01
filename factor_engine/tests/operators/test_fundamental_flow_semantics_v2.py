from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"A": values}, index=index)


def test_quarterly_and_cumulative_ttm_are_not_conflated():
    from cleaned_operators.fundamental.flow_semantics_v2 import (
        fin_quarter_from_cumulative,
        fin_ttm_cumulative,
        fin_ttm_quarterly,
    )

    period_id = _panel(["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    fiscal_quarter = _panel([1, 2, 3, 4])
    quarterly = _panel([10.0, 20.0, 30.0, 40.0])
    cumulative = _panel([10.0, 30.0, 60.0, 100.0])

    converted = fin_quarter_from_cumulative(
        cumulative, period_id, fiscal_quarter
    )
    np.testing.assert_allclose(
        converted["A"].to_numpy(), quarterly["A"].to_numpy(), equal_nan=True
    )
    assert fin_ttm_quarterly(quarterly, period_id, 4).iloc[-1, 0] == 100.0
    assert (
        fin_ttm_cumulative(cumulative, period_id, fiscal_quarter, 4).iloc[-1, 0]
        == 100.0
    )
    # The ambiguous old behavior would have summed cumulative values to 200.
    assert cumulative["A"].sum() == 200.0


def test_cumulative_quarter_conversion_fails_closed_on_missing_predecessor():
    from cleaned_operators.fundamental.flow_semantics_v2 import (
        fin_quarter_from_cumulative,
    )

    cumulative = _panel([10.0, 60.0])
    period_id = _panel(["2023Q1", "2023Q3"])
    fiscal_quarter = _panel([1, 3])
    output = fin_quarter_from_cumulative(cumulative, period_id, fiscal_quarter)
    assert output.iloc[0, 0] == 10.0
    assert np.isnan(output.iloc[1, 0])


def test_same_period_cumulative_revision_changes_results_only_when_visible():
    from cleaned_operators.fundamental.flow_semantics_v2 import (
        fin_quarter_from_cumulative,
    )

    cumulative = _panel([10.0, 10.0, 30.0, 32.0, 32.0])
    period_id = _panel(["2023Q1", "2023Q1", "2023Q2", "2023Q2", "2023Q2"])
    fiscal_quarter = _panel([1, 1, 2, 2, 2])
    output = fin_quarter_from_cumulative(cumulative, period_id, fiscal_quarter)
    np.testing.assert_allclose(
        output["A"].to_numpy(), [10.0, 10.0, 20.0, 22.0, 22.0], equal_nan=True
    )


def test_reviewed_fundamental_extensions_are_registered():
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    expected = {
        "fin_ttm_quarterly",
        "fin_quarter_from_cumulative",
        "fin_ttm_cumulative",
        "fin_surprise",
        "fin_surprise_zscore",
        "fin_expectation_revision",
        "fin_expectation_revision_pct",
        "fin_expectation_revision_speed",
        "fin_expectation_dispersion",
        "fin_actual_expectation_divergence",
        "fin_beat_streak",
        "fin_miss_streak",
    }
    assert expected <= set(OperatorRegistry.list_canonical())
    legacy = OperatorRegistry._catalog["fin_ttm"]
    assert legacy["compatibility_only"] is True
    assert set(legacy["preferred_replacements"]) == {
        "fin_ttm_quarterly",
        "fin_ttm_cumulative",
    }


def test_new_fundamental_recipes_do_not_use_ambiguous_fin_ttm():
    import factor_recipes  # noqa: F401
    from factor_recipes.registry import FactorRecipeRegistry

    for name, row in FactorRecipeRegistry.catalog().items():
        if row["category"] != "fundamental":
            continue
        expression = str(row["expression"])
        assert "fin_ttm(" not in expression, f"{name} uses ambiguous fin_ttm"
