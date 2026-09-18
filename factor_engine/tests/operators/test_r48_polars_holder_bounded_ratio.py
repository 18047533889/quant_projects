"""Domain parity for Polars holder share-count ratios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.common.polars_holder import (
    HolderFreezeRatioNative,
    HolderLockedShareRatioNative,
    HolderPledgeRatioNative,
)


_RATIO_CLASSES = (
    HolderFreezeRatioNative,
    HolderLockedShareRatioNative,
    HolderPledgeRatioNative,
)


@pytest.mark.parametrize("operator_cls", _RATIO_CLASSES)
def test_holder_ratio_accepts_only_finite_bounded_share_counts(operator_cls):
    dates = pl.Series("date", ["2024-01-02"] * 8)
    stocks = pl.Series("stock_code", [f"C{i}" for i in range(8)])
    numerator = pl.DataFrame(
        {
            "date": dates,
            "stock_code": stocks,
            "value": [25.0, 0.0, 101.0, -1.0, 1.0, np.nan, np.inf, 1.0],
        }
    )
    denominator = pl.DataFrame(
        {
            "date": dates,
            "stock_code": stocks,
            "value": [100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0, np.inf],
        }
    )

    result = operator_cls()._calculate_series(numerator, denominator)

    np.testing.assert_allclose(result["value"][:2].to_numpy(), [0.25, 0.0])
    assert np.isnan(result["value"][2:].to_numpy()).all()
    assert result["date"].to_list() == numerator["date"].to_list()
    assert result["stock_code"].to_list() == numerator["stock_code"].to_list()
    assert result.columns == numerator.columns


@pytest.mark.parametrize("operator_cls", _RATIO_CLASSES)
def test_holder_ratio_rejects_missing_denominator(operator_cls):
    frame = pl.DataFrame({"date": ["2024-01-02"], "value": [1.0]})
    with pytest.raises(ValueError, match="requires total"):
        operator_cls()._calculate_series(frame, None)


@pytest.mark.parametrize(
    "canonical",
    ("holder_freeze_ratio", "holder_locked_share_ratio", "holder_pledge_ratio"),
)
def test_public_registry_winner_enforces_mask_and_preserves_timestamp(canonical):
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()

    timestamps = pd.date_range("2024-01-02", periods=4, freq="D")
    numerator = pl.DataFrame(
        {"timestamp": timestamps, "A": [2.0, 11.0, -1.0, np.nan]}
    )
    denominator = pl.DataFrame(
        {"timestamp": timestamps, "A": [4.0, 10.0, 10.0, 10.0]}
    )
    operator = OperatorRegistry.get(canonical, backend="polars")

    result = operator.calculate(numerator, denominator)
    param = {
        "holder_freeze_ratio": "freeze_shares",
        "holder_locked_share_ratio": "locked_shares",
        "holder_pledge_ratio": "pledge_shares",
    }[canonical]
    by_name = operator.calculate(**{param: numerator, "total_capital": denominator})
    np.testing.assert_allclose(by_name["A"].to_numpy(), result["A"].to_numpy(), equal_nan=True)

    assert result["timestamp"].to_list() == numerator["timestamp"].to_list()
    assert result["A"][0] == pytest.approx(0.5)
    assert np.isnan(result["A"][1:].to_numpy()).all()


@pytest.mark.parametrize("operator_cls", _RATIO_CLASSES)
def test_holder_ratio_rejects_reordered_time_axis(operator_cls):
    timestamps = pd.date_range("2024-01-02", periods=2, freq="D")
    numerator = pl.DataFrame({"timestamp": timestamps, "A": [1.0, 2.0]})
    denominator = pl.DataFrame(
        {"timestamp": timestamps[::-1], "A": [2.0, 4.0]}
    )
    with pytest.raises(ValueError, match="different PanelIdentity"):
        operator_cls()._calculate_series(numerator, denominator)
