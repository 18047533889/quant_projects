"""End-to-end parity for FP missing_indicator through FE pandas_numpy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_operator import get_fe_executor
from factor_preprocess.registry.transforms import get_default_registry


_FE_AVAILABLE = get_fe_executor("is_null") is not None
needs_fe = pytest.mark.skipif(
    not _FE_AVAILABLE,
    reason="FactorEngine unavailable; FP-only environments have no FE route",
)


@needs_fe
def test_registry_routes_missing_indicator_to_fe_and_handles_missing_values():
    registry = get_default_registry()
    metadata = registry.get("missing_indicator")
    assert metadata.implementation_origin == "FE_OPERATOR"
    assert metadata.fe_operator_id == "is_null"
    assert metadata.fit_kind == "stateless"

    # Deliberately interleaved and unsorted, with a non-default index. The FE
    # adapter must pivot for execution then restore the caller's row order.
    values = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01",
                "2024-01-03", "2024-01-04",
            ]),
            "asset_id": ["B", "A", "A", "B", "B", "A"],
            "value": pd.array(
                [pd.NA, None, np.nan, np.inf, -np.inf, 3.0], dtype=object
            ),
        },
        index=[81, 7, 45, 22, 90, 13],
    )
    actual = registry.get_execution("missing_indicator")(values)
    expected = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=values.index,
                         name="value")
    pd.testing.assert_series_equal(actual, expected)


@needs_fe
def test_missing_indicator_fe_nullable_float64_preserves_null_semantics():
    registry = get_default_registry()
    values = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_id": ["A", "A", "A"],
            "value": pd.array([1.0, pd.NA, np.inf], dtype="Float64"),
        },
        index=[20, 5, 30],
    )
    actual = registry.get_execution("missing_indicator")(values)
    expected = pd.Series([0.0, 1.0, 0.0], index=values.index, name="value")
    pd.testing.assert_series_equal(actual, expected)


@needs_fe
@pytest.mark.parametrize("bad_identity", ["null", "duplicate"])
def test_missing_indicator_rejects_ambiguous_long_panel_identity(bad_identity):
    values = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "asset_id": ["A", "B"],
            "value": [np.nan, 1.0],
        }
    )
    if bad_identity == "null":
        values.loc[0, "asset_id"] = None
    else:
        values.loc[1, "asset_id"] = "A"
    with pytest.raises(ValueError, match="null|duplicate"):
        get_default_registry().get_execution("missing_indicator")(values)
