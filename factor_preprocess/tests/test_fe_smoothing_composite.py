import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_smoothing import execute_trailing_sma
from factor_preprocess.transforms.smoothing import trailing_sma


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3])
def test_fe_composite_matches_native_with_duplicate_times_and_indices(min_periods):
    frame = pd.DataFrame(
        {
            "asset_id": pd.Categorical(
                ["a", "b", "a", None, "a", "b", "a"],
                categories=["a", "b", "unused"],
            ),
            "date": [1, 1, 1, 1, 2, 2, 3],
            "value": [10.0, 1.0, 20.0, 999.0, np.nan, 3.0, 30.0],
        },
        index=[7, 7, 4, 4, 7, 1, 1],
    )
    original = frame.copy(deep=True)
    actual = execute_trailing_sma(frame, window=3, min_periods=min_periods)
    expected = trailing_sma(frame, window=3, min_periods=min_periods)
    assert actual.index.equals(frame.index)
    assert actual.name == expected.name
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)


def test_fe_composite_handles_nullable_ids_empty_and_all_missing_assets():
    frame = pd.DataFrame(
        {
            "asset_id": pd.array([1, 1, pd.NA, 1], dtype="Int64"),
            "date": pd.array([1, 2, 1, 3], dtype="Int64"),
            "value": [2.0, 4.0, 99.0, 8.0],
        },
        index=[5, 5, 8, 5],
    )
    actual = execute_trailing_sma(frame, window=2)
    expected = trailing_sma(frame, window=2)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert np.isnan(actual.iloc[2])

    empty = frame.iloc[:0]
    result = execute_trailing_sma(empty, window=2)
    assert result.empty and result.index.equals(empty.index)

    missing = frame.copy()
    missing["asset_id"] = pd.array([pd.NA] * len(missing), dtype="Int64")
    assert execute_trailing_sma(missing, window=2).isna().all()


def test_fe_composite_rejects_null_dates_for_observed_assets():
    frame = pd.DataFrame(
        {"asset_id": ["a", "a"], "date": [1, None], "value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="dates must be non-null"):
        execute_trailing_sma(frame, window=1)


def test_registry_routes_trailing_sma_as_composite_without_operator_id():
    from factor_preprocess.registry.transforms import get_default_registry

    registry = get_default_registry()
    meta = registry.get("trailing_sma")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_operator_id is None
    assert meta.fe_equivalent_semantics == "FE_COMPOSITE:long_smoothing.lagged_mean:v1"
    routed = registry.get_execution("trailing_sma")
    sample = pd.DataFrame(
        {"asset_id": ["a", "a", "a", "a"], "date": [1, 2, 3, 4], "value": [1.0, 2.0, 4.0, 7.0]}
    )
    np.testing.assert_allclose(
        routed(sample, window=3).to_numpy(),
        trailing_sma(sample, window=3).to_numpy(),
        equal_nan=True,
    )
    assert callable(registry.get_execution("trailing_sma", allow_research=True))

def test_composite_resolver_rejects_unknown_recipe():
    from factor_preprocess.adapters.fe_smoothing import get_fe_composite_executor
    from factor_preprocess.errors import GovernanceError

    with pytest.raises(GovernanceError, match="recipe is not registered"):
        get_fe_composite_executor("unknown", "FE_COMPOSITE:long_smoothing.lagged_mean:v1")
    with pytest.raises(GovernanceError, match="recipe is not registered"):
        get_fe_composite_executor("trailing_sma", "unrecognized:v9")


def test_missing_fe_dependency_fails_closed_and_research_uses_native(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("trailing_sma")
    research = registry.get_execution("trailing_sma", allow_research=True)
    sample = pd.DataFrame({"asset_id": ["a", "a"], "date": [1, 2], "value": [2.0, 4.0]})
    np.testing.assert_allclose(
        research(sample, window=3).to_numpy(),
        trailing_sma(sample, window=3).to_numpy(),
        equal_nan=True,
    )


def test_fe_composite_does_not_mutate_infinite_input_values():
    frame = pd.DataFrame(
        {"asset_id": ["a", "a", "a"], "date": [1, 2, 3],
         "value": np.array([1.0, np.inf, 3.0])}
    )
    original = frame.copy(deep=True)
    actual = execute_trailing_sma(frame, window=2, min_periods=1)
    expected = trailing_sma(frame, window=2, min_periods=1)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)
