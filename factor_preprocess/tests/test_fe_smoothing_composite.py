import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.fe_smoothing import execute_trailing_median
from factor_preprocess.adapters.fe_smoothing import execute_trailing_sma
from factor_preprocess.adapters.fe_smoothing import execute_rolling_std
from factor_preprocess.transforms.rolling import rolling_mean
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

@pytest.mark.parametrize("min_periods", [0, 1, 2, 3])
def test_rolling_mean_registry_reuses_lagged_mean_composite(min_periods):
    from factor_preprocess.adapters.fe_smoothing import (
        ROLLING_MEAN_RECIPE, execute_trailing_sma, get_fe_composite_executor,
    )
    from factor_preprocess.registry.transforms import get_default_registry

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
    registry = get_default_registry()
    meta = registry.get("rolling_mean")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_operator_id is None
    assert meta.fe_equivalent_semantics == ROLLING_MEAN_RECIPE
    assert ROLLING_MEAN_RECIPE == "FE_COMPOSITE:long_smoothing.lagged_mean:v1"
    assert get_fe_composite_executor("rolling_mean", ROLLING_MEAN_RECIPE) is execute_trailing_sma

    routed = registry.get_execution("rolling_mean")
    actual = routed(frame, window=3, min_periods=min_periods)
    expected = rolling_mean(frame, window=3, min_periods=min_periods)
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_rolling_mean_fe_authority_fails_closed_and_research_fallback_is_explicit(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("rolling_mean")

    frame = pd.DataFrame(
        {"asset_id": ["a", "a", "a"], "date": [1, 2, 3], "value": [2.0, 4.0, 8.0]}
    )
    research = registry.get_execution("rolling_mean", allow_research=True)
    np.testing.assert_allclose(
        research(frame, window=2, min_periods=1).to_numpy(),
        rolling_mean(frame, window=2, min_periods=1).to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("kwargs", [
    {"window": True},
    {"window": 2, "min_periods": True},
    {"window": 2, "min_periods": 1.5},
])
def test_rolling_mean_fe_composite_rejects_invalid_integer_parameters(kwargs):
    from factor_preprocess.registry.transforms import get_default_registry

    frame = pd.DataFrame(
        {"asset_id": ["a", "a"], "date": [1, 2], "value": [2.0, 4.0]}
    )
    routed = get_default_registry().get_execution("rolling_mean")
    with pytest.raises(ValueError):
        routed(frame, **kwargs)


def test_rolling_mean_fe_composite_rejects_null_dates_for_observed_assets():
    from factor_preprocess.registry.transforms import get_default_registry

    frame = pd.DataFrame(
        {"asset_id": ["a", "a"], "date": [1, None], "value": [1.0, 2.0]}
    )
    routed = get_default_registry().get_execution("rolling_mean")
    with pytest.raises(ValueError, match="dates must be non-null"):
        routed(frame, window=1)

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

@pytest.mark.parametrize("min_periods", [0, 1, 2, 3])
def test_lagged_median_matches_native_for_interleaved_rows_and_duplicate_labels(min_periods):
    from factor_preprocess.transforms.smoothing import trailing_median

    frame = pd.DataFrame(
        {
            "asset_id": pd.Categorical(
                ["a", "b", "a", "b", None, "a", "b", "a"],
                categories=["a", "b", "unused"],
            ),
            "date": [1, 1, 1, 2, 1, 2, 3, 3],
            "value": [10.0, 1.0, 20.0, np.inf, 999.0, np.nan, 5.0, -np.inf],
        },
        index=[4, 4, 2, 2, 4, 9, 9, 4],
    )
    original = frame.copy(deep=True)
    actual = execute_trailing_median(frame, window=3, min_periods=min_periods)
    expected = trailing_median(frame, window=3, min_periods=min_periods)
    assert actual.index.equals(frame.index)
    assert actual.name == expected.name
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)
    # Current-row mutations cannot affect the corresponding output.
    changed = frame.copy()
    changed.iloc[5, changed.columns.get_loc("value")] = 123456.0
    changed_actual = execute_trailing_median(changed, window=3, min_periods=min_periods)
    np.testing.assert_allclose(actual.iloc[:6].to_numpy(), changed_actual.iloc[:6].to_numpy(), equal_nan=True)


def test_lagged_median_registry_uses_fe_composite_route():
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.smoothing import trailing_median

    registry = get_default_registry()
    meta = registry.get("trailing_median")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_operator_id is None
    assert meta.fe_equivalent_semantics == "FE_COMPOSITE:long_smoothing.lagged_median:v1"
    routed = registry.get_execution("trailing_median")
    sample = pd.DataFrame(
        {"asset_id": ["a", "b", "a", "b", "a"], "date": [1, 1, 2, 2, 3],
         "value": [1.0, 10.0, 2.0, 20.0, 4.0]}
    )
    np.testing.assert_allclose(
        routed(sample, window=3, min_periods=1).to_numpy(),
        trailing_median(sample, window=3, min_periods=1).to_numpy(),
        equal_nan=True,
    )


def test_lagged_median_rejects_null_dates_for_observed_assets():
    frame = pd.DataFrame({"asset_id": ["a", "a"], "date": [1, None], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="dates must be non-null"):
        execute_trailing_median(frame, window=1)


def test_lagged_median_min_periods_counts_only_finite_values():
    frame = pd.DataFrame(
        {"asset_id": ["a"] * 7, "date": range(7),
         "value": [3.0, np.nan, 5.0, np.inf, 7.0, -np.inf, 9.0]},
        index=[1, 1, 2, 2, 1, 1, 2],
    )
    result = execute_trailing_median(frame, window=4, min_periods=2)
    np.testing.assert_allclose(
        result.to_numpy(), [np.nan, np.nan, np.nan, 4.0, 4.0, 6.0, 6.0], equal_nan=True
    )


def test_lagged_median_registry_stays_fail_closed_without_fe(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        get_default_registry().get_execution("trailing_median")
    assert callable(get_default_registry().get_execution("trailing_median", allow_research=True))


@pytest.mark.parametrize("window", [1, 3, 5])
@pytest.mark.parametrize("min_periods_mode", ["default", "zero", "window"])
def test_lagged_median_parameter_edges_preserve_nullable_assets_and_prefix(window, min_periods_mode):
    from factor_preprocess.transforms.smoothing import trailing_median
    min_periods = {"default": None, "zero": 0, "window": window}[min_periods_mode]
    frame = pd.DataFrame(
        {
            "asset_id": pd.array([1, 2, 1, 1, 2, 1, 1], dtype="Int64"),
            "date": pd.array([1, 1, 2, 3, 2, 4, 5], dtype="Int64"),
            "value": [2.0, 10.0, np.nan, 6.0, 12.0, np.inf, 10.0],
        },
        index=[8, 8, 3, 3, 8, 5, 5],
    )
    original = frame.copy(deep=True)
    actual = execute_trailing_median(frame, window=window, min_periods=min_periods)
    expected = trailing_median(frame, window=window, min_periods=min_periods)
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)

    prefix_len = 5
    prefix_result = execute_trailing_median(
        frame.iloc[:prefix_len], window=window, min_periods=min_periods
    )
    np.testing.assert_allclose(
        actual.iloc[:prefix_len].to_numpy(), prefix_result.to_numpy(), equal_nan=True
    )
    pd.testing.assert_frame_equal(frame, original)


def test_lagged_median_empty_and_all_null_assets_keep_shape_and_index():
    frame = pd.DataFrame(
        {"asset_id": pd.array([1, pd.NA, 1], dtype="Int64"),
         "date": pd.array([1, 1, 2], dtype="Int64"), "value": [2.0, 99.0, 4.0]},
        index=[6, 6, 2],
    )
    empty = frame.iloc[:0]
    result = execute_trailing_median(empty, window=3)
    assert result.empty and result.index.equals(empty.index)

    all_null = frame.copy()
    all_null["asset_id"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    result = execute_trailing_median(all_null, window=3)
    assert result.index.equals(all_null.index)
    assert result.isna().all()


@pytest.mark.parametrize("ddof", [0, 1, 2, -1, 0.5, 1.5, -1.5, True, False])
@pytest.mark.parametrize("min_periods", [0, 1, 2, 3])
def test_lagged_std_matches_native_with_nonfinite_interleaved_rows(ddof, min_periods):
    from factor_preprocess.transforms.rolling import rolling_std

    frame = pd.DataFrame(
        {
            "asset_id": pd.array([1, 2, 1, 2, 1, 1, pd.NA, 1, 1], dtype="Int64"),
            "date": pd.array([1, 1, 2, 2, 2, 3, 3, 4, 5], dtype="Int64"),
            "value": [1.0, 10.0, 3.0, 20.0, np.nan, np.inf, 999.0, 5.0, -np.inf],
        },
        index=[4, 4, 2, 2, 4, 9, 9, 4, 2],
    )
    original = frame.copy(deep=True)
    actual = execute_rolling_std(
        frame, window=3, min_periods=min_periods, ddof=ddof
    )
    expected = rolling_std(
        frame, window=3, min_periods=min_periods, ddof=ddof
    )
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(frame, original)

    prefix = 7
    prefix_result = execute_rolling_std(
        frame.iloc[:prefix], window=3, min_periods=min_periods, ddof=ddof
    )
    np.testing.assert_allclose(
        actual.iloc[:prefix].to_numpy(), prefix_result.to_numpy(), equal_nan=True
    )


def test_lagged_std_registry_route_and_singleton_ddof_boundary():
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.rolling import rolling_std

    registry = get_default_registry()
    meta = registry.get("rolling_std")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fit_kind == "stateless"
    assert meta.fe_equivalent_semantics == "FE_COMPOSITE:long_smoothing.lagged_std:v1"

    frame = pd.DataFrame(
        {"asset_id": ["a"] * 4, "date": [1, 2, 3, 4],
         "value": [4.0, 7.0, 10.0, 13.0]}
    )
    actual = registry.get_execution("rolling_std")(
        frame, window=1, min_periods=0, ddof=1
    )
    expected = rolling_std(frame, window=1, min_periods=0, ddof=1)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert actual.isna().all()
    routed = registry.get_execution("rolling_std")(
        frame, window=2, min_periods=1, ddof=1.5
    )
    routed_expected = rolling_std(frame, window=2, min_periods=1, ddof=1.5)
    np.testing.assert_allclose(
        routed.to_numpy(), routed_expected.to_numpy(), equal_nan=True
    )


def test_lagged_std_matches_pandas_ddof_coercion_and_fails_closed_without_fe(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    frame = pd.DataFrame(
        {"asset_id": ["a", "a"], "date": [1, 2], "value": [1.0, 2.0]}
    )
    from factor_preprocess.transforms.rolling import rolling_std
    np.testing.assert_allclose(
        execute_rolling_std(frame, window=2, ddof=2).to_numpy(),
        rolling_std(frame, window=2, ddof=2).to_numpy(),
        equal_nan=True,
    )
    for ddof, error in ((np.nan, ValueError), (np.inf, OverflowError), ("1", TypeError)):
        with pytest.raises(error):
            execute_rolling_std(frame, window=2, ddof=ddof)
    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("rolling_std")
    research = registry.get_execution("rolling_std", allow_research=True)
    assert callable(research)
    from factor_preprocess.transforms.rolling import rolling_std
    np.testing.assert_allclose(
        research(frame, window=2, ddof=2).to_numpy(),
        rolling_std(frame, window=2, ddof=2).to_numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "values,all_missing",
    [
        ([2.0, 2.0, 2.0, 2.0], False),
        ([np.nan, np.inf, -np.inf], True),
    ],
)
def test_lagged_std_ties_and_all_missing_match_native(values, all_missing):
    from factor_preprocess.transforms.rolling import rolling_std

    frame = pd.DataFrame(
        {"asset_id": ["a"] * len(values), "date": range(len(values)), "value": values}
    )
    actual = execute_rolling_std(frame, window=2, min_periods=1, ddof=1)
    expected = rolling_std(frame, window=2, min_periods=1, ddof=1)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    if all_missing:
        assert actual.isna().all()
    else:
        np.testing.assert_allclose(actual.iloc[2:].to_numpy(), [0.0, 0.0])


def test_lagged_zscore_registry_route_preserves_ieee_and_index():
    from factor_preprocess.adapters.fe_smoothing import execute_rolling_zscore
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.rolling import rolling_zscore

    frame = pd.DataFrame(
        {"asset_id": ["a"] * 8, "date": range(8),
         "value": [2.0, 2.0, 2.0, 2.0, 3.0, np.nan, 1.0, 2.0]},
        index=[7, 7, 3, 3, 8, 8, 4, 4],
    )
    registry = get_default_registry()
    meta = registry.get("rolling_zscore")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_equivalent_semantics == "FE_COMPOSITE:long_smoothing.lagged_zscore:v1"
    expected = rolling_zscore(frame, window=3, min_periods=1, ddof=1)
    actual = registry.get_execution("rolling_zscore")(
        frame, window=3, min_periods=1, ddof=1
    )
    direct = execute_rolling_zscore(frame, window=3, min_periods=1, ddof=1)
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(direct.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert np.isposinf(actual.iloc[4])


def test_lagged_zscore_registry_fails_closed_without_fe(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("rolling_zscore")
    assert callable(registry.get_execution("rolling_zscore", allow_research=True))


def test_ewma_resolver_does_not_require_expression_emitter(monkeypatch):
    import builtins
    import factor_preprocess.adapters.fe_smoothing as adapter

    original_import = builtins.__import__
    def blocked_emitter(name, *args, **kwargs):
        if name == "factor_engine.backend.polars_expr_emitter":
            raise ModuleNotFoundError("emitter unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_emitter)
    assert adapter.get_fe_composite_executor(
        "ewma", adapter.EWMA_RECIPE
    ) is adapter.execute_ewma


def test_ewma_public_route_matches_pandas_for_fractional_halflife():
    from factor_preprocess.transforms import rolling as R
    from factor_preprocess.transforms.rolling import _ewma_reference
    frame = pd.DataFrame({
        "asset_id": ["a"] * 6 + ["b"] * 6, "date": list(range(6)) * 2,
        "value": [1.0, np.nan, 2.0, 4.0, -1.0, 3.0,
                  9.0, 8.0, 7.0, np.nan, 5.0, 4.0],
    })
    actual = R.ewma(frame, halflife=2.5, min_periods=1)
    expected = _ewma_reference(frame, halflife=2.5, min_periods=1)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(),
                               rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ewma_direct_adapter_registry_are_bit_exact_in_production_domain():
    from factor_preprocess.adapters.fe_smoothing import execute_ewma
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms import rolling as R
    from factor_preprocess.transforms.rolling import _ewma_fp_research
    frame = pd.DataFrame({
        "asset_id": ["a", "a", "b", "a", "b", "b"],
        "date": [1, 2, 1, 3, 2, 3],
        "value": [1.0, np.nan, -2.0, 3.5, 4.0, 5.0],
    }).sort_values(["asset_id", "date"], kind="stable")
    direct = R.ewma(frame, halflife=3.7, min_periods=1)
    adapted = execute_ewma(frame, halflife=3.7, min_periods=1)
    routed = get_default_registry().get_execution("ewma")(
        frame, halflife=3.7, min_periods=1
    )
    for actual in (adapted, routed):
        np.testing.assert_array_equal(actual.to_numpy(), direct.to_numpy())
    assert get_default_registry().get("ewma").func is _ewma_fp_research


def test_ewma_public_authority_fails_closed(monkeypatch):
    from factor_preprocess.adapters import fe_composite
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.transforms import rolling as R
    monkeypatch.setattr(fe_composite, "get_fe_composite_executor",
                        lambda name, recipe_identity: None)
    frame = pd.DataFrame({"asset_id": ["a"], "date": [1], "value": [1.0]})
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        R.ewma(frame, halflife=3.7)


def test_ewma_fe_authority_fails_closed_and_research_fallback_is_explicit(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.rolling import _ewma_fp_research
    frame = pd.DataFrame({"asset_id": ["a", "a"], "date": [1, 2], "value": [1.0, 2.0]})
    registry = get_default_registry()
    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda name, recipe_identity: None)
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("ewma")
    research = registry.get_execution("ewma", allow_research=True)
    assert research.executor is _ewma_fp_research
    np.testing.assert_allclose(research(frame, halflife=3.7).to_numpy(),
                               _ewma_fp_research(frame, halflife=3.7).to_numpy(),
                               equal_nan=True)


@pytest.mark.parametrize("halflife", [float.fromhex("0x0.0000000000001p-1022"),
                                      2.0 ** 53, float.fromhex("0x1.fffffffffffffp+1023")])
def test_ewma_fe_extreme_halflife_matches_stable_alpha_recurrence(halflife):
    import math
    from factor_preprocess.transforms import rolling as R
    frame = pd.DataFrame({
        "asset_id": ["a"] * 5, "date": range(5),
        "value": [1.0, -2.0, 5.0, 4.0, -3.0],
    })
    alpha = -math.expm1(-math.log(2.0) / halflife)
    prior = None
    expected = [np.nan]
    for value in frame["value"].to_numpy()[:-1]:
        prior = value if prior is None else alpha * value + (1.0 - alpha) * prior
        expected.append(prior)
    actual = R.ewma(frame, halflife=halflife, min_periods=1)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-14,
                               atol=0.0, equal_nan=True)


def test_ewma_fe_empty_and_min_periods_beyond_input_length():
    from factor_preprocess.transforms import rolling as R
    frame = pd.DataFrame({"asset_id": ["a", "a"], "date": [1, 2], "value": [1.0, 2.0]})
    empty = R.ewma(frame.iloc[:0], halflife=2.0)
    assert empty.empty and empty.index.equals(frame.iloc[:0].index)
    result = R.ewma(frame, halflife=2.0, min_periods=3)
    assert result.isna().all()


def _iir_scalar_oracle(frame, alpha, asset_col="asset_id", value_col="value"):
    out = np.full(len(frame), np.nan, dtype=float)
    raw = frame[value_col].reset_index(drop=True).to_numpy(dtype=float)
    for positions in frame.groupby(asset_col, sort=False).indices.values():
        state = np.nan
        for local_pos, pos in enumerate(positions):
            lagged = raw[positions[local_pos - 1]] if local_pos else np.nan
            if not np.isfinite(lagged):
                state = np.nan
            elif not np.isfinite(state):
                state = lagged
            else:
                state = (1.0 - alpha) * state + alpha * lagged
            out[pos] = state
    return pd.Series(out, index=frame.index)


@pytest.mark.parametrize("alpha", [0.05, 0.37, 0.5, 1.0, np.nextafter(0.0, 1.0)])
def test_fe_iir_lowpass_exact_alpha_matches_scalar_oracle_and_routes(alpha):
    from factor_preprocess.adapters.fe_smoothing import (
        IIR_RECIPE, execute_iir_lowpass, get_fe_composite_executor,
    )
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.smoothing import one_sided_iir_lowpass

    frame = pd.DataFrame({
        "asset_id": pd.Categorical(
            ["a", "b", "a", None, "b", "a", "b", "a", "b", "a"],
            categories=["a", "b", "unused"],
        ),
        "date": [1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        "value": [1e100, 7., 2e100, 999., -3., np.inf, 5., 4., -np.inf, 8.],
    }, index=[4, 4, 7, 7, 4, 1, 7, 1, 4, 1])
    original = frame.copy(deep=True)
    expected = _iir_scalar_oracle(frame, alpha)
    direct = one_sided_iir_lowpass(frame, alpha=alpha)
    registry = get_default_registry()
    routed = registry.get_execution("one_sided_iir_lowpass")
    # Search-domain governance remains narrower than the numerical API.
    if 0.05 <= alpha <= 0.5:
        execute = routed
    else:
        with pytest.raises(ValueError, match="alpha"):
            routed(frame, alpha=alpha)
        execute = execute_iir_lowpass
    actual = execute(frame, alpha=alpha)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=2e-15,
                               atol=0.0, equal_nan=True)
    np.testing.assert_allclose(direct.to_numpy(), expected.to_numpy(), rtol=2e-15,
                               atol=0.0, equal_nan=True)
    assert actual.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, original)
    meta = registry.get("one_sided_iir_lowpass")
    assert meta.implementation_origin == "FE_COMPOSITE"
    assert meta.fe_equivalent_semantics == IIR_RECIPE
    assert get_fe_composite_executor("one_sided_iir_lowpass", IIR_RECIPE) is execute_iir_lowpass
    for stop in range(1, len(frame) + 1):
        prefix = execute(frame.iloc[:stop].copy(), alpha=alpha)
        np.testing.assert_allclose(prefix.to_numpy(), actual.iloc[:stop].to_numpy(),
                                   rtol=2e-15, atol=0.0, equal_nan=True)


def test_iir_native_route_preserves_true_alpha_and_fails_closed(monkeypatch):
    import factor_preprocess.adapters.fe_smoothing as adapter
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry
    from factor_preprocess.transforms.smoothing import one_sided_iir_lowpass

    frame = pd.DataFrame({"asset_id": ["a", "a"], "date": [1, 2], "value": [2., 4.]})
    registry = get_default_registry()
    routed = registry.get_execution("one_sided_iir_lowpass")
    np.testing.assert_allclose(adapter.execute_iir_lowpass(frame, alpha=True),
                               one_sided_iir_lowpass(frame, alpha=True), equal_nan=True)
    with pytest.raises(ValueError, match="alpha"):
        routed(frame, alpha=True)
    monkeypatch.setattr(adapter, "get_fe_composite_executor", lambda *args: None)
    with pytest.raises(GovernanceError, match="FE composite authority unavailable"):
        registry.get_execution("one_sided_iir_lowpass")
    research = registry.get_execution("one_sided_iir_lowpass", allow_research=True)
    np.testing.assert_allclose(research(frame, alpha=0.5),
                               one_sided_iir_lowpass(frame, alpha=0.5), equal_nan=True)
