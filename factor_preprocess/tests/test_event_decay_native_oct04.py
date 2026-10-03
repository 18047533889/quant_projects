import numpy as np
import pandas as pd
import pytest


def test_event_decay_registry_executes_fe_native_recipe_with_fp_semantics():
    """Catch a registry label that does not route event_decay into FE math."""
    from factor_preprocess.registry.transforms import get_default_registry

    registry = get_default_registry()
    metadata = registry.get("event_decay")
    assert metadata.implementation_origin == "FE_COMPOSITE"
    assert metadata.fe_equivalent_semantics == (
        "FE_COMPOSITE:long_ewm.event_decay_native:v1"
    )

    frame = pd.DataFrame({
        "asset_id": ["a"] * 8,
        "date": list(range(8)),
        "value": [2.0, 4.0, np.nan, 8.0, 10.0, np.inf, 3.0, 5.0],
    }, index=[4, 4, 7, 7, 4, 9, 9, 4])
    actual = registry.get_execution("event_decay")(
        frame, halflife=2.0, min_periods=2
    )
    expected = np.array([np.nan, np.nan, 2.6931471805599454, np.nan, np.nan, 8.693147180559945, np.nan, np.nan])
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=0, atol=4e-16,
                               equal_nan=True)


@pytest.mark.parametrize("halflife,min_periods", [(1.0, 1), (2.0, 3)])
def test_registry_event_decay_interleaved_assets_matches_scalar_oracle(halflife, min_periods):
    """Catch ordering, asset leakage, reset, and contiguous-warmup regressions."""
    import runpy
    from factor_preprocess.registry.transforms import get_default_registry

    oracle = runpy.run_path(
        "factor_preprocess/tests/test_event_freshness_numeric_contract_oct04.py"
    )["_event_decay_reference"]
    frame = pd.DataFrame({
        "asset_id": ["a", "b", "a", "b", "a", "b", "a", "b", None],
        "date": [1, 1, 2, 2, 2, 3, 3, 4, 1],
        "value": [2.0, 20.0, 4.0, 40.0, np.nan, 80.0, np.inf, 160.0, 999.0],
    }, index=[7, 7, 3, 3, 7, 1, 1, 3, 7])
    actual = get_default_registry().get_execution("event_decay")(
        frame, halflife=halflife, min_periods=min_periods
    )
    expected = np.full(len(frame), np.nan)
    for asset in ("a", "b"):
        positions = np.flatnonzero(frame.asset_id.eq(asset).to_numpy())
        expected[positions] = oracle(
            frame.value.to_numpy()[positions], halflife, min_periods
        )
    assert actual.index.equals(frame.index)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=0, atol=4e-16,
                               equal_nan=True)


def test_registry_event_decay_is_prefix_invariant_with_future_extreme():
    """Catch global extreme routing or future-dependent output changes."""
    from factor_preprocess.registry.transforms import get_default_registry

    route = get_default_registry().get_execution("event_decay")
    prefix = pd.DataFrame({
        "asset_id": ["a"] * 4, "date": [1, 2, 3, 4],
        "value": [2.0, 4.0, -3.0, 8.0],
    }, index=[1, 1, 2, 2])
    extended = pd.concat([
        prefix,
        pd.DataFrame({"asset_id": ["a"], "date": [5], "value": [1e308]}),
    ])
    before = route(prefix, halflife=2.0, min_periods=1)
    after = route(extended, halflife=2.0, min_periods=1).iloc[:len(prefix)]
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=0,
                               atol=4e-16, equal_nan=True)


def test_registry_event_decay_nullable_values_missing_assets_and_empty_frame():
    """Catch conversion, row-restoration, or null-asset output regressions."""
    from factor_preprocess.registry.transforms import get_default_registry

    route = get_default_registry().get_execution("event_decay")
    frame = pd.DataFrame({
        "asset_id": pd.array([1, 1, pd.NA, 1], dtype="Int64"),
        "date": pd.array([1, 2, 1, 3], dtype="Int64"),
        "value": pd.array([2.0, pd.NA, 99.0, 4.0], dtype="Float64"),
    }, index=[5, 5, 8, 5])
    result = route(frame, halflife=2.0, min_periods=1)
    assert result.index.equals(frame.index)
    assert result.iloc[2] != result.iloc[2]
    assert result.iloc[3] != result.iloc[3]
    empty = route(frame.iloc[:0], halflife=2.0)
    assert empty.empty and empty.index.equals(frame.iloc[:0].index)


def test_registry_event_decay_identity_binds_selected_kernel(monkeypatch):
    """Catch identity that stays unchanged after the selected numeric kernel changes."""
    from factor_engine.backend import native_event_decay
    from factor_preprocess.registry.transforms import get_default_registry

    registry = get_default_registry()
    original = registry.execution_identity("event_decay")
    assert original["adapter"] == "factor_preprocess.adapters.fe_smoothing.execute_event_decay"
    assert any(
        label == "factor_engine.backend.native_event_decay.collect_event_decay"
        for label, _ in original["fe_callable_digests"]
    )

    selected = native_event_decay.collect_event_decay
    def changed_kernel(*args, **kwargs):
        return selected(*args, **kwargs)
    monkeypatch.setattr(native_event_decay, "collect_event_decay", changed_kernel)
    changed = registry.execution_identity("event_decay")
    assert changed["digest"] != original["digest"]


def test_registry_event_decay_fails_closed_without_fe_authority(monkeypatch):
    """Catch an implicit FP math fallback when the registered FE recipe is absent."""
    from factor_preprocess.adapters import fe_smoothing
    from factor_preprocess.errors import GovernanceError
    from factor_preprocess.registry.transforms import get_default_registry

    monkeypatch.setattr(fe_smoothing, "get_fe_composite_executor",
                        lambda name, identity: None)
    registry = get_default_registry()
    with pytest.raises(GovernanceError, match="no implicit FP-native fallback"):
        registry.get_execution("event_decay")
    research = registry.get_execution("event_decay", allow_research=True)
    frame = pd.DataFrame({
        "asset_id": ["a", "a", "a"], "date": [1, 2, 3], "value": [2., 4., 8.],
    })
    np.testing.assert_allclose(
        research(frame, halflife=2.0).to_numpy(),
        [np.nan, 2.0, 2.0 + 2.0 * np.log(2.0) / 2.0],
        rtol=0, atol=4e-16, equal_nan=True,
    )


def test_registry_event_decay_rejects_nonmonotone_asset_dates():
    """Catch accidental sorting that changes duplicate-tie input semantics."""
    from factor_preprocess.registry.transforms import get_default_registry

    frame = pd.DataFrame({
        "asset_id": ["a", "a"], "date": [2, 1], "value": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="monotone increasing"):
        get_default_registry().get_execution("event_decay")(
            frame, halflife=2.0
        )


def test_registry_event_decay_preserves_fp_parameter_coercion_and_rejection():
    """Keep the registry domain while checking the FE wrapper's coercive API."""
    from factor_preprocess.registry.transforms import get_default_registry

    route = get_default_registry().get_execution("event_decay")
    assert get_default_registry().get("event_decay").parameter_domain["halflife"] == (1.0, 5.0)
    frame = pd.DataFrame({
        "asset_id": ["a", "a", "a"], "date": [1, 2, 3], "value": [2.0, 4.0, 8.0],
    })
    from factor_preprocess.adapters.fe_smoothing import execute_event_decay
    expected = route(frame, halflife=2.0).to_numpy()
    np.testing.assert_allclose(
        execute_event_decay(frame, halflife="2").to_numpy(), expected,
        rtol=0, atol=4e-16, equal_nan=True,
    )
    for invalid in ("2", 0.5, 6.0):
        with pytest.raises(ValueError):
            route(frame, halflife=invalid)
    with pytest.raises(ValueError):
        route(frame, halflife=True)
    with pytest.raises(ValueError):
        route(frame, halflife=2.0, min_periods=True)


