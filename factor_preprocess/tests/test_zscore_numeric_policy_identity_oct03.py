"""Stable z-score defaults, legacy replay and versioned registry identity."""
import inspect
import warnings

import numpy as np
import pytest

from factor_preprocess.transforms.cross_sectional import cs_zscore
from factor_preprocess.registry.transforms import create_default_registry


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_explicit_legacy_replays_numpy_v1_even_when_numerically_unstable(ddof):
    values = np.array([[1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6],
                       [1., 2., np.inf, np.nan], [4., 4., 4., np.nan]])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=-1, keepdims=True)
        std = np.nanstd(values, axis=-1, ddof=ddof, keepdims=True)
        expected = np.where(std > 0, (values - mean) / std, 3.25)
        expected = np.where(np.isnan(values), np.nan, expected)
        actual = cs_zscore(values, ddof=ddof, constant_value=3.25,
                           numeric_policy="legacy_numpy_v1")
    np.testing.assert_array_equal(actual, expected)


def test_default_policy_is_stable_and_registry_has_distinct_v2_identity():
    assert inspect.signature(cs_zscore).parameters["numeric_policy"].default == "finite_anchor_centered_v2"
    meta = create_default_registry().get("cs_zscore")
    assert meta.version == "2.0.0"
    assert meta.semantic_id == "CROSS_SECTIONAL_ZSCORE:cs"
    assert meta.numeric_policy == "finite_anchor_centered_v2"
    assert meta.fe_equivalent_semantics is None
    assert meta.implementation_hash
    assert meta.numeric_policy_hash


def test_invalid_policy_is_not_silently_accepted_for_empty_input():
    with pytest.raises(ValueError, match="numeric policy"):
        cs_zscore(np.empty((0, 3)), numeric_policy="unknown")


def test_explicit_polars_legacy_remains_available_and_native_v2_is_default():
    pl = pytest.importorskip("polars")
    from factor_preprocess.backends.polars_backend import cs_zscore_polars
    assert inspect.signature(cs_zscore_polars).parameters["numeric_policy"].default == "finite_anchor_centered_v2"
    frame = pl.DataFrame({"date": [1, 1, 1, 1],
                          "value": [1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6]})
    actual = cs_zscore_polars(frame, "value", ddof=0,
                              numeric_policy="legacy_polars_v1")
    reference = frame.with_columns(
        ((pl.col("value") - pl.col("value").mean().over("date")) /
         pl.col("value").std(ddof=0).over("date")).alias("z"))
    np.testing.assert_array_equal(actual.to_numpy(), reference["z"].to_numpy())


def test_stable_default_fixes_large_offset_population_zscore():
    values = np.array([1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6])
    expected = np.array([-3., -1., 1., 3.]) / np.sqrt(5.)
    np.testing.assert_allclose(cs_zscore(values, ddof=0), expected,
                               rtol=2e-15, atol=2e-15)
