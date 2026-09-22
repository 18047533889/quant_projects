import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.smoothing import trailing_sma


def _scalar_oracle(values, window, min_periods):
    result = np.full(len(values), np.nan, dtype=float)
    for positions in values.groupby("asset_id", sort=False, observed=True).indices.values():
        positions = list(positions)
        result[positions] = (
            values.iloc[positions]["value"]
            .shift(1)
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .to_numpy()
        )
    return pd.Series(result, index=values.index)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("window,min_periods", [(1, 1), (3, 1), (3, 2), (3, 3)])
def test_trailing_sma_matches_scalar_oracle_on_interleaved_edge_cases(
    dtype, window, min_periods
):
    values = pd.DataFrame(
        {
            "asset_id": pd.Categorical(
                ["b", "a", "b", "a", "b", "a"],
                categories=["a", "b", "unused"],
            ),
            "date": [1, 1, 2, 2, 2, 3],
            "value": np.array([1.0, 10.0, np.nan, 20.0, 3.0, 30.0], dtype=dtype),
        },
        index=[7, 7, 2, 2, 7, 1],
    )

    actual = trailing_sma(values, window=window, min_periods=min_periods)
    expected = _scalar_oracle(values, window=window, min_periods=min_periods)

    assert actual.index.equals(values.index)
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())


def test_trailing_sma_excludes_current_value_and_is_prefix_invariant():
    values = pd.DataFrame(
        {
            "asset_id": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "date": [1, 1, 2, 2, 3, 3, 4, 4],
            "value": [1.0, 10.0, 2.0, 20.0, 1000.0, 30.0, 4.0, 40.0],
        }
    )

    full = trailing_sma(values, window=2, min_periods=2)
    prefix = trailing_sma(values.iloc[:6], window=2, min_periods=2)

    np.testing.assert_array_equal(full.iloc[:6].to_numpy(), prefix.to_numpy())
    assert full.iloc[4] == 1.5
    assert full.iloc[6] == 501.0


def test_trailing_sma_leaves_missing_asset_groups_uncomputed():
    values = pd.DataFrame(
        {
            "asset_id": ["a", None, "a", None],
            "date": [1, 1, 2, 2],
            "value": [1.0, 8.0, 2.0, 9.0],
        }
    )

    actual = trailing_sma(values, window=1)

    np.testing.assert_array_equal(actual.to_numpy(), [np.nan, np.nan, 1.0, np.nan])


@pytest.mark.parametrize(
    "values",
    [
        pd.DataFrame(
            {
                "asset_id": pd.Series(dtype=object),
                "date": pd.Series(dtype=int),
                "value": pd.Series(dtype=float),
            }
        ),
        pd.DataFrame(
            {"asset_id": [None, None], "date": [1, 2], "value": [1.0, 2.0]},
            index=[4, 4],
        ),
    ],
)
def test_trailing_sma_handles_no_observed_asset_groups(values):
    actual = trailing_sma(values, window=2)

    assert actual.index.equals(values.index)
    assert len(actual) == len(values)
    assert actual.isna().all()
