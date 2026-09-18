"""Independent normalized share-entropy domain and scale contracts."""
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.shareholder.churn_network import _weighted_entropy


def _frames(rows):
    array = np.asarray(rows, dtype=float)
    return [pd.DataFrame({"A": array[:,i]}) for i in range(8)]


def test_negative_and_infinite_rank_make_share_entropy_unknown():
    frames = _frames([
        [3,1,0,0,0,0,0,0],
        [3,-1,1,1,0,0,0,0],
        [3,np.inf,1,0,0,0,0,0],
        [3,-np.inf,1,0,0,0,0,0],
        [3,np.nan,1,0,0,0,0,0],
    ])
    values = _weighted_entropy(*frames)["A"].to_numpy()
    expected = -(0.75*np.log(.75)+.25*np.log(.25))/np.log(2)
    assert values[0] == pytest.approx(expected)
    assert values[4] == pytest.approx(expected)
    assert np.isnan(values[1:4]).all()


def test_share_entropy_is_scale_safe_and_defined_only_with_two_positive_ranks():
    frames = _frames([
        [1e308,1e308,0,0,0,0,0,0],
        [1e-308,1e-308,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [np.nan]*8,
        [1]*8,
    ])
    values = _weighted_entropy(*frames)["A"].to_numpy()
    np.testing.assert_allclose(values[[0,1,4]], 1.)
    assert np.isnan(values[[2,3]]).all()
