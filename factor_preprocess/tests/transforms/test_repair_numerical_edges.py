import numpy as np
import pandas as pd
import pytest
from factor_preprocess.transforms.repair_shapes import capped_zscore
from factor_preprocess.transforms.smoothing import kama


def frame(values):
    return pd.DataFrame({"asset_id": "A", "date": np.arange(len(values)), "value": values})


def test_capped_zscore_does_not_erase_finite_extreme_signal():
    values = pd.DataFrame({"date": [0,0,0,0], "asset_id": ["a","b","c","d"],
                           "value": [-1e308, 0., 1e308, np.nan]})
    np.testing.assert_allclose(capped_zscore(values, cap=3), [-1., 0., 1., np.nan], equal_nan=True)


@pytest.mark.parametrize("gap", [np.nan, np.inf, -np.inf])
def test_kama_holds_state_through_missing_er_window_then_resumes(gap):
    # fast=2 => valid trending-window gain 4/9; seed lagged x=2 at t=2.
    out = kama(frame([1.,2.,3.,gap,5.,6.,7.,8.]), period_fast=2, period_slow=5, period_er=2)
    expected_level = 2. + 4./9.
    np.testing.assert_allclose(out.iloc[3:7], expected_level)
    assert out.iloc[7] == pytest.approx(expected_level + (7. - expected_level) * 4./9.)
    # Missing future observations cannot change earlier results.
    prefix = kama(frame([1.,2.,3.,gap,5.]), period_fast=2, period_slow=5, period_er=2)
    np.testing.assert_allclose(out.iloc[:5], prefix, equal_nan=True)
