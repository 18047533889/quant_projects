"""Independent membership-transition oracle; first observations are not entries."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators.index_listing.ops_v2 import _reconstitution_churn as pandas_churn
from factor_engine.cleaned_operators.index_listing.polars_ops_v2 import _reconstitution_churn as polars_churn
from factor_engine.cleaned_operators.common.polars_market_structure import IndexReconstitutionChurnNative


@pytest.mark.parametrize("window", [1,3,6])
def test_all_implementations_ignore_unknown_boundaries_and_first_member(window):
    values = np.array([1.,1.,0.,np.nan,1.,0.,np.inf,0.,1.,1.])
    dates = pd.date_range("2024-01-01", periods=len(values))
    events = np.zeros(len(values))
    for i in range(1,len(values)):
        if np.isfinite(values[i-1:i+1]).all():
            events[i] = float(values[i] != values[i-1])
    expected = np.full(len(values), np.nan)
    for i in range(window-1,len(values)):
        expected[i] = events[i-window+1:i+1].sum()
    frame = pl.DataFrame({"timestamp": dates, "B": values})
    pd_out = pandas_churn(pd.DataFrame({"B": values}, index=dates), window)
    np.testing.assert_allclose(pd_out["B"], expected, equal_nan=True)
    for fn in (polars_churn, IndexReconstitutionChurnNative().calculate):
        out = fn(frame, window=window)
        assert out["timestamp"].equals(frame["timestamp"])
        np.testing.assert_allclose(out["B"], expected, equal_nan=True)
        changed = frame.with_columns(
            pl.when(pl.int_range(pl.len()) >= 7).then(1.).otherwise(pl.col("B")).alias("B"))
        prefix = fn(changed, window=window)
        np.testing.assert_allclose(prefix["B"].to_numpy()[:7], out["B"].to_numpy()[:7], equal_nan=True)


def test_invalid_finite_membership_is_not_coerced_to_boolean():
    frame = pl.DataFrame({"A": [0.,.5,1.]})
    for fn in (polars_churn, IndexReconstitutionChurnNative().calculate):
        with pytest.raises(ValueError, match="ConditionBool"):
            fn(frame, window=2)
    with pytest.raises(ValueError, match="ConditionBool"):
        pandas_churn(frame.to_pandas(), 2)
