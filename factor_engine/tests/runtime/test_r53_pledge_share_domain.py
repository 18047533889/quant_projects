"""Independent finite/nonnegative share-count and scale contracts."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators.shareholder import churn_network as reference
from factor_engine.cleaned_operators.shareholder import polars_churn_network as native


def _panels():
    rows = np.array([
        [3.,1.], [1e308,1e308], [1e-308,1e-308],
        [-1.,3.], [np.inf,3.], [-np.inf,3.], [np.nan,3.], [0.,0.],
    ])
    rows = np.pad(rows, ((0,0),(0,8)))
    index = pd.date_range("2024-01-01", periods=len(rows))
    return [pd.DataFrame({"A": rows[:,i]}, index=index) for i in range(10)]


def _polars(panels):
    return [pl.from_pandas(p.rename_axis("timestamp").reset_index()) for p in panels]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_share_hhi_is_scale_safe_and_invalid_counts_remain_unknown(backend):
    panels = _panels()
    if backend == "pandas":
        out = reference._share_hhi(*panels)["A"].to_numpy()
    else:
        result = native._hhi(*_polars(panels))
        assert result["timestamp"].to_list() == list(panels[0].index.to_pydatetime())
        out = result["A"].to_numpy()
    np.testing.assert_allclose(out[:3], [.625,.5,.5], rtol=1e-12)
    assert np.isnan(out[3:]).all()


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pledged_count_rejects_negative_and_infinite_counts(backend):
    panels = _panels()
    out = (reference._pledged_holder_count(*panels)["A"].to_numpy()
           if backend == "pandas" else native._pledged_count(*_polars(panels))["A"].to_numpy())
    np.testing.assert_array_equal(out[:3], [2.,2.,2.])
    assert np.isnan(out[3:7]).all()
    assert out[7] == 0.


def test_pledge_churn_never_replaces_infinity_with_a_large_share_count():
    panels = _panels()
    previous = [p * 0 for p in panels]
    for p in previous:
        p.iloc[:,:] = 0.
    result = reference._pledge_churn(*panels, *previous)["A"].to_numpy()
    assert result[0] == 4.
    assert np.isnan(result[1])  # finite inputs whose mathematical sum exceeds float64
    assert result[2] == pytest.approx(2e-308, abs=0.)
    assert np.isnan(result[3:7]).all()
    assert result[7] == 0.


def test_ranked_native_rejects_mismatched_time_axis():
    panels = _polars(_panels())
    panels[-1] = panels[-1].reverse()
    with pytest.raises(ValueError, match="different PanelIdentity"):
        native._hhi(*panels)
