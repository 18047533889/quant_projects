"""Public-registry share-domain checks after complete governance loading."""
import numpy as np
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.tests.runtime.test_r53_pledge_share_domain import _panels, _polars

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("canonical", [
    "holder_freeze_concentration", "holder_pledge_concentration",
    "holder_pledged_holder_count", "holder_pledge_churn",
])
def test_registered_share_counts_keep_domain_and_coordinates(canonical, backend):
    load_all()
    panels = _panels()
    if canonical == "holder_pledge_churn":
        previous = [p.copy() for p in panels]
        for panel in previous:
            panel.iloc[:, :] = 0.
        panels = panels + previous
        expected = [4., np.nan, 2e-308, np.nan, np.nan, np.nan, np.nan, 0.]
    elif canonical == "holder_pledged_holder_count":
        expected = [2., 2., 2., np.nan, np.nan, np.nan, np.nan, 0.]
    else:
        expected = [.625, .5, .5, np.nan, np.nan, np.nan, np.nan, np.nan]
    args = _polars(panels) if backend == "polars" else panels
    out = OperatorRegistry.get(canonical, backend=backend).calculate(*args)
    if isinstance(out, pl.DataFrame):
        assert out.columns == ["timestamp", "A"]
        assert out["timestamp"].to_list() == list(panels[0].index.to_pydatetime())
        actual = out["A"].to_numpy()
    else:
        assert out.index.equals(panels[0].index)
        assert list(out.columns) == ["A"]
        actual = out["A"].to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=0.)
