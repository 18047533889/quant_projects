import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.weighted_moment_ext import _ts_weighted_standardized_moment


@pytest.mark.parametrize("xscale", [1., 1e-100, 1e100, 1e-300, 1e300])
@pytest.mark.parametrize("wscale", [1., 1e-200, 1e200])
@pytest.mark.parametrize("order", [3, 4])
def test_weighted_moment_units_preserve_independent_reference(xscale, wscale, order):
    x = np.array([0., 1., 2., 5.])
    mean = sum(x)/4
    variance = sum((float(v)-mean)**2 for v in x)/4
    expected = sum((float(v)-mean)**order for v in x)/4 / variance**(order/2)
    result = _ts_weighted_standardized_moment(pd.DataFrame(x*xscale),
        pd.DataFrame(np.ones(4)*wscale), window=4, order=order)
    assert result.iloc[-1, 0] == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("weights", [[0., 0., 0., 0.], [1., 0., 0., 0.],
    [1., -1., 1., 1.], [1., np.nan, 1., 1.], [1., np.inf, 1., 1.]])
def test_invalid_weight_or_effective_support_fails_closed(weights):
    out = _ts_weighted_standardized_moment(pd.DataFrame([0., 1., 2., 5.]),
        pd.DataFrame(weights), window=4)
    assert np.isnan(out.iloc[-1, 0])


def test_true_constant_is_undefined():
    out = _ts_weighted_standardized_moment(pd.DataFrame([1e-200]*4),
        pd.DataFrame([1.]*4), window=4)
    assert np.isnan(out.iloc[-1, 0])

@pytest.mark.parametrize("order", [3, 4])
def test_weighted_moment_large_representable_offset(order):
    # Integer increments remain exact in binary64 at this offset.
    x = np.array([0., 1., 2., 5.])
    mean = sum(x)/len(x)
    variance = sum((float(v)-mean)**2 for v in x)/len(x)
    expected = sum((float(v)-mean)**order for v in x)/len(x)/variance**(order/2)
    out = _ts_weighted_standardized_moment(
        pd.DataFrame(x + 1e12), pd.DataFrame(np.ones(4)), window=4, order=order)
    assert out.iloc[-1, 0] == pytest.approx(expected, rel=2e-14)
