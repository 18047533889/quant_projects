from fractions import Fraction
import numpy as np
import pandas as pd
import pytest
from factor_engine.cleaned_operators.robust_scale import _hodges_lehmann
from factor_engine.cleaned_operators.binned_response import (
    TsBinnedResponseMonotonicity, TsBinnedResponseCurvature,
)


@pytest.mark.parametrize("values", [
    [1e308] * 3, [-1e308] * 3, [1e308] * 4,
    [1e308, -1e308, 1e308, -1e308],
    [np.nextafter(0., 1.)] * 3,
    [np.nextafter(0., 1.), 2*np.nextafter(0., 1.), 3*np.nextafter(0., 1.)],
    [-4., 1., 3., 10.],
])
def test_hodges_lehmann_extremes_against_exact_rational(values):
    # Each stored binary64 midpoint and the final median round separately.
    mids = sorted(float((Fraction(float(a)) + Fraction(float(b))) / 2)
                  for i, a in enumerate(values) for b in values[i:])
    middle = len(mids) // 2
    expected = (mids[middle] if len(mids) % 2 else
                float((Fraction(mids[middle-1]) + Fraction(mids[middle])) / 2))
    with np.errstate(over="raise", invalid="raise"):
        actual = _hodges_lehmann(np.array(values))
    assert np.isfinite(actual)
    assert actual == expected


@pytest.mark.parametrize("cls", [TsBinnedResponseMonotonicity, TsBinnedResponseCurvature])
@pytest.mark.parametrize("window", [14, 15, 16])
def test_joint_window_domain_public_and_cpu(cls, window):
    x = pd.DataFrame(np.arange(40.), columns=["a"])
    y = x*x + x
    op = cls()
    if window < 15:
        for fn in (op.calculate, op._calculate_series):
            with pytest.raises(ValueError, match="INFEASIBLE_PARAMETER_DOMAIN"):
                fn(y, x, window=window, bins=5, min_per_bin=3)
    else:
        assert np.isfinite(op.calculate(y, x, window=window, bins=5, min_per_bin=3).to_numpy()).any()
    relation, = op.metadata.relational_specs
    assert relation.check(dict(window=window, bins=5, min_per_bin=3)) == (window >= 15)


def test_curvature_minimum_four_bins_preserved():
    x = pd.DataFrame(np.arange(40.))
    with pytest.raises(ValueError):
        TsBinnedResponseCurvature().calculate(x, x, window=40, bins=3)


def test_short_data_is_warmup_not_invalid_parameter():
    x = pd.DataFrame(np.arange(5.))
    result = TsBinnedResponseMonotonicity().calculate(x, x, window=15, bins=5)
    assert result.isna().all().all()
