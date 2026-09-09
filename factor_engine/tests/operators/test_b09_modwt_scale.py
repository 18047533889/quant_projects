import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.conditional_dependence import (
    TsModwtBandCorr,
    _modwt_band_corr_chunk,
)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_modwt_band_corr_public_is_scale_invariant(scale):
    t = np.arange(100, dtype=float)
    values = np.sin(t / 5.0) + 0.3 * np.cos(t / 11.0)
    x = pd.DataFrame({"A": values * scale})
    y = pd.DataFrame({"A": values * scale})
    out = TsModwtBandCorr().calculate(x, y, window=40, level=2, band=2)
    finite = out["A"].dropna()
    assert not finite.empty
    assert finite.iloc[-1] == pytest.approx(1.0, abs=2e-14)


def test_modwt_band_corr_sign_and_zero_energy_contract():
    t = np.arange(60, dtype=float)
    x = np.sin(t / 4.0)
    assert _modwt_band_corr_chunk(x, -x, 2, 2) == pytest.approx(-1.0)
    assert np.isnan(_modwt_band_corr_chunk(np.ones(60), np.ones(60), 2, 2))


def test_modwt_band_corr_missing_clock_and_support_unchanged_by_scale():
    t = np.arange(80, dtype=float)
    x = np.sin(t / 7.0)
    y = np.cos(t / 9.0)
    x[[10, 31]] = np.nan
    y[[10, 31]] = np.nan
    base = _modwt_band_corr_chunk(x, y, 3, 2)
    tiny = _modwt_band_corr_chunk(x * 1e-200, y * 1e-200, 3, 2)
    huge = _modwt_band_corr_chunk(x * 1e200, y * 1e200, 3, 2)
    assert tiny == pytest.approx(base, abs=2e-14)
    assert huge == pytest.approx(base, abs=2e-14)
