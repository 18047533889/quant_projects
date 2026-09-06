import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.microstructure.intraday_agg import (
    IntraLimitDuration, IntraLimitFirstHitTime, IntraLimitReopenCount,
)


@pytest.mark.parametrize("side", ["up", "down"])
@pytest.mark.parametrize("reverse", [False, True])
def test_limits_are_bound_per_instrument(side, reverse):
    slots = pd.date_range("2024-01-02 09:31", periods=120, freq="min").append(
        pd.date_range("2024-01-02 13:01", periods=120, freq="min"))
    delta = -1 if side == "up" else 1
    close = pd.DataFrame({"A": np.full(240, 100.+delta), "B": np.full(240, 200.+delta)}, index=slots)
    close.iloc[1:3] = [100., 200.]
    limits = pd.DataFrame({"A": [100.], "B": [200.]}, index=pd.DatetimeIndex(["2024-01-02"]))
    if reverse:
        limits = limits[["B", "A"]]
    kwargs = {"side": side, "high_limit" if side == "up" else "low_limit": limits}
    for cls, expected in ((IntraLimitDuration, 2/240), (IntraLimitFirstHitTime, 1/240),
                          (IntraLimitReopenCount, 1.)):
        actual = cls()._calculate_series(close, **kwargs)
        np.testing.assert_allclose(actual.to_numpy(), [[expected, expected]], rtol=0, atol=1e-14)


def test_missing_instrument_limit_never_uses_other_security():
    slots = pd.date_range("2024-01-02 09:31", periods=120, freq="min").append(
        pd.date_range("2024-01-02 13:01", periods=120, freq="min"))
    close = pd.DataFrame({"A": np.full(240, 100.), "B": np.full(240, 200.)}, index=slots)
    limits = pd.DataFrame({"A": [100.]}, index=pd.DatetimeIndex(["2024-01-02"]))
    actual = IntraLimitDuration()._calculate_series(close, high_limit=limits)
    assert actual["A"].iloc[0] == 1.
    assert np.isnan(actual["B"].iloc[0])
