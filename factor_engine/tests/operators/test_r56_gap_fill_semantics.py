"""Gap completion counts crossings back to the previous close, not continuation."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

def evaluate(close, open_px, previous, window, backend):
    load_all()
    index = pd.date_range("2024-01-01", periods=len(close))
    frames = [pd.DataFrame({"A": v}, index=index) for v in (close, open_px, previous)]
    if backend == "polars":
        frames = [pl.from_pandas(f.rename_axis("timestamp").reset_index()) for f in frames]
    result = OperatorRegistry.get("ts_gap_fill_ratio", backend).calculate(*frames, window=window)
    if backend == "polars":
        assert result["timestamp"].to_list() == list(index.to_pydatetime())
    else:
        pd.testing.assert_index_equal(result.index, index)
    return result["A"].to_numpy()

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_both_gap_directions_and_no_gap_denominator(backend):
    # Up filled; down filled; up continuation; down continuation; no gap; up filled.
    actual = evaluate([100,100,105,95,105,99], [102,98,102,98,100,102], [100]*6, 3, backend)
    np.testing.assert_allclose(actual, [np.nan,np.nan,2/3,1/3,0,1/2], equal_nan=True)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_down_gaps_are_not_discarded(backend):
    actual = evaluate([100,101,97], [98]*3, [100]*3, 1, backend)
    np.testing.assert_allclose(actual, [1,1,0])

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 0.])
def test_missing_or_invalid_prices_do_not_count_as_non_events(backend, bad):
    actual = evaluate([100,bad,100,100], [102]*4, [100]*4, 2, backend)
    np.testing.assert_allclose(actual, [np.nan,np.nan,np.nan,1], equal_nan=True)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_no_gap_window_is_undefined(backend):
    actual = evaluate([101]*4, [100]*4, [100]*4, 2, backend)
    assert np.isnan(actual).all()

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_multistock_prefix_and_price_unit_invariance(backend):
    load_all()
    index = pd.date_range("2024-01-01", periods=12)
    opening = np.array([102.,98.,102.,98.,100.,102.] * 2)
    closing = np.array([100.,100.,105.,95.,105.,99.] * 2)
    c = pd.DataFrame({"B": closing, "A": closing[::-1]}, index=index)
    o = pd.DataFrame({"B": opening, "A": opening[::-1]}, index=index)
    p = pd.DataFrame(100., index=index, columns=c.columns)
    op = OperatorRegistry.get("ts_gap_fill_ratio", backend)
    def call(frames):
        if backend == "polars":
            frames = [pl.from_pandas(f.rename_axis("timestamp").reset_index()) for f in frames]
        out = op.calculate(*frames, window=3)
        return out.select("B","A").to_numpy() if backend == "polars" else out.to_numpy()
    full = call([c,o,p])
    np.testing.assert_allclose(call([f.iloc[:8] for f in (c,o,p)]), full[:8], equal_nan=True)
    for scale in (1e-200,1e200):
        np.testing.assert_allclose(call([f*scale for f in (c,o,p)]), full, equal_nan=True)
    for col_idx,col in enumerate(c.columns):
        expected = []
        for t in range(len(c)):
            if t < 2:
                expected.append(np.nan)
                continue
            trials = [(o[col].iloc[s], c[col].iloc[s]) for s in range(t-2,t+1) if o[col].iloc[s] != 100.]
            fills = sum((opn > 100. and cls <= 100.) or (opn < 100. and cls >= 100.) for opn,cls in trials)
            expected.append(fills/len(trials) if trials else np.nan)
        np.testing.assert_allclose(full[:,col_idx], expected, equal_nan=True)
