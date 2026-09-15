import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

NAMES = ("ts_weighted_permutation_entropy", "ts_permutation_transition_entropy", "ts_hurst_dfa", "ts_higuchi_fractal_dimension", "ts_autocorr_decay_half_life")

def _op(name, backend): return OperatorRegistry.get(name, backend=backend)
def _frame(v, backend):
    x = pd.DataFrame({"A": v})
    return pl.from_pandas(x) if backend == "polars" else x
def _last(x):
    if isinstance(x, pl.DataFrame): x = x.to_pandas()
    return float(x.iloc[-1, 0])

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_weighted_entropy_scale_stability_and_prefix(backend):
    v = np.array([1.,4.,2.,8.,3.,9.,5.,11.,6.,13.,7.,15.])
    op = _op("ts_weighted_permutation_entropy", backend)
    a = op.calculate(_frame(v, backend), window=12, order=3, delay=1, weight="variance", normalize=True)
    b = op.calculate(_frame(v * 1e300, backend), 12, 3, 1, "variance", True)
    assert np.isfinite(_last(a)) and _last(a) == pytest.approx(_last(b), rel=1e-12)
    extended = op.calculate(_frame(np.r_[v, 999.], backend), window=12)
    aa = a.to_pandas() if isinstance(a, pl.DataFrame) else a
    ee = extended.to_pandas() if isinstance(extended, pl.DataFrame) else extended
    pd.testing.assert_frame_equal(aa, ee.iloc[:-1].reset_index(drop=True))

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_dfa_scale_stability_higuchi_and_domains(backend):
    t = np.arange(120., dtype=float)
    v = np.sin(t / 5.) + .01 * t
    dfa = _op("ts_hurst_dfa", backend)
    a = _last(dfa.calculate(_frame(v, backend), 120, 4, 30, 6))
    b = _last(dfa.calculate(_frame(v * 1e-20, backend), window=120, min_scale=4, max_scale=30, n_scales=6))
    assert np.isfinite(a) and a == pytest.approx(b, rel=1e-8)
    for bad in ((8, 8), (4, 61)):
        with pytest.raises((TypeError, ValueError)):
            dfa.calculate(_frame(v, backend), window=120, min_scale=bad[0], max_scale=bad[1])
    h = _last(_op("ts_higuchi_fractal_dimension", backend).calculate(_frame(v, backend), 120, 8))
    assert np.isfinite(h)

def test_five_contracts_are_verified_and_typed():
    for name in NAMES:
        op = _op(name, "pandas_numpy")
        schema, _, verified = _parameter_contract(op, ("x",))
        assert verified and set(schema["properties"]) == set(op.metadata.param_names)
    x = _frame(np.arange(60.), "pandas_numpy")
    with pytest.raises((TypeError, ValueError)):
        _op("ts_autocorr_decay_half_life", "pandas_numpy").calculate(x, use_abs=1)
