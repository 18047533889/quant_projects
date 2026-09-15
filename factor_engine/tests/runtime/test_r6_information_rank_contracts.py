"""Independent salience-rank and Decimal Benford checks on final loaded backends."""
from decimal import Decimal
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import rankdata
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

SCORE = "ts_score_rank_weighted_mean"
BENFORD = "report_benford_js_divergence"
@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()
def frame(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2025-01-01", periods=len(values), tz="Asia/Hong_Kong"))
def run(name, backend, inputs, **kwargs):
    op = OperatorRegistry.get(name, backend, mode="research")
    names = ("target", "score") if name == SCORE else ("amount",)
    assert op.metadata.panel_params == names
    assert _parameter_contract(op, names)[2]
    if backend == "polars":
        inputs = [pl.from_pandas(v.rename_axis("date").reset_index()) for v in inputs]
    out = op.calculate(**dict(zip(names, inputs)), **kwargs)
    return out.to_pandas().set_index("date").rename_axis(None) if backend == "polars" else out

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_rank_ties_true_decay_and_finite_extremes(backend):
    target = frame([1., 8., 3., -2., 5., 7., 4., 12.])
    score = frame([2., 2., 1., 4., 4., 3., 2., 1.])
    rank = rankdata(-score.A.to_numpy()) - 1
    weight = .3 ** rank
    expected = np.sum(weight * target.A.to_numpy()) / weight.sum()
    got = run(SCORE, backend, [target, score], decay=.3)
    assert got.iloc[-1, 0] == pytest.approx(expected)
    np.testing.assert_allclose(run(SCORE, backend, [target, score]),
        run(SCORE, backend, [target, score], window=60, decay=.85), equal_nan=True)
    np.testing.assert_allclose(run(SCORE, backend, [target.iloc[:5], score.iloc[:5]], decay=.3),
        got.iloc[:5], equal_nan=True)
    # All tied scores must stay equally weighted even for subnormal decay.
    huge = frame([1e308] * 8)
    tied = frame([1.] * 8)
    out = run(SCORE, backend, [huge, tied], decay=1e-300)
    np.testing.assert_allclose(out, huge, rtol=1e-15)
    invalid = target.copy(); invalid.iloc[-1] = np.nan
    assert np.isfinite(run(SCORE, backend, [invalid, score]).iloc[-1, 0])
    for bad in (True, 0., np.inf):
        with pytest.raises((ValueError, TypeError)):
            run(SCORE, backend, [target, score], decay=bad)
    with pytest.raises((ValueError, TypeError)):
        run(SCORE, backend, [target, score.rename(columns={"A": "B"})])

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_decimal_first_digit_oracle_including_subnormals(backend):
    values = np.array([5e-324, 1e-323, 1.5e-323, 2e-323, 3e-323, 9e-323,
        1e308, 9e307, 100., 999., -345., 0., np.nan])
    digits = [int(Decimal.from_float(float(abs(v))).as_tuple().digits[0])
        for v in values if np.isfinite(v) and v != 0]
    counts = np.bincount(digits, minlength=10)[1:]
    p = counts / counts.sum(); b = np.log10(1 + 1 / np.arange(1, 10))
    m = (p+b)/2
    expected = np.sqrt((np.sum(p[p>0]*np.log(p[p>0]/m[p>0])) + np.sum(b*np.log(b/m)))/(2*np.log(2)))
    got = run(BENFORD, backend, [frame(values)])
    assert got.iloc[-1, 0] == pytest.approx(expected, abs=1e-12)
    for bad in (9, True, 10.5):
        with pytest.raises((ValueError, TypeError)):
            run(BENFORD, backend, [frame(values)], window=bad)
    assert got.iloc[:9].isna().all().all()

def test_polars_paths_do_not_convert_to_pandas(monkeypatch):
    op1 = OperatorRegistry.get(SCORE, "polars", mode="research")
    op2 = OperatorRegistry.get(BENFORD, "polars", mode="research")
    x = pl.DataFrame({"A": np.arange(1., 15.)})
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected Pandas conversion")
    monkeypatch.setattr(pl.DataFrame, "to_pandas", forbidden)
    for op, kwargs in ((op1, dict(target=x, score=x)), (op2, dict(amount=x))):
        assert isinstance(op.calculate(**kwargs), pl.DataFrame)
        assert op.physical_spec().execution_kind.value == "polars_numpy_kernel"
