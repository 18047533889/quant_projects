"""Independent numerical oracles for non-finite Polars result handling."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()


@pytest.mark.parametrize("canonical,oracle", [
    ("sqrt_abs", lambda x: np.sqrt(np.abs(x))),
    ("log_abs", lambda x: np.log(np.abs(x))),
    ("exp_neg", lambda x: np.exp(-x)),
])
@pytest.mark.parametrize("nan_to_null", [True, False])
def test_elementwise_nonfinite_oracle(canonical, oracle, nan_to_null):
    data = pd.DataFrame({"z": [-np.inf, -1000., -8., -1., -0., np.nan, 1., 8., np.inf],
                         "a": [1., 2., 3., 4., 5., 6., 7., 8., 9.]})
    with np.errstate(all="ignore"):
        expected = oracle(data.to_numpy())
    expected[~np.isfinite(expected)] = np.nan
    for backend, frame in [("pandas_numpy", data), ("polars", pl.from_pandas(data, nan_to_null=nan_to_null))]:
        actual = OperatorRegistry.get(canonical, backend, mode="any").calculate(frame)
        assert list(actual.columns) == list(data.columns)
        np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("nan_to_null", [True, False])
def test_signed_log_native_matches_oracle_without_pandas(monkeypatch, nan_to_null):
    from factor_engine.cleaned_operators.common.polars_daily_native import SignedLogNative
    from factor_engine.backend.polars_backend_kind import canonical_polars_kind, PolarsImplementationKind
    values = np.array([-np.inf, -1e300, -8., -1e-14, -0., 0., 1e-14, 1., 8., np.inf, np.nan])
    data = pd.DataFrame({"z": values, "a": np.arange(len(values), dtype=float)})
    with np.errstate(all="ignore"):
        expected = np.sign(data.to_numpy()) * np.log(np.abs(data.to_numpy()) + 1e-10)
    reference = OperatorRegistry.get("signed_log", "pandas_numpy", mode="any").calculate(data)
    np.testing.assert_allclose(reference.to_numpy(), expected, equal_nan=True)
    op = OperatorRegistry.get("signed_log", "polars", mode="any")
    assert isinstance(op, SignedLogNative)
    assert canonical_polars_kind("signed_log") == PolarsImplementationKind.POLARS_NATIVE
    assert not op._physical_spec.validation_errors()
    frame = pl.from_pandas(data, nan_to_null=nan_to_null)
    def forbidden(*args, **kwargs):
        raise AssertionError("native kernel must not materialize pandas")
    monkeypatch.setattr(pl.DataFrame, "to_pandas", forbidden)
    actual = op.calculate(frame)
    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
    # Prefix execution is independent of future values.
    np.testing.assert_allclose(op.calculate(frame.head(7)).to_numpy(), expected[:7], equal_nan=True)


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("nan_to_null", [True, False])
def test_pct_zero_and_overflow_oracle(d, nan_to_null):
    data = pd.DataFrame({"z": [0., 2., np.nan, 4., np.inf, -np.inf, 0., -3., 1e-300, 1e300],
                         "a": [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]})
    with np.errstate(all="ignore"):
        expected = data.to_numpy() / data.shift(d).to_numpy() - 1
    # The canonical percentage-change contract masks zero/missing lagged
    # values; finite nonzero divisors may still produce an infinite result.
    previous = data.shift(d).to_numpy()
    expected[np.isnan(previous) | (previous == 0)] = np.nan
    for backend, frame in [("pandas_numpy", data), ("polars", pl.from_pandas(data, nan_to_null=nan_to_null))]:
        actual = OperatorRegistry.get("ts_pct", backend, mode="any").calculate(frame, d=d)
        assert list(actual.columns) == list(data.columns)
        np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
