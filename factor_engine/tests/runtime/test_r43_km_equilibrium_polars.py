"""Focused contracts for Polars KM equilibrium-root selection."""
import numpy as np
import pandas as pd
import polars as pl

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.polars_dynamics import _stable_attractor_root
from factor_engine.cleaned_operators.registry import OperatorRegistry


def test_stable_root_selects_nearest_of_two_attractors():
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    drift = np.array([2.0, -1.0, 1.0, -2.0, -3.0])
    left = -2.0 + (1.0 * 2.0 / 3.0)
    right = 0.0 + (1.0 * 1.0 / 3.0)
    assert np.isclose(_stable_attractor_root(drift, centers, centers[3]), right)
    assert np.isclose(_stable_attractor_root(drift, centers, centers[0]), left)


def test_stable_root_plateau_single_nan_and_no_root_contracts():
    assert np.isclose(
        _stable_attractor_root(np.array([2.0, 0.0, 0.0, -1.0]), np.arange(4.0), 3.0),
        1.5,
    )
    assert np.isclose(
        _stable_attractor_root(np.array([2.0, 1.0, -1.0, -2.0]), np.arange(4.0), 0.0),
        1.5,
    )
    assert np.isnan(_stable_attractor_root(np.array([2.0, np.nan, -1.0]), np.arange(3.0), 1.0))
    assert np.isnan(_stable_attractor_root(np.array([1.0, 0.5, 0.25]), np.arange(3.0), 1.0))
    assert np.isnan(_stable_attractor_root(np.array([0.0, 0.0, -1.0]), np.arange(3.0), 1.0))


def test_km_equilibrium_polars_matches_canonical_and_prefix():
    load_all()
    rng = np.random.default_rng(2)
    values = rng.normal(size=(200, 2))
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    pandas_frame = pd.DataFrame(values, columns=["A", "B"], index=dates)
    polars_frame = pl.DataFrame({"date": dates, "A": values[:, 0], "B": values[:, 1]})
    pandas_op = OperatorRegistry.get("ts_km_equilibrium_distance", "pandas_numpy", mode="any")
    polars_op = OperatorRegistry.get("ts_km_equilibrium_distance", "polars", mode="any")
    expected = pandas_op.calculate(pandas_frame).to_numpy()
    actual = polars_op.calculate(polars_frame).select(["A", "B"]).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-12)
    changed = polars_frame.with_columns(
        pl.when(pl.int_range(pl.len()) >= 150).then(pl.col("A") * -9 + 3).otherwise(pl.col("A")).alias("A")
    )
    after = polars_op.calculate(changed).select(["A", "B"]).to_numpy()
    np.testing.assert_allclose(actual[:150], after[:150], equal_nan=True, atol=1e-12)


def test_km_equilibrium_physical_path_is_truthful():
    load_all()
    op = OperatorRegistry.get("ts_km_equilibrium_distance", "polars", mode="any")
    spec = op.physical_spec()
    assert spec.execution_kind is ExecutionKind.POLARS_NUMPY_KERNEL
    assert spec.materializes_full_panel
    assert not spec.supports_lazy
    assert not spec.supports_streaming
