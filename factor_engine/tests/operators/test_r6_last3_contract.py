from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_all()


def _op(name, backend="pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None
    return op


def test_final_contracts_and_honest_polars_physical_identity():
    row = _op("row_sum_skipna")
    assert row.metadata.scalar_params == ("min_count",)
    assert row.metadata.variadic_mixed_param == "inputs"
    assert row.metadata.param_specs["min_count"].default == 1
    assert row.metadata.param_specs["min_count"].min == 1
    expected = {
        "group_signal_attraction_share": (("x", "group"), ("damping",)),
        "relation_diffusion_score": (("x", "group"), ("alpha", "steps")),
    }
    for name, (panels, scalars) in expected.items():
        assert _op(name).metadata.panel_params == panels
        assert _op(name).metadata.scalar_params == scalars
    for name in ("row_sum_skipna", *expected):
        pandas_meta = _op(name).metadata
        polars = _op(name, "polars")
        assert pandas_meta.param_specs == polars.metadata.param_specs
        spec = polars._physical_spec
        assert spec.execution_kind in {
            ExecutionKind.POLARS_NUMPY_KERNEL, ExecutionKind.POLARS_PANDAS_DELEGATE,
        }
        assert len(spec.implementation_source_hash) == 64
        int(spec.implementation_source_hash, 16)
        assert spec.kernel_identity


def test_row_sum_variadic_panels_scalar_mix_and_strict_min_count():
    idx = pd.date_range("2026-01-01", periods=2)
    a = pd.DataFrame([[1., np.nan], [2., 4.]], index=idx, columns=list("AB"))
    b = pd.DataFrame([[2., 3.], [np.inf, 5.]], index=idx, columns=list("AB"))
    expected = np.array([[4., 4.], [3., 10.]])
    for backend in ("pandas_numpy", "polars"):
        left, right = (pl.from_pandas(a), pl.from_pandas(b)) if backend == "polars" else (a, b)
        got = _op("row_sum_skipna", backend).calculate(left, right, 1.0, min_count=2)
        got = got.to_numpy() if isinstance(got, pl.DataFrame) else got.to_numpy()
        np.testing.assert_allclose(got, expected, equal_nan=True)
        for bad in (0, True, 1.5):
            with pytest.raises((TypeError, ValueError)):
                _op("row_sum_skipna", backend).calculate(left, right, min_count=bad)


def test_attraction_and_diffusion_independent_closed_forms_and_alignment():
    x = pd.DataFrame([[1., 2., 3.]], columns=list("ABC"))
    group = pd.DataFrame([[1, 1, 1]], columns=x.columns)
    attraction = _op("group_signal_attraction_share").calculate(x, group)
    np.testing.assert_allclose(attraction.iloc[0], [.1916666666666667, 1/3, .475])

    alpha, steps, m = .4, 3, 3
    transition = (np.ones((m, m)) - np.eye(m)) / (m - 1)
    expected = sum((1-alpha) * alpha**(k-1) * np.linalg.matrix_power(transition, k) @ x.iloc[0]
                   for k in range(1, steps + 1))
    expected += alpha**steps * np.linalg.matrix_power(transition, steps) @ x.iloc[0]
    diffusion = _op("relation_diffusion_score").calculate(x, group, alpha=alpha, steps=steps)
    np.testing.assert_allclose(diffusion.iloc[0], expected, atol=1e-14, rtol=1e-14)
    bad_group = group.rename(columns={"C": "D"})
    for name in ("group_signal_attraction_share", "relation_diffusion_score"):
        with pytest.raises((TypeError, ValueError)):
            _op(name).calculate(x, bad_group)


def test_row_sum_real_run_many_variadic_binding_with_scalar_literal():
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    from factor_engine.api.factor import Factor
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from tests.helpers import InMemorySeriesSource

    dates = pd.date_range("2026-01-01", periods=3)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    source = InMemorySeriesSource(data={
        "a": pd.Series([1., np.nan, 2., 4., 3., 6.], index=index),
        "b": pd.Series([2., 3., np.inf, 5., 4., 7.], index=index),
    })
    expr = make_cleaned_call_factory("row_sum_skipna")(col("a"), col("b"), 1.0, min_count=2)
    engine = FactorEngine(backend=build_backend("polars_long"), data_source=source, run_mode="research")
    factor = Factor(name="sum", expr=expr)
    single = engine.run(factor)["result"]
    many = engine.run_many([factor])["results"]["sum"]
    expected = pd.Series(
        [4., 4., 3., 10., 8., 14.],
        index=index,
        name="sum",
    )
    pd.testing.assert_series_equal(single, many, check_names=False)
    pd.testing.assert_series_equal(single, expected, check_names=False)
