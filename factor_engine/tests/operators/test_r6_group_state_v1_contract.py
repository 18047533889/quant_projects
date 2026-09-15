from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

NAMES = ("beta_residual_z", "beta_divergence_pct", "relative_strength_group_pct")


def op(name, backend="pandas_numpy"):
    value = OperatorRegistry.get(name, backend)
    assert value is not None
    return value


def panels(rows=30):
    market = np.linspace(-0.03, 0.04, rows) + 0.004 * np.sin(np.arange(rows))
    noise = 0.0005 * np.cos(np.arange(rows) * 1.7)
    ret = np.column_stack(
        (0.002 + 1.4 * market + noise, -0.001 + 0.7 * market - noise)
    )
    ret[17, 0] += 0.02
    columns = ["A", "B"]
    return (
        pd.DataFrame(ret, columns=columns),
        pd.DataFrame(np.column_stack((market, market)), columns=columns),
        pd.DataFrame(np.full((rows, 2), "sector"), columns=columns),
    )


def call(name, backend="pandas_numpy", window=8):
    ret, market, group = panels()
    second = group if name == "relative_strength_group_pct" else market
    if backend == "polars":
        ret, second = pl.from_pandas(ret), pl.from_pandas(second)
    result = op(name, backend).calculate(ret, second, window=window)
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


def test_contract_defaults_topology_and_physical_truth():
    for name in NAMES:
        pandas_op, polars_op = op(name), op(name, "polars")
        assert pandas_op.metadata == polars_op.metadata
        meta = pandas_op.metadata
        expected_second = "group" if name == "relative_strength_group_pct" else "market_ret"
        assert meta.panel_params == ("ret", expected_second)
        assert meta.panel_arity == 2
        assert meta.scalar_params == ("window",)
        assert meta.output_unit == "dimensionless"
        assert meta.param_specs["window"].default == 20
        assert meta.param_specs["window"].history_semantics == "max_rows"
        spec = polars_op.physical_spec()
        assert spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
        assert spec.materializes_full_panel
        assert not spec.supports_lazy and not spec.supports_streaming
        assert len(spec.implementation_source_hash) == 64
        assert name in spec.kernel_identity


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_exact_backend_parity_prefix_causality_and_strict_window(backend):
    for name in NAMES:
        expected = call(name)
        actual = call(name, backend)
        pd.testing.assert_frame_equal(actual, expected)
        with pytest.raises((TypeError, ValueError)):
            call(name, backend, window=8.5)

        ret, market, group = panels()
        second = group if name == "relative_strength_group_pct" else market
        base = op(name).calculate(ret, second, window=8)
        ret.iloc[-1] = 1e6
        changed = op(name).calculate(ret, second, window=8)
        pd.testing.assert_frame_equal(base.iloc[:-1], changed.iloc[:-1])


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_strict_prior_beta_fit_and_group_ex_self_semantics(backend):
    z = call("beta_residual_z", backend)
    divergence = call("beta_divergence_pct", backend)
    # The first eight rows have no complete prior fit. A current-row shock is
    # evaluated against, and never included in, that prior fit.
    assert z.iloc[:8].isna().all().all()
    assert divergence.iloc[:8].isna().all().all()
    assert z.iloc[17, 0] > 1.0
    assert divergence.iloc[17, 0] > 0.0

    relative = call("relative_strength_group_pct", backend)
    np.testing.assert_allclose(relative.iloc[7:].sum(axis=1), 0.0, atol=1e-15)
    assert relative.iloc[:7].isna().all().all()


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_degenerate_denominators_and_unknown_windows_fail_closed(backend):
    ret, market, group = panels()
    market.iloc[:9] = 0.0
    left, right = ret, market
    if backend == "polars":
        left, right = pl.from_pandas(left), pl.from_pandas(right)
    for name in ("beta_residual_z", "beta_divergence_pct"):
        result = op(name, backend).calculate(left, right, window=8)
        result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
        assert result.iloc[8].isna().all()

    group.iloc[:, 1] = "other"
    left, right = ret, group
    if backend == "polars":
        left, right = pl.from_pandas(left), pl.from_pandas(right)
    result = op("relative_strength_group_pct", backend).calculate(left, right, window=8)
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    assert result.iloc[7:].isna().all().all()
