"""R28: ts_vol_of_vol must reject guaranteed-all-NaN parameter domains."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _ret() -> pd.DataFrame:
    t = np.arange(64, dtype=float)[:, None]
    a = np.arange(3, dtype=float)[None, :]
    return pd.DataFrame(0.01 * np.sin(t / 3.0 + a) + 0.004 * np.cos(t / 7.0 + a / 2.0))


def _backend_ret(backend: str):
    if backend == "polars":
        import polars as pl
        return pl.from_pandas(_ret())
    return _ret()


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize(
    "params",
    [
        {"inner_window": 5, "outer_window": 12, "min_periods": 6},
        {"inner_window": 12, "outer_window": 5, "min_periods": 6},
        {"inner_window": 5, "outer_window": 12, "min_periods": 4.5},
        {"inner_window": 5, "outer_window": 12, "min_periods": True},
    ],
)
def test_rejects_infeasible_or_non_integer_min_periods(backend: str, params: dict[str, object]) -> None:
    op = OperatorRegistry.get("ts_vol_of_vol", backend, mode="any")
    with pytest.raises((TypeError, ValueError)):
        op.calculate(_backend_ret(backend), **params)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_valid_shared_min_periods_produces_finite_output(backend: str) -> None:
    op = OperatorRegistry.get("ts_vol_of_vol", backend, mode="any")
    out = op.calculate(_backend_ret(backend), inner_window=5, outer_window=12, min_periods=4)
    values = out.to_numpy() if hasattr(out, "to_numpy") else np.asarray(out)
    assert np.isfinite(values).any()


def test_declares_both_relational_constraints() -> None:
    op = OperatorRegistry.get("ts_vol_of_vol", "pandas_numpy", mode="any")
    expressions = {spec.expression for spec in op.metadata.relational_specs}
    assert "min_periods <= inner_window" in expressions
    assert "min_periods <= outer_window" in expressions


def test_pre_lowering_contract_rejects_same_infeasible_domain() -> None:
    from factor_engine.cleaned_operators.base import validate_operator_call

    op = OperatorRegistry.get("ts_vol_of_vol", "pandas_numpy", mode="any")
    with pytest.raises(ValueError, match="min_periods must be <= inner_window"):
        validate_operator_call(
            op,
            (_ret(),),
            {"inner_window": 5, "outer_window": 12, "min_periods": 8},
        )
