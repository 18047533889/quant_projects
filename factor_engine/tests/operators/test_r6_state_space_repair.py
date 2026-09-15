"""Fresh finalized-registry and numerical oracles for Kalman state operators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "ts_kalman_level", "ts_kalman_trend", "ts_kalman_innovation_z",
    "ts_kalman_beta", "ts_kalman_beta_change", "ts_kalman_beta_uncertainty",
)


def _panel(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-02", periods=len(values)), dtype=float)


def _as_backend(frame, backend):
    if backend == "pandas_numpy":
        return frame
    import polars as pl
    out = frame.copy(); out.index.name = "date"
    return pl.from_pandas(out.reset_index())


def _values(frame):
    if isinstance(frame, pd.DataFrame):
        return frame["A"].to_numpy(dtype=float)
    return frame["A"].to_numpy()


def test_fresh_final_registry_has_exact_topology_on_every_backend():
    load_all()
    for name in NAMES:
        panels = ("y", "x") if "beta" in name else ("x",)
        for backend in OperatorRegistry.backends_for(name):
            op = OperatorRegistry.get(name, backend, mode="any")
            assert tuple(op.metadata.panel_params) == panels
            assert op.metadata.panel_arity == len(panels)
            assert set(op.metadata.scalar_params) == set(op.metadata.param_names) - set(panels)


def test_all_backends_default_keyword_positional_and_prefix_equivalence():
    load_all()
    x = _panel([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    y = 2.0 * x
    for name in NAMES:
        for backend in OperatorRegistry.backends_for(name):
            op = OperatorRegistry.get(name, backend, mode="any")
            bx, by = _as_backend(x, backend), _as_backend(y, backend)
            if "beta" in name:
                positional = op.calculate(by, bx, 0.001, 1.0, "absolute", 2)
                keyword = op.calculate(y=by, x=bx, q=0.001, r=1.0, scale_mode="absolute", min_warmup=2)
                default = op.calculate(by, bx)
                prefix = op.calculate(_as_backend(y.iloc[:4], backend), _as_backend(x.iloc[:4], backend), 0.001, 1.0, "absolute", 2)
            elif name == "ts_kalman_trend":
                positional = op.calculate(bx, 1e-5, 1e-5, 1.0, "absolute")
                keyword = op.calculate(x=bx, q_level=1e-5, q_trend=1e-5, r=1.0, scale_mode="absolute")
                default = op.calculate(bx)
                prefix = op.calculate(_as_backend(x.iloc[:4], backend), 1e-5, 1e-5, 1.0, "absolute")
            else:
                positional = op.calculate(bx, 1e-4, 1.0, "absolute")
                keyword = op.calculate(x=bx, q=1e-4, r=1.0, scale_mode="absolute")
                default = op.calculate(bx)
                prefix = op.calculate(_as_backend(x.iloc[:4], backend), 1e-4, 1.0, "absolute")
            np.testing.assert_allclose(_values(positional), _values(keyword), equal_nan=True, err_msg=f"{name}:{backend}")
            assert len(_values(default)) == len(x)
            np.testing.assert_allclose(_values(prefix), _values(positional)[:4], equal_nan=True, err_msg=f"{name}:{backend}:prefix")


def test_independent_local_level_and_beta_oracles():
    load_all()
    x = _panel([1.0, 2.0, np.nan])
    level = OperatorRegistry.get("ts_kalman_level", mode="any").calculate(x, q=0.1, r=1.0)
    expected_second = 1.0 + (1.1 / 2.1) * (2.0 - 1.0)
    np.testing.assert_allclose(level["A"], [1.0, expected_second, expected_second], atol=1e-12)

    constant = _panel([3.0] * 6)
    np.testing.assert_allclose(OperatorRegistry.get("ts_kalman_level", mode="any").calculate(constant)["A"], 3.0)
    np.testing.assert_allclose(OperatorRegistry.get("ts_kalman_trend", mode="any").calculate(constant)["A"], 0.0, atol=1e-12)
    innovation = OperatorRegistry.get("ts_kalman_innovation_z", mode="any").calculate(constant)["A"].to_numpy()
    np.testing.assert_allclose(innovation[np.isfinite(innovation)], 0.0, atol=1e-12)

    bx = _panel([1.0, 2.0, 3.0, 4.0])
    by = 2.0 * bx
    beta = OperatorRegistry.get("ts_kalman_beta", mode="any").calculate(by, bx, q=0.0, r=1.0, min_warmup=2)
    change = OperatorRegistry.get("ts_kalman_beta_change", mode="any").calculate(by, bx, q=0.0, r=1.0, min_warmup=2)
    uncertainty = OperatorRegistry.get("ts_kalman_beta_uncertainty", mode="any").calculate(by, bx, q=0.0, r=1.0, min_warmup=2)
    np.testing.assert_allclose(beta["A"].iloc[1:], 2.0, atol=1e-12)
    np.testing.assert_allclose(change["A"].iloc[2:], 0.0, atol=1e-12)
    assert uncertainty["A"].iloc[1] == 0.2
    assert uncertainty["A"].iloc[2] < 0.2
