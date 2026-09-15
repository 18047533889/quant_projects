from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


NAMES = ("intraday_medrv", "intraday_minrv", "intraday_jump_test_stat")
MEDRV_CONST = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
MINRV_CONST = np.pi / (np.pi - 2.0)
THETA_MINUS_2 = (np.pi / 2.0) ** 2 + np.pi - 5.0


def _op(name, backend="pandas_numpy"):
    op = OperatorRegistry.get(name, backend=backend)
    assert op is not None
    return op


def _references(values):
    v = np.asarray(values, dtype=float)
    a = np.abs(v)
    n = len(v)
    med = np.array([np.median(a[i - 2:i + 1]) for i in range(2, n)])
    medrv = MEDRV_CONST * n / (n - 2) * np.sum(med ** 2)
    minrv = MINRV_CONST * n / (n - 1) * np.sum(np.minimum(a[1:], a[:-1]) ** 2)
    rv = np.sum(v ** 2)
    bv = (np.pi / 2.0) * np.sum(a[1:] * a[:-1])
    jump = (rv - bv) / np.sqrt(THETA_MINUS_2 / 3.0 * np.sum(v ** 4))
    return medrv, minrv, jump


def _calculate(name, values, backend="pandas_numpy", **kwargs):
    panel = pd.DataFrame({"B": values, "A": np.asarray(values) * -0.5})
    if backend == "polars":
        panel = pl.from_pandas(panel)
    out = _op(name, backend).calculate(panel, **kwargs)
    return out.to_pandas() if isinstance(out, pl.DataFrame) else out


def test_fresh_load_contracts_and_exact_delegate_twins():
    for name in NAMES:
        pandas_op = _op(name)
        polars_op = _op(name, "polars")
        assert pandas_op.metadata == polars_op.metadata
        meta = pandas_op.metadata
        assert meta.panel_params == ("returns",)
        assert meta.panel_arity == 1
        assert meta.scalar_params == ("window",)
        assert meta.param_specs["window"].default == 240
        assert meta.param_specs["window"].min == 5
        assert meta.param_specs["window"].history_semantics == "max_rows"
        assert meta.param_specs["window"].param_role.value == "horizon"
        assert type(polars_op).__name__ == "JumpRobustPolars"
        assert polars_op.physical_spec().execution_kind.value == "polars_pandas_delegate"


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_independent_numerical_authorities_and_axis_preservation(backend):
    values = np.array([.3, -.4, .2, .9, -.1, .05, -.3, .4])
    refs = dict(zip(NAMES, _references(values)))
    for name in NAMES:
        out = _calculate(name, values, backend, window=8)
        assert out.iloc[-1, 0] == pytest.approx(refs[name], rel=2e-14, abs=2e-14)
        assert list(out.columns) == ["B", "A"]
        assert out.shape == (8, 2)


@pytest.mark.parametrize("name", NAMES)
def test_finite_scale_law_and_no_inf(name):
    values = np.array([.3, -.4, .2, .9, -.1, .05, -.3, .4])
    base = _calculate(name, values, window=8).iloc[-1, 0]
    for scale in (1e-150, 1e150):
        got = _calculate(name, values * scale, window=8).iloc[-1, 0]
        assert np.isfinite(got)
        if name == "intraday_jump_test_stat":
            assert got == pytest.approx(base, rel=2e-14, abs=2e-14)
        else:
            assert got / (scale * scale) == pytest.approx(base, rel=3e-14)
    overflow = _calculate(name, values * 1e200, window=8).iloc[-1, 0]
    assert not np.isinf(overflow)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_missing_slot_fail_closed_domain_and_degenerate_rules(backend):
    leading = np.array([np.nan, .2, -.1, .3, -.2, .4])
    interior = np.array([.1, .2, np.nan, .3, -.2, .4])
    infinite = np.array([np.inf, .2, -.1, .3, -.2, .4])
    for name in NAMES:
        assert np.isfinite(_calculate(name, leading, backend, window=6).iloc[-1, 0])
        assert np.isnan(_calculate(name, interior, backend, window=6).iloc[-1, 0])
        assert np.isnan(_calculate(name, infinite, backend, window=6).iloc[-1, 0])
        assert np.isnan(_calculate(name, np.zeros(6), backend, window=6).iloc[-1, 0])
        panel = pl.DataFrame({"A": np.arange(8.0)}) if backend == "polars" else pd.DataFrame({"A": np.arange(8.0)})
        with pytest.raises((TypeError, ValueError)):
            _op(name, backend).calculate(panel, window=4)
        with pytest.raises((TypeError, ValueError)):
            _op(name, backend).calculate(panel, window=5.5)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_prefix_causality_and_default_backend_parity(backend):
    values = np.sin(np.arange(30.0) / 3.0) * .01
    panel = pd.DataFrame({"B": values, "A": values[::-1]})
    longer = pd.concat([panel, pd.DataFrame({"B": [99.], "A": [-99.]})], ignore_index=True)
    if backend == "polars":
        panel_in, longer_in = pl.from_pandas(panel), pl.from_pandas(longer)
    else:
        panel_in, longer_in = panel, longer
    for name in NAMES:
        first = _op(name, backend).calculate(panel_in)
        extended = _op(name, backend).calculate(longer_in)
        if backend == "polars":
            first, extended = first.to_pandas(), extended.to_pandas()
        pd.testing.assert_frame_equal(first, extended.iloc[:-1].reset_index(drop=True))
    if backend == "polars":
        for name in NAMES:
            expected = _op(name).calculate(panel)
            actual = _op(name, "polars").calculate(pl.from_pandas(panel)).to_pandas()
            pd.testing.assert_frame_equal(expected, actual)
