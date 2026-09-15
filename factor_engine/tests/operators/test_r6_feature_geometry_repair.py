"""Contract and independent numerical coverage for repaired feature geometry operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "ts_beta_break_score",
    "ts_feature_effective_rank",
    "ts_feature_mode_share",
    "ts_feature_subspace_rotation",
)


def _panel(values, *, start="2024-01-02", column="A"):
    return pd.DataFrame({column: np.asarray(values, dtype=float)}, index=pd.date_range(start, periods=len(values)))


def _backend(panel, backend):
    if backend == "pandas_numpy":
        return panel
    import polars as pl
    return pl.DataFrame({c: panel[c].to_list() for c in panel.columns})


def _values(result):
    if isinstance(result, pd.DataFrame):
        return result.to_numpy(dtype=float)
    return result.to_numpy()


def _operators(name):
    load_all()
    assert set(OperatorRegistry.backends_for(name)) == {"pandas_numpy", "polars"}
    result = []
    for backend in OperatorRegistry.backends_for(name):
        op = OperatorRegistry.get(name, backend, mode="any")
        op._r6_backend = backend
        result.append(op)
    return result


def test_exact_topology_defaults_history_and_honest_final_backends():
    expected = {
        "ts_beta_break_score": (("y", "x"), {"recent_window": 30, "prior_window": 90}),
        "ts_feature_effective_rank": (("f1", "f2", "f3"), {"window": 60}),
        "ts_feature_mode_share": (("f1", "f2", "f3"), {"window": 60}),
        "ts_feature_subspace_rotation": (
            ("f1", "f2", "f3"), {"recent_window": 30, "prior_window": 90, "eigen_gap": 0.02}
        ),
    }
    for name, (panels, defaults) in expected.items():
        for op in _operators(name):
            meta = op.metadata
            assert tuple(meta.panel_params) == panels
            assert meta.panel_arity == len(panels)
            assert tuple(meta.scalar_params) == tuple(defaults)
            assert {k: meta.param_specs[k].default for k in defaults} == defaults
            for key in set(defaults) & {"window", "recent_window", "prior_window"}:
                spec = meta.param_specs[key]
                assert spec.dtype is int and spec.min == 5 and spec.param_role is ParamRole.HORIZON
            if "recent_window" in defaults:
                assert meta.param_specs["recent_window"].history_semantics == "max_rows"
                assert meta.param_specs["recent_window"].history_formula == "recent_window + prior_window"
            if "window" in defaults:
                assert meta.param_specs["window"].history_semantics == "max_rows"
            if op._r6_backend == "polars":
                assert op._physical_spec.execution_kind.value == "polars_pandas_delegate"


def test_call_forms_prefix_and_backend_parity():
    t = np.arange(1.0, 41.0)
    f1 = _panel(np.sin(t / 3) + t / 20)
    f2 = _panel(np.cos(t / 4) - t / 30)
    f3 = _panel(np.sin(t / 7) + np.cos(t / 5))
    y, x = _panel(1.7 * t + np.sin(t)), _panel(t)
    cases = {
        "ts_feature_mode_share": ((f1, f2, f3), (12,), {"window": 12}),
        "ts_feature_effective_rank": ((f1, f2, f3), (12,), {"window": 12}),
        "ts_feature_subspace_rotation": ((f1, f2, f3), (8, 12, 0.0), {"recent_window": 8, "prior_window": 12, "eigen_gap": 0.0}),
        "ts_beta_break_score": ((y, x), (8, 12), {"recent_window": 8, "prior_window": 12}),
    }
    for name, (panels, scalars, kwargs) in cases.items():
        reference = None
        for op in _operators(name):
            bp = tuple(_backend(p, op._r6_backend) for p in panels)
            positional = _values(op.calculate(*bp, *scalars))
            keyword = _values(op.calculate(**dict(zip(op.metadata.panel_params, bp)), **kwargs))
            mixed = _values(op.calculate(bp[0], **dict(zip(op.metadata.panel_params[1:], bp[1:])), **kwargs))
            np.testing.assert_allclose(positional, keyword, equal_nan=True)
            np.testing.assert_allclose(positional, mixed, equal_nan=True)
            prefix = tuple(_backend(p.iloc[:31], op._r6_backend) for p in panels)
            np.testing.assert_allclose(_values(op.calculate(*prefix, *scalars)), positional[:31], equal_nan=True)
            if reference is None:
                reference = positional
            else:
                np.testing.assert_allclose(positional, reference, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name,param,bad", [
    ("ts_feature_mode_share", "window", 4),
    ("ts_feature_effective_rank", "window", 5.5),
    ("ts_feature_mode_share", "window", np.nan),
    ("ts_feature_effective_rank", "window", np.inf),
    ("ts_feature_subspace_rotation", "recent_window", 4),
    ("ts_feature_subspace_rotation", "prior_window", 9.5),
    ("ts_beta_break_score", "recent_window", np.nan),
    ("ts_beta_break_score", "prior_window", np.inf),
])
def test_invalid_integer_scalars_rejected_by_every_backend(name, param, bad):
    p = _panel(np.arange(12.0))
    count = 2 if name == "ts_beta_break_score" else 3
    for op in _operators(name):
        panels = [_backend(p, op._r6_backend)] * count
        with pytest.raises(Exception):
            op.calculate(*panels, **{param: bad})


def test_required_panels_and_exact_axes_are_enforced():
    p = _panel(np.arange(12.0))
    shifted = _panel(np.arange(12.0), start="2024-01-03")
    for name in NAMES:
        for op in _operators(name):
            with pytest.raises(Exception):
                op.calculate()
            if op._r6_backend == "pandas_numpy":
                panels = [p] * len(op.metadata.panel_params)
                panels[-1] = shifted
                with pytest.raises(Exception):
                    op.calculate(*panels)


def test_independent_geometry_oracles_scale_extremes_and_degeneracy():
    rng = np.random.default_rng(20260914)
    n = 120
    common = rng.normal(size=n)
    identical = [_panel(common), _panel(2 * common + 7), _panel(-3 * common + 2)]
    independent = [_panel(rng.normal(size=n)) for _ in range(3)]
    for name in ("ts_feature_mode_share", "ts_feature_effective_rank"):
        for op in _operators(name):
            same = _values(op.calculate(*[_backend(p, op._r6_backend) for p in identical], 60))[-1, 0]
            indep = _values(op.calculate(*[_backend(p, op._r6_backend) for p in independent], 60))[-1, 0]
            if name.endswith("mode_share"):
                assert same > 0.99 and indep < same
            else:
                assert same < 1.05 and indep > same
            scaled = [identical[0] * 1e100, identical[1] * 1e-100, identical[2] * 1e50]
            got = _values(op.calculate(*[_backend(p, op._r6_backend) for p in scaled], 60))[-1, 0]
            np.testing.assert_allclose(got, same, rtol=1e-10, atol=1e-10)
            huge = [_panel((0.8 + 0.01 * np.arange(n)) / (0.8 + 0.01 * (n - 1)) * (np.finfo(float).max / 4)) for _ in range(3)]
            assert np.isfinite(_values(op.calculate(*[_backend(p, op._r6_backend) for p in huge], 60))[-1, 0])
            const = [_panel(np.ones(n)), _panel(np.arange(n)), _panel(np.arange(n) ** 2)]
            assert np.isnan(_values(op.calculate(*[_backend(p, op._r6_backend) for p in const], 60))[-1, 0])


def test_rotation_and_beta_break_independent_directional_oracles():
    rng = np.random.default_rng(77)
    prior, recent = 30, 20
    a = rng.normal(size=prior + recent)
    b = rng.normal(size=prior + recent)
    f1 = a
    f2 = np.r_[a[:prior] + 0.02 * b[:prior], b[prior:] + 0.02 * a[prior:]]
    f3 = np.r_[0.4 * a[:prior] + b[:prior], a[prior:] - 0.4 * b[prior:]]
    x = np.linspace(-2, 2, prior + recent)
    y = np.r_[2.0 * x[:prior], 6.0 * x[prior:]]
    panels = [_panel(f1), _panel(f2), _panel(f3)]
    for op in _operators("ts_feature_subspace_rotation"):
        value = _values(op.calculate(*[_backend(p, op._r6_backend) for p in panels], recent, prior, 0.0))[-1, 0]
        assert np.isfinite(value) and value > 0.05
    for op in _operators("ts_beta_break_score"):
        value = _values(op.calculate(_backend(_panel(y), op._r6_backend), _backend(_panel(x), op._r6_backend), recent, prior))[-1, 0]
        assert np.isfinite(value) and value > 0
