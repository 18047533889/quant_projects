from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

ensure_cleaned_loaded()

from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = ("ts_l_skewness", "ts_l_kurtosis", "ts_hartigan_dip")


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend=backend)
    assert op is not None
    return op


def _oracle_l_ratios(values: np.ndarray) -> tuple[float, float]:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    b = []
    for r in range(4):
        b.append(sum(comb(i, r) / comb(n - 1, r) * x[i] for i in range(r, n)) / n)
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return l3 / l2, l4 / l2


def _last(name: str, values, backend="pandas_numpy", **kwargs) -> float:
    panel = pd.DataFrame({"A": values}, dtype=float)
    if backend == "polars":
        panel = pl.from_pandas(panel)
    out = _op(name, backend).calculate(panel, **kwargs)
    if isinstance(out, pl.DataFrame):
        out = out.to_pandas()
    return float(out.iloc[-1, 0])


def test_complete_contracts_and_polars_metadata_parity():
    defaults = {
        "ts_l_skewness": (60, 20, .5, 4),
        "ts_l_kurtosis": (60, 20, .5, 4),
        "ts_hartigan_dip": (120, 20, .8, 2),
    }
    for name, (window, periods, coverage, minimum) in defaults.items():
        pandas_meta = _op(name).metadata
        polars_meta = _op(name, "polars").metadata
        assert pandas_meta == polars_meta
        assert pandas_meta.panel_params == ("x",)
        assert pandas_meta.panel_arity == 1
        assert pandas_meta.scalar_params == ("window", "min_periods", "min_coverage_fraction")
        assert set(pandas_meta.param_specs) == set(pandas_meta.scalar_params)
        specs = pandas_meta.param_specs
        assert (specs["window"].default, specs["window"].min) == (window, minimum)
        assert specs["window"].history_semantics == "max_rows"
        assert (specs["min_periods"].default, specs["min_periods"].min) == (periods, minimum)
        assert specs["min_coverage_fraction"].default == coverage
        assert pandas_meta.output_unit == "dimensionless"
        assert [r.expression for r in pandas_meta.relational_specs] == ["min_periods <= window"]


def test_l_moments_match_independent_combinatorial_pwm_oracle():
    values = np.array([-7., -2., -1., 0., 1., 3., 4., 11.])
    expected_skew, expected_kurt = _oracle_l_ratios(values)
    kwargs = {"window": len(values), "min_periods": len(values), "min_coverage_fraction": 1.0}
    assert _last("ts_l_skewness", values, **kwargs) == pytest.approx(expected_skew, abs=1e-14)
    assert _last("ts_l_kurtosis", values, **kwargs) == pytest.approx(expected_kurt, abs=1e-14)


@pytest.mark.parametrize("name", ["ts_l_skewness", "ts_l_kurtosis"])
def test_l_ratio_scale_invariance_for_tiny_and_huge_finite_values(name):
    values = np.array([-5., -2., -.5, 0., 1., 4., 9., 15.])
    kwargs = {"window": 8, "min_periods": 8, "min_coverage_fraction": 1.0}
    base = _last(name, values, **kwargs)
    assert _last(name, values * 1e-250, **kwargs) == pytest.approx(base, abs=2e-14)
    assert _last(name, values * 1e250, **kwargs) == pytest.approx(base, abs=2e-14)
    symmetric = np.array([-9., -4., -2., -1., 1., 2., 4., 9.])
    assert abs(_last("ts_l_skewness", symmetric, **kwargs)) < 1e-14


def test_hartigan_dip_affine_scale_invariance_and_bounds():
    values = np.r_[np.linspace(-5., -3., 12), np.linspace(4., 7., 12)]
    kwargs = {"window": 24, "min_periods": 24, "min_coverage_fraction": 1.0}
    base = _last("ts_hartigan_dip", values, **kwargs)
    shifted_scaled = _last("ts_hartigan_dip", values * 1e220 + 2e220, **kwargs)
    tiny = _last("ts_hartigan_dip", values * 1e-220, **kwargs)
    assert 0.0 <= base <= .25
    assert shifted_scaled == pytest.approx(base, abs=2e-14)
    assert tiny == pytest.approx(base, abs=2e-14)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_domain_rejection_prefix_causality_and_default_parity(backend):
    values = np.linspace(-3., 4., 40) + np.sin(np.arange(40))
    panel = pd.DataFrame({"B": values, "A": values[::-1]}, index=pd.date_range("2026-01-01", periods=40))
    longer = pd.concat([panel, pd.DataFrame({"B": [1e100], "A": [-1e100]}, index=[pd.Timestamp("2026-02-10")])])
    if backend == "polars":
        panel_in = pl.from_pandas(panel.reset_index(drop=True))
        longer_in = pl.from_pandas(longer.reset_index(drop=True))
    else:
        panel_in, longer_in = panel, longer
    for name in NAMES:
        op = _op(name, backend)
        first = op.calculate(panel_in)
        extended = op.calculate(longer_in)
        if backend == "polars":
            first, extended = first.to_pandas(), extended.to_pandas()
            pd.testing.assert_frame_equal(first, extended.iloc[:-1].reset_index(drop=True))
        else:
            pd.testing.assert_frame_equal(first, extended.iloc[:-1], check_freq=False)
            assert first.index.equals(panel.index) and first.columns.equals(panel.columns)
        minimum = 4 if name.startswith("ts_l_") else 2
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel_in, window=minimum - 1, min_periods=minimum)
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel_in, window=8, min_periods=9)
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel_in, min_coverage_fraction=0.0)
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel_in, min_coverage_fraction=np.inf)
        with pytest.raises((TypeError, ValueError)):
            op.calculate(panel_in, window=8.5, min_periods=minimum)

    if backend == "polars":
        for name in NAMES:
            p = _op(name).calculate(panel.reset_index(drop=True))
            q = _op(name, "polars").calculate(pl.from_pandas(panel.reset_index(drop=True))).to_pandas()
            pd.testing.assert_frame_equal(p, q)
            assert list(q.columns) == ["B", "A"]

def test_polars_registry_uses_exact_declared_delegates():
    expected = {
        "ts_l_skewness": "MomentsPolars",
        "ts_l_kurtosis": "MomentsPolars",
        "ts_hartigan_dip": "MomentsPolars",
    }
    for name, class_name in expected.items():
        op = _op(name, "polars")
        assert type(op).__name__ == class_name
        assert op.physical_spec().execution_kind.value == "polars_pandas_delegate"
