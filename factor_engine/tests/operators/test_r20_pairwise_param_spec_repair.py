# -*- coding: utf-8 -*-
"""R20 pairwise ParamSpec repair oracles — ts_ewm_corr / ts_ewm_cov."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import polars as pl

def _ensure_chain():
    from cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry.lifecycle() == "frozen":
        return
    from cleaned_operators import load_all
    load_all()

@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_chain()

def _op(name):
    from cleaned_operators.registry import OperatorRegistry
    # ts_ewm_corr/ts_ewm_cov only have polars/sql backends, no pandas_numpy.
    op = OperatorRegistry.get(name, "polars") or OperatorRegistry.get(name, "sql")
    assert op is not None, f"{name} not registered"
    return op

def _to_polars(df):
    from backend.panel_polars import panel_to_polars
    return panel_to_polars(df)

def _span_key(op):
    # Review P2: a silent "span else window" probe masks naming drift.  The
    # pairwise EWM contract is intentionally split across backends — pandas
    # ewm_corr/ewm_cov expose `span`, the polars TSEwm* natives expose
    # `window` (kernel signature: strict_integer(window, ...)).  Assert the
    # resolved key is exactly one of the two contracted names and that it is
    # a declared param of this operator, so any third name fails loudly.
    specs = op.metadata.param_specs
    keys = [k for k in ("span", "window") if k in specs]
    assert keys, (
        f"{op.metadata.name} param_specs keys={sorted(specs)} — expected "
        "'span' or 'window' (naming regression must fail loudly here)"
    )
    assert len(keys) == 1, (
        f"{op.metadata.name} declares both {keys} — ambiguous horizon knob"
    )
    names = set(getattr(op.metadata, "param_names", None) or [])
    assert keys[0] in names, (
        f"{op.metadata.name}: spec key {keys[0]!r} missing from param_names {sorted(names)}"
    )
    return keys[0]

class TestParamSpecExistence:
    def test_ts_ewm_corr_has_param_specs(self):
        op = _op("ts_ewm_corr")
        assert hasattr(op.metadata, "param_specs")
        specs = op.metadata.param_specs
        span_spec = specs[_span_key(op)]
        assert span_spec.dtype is int
        assert span_spec.min == 2

    def test_ts_ewm_cov_has_param_specs(self):
        op = _op("ts_ewm_cov")
        assert hasattr(op.metadata, "param_specs")
        specs = op.metadata.param_specs
        span_spec = specs[_span_key(op)]
        assert span_spec.dtype is int
        assert span_spec.min == 2


class TestSpanValidation:
    def test_ts_ewm_corr_span_1_rejected(self):
        op = _op("ts_ewm_corr")
        rng = np.random.default_rng(31)
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        with pytest.raises((ValueError, TypeError)):
            op.calculate(_to_polars(x), _to_polars(y), span=1)

    def test_ts_ewm_cov_span_1_rejected(self):
        op = _op("ts_ewm_cov")
        rng = np.random.default_rng(32)
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        with pytest.raises((ValueError, TypeError)):
            op.calculate(_to_polars(x), _to_polars(y), span=1)

    def test_ts_ewm_corr_span_2_valid(self):
        rng = np.random.default_rng(21)
        op = _op("ts_ewm_corr")
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        result = op.calculate(_to_polars(x), _to_polars(y), span=2)
        assert not result.null_count().get_column("A")[0] == 20
        valid = result.drop_nulls()
        # pandas ewm corr can numerically exceed [-1, 1] by ~1e-16 — compare
        # with epsilon, never exactly (review: unseeded RNG + exact bound made
        # this test order-dependent flaky).
        assert (valid["A"].abs() <= 1.0 + 1e-9).all()

    def test_ts_ewm_cov_span_2_valid(self):
        rng = np.random.default_rng(33)
        op = _op("ts_ewm_cov")
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        result = op.calculate(_to_polars(x), _to_polars(y), span=2)
        assert not result.null_count().get_column("A")[0] == 20

    def test_ts_ewm_corr_span_non_int_rejected(self):
        op = _op("ts_ewm_corr")
        rng = np.random.default_rng(34)
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        with pytest.raises((ValueError, TypeError)):
            op.calculate(_to_polars(x), _to_polars(y), span=1.5)

    def test_ts_ewm_cov_span_non_int_rejected(self):
        op = _op("ts_ewm_cov")
        rng = np.random.default_rng(35)
        x = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 20)})
        with pytest.raises((ValueError, TypeError)):
            op.calculate(_to_polars(x), _to_polars(y), span=1.5)

    def test_pandas_ewm_corr_cov_span_2_valid(self):
        # Review P2: the focused module never exercised the pandas ewm_corr /
        # ewm_cov path — the same registration whose string param_specs crashed
        # strict validation (the P0).  Pin it directly.
        from cleaned_operators.registry import OperatorRegistry
        rng = np.random.default_rng(36)
        x = pd.DataFrame({"A": rng.normal(0, 1, 30)})
        y = pd.DataFrame({"A": rng.normal(0, 1, 30)})
        for name in ("ewm_corr", "ewm_cov"):
            op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name, mode="any")
            assert op is not None, f"{name} not registered"
            specs = getattr(op.metadata, "param_specs", None) or {}
            # every declared spec must be a real ParamSpec — string placeholders
            # here were the P0 crash
            from cleaned_operators.base import ParamSpec
            bad = {k: type(v).__name__ for k, v in specs.items() if not isinstance(v, ParamSpec)}
            assert not bad, f"{name} param_specs non-ParamSpec entries: {bad}"
            out = op.calculate(x, y, span=2)
            arr = np.asarray(out, dtype=float).ravel()
            assert np.isfinite(arr[~np.isnan(arr)]).all()
            if name == "ewm_corr":
                finite = arr[np.isfinite(arr)]
                assert (np.abs(finite) <= 1.0 + 1e-9).all()


class TestDegenerateAndEdgeCases:
    def test_ts_ewm_corr_identical_inputs(self):
        rng = np.random.default_rng(37)
        op = _op("ts_ewm_corr")
        x = pd.DataFrame({"A": rng.normal(0, 1, 50)})
        result = op.calculate(_to_polars(x), _to_polars(x.copy()), span=10)
        valid = result.drop_nulls()
        # identical inputs must give corr ≈ 1; 1e-9 eps (not 1e-10) — pandas
        # ewm corr can exceed the bound by float epsilon (review P2 flake).
        assert (valid["A"] <= 1.0 + 1e-9).all()
        assert (valid["A"] >= -1.0 - 1e-9).all()

    def test_ts_ewm_corr_nan_in_window(self):
        op = _op("ts_ewm_corr")
        x = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
        y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        result = op.calculate(_to_polars(x), _to_polars(y), span=3)
        assert result["A"][0] is None
        assert result["A"][1] is None


class TestRegistration:
    def test_ts_ewm_corr_registered(self):
        from cleaned_operators.registry import OperatorRegistry
        op = OperatorRegistry.get("ts_ewm_corr", "polars")
        assert op is not None

    def test_ts_ewm_cov_registered(self):
        from cleaned_operators.registry import OperatorRegistry
        op = OperatorRegistry.get("ts_ewm_cov", "polars")
        assert op is not None

    def test_operator_policy_alignment(self):
        from cleaned_operators.operator_policy import infer_operator_policy
        corr_policy = infer_operator_policy(None, canonical="ts_ewm_corr")
        cov_policy = infer_operator_policy(None, canonical="ts_ewm_cov")
        assert corr_policy.min_periods == 2
        assert cov_policy.min_periods == 2
