# -*- coding: utf-8 -*-
"""R11 round-3 audit tests for prospect-theory and robust-stats fixes.

Covers:
* #118  ``ts_cpt_value`` trailing-window coverage gate (default 0.8).
* #119  ``ts_cpt_value`` output unit is ``behavioral_score``, not ``level``.
* #120  ``ts_quantile_range`` output unit is ``same_as:x``, not ``ratio``.
* #121  ``ts_trimmed_mean`` output unit is ``same_as:x``, not ``ratio``.
* #122  ``ts_robust_zscore`` split into ``ts_robust_zscore_inclusive`` (old
        behaviour) + ``ts_robust_zscore_prior`` (baseline on ``[t-W, t-1]``);
        ``ts_robust_zscore`` stays a back-compat alias of the inclusive kernel.
* #123  robust-stats default ``min_periods`` >= ``max(5, 0.5*window)`` for the
        searchable canonicals (knob declared ``searchable=False``).
* pandas/polars parity for every touched operator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import ParamRole
from factor_engine.cleaned_operators.operator_surface import classify_canonical
from factor_engine.cleaned_operators.registry import OperatorRegistry

pl = pytest.importorskip("polars")


def _frame(rows: int = 70, cols: int = 3, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.5, 2.0, (rows, cols)), index=idx, columns=["A", "B", "C"][:cols])


def _returns(rows: int = 70, cols: int = 2, seed: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 0.01, (rows, cols)), index=idx, columns=["A", "B"][:cols])


def _pl(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({c: frame[c].to_numpy() for c in frame.columns})


def _unit_tag(operator) -> str | None:
    for tag in getattr(operator.metadata, "tags", None) or []:
        if str(tag).startswith("unit:"):
            return str(tag)[len("unit:"):]
    return None


# ---------------------------------------------------------------------------
# #118 CPT coverage gate
# ---------------------------------------------------------------------------

def test_ts_cpt_value_coverage_gate_sparse_sample_is_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    r = pd.DataFrame(np.nan, index=idx, columns=["A"])
    r.iloc[10:18, 0] = np.random.default_rng(3).normal(0.0, 0.01, 8)
    out = OperatorRegistry.get("ts_cpt_value").calculate(r, window=60)
    # 8 valid of a 60-window = 0.133 coverage < 0.8 -> NaN (no 1/8-probability
    # decision weights compared against full-60 peers).
    assert np.isnan(out.iloc[-1, 0])


def test_ts_cpt_value_coverage_gate_mature_window_finite() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    r = pd.DataFrame(np.random.default_rng(3).normal(0.0, 0.01, (60, 1)), index=idx, columns=["A"])
    r.iloc[0:10, 0] = np.nan  # 50 valid in the 60-window = 0.833 coverage
    out = OperatorRegistry.get("ts_cpt_value").calculate(r, window=60)
    assert np.isfinite(out.iloc[-1, 0])
    # Raising the bar to 0.9 fails the same window closed.
    strict = OperatorRegistry.get("ts_cpt_value").calculate(r, window=60, min_coverage=0.9)
    assert np.isnan(strict.iloc[-1, 0])


def test_ts_cpt_value_min_coverage_validation() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    r = pd.DataFrame(np.zeros((30, 1)), index=idx, columns=["A"])
    op = OperatorRegistry.get("ts_cpt_value")
    with pytest.raises(ValueError, match="min_coverage"):
        op.calculate(r, window=20, min_coverage=0.0)
    with pytest.raises(ValueError, match="min_coverage"):
        op.calculate(r, window=20, min_coverage=1.5)


def test_ts_cpt_value_min_coverage_is_policy_knob() -> None:
    op = OperatorRegistry.get("ts_cpt_value")
    spec = op.metadata.param_specs.get("min_coverage")
    assert spec is not None
    assert spec.searchable is False
    assert spec.param_role is ParamRole.POLICY
    assert spec.default == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# #119 / #120 / #121 output units
# ---------------------------------------------------------------------------

def test_ts_cpt_value_output_unit_behavioral_score() -> None:
    op = OperatorRegistry.get("ts_cpt_value")
    assert _unit_tag(op) == "behavioral_score"
    assert op.metadata.output_unit == "behavioral_score"
    assert OperatorRegistry._catalog["ts_cpt_value"].get("output_unit") == "behavioral_score"


def test_ts_quantile_range_output_unit_same_as_x() -> None:
    op = OperatorRegistry.get("ts_quantile_range")
    assert _unit_tag(op) == "same_as:x"
    assert op.metadata.output_unit == "same_as:x"
    assert OperatorRegistry._catalog["ts_quantile_range"].get("output_unit") == "same_as:x"


def test_ts_trimmed_mean_output_unit_same_as_x() -> None:
    op = OperatorRegistry.get("ts_trimmed_mean")
    assert _unit_tag(op) == "same_as:x"
    assert op.metadata.output_unit == "same_as:x"
    assert OperatorRegistry._catalog["ts_trimmed_mean"].get("output_unit") == "same_as:x"


# ---------------------------------------------------------------------------
# #122 ts_robust_zscore split
# ---------------------------------------------------------------------------

def test_ts_robust_zscore_split_registered() -> None:
    assert OperatorRegistry.resolve_canonical("ts_robust_zscore") == "ts_robust_zscore_inclusive"
    for name in ("ts_robust_zscore_inclusive", "ts_robust_zscore_prior"):
        op = OperatorRegistry.get(name)
        assert op is not None, name
        assert classify_canonical(name) == "extended", name
    # Both canonicals carry both backends.
    assert set(OperatorRegistry.backends_for("ts_robust_zscore_inclusive")) == {"pandas_numpy", "polars"}
    assert set(OperatorRegistry.backends_for("ts_robust_zscore_prior")) == {"pandas_numpy", "polars"}


def test_ts_robust_zscore_alias_keeps_old_behaviour() -> None:
    x = _frame(rows=60, seed=7)
    alias = OperatorRegistry.get("ts_robust_zscore").calculate(x, window=20)
    inclusive = OperatorRegistry.get("ts_robust_zscore_inclusive").calculate(x, window=20)
    pd.testing.assert_frame_equal(alias, inclusive, check_dtype=False)


def test_ts_robust_zscore_prior_excludes_current_row() -> None:
    x = _frame(rows=60, seed=7)
    x.iloc[30, 0] = 1000.0  # extreme current value
    prior = OperatorRegistry.get("ts_robust_zscore_prior").calculate(x, window=20)
    inclusive = OperatorRegistry.get("ts_robust_zscore_inclusive").calculate(x, window=20)
    # Row 0 has an empty prior window -> NaN (prior baseline is never contaminated
    # by the row it scores).
    assert np.isnan(prior.iloc[0, 0])
    # The extreme is scored against a CLEAN prior baseline -> its z-score is more
    # extreme than the inclusive one (whose MAD is inflated by the outlier).
    assert abs(prior.iloc[30, 0]) > abs(inclusive.iloc[30, 0])


def test_ts_robust_zscore_prior_min_periods_knob() -> None:
    op = OperatorRegistry.get("ts_robust_zscore_prior")
    spec = op.metadata.param_specs.get("min_periods")
    assert spec is not None
    assert spec.searchable is False
    assert spec.param_role is ParamRole.ESTIMATOR_RESOLUTION


# ---------------------------------------------------------------------------
# #123 robust-stats default min_periods
# ---------------------------------------------------------------------------

def test_ts_quantile_range_default_min_periods_not_loose() -> None:
    x = _frame(rows=30, seed=11)
    out = OperatorRegistry.get("ts_quantile_range").calculate(x, window=20)
    # Default min_periods = max(5, 10) = 10: a 5-observation warm-up row is NaN.
    assert np.isnan(out.iloc[4, 0])
    assert np.isfinite(out.iloc[-1, 0])
    # Explicit permissive min_periods restores the early finite value.
    loose = OperatorRegistry.get("ts_quantile_range").calculate(x, window=20, min_periods=2)
    assert np.isfinite(loose.iloc[4, 0])


def test_ts_trimmed_mean_default_min_periods_not_loose() -> None:
    x = _frame(rows=30, seed=12)
    out = OperatorRegistry.get("ts_trimmed_mean").calculate(x, window=20)
    assert np.isnan(out.iloc[4, 0])
    assert np.isfinite(out.iloc[-1, 0])


def test_robust_stats_min_periods_knobs_are_non_searchable() -> None:
    for name in ("ts_quantile_range", "ts_trimmed_mean", "ts_robust_zscore_prior"):
        spec = OperatorRegistry.get(name).metadata.param_specs.get("min_periods")
        assert spec is not None, name
        assert spec.searchable is False, name
        assert spec.param_role is ParamRole.ESTIMATOR_RESOLUTION, name


# ---------------------------------------------------------------------------
# pandas/polars parity for every touched operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("ts_quantile_range", {"window": 20}),
        ("ts_trimmed_mean", {"window": 20}),
        ("ts_robust_zscore_inclusive", {"window": 20}),
        ("ts_robust_zscore", {"window": 20}),
        ("ts_robust_zscore_prior", {"window": 20}),
        ("ts_robust_zscore_prior", {"window": 20, "min_periods": 3}),
        ("ts_cpt_value", {"window": 60}),
        ("ts_cpt_value", {"window": 60, "min_coverage": 0.9}),
    ],
)
def test_touched_operators_polars_parity(name, kwargs) -> None:
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    if name == "ts_cpt_value":
        x = _returns(rows=70, cols=2, seed=21)
        x.iloc[0:12, 0] = np.nan
    else:
        x = _frame(rows=70, cols=3, seed=22)
        x.iloc[5, 1] = np.nan
        x.iloc[30, 0] = 1000.0
    pandas_out = pandas_op.calculate(x, **kwargs)
    polars_out = polars_op.calculate(_pl(x), **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )
