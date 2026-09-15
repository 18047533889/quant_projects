from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.gather_ext import (
    _cs_weighted_percentile_rank,
    _event_level_survival_share,
    _group_distribution_js_divergence,
    _group_topk_mean,
    _ts_value_at_argextreme,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "group_topk_mean", "ts_value_at_argextreme",
    "cs_weighted_percentile_rank", "group_distribution_js_divergence",
    "event_level_survival_share",
)


def _pd(values):
    return pd.DataFrame([values], columns=[f"c{i}" for i in range(len(values))], dtype=float)


def test_topk_fractional_ties_and_overflow_safe_mean():
    target = _pd([1e308, 1e308, -1e308, 0.0])
    score = _pd([3.0, 3.0, 3.0, 0.0])
    group = pd.DataFrame([["g"] * 4], columns=target.columns)
    out = _group_topk_mean(target, score, group, k=2, exclude_self=False)
    assert np.isfinite(out.to_numpy()).all()
    assert out.iloc[0, 0] == pytest.approx(1e308 / 3.0)
    with pytest.raises(Exception):
        _group_topk_mean(target, score, group, k=2.5)


@pytest.mark.parametrize("factor", [1e-300, 1e300])
def test_weighted_rank_scale_invariance_without_absolute_eps(factor):
    x = _pd([1.0, 2.0, 2.0, 4.0])
    weight = _pd(np.array([1.0, 2.0, 3.0, 4.0]) * factor)
    expected = np.array([[0.05, 0.35, 0.35, 0.80]])
    np.testing.assert_allclose(_cs_weighted_percentile_rank(x, weight), expected)


def test_argextreme_latest_tie_and_excluded_current_history():
    idx = pd.RangeIndex(5)
    value = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)
    score = pd.DataFrame({"A": [1.0, 9.0, 9.0, 2.0, 100.0]}, index=idx)
    out = _ts_value_at_argextreme(value, score, window=4, mode="max", include_current=False)
    assert out.iloc[-1, 0] == 30.0
    with pytest.raises(Exception):
        _ts_value_at_argextreme(value, score, window=3.5)


def test_survival_unknown_path_is_censored_until_age_out():
    idx = pd.RangeIndex(6)
    event = pd.DataFrame({"A": [1.0, 0, 0, 0, 0, 0]}, index=idx)
    level = pd.DataFrame({"A": [10.0] * 6}, index=idx)
    x = pd.DataFrame({"A": [10.0, 11.0, np.nan, 12.0, 12.0, 12.0]}, index=idx)
    out = _event_level_survival_share(event, level, x, history_window=3)
    assert out.iloc[1, 0] == 1.0
    assert out.iloc[2:4, 0].isna().all()
    assert np.isnan(out.iloc[4, 0])  # cohort ages out; never resurrected
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(Exception):
            _event_level_survival_share(event, level, x, tolerance=bad)


def test_survival_censoring_is_per_cohort_not_whole_window_nan():
    idx = pd.RangeIndex(6)
    # The row-0 cohort crosses an unknown observation at row 2.  A fresh event
    # at row 3 has a fully observed path and must remain estimable even while
    # the older censored cohort is still inside history_window.
    event = pd.DataFrame({"A": [1.0, 0, 0, 1.0, 0, 0]}, index=idx)
    level = pd.DataFrame({"A": [10.0] * 6}, index=idx)
    x = pd.DataFrame({"A": [10.0, 11.0, np.nan, 10.0, 11.0, 12.0]}, index=idx)
    out = _event_level_survival_share(event, level, x, history_window=5)
    assert np.isnan(out.iloc[2, 0])
    assert out.iloc[4, 0] == 1.0
    assert out.iloc[5, 0] == 1.0


def test_js_ex_group_reference_and_integer_contract():
    values = list(range(24))
    x = _pd(values)
    labels = ["a"] * 4 + ["b"] * 20
    group = pd.DataFrame([labels], columns=x.columns)
    out = _group_distribution_js_divergence(
        x, group, bins=4, min_group_size=4, exclude_group_from_reference=True
    )
    assert np.isfinite(out.iloc[0, :4]).all()
    assert (out.iloc[0, :4] > 0).all()
    with pytest.raises(Exception):
        _group_distribution_js_divergence(x, group, bins=4.5)


def test_default_registry_contracts_and_polars_keyword_calls():
    ensure_cleaned_loaded()
    expected_panels = {
        "group_topk_mean": ("target", "score", "group"),
        "ts_value_at_argextreme": ("value", "score"),
        "cs_weighted_percentile_rank": ("x", "weight"),
        "group_distribution_js_divergence": ("x", "group"),
        "event_level_survival_share": ("event", "level", "x"),
    }
    for name in NAMES:
        pandas_op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
        polars_op = OperatorRegistry.get(name, "polars", mode="any")
        assert pandas_op.metadata.panel_params == expected_panels[name]
        assert polars_op.metadata.panel_params == expected_panels[name]
        assert pandas_op.metadata.param_specs == polars_op.metadata.param_specs
        assert polars_op._physical_spec.execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE
        assert polars_op._physical_spec.is_production_eligible() is False

    target = pl.DataFrame({"A": [1e308], "B": [1e308], "C": [-1e308]})
    score = pl.DataFrame({"A": [3.0], "B": [3.0], "C": [3.0]})
    group = pl.DataFrame({"A": ["g"], "B": ["g"], "C": ["g"]})
    op = OperatorRegistry.get("group_topk_mean", "polars", mode="any")
    got = op.calculate(target=target, score=score, group=group, k=2, exclude_self=False)
    assert np.isfinite(got["A"][0])


def test_fresh_default_bootstrap_selects_callable_final_owners():
    code = r'''\
import json
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
ensure_cleaned_loaded()
names = ("group_topk_mean", "ts_value_at_argextreme", "cs_weighted_percentile_rank", "group_distribution_js_divergence", "event_level_survival_share")
print(json.dumps([{"name": n, "module": type(OperatorRegistry.get(n, "polars", mode="any")).__module__, "params": OperatorRegistry.get(n, "polars", mode="any").metadata.param_names} for n in names]))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ), text=True,
        capture_output=True, check=True,
    )
    rows = json.loads(completed.stdout.strip().splitlines()[-1])
    assert [row["name"] for row in rows] == list(NAMES)
    assert all(row["params"] for row in rows)
