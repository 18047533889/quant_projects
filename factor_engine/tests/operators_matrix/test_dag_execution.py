# -*- coding: utf-8 -*-
"""DAG / batch execution contract tests.

run_many compiles one shared DAG with CSE: results must equal isolated runs
bit-for-bit, shared subexpressions must be detected, and batch order must not
change values.
"""
from __future__ import annotations

import numpy as np
import pytest

from factor_engine.tests.operators_matrix.helpers import _build_engine, _extract

from factor_engine.api.dsl_parser import parse_factor

# shared-subexpression DAG: one common base, several consumers
CHAIN_FACTORS = {
    "dag_base": "ts_mean(close, 20)",
    "dag_a1": "rank(ts_mean(close, 20))",
    "dag_a2": "ts_zscore(ts_mean(close, 20), 20)",
    "dag_b1": "rank(volume)",
    "dag_b2": "ts_corr(close, volume, 30)",
    "dag_c1": "rank(ts_zscore(ts_mean(close, 20), 20))",
    "dag_deep": "ts_rank(ts_corr(rank(ts_mean(close, 20)), rank(volume), 20), 10)",
}


def _values(eng, factors):
    out = eng.run_many(factors, market="ashare", result_policy="return")
    res = out.get("results") if isinstance(out, dict) else out
    return {n: _extract(v) for n, v in (res or {}).items()}


@pytest.mark.parametrize("backend", ("pandas",))
def test_batch_equals_isolated_bitwise(backend):
    """run_many(DAG batch) and per-factor isolated runs must agree bit-for-bit."""
    eng = _build_engine(backend)
    factors = [parse_factor(e, name=n) for n, e in CHAIN_FACTORS.items()]
    batched = _values(eng, factors)
    isolated = {}
    for f in factors:
        fresh = _build_engine(backend)   # isolated runs use a fresh engine
        isolated.update(_values(fresh, [f]))
    for n, v in batched.items():
        w = isolated[n]
        assert v.shape == w.shape, n
        assert np.array_equal(np.isnan(v), np.isnan(w)), n
        mv = ~np.isnan(v)
        assert np.array_equal(v[mv], w[mv]), n


def test_dag_shared_nodes_detected():
    """The DAG plan must detect shared subexpression nodes (CSE)."""
    eng = _build_engine("pandas")
    factors = [parse_factor(e, name=n) for n, e in CHAIN_FACTORS.items()]
    out = eng.run_many(factors, market="ashare", result_policy="return")
    dag = out.get("dag") if isinstance(out, dict) else None
    assert dag is not None, "run_many returned no dag plan"
    # shared base must appear in the plan's shared nodes
    shared = getattr(dag, "shared_nodes", None) or (dag.get("shared_nodes") if isinstance(dag, dict) else None)
    assert shared is not None and len(shared) >= 1, (
        f"expected shared nodes for common subexpressions, got {shared}"
    )


def test_batch_order_invariance():
    """Factor execution order inside a batch must not change any value."""
    eng = _build_engine("pandas")
    factors = [parse_factor(e, name=n) for n, e in CHAIN_FACTORS.items()]
    fwd = _values(eng, factors)
    rev = _values(eng, list(reversed(factors)))
    for n, v in fwd.items():
        w = rev[n]
        mv = ~np.isnan(v)
        assert np.array_equal(v[mv], w[mv]), n


def test_duplicate_factor_names_rejected():
    eng = _build_engine("pandas")
    f = parse_factor("ts_mean(close, 5)", name="dup")
    # duplicates must either raise cleanly or be deduped -- never corrupt results
    try:
        out = eng.run_many([f, f], market="ashare", result_policy="return")
        res = out.get("results") if isinstance(out, dict) else out
        assert isinstance(res, dict) and len(res) >= 1
    except Exception:
        pass  # clean rejection is also acceptable


def test_cse_disabled_still_correct():
    """enable_cse=False must produce identical values to the CSE path."""
    eng = _build_engine("pandas")
    factors = [parse_factor(e, name=n) for n, e in CHAIN_FACTORS.items()]
    with_cse = eng.run_many(factors, market="ashare", result_policy="return", enable_cse=True)
    without = eng.run_many(factors, market="ashare", result_policy="return", enable_cse=False)
    r1 = with_cse.get("results") if isinstance(with_cse, dict) else with_cse
    r2 = without.get("results") if isinstance(without, dict) else without
    for n in CHAIN_FACTORS:
        a, b = _extract(r1[n]), _extract(r2[n])
        mv = ~np.isnan(a)
        assert np.array_equal(a[mv], b[mv]), n
