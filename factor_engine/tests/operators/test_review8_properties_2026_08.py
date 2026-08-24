# -*- coding: utf-8 -*-
"""Review-8 property tests (D / E / F / G / B).

* G  — column identity: sourceA.Close vs sourceB.Close are different data, even
       though both display "Close".  ``_same_column`` and the rolling-CSE
       semantic key must not conflate them.
* D  — axis metamorphic: instrument permutation only permutes output columns;
       relabelling group ids does not change group values.
* E  — unit/scale metamorphic: scale-invariant operators must be invariant;
       unit-algebra operators must scale according to their unit.
* B  — prefix causality: ``run(data[:t])`` == ``run(data_full).loc[t]``.
* F  — rewrite semantic equivalence: rewritten forms equal their source on
       adversarial inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

_N = 80
_IDX = pd.date_range("2024-01-01", periods=_N, freq="B")


def _mk_panel(cols, seed=1, scale=1.0, with_nan=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        scale * rng.standard_normal((_N, len(cols))), index=_IDX, columns=cols
    )
    if with_nan:
        frame.iloc[0, 0] = np.nan
        frame.iloc[3, 1] = np.nan
    return frame


def _run(canon: str, panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    op = OperatorRegistry.get(canon, backend="pandas_numpy")
    assert op is not None, f"missing {canon}"
    return op.calculate(panel.copy(), *args, **kwargs)


# ===========================================================================
# G — column identity
# ===========================================================================

def test_same_column_distinguishes_source_identity():
    """sourceA.Close vs sourceB.Close (same display name, different source)."""
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.rewrite_fastpath import _same_column

    col_a = PlanNode(op="column", attrs={
        "name": "Close", "field_id": "fa", "source_table": "sourceA",
        "source_field": "Close",
    }, inputs=[])
    col_b = PlanNode(op="column", attrs={
        "name": "Close", "field_id": "fb", "source_table": "sourceB",
        "source_field": "Close",
    }, inputs=[])
    assert not _same_column(col_a, col_b)
    # same identity -> same column
    col_a2 = PlanNode(op="column", attrs={
        "name": "Close", "field_id": "fa", "source_table": "sourceA",
        "source_field": "Close",
    }, inputs=[])
    assert _same_column(col_a, col_a2)
    # synthetic columns without source identity still compare by name
    bare1 = PlanNode(op="column", attrs={"name": "Close"}, inputs=[])
    bare2 = PlanNode(op="column", attrs={"name": "Close"}, inputs=[])
    assert _same_column(bare1, bare2)


def test_rolling_cse_semantic_key_distinguishes_sources():
    """CSE must not reuse sourceA's rolling result for sourceB."""
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.rolling_cse import rolling_semantic_key

    def _ts_mean(name, source_table, field_id):
        return PlanNode(op="ts_mean", attrs={"d": 20}, inputs=[
            PlanNode(op="column", attrs={
                "name": name, "source_table": source_table, "field_id": field_id,
                "source_field": name,
            }, inputs=[]),
        ])

    k_a = rolling_semantic_key(_ts_mean("Close", "sourceA", "fa"))
    k_b = rolling_semantic_key(_ts_mean("Close", "sourceB", "fb"))
    assert k_a is not None and k_b is not None
    assert k_a != k_b
    # same source -> same key
    k_a2 = rolling_semantic_key(_ts_mean("Close", "sourceA", "fa"))
    assert k_a == k_a2


# ===========================================================================
# D — axis metamorphic
# ===========================================================================

def test_cross_section_ops_permutation_invariance():
    """cs_rank is invariant to instrument column order (only columns permute)."""
    panel = _mk_panel(["A", "B", "C"], seed=3)
    permuted = panel[["C", "A", "B"]]
    out = _run("cs_rank", panel)
    out_p = _run("cs_rank", permuted)
    # same values, columns permuted accordingly
    assert out["A"].equals(out_p["A"])
    assert out["B"].equals(out_p["B"])
    assert out["C"].equals(out_p["C"])
    # and the permuted output has permuted column order
    assert list(out_p.columns) == ["C", "A", "B"]


def test_group_relabel_invariance():
    """Relabelling group ids (1->101, 2->202) must not change group values."""
    panel = _mk_panel(["A", "B", "C"], seed=4)
    groups = pd.DataFrame({"A": 1, "B": 2, "C": 1}, index=_IDX)
    relabel = pd.DataFrame({"A": 101, "B": 202, "C": 101}, index=_IDX)
    out1 = _run("group_mean", panel, groups)
    out2 = _run("group_mean", panel, relabel)
    pd.testing.assert_frame_equal(out1, out2)


# ===========================================================================
# E — unit / scale metamorphic
# ===========================================================================

def test_scale_invariant_ops():
    """return / rank / zscore / corr are invariant to a price-level rescale."""
    x = _mk_panel(["A", "B"], seed=5)
    x10 = _mk_panel(["A", "B"], seed=5) * 10.0

    def _cols_equal(a, b):
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-9)

    _cols_equal(_run("cs_rank", x), _run("cs_rank", x10))

    # ts_log_return uses ratios -> level invariant
    lr1 = _run("ts_log_return", x)
    lr2 = _run("ts_log_return", x10)
    _cols_equal(lr1, lr2)

    # zscore is (x-mean)/std -> level invariant
    z1 = _run("ts_zscore", x, window=20)
    z2 = _run("ts_zscore", x10, window=20)
    _cols_equal(z1, z2)


def test_unit_algebra_scaling():
    """Level operators scale by the factor; covariance scales once."""
    x = _mk_panel(["A", "B"], seed=6)
    x10 = _mk_panel(["A", "B"], seed=6) * 10.0

    # price level scales linearly
    m1 = _run("ts_mean", x, window=20)
    m2 = _run("ts_mean", x10, window=20)
    np.testing.assert_allclose(m2.to_numpy(), 10.0 * m1.to_numpy(), rtol=1e-6)

    # ts_cov scales by the product of the two scales (10 * 1 = 10)
    y = _mk_panel(["A", "B"], seed=7)
    cov1 = _run("ts_cov", x, y, window=20)
    cov2 = _run("ts_cov", x10, y, window=20)
    np.testing.assert_allclose(cov2.to_numpy(), 10.0 * cov1.to_numpy(), rtol=1e-6)


# ===========================================================================
# B — prefix causality (no future leakage)
# ===========================================================================

@pytest.mark.parametrize("canon", ["ts_mean", "ts_rank", "ts_std"])
def test_prefix_causality_trailing_ops(canon):
    """For trailing-window ops, computing on data[:t] and taking the last row
    equals the full-run value at t (no future leakage)."""
    panel = _mk_panel(["A", "B"], seed=8)
    full = _run(canon, panel, window=20)
    for t in (10, 20, 40, 70):
        prefix = _run(canon, panel.iloc[:t], window=20)
        np.testing.assert_allclose(
            prefix.iloc[-1].to_numpy(), full.iloc[t - 1].to_numpy(), rtol=1e-8, atol=1e-9
        )


# ===========================================================================
# F — rewrite semantic equivalence
# ===========================================================================

def test_zscore_rewrite_equivalence():
    """(col - ts_mean) / ts_std == ts_zscore on adversarial inputs."""
    panel = _mk_panel(["A", "B"], seed=9, with_nan=True)
    # constant + zero-std column
    panel["C"] = 5.0

    z = _run("ts_zscore", panel, window=20)
    mean = _run("ts_mean", panel, window=20)
    std = _run("ts_std", panel, window=20)
    manual = (panel - mean) / std
    # compare finite positions
    for col in panel.columns:
        m = z[col].notna() & manual[col].notna()
        np.testing.assert_allclose(
            z[col][m].to_numpy(), manual[col][m].to_numpy(), rtol=1e-6, atol=1e-9
        )


def test_log_return_rewrite_equivalence():
    """ts_log_return == ln(x_t / x_{t-1}) exactly."""
    panel = _mk_panel(["A", "B"], seed=10, with_nan=True)
    lr = _run("ts_log_return", panel)
    manual = np.log(panel / panel.shift(1))
    pd.testing.assert_frame_equal(lr, manual, check_exact=False, rtol=1e-6, atol=1e-9)
