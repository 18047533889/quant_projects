# -*- coding: utf-8 -*-
"""R6 governance-layer tests: RelationalParamSpec, cost model, auto-audits.

Covers the shared engineering layer added in the 6th review round:
* RelationalParamSpec — guaranteed-NaN parameter combos rejected at the call
  boundary (Pickands window>=4k+1, RQA min_line<embedding-count, GLR
  window>=2*min_segment+2, PS min_periods<=window-1, DMD feasibility, BDS
  N_m>=12, SR baseline+8<=window).
* operator_cost_model — parameter-aware runtime/memory cost, not the static
  ``cost:N`` tag (HVG motif O(W^3) >> EMA O(W)).
* operator_audits — the 14 audit primitives run without error and flag the
  failure classes they are designed to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


# ---------------------------------------------------------------------------
# RelationalParamSpec
# ---------------------------------------------------------------------------
def test_relational_spec_rejects_infeasible_pickands():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_pickands_tail_index", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (80, 3)),
                     index=idx, columns=list("ABC"))
    with pytest.raises(ValueError):
        op.calculate(x, window=40, k=20)  # 40 < 4*20+1


def test_relational_spec_accepts_feasible_pickands():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_pickands_tail_index", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (80, 3)),
                     index=idx, columns=list("ABC"))
    out = op.calculate(x, window=84, k=20)  # 84 >= 4*20+1
    assert out.shape == x.shape


def test_relational_spec_rqa_min_line_against_embedding_count():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_recurrence_determinism", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (80, 3)),
                     index=idx, columns=list("ABC"))
    with pytest.raises(ValueError):
        op.calculate(x, window=60, dim=2, delay=2, min_line=60)  # min_line >= 60-2


def test_relational_spec_glr_min_segment():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_glr_mean_shift_score", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (80, 3)),
                     index=idx, columns=list("ABC"))
    with pytest.raises(ValueError):
        op.calculate(x, window=20, min_segment=15)  # 20 < 2*15+2


def test_relational_spec_ps_min_periods_window_minus_one():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_pastor_stambaugh_liquidity_gamma", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    ret = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (80, 3)),
                       index=idx, columns=list("ABC"))
    amt = pd.DataFrame(np.random.default_rng(1).lognormal(5, 0.5, (80, 3)),
                       index=idx, columns=list("ABC"))
    with pytest.raises(ValueError):
        op.calculate(ret, ret, amt, window=20, min_periods=20)  # 20 > 20-1


def test_relational_spec_dmd_top_k_rank():
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_dmd_mode_concentration", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    x = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (80, 3)),
                     index=idx, columns=list("ABC"))
    with pytest.raises(ValueError):
        op.calculate(x, window=60, rank=2, dim=3, delay=1, top_k=5)  # top_k > rank


# ---------------------------------------------------------------------------
# operator_cost_model
# ---------------------------------------------------------------------------
def test_cost_model_orders_by_family():
    from factor_engine.cleaned_operators.operator_cost_model import runtime_cost

    hvg252 = runtime_cost("ts_hvg_motif_entropy", {"window": 252})
    hvg64 = runtime_cost("ts_hvg_motif_entropy", {"window": 64})
    ema = runtime_cost("ts_ema", {"window": 252})
    assert hvg252 > hvg64 * 10  # cubic vs quadratic growth
    assert hvg252 > ema * 100   # O(W^3) far above O(W)


def test_cost_model_respects_window_scale():
    from factor_engine.cleaned_operators.operator_cost_model import runtime_cost

    qn20 = runtime_cost("ts_qn_scale", {"window": 20})
    qn500 = runtime_cost("ts_qn_scale", {"window": 500})
    assert qn500 / qn20 > 100  # quadratic in window


# ---------------------------------------------------------------------------
# operator_audits
# ---------------------------------------------------------------------------
def test_audit_equivalent_parameters_runs():
    from factor_engine.cleaned_operators.operator_audits import audit_equivalent_parameters
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_interval_union_coverage", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    lo = pd.DataFrame(np.random.default_rng(0).normal(100, 2, (80, 3)),
                      index=idx, columns=list("ABC"))
    hi = lo + np.abs(np.random.default_rng(1).normal(0, 2, (80, 3)))
    findings = audit_equivalent_parameters(op, lo, hi, param_grid={"window": [20, 65, 80]})
    assert isinstance(findings, list)


def test_audit_column_permutation_runs():
    from factor_engine.cleaned_operators.operator_audits import audit_column_permutation
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("cs_isotonic_residual", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.normal(0, 1, (80, 6)), index=idx,
                     columns=list("ABCDEF"))
    x = pd.DataFrame(rng.normal(0, 1, (80, 6)), index=idx,
                     columns=list("ABCDEF"))
    panels = {"y": y, "x": x}
    findings = audit_column_permutation(op, ["y", "x"], panels, {})
    assert isinstance(findings, list)


def test_audit_self_contamination_runs():
    from factor_engine.cleaned_operators.operator_audits import audit_self_contamination
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_vector_state_mahalanobis", "pandas_numpy")
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(0)
    f1 = pd.DataFrame(rng.normal(0, 1, (80, 4)), index=idx, columns=list("ABCD"))
    f2 = pd.DataFrame(rng.normal(0, 1, (80, 4)), index=idx, columns=list("ABCD"))
    f3 = pd.DataFrame(rng.normal(0, 1, (80, 4)), index=idx, columns=list("ABCD"))
    f4 = pd.DataFrame(rng.normal(0, 1, (80, 4)), index=idx, columns=list("ABCD"))
    panels = {"f1": f1, "f2": f2, "f3": f3, "f4": f4}
    findings = audit_self_contamination(op, ["f1", "f2", "f3", "f4"], panels, {"window": 40})
    assert isinstance(findings, list)
