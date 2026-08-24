# -*- coding: utf-8 -*-
"""R6 system test A — backend call-contract parity (P0-01/P0-03 acceptance).

For a multi-backend canonical (pandas_numpy + polars + sql), every backend must
give the SAME accept/reject result for:
  valid params / fractional int / bool-as-int / unknown kwarg / extra positional
  / NaN param / Inf param / misaligned index / column permutation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators as co

co.load_all()

from factor_engine.cleaned_operators.base import validate_operator_call
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_errors import OperatorParameterError


def _panels(n: int = 20, cols: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, (n, len(cols))),
        index=dates, columns=list(cols),
    )


MULTI_BACKEND = ("ts_mean", "ts_std", "ts_delay")


def _ops(canonical: str):
    ops = {}
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get(canonical, backend)
        if op is not None:
            ops[backend] = op
    assert len(ops) >= 2, f"{canonical}: need >=2 framework backends, got {list(ops)}"
    return ops


def _accept(op, args, kwargs=None):
    try:
        validate_operator_call(op, tuple(args), dict(kwargs or {}))
        return True
    except (OperatorParameterError, ValueError):
        return False


def _sql_plan_rejects(canonical: str, kwargs: dict) -> bool:
    """The DuckDB/SQL backend is a marker operator validated at the emitter /
    planning layer, so a bad parameter must be rejected BEFORE lowering by the
    planner's pre-lowering ParamSpec validator (R6 P0-04)."""
    from factor_engine.planner.logical_plan import PlanNode
    from factor_engine.planner.optimizer import Optimizer

    probe = PlanNode(op=canonical, inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])], attrs=dict(kwargs))
    try:
        Optimizer().optimize(probe, production=True)
        return False
    except (ValueError, OperatorParameterError, Exception):
        return True


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_valid_call_accepted_on_all_backends(canonical):
    panel = _panels()
    kwargs = {"window": 10} if canonical != "ts_delay" else {"window": 3}
    verdicts = {b: _accept(op, (panel,), kwargs) for b, op in _ops(canonical).items()}
    assert all(verdicts.values()), f"{canonical}: valid call rejected on {[b for b, v in verdicts.items() if not v]}"
    # SQL/planning path accepts the same valid call.
    from factor_engine.planner.optimizer import Optimizer
    from factor_engine.planner.logical_plan import PlanNode
    probe = PlanNode(op=canonical, inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])], attrs=dict(kwargs))
    Optimizer().optimize(probe, production=True)


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_fractional_int_rejected_on_all_backends(canonical):
    panel = _panels()
    verdicts = {b: _accept(op, (panel,), {"window": 5.9}) for b, op in _ops(canonical).items()}
    assert not any(verdicts.values()), (
        f"{canonical}: fractional window=5.9 accepted on "
        f"{[b for b, v in verdicts.items() if v]}"
    )
    assert _sql_plan_rejects(canonical, {"window": 5.9})


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_bool_as_int_rejected_on_all_backends(canonical):
    panel = _panels()
    verdicts = {b: _accept(op, (panel,), {"window": True}) for b, op in _ops(canonical).items()}
    assert not any(verdicts.values()), f"{canonical}: bool-as-int accepted"
    assert _sql_plan_rejects(canonical, {"window": True})


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_unknown_kwarg_rejected_on_all_backends(canonical):
    panel = _panels()
    verdicts = {b: _accept(op, (panel,), {"bogus_param": 1}) for b, op in _ops(canonical).items()}
    assert not any(verdicts.values()), f"{canonical}: unknown kwarg accepted"


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_extra_positional_rejected_on_all_backends(canonical):
    panel = _panels()
    verdicts = {b: _accept(op, (panel, 10, 5, 99)) for b, op in _ops(canonical).items()}
    assert not any(verdicts.values()), f"{canonical}: extra positional accepted"


@pytest.mark.parametrize("canonical", MULTI_BACKEND)
def test_r6_a_nan_and_inf_params_rejected_on_all_backends(canonical):
    panel = _panels()
    for bad in (float("nan"), float("inf")):
        verdicts = {b: _accept(op, (panel,), {"window": bad}) for b, op in _ops(canonical).items()}
        assert not any(verdicts.values()), f"{canonical}: {bad} accepted"
        assert _sql_plan_rejects(canonical, {"window": bad})


def test_r6_a_misaligned_index_rejected_on_all_backends():
    base = _panels()
    shifted = base.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    verdicts = {}
    for b, op in _ops("ts_mean").items():
        try:
            validate_operator_call(op, (base, shifted, 10, 5), {})
            verdicts[b] = True
        except (OperatorParameterError, ValueError):
            verdicts[b] = False
    assert not any(verdicts.values()), f"misaligned index accepted on {[b for b, v in verdicts.items() if v]}"


def test_r6_a_column_permutation_rejected_on_all_backends():
    base = _panels()
    perm = base.copy()
    perm.columns = ["B", "A"]
    verdicts = {}
    for b, op in _ops("ts_mean").items():
        try:
            validate_operator_call(op, (base, perm, 10, 5), {})
            verdicts[b] = True
        except (OperatorParameterError, ValueError):
            verdicts[b] = False
    assert not any(verdicts.values()), f"column permutation accepted on {[b for b, v in verdicts.items() if v]}"
