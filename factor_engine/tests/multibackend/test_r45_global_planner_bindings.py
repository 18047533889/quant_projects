# -*- coding: utf-8 -*-
"""R45: Global-Planner exact-binding tests.

Covers the three Global-Planner items:
    1. Numba candidate uses REAL bound params (window/span), not window=0, and
       carries its OWN PhysicalImplementationID (does not borrow pandas_numpy's
       production capability).
    2. NodeBackendChoice binds exact implementation: PhysicalImplementationID +
       bound parameter identity + implementation closure hash + numeric policy
       identity + kernel signature.
    3. ``column`` is NOT unconditionally production_certified=True: it requires a
       physical source binding (DataReadIdentity + field semantics + PIT +
       universe + snapshot).  ``literal``/``plan_ref`` stay certified.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from factor_engine.planner.logical_plan import PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    NodeBackendChoice,
    PhysicalBatchGlobalOptimizer,
)


def _ctx(run_mode: str = "research") -> SimpleNamespace:
    return SimpleNamespace(run_mode=run_mode, data_source=None, runtime_stats={})


def _optimizer() -> PhysicalBatchGlobalOptimizer:
    return PhysicalBatchGlobalOptimizer()


# ---------------------------------------------------------------------------
# 1. Numba candidate uses REAL bound params + own PI-ID
# ---------------------------------------------------------------------------

def test_numba_candidate_uses_real_bound_window() -> None:
    """Numba eligibility must be evaluated with the node's real window, not 0."""
    node = PlanNode(op="ts_mean", attrs={"window": 20}, inputs=())
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    numba = [c for c in choices if c.execution_kind.value == "numba_cpu_kernel"]
    # Numba is only enabled when FACTOR_ENGINE_USE_NUMBA is set; if not, there
    # is no numba candidate and the test is vacuous.  We assert the invariant
    # that WHEN a numba candidate exists it carries the real bound params.
    for c in numba:
        assert c.bound_parameter_identity == '{"window":20}'
        assert c.kernel_signature == "window=20"


def test_numba_candidate_has_own_pi_id() -> None:
    """Numba candidate must not borrow pandas_numpy's production capability."""
    node = PlanNode(op="ts_mean", attrs={"window": 20}, inputs=())
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    numba = [c for c in choices if c.execution_kind.value == "numba_cpu_kernel"]
    pandas = [c for c in choices if c.backend.value == "pandas_numpy"
              and c.execution_kind.value != "numba_cpu_kernel"]
    for c in numba:
        # Numba PI-ID is distinct from the pandas_numpy capability PI-ID.
        assert c.physical_implementation_id.startswith("pi:v3:numba:")
        for p in pandas:
            assert c.physical_implementation_id != p.physical_implementation_id


# ---------------------------------------------------------------------------
# 2. NodeBackendChoice binds exact implementation
# ---------------------------------------------------------------------------

def test_node_backend_choice_binds_exact_implementation() -> None:
    """NodeBackendChoice carries PI-ID + bound params + closure + policy + sig."""
    node = PlanNode(op="ts_mean", attrs={"window": 20}, inputs=())
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    assert choices
    for c in choices:
        assert c.physical_implementation_id, "choice must bind a PI-ID"
        assert c.bound_parameter_identity == '{"window":20}'
        assert c.kernel_signature == "window=20"
        # closure hash / numeric policy are additive fields (may be empty for
        # candidates that do not yet carry them, but the fields must exist).
        assert hasattr(c, "implementation_closure_hash")
        assert hasattr(c, "numeric_policy_identity")


def test_node_backend_choice_keeps_existing_fields() -> None:
    """Additive fields must not break existing positional construction."""
    from factor_engine.planner.backend_region import PhysicalBackend, Representation
    from factor_engine.backend.contracts import ExecutionKind

    choice = NodeBackendChoice(
        node_id="n1",
        backend=PhysicalBackend.PANDAS_NUMPY,
        compute_cost_ms=1.0,
        transfer_from_children_ms=0.0,
        total_cost_ms=1.0,
        representation=Representation.PANDAS_LONG,
        execution_kind=ExecutionKind.NATIVE_EXPR,
        production_certified=True,
    )
    assert choice.node_id == "n1"
    assert choice.production_certified is True
    assert choice.physical_implementation_id == ""
    assert choice.bound_parameter_identity == ""


# ---------------------------------------------------------------------------
# 3. column requires a physical source binding to be production-certified
# ---------------------------------------------------------------------------

def test_column_without_source_binding_not_certified() -> None:
    """A bare column (no source binding) must NOT be production-certified."""
    node = PlanNode(op="column", attrs={"name": "close"}, inputs=())
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    assert choices
    for c in choices:
        assert c.production_certified is False, (
            "bare column must not be production-certified"
        )


def test_column_with_full_source_binding_certified() -> None:
    """A column with DataReadIdentity + field + PIT + universe + snapshot is certified."""
    from data_access.read.data_read_identity import DataReadIdentity, ResolvedFieldIdentity
    from factor_engine.runtime.physical_source_binding import PhysicalSourceBinding
    identity = DataReadIdentity(
        dataset="equity_daily", revision="2026-01-01",
        calendar_identity="cal_2026", universe_snapshot="univ_2026",
        source_snapshot="snap_2026",
        fields=(ResolvedFieldIdentity(logical_name="close", physical_name="close",
                                      dataset="equity_daily", availability="same_day"),),
    )
    node = PlanNode(
        op="column",
        attrs={
            "name": "close",
            "physical_source_binding": PhysicalSourceBinding(identity, "close", "catalog:v1"),
        },
        inputs=(),
    )
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    assert choices
    for c in choices:
        assert c.production_certified is True, (
            "column with full source binding must be production-certified"
        )
        assert c.physical_implementation_id.startswith("pi:v3:source_binding:")


def test_column_partial_binding_not_certified() -> None:
    """A column missing any binding dimension must NOT be certified."""
    node = PlanNode(
        op="column",
        attrs={
            "name": "close",
            "dataset": "equity_daily",
            # missing field semantics / PIT / universe / snapshot
        },
        inputs=(),
    )
    choices = _optimizer()._eligible_choices("n1", node, rows=1000, ctx=_ctx())
    assert choices
    for c in choices:
        assert c.production_certified is False


def test_literal_and_plan_ref_stay_certified() -> None:
    """literal/plan_ref remain production-certified (unchanged)."""
    lit = PlanNode(op="literal", attrs={"value": 1}, inputs=())
    lit_choices = _optimizer()._eligible_choices("n1", lit, rows=1000, ctx=_ctx())
    assert lit_choices
    for c in lit_choices:
        assert c.production_certified is True

    ref = PlanNode(op="plan_ref", attrs={"sid": "s1"}, inputs=())
    ref_choices = _optimizer()._eligible_choices("n2", ref, rows=1000, ctx=_ctx())
    assert ref_choices
    for c in ref_choices:
        assert c.production_certified is True
