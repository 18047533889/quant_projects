# -*- coding: utf-8 -*-
"""R21-LOGICAL-NODE-ID: duplicate canonical node_id detection (fail closed).

The batch-global optimizer discovers the physical graph from ``PlanNode.inputs``
and keys it by canonical ``node_id``.  Two DISTINCT objects that carry the same
``node_id`` used to be silently renamed (``<id>_<object-id>``), which made the
physical graph depend on traversal order / object identity — the same formula
could produce different physical graphs across runs.  These regression tests pin
the fail-closed behavior: a duplicate canonical id on distinct nodes raises
``DuplicateLogicalNodeIdentityError`` and the graph is never silently renamed.
"""
from __future__ import annotations

import os

# Serial execution: pin BLAS/Polars thread counts before any library import.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_var] = "1"

import pytest

from factor_engine.planner.logical_plan import DuplicateLogicalNodeIdentityError, PlanNode
from factor_engine.runtime.multibackend.batch_global_optimizer import (
    _raise_duplicate_logical_node_identity,
)
from factor_engine.planner.batch_global_optimizer import BatchGlobalOptimizer


def _ctx():
    from types import SimpleNamespace

    return SimpleNamespace(
        run_mode="research",
        runtime_stats={"row_count_estimate": 100},
    )


def test_duplicate_node_id_on_distinct_objects_raises() -> None:
    """Two distinct nodes with the same canonical node_id fail closed."""
    left = PlanNode("column", attrs={"name": "left"}, node_id="dup")
    right = PlanNode("literal", attrs={"value": 7}, node_id="dup")
    root = PlanNode("add", inputs=(left, right), node_id="root")

    with pytest.raises(DuplicateLogicalNodeIdentityError, match="duplicate canonical logical node_id='dup'"):
        BatchGlobalOptimizer().optimize_batch({"root": root}, {}, {}, _ctx())


def test_duplicate_node_id_never_silently_renames() -> None:
    """The collision is never renamed into a traversal-order dependent id."""
    left = PlanNode("column", attrs={"name": "left"}, node_id="dup")
    right = PlanNode("literal", attrs={"value": 7}, node_id="dup")
    root = PlanNode("add", inputs=(left, right), node_id="root")

    # Swap traversal order: a silent rename would flip which node gets the base
    # id, yielding a different physical graph for the same formula.  Both orders
    # must fail identically — there is no "first one wins".
    root_flipped = PlanNode("add", inputs=(right, left), node_id="root")
    with pytest.raises(DuplicateLogicalNodeIdentityError):
        BatchGlobalOptimizer().optimize_batch({"root": root_flipped}, {}, {}, _ctx())


def test_shared_plan_node_reference_still_allowed() -> None:
    """A shared PlanNode object referenced twice is NOT a collision."""
    leaf = PlanNode("column", attrs={"name": "close"}, node_id="leaf")
    root = PlanNode("add", inputs=(leaf, leaf), node_id="root")

    result = BatchGlobalOptimizer().optimize_batch(
        {"root": root}, {}, {}, _ctx()
    )
    assert "leaf" in result.per_node_choices
    assert "root" in result.per_node_choices


def test_duplicate_node_id_in_discovered_graph_fails_closed() -> None:
    """A collision introduced deeper than the roots is still caught.

    Both children share one canonical node_id but are DISTINCT objects.  The
    roots must pass ``node_id=`` (not a preferred root id) so discovery keys by
    the canonical id and hits the collision guard.
    """
    leaf_a = PlanNode("column", attrs={"name": "x"}, node_id="dup-child")
    leaf_b = PlanNode("column", attrs={"name": "y"}, node_id="dup-child")
    root = PlanNode("add", inputs=(leaf_a, leaf_b), node_id="root")

    with pytest.raises(DuplicateLogicalNodeIdentityError):
        BatchGlobalOptimizer().optimize_batch({"root": root}, {}, {}, _ctx())


def test_error_carries_both_distinct_node_ops() -> None:
    """The message identifies the two distinct objects, not a rename hint."""
    left = PlanNode("column", attrs={"name": "left"}, node_id="dup")
    right = PlanNode("literal", attrs={"value": 7}, node_id="dup")

    with pytest.raises(DuplicateLogicalNodeIdentityError) as excinfo:
        _raise_duplicate_logical_node_identity(
            node_id="dup", existing_node=left, new_node=right
        )

    message = str(excinfo.value)
    assert "dup" in message
    assert "column" in message
    assert "literal" in message
