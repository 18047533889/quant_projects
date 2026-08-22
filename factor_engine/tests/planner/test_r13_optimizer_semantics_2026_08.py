# -*- coding: utf-8
"""R13 planner audit round (2026-08): optimizer semantic-attr preservation,
execution-vs-hash parameter canonicalization, CSE semantic binding.

Covers:
- NEW-P0-14/15 — ``semantic_attrs`` survive canonicalize_plan_parameters and
  every fastpath rewrite (single ``rewrite_node`` constructor);
- NEW-P0-16 — log-return fastpath reuses the ORIGINAL column node so typed
  source identity (field_id/source_table/...) is not dropped;
- NEW-P0-17 — execution parameters are never rewritten for dedup; only hash
  keys canonicalize;
- NEW-P0-20 — planning-time validation runs the FULL runtime authority
  (active_when + min_periods<=window + RelationalParamSpec), not just ints;
- NEW-P0-21 — CSE ``plan_ref`` carries the shared subtree's semantic attrs;
- NEW-P1-22 — ``structural_key`` binds a semantic digest when present.
"""
from __future__ import annotations

import pytest

from planner.logical_plan import PlanNode
from planner.optimizer import Optimizer
from planner.cse import apply_cse, collect_consumed_sids
from planner.plan_hash import structural_key
from planner.rewrite_fastpath import rewrite_plan_for_fastpath
from planner.canonicalize_params import (
    canonicalize_parameter_values,
    canonicalize_plan_parameters,
    validate_plan_params,
)

from cleaned_operators import load_all

load_all()  # real ParamSpec contracts for ts_mean validation


def _col(name: str = "close", **source_keys: object) -> PlanNode:
    attrs = {"name": name, **source_keys}
    return PlanNode(op="column", attrs=attrs, inputs=[])


def _semantic(**kw) -> dict:
    return {"unit": "return", "grain": "daily", **kw}


# ---------------------------------------------------------------------------
# NEW-P0-14/15: semantic_attrs survive the optimizer pipeline
# ---------------------------------------------------------------------------

def test_canonicalize_plan_parameters_preserves_semantic_attrs():
    plan = PlanNode(
        op="ts_mean",
        inputs=[_col()],
        attrs={"window": 5},
        semantic_attrs=_semantic(price_basis="RAW"),
    )
    out = canonicalize_plan_parameters(plan)
    assert out.semantic_attrs["price_basis"] == "RAW"
    assert out.semantic_attrs["unit"] == "return"
    assert out.op == "ts_mean"
    assert out.attrs["window"] == 5


def test_optimizer_pipeline_preserves_semantic_attrs():
    col = _col()
    mean = PlanNode(
        op="ts_mean", inputs=[col], attrs={"window": 5},
        semantic_attrs=_semantic(price_basis="RAW"),
    )
    plan = PlanNode(
        op="subtract", inputs=[col, mean], attrs={},
        semantic_attrs=_semantic(price_basis="RAW"),
    )
    out = Optimizer().optimize(plan)

    def walk(n: PlanNode):
        # leaf nodes (column/literal) carry no semantic attrs; every OPERATOR
        # node that had them must still have them after folding+lowering+rewrite.
        if n.op not in {"column", "literal", "plan_ref", "materialized_series"}:
            assert n.semantic_attrs.get("unit") == "return", f"semantic lost at {n.op}"
            assert n.semantic_attrs.get("price_basis") == "RAW", f"basis lost at {n.op}"
        for child in n.inputs:
            walk(child)

    walk(out)


def test_rewrite_node_preserves_semantic_attrs_by_default():
    from planner.rewrite_fastpath import rewrite_node

    old = PlanNode(op="x", inputs=[_col()], attrs={"a": 1}, semantic_attrs=_semantic(grain="minute"))
    new = rewrite_node(old, op="y", inputs=[_col("b")])
    assert new.op == "y"
    assert new.semantic_attrs["grain"] == "minute"


# ---------------------------------------------------------------------------
# NEW-P0-16: log-return fastpath keeps the original column's source identity
# ---------------------------------------------------------------------------

def test_fastpath_log_returns_keeps_source_identity():
    col = _col("Close", field_id="fieldA", source_table="sourceA", source_field="Close")
    delay = PlanNode(op="ts_delay", attrs={"d": 1}, inputs=[col])
    div = PlanNode(op="divide", inputs=[col, delay], attrs={})
    plan = PlanNode(op="log", inputs=[div], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    assert out.op == "ts_log_return"
    # the column node is REUSED, not rebuilt from the display name
    assert out.inputs[0].attrs["field_id"] == "fieldA"
    assert out.inputs[0].attrs["source_table"] == "sourceA"


def test_fastpath_log_returns_not_applied_across_same_name_different_source():
    a = _col("Close", field_id="fieldA", source_table="sourceA")
    b = _col("Close", field_id="fieldB", source_table="sourceB")
    delay = PlanNode(op="ts_delay", attrs={"d": 1}, inputs=[b])
    div = PlanNode(op="divide", inputs=[a, delay], attrs={})
    plan = PlanNode(op="log", inputs=[div], attrs={})
    out = rewrite_plan_for_fastpath(plan)
    # sourceA.Close and sourceB.Close are DIFFERENT data — no fusion
    assert out.op == "log"


# ---------------------------------------------------------------------------
# NEW-P0-17: execution parameters verbatim; only hash keys canonicalize
# ---------------------------------------------------------------------------

def test_execution_params_never_rewritten_for_dedup():
    plan = PlanNode(
        op="custom_op",
        inputs=[_col()],
        attrs={"window": 0.123456789012345},
    )
    out = canonicalize_plan_parameters(plan)
    # exact user value survives into the execution plan
    assert out.attrs["window"] == 0.123456789012345


def test_hash_side_canonicalizes_float_noise():
    a = canonicalize_parameter_values({"window": 0.123456789012345}, canonical="custom_op")
    b = canonicalize_parameter_values({"window": 0.123456789012}, canonical="custom_op")
    assert a == b
    assert a["window"] != 0.123456789012345


# ---------------------------------------------------------------------------
# NEW-P0-20: planning-time validation = full runtime authority
# ---------------------------------------------------------------------------

def test_validate_plan_params_rejects_min_periods_gt_window():
    plan = PlanNode(op="ts_argmax", inputs=[_col()], attrs={"window": 20, "min_periods": 30})
    with pytest.raises(Exception, match="planning-time"):
        validate_plan_params(plan)


def test_optimizer_rejects_infeasible_combination_before_lowering():
    plan = PlanNode(op="ts_argmax", inputs=[_col()], attrs={"window": 20, "min_periods": 30})
    with pytest.raises(Exception, match="planning-time"):
        Optimizer().optimize(plan)


def test_validate_plan_params_enforces_active_when():
    """R13 NEW-P0-20: a dead knob (inactive parameter set to a non-default value)
    is rejected at PLANNING time, not left for runtime."""
    def _candle(**scalar):
        return PlanNode(
            op="candlestick_pattern",
            inputs=[_col("o"), _col("h"), _col("l"), _col("c")],
            attrs=scalar,
        )

    # pattern='doji' is NOT an allowed controller for body_window -> body_window
    # is inactive; providing a non-default value must be rejected at planning.
    with pytest.raises(Exception, match="planning-time"):
        validate_plan_params(_candle(pattern="doji", body_window=999))
    # pattern='3_stars_south' IS allowed -> body_window active, tolerated.
    validate_plan_params(_candle(pattern="3_stars_south", body_window=999))


# ---------------------------------------------------------------------------
# NEW-P0-21: CSE plan_ref carries the shared subtree's semantic contract
# ---------------------------------------------------------------------------

def test_cse_plan_ref_carries_semantic_attrs():
    shared = PlanNode(
        op="ts_mean",
        inputs=[_col()],
        attrs={"window": 5},
        semantic_attrs=_semantic(price_basis="RAW"),
    )
    root_a = PlanNode(op="divide", inputs=[shared, _col("x")], attrs={})
    root_b = PlanNode(op="multiply", inputs=[shared, _col("y")], attrs={})
    roots, _ = apply_cse([root_a, root_b])

    def find_ref(n: PlanNode):
        if n.op == "plan_ref":
            return n
        for c in n.inputs:
            hit = find_ref(c)
            if hit is not None:
                return hit
        return None

    ref = find_ref(roots[0])
    assert ref is not None, "expected a plan_ref for the shared subtree"
    assert ref.semantic_attrs["price_basis"] == "RAW"
    assert ref.semantic_attrs["unit"] == "return"


# ---------------------------------------------------------------------------
# NEW-P1-22: structural_key binds output semantics when present
# ---------------------------------------------------------------------------

def test_structural_key_binds_semantic_digest():
    plain = PlanNode(op="ts_mean", inputs=[_col()], attrs={"window": 5})
    raw = PlanNode(
        op="ts_mean",
        inputs=[_col()],
        attrs={"window": 5},
        semantic_attrs={"price_basis": "RAW"},
    )
    continuous = PlanNode(
        op="ts_mean",
        inputs=[_col()],
        attrs={"window": 5},
        semantic_attrs={"price_basis": "CONTINUOUS"},
    )
    assert structural_key(plain) == structural_key(PlanNode(op="ts_mean", inputs=[_col()], attrs={"window": 5}))
    assert structural_key(raw) != structural_key(continuous)
    assert structural_key(raw) != structural_key(plain)


def test_cse_does_not_share_across_different_semantics():
    """Structurally identical subtrees with different output semantics must NOT
    be CSE-merged (they compute different quantities).  The shared column may be
    merged (same semantics); the ts_mean nodes differ by price_basis."""
    a = PlanNode(
        op="ts_mean", inputs=[_col()], attrs={"window": 5},
        semantic_attrs={"price_basis": "RAW"},
    )
    b = PlanNode(
        op="ts_mean", inputs=[_col()], attrs={"window": 5},
        semantic_attrs={"price_basis": "CONTINUOUS"},
    )
    root_a = PlanNode(op="divide", inputs=[a, _col("x")], attrs={})
    root_b = PlanNode(op="multiply", inputs=[b, _col("y")], attrs={})
    roots, shared = apply_cse([root_a, root_b])
    merged_ops = {n.op for n in shared.values()}
    assert "ts_mean" not in merged_ops, "different-semantic ts_mean subtrees were CSE-merged"

    def find_ref(n: PlanNode):
        if n.op == "plan_ref":
            return n
        for c in n.inputs:
            hit = find_ref(c)
            if hit is not None:
                return hit
        return None

    for root in roots:
        ref = find_ref(root)
        if ref is not None:
            sid = ref.attrs["sid"]
            # the only allowed shared sid is the (semantically identical) column
            assert shared[sid].op == "column", f"ts_mean sid shared across semantics: {sid}"
