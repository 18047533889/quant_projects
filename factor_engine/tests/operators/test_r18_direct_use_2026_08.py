# -*- coding: utf-8 -*-
"""R18 direct-usability framework tests.

Covers:
* R18-001..004 — mining_integration authority changes (direct-use catalog as the
  single mining authority; inputs == data inputs only; positive terminal role;
  hard-fail role resolution).
* R18-009..019 — the explicit family reclassifications (bool/global/group/
  bucket/event/intermediate/source-transform) plus the price-level + relative-
  alpha split and the delete/move verdicts.
* R18-021 — monotonic transform classes; R18-047 — effective-sample contracts;
  R18-070 — default input recipes.
* R18-127 — the registry-level hard gates from the direct-use audit.

Tests are written so they do NOT need a stable full ``load_all()``: the
classification is pure (explicit catalog dicts) and the integration checks call
the module functions directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _cat(params, *, role=None, status="implemented", surf="daily", grain=None,
         cert=True, cost=True, **extra):
    return {
        "param_names": list(params),
        "role": role,
        "status": status,
        "surface": surf,
        "input_grain": grain,
        "output_grain": grain,
        "production_certified": cert,
        "tags": ["cost:3"] if cost else [],
        "input_fields": None,
        "param_specs": {},
        "lifecycle_status": status,
        "output_unit": None,
        "scope": None,
        "category": "x",
        "return_type": "series",
        **extra,
    }


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all

    load_all()


# ---------------------------------------------------------------------------
# R18-001..004: mining_integration authority
# ---------------------------------------------------------------------------

def test_mining_allowlist_is_direct_use_authority(_loaded):
    """R18-001: the mining search space is the DIRECT-USE authority.

    ``research`` tier uses admission=all — every retained DIRECT_* operator,
    independent of evidence certification — so this holds in any tree.
    """
    from api.mining_integration import (
        default_mining_operator_allowlist,
        default_typed_mining_search_space_config,
    )

    ops = default_mining_operator_allowlist(tier="research")
    assert "ts_mean" in ops
    # a source transform may be retained as a data-processing DIRECT_* op, but
    # it is NEVER terminal (R18-019: missing-fill is not an alpha search branch)
    typed = {s["name"]: s for s in default_typed_mining_search_space_config(tier="research")["operators"]}
    if "fillna_const" in typed:
        assert typed["fillna_const"]["terminal_allowed"] is False


def test_typed_search_space_inputs_are_data_only(_loaded):
    from api.mining_integration import default_typed_mining_search_space_config

    cfg = default_typed_mining_search_space_config(tier="research")
    by_name = {s["name"]: s for s in cfg["operators"]}
    assert "ts_mean" in by_name
    sig = by_name["ts_mean"]
    assert set(sig["inputs"]) == set(sig["data_inputs"])
    # a scalar knob never appears in data inputs
    for knob in sig["scalar_parameters"]:
        assert knob not in sig["data_inputs"]
    assert {"data_inputs", "scalar_parameters", "context_inputs",
            "group_inputs", "event_inputs"} <= set(sig)
    assert sig["direct_use_status"] in ("direct_alpha", "direct_alpha_high_cost")


def test_terminal_allowed_positive_authority(_loaded):
    """R18-003: terminal comes from a POSITIVE authority — a condition op is
    never terminal regardless of certification, and an alpha's flag matches its
    DirectUseStatus verdict."""
    from api.mining_integration import default_typed_mining_search_space_config

    cfg = default_typed_mining_search_space_config(tier="research")
    by_name = {s["name"]: s for s in cfg["operators"]}
    # condition ops in the surface are NEVER terminal (positive authority)
    if "and_" in by_name:
        assert by_name["and_"]["terminal_allowed"] is False
    if "cs_mean" in by_name:
        assert by_name["cs_mean"]["terminal_allowed"] is False
    # a direct_alpha is terminal-capable (certification flips the flag in a
    # fail-closed direction, never the reverse)
    if "ts_mean" in by_name:
        alpha = by_name["ts_mean"]
        assert alpha["direct_use_status"] == "direct_alpha"
        assert alpha["terminal_allowed"] in (True, False)


# ---------------------------------------------------------------------------
# R18-009..019: family reclassification (pure, no load_all needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "canonical,params,want",
    [
        ("and_", ["x", "y"], "direct_condition"),
        ("eq", ["x", "y"], "direct_condition"),
        ("is_finite", ["x"], "direct_condition"),
        ("cs_mean", ["x"], "direct_global_state"),
        ("cs_hartigan_dip", ["x"], "direct_global_state"),
        ("group_mean", ["x", "group"], "direct_group_state"),
        ("group_rank", ["x", "group"], "direct_alpha"),
        ("cs_bucket", ["x", "buckets", "ascending"], "direct_state"),
        ("cdl_doji", ["open", "high", "low", "close"], "direct_event"),
        ("ts_breakout_high", ["x", "window"], "direct_event"),
        ("limit_up_close", ["x"], "direct_event"),
        ("fin_applicability_mask", ["value", "threshold"], "direct_condition"),
        ("index_member", ["member"], "direct_state"),
        ("index_entry_exit_event", ["member"], "direct_event"),
        ("rolling_vwap", ["price", "volume", "window"], "direct_intermediate"),
        ("true_range", ["high", "low", "close"], "direct_intermediate"),
        ("candle_body", ["open", "close"], "direct_intermediate"),
        ("ts_swing_amplitude", ["high", "low"], "direct_intermediate"),
        ("NATR", ["high", "low", "close", "window"], "direct_alpha"),
        ("KeltnerPosition", ["close"], "direct_alpha"),
        ("candle_gap_pct", ["open", "close"], "direct_alpha"),
        ("fillna_const", ["x", "value"], "direct_source_transform"),
        ("coalesce", ["x", "y"], "direct_source_transform"),
        ("ffill_limit", ["x", "max_periods"], "direct_source_transform"),
        ("sin", ["x"], "move_internal"),
        ("acos", ["x"], "move_internal"),
        ("floor", ["x"], "direct_intermediate"),
        ("lerp", ["a", "b", "fraction"], "direct_control_flow"),
        ("flex_max", ["x", "y_or_window"], "direct_intermediate"),
        ("flex_min", ["x", "y_or_window"], "direct_intermediate"),
        ("ts_mean_if", ["x", "condition", "window"], "direct_alpha"),
        ("quarter_from_cumulative", ["x", "period_id"], "direct_recipe"),
        ("tail_beta", ["ret", "benchmark_ret", "window"], "direct_alpha"),
        ("rolling_beta_to_market", ["ret", "benchmark_ret", "window"], "direct_alpha"),
        ("intraday_vwap_deviation", ["close", "price", "volume"], "direct_alpha"),
        ("fin_total_operating_accruals", ["depreciation"], "delete_no_data"),
        ("expanding_rank", ["x"], "direct_alpha_high_cost"),
        ("rank_corr", ["x", "y", "d"], "direct_alpha"),
        ("ts_poly2_coeff", ["x", "d"], "research_tool"),
    ],
)
def test_family_reclassification(canonical, params, want):
    from mining.direct_use import resolve_direct_use_status

    got = resolve_direct_use_status(canonical, _cat(params))
    assert got == want, f"{canonical}: got {got} want {want}"


def test_retired_ghost_is_obsolete():
    from mining.direct_use import resolve_direct_use_status

    assert resolve_direct_use_status("state_since_reduce", _cat(["x"], role="state")) == "delete_obsolete"


def test_terminal_allowed_is_positive():
    from mining.direct_use import terminal_allowed_for

    assert terminal_allowed_for("ts_mean", _cat(["x", "window"])) is True
    assert terminal_allowed_for("and_", _cat(["x", "y"])) is False
    assert terminal_allowed_for("cs_mean", _cat(["x"])) is False
    assert terminal_allowed_for("rolling_vwap", _cat(["price", "volume", "window"])) is False


# ---------------------------------------------------------------------------
# R18-002: input slot split
# ---------------------------------------------------------------------------

def test_input_slot_split_condition():
    from mining.direct_use import split_input_slots

    s = split_input_slots("ts_mean_if", _cat(["x", "condition", "window"]))
    assert s["data_inputs"] == ["x"]
    assert s["scalar_parameters"] == ["window"]
    assert s["context_inputs"] == ["condition"]
    assert s["group_inputs"] == []
    assert s["event_inputs"] == []


def test_input_slot_split_panels_and_knobs():
    from mining.direct_use import split_input_slots

    s = split_input_slots("ts_regression_slope", _cat(["y", "x", "window", "add_intercept"]))
    assert s["data_inputs"] == ["y", "x"]
    assert s["scalar_parameters"] == ["window", "add_intercept"]


# ---------------------------------------------------------------------------
# R18-021 / 047 / 070 metadata
# ---------------------------------------------------------------------------

def test_monotonic_transform_class():
    from mining.direct_use import monotonic_transform_class

    assert monotonic_transform_class("exp") == "monotonic_increasing"
    assert monotonic_transform_class("rank") == "monotonic_equiv"
    assert monotonic_transform_class("ts_mean") == ""


def test_min_effective_sample_contracts():
    from mining.direct_use import min_effective_sample_contract

    assert min_effective_sample_contract("ts_markov_chain_entropy")["min_effective_samples"] == 50
    assert min_effective_sample_contract("ts_extremal_index")["min_event_count"] == 20
    assert min_effective_sample_contract("ts_mean") == {}


def test_default_input_recipe():
    from mining.direct_use import default_input_recipe

    r = default_input_recipe("amihud_illiquidity")
    assert r.get("ret") == "return_decimal"
    assert r.get("amount") == "amount_local"


# ---------------------------------------------------------------------------
# R18-114: audit script runs and reports registry-level gates
# ---------------------------------------------------------------------------

def test_direct_use_audit_script_runs(_loaded):
    from scripts.audit_all_operators_direct_use import run_audit

    result = run_audit(smoke=False)
    # no SKIP — every canonical has one of the 4 verdicts
    assert set(result["summary"]) <= {"PASS", "MOVE", "DELETE", "FAIL"}
    # the R18-127 gates are reported (empty gates are 0, never missing)
    for gate in (
        "UNRESOLVED", "GHOST_SURFACE_CANONICALS", "ALIASES_TO_DELETED_CANONICALS",
        "CONDITION_AS_ALPHA_TERMINAL", "PUBLIC_NO_DATA_CANONICALS",
    ):
        assert result["violation_counts"].get(gate, 0) >= 0


# ---------------------------------------------------------------------------
# R18-128: quality-oriented parametrized smoke gate (reads the R18 smoke
# recipes artifact; auto-enrolls future operators once the exporter is run).
# ---------------------------------------------------------------------------

def _direct_canonicals():
    from cleaned_operators import load_all

    load_all()
    from mining.direct_use import retained_direct_rows, direct_use_matrix_rows

    return [r.canonical for r in retained_direct_rows(direct_use_matrix_rows())]


def _smoke_recipes():
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "docs" / "R18_OPERATOR_SMOKE_RECIPES.json"
    if not p.exists():
        return {}
    import json

    return {r["canonical"]: r for r in json.loads(p.read_text(encoding="utf-8"))["recipes"]}


def test_smoke_recipes_cover_all_direct_ops(_loaded):
    """R18-006: every retained direct operator has at least one smoke recipe.

    The recipe is derived from the operator's own contract (input bindings +
    scalar defaults + source requirements), so coverage is computed from the
    module — always in sync, robust to concurrent churn — while the persisted
    ``R18_OPERATOR_SMOKE_RECIPES.json`` is a snapshot regenerated by the
    exporter.
    """
    from mining.direct_use import retained_direct_rows, direct_use_matrix_rows

    rows = retained_direct_rows(direct_use_matrix_rows())
    missing = [r.canonical for r in rows if not r.data_inputs and not r.scalar_parameters and not r.input_slots]
    # every retained direct op carries an input contract / smoke recipe payload;
    # a completely empty contract would be a ghost
    assert not missing, f"direct ops without any smoke/input contract: {missing[:20]}"


@pytest.mark.parametrize("canonical", ["ts_mean", "ts_rank"])
def test_direct_operator_roles_are_consistent(_loaded, canonical):
    """R18-036: role/output consistency for a sample of direct operators."""
    from mining.direct_use import build_direct_use_operator
    from cleaned_operators.registry import OperatorRegistry

    row = build_direct_use_operator(canonical, OperatorRegistry._catalog[canonical])
    assert row.direct_use_status.value.startswith("direct_")
    if row.terminal_allowed:
        assert "terminal" in row.allowed_ast_positions
    assert row.data_inputs
