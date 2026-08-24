# -*- coding: utf-8 -*-
"""R10 #2/#6 regression tests: PIT eligibility is tri-state (True/False/
None=UNKNOWN) everywhere; UNKNOWN propagates through the semantic lattice and
rejects in production (never silently treated as safe)."""
from __future__ import annotations

from factor_engine.ir.types import lattice_join_semantic_attrs


def _attrs(pit: object | None) -> dict:
    d = {"semantic_kind": "price", "domain": "price", "unit": "price"}
    d["pit_safe"] = pit
    return d


def test_lattice_all_true_is_true():
    out = lattice_join_semantic_attrs(
        [_attrs(True), _attrs(True), _attrs(True)]
    )
    assert out["pit_safe"] is True


def test_lattice_any_false_is_false():
    out = lattice_join_semantic_attrs(
        [_attrs(True), _attrs(False), _attrs(True)]
    )
    assert out["pit_safe"] is False


def test_lattice_missing_declaration_is_unknown_not_safe():
    # R10 #6: a child WITHOUT a pit_safe declaration must NOT be treated as safe.
    child = {"semantic_kind": "price", "domain": "price"}
    assert "pit_safe" not in child
    out = lattice_join_semantic_attrs([_attrs(True), child, _attrs(True)])
    assert out["pit_safe"] is None


def test_lattice_unknown_propagates_not_silently_safe():
    out = lattice_join_semantic_attrs([_attrs(True), _attrs(None), _attrs(True)])
    assert out["pit_safe"] is None


def test_unknown_is_distinct_from_false():
    out = lattice_join_semantic_attrs([_attrs(False), _attrs(None)])
    assert out["pit_safe"] is False  # proven-unsafe dominates unknown
