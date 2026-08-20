# -*- coding: utf-8 -*-
"""R21-P0-MATRIX-TRUTH smoke: generator runs; matrix non-empty and self-consistent.

Asserts that the counts summary is derivable from the matrix rows themselves
(no hand-written numbers can drift), and that fail-closed fields stay honest.
"""
from __future__ import annotations

import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[_var] = "1"

import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(FE_ROOT.parent), str(FE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.generate_physical_implementation_matrix import (  # noqa: E402
    build_direct_use_rows,
    build_matrix,
    summarize,
)


def test_matrix_non_empty_and_fields_present() -> None:
    rows = build_matrix()
    assert rows, "physical implementation matrix must not be empty"
    required = {
        "canonical",
        "registry_slot",
        "physical_backend",
        "implementation_id",
        "execution_kind",
        "implemented",
        "selectable",
        "spec_complete",
        "oracle_passed",
        "edge_passed",
        "parity_passed",
        "production_evidence_passed",
        "production_admitted",
        "source_hash",
        "semantic_hash",
        "parameter_domain_hash",
        "current_head_sha",
    }
    for row in rows[:50]:
        assert required <= set(row), f"row missing fields: {sorted(required - set(row))}"
    assert len({row["current_head_sha"] for row in rows}) == 1


def test_summary_derives_from_rows() -> None:
    rows = build_matrix()
    summary = summarize(rows, build_direct_use_rows())
    assert summary["total_rows"] == len(rows)
    assert summary["total_canonicals"] == len({r["canonical"] for r in rows})
    for backend, counts in summary["backends"].items():
        members = [r for r in rows if r["physical_backend"] == backend]
        assert counts["implemented"] == len(members)


def test_fail_closed_truth() -> None:
    """Missing spec ⇒ no implementation_id and spec_complete=false (R21 §28)."""
    rows = build_matrix()
    for row in rows:
        if not row["implementation_id"]:
            assert not row["spec_complete"], row["canonical"]
            assert not row["production_admitted"], row["canonical"]
        # evidence-backed oracle verdicts only; never fabricated
        assert row["oracle_passed"] in {"yes", "no", "NOT_RUN"}
    # production admission must be the strictest intersection
    admitted = [
        r
        for r in rows
        if r["production_admitted"]
    ]
    for row in admitted:
        assert row["spec_complete"] and row["implementation_id"]


def test_direct_use_split_monotone() -> None:
    direct_rows = build_direct_use_rows()
    assert direct_rows
    for row in direct_rows:
        # composition ⇒ mining_visible; terminal ⇒ composition (R22-003..008)
        if row["composition_usable"]:
            assert row["mining_visible"], row["canonical"]
        if row["terminal_usable"]:
            assert row["composition_usable"], row["canonical"]
        if row["directly_usable"]:
            assert row["mining_visible"] and row["production_admitted"]
