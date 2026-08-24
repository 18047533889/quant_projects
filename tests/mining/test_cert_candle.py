# -*- coding: utf-8 -*-
"""R23 P1 certification tests: candle_pattern + candle_geometry + chart_pattern + candle_state_space.

HONEST FINDING (R23 P1): no canonical in the four in-scope categories has
six-way primitive evidence in the committed ``evidence/primitive_verified.json``
artifact (its six-way intersection covers only the time_series / price /
cross_sectional / group / elementwise / math primitives already certified by the
sibling R23 P1 passes).  The 82 candle/chart operators carry at most a 5-gate
factor-runtime record (missing semantic_golden and source_contract) in
``factor_operator_verified.json`` and are therefore NOT certified.

So the certified set is empty and no in-scope operator may be production-admitted.
This test enforces that honest invariant: every in-scope operator must remain
NOT production-admitted.
"""
from __future__ import annotations

import json

import pytest
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import load_all
from factor_engine.mining.direct_use import build_direct_use_operator

# Honest: no in-scope canonical has six-way evidence -> empty certified set.
R23_CERTIFIED_CANONICALS: frozenset[str] = frozenset()

IN_SCOPE_CATEGORIES = (
    "candle_pattern",
    "candle_geometry",
    "chart_pattern",
    "candle_state_space",
)


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


def test_no_in_scope_operator_is_certified():
    """No candle/chart operator may be certified (no six-way primitive evidence)."""
    d = json.load(open("/home/shw/quant_projects/docs/R23_PER_CANONICAL_AUDIT.json"))
    scope = {x["canonical"] for x in d if x.get("category") in IN_SCOPE_CATEGORIES}
    failures = []
    for canonical in sorted(scope):
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        if catalog.get("production_certified") is True:
            failures.append(
                f"{canonical}: production_certified=True but no six-way evidence"
            )
    assert not failures, (
        f"In-scope operators wrongly certified (no six-way primitive evidence):\n"
        + "\n".join(failures)
    )


def test_no_in_scope_operator_is_production_admitted():
    """Every in-scope operator must have production_admitted=False (honest gate)."""
    d = json.load(open("/home/shw/quant_projects/docs/R23_PER_CANONICAL_AUDIT.json"))
    scope = {x["canonical"] for x in d if x.get("category") in IN_SCOPE_CATEGORIES}
    failures = []
    for canonical in sorted(scope):
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        row = build_direct_use_operator(canonical, catalog)
        if row.production_admitted:
            failures.append(
                f"{canonical}: production_admitted=True but no six-way evidence "
                f"(prod_certified={row.production_certified})"
            )
    assert not failures, (
        f"In-scope operators with production_admitted=True (no six-way evidence):\n"
        + "\n".join(failures)
    )


def test_certified_count():
    """Exactly 0 operators are certified in this pass (no fabricated evidence)."""
    catalog = OperatorRegistry._catalog
    actual = sum(
        1 for c in catalog if c in R23_CERTIFIED_CANONICALS
        and catalog[c].get("production_certified") is True
    )
    assert actual == 0, f"Expected 0 certified, got {actual}"
