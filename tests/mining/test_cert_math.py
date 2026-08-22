# -*- coding: utf-8 -*-
"""R23 P1 certification tests: math + elementwise_math (+ robust_statistics / information_theory).

Asserts that the operators certified by R23 P1 math-family certification have
production_admitted=True in the DirectUseOperator verdict.
"""
from __future__ import annotations

import pytest
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators import load_all
from mining.direct_use import build_direct_use_operator

# The canonicals certified by R23 P1 math-family certification
R23_CERTIFIED_MATH = frozenset({
    "abs", "ceil", "clip", "exp", "floor", "log", "log_abs",
    "maximum", "minimum", "neg", "normalize", "power", "sign",
    "sqrt", "tanh", "winsorize",
})
R23_CERTIFIED_ELEMENTWISE_MATH = frozenset({
    "add", "and_", "coalesce", "divide", "eq", "ge", "gt", "inverse",
    "le", "lt", "multiply", "ne", "not_", "or_", "signed_sqrt",
    "subtract", "where",
})
R23_CERTIFIED_CANONICALS = R23_CERTIFIED_MATH | R23_CERTIFIED_ELEMENTWISE_MATH


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


def test_all_certified_have_production_admitted():
    """Every R23 P1 certified math-family canonical must have production_admitted=True."""
    failures = []
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None, f"{canonical} not in registry catalog"
        row = build_direct_use_operator(canonical, catalog)
        if not row.production_admitted:
            failures.append(
                f"{canonical}: production_admitted={row.production_admitted} "
                f"(prod_certified={row.production_certified}, "
                f"mining_visible={row.mining_visible})"
            )
    assert not failures, (
        f"R23 P1 certified math-family operators with production_admitted=False:\n"
        + "\n".join(failures)
    )


def test_all_certified_have_production_certified():
    """Every R23 P1 certified math-family canonical must have production_certified=True."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("production_certified") is True, (
            f"{canonical}: catalog production_certified is not True"
        )


def test_all_certified_have_production_status():
    """Every R23 P1 certified math-family canonical must have status=production."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("status") == "production", (
            f"{canonical}: status is {catalog.get('status')!r}, expected 'production'"
        )


def test_all_certified_have_cost_tag():
    """Every certified math-family canonical must carry a cost: tier tag so
    cost_contract_declared() admits it."""
    import re

    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        tags = tuple(str(t) for t in (catalog.get("tags") or ()))
        assert any(re.fullmatch(r"cost:\d+", t.strip()) for t in tags), (
            f"{canonical}: no cost: tag in {tags}"
        )


def test_robust_stats_info_theory_not_certified():
    """robust_statistics / information_theory must NOT be certified: no six-way
    evidence in the artifact for those categories."""
    import json

    d = json.load(open("/home/shw/quant_projects/docs/R23_PER_CANONICAL_AUDIT.json"))
    cats = ("robust_statistics", "information_theory")
    scope = {x["canonical"] for x in d if x.get("category") in cats}
    for canonical in sorted(scope):
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        assert catalog.get("production_certified") is not True, (
            f"{canonical}: robust_statistics/information_theory must not be certified"
        )


def test_certified_count():
    """Exactly 33 operators are certified (16 math + 17 elementwise_math)."""
    catalog = OperatorRegistry._catalog
    actual = sum(
        1 for c in catalog if c in R23_CERTIFIED_CANONICALS
        and catalog[c].get("production_certified") is True
    )
    assert actual == len(R23_CERTIFIED_CANONICALS) == 33, (
        f"Expected 33 certified, got {actual}"
    )
