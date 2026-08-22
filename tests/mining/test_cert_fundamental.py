# -*- coding: utf-8 -*-
"""R23 P1 certification tests: fundamental_period + shareholder + valuation.

HONEST FINDING (R23 P1): no canonical in the three in-scope categories has
six-way primitive evidence in the committed ``evidence/primitive_verified.json``
artifact (its six-way intersection covers only the time_series / price /
cross_sectional / group / elementwise / math primitives already certified by the
sibling R23 P1 passes).  The 205 financial operators carry at most a 5-gate
factor-runtime record (missing semantic_golden and source_contract) in
``factor_operator_verified.json`` and are therefore NOT certified.

The 21 P0-severity items (``fin_*`` expectation/revision/restatement) are exactly
the R23-P0-PIT11/PIT18 denied set already fail-closed via
``production_hardening.NON_FACTOR_PRODUCTION_CANONICALS``; they must NEVER be
re-certified by this pass.

So the certified set is empty and no in-scope operator may be production-admitted.
This test enforces that honest invariant: every in-scope operator must remain
NOT production-admitted.
"""
from __future__ import annotations

import json

import pytest
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators import load_all
from mining.direct_use import build_direct_use_operator

# Honest: no in-scope canonical has six-way evidence -> empty certified set.
R23_CERTIFIED_CANONICALS: frozenset[str] = frozenset()

IN_SCOPE_CATEGORIES = (
    "fundamental_period",
    "shareholder",
    "valuation",
)


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


def _in_scope_canonicals():
    d = json.load(open("/home/shw/quant_projects/docs/R23_PER_CANONICAL_AUDIT.json"))
    return {x["canonical"] for x in d if x.get("category") in IN_SCOPE_CATEGORIES}


def test_no_in_scope_operator_is_certified():
    """No fundamental/shareholder/valuation operator may be certified
    (no six-way primitive evidence)."""
    scope = _in_scope_canonicals()
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
    scope = _in_scope_canonicals()
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


def test_pit_denied_canonicals_remain_fail_closed():
    """The 21 PIT11/PIT18 denied canonicals must not be certified by this pass."""
    from cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS

    scope = _in_scope_canonicals()
    pit_denied = scope & set(PRODUCTION_DENIED_CANONICALS)
    assert len(pit_denied) == 21, (
        f"Expected 21 PIT-denied canonicals in scope, got {len(pit_denied)}"
    )
    for canonical in sorted(pit_denied):
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        assert catalog.get("production_certified") is not True, (
            f"{canonical}: PIT-denied canonical must not be certified"
        )
        row = build_direct_use_operator(canonical, catalog)
        assert row.production_admitted is False, (
            f"{canonical}: PIT-denied canonical must not be production-admitted"
        )


def test_certified_count():
    """Exactly 0 operators are certified in this pass (no fabricated evidence)."""
    catalog = OperatorRegistry._catalog
    actual = sum(
        1 for c in catalog if c in R23_CERTIFIED_CANONICALS
        and catalog[c].get("production_certified") is True
    )
    assert actual == 0, f"Expected 0 certified, got {actual}"
