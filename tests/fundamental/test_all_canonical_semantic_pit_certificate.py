# -*- coding: utf-8 -*-
"""R23-001/002/003/263/264: every canonical has an explicit semantic/PIT
certificate with a legal final state — none may be UNKNOWN / SKIPPED / except-
swallowed, and the audit row count must equal the canonical count.
"""
from __future__ import annotations

import os

import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

LEGAL_STATES = frozenset({
    "CERTIFIED",
    "CERTIFIED_CONTEXTUAL",
    "CERTIFIED_HIGH_COST",
    "SUPPORTING_ONLY",
    "RESEARCH_TOOL",
    "INTERNAL_ONLY",
    "DELETE",
    "BLOCKED_NO_DATA",
})
FORBIDDEN_STATES = frozenset({"UNKNOWN", "MAYBE", "TODO", "PENDING_WITHOUT_ACTION", "SKIPPED"})
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
AUDIT_CSV = os.path.join(DOCS, "R23_PER_CANONICAL_AUDIT.csv")


def test_canonical_audit_covers_every_canonical():
    load_all()
    canonicals = sorted(OperatorRegistry._catalog)
    assert os.path.exists(AUDIT_CSV), "R23_PER_CANONICAL_AUDIT.csv must exist (run scripts/r23_audit_generate.py)"
    df = pd.read_csv(AUDIT_CSV)
    assert len(df) == len(canonicals), (
        f"audit row count ({len(df)}) must equal canonical count ({len(canonicals)}) — "
        "a canonical was skipped"
    )
    assert set(df["canonical"]) == set(canonicals), "canonical set mismatch"


def test_no_canonical_is_unknown_or_skipped():
    assert os.path.exists(AUDIT_CSV), "run scripts/r23_audit_generate.py first"
    df = pd.read_csv(AUDIT_CSV)
    bad = df[df["final_status"].isin(FORBIDDEN_STATES)]
    assert len(bad) == 0, f"forbidden certificate states: {bad['canonical'].tolist()}"


def test_every_certificate_state_is_legal():
    assert os.path.exists(AUDIT_CSV), "run scripts/r23_audit_generate.py first"
    df = pd.read_csv(AUDIT_CSV)
    illegal = set(df["final_status"]) - LEGAL_STATES
    assert not illegal, f"illegal certificate states: {illegal}"


def test_fundamental_audit_covers_all_fundamental_canonicals():
    load_all()
    canonicals = sorted(OperatorRegistry._catalog)
    fund = [
        c for c in canonicals
        if str(OperatorRegistry._catalog[c].get("category", "")) in {"fundamental", "fundamental_period"}
    ]
    fund_csv = os.path.join(DOCS, "R23_FUNDAMENTAL_OPERATOR_AUDIT.csv")
    assert os.path.exists(fund_csv), "run scripts/r23_audit_generate.py first"
    df = pd.read_csv(fund_csv)
    assert len(df) == len(fund), f"fundamental audit rows ({len(df)}) != fundamental canonicals ({len(fund)})"
