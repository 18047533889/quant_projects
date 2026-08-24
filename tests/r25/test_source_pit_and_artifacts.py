# -*- coding: utf-8 -*-
"""R25-153/179/180/159: cross-market source certificates, strict_pit
contradiction, and artifact snapshot coherence.
"""
from __future__ import annotations

import json
import os

from factor_engine.fields.catalog import ASHARE_TABLE_SPECS
from factor_engine.fields.catalog_us import US_TABLE_SPECS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cross_market_source_cert_both_markets_present():
    # R25-064..066/153: the temporal source certificate keys A/US same-name
    # tables separately — both markets must be present, never one overriding
    # the other.
    cert_path = os.path.join(ROOT, "docs", "R23_TEMPORAL_SOURCE_CERTIFICATES.json")
    assert os.path.exists(cert_path), "run scripts/r23_audit_generate.py first"
    certs = json.load(open(cert_path))
    for name in ("StockIncome", "StockBalance", "StockDailyBar", "StockList", "Calendar"):
        markets = sorted({v["market"] for v in certs.values() if v["table"] == name})
        assert markets == ["ashare", "us"], f"{name}: markets={markets} (must include BOTH A and US)"
    # keys use market::dataset::table (no bare-table collision)
    keys = list(certs)
    assert all(k.count("::") == 2 for k in keys), "keys must be market::dataset::table"


def test_no_strict_pit_true_with_unproven_knowledge_clock():
    # R25-067/068/180: a table that claims strict_pit_allowed=True must carry an
    # explicit knowledge-time resolution (or N/A_EXACT_OBSERVATION for exact
    # observed data) — never a bare UNPROVEN contradiction.
    cert_path = os.path.join(ROOT, "docs", "R23_TEMPORAL_SOURCE_CERTIFICATES.json")
    assert os.path.exists(cert_path), "run scripts/r23_audit_generate.py first"
    certs = json.load(open(cert_path))
    contradictions = []
    for key, cert in certs.items():
        if cert["strict_pit_allowed"] and cert["knowledge_time_resolution"] == "UNPROVEN":
            contradictions.append(key)
    assert not contradictions, f"strict_pit=True + UNPROVEN resolution: {contradictions}"


def test_revision_operators_blocked_without_vintage_source():
    # R25-069..072/154: revision family declares requires:RevisionEventSource +
    # revision_vintage_pit_certified=false (R23) — production admission must
    # not pass them.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    for name in ("fin_revision_delta", "fin_revision_pct", "fin_restated_flag",
                 "fin_days_since_update", "fin_staleness"):
        ops = OperatorRegistry._operators.get(name, {})
        pd_op = ops.get("pandas_numpy")
        assert pd_op is not None, name
        tags = list(getattr(pd_op.metadata, "tags", None) or [])
        assert "requires:RevisionEventSource" in tags, name
        assert "revision_vintage_pit_certified:false" in tags, name


def test_artifact_snapshot_digest_coherent():
    # R25-023/159: R25_CANONICAL_SNAPSHOT must exist and its count must match
    # the live registry count.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    snapshot_path = os.path.join(ROOT, "docs", "R25_CANONICAL_SNAPSHOT.json")
    assert os.path.exists(snapshot_path), "run scripts/audit_r25_genuine_usability.py first"
    snapshot = json.load(open(snapshot_path))
    assert snapshot["count"] == len(OperatorRegistry._catalog), "snapshot count != live registry count"


def test_genuine_usability_requires_independent_evidence():
    # R25-010: a canonical is genuinely usable only when semantic_golden AND
    # source_contract are independently proven.  With the R25-186 certifier fix
    # neither is granted by the runtime audit, so the honest matrix must not
    # mark everything usable.
    matrix_path = os.path.join(ROOT, "docs", "R25_GENUINE_USABILITY_MATRIX.json")
    assert os.path.exists(matrix_path), "run scripts/audit_r25_genuine_usability.py first"
    matrix = json.load(open(matrix_path))
    claimed = [r["canonical"] for r in matrix if r["genuine_usable"]]
    for name in claimed:
        entry = [r for r in matrix if r["canonical"] == name][0]
        assert entry["semantic_golden_verified"] is True, name
        assert entry["source_contract_verified"] is True, name


def test_state_machine_no_silent_int_coercion():
    # R25-005/167: A-share state-machine must use strict integer authority.
    src = open(os.path.join(ROOT, "cleaned_operators", "ashare", "state_machine.py")).read()
    assert "strict_nonnegative_int(max_lookback, \"max_lookback\")" in src
    assert "int(max_lookback)" not in src, "silent int() coercion must be gone"
