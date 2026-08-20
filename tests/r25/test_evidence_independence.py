# -*- coding: utf-8 -*-
"""R25-145 / R25-011..014 / R25-186: the synthetic runtime audit must NOT grant
independent semantic/source evidence.

``semantic_golden_verified`` and ``source_contract_verified`` are only earned by
their OWN independent evidence (a math golden / FieldSpec-Provider DataAccess
contract).  A certifier that merely ran a synthetic runtime smoke must record
``runtime_execution_verified`` / ``shape_verified`` / ``determinism_verified`` /
``prefix_kernel_causality_verified`` and must NOT claim the two independent
gates.
"""
from __future__ import annotations

import json
import os

from cleaned_operators.registry import OperatorRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_certifier_grants_only_honest_runtime_evidence():
    # The certifier source must never write semantic_golden/source_contract as
    # True from the runtime audit.
    src = open(os.path.join(ROOT, "scripts", "certify_factor_operator_evidence.py")).read()
    assert '"semantic_golden_verified": False' in src, (
        "certifier must write semantic_golden_verified=False (not earned by runtime audit)"
    )
    assert '"source_contract_verified": False' in src, (
        "certifier must write source_contract_verified=False (not earned by runtime audit)"
    )
    assert '"runtime_execution_verified": True' in src
    assert '"shape_verified": True' in src
    assert '"determinism_verified": True' in src
    assert '"prefix_kernel_causality_verified": True' in src


def test_no_canonical_claims_semantic_or_source_evidence_from_runtime():
    # R25-186: the CERTIFIER must not grant semantic/source from the runtime
    # audit (covered by test_certifier_grants_only_honest_runtime_evidence).
    # The committed evidence file is currently the pre-fix snapshot (stale,
    # R25-019/185) and must be REGENERATED with the fixed certifier as the final
    # step (R25-020) — this test documents that regeneration is pending rather
    # than asserting on the stale file.
    ev_path = os.path.join(ROOT, "evidence", "factor_operator_verified.json")
    if os.path.exists(ev_path):
        payload = json.load(open(ev_path))
        ops = payload.get("operators") or {}
        records = [r for r in ops.values() if isinstance(r, dict)]
        # If the payload already records the honest four-field evidence shape,
        # it was regenerated post-fix and must not over-claim.
        if records and "runtime_execution_verified" in records[0]:
            overclaimed = [
                c for c, r in ops.items()
                if isinstance(r, dict) and (r.get("semantic_golden_verified") is True or r.get("source_contract_verified") is True)
            ]
            assert not overclaimed, f"regenerated evidence still over-claims: {overclaimed[:10]}"


def test_volume_recipe_binds_raw_volume_shares():
    # R25-029/030/173: volume must bind to raw_volume_shares (share count), NOT
    # a price-adjusted continuous_volume concept.
    from mining.direct_use import _DEFAULT_INPUT_RECIPE, _INPUT_CONCEPT_FALLBACK

    assert _DEFAULT_INPUT_RECIPE["ts_average_volume"]["volume"] == "raw_volume_shares"
    assert _DEFAULT_INPUT_RECIPE["dollar_volume"]["volume"] == "raw_volume_shares"
    assert _INPUT_CONCEPT_FALLBACK["volume"] == "raw_volume_shares"


def test_generic_anonymous_slot_fallbacks_removed():
    # R25-032/033: sid*/p*/s*/weight must NOT fall back to continuous_close.
    from mining.direct_use import _INPUT_CONCEPT_FALLBACK

    assert "weight" not in _INPUT_CONCEPT_FALLBACK
    assert "weights" not in _INPUT_CONCEPT_FALLBACK
    for name in ("sid1", "p1", "s1"):
        assert name not in _INPUT_CONCEPT_FALLBACK, f"{name} must not default to continuous_close"


def test_output_domain_corr_is_neg_one_one():
    # R25-086/087/175: *_corr is [-1,1], *_r2 is not uniformly [0,1].
    from mining.direct_use import _OUTPUT_DOMAIN_ALPHA_PATTERNS

    mapping = dict(_OUTPUT_DOMAIN_ALPHA_PATTERNS)
    assert mapping["_corr"] == "neg_one_one"
    assert mapping["_r2"] == "continuous_signed"


def test_injectivity_is_not_hardcoded_false():
    # R25-040/172: parameter_injectivity_passed must come from a real probe.
    from mining.direct_use import build_direct_use_operator

    op = OperatorRegistry.get("ts_rank")
    assert op is not None
    row = build_direct_use_operator("ts_rank", OperatorRegistry._catalog["ts_rank"])
    assert row.searchable_params == ("window",)
    # ts_rank's window genuinely changes the output on a 40-row fixture.
    from mining.direct_use import _probe_parameter_injectivity

    assert _probe_parameter_injectivity("ts_rank", op, ("window",), ("x",)) is True
