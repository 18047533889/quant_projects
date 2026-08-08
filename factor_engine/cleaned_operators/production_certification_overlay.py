# -*- coding: utf-8 -*-
"""Derive Pandas production certification from immutable test evidence.

``status=production`` means an operator is a reviewed production *target*.
Physical backend admission is stronger: Daily primitives need primitive backend
evidence; non-Daily Factor operators need the factor-operator evidence artifact.
Metadata alone is never sufficient to route production traffic.
"""
from __future__ import annotations

from typing import Any


def _per_gate_verified(canonical: str) -> dict[str, bool]:
    """Return each gate's OWN independent evidence record from the payloads.

    P0-18: the semantic / temporal / source-contract certificates must each be
    earned by their own verified record (``semantic_golden_verified`` /
    ``temporal_prefix_verified`` / ``source_contract_verified``).  Implementation
    evidence proves the backend ran; it never back-stops these gates.  Records
    are read from the per-operator evidence payload when present (``operators``
    as a dict), and default to False when absent — fail-closed.
    """
    out = {
        "semantic_golden_verified": False,
        "temporal_prefix_verified": False,
        "source_contract_verified": False,
    }
    try:
        from backend.factor_operator_evidence import load_factor_operator_evidence

        payload = load_factor_operator_evidence() or {}
        operators = payload.get("operators")
        if isinstance(operators, dict):
            record = operators.get(canonical)
            if isinstance(record, dict):
                for key in out:
                    value = record.get(key)
                    if value is not None:
                        out[key] = bool(value)
    except Exception:
        pass
    return out


def _record_alias_evidence_origin(
    canonical: str, catalog: dict[str, Any], certified: bool
) -> None:
    """P0-19: record which canonical each alias derives from.

    Aliases/wrappers inherit the parent canonical's evidence; this records a
    per-alias ``evidence_origin`` marker so the inheritance is explicit rather
    than implicit.  An alias must never *upgrade* a parent's certification: if
    the parent is experimental / not evidence-certified, any alias record that
    exists is forced back to experimental / non-production.
    """
    from cleaned_operators.registry import OperatorRegistry

    aliases = list(catalog.get("aliases") or [])
    if aliases:
        catalog["alias_evidence_origins"] = {alias: canonical for alias in aliases}
    for alias in aliases:
        alias_entry = OperatorRegistry._catalog.get(alias)
        if alias_entry is None:
            # Most aliases have no catalog record of their own (finalize()
            # forbids alias/canonical collisions); the origin map above is the
            # durable per-alias marker.  Nothing to force closed.
            continue
        alias_entry["evidence_origin"] = canonical
        if not certified:
            # An alias of an uncertified parent can never claim production.
            alias_entry["production_certified"] = False
            alias_entry["pit_safe"] = False
            alias_entry["status"] = "experimental"
            alias_entry["lifecycle_status"] = "experimental"


def apply_evidence_certification_overlay() -> None:
    # Install the base-blob-bound evidence delta before any capability module
    # snapshots the verified primitive sets.  The delta loader remains
    # fail-closed: every override must equal the value recomputed from current
    # source and the immutable base JSON must match its recorded Git blob SHA.
    from backend.evidence_delta import install_evidence_delta

    install_evidence_delta()

    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry

    primitive_valid = False
    primitive_certified: set[str] = set()
    try:
        from backend.evidence_provenance import evidence_artifact_valid
        from backend.primitive_evidence import PRIMITIVE_BACKEND_EXECUTION_CERTIFIED

        primitive_valid = bool(evidence_artifact_valid())
        primitive_certified = set(PRIMITIVE_BACKEND_EXECUTION_CERTIFIED)
    except Exception:
        primitive_valid = False

    for canonical in sorted(factor_production_targets()):
        catalog = OperatorRegistry._catalog.get(canonical, {})
        meta = catalog.setdefault("backend_meta", {}).setdefault("pandas_numpy", {})
        if canonical in DAILY_CANONICALS:
            certified = primitive_valid and canonical in primitive_certified
            source = "primitive_verified.json+primitive_verified_delta.json"
        else:
            try:
                from backend.factor_operator_evidence import pandas_reference_production_safe

                certified = bool(pandas_reference_production_safe(canonical))
            except Exception:
                certified = False
            source = "factor_operator_verified.json"
        meta["production_certified"] = certified
        meta["certification_source"] = source
        meta["certification_tier"] = "pandas_reference" if certified else "candidate"
        meta["reference_backend"] = True

        # P0-18: bind each gate's OWN independent evidence.  Implementation
        # evidence (``certified``) is required but never sufficient for the
        # semantic/temporal/source gates; each needs its own payload record.
        per_gate = _per_gate_verified(canonical)
        for _key in (
            "semantic_golden_verified",
            "temporal_prefix_verified",
            "source_contract_verified",
        ):
            catalog[_key] = bool(certified) and per_gate.get(_key, False)

        # P0-19: per-alias evidence origin; an alias never upgrades its parent.
        _record_alias_evidence_origin(canonical, catalog, certified)

    # The evidence overlay is the sole promotion authority: converge the
    # six-gate composite and the top-level lifecycle fields from the now-bound
    # backend certification.  ``reconcile_operator_certification`` is the single
    # writer of ``production_certified`` / ``status`` / ``lifecycle_status`` /
    # ``pit_safe`` after this point (review §2.3, §2.4).
    from cleaned_operators.semantic_certification import (
        reconcile_operator_certification,
    )

    for canonical in sorted(factor_production_targets()):
        catalog = OperatorRegistry._catalog.get(canonical, {})
        reconcile_operator_certification(canonical, catalog)
