# -*- coding: utf-8 -*-
"""Derive Pandas production certification from immutable test evidence.

``status=production`` means an operator is a reviewed production *target*.
Physical backend admission is stronger: Daily primitives need primitive backend
evidence; non-Daily Factor operators need the factor-operator evidence artifact.
Metadata alone is never sufficient to route production traffic.
"""
from __future__ import annotations


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
