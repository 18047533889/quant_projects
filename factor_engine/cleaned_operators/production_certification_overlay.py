# -*- coding: utf-8 -*-
"""Derive Pandas production certification from immutable test evidence.

``status=production`` means an operator is a reviewed production *target*.
Physical backend admission is stronger: Daily primitives need primitive backend
evidence; non-Daily Factor operators need the factor-operator evidence artifact.
Metadata alone is never sufficient to route production traffic.
"""
from __future__ import annotations


def apply_evidence_certification_overlay() -> None:
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
            source = "primitive_verified.json"
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
