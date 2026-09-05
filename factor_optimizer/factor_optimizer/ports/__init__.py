"""Ports: narrow consumer-side contracts FO binds other packages through.

FO occupies the middle of the pipeline (``factor_engine`` -> ``quant_evaluator``
-> ``factor_preprocess`` -> ``factor_optimizer`` -> ``factor_assets``) but never
imports its neighbours directly — every integration goes through an adapter
protocol.  ``ports/`` holds those *consumer-boundary* protocols (a service-port
/ driving-side contract): what FO needs from an upstream package, expressed in
FO's own vocabulary, with the adapter implementations behind the provider
boundary in :mod:`factor_optimizer.adapters`.

The single seam defined here, R61-FI-014, is the Factor Intelligence provider
boundary (plan §20 E6 / matrix E6): FA is the canonical authority for factor
taxonomy/health/diagnosis, and FO must *consume views* — never recompute grades
or re-implement thresholds.
"""

from factor_optimizer.ports.factor_intelligence import (
    DiagnosisSeverity,
    FactorHealthView,
    FactorIntelligenceProvider,
    FactorIntelligenceUnknownFactorError,
    FactorIntelligenceView,
    FactorTaxonomyView,
    HealthDimension,
    HealthGrade,
    DiagnosisView,
)

__all__ = [
    "FactorIntelligenceProvider",
    "FactorTaxonomyView",
    "FactorHealthView",
    "DiagnosisView",
    "HealthDimension",
    "HealthGrade",
    "DiagnosisSeverity",
    "FactorIntelligenceView",
    "FactorIntelligenceUnknownFactorError",
]