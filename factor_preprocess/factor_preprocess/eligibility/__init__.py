"""Eligibility package for the auto-treatment optimizer."""
from factor_preprocess.eligibility.engine import (
    TreatmentEligibilityEngine,
    TreatmentSearchSpace,
    PRICE_VOLUME,
    HIGH_TURNOVER,
    FUNDAMENTAL,
    SPARSE_UPDATE,
    EVENT,
    BINARY,
    DISCRETE,
    RAW_SEMANTIC_ID,
)
from factor_preprocess.eligibility.rules import (
    TreatmentFamily,
    FactorFamily,
    FamilyRule,
    EligibilityRuleRegistry,
    create_default_eligibility_rules,
    get_default_eligibility_rules,
)

__all__ = [
    "TreatmentEligibilityEngine",
    "TreatmentSearchSpace",
    "PRICE_VOLUME",
    "HIGH_TURNOVER",
    "FUNDAMENTAL",
    "SPARSE_UPDATE",
    "EVENT",
    "BINARY",
    "DISCRETE",
    "RAW_SEMANTIC_ID",
    "TreatmentFamily",
    "FactorFamily",
    "FamilyRule",
    "EligibilityRuleRegistry",
    "create_default_eligibility_rules",
    "get_default_eligibility_rules",
]
