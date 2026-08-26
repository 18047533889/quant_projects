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
]
