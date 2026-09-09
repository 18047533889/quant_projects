"""FO-side statistical-search governance contracts."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Optional, Tuple


class SplitPurpose(str, Enum):
    DIAGNOSTIC = "diagnostic"
    PREDICTION_VALIDATION = "prediction_validation"
    PRODUCTION_OOS = "production_oos"


@dataclass(frozen=True)
class SplitEvidenceRef:
    split_ref: str
    purpose: SplitPurpose
    chronological_forward: bool
    candidate_universe_complete: bool

    def require_production_oos(self) -> None:
        if self.purpose is not SplitPurpose.PRODUCTION_OOS or not self.chronological_forward:
            raise ValueError("diagnostic/non-forward split cannot satisfy production OOS")
        if not self.candidate_universe_complete:
            raise ValueError("winner-only evidence cannot satisfy production OOS")


@dataclass(frozen=True)
class RetentionEvidence:
    train_value: float
    validation_value: float
    signed_difference: float
    retention_ratio: Optional[float]
    ratio_applicable: bool
    sign_reversal: bool


def retention_evidence(train: float, validation: float, *, near_zero: float = 1e-3) -> RetentionEvidence:
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (train, validation, near_zero)) or near_zero <= 0:
        raise ValueError("retention inputs must be finite and near_zero positive")
    reversal = train * validation < 0
    applicable = abs(train) >= near_zero and not reversal
    return RetentionEvidence(
        float(train), float(validation), float(validation - train),
        float(validation / train) if applicable else None, applicable, reversal,
    )


@dataclass(frozen=True)
class HorizonDiagnosis:
    horizons: Tuple[int, ...]
    values: Tuple[float, ...]
    best_horizon: Optional[int]
    half_life: Optional[float]
    fit_status: str
    hypothesis_count: int


def diagnose_horizon_curve(values: Mapping[int, float]) -> HorizonDiagnosis:
    if len(values) < 2 or not all(isinstance(h, int) and h > 0 for h in values):
        raise ValueError("at least two positive horizons are required")
    ordered = tuple(sorted(values))
    series = tuple(float(values[h]) for h in ordered)
    if not all(math.isfinite(v) for v in series):
        raise ValueError("horizon values must be finite")
    signs = {1 if v > 0 else -1 if v < 0 else 0 for v in series} - {0}
    peaks = sum(1 for i in range(1, len(series) - 1) if series[i] > series[i-1] and series[i] > series[i+1])
    if len(signs) > 1:
        status, half_life = "SIGN_CHANGE", None
    elif peaks > 1:
        status, half_life = "MULTI_PEAK", None
    else:
        status, half_life = "NOT_FIT_WITHOUT_QE_MODEL", None
    best = ordered[max(range(len(series)), key=lambda i: abs(series[i]))]
    return HorizonDiagnosis(ordered, series, best, half_life, status, len(ordered))
