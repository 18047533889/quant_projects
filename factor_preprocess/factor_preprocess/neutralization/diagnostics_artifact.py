"""Neutralization diagnostics artifact (DLIB-FP-020 / DLIB-FP-021).

Every neutralization production run must emit at least:
effective_n, rank, condition_number, R², residual_variance,
missing_exposure_count, industry_coverage, solver, regularization, warnings.

The artifact is deep-immutable and content-hashable so it can serve as
audit evidence for the neutralization step of a treatment.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from collections.abc import Mapping
import hashlib

from factor_preprocess.contracts._deep_freeze import deep_freeze
from factor_preprocess.errors import InvalidContractError


def _stable_repr(value: Any) -> str:
    if isinstance(value, Mapping):
        value = dict(value)
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{_stable_repr(k)}:{_stable_repr(v)}"
            for k, v in sorted(value.items(), key=lambda kv: _stable_repr(kv[0]))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(item) for item in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "<" + ",".join(sorted(_stable_repr(item) for item in value)) + ">"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    raise TypeError(
        "_stable_repr does not support type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


class RankDeficientResolution(str):
    """How a rank-deficient neutralization was resolved."""

    FAIL = "FAIL"
    FALLBACK_REGULARIZED = "FALLBACK_REGULARIZED"
    WARN_CONTINUED = "WARN_CONTINUED"


@dataclass(frozen=True)
class NeutralizationDiagnostics:
    """Deep-immutable audit record of one neutralization run.

    A rank-deficient run must NOT silently continue: either the run FAILED or
    it fell back to a regularized solver, and the fallback is recorded here as
    audit evidence (DLIB-FP-021).
    """

    neutralization_ref: str
    effective_n: int = 0
    rank: Optional[int] = None
    condition_number: Optional[float] = None
    r_squared: Optional[float] = None
    residual_variance: Optional[float] = None
    missing_exposure_count: int = 0
    industry_coverage: Optional[float] = None
    solver: Optional[str] = None
    regularization: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    rank_deficient_resolution: Optional[str] = None
    artifact_evidence_ref: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.neutralization_ref:
            raise InvalidContractError("NeutralizationDiagnostics.neutralization_ref cannot be empty")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        actual = self._derive_content_hash()
        if self.content_hash and self.content_hash != actual:
            raise InvalidContractError(
                "NeutralizationDiagnostics.content_hash does not match content"
            )
        object.__setattr__(self, "content_hash", actual)

    def _derive_content_hash(self) -> str:
        components = {
            "neutralization_ref": self.neutralization_ref,
            "effective_n": self.effective_n,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "r_squared": self.r_squared,
            "residual_variance": self.residual_variance,
            "missing_exposure_count": self.missing_exposure_count,
            "industry_coverage": self.industry_coverage,
            "solver": self.solver,
            "regularization": self.regularization,
            "warnings": self.warnings,
            "rank_deficient_resolution": self.rank_deficient_resolution,
            "artifact_evidence_ref": self.artifact_evidence_ref,
        }
        return hashlib.sha256(_stable_repr(components).encode("utf-8")).hexdigest()


__all__ = [
    "NeutralizationDiagnostics",
    "RankDeficientResolution",
]
