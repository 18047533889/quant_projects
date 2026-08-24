"""Unified artifact model for the research platform.

Pure standard-library module. Does not import any package from the wider
quant_projects tree; it only depends on ``dataclasses``, ``hashlib``,
``typing`` and ``datetime``.

Design notes
------------
- Every artifact is a frozen dataclass. Immutability keeps hashes stable and
  makes artifacts safe to share across graph nodes / subagents.
- ``content_hash`` is the SHA-256 over the *semantic* fields only (all fields
  except the artifact id, the hash itself and ``created_at``). Two artifacts
  with identical semantic payload hash identically regardless of their id or
  the instant they were created, which lets us detect result-identity
  duplicates cheaply.
- ``to_dict`` / ``from_dict`` are symmetric and used as the serialization
  boundary (JSON-safe).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Version marker
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0"

# Type alias for the primitive JSON-ish representation we serialize to.
ArtifactDict = Dict[str, Any]


def _to_primitive(value: Any) -> Any:
    """Recursively convert a dataclass/date into JSON-safe primitives."""
    if is_dataclass(value):
        return {
            f.name: _to_primitive(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_to_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    return value


def utcnow() -> datetime:
    """Timezone-aware UTC now (naive datetimes are refused everywhere)."""
    return datetime.now(timezone.utc)


def _assert_tz(value: Optional[datetime], name: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC)")


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


def _dt_from_str(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _dt_to_primitive(value: Optional[datetime]) -> Any:
    return _dt_to_str(value)


def _dt_from_primitive(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    return _dt_from_str(value)


# ---------------------------------------------------------------------------
# Base artifact
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact:
    """Base class for all research-platform artifacts.

    Subclasses may add extra semantic fields. The base supplies the identity,
    provenance, content-hash, and serialization machinery.
    """

    artifact_id: str
    artifact_type: str
    schema_version: str = SCHEMA_VERSION
    content_hash: str = ""
    producer: str = ""
    producer_version: str = ""
    source_refs: tuple = ()
    data_snapshot_ref: str = ""
    universe_ref: str = ""
    policy_ref: str = ""
    environment_ref: str = ""
    created_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> ArtifactDict:
        d: ArtifactDict = {}
        for f in fields(self):
            if f.name == "created_at":
                d[f.name] = _dt_to_str(getattr(self, f.name))
            elif f.name == "content_hash":
                # persist the computed hash so round-trips are stable
                d[f.name] = getattr(self, f.name)
            else:
                d[f.name] = _to_primitive(getattr(self, f.name))
        return d

    @classmethod
    def from_dict(cls, data: ArtifactDict) -> "Artifact":
        kwargs: Dict[str, Any] = dict(data)
        kwargs["created_at"] = _dt_from_primitive(kwargs.get("created_at"))
        kwargs["source_refs"] = tuple(kwargs.get("source_refs", ()) or ())
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Content hash
    # ------------------------------------------------------------------
    def compute_content_hash(self) -> str:
        """SHA-256 over semantic fields (excludes id/hash/created_at)."""
        h = hashlib.sha256()
        h.update(self.artifact_type.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.schema_version.encode("utf-8"))
        h.update(b"\x00")
        for f in fields(self):
            name = f.name
            if name in ("artifact_id", "content_hash", "created_at"):
                continue
            value = getattr(self, name)
            h.update(name.encode("utf-8"))
            h.update(b"=")
            h.update(str(_to_primitive(value)).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def with_hash(self) -> "Artifact":
        """Return a copy with ``content_hash`` populated from semantics."""
        return _rebuild(self, content_hash=self.compute_content_hash())

    @property
    def is_hashed(self) -> bool:
        return bool(self.content_hash)


def _rebuild(art: Artifact, **overrides: Any) -> Artifact:
    """Construct a same-class copy with a few fields overridden."""
    values = {
        f.name: getattr(art, f.name)
        for f in fields(art)
        if hasattr(art, f.name)
    }
    values.update(overrides)
    return type(art)(**values)


# ---------------------------------------------------------------------------
# Domain subclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataSnapshotArtifact(Artifact):
    """A point-in-time data snapshot referenced by many downstream artifacts."""

    snapshot_ref: str = ""
    source_system: str = ""
    instrument_count: int = 0
    field_list: tuple = ()
    asof: Optional[datetime] = None
    manifest_hash: str = ""
    coverage_summary: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", _normalize_dt(self.asof))
        object.__setattr__(self, "snapshot_ref", self.snapshot_ref or self.artifact_id)
        _hash_tz(self.asof, "asof")


@dataclass(frozen=True)
class FactorDefinitionArtifact(Artifact):
    """Definition of a factor: its formula + operator provenance."""

    factor_id: str = ""
    formula: str = ""
    operator: str = ""
    operator_version: str = ""
    input_fields: tuple = ()
    params: Dict[str, Any] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FactorValueArtifact(Artifact):
    """Computed factor values for a snapshot."""

    factor_id: str = ""
    snapshot_ref: str = ""
    value_format: str = ""
    row_count: int = 0


@dataclass(frozen=True)
class EvaluationBundle(Artifact):
    """Bundle of evaluation metrics for a factor over a snapshot."""

    factor_id: str = ""
    snapshot_ref: str = ""
    metrics: Dict[str, Any] = None  # type: ignore[assignment]
    ic: float = 0.0
    icir: float = 0.0


@dataclass(frozen=True)
class SimilarityArtifact(Artifact):
    """Similarity matrix / fingerprints among a set of factor candidates."""

    fingerprint_version: str = ""
    candidate_ids: tuple = ()
    pairs: tuple = ()  # (id_a, id_b, similarity) tuples
    method: str = ""


@dataclass(frozen=True)
class AdmissionDecisionArtifact(Artifact):
    """Record of a factor candidate admission / rejection decision."""

    candidate_id: str = ""
    decision: str = ""  # 'admit' | 'reject'
    stage: str = ""
    reason: str = ""
    policy_ref: str = ""


@dataclass(frozen=True)
class FactorSetArtifact(Artifact):
    """A named, ordered collection of factors (a candidate universe)."""

    factor_ids: tuple = ()
    description: str = ""


@dataclass(frozen=True)
class FeatureBundle(Artifact):
    """Bundle of extracted features for modeling."""

    feature_set: tuple = ()
    feature_count: int = 0
    source_factors: tuple = ()


@dataclass(frozen=True)
class ModelDatasetArtifact(Artifact):
    """A prepared model training / evaluation dataset."""

    dataset_ref: str = ""
    rows: int = 0
    features: tuple = ()
    label: str = ""
    split: Dict[str, int] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelTrainingArtifact(Artifact):
    """Record of a single model training run."""

    model_ref: str = ""
    model_type: str = ""
    dataset_ref: str = ""
    hyperparams: Dict[str, Any] = None  # type: ignore[assignment]
    train_metrics: Dict[str, Any] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BacktestArtifact(Artifact):
    """A backtest run over a strategy built on admitted factors."""

    backtest_ref: str = ""
    strategy_ref: str = ""
    universe_ref: str = ""
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metrics: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "period_start", _normalize_dt(self.period_start))
        object.__setattr__(self, "period_end", _normalize_dt(self.period_end))
        _hash_tz(self.period_start, "period_start")
        _hash_tz(self.period_end, "period_end")


@dataclass(frozen=True)
class PortfolioArtifact(Artifact):
    """A realized (or simulated) portfolio / allocation snapshot."""

    portfolio_ref: str = ""
    holdings: Dict[str, float] = None  # type: ignore[assignment]
    weights_hash: str = ""
    nav: float = 0.0


@dataclass(frozen=True)
class RiskSnapshotArtifact(Artifact):
    """A risk model snapshot (exposures, factor risk, variance)."""

    snapshot_ref: str = ""
    risk_model_ref: str = ""
    exposures: Dict[str, Any] = None  # type: ignore[assignment]
    total_var: float = 0.0


@dataclass(frozen=True)
class ExpectedReturnArtifact(Artifact):
    """An expected-return estimate, produced by a model for a universe."""

    universe_ref: str = ""
    model_ref: str = ""
    estimator: str = ""
    mean_expected_return: float = 0.0
    per_name: Dict[str, Any] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers used by __post_init__
# ---------------------------------------------------------------------------
def _normalize_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return _dt_from_str(value)
    return value


def _hash_tz(value: Optional[datetime], name: str) -> None:
    _hash_tz_impl(value, name)


def _hash_tz_impl(value: Optional[datetime], name: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (got naive {value})")


# Registry for validation tests
ARTIFACT_TYPES: Dict[str, Any] = {
    cls.__name__: cls
    for cls in (
        DataSnapshotArtifact,
        FactorDefinitionArtifact,
        FactorValueArtifact,
        EvaluationBundle,
        SimilarityArtifact,
        AdmissionDecisionArtifact,
        FactorSetArtifact,
        FeatureBundle,
        ModelDatasetArtifact,
        ModelTrainingArtifact,
        BacktestArtifact,
        PortfolioArtifact,
        RiskSnapshotArtifact,
        ExpectedReturnArtifact,
    )
}
