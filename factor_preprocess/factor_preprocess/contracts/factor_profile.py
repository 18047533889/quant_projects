"""
Factor profile artifact contract for the auto-treatment optimizer.

A :class:`FactorProfileArtifact` is an immutable, content-hashed description
of a factor's statistical and behavioral characteristics. It is the *input*
to the treatment eligibility engine: it describes *what the factor is* so the
engine can decide *which treatments are legal*.

Data-scope guarantee
--------------------
Profile metrics (distribution, time_behavior, data_behavior, exposures) MUST
only be computed from authorized TRAIN data, or from an explicitly
unsupervised PIT-safe scope. Any metric that would require future information
or the test split is forbidden here. This is a governance contract, not just a
convention: the artifact carries ``split_ref`` so downstream consumers can
verify which split the profile was derived from.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from factor_preprocess.errors import InvalidContractError


def _stable_repr(value: Any) -> str:
    """Deterministic canonical string form for content hashing."""
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


def _content_hash(value: Any) -> str:
    """SHA-256 hex digest derived from the actual content of ``value``."""
    return hashlib.sha256(_stable_repr(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorProfileArtifact:
    """
    Immutable profile of a factor for treatment eligibility.

    Parameters
    ----------
    factor_id : str
        Unique identifier of the factor.
    factor_version : str
        Version of the factor definition.
    semantic_family : str
        High-level semantic family. One of the known families
        (``PRICE_VOLUME``, ``HIGH_TURNOVER``, ``FUNDAMENTAL``,
        ``SPARSE_UPDATE``, ``EVENT``, ``BINARY``, ``DISCRETE``) or a custom
        string. The eligibility engine matches on these known families.
    source_type : str
        Provenance of the factor (e.g. ``price``, ``fundamental``, ``event``).
    update_frequency : str
        Expected update cadence (e.g. ``daily``, ``weekly``, ``quarterly``).
    natural_horizon : int
        Natural lookback/decay horizon of the factor in periods.
    distribution : dict
        Distributional statistics: ``mean``, ``std``, ``skew``, ``kurtosis``,
        ``tailness``, ``unique_ratio``.
    time_behavior : dict
        Time-series behavior: ``autocorr_1``, ``autocorr_5``,
        ``rank_persistence``, ``raw_turnover``, ``signal_decay``,
        ``ic_decay``.
    data_behavior : dict
        Data-quality behavior: ``coverage``, ``missing_rate``, ``staleness``,
        ``update_gap``.
    exposures : dict
        Style exposures: ``industry``, ``size``, ``beta``, ``liquidity``,
        ``volatility``, ``momentum``, ``value``.
    existing_transform_lineage : tuple
        Ordered tuple of transform semantic IDs already applied to the factor.
    snapshot_ref : str, optional
        Reference to the data snapshot the profile was computed from.
    universe_ref : str, optional
        Reference to the universe used.
    split_ref : str, optional
        Reference to the train/test split used. Profile metrics must only use
        the TRAIN side (or an explicit unsupervised PIT-safe scope).
    content_hash : str
        Content-derived hash over the full profile surface. Computed in
        ``__post_init__``; a caller-supplied value is validated against the
        recomputation and rejected on mismatch (fail-closed).

    Notes
    -----
    Profile metrics must only use authorized TRAIN data or an explicit
    unsupervised PIT-safe scope. Any metric derived from the test split or
    from future information is a governance violation.
    """

    factor_id: str
    factor_version: str
    semantic_family: str
    source_type: str
    update_frequency: str
    natural_horizon: int

    distribution: Dict[str, float] = field(default_factory=dict)
    time_behavior: Dict[str, float] = field(default_factory=dict)
    data_behavior: Dict[str, float] = field(default_factory=dict)
    exposures: Dict[str, float] = field(default_factory=dict)

    existing_transform_lineage: Tuple[str, ...] = field(default_factory=tuple)

    snapshot_ref: Optional[str] = None
    universe_ref: Optional[str] = None
    split_ref: Optional[str] = None

    content_hash: str = ""

    def __post_init__(self):
        """Validate and derive the content hash."""
        if not self.factor_id:
            raise InvalidContractError("FactorProfileArtifact.factor_id cannot be empty")
        if not self.factor_version:
            raise InvalidContractError("FactorProfileArtifact.factor_version cannot be empty")
        if not self.semantic_family:
            raise InvalidContractError("FactorProfileArtifact.semantic_family cannot be empty")
        if self.natural_horizon < 0:
            raise InvalidContractError("natural_horizon must be >= 0")

        # Freeze mutable containers.
        object.__setattr__(self, "distribution", dict(self.distribution))
        object.__setattr__(self, "time_behavior", dict(self.time_behavior))
        object.__setattr__(self, "data_behavior", dict(self.data_behavior))
        object.__setattr__(self, "exposures", dict(self.exposures))
        object.__setattr__(
            self, "existing_transform_lineage", tuple(self.existing_transform_lineage)
        )

        actual_hash = self._derive_content_hash()
        if self.content_hash and self.content_hash != actual_hash:
            raise InvalidContractError(
                "FactorProfileArtifact.content_hash does not match profile content"
            )
        object.__setattr__(self, "content_hash", actual_hash)

    def _derive_content_hash(self) -> str:
        """Content-derived identity over the full profile surface."""
        components = {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "semantic_family": self.semantic_family,
            "source_type": self.source_type,
            "update_frequency": self.update_frequency,
            "natural_horizon": self.natural_horizon,
            "distribution": self.distribution,
            "time_behavior": self.time_behavior,
            "data_behavior": self.data_behavior,
            "exposures": self.exposures,
            "existing_transform_lineage": self.existing_transform_lineage,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "split_ref": self.split_ref,
        }
        return _content_hash(components)

    def with_lineage(self, lineage: Tuple[str, ...]) -> "FactorProfileArtifact":
        """Return a copy with a replaced transform lineage (content hash updated)."""
        return FactorProfileArtifact(
            factor_id=self.factor_id,
            factor_version=self.factor_version,
            semantic_family=self.semantic_family,
            source_type=self.source_type,
            update_frequency=self.update_frequency,
            natural_horizon=self.natural_horizon,
            distribution=self.distribution,
            time_behavior=self.time_behavior,
            data_behavior=self.data_behavior,
            exposures=self.exposures,
            existing_transform_lineage=lineage,
            snapshot_ref=self.snapshot_ref,
            universe_ref=self.universe_ref,
            split_ref=self.split_ref,
        )


__all__ = ["FactorProfileArtifact"]
