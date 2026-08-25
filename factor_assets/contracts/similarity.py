"""
Similarity artifact contract.

A complete, self-describing record of the similarity between two factors,
capturing *which* similarity views were computed, on *which* data snapshot,
universe and window, and who produced them.  Unlike a bare ``float`` from
:class:`factor_assets.similarity.exact.SimilarityResult`, a
:class:`SimilarityArtifact` carries the full provenance needed to reproduce
the measurement — the exact same problem the ``diverse`` (MMR) policy had
when it consumed a naked ``Optional[float]``.

Legacy similarity computations (:class:`factor_assets.similarity`) are
retained untouched.  This contract is the *new* typed boundary that consumers
(such as :class:`factor_assets.assembly.engine.FactorSetAssembler` in
production mode) may adopt to carry snapshot/window/universe provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Optional

__all__ = [
    "SimilarityArtifact",
    "SimilarityView",
    "UNKNOWN_SIMILARITY",
]

#: Canonical similarity view keys understood by consumers.  ``rank_corr`` is
#: the primary view used by MMR; the remaining keys are common residual /
#: overlap / horizon views a similarity producer may populate.
SIMILARITY_VIEW_KEYS = (
    "rank_corr",
    "pnl_corr",
    "top_overlap",
    "bottom_overlap",
    "residual_similarity",
    "horizon_similarity",
)

DEFAULT_SIMILARITY_VIEW = "rank_corr"


@dataclass(frozen=True)
class SimilarityView:
    """One named, self-describing similarity measurement."""

    key: str
    value: float
    #: Lower is better / absolute-value semantics for the ``diverse`` MMR
    #: consumer: ``False`` means the view is already canonical (e.g. cosine,
    #: correlation magnitude) and ``abs()`` must not be applied twice.
    is_absolute: bool = True

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key is required")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError("value must be a non-boolean number")
        value = float(self.value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("value must be finite (NaN/±inf is not a similarity)")


def _length_prefixed(digest: "hashlib._Hash", field_value: object) -> None:
    """Hash a field with length-prefixing so delimiters cannot collide."""
    encoded = str(field_value).encode("utf-8")
    digest.update(str(len(encoded)).encode("ascii"))
    digest.update(b":")
    digest.update(encoded)


def _deep_freeze_views(
    views: dict[str, Optional[float]],
) -> Mapping[str, Optional[float]]:
    """Deep-freeze a normalized views mapping into an immutable snapshot.

    The returned mapping is a :class:`types.MappingProxyType` backed by a
    private copy of the caller's dict, so no mutation of the caller's dict
    (nor of the mapping itself) can change an artifact after construction.
    """
    snapshot = dict(views)
    return MappingProxyType(snapshot)


#: Sentinel distinguishing an explicitly ``UNKNOWN`` measurement from a
#: genuinely computed zero.  ``UNKNOWN`` means the view was not measured and
#: MUST NOT be conflated with ``COMPUTED_ZERO`` (similarity 0.0) by consumers
#: such as the diverse (MMR) policy.
UNKNOWN_SIMILARITY = None


def _similarity_spec_digest(
    factor_a: str,
    factor_b: str,
    views: Mapping[str, Optional[float]],
    snapshot_ref: Optional[str],
    window_ref: Optional[str],
    universe_ref: Optional[str],
) -> str:
    """sha256 over the semantic fields that define a similarity measurement.

    Covers the factor pair, every view (including its None-ness), and the
    snapshot/window/universe provenance.  Two artifacts with the same
    semantic content but different producer or timestamp share the same
    ``similarity_spec_hash``; a different data snapshot or view set yields a
    different hash.
    """
    digest = hashlib.sha256()
    _length_prefixed(digest, factor_a)
    _length_prefixed(digest, factor_b)
    for key in sorted(views):
        _length_prefixed(digest, key)
        _length_prefixed(digest, views[key])
    _length_prefixed(digest, snapshot_ref)
    _length_prefixed(digest, window_ref)
    _length_prefixed(digest, universe_ref)
    return digest.hexdigest()


@dataclass(frozen=True)
class SimilarityArtifact:
    """Complete, hash-addressed record of pairwise factor similarity.

    Fields:
        factor_a: First factor ID.
        factor_b: Second factor ID.
        views: Named similarity views.  Keys follow
            :data:`SIMILARITY_VIEW_KEYS` (``rank_corr``, ``pnl_corr``,
            ``top_overlap``, ``bottom_overlap``, ``residual_similarity``,
            ``horizon_similarity``); values are finite floats or ``None``
            when the view was not measurable.  At least one view must be
            non-None.
        snapshot_ref: Data snapshot the similarity was computed on.
        window_ref: Evaluation window (e.g. ``"2024-01-01/2024-12-31"``).
        universe_ref: Universe the measurement was restricted to.
        similarity_spec_hash: sha256 over the semantic fields (factor pair,
            views, snapshot/window/universe).  Computed automatically when
            not supplied.
        producer: Identifier of the similarity producer.
        created_at: ISO 8601 creation timestamp.
        primary_view: Which view key MMR should consume as the similarity
            value.  Defaults to ``rank_corr``; ``None`` means the only
            non-None view is the primary view.
    """

    factor_a: str
    factor_b: str
    views: Mapping[str, Optional[float]]
    snapshot_ref: Optional[str] = None
    window_ref: Optional[str] = None
    universe_ref: Optional[str] = None
    similarity_spec_hash: str = ""
    producer: Optional[str] = None
    created_at: Optional[str] = None
    primary_view: Optional[str] = DEFAULT_SIMILARITY_VIEW

    def __post_init__(self) -> None:
        if not self.factor_a:
            raise ValueError("factor_a is required")
        if not self.factor_b:
            raise ValueError("factor_b is required")
        if self.factor_a == self.factor_b:
            raise ValueError("factor_a and factor_b must differ")

        if self.views is None:
            raise TypeError("views is required")
        normalized: dict[str, Optional[float]] = {}
        for key, value in self.views.items():
            if not isinstance(key, str) or not key:
                raise TypeError("views keys must be non-empty strings")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"views[{key!r}] must be a non-boolean number or None")
                value = float(value)
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError(f"views[{key!r}] must be finite")
            normalized[key] = value
        if not any(value is not None for value in normalized.values()):
            raise ValueError("views must contain at least one non-None similarity")
        object.__setattr__(self, "views", _deep_freeze_views(normalized))

        computed_hash = _similarity_spec_digest(
            self.factor_a,
            self.factor_b,
            self.views,
            self.snapshot_ref,
            self.window_ref,
            self.universe_ref,
        )
        if not self.similarity_spec_hash:
            object.__setattr__(self, "similarity_spec_hash", computed_hash)
        elif self.similarity_spec_hash != computed_hash:
            raise ValueError(
                "similarity_spec_hash does not match the recomputed spec hash "
                "(factor pair / views / snapshot / window / universe); a caller "
                "may not self-report an arbitrary hash — FAIL CLOSED"
            )
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

        if self.primary_view is not None:
            if self.primary_view not in self.views:
                raise ValueError(
                    f"primary_view {self.primary_view!r} not present in views"
                )
            if self.views.get(self.primary_view) is None:
                raise ValueError(
                    f"primary_view {self.primary_view!r} must not be None"
                )
        else:
            non_none = [key for key, value in self.views.items() if value is not None]
            if len(non_none) != 1:
                raise ValueError(
                    "primary_view=None is only valid when exactly one view is non-None"
                )

    @classmethod
    def for_spec(
        cls,
        factor_a: str,
        factor_b: str,
        similarity_spec_hash: str,
        views: Optional[Mapping[str, Optional[float]]] = None,
        *,
        snapshot_ref: Optional[str] = None,
        window_ref: Optional[str] = None,
        universe_ref: Optional[str] = None,
        producer: Optional[str] = None,
        created_at: Optional[str] = None,
        primary_view: Optional[str] = DEFAULT_SIMILARITY_VIEW,
    ) -> "SimilarityArtifact":
        """Factory for an artifact with an explicitly supplied spec hash.

        The spec hash identifies *which similarity definition* was measured
        (factor pair + view semantics + snapshot/window/universe).  A caller
        attaching a stored hash MUST supply a hash that equals the one
        recomputed from the supplied views/provenance; otherwise the artifact
        fails closed (``ValueError``), because a self-reported arbitrary hash
        would let content and hash diverge.
        """
        return cls(
            factor_a=factor_a,
            factor_b=factor_b,
            views=(
                views if views is not None else {DEFAULT_SIMILARITY_VIEW: 0.0}
            ),
            snapshot_ref=snapshot_ref,
            window_ref=window_ref,
            universe_ref=universe_ref,
            similarity_spec_hash=similarity_spec_hash,
            producer=producer,
            created_at=created_at,
            primary_view=primary_view,
        )

    @classmethod
    def from_similarity_result(
        cls,
        result: object,
        *,
        snapshot_ref: Optional[str] = None,
        producer: Optional[str] = None,
    ) -> "SimilarityArtifact":
        """Adapt a legacy :class:`SimilarityResult` into a SimilarityArtifact.

        The legacy result carries a single method-scoped correlation score;
        the new artifact's views carry it under the ``rank_corr`` key (the
        canonical primary view for MMR).  ``universe_ref`` and the period
        bounds are lifted from the result when present.
        """
        factor_a = getattr(result, "factor_id_a", None)
        factor_b = getattr(result, "factor_id_b", None)
        score = getattr(result, "similarity_score", None)
        if factor_a is None or factor_b is None or score is None:
            raise TypeError(
                "result must expose factor_id_a/factor_id_b/similarity_score "
                "(e.g. factor_assets.similarity.SimilarityResult)"
            )
        return cls(
            factor_a=factor_a,
            factor_b=factor_b,
            views={"rank_corr": float(score)},
            snapshot_ref=(
                snapshot_ref
                or getattr(result, "snapshot_ref", None)
                or getattr(result, "data_snapshot_ref", None)
            ),
            window_ref=_legacy_window_ref(result),
            universe_ref=getattr(result, "universe_ref", None),
            producer=producer or "legacy:SimilarityResult",
            created_at=getattr(result, "timestamp", None),
        )

    @property
    def primary_value(self) -> Optional[float]:
        """The value MMR / downstream consumers should treat as similarity."""
        if self.primary_view is not None:
            return self.views.get(self.primary_view)
        for value in self.views.values():
            if value is not None:
                return value
        return None

    @property
    def similarity_spec_key(self) -> str:
        """Short, readable identifier derived from the spec hash."""
        return self.similarity_spec_hash[:16]

    def view(self, key: str) -> Optional[float]:
        """Return one similarity view by key (None when absent/not measured)."""
        return self.views.get(key)

    def to_dict(self) -> dict:
        """Serializable dict form (producer + created_at dropped for purity)."""
        return {
            "factor_a": self.factor_a,
            "factor_b": self.factor_b,
            "views": dict(self.views),
            "snapshot_ref": self.snapshot_ref,
            "window_ref": self.window_ref,
            "universe_ref": self.universe_ref,
            "similarity_spec_hash": self.similarity_spec_hash,
            "primary_view": self.primary_view,
        }

    def with_provenance(
        self,
        *,
        snapshot_ref: Optional[str] = None,
        window_ref: Optional[str] = None,
        universe_ref: Optional[str] = None,
    ) -> "SimilarityArtifact":
        """Return a copy carrying the given provenance refs.

        The spec hash is RECOMPUTED from the merged provenance — changing the
        snapshot/window/universe is a change to the measurement's identity, so
        a stale caller-supplied hash would be a lie.
        """
        merged_snapshot = snapshot_ref if snapshot_ref is not None else self.snapshot_ref
        merged_window = window_ref if window_ref is not None else self.window_ref
        merged_universe = universe_ref if universe_ref is not None else self.universe_ref
        return SimilarityArtifact(
            factor_a=self.factor_a,
            factor_b=self.factor_b,
            views=dict(self.views),
            snapshot_ref=merged_snapshot,
            window_ref=merged_window,
            universe_ref=merged_universe,
            producer=self.producer,
            created_at=self.created_at,
            primary_view=self.primary_view,
        )


def _legacy_window_ref(result: object) -> Optional[str]:
    """Build a ``window_ref`` from a legacy result's period bounds."""
    start = getattr(result, "period_start", None)
    end = getattr(result, "period_end", None)
    if start is None and end is None:
        return None
    if start is not None and end is not None:
        return f"{start}/{end}"
    return start if start is not None else end
