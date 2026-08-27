"""FeatureSet — typed members + versioned artifact + retrain decision.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.8 (spec §17,
§18). PURE stdlib frozen dataclasses + enum. ``[RECONCILE]`` against
``factor_assets/contracts/factor_set.py`` before freeze.

Ordered feature identity MUST participate in the hash.

IMPORTANT content-hash rule (§5.5): ``FeatureSetArtifact.content_hash`` IS
verifiable here because it is a **semantic hash over local fields** (unlike
``ArtifactRef``'s object-bytes hash, which this DTO cannot recompute). When the
caller supplies a ``content_hash``, we VERIFY it against the recomputed value and
fail closed on mismatch.

P8 ADDITIVE EXTENSION (2026-08-28) — purely additive; nothing here removes or
renames an existing exported symbol:

- :class:`FeatureMemberRef` gains a ``metadata`` provenance map and
  ``FactorValueRef``-shaped transport aliases/projection
  (``factor_value_id`` / ``factor_ids`` / ``source_ref`` / ``metadata``).
  ``metadata`` participates in the content hash (fail-closed provenance).
- :class:`FeatureSetDiffCategory` gains a granular retrain *reason code*
  (:attr:`retrain_reason_code`); the boolean ``retrain_required`` is derived
  from it so the two can never disagree.
- :class:`FeatureSetDiff` is extended from ``(category, reason)`` to a full diff
  ledger — ``added_members`` / ``removed_members`` / ``changed_members`` (each a
  :class:`FeatureMemberChange` distinguishing a **VERSION** change from a
  **SOURCE_REF** change), ``metadata_changed`` and ``retrain_reason_codes``.
  ``category`` / ``reason`` are preserved as the legacy scalar projection.
- :class:`ModelRetrainRequiredEvent` gains ``before`` / ``after``
  ``FeatureSetVersion`` references, a :class:`FeatureSetDiffSummary` digest and
  the granular ``retrain_reason_codes`` list.
- ``classify_feature_set_diff`` keeps its exact legacy decision (so existing
  tests keep passing) while building the richer member-level ledger.
- ``retrain_required_for_diff`` is the injectable-policy entry point. It accepts
  a ``FeatureSetDiff`` + ``RetrainPolicy``, or two ``FeatureSetVersion``
  snapshots (diffs first), or — legacy — a bare ``FeatureSetDiffCategory``
  (returns the boolean). Default policy: semantic
  transform/orientation/schema/label/data-revision changes and *any* removed
  member force retrain; added members with unchanged semantics and pure
  financial-caliber / metadata / evidence changes do NOT retrain; a
  changed-member count/ratio over the injectable threshold forces retrain.
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ._contenthash import content_hash
from .rbac import SecurityClassification
from .timing import TimingContract

__all__ = [
    "FeatureSetDiffCategory",
    "FeatureSetArtifact",
    "FeatureSetVersion",
    "FeatureMemberRef",
    "FeatureMemberChange",
    "ModelRetrainRequiredEvent",
    "FeatureSetDiff",
    "FeatureSetDiffSummary",
    "RetrainPolicy",
    "compute_feature_set_content_hash",
    "classify_feature_set_diff",
    "retrain_required_for_diff",
]


# --------------------------------------------------------------------------- #
# Retrain policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrainPolicy:
    """Injectable threshold policy for :func:`retrain_required_for_diff`.

    Attributes:
        retrain_on_added: Default ``False`` — adding members whose semantics
            are unchanged does NOT force retrain.
        retrain_on_removed: Default ``True`` — ANY removed member forces retrain.
        max_changed_ratio: Default ``0.35`` — changed/removed membership ratio
            ABOVE this cap forces retrain (guards mass re-calibering).
        max_changed_count: Default ``2`` — count backstop for the ratio rule
            (protects tiny feature sets where one row is already a big ratio).
    """

    retrain_on_added: bool = False
    retrain_on_removed: bool = True
    max_changed_ratio: float = 0.35
    max_changed_count: int = 2

    def __post_init__(self) -> None:
        if self.max_changed_ratio <= 0.0 or self.max_changed_ratio > 1.0:
            raise ValueError(
                "RetrainPolicy.max_changed_ratio must be > 0.0 and <= 1.0, got "
                f"{self.max_changed_ratio!r}"
            )
        if self.max_changed_count < 0:
            raise ValueError(
                "RetrainPolicy.max_changed_count must be >= 0, got "
                f"{self.max_changed_count!r}"
            )


# --------------------------------------------------------------------------- #
# Reason-code ledger
# --------------------------------------------------------------------------- #

# Retrain-causing granular codes, at MODULE scope: Enum metaclass would turn a
# leading-underscore class attribute like ``_RETRAIN_CODES`` into an enum MEMBER
# (so ``self._RETRAIN_CODES`` would resolve to the enum, not the set). Kept at
# module level so method bodies resolve it as a plain global.
_RETRAIN_TRIGGER_CODES: frozenset[str] = frozenset(
    {
        "CHANGED_MEMBERSHIP",
        "CHANGED_VERSION",
        "CHANGED_TREATMENT",
        "CHANGED_ORIENTATION",
        "CHANGED_SCHEMA",
        "CHANGED_LABEL",
        "CHANGED_DATA_REVISION",
    }
)


class FeatureSetDiffCategory(enum.Enum):
    """FeatureSetDiff categories (spec §18).

    Each category carries a granular retrain *reason code*
    (:attr:`retrain_reason_code`); the boolean :attr:`retrain_required` is
    derived from that code so the decision and its attribution can never
    disagree.
    """

    METADATA_ONLY = "METADATA_ONLY"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    FEATURE_MEMBERSHIP_CHANGE = "FEATURE_MEMBERSHIP_CHANGE"
    FEATURE_TRANSFORM_CHANGE = "FEATURE_TRANSFORM_CHANGE"
    FEATURE_ORIENTATION_CHANGE = "FEATURE_ORIENTATION_CHANGE"
    FEATURE_SCHEMA_CHANGE = "FEATURE_SCHEMA_CHANGE"
    LABEL_CHANGE = "LABEL_CHANGE"
    DATA_REVISION = "DATA_REVISION"

    @property
    def retrain_reason_code(self) -> str:
        """Granular reason code for this category ("" when no retrain)."""
        code = {
            FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE: "CHANGED_MEMBERSHIP",
            FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE: "CHANGED_TREATMENT",
            FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE: "CHANGED_ORIENTATION",
            FeatureSetDiffCategory.FEATURE_SCHEMA_CHANGE: "CHANGED_SCHEMA",
            FeatureSetDiffCategory.LABEL_CHANGE: "CHANGED_LABEL",
            FeatureSetDiffCategory.DATA_REVISION: "CHANGED_DATA_REVISION",
        }.get(self, "")
        return code if isinstance(code, str) else ""

    @property
    def retrain_required(self) -> bool:
        """Default retrain decision (spec §18)."""
        code = self.retrain_reason_code
        return bool(code and code in _RETRAIN_TRIGGER_CODES)


# Semantic-field -> granular reason code / change-kind map for per-member diffs.
# "VERSION" = the feature itself changed (definition/payload/treatment/…);
# "SOURCE_REF" = the upstream factor-value / raw-value source changed (financial
#   caliber / provenance — must NOT retrain on its own);
# "METADATA" = free-form provenance changed (identity-relevant hash, not retrain).
_SEMANTIC_FIELD_KINDS: dict[str, tuple[str, str]] = {
    "factor_definition_ref": ("VERSION", "CHANGED_VERSION"),
    "treatment_selection_ref": ("VERSION", "CHANGED_TREATMENT"),
    "treated_feature_ref": ("VERSION", "CHANGED_VERSION"),
    "orientation": ("VERSION", "CHANGED_ORIENTATION"),
    "dtype": ("VERSION", "CHANGED_SCHEMA"),
    "channel": ("VERSION", "CHANGED_SCHEMA"),
    "timing_ref": ("VERSION", "CHANGED_VERSION"),
    "availability_semantics": ("VERSION", "CHANGED_VERSION"),
    "security_classification": ("VERSION", "CHANGED_VERSION"),
    "metadata": ("METADATA", "CHANGED_METADATA"),
}


# --------------------------------------------------------------------------- #
# Member reference (mirrors the FactorValueRef transport shape)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureMemberRef:
    """One ordered feature within a FeatureSetVersion (§5.5).

    The DTO mirrors the ``FactorValueRef`` transport shape
    (``factor_value_id`` / ``factor_ids`` / ``source_ref`` / ``metadata`` in
    ``quant_evaluator/contracts/evaluation_refs.py``): the PURE-DTO layer
    carries provenance references — never the raw value panels. The existing
    strongest ref fields remain the canonical carriers:

    - ``factor_value_id``  -> ``factor_definition_ref`` (stable identity).
    - ``factor_ids``       -> projected ``(factor_definition_ref, raw_value_ref,
      …)`` provenance tuple (:attr:`factor_ids`).
    - ``source_ref``       -> ``source_artifact_id`` / ``raw_value_ref``
      (:attr:`source_ref`).
    - ``metadata``         -> new free-form provenance map (deep-copied), NEW.

    ``metadata`` participates in the content hash. This is deliberate (fail
    closed): a change that is semantically a new source / new treatment must
    surface as a NEW FeatureSet identity even if the explicit ref fields were
    kept unchanged. It is, however, NOT a retrain trigger by itself — that
    decision belongs to :func:`retrain_required_for_diff` (pure financial-caliber
    / provenance changes must not force a retrain).
    """

    position: int
    feature_name: str
    factor_definition_ref: str
    raw_value_ref: str | None = None
    treatment_selection_ref: str | None = None
    treated_feature_ref: str | None = None
    orientation: str | None = None
    dtype: str | None = None
    channel: str | None = None
    timing_ref: str | None = None
    source_artifact_id: str | None = None
    availability_semantics: str | None = None
    security_classification: SecurityClassification | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be >= 0")
        if not self.feature_name:
            raise ValueError("feature_name is required")
        if not self.factor_definition_ref:
            raise ValueError("factor_definition_ref is required")
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata or {})))

    # -- FactorValueRef-shaped provenance projection ------------------------ #
    @property
    def factor_value_id(self) -> str:
        """Transport alias for ``factor_definition_ref`` (stable identity)."""
        return self.factor_definition_ref

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Transport alias: identity + raw-value source refs (provenance)."""
        ids: list[str] = [self.factor_definition_ref]
        if self.raw_value_ref:
            ids.append(self.raw_value_ref)
        return tuple(ids)

    @property
    def source_ref(self) -> str | None:
        """Transport alias: upstream artifact / raw-value source."""
        if self.source_artifact_id and self.raw_value_ref:
            return f"{self.raw_value_ref}@{self.source_artifact_id}"
        return self.raw_value_ref or self.source_artifact_id

    def to_dict(self) -> dict[str, Any]:
        """Serializable transport form (FactorValueRef-compatible shape)."""
        payload: dict[str, Any] = {
            "factor_value_id": self.factor_value_id,
            "factor_ids": list(self.factor_ids),
            "source_ref": self.source_ref,
            "metadata": dict(self.metadata),
        }
        optional = {
            "position": self.position,
            "feature_name": self.feature_name,
            "treatment_selection_ref": self.treatment_selection_ref,
            "treated_feature_ref": self.treated_feature_ref,
            "orientation": self.orientation,
            "dtype": self.dtype,
            "channel": self.channel,
            "timing_ref": self.timing_ref,
            "availability_semantics": self.availability_semantics,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        return payload


# --------------------------------------------------------------------------- #
# Version + artifact
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSetVersion:
    """Versioned feature set with typed, ordered members (§5.5)."""

    feature_set_id: str
    version: str
    ordered_members: tuple[FeatureMemberRef, ...] = ()
    consumer_profile: str | None = None
    source_library_versions: tuple[str, ...] = ()
    label_definition_ref: str | None = None
    data_revision_ref: str | None = None
    schema_hash: str = ""
    semantic_hash: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if not self.version:
            raise ValueError("version is required")
        if not self.schema_hash:
            object.__setattr__(
                self,
                "schema_hash",
                content_hash(
                    self.feature_set_id,
                    self.version,
                    self.ordered_members,
                ),
            )
        if not self.semantic_hash:
            object.__setattr__(
                self,
                "semantic_hash",
                content_hash(
                    self.feature_set_id,
                    self.version,
                    self.ordered_members,
                    self.source_library_versions,
                    self.consumer_profile,
                ),
            )


@dataclass(frozen=True)
class FeatureSetArtifact:
    """Ordered feature-set artifact (spec §17).

    ``content_hash`` is a **semantic hash over local fields** — it IS recomputable
    by this DTO. When the caller supplies ``content_hash``, we VERIFY it against
    the recomputed value and fail closed on mismatch. (Contrast with
    ``ArtifactRef.content_hash``, an object-bytes hash this DTO cannot recompute.)
    """

    feature_set_id: str
    feature_set_version: str
    source_library_versions: tuple[str, ...] = ()
    ordered_feature_manifest: tuple[str, ...] = ()
    created_at: datetime | None = None
    content_hash: str = ""

    def recomputed_hash(self) -> str:
        """The authoritative hash over all semantic fields incl. ordering."""
        return content_hash(
            self.feature_set_id,
            self.feature_set_version,
            self.source_library_versions,
            self.ordered_feature_manifest,
        )

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if not self.feature_set_version:
            raise ValueError("feature_set_version is required")
        computed = self.recomputed_hash()
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match recomputed value — fail closed "
                f"(got {self.content_hash!r}, recomputed {computed!r})"
            )


# --------------------------------------------------------------------------- #
# Diff ledger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureMemberChange:
    """Per-member change inside a FeatureSetDiff.

    ``change_kind`` distinguishes a **VERSION** change (the feature itself —
    definition / treatment / orientation / schema / payload — changed) from a
    **SOURCE_REF** change (the upstream factor-value / raw-value source the
    member points at changed — financial caliber / provenance) and a
    **METADATA** change (free-form provenance). ``reason_code`` is the granular
    attribution for this member.
    """

    member: FeatureMemberRef
    change_kind: str = ""  # "VERSION" | "SOURCE_REF" | "METADATA"
    reason_code: str = ""  # e.g. "CHANGED_VERSION" | "CHANGED_SOURCE_REF"
    attribute: str = ""
    before: Any = None
    after: Any = None


def _member_changes(
    old: tuple[FeatureMemberRef, ...],
    new: tuple[FeatureMemberRef, ...],
) -> tuple[FeatureMemberChange, ...]:
    """Per-member semantic diff for same-position same-identity member pairs."""
    changes: list[FeatureMemberChange] = []
    for a, b in zip(old, new):
        if a.feature_name != b.feature_name or a.factor_definition_ref != b.factor_definition_ref:
            # identity differs — that is a membership change, not a member
            # semantic change; the caller decides the category.
            continue
        for attr, (kind, code) in _SEMANTIC_FIELD_KINDS.items():
            if getattr(a, attr) != getattr(b, attr):
                changes.append(
                    FeatureMemberChange(
                        member=b,
                        change_kind=kind,
                        reason_code=code,
                        attribute=attr,
                        before=getattr(a, attr),
                        after=getattr(b, attr),
                    )
                )
        src_before = a.raw_value_ref or a.source_artifact_id
        src_after = b.raw_value_ref or b.source_artifact_id
        if src_before != src_after:
            changes.append(
                FeatureMemberChange(
                    member=b,
                    change_kind="SOURCE_REF",
                    reason_code="CHANGED_SOURCE_REF",
                    attribute="source_ref",
                    before=a.source_ref,
                    after=b.source_ref,
                )
            )
    return tuple(changes)


def _membership_ledger(
    old: tuple[FeatureMemberRef, ...],
    new: tuple[FeatureMemberRef, ...],
) -> tuple[tuple[FeatureMemberRef, ...], tuple[FeatureMemberRef, ...]]:
    """Position-independent added/removed members (by feature identity)."""
    old_by_id = {(m.feature_name, m.factor_definition_ref): m for m in old}
    new_by_id = {(m.feature_name, m.factor_definition_ref): m for m in new}
    added = tuple(m for key, m in new_by_id.items() if key not in old_by_id)
    removed = tuple(m for key, m in old_by_id.items() if key not in new_by_id)
    return added, removed


@dataclass(frozen=True)
class FeatureSetDiffSummary:
    """Compact classification digest (JSON-friendly)."""

    category: str
    reason: str
    added: int
    removed: int
    changed: int
    retrain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "reason": self.reason,
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "retrain": self.retrain,
        }


@dataclass(frozen=True)
class FeatureSetDiff:
    """Result of classifying a change between two FeatureSetVersions (spec §18).

    ``category`` and ``reason`` remain the legacy scalar projection. The new
    member-level fields give the full ledger:

    - ``added_members`` / ``removed_members`` — FeatureMemberRef members only in
      the new / old version (by feature identity).
    - ``changed_members`` — :class:`FeatureMemberChange` entries distinguishing
      VERSION vs SOURCE_REF vs METADATA changes.
    - ``metadata_changed`` — member provenance / 口径 metadata changed with
      unchanged membership (never a retrain trigger by itself).
    - ``retrain_reason_codes`` — the granular attribution set (informational;
      the retrain decision comes from :attr:`retrain_required`).
    """

    category: FeatureSetDiffCategory
    reason: str = ""
    added_members: tuple[FeatureMemberRef, ...] = ()
    removed_members: tuple[FeatureMemberRef, ...] = ()
    changed_members: tuple[FeatureMemberChange, ...] = ()
    metadata_changed: bool = False
    retrain_reason_codes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_members", tuple(self.added_members))
        object.__setattr__(self, "removed_members", tuple(self.removed_members))
        object.__setattr__(self, "changed_members", tuple(self.changed_members))
        object.__setattr__(self, "retrain_reason_codes", frozenset(self.retrain_reason_codes))

    # -- diff ledger helpers ------------------------------------------------- #
    @property
    def added_feature_names(self) -> tuple[str, ...]:
        return tuple(m.feature_name for m in self.added_members)

    @property
    def removed_feature_names(self) -> tuple[str, ...]:
        return tuple(m.feature_name for m in self.removed_members)

    @property
    def changed_feature_names(self) -> tuple[str, ...]:
        return tuple(m.member.feature_name for m in self.changed_members)

    @property
    def total_members(self) -> int:
        """New-sample membership size (denominator for the ratio rule)."""
        return max(
            len(self.added_members) + len(self.changed_members),
            len(self.removed_members) + len(self.changed_members),
            1,
        )

    @classmethod
    def _assemble(
        cls,
        category: FeatureSetDiffCategory,
        reason: str,
        added_members: tuple[FeatureMemberRef, ...] = (),
        removed_members: tuple[FeatureMemberRef, ...] = (),
        changed_members: tuple[FeatureMemberChange, ...] = (),
        metadata_changed: bool = False,
    ) -> "FeatureSetDiff":
        changed_reasons = frozenset(c.reason_code for c in changed_members)
        codes: set[str] = set()
        if category.retrain_reason_code:
            codes.add(category.retrain_reason_code)
        if removed_members:
            codes.add("CHANGED_MEMBERSHIP")
        if added_members:
            codes.add("CHANGED_MEMBERSHIP")
        codes.update(changed_reasons)
        if metadata_changed:
            codes.add("CHANGED_METADATA")
        return cls(
            category=category,
            reason=reason,
            added_members=added_members,
            removed_members=removed_members,
            changed_members=changed_members,
            metadata_changed=metadata_changed,
            retrain_reason_codes=frozenset({c for c in codes if c}),
        )

    @property
    def retrain_required(self) -> bool:
        return self.category.retrain_required

    @property
    def summary(self) -> FeatureSetDiffSummary:
        return FeatureSetDiffSummary(
            category=self.category.value,
            reason=self.reason,
            added=len(self.added_members),
            removed=len(self.removed_members),
            changed=len(self.changed_members),
            retrain=self.retrain_required,
        )

    def to_event(
        self,
        feature_set_id: str,
        before: FeatureSetVersion | None = None,
        after: FeatureSetVersion | None = None,
    ) -> ModelRetrainRequiredEvent | None:
        """Emit a ModelRetrainRequiredEvent iff this diff requires retrain."""
        if not self.retrain_required:
            return None
        return ModelRetrainRequiredEvent(
            feature_set_id=feature_set_id,
            diff_category=self.category,
            reason=self.reason,
            before=before,
            after=after,
        )


# --------------------------------------------------------------------------- #
# Retrain event
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelRetrainRequiredEvent:
    """Event emitted when a FeatureSetDiff requires model retrain (spec §18).

    Enriched (P8): binds the ``before`` / ``after`` ``FeatureSetVersion``
    snapshots this diff was computed from, a :class:`FeatureSetDiffSummary`
    classification digest and the granular ``retrain_reason_codes`` list.
    """

    feature_set_id: str
    diff_category: FeatureSetDiffCategory
    reason: str = ""
    before: FeatureSetVersion | None = None
    after: FeatureSetVersion | None = None
    diff_summary: FeatureSetDiffSummary | None = None
    retrain_reason_codes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.feature_set_id:
            raise ValueError("feature_set_id is required")
        if self.before is not None and not isinstance(self.before, FeatureSetVersion):
            raise TypeError(
                "ModelRetrainRequiredEvent.before must be a FeatureSetVersion or "
                f"None, got {type(self.before).__name__}"
            )
        if self.after is not None and not isinstance(self.after, FeatureSetVersion):
            raise TypeError(
                "ModelRetrainRequiredEvent.after must be a FeatureSetVersion or "
                f"None, got {type(self.after).__name__}"
            )
        if (self.before is None) != (self.after is None):
            raise ValueError(
                "ModelRetrainRequiredEvent requires BOTH before and after "
                "references (or neither)"
            )
        if self.before is not None and self.after is not None:
            if self.before.feature_set_id != self.after.feature_set_id:
                raise ValueError(
                    "before/after feature set ids must match: "
                    f"{self.before.feature_set_id!r} vs {self.after.feature_set_id!r}"
                )
            if self.feature_set_id != self.before.feature_set_id:
                raise ValueError(
                    "event feature_set_id must match the before/after refs: "
                    f"{self.feature_set_id!r} vs {self.before.feature_set_id!r}"
                )
        object.__setattr__(self, "retrain_reason_codes", frozenset(self.retrain_reason_codes))

    @property
    def before_version(self) -> str | None:
        return self.before.version if self.before else None

    @property
    def after_version(self) -> str | None:
        return self.after.version if self.after else None


# --------------------------------------------------------------------------- #
# Diff classification
# --------------------------------------------------------------------------- #


def compute_feature_set_content_hash(ordered_features: tuple[FeatureMemberRef, ...]) -> str:
    """Deterministic hash of the ordered feature identity (spec §17).

    The hash covers the *semantic identity* of each member — feature_name,
    factor_definition_ref, treatment_selection_ref, orientation, source_artifact_id,
    availability_semantics, dtype, channel, and column_position — in order. A
    treatment or orientation change therefore yields a NEW FeatureSet identity,
    while a metadata-only change (e.g. consumer_profile) does not.

    ``position`` is intentionally excluded from the per-member identity: the
    *ordering* is captured by the tuple order itself, so renumbering positions
    without reordering must not change identity.
    """
    return content_hash(ordered_features)


def classify_feature_set_diff(
    old: FeatureSetVersion,
    new: FeatureSetVersion,
) -> FeatureSetDiff:
    """Classify a change between two FeatureSetVersions (spec §18).

    Pure and deterministic. Returns a ``FeatureSetDiff`` whose ``category`` drives
    the retrain decision:

    - METADATA_ONLY / EVIDENCE_ONLY -> no retrain;
    - FEATURE_MEMBERSHIP_CHANGE / FEATURE_TRANSFORM_CHANGE /
      FEATURE_ORIENTATION_CHANGE / FEATURE_SCHEMA_CHANGE / LABEL_CHANGE /
      DATA_REVISION -> ``ModelRetrainRequiredEvent`` emitted.
    """
    if old.feature_set_id != new.feature_set_id:
        raise ValueError(
            "cannot diff feature sets with different ids: "
            f"{old.feature_set_id!r} vs {new.feature_set_id!r}"
        )

    old_members = old.ordered_members
    new_members = new.ordered_members

    # ---- full member-level ledger (position-independent identity) ---------- #
    added, removed = _membership_ledger(old_members, new_members)
    changes = _member_changes(old_members, new_members)

    # ---- legacy category decision (unchanged) ------------------------------ #
    if len(old_members) != len(new_members):
        return FeatureSetDiff._assemble(
            FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE,
            reason="member count changed",
            added_members=added,
            removed_members=removed,
            changed_members=changes,
        )
    for a, b in zip(old_members, new_members):
        if a.feature_name != b.feature_name or a.factor_definition_ref != b.factor_definition_ref:
            return FeatureSetDiff._assemble(
                FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE,
                reason=f"member identity changed at position {a.position}",
                added_members=added,
                removed_members=removed,
                changed_members=changes,
            )

    # Same membership. Now compare per-member semantic fields.
    for a, b in zip(old_members, new_members):
        if a.treatment_selection_ref != b.treatment_selection_ref:
            return FeatureSetDiff._assemble(
                FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE,
                reason=f"treatment changed at position {a.position}",
                changed_members=changes,
            )
        if a.orientation != b.orientation:
            return FeatureSetDiff._assemble(
                FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE,
                reason=f"orientation changed at position {a.position}",
                changed_members=changes,
            )
        if a.dtype != b.dtype or a.channel != b.channel:
            return FeatureSetDiff._assemble(
                FeatureSetDiffCategory.FEATURE_SCHEMA_CHANGE,
                reason=f"schema changed at position {a.position}",
                changed_members=changes,
            )

    # Label / data-revision changes are carried on the FeatureSetVersion itself.
    if old.label_definition_ref != new.label_definition_ref:
        return FeatureSetDiff._assemble(
            FeatureSetDiffCategory.LABEL_CHANGE,
            reason="label definition changed",
        )
    if old.data_revision_ref != new.data_revision_ref:
        return FeatureSetDiff._assemble(
            FeatureSetDiffCategory.DATA_REVISION,
            reason="data revision changed",
        )

    # Remaining differences are metadata / evidence only. Evidence is carried by
    # the source library versions; everything else (consumer_profile, version
    # label) is metadata.
    if old.source_library_versions != new.source_library_versions:
        return FeatureSetDiff._assemble(
            FeatureSetDiffCategory.EVIDENCE_ONLY,
            reason="source-library / evidence metadata changed",
            changed_members=changes,
        )
    # Source-ref-only / metadata-only member changes with identical feature
    # identity are financial-caliber / provenance changes — classified as
    # METADATA_ONLY (no retrain) but recorded in the ledger.
    if changes:
        return FeatureSetDiff._assemble(
            FeatureSetDiffCategory.METADATA_ONLY,
            reason="membership unchanged; non-semantic member metadata changed",
            changed_members=changes,
            metadata_changed=True,
        )
    return FeatureSetDiff._assemble(
        FeatureSetDiffCategory.METADATA_ONLY,
        reason="no semantic change",
    )


# --------------------------------------------------------------------------- #
# Retrain decision with injectable policy
# --------------------------------------------------------------------------- #


def _over_threshold(diff: FeatureSetDiff, policy: RetrainPolicy) -> bool:
    """Changed/removed count or ratio above the injectable policy threshold."""
    total = diff.total_members
    touched = len(diff.changed_members) + len(diff.removed_members)
    return touched > policy.max_changed_count or (
        total > 0 and (touched / total) > policy.max_changed_ratio
    )


def retrain_required_for_diff(
    diff_or_old: FeatureSetDiff | FeatureSetVersion | FeatureSetDiffCategory,
    policy_or_new: RetrainPolicy | FeatureSetVersion | None = None,
    *,
    policy: RetrainPolicy | None = None,
    feature_set_id: str | None = None,
) -> ModelRetrainRequiredEvent | None | bool:
    """Injectably decide whether a FeatureSetDiff requires model retrain.

    Three call forms:

    - ``retrain_required_for_diff(diff, policy=None)`` — a pre-classified
      ``FeatureSetDiff`` plus an optional injectable ``RetrainPolicy``. When a
      retrain event is produced you must pass ``feature_set_id=`` (the diff
      cannot recover it), or use the two-snapshot form below.
    - ``retrain_required_for_diff(old, new, policy=None)`` — two
      ``FeatureSetVersion`` snapshots; the diff is classified first and the
      event binds ``before=old`` / ``after=new``.
    - ``retrain_required_for_diff(category)`` — legacy: returns the boolean.

    Default policy: semantic transform / orientation / schema / label /
    data-revision changes and *any* removed member force retrain; added members
    with unchanged semantics and pure financial-caliber / metadata / evidence
    changes do NOT; a changed-member count/ratio above the injectable threshold
    forces retrain. Returns ``ModelRetrainRequiredEvent | None`` (or ``bool``
    for the legacy category form).
    """
    # Legacy bool form.
    if isinstance(diff_or_old, FeatureSetDiffCategory):
        if policy_or_new is not None or policy is not None:
            raise TypeError("legacy category form takes no policy")
        return diff_or_old.retrain_required

    # Two-snapshot form.
    if isinstance(diff_or_old, FeatureSetVersion):
        if not isinstance(policy_or_new, FeatureSetVersion):
            raise TypeError(
                "retrain_required_for_diff(old, new) requires a FeatureSetVersion "
                f"as the second argument, got {type(policy_or_new).__name__}"
            )
        if policy is not None and not isinstance(policy, RetrainPolicy):
            raise TypeError(
                "policy must be a RetrainPolicy; got "
                f"{type(policy).__name__}"
            )
        old: FeatureSetVersion = diff_or_old
        new: FeatureSetVersion = policy_or_new
        diff: FeatureSetDiff = classify_feature_set_diff(old, new)
        eff_policy: RetrainPolicy | None = policy
        fsid: str = old.feature_set_id
        before_arg: FeatureSetVersion | None = old
        after_arg: FeatureSetVersion | None = new

    # Diff form.
    elif isinstance(diff_or_old, FeatureSetDiff):
        diff = diff_or_old
        if policy is not None and policy_or_new is not None:
            raise ValueError("pass policy either positionally or by keyword, not both")
        eff_policy = policy if policy is not None else policy_or_new
        if eff_policy is not None and not isinstance(eff_policy, RetrainPolicy):
            raise TypeError(
                "policy must be a RetrainPolicy; got "
                f"{type(eff_policy).__name__}"
            )
        fsid = feature_set_id or ""
        before_arg = after_arg = None

    else:
        raise TypeError(
            "retrain_required_for_diff expects a FeatureSetDiff, "
            f"FeatureSetVersion or FeatureSetDiffCategory, got {type(diff_or_old).__name__}"
        )

    eff_policy = eff_policy if eff_policy is not None else RetrainPolicy()

    # ---- policy decision ---------------------------------------------------- #
    category = diff.category

    if category is FeatureSetDiffCategory.FEATURE_TRANSFORM_CHANGE or (
        category is FeatureSetDiffCategory.FEATURE_ORIENTATION_CHANGE
    ):
        retrain = True
    elif category is FeatureSetDiffCategory.FEATURE_MEMBERSHIP_CHANGE:
        # Membership counts (added/removed) are decided by the explicit policy
        # flags; the threshold governs changed members KEPT in the set,
        # and financial-caliber changes are never a retrain trigger.
        retrain = False
        if diff.added_members and eff_policy.retrain_on_added:
            retrain = True
        elif diff.removed_members:
            retrain = True
        elif diff.changed_members and any(
            c.reason_code in _RETRAIN_TRIGGER_CODES for c in diff.changed_members
        ):
            retrain = _over_threshold(diff, eff_policy)
        # else: pure renumbering / pure source-ref changes -> no retrain.
    elif category is FeatureSetDiffCategory.LABEL_CHANGE:
        retrain = True
    elif category is FeatureSetDiffCategory.DATA_REVISION:
        retrain = True
    elif category is FeatureSetDiffCategory.FEATURE_SCHEMA_CHANGE:
        # Schema change forces retrain unless a fully permissive injectable
        # policy lifts it.
        if diff.changed_members:
            retrain = not (
                eff_policy.max_changed_ratio >= 1.0
                and eff_policy.max_changed_count >= 10
            )
        else:
            retrain = True
    else:  # METADATA_ONLY / EVIDENCE_ONLY — pure 口径 / provenance changes.
        retrain = False

    if not retrain:
        return None

    if not fsid:
        raise ValueError(
            "retrain is required but no feature_set_id is available — pass "
            "feature_set_id= or call retrain_required_for_diff(old, new)"
        )

    return ModelRetrainRequiredEvent(
        feature_set_id=fsid,
        diff_category=diff.category,
        reason=diff.reason,
        before=before_arg,
        after=after_arg,
        diff_summary=diff.summary,
        retrain_reason_codes=diff.retrain_reason_codes,
    )
