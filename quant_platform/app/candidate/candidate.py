"""QRP-P3 — candidate normalization / reconciliation / batch fingerprint.

Pure stdlib logic layer (the *parallel enriched-normalization* variant of
``candidate/ingest.py``).

AUTHORITY (R55 P0-5, task #95): the platform does NOT mint or judge factor
identities. Every identity a candidate carries is produced by the owning
DOMAIN package and validated here FORMAT-ONLY:

* ``content_hash`` — the publisher's spec-bytes sha256 (64-char lowercase hex);
* ``semantic_hash`` — the domain's factor-definition semantic digest
  (``factor_assets/identity``), carried verbatim, never re-derived from the
  formula / parameter dict / 口径;
* ``factor_definition_ref`` — the carried :class:`FactorDefinitionRef`
  (``contracts/identities.py``), whose ``hash`` is the domain digest itself.

This module contains no hash function over factor semantics and no admission /
threshold logic: whether a candidate is admitted is delegated to the owning
domain package through ``contracts/admission.AdmissionAuthority`` (P0-6).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..contracts._contenthash import canonical_str
from ..contracts.candidate import FactorCandidateManifest
from ..contracts.identities import FactorDefinitionRef, sha256_hex

__all__ = [
    "NormalizationError",
    "NormalizationErrorCode",
    "NORMALIZATION_ERROR_CODES",
    "NormalizedCandidate",
    "ConflictRecord",
    "RECONCILE_REASON_NEW",
    "RECONCILE_REASON_DUPLICATE",
    "RECONCILE_REASON_CONFLICT_SEMANTIC",
    "RECONCILE_REASON_CONFLICT_CONTENT",
    "RECONCILE_REASON_INVALID_DUPLICATE",
    "RECONCILE_REASON_INVALID_CONFLICT",
    "RECONCILE_REASONS",
    "ReconcileReport",
    "normalize_candidate",
    "reconcile_candidates",
    "batch_fingerprint",
]

# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #

NormalizationErrorCode = str

# fail-closed reason codes
NORMALIZATION_ERROR_CODES = frozenset(
    {
        "MISSING_CONTENT_HASH",
        "MISSING_SEMANTIC_ID",
        "INVALID_CONTENT_HASH",
        "INVALID_SEMANTIC_HASH",
        "NON_FINITE_PARAMETER",
        "UNSUPPORTED_PARAMETER_VALUE",
        "INVALID_MARKET",
        "INVALID_SUBMITTED_AT",
    }
)


class NormalizationError(ValueError):
    """Raised when a raw candidate fails canonical normalization (fail-closed).

    ``code`` ∈ :data:`NORMALIZATION_ERROR_CODES`.
    """

    def __init__(self, code: str, reason: str) -> None:
        if code not in NORMALIZATION_ERROR_CODES:
            raise TypeError(f"unknown normalization error code: {code!r}")
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _as_iso_datetime(value: Any, code: str, field_name: str) -> str:
    """Accept str, datetime, or date insertions → canonical UTC ``Z`` iso str;
    anything else / absurd values fail closed."""
    if isinstance(value, str):
        if not value:
            raise NormalizationError(code, f"{field_name} must not be empty")
        value = value.strip()
        if not value:
            raise NormalizationError(code, f"{field_name} must not be blank")
        try:
            _ = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise NormalizationError(code, f"{field_name} not ISO-8601: {value!r}") from exc
        return value
    if isinstance(value, datetime):
        try:
            return _canonical_iso(value)
        except ValueError as exc:
            raise NormalizationError(code, str(exc)) from exc
    if isinstance(value, type):
        raise NormalizationError(code, f"{field_name} must be str/datetime, got {type(value).__name__}")
    if isinstance(value, object) and not isinstance(value, (int, float, bool)):
        raise NormalizationError(code, f"{field_name} must be str/datetime, got {type(value).__name__}")
    raise NormalizationError(code, f"{field_name} must be str/datetime, got {type(value).__name__}")


def _canonical_iso(dt: datetime) -> str:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(
            f"naive datetime not allowed in canonical content hash; got {dt!r} — "
            "provide a timezone-aware datetime"
        )
    from datetime import timezone

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return value.strip()
    raise NormalizationError(
        "UNSUPPORTED_PARAMETER_VALUE",
        f"{field_name} must be a str, got {type(value).__name__}",
    )


def _required_text(value: Any, field_name: str) -> str:
    text = _clean_text(value, field_name)
    if not text:
        raise NormalizationError(
            "UNSUPPORTED_PARAMETER_VALUE", f"{field_name} is required and must not be blank"
        )
    return text


def _hash_hex(value: Any, field_name: str) -> str:
    """FORMAT-ONLY sha256 hex validation of a carried digest (never recomputed)."""
    text = _clean_text(value, field_name)
    try:
        return sha256_hex(text, field_name)
    except (TypeError, ValueError) as exc:
        raise NormalizationError(
            "INVALID_CONTENT_HASH",
            f"{field_name} must be a 64-char lowercase hex sha256 minted by the "
            f"owning domain package ({exc})",
        ) from exc


def _finite_parameter(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NormalizationError(
            "UNSUPPORTED_PARAMETER_VALUE",
            f"parameter must be a finite int/float, got {type(value).__name__}",
        )
    param = float(value)
    if math.isnan(param) or math.isinf(param):
        raise NormalizationError("NON_FINITE_PARAMETER", "parameter values must be finite")
    return param


def _normalize_parameter_domain(raw: Any) -> tuple[tuple[str, str], ...]:
    """Normalize a parameter domain into a canonical tuple of (name, value).

    Accepted shapes (each value must be finite):
    - mapping ``{name: value, ...}``
    - sequence of ``(name, value)`` pairs
    - sequence of bare scalar values → param0/param1/…
      (bare values come from discovery records, e.g. recipe JSON)

    The normalized domain is CARRIED as opaque manifest data: it is never hashed
    into a platform-computed identity (R55 P0-5) — the domain package that owns
    the factor definition folds its own parameters into its own digest.
    """
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, (tuple, list)):
        items = list(raw)
    else:
        raise NormalizationError(
            "UNSUPPORTED_PARAMETER_VALUE",
            f"parameter_domain must be a mapping or sequence, got {type(raw).__name__}",
        )

    out: list[tuple[str, str]] = []
    for idx, entry in enumerate(items):
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and not isinstance(entry, (str, bytes)):
            try:
                key = _required_text(entry[0], f"parameter_domain[{idx}].name")
            except NormalizationError:
                raise
            value = _finite_parameter(entry[1])
            out.append((key, repr(value)))
        elif isinstance(entry, str) or isinstance(entry, (int, float)) and not isinstance(entry, bool):
            # bare scalar → positional param name
            value = _finite_parameter(entry)
            out.append((f"param{idx}", repr(value)))
        elif isinstance(entry, bool):
            raise NormalizationError(
                "UNSUPPORTED_PARAMETER_VALUE",
                f"parameter_domain[{idx}] is bool — domains admit numbers only",
            )
        else:
            raise NormalizationError(
                "UNSUPPORTED_PARAMETER_VALUE",
                f"parameter_domain[{idx}] must be (name, value) pair or bare scalar, got {type(entry).__name__}",
            )
    return tuple(out)


# --------------------------------------------------------------------------- #
# NormalizedCandidate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NormalizedCandidate(FactorCandidateManifest):
    """Canonical normalized candidate.

    Extends :class:`FactorCandidateManifest` with the three enrichment slots
    required by reconciliation: a stable ``content_hash`` (the publisher's
    ``factor_spec_sha256``), a ``semantic_hash`` CARRIED from the domain
    package, and a canonical ``parameter_domain``.

    R55 P0-5: ``semantic_hash`` is the OWNING DOMAIN PACKAGE's factor-definition
    digest, validated for hex format only — this class performs NO identity
    negotiation, no recomputation, and embeds no 口径 (the vwap->vwap basis
    lives inside the domain's own digest, not in a platform-side hash input).
    ``factor_definition_ref`` carries the same digest in typed-ref form for
    consumers that want the ref object rather than the raw string.
    """

    content_hash: str = ""
    semantic_hash: str = ""
    parameter_domain: tuple[tuple[str, str], ...] = ()
    formula_expression: str = ""
    operator_semantics_version: str = ""

    def __post_init__(self) -> None:
        summary = super().__post_init__()
        if not self.content_hash:
            raise ValueError("content_hash is required")
        if len(self.content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.content_hash):
            raise ValueError(f"content_hash must be a 64-char lowercase hex sha256, got {self.content_hash!r}")
        if not self.semantic_hash:
            raise ValueError("semantic_hash is required (minted by the owning domain package)")
        if len(self.semantic_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.semantic_hash):
            raise ValueError(f"semantic_hash must be a 64-char lowercase hex sha256, got {self.semantic_hash!r}")
        # factor_spec_sha256 must equal content_hash — a candidate whose spec
        # bytes hash differs from its manifest hash is self-inconsistent.
        if self.factor_spec_sha256 != self.content_hash:
            raise ValueError(
                "factor_spec_sha256 must equal content_hash "
                f"(got {self.factor_spec_sha256!r} vs {self.content_hash!r})"
            )
        # R55 P0-5: FORMAT-only identity validation. The platform carries the
        # domain digest; it never re-derives one. Building the ref type here is
        # the whole validation surface (hex format + required non-empty
        # factor_version at construction time).
        object.__setattr__(
            self,
            "_definition_ref",
            FactorDefinitionRef(
                self.semantic_hash,
                factor_version=self.generator_version,
                descriptor={
                    "formula_expression": self.consumed_formula_expression(),
                    "parameter_domain": tuple(self.parameter_domain or ()),
                },
            ),
        )
        for name, value in self.parameter_domain:
            if not name or not isinstance(name, str):
                raise ValueError("parameter_domain names must be non-empty strings")
            if not isinstance(value, str) or not value:
                raise ValueError("parameter_domain values must be non-empty canonical strings")

    def factor_definition_ref(self) -> FactorDefinitionRef:
        """The carried (not recomputed) domain factor-definition identity ref."""
        return self._definition_ref

    def consumed_formula_expression(self) -> str:
        """The formula expression this candidate was submitted with.

        Carried provenance ONLY — never hashed by the platform. Prefers the
        enriched slot; falls back to ``semantic_family_hint`` (the domain's
        semantic digest carrier) so past-round manifests stay self-consistent.
        """
        if self.formula_expression and self.formula_expression.strip():
            return self.formula_expression
        return self.semantic_family_hint or self.expression()

    def semantic_id(self) -> str:
        """The carried domain semantic identity digest (opaque)."""
        return self.semantic_hash


def reconcile_id(candidate: NormalizedCandidate) -> str:
    return f"candidate:{candidate.content_hash}"


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #

# reason codes (per class)
RECONCILE_REASON_NEW = "NEW"
RECONCILE_REASON_DUPLICATE = "DUPLICATE"
RECONCILE_REASON_CONFLICT_SEMANTIC = "CONFLICT_SEMANTIC"
RECONCILE_REASON_CONFLICT_CONTENT = "CONFLICT_CONTENT"
# in-batch identification failures (a batch that would accidentally ingest a
# duplicate or a conflicting candidate must fail closed)
RECONCILE_REASON_INVALID_DUPLICATE = "INVALID_DUPLICATE"
RECONCILE_REASON_INVALID_CONFLICT = "INVALID_CONFLICT"

RECONCILE_REASONS = frozenset(
    {
        RECONCILE_REASON_NEW,
        RECONCILE_REASON_DUPLICATE,
        RECONCILE_REASON_CONFLICT_SEMANTIC,
        RECONCILE_REASON_CONFLICT_CONTENT,
        RECONCILE_REASON_INVALID_DUPLICATE,
        RECONCILE_REASON_INVALID_CONFLICT,
    }
)


@dataclass(frozen=True)
class ConflictRecord:
    """One conflicting candidate + the registry entry it conflicts with."""

    candidate: Any  # NormalizedCandidate
    registered: Any  # NormalizedCandidate
    reason: str  # CONFLICT_SEMANTIC | CONFLICT_CONTENT

    def __post_init__(self) -> None:
        if self.reason not in (RECONCILE_REASON_CONFLICT_SEMANTIC, RECONCILE_REASON_CONFLICT_CONTENT):
            raise ValueError(f"ConflictRecord must carry a conflict reason, got {self.reason!r}")
        if self.candidate is self.registered:
            raise ValueError("ConflictRecord candidate and registered must not be the same object")


_T = Any


@dataclass(frozen=True)
class ReconcileReport:
    """Immutable reconciliation result."""

    new: tuple[Any, ...] = ()
    duplicates: tuple[Any, ...] = ()
    conflicts: tuple[ConflictRecord, ...] = ()
    batch_fingerprint: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "duplicates": len(self.duplicates),
            "conflicts": len(self.conflicts),
        }

    @property
    def total(self) -> int:
        return self.counts["new"] + self.counts["duplicates"] + self.counts["conflicts"]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "total": self.total,
            "new": [cat.candidate_id for cat in self.new],
            "duplicates": [cat.candidate_id for cat in self.duplicates],
            "conflict_details": [
                {
                    "candidate_id": rec.candidate.candidate_id,
                    "registered_id": rec.registered.candidate_id,
                    "reason": rec.reason,
                    "content_hash": rec.candidate.content_hash,
                    "semantic_hash": rec.candidate.semantic_hash,
                }
                for rec in self.conflicts
            ],
            "batch_fingerprint": self.batch_fingerprint,
        }


def _reconcile_identity(
    candidate: NormalizedCandidate,
    by_content: dict[str, Any],
    by_semantic: dict[str, Any],
) -> tuple[str, Any | None]:
    """Classify a single candidate against the dual-key indexes.

    Returns ``(reason, registered_entry)`` where ``registered_entry`` is the
    registry candidate it *matched* (for NEW/DUPLICATE) or the entry it is in
    conflict with (for CONFLICT_*).
    """
    content_hit = by_content.get(candidate.content_hash)
    semantic_hit = by_semantic.get(candidate.semantic_hash)
    if content_hit is None and semantic_hit is None:
        return RECONCILE_REASON_NEW, None
    if content_hit is not None and semantic_hit is not None and content_hit is semantic_hit:
        return RECONCILE_REASON_DUPLICATE, content_hit
    # ambiguous — one hash matched a different entry
    if content_hit is not None and (semantic_hit is None or semantic_hit is not content_hit):
        return RECONCILE_REASON_CONFLICT_CONTENT, content_hit
    if semantic_hit is not None and (content_hit is None or content_hit is not semantic_hit):
        return RECONCILE_REASON_CONFLICT_SEMANTIC, semantic_hit
    return RECONCILE_REASON_NEW, None


recognized_types = (NormalizedCandidate,)


def _slot(candidate: NormalizedCandidate) -> tuple[str, str]:
    """The identity slot pair used for in-batch dedup and registry keys."""
    return (candidate.content_hash, candidate.semantic_hash)


def normalize_candidate(raw: Mapping[str, Any], *, parameter_defaults: Mapping[str, Any] | None = None) -> NormalizedCandidate:
    """Normalize one heterogeneous raw candidate record (fail-closed).

    Every identity is CARRIED from the producing domain package and validated
    for hex FORMAT only:

      ``content_hash`` / ``factor_spec_sha256`` (publisher spec-bytes digest),
      ``semantic_hash`` (the domain's factor-definition semantic digest —
      mandatory, non-empty, 64-char lowercase hex; the platform never derives
      one from the expression / parameter dict, so ``MISSING_SEMANTIC_ID`` /
      ``INVALID_SEMANTIC_HASH`` are the fail-closed paths).

    Raises :class:`NormalizationError` with a reason code on any missing /
    non-finite / malformed / non-canonical input.
    """
    raw = dict(raw)
    param_defaults_mapping: Mapping[str, Any] = dict(parameter_defaults or {})

    # ---- content hash (publisher-reported spec digest, format-checked) ----
    content_from_spec = raw.get("content_hash")
    if content_from_spec is None:
        content_from_spec = raw.get("factor_spec_sha256")
    if content_from_spec is None:
        # fail-closed: a raw record without a content hash cannot be carried
        raise NormalizationError(
            "MISSING_CONTENT_HASH",
            "raw candidate must carry 'content_hash' or 'factor_spec_sha256' "
            "(sha256 over the canonical spec, minted by the publisher) — none provided",
        )
    content_hash_value = _hash_hex(content_from_spec, "content_hash")

    # ---- semantic identity (domain-minted digest, carried verbatim) ----
    raw_semantic = raw.get("semantic_id")
    if raw_semantic is None:
        raw_semantic = raw.get("semantic_hash")
    if not isinstance(raw_semantic, str) or not raw_semantic.strip():
        raise NormalizationError(
            "MISSING_SEMANTIC_ID",
            "raw candidate must carry the OWNING DOMAIN PACKAGE's semantic digest "
            "('semantic_id' / 'semantic_hash', 64-char lowercase hex) — none "
            "provided; the platform refuses to fabricate identity",
        )
    raw_semantic = raw_semantic.strip()
    if len(raw_semantic) != 64 or any(ch not in "0123456789abcdef" for ch in raw_semantic):
        raise NormalizationError(
            "INVALID_SEMANTIC_HASH",
            f"semantic_hash must be a 64-char lowercase hex sha256, got {raw_semantic!r}",
        )

    # ---- remaining manifest fields ----
    candidate_id = _required_text(
        raw.get("candidate_id") or raw.get("recipe_name") or raw.get("id"),
        "candidate_id",
    )
    submitted_at = _as_iso_datetime(
        raw.get("submitted_at") or raw.get("discovered_at") or raw.get("published_at") or raw.get("created_at") or "",
        "INVALID_SUBMITTED_AT",
        "submitted_at",
    )
    submitted_by = _required_text(raw.get("submitted_by") or "discovery-mining", "submitted_by")
    all_generator = _clean_text(raw.get("generator_type") or raw.get("generator") or "mining", "generator_type")
    generator_type = _required_text(all_generator or "mining", "generator_type")
    generator_version = _required_text(raw.get("generator_version") or "1.0.0", "generator_version")
    market = _required_text(raw.get("market") or raw.get("universe") or "", "market")
    frequency = _required_text(raw.get("frequency") or "", "frequency")
    formula_language = _required_text(raw.get("formula_language") or raw.get("language") or "", "formula_language")
    factor_spec_uri = _required_text(raw.get("factor_spec_uri") or f"recipe://{candidate_id}", "factor_spec_uri")
    parent_factor_ids = tuple(
        str(v) for v in (raw.get("parent_factor_ids") if isinstance(raw.get("parent_factor_ids"), (tuple, list)) else ())
    )
    required_fields = tuple(
        str(v) for v in (raw.get("required_fields") if isinstance(raw.get("required_fields"), (tuple, list)) else ())
    )
    semantic_family_hint = _clean_text(raw.get("semantic_family_hint") or raw_semantic, "semantic_family_hint") or None
    campaign_id = _clean_text(raw.get("campaign_id") or "", "campaign_id") or None
    attempt_id = _clean_text(raw.get("attempt_id") or "", "attempt_id") or None

    # ---- parameter domain (fail closed on non-finite / unsupported) ----
    param_raw = raw.get("parameter_domain")
    if param_raw is None:
        param_raw = raw.get("params")
    parameters = _normalize_parameter_domain(param_raw)

    if parameters:
        # fill missing defaulted params, ALWAYS validating (non-finite → raise)
        present = {name for name, _ in parameters}
        for name, default in param_defaults_mapping.items():
            if name not in present:
                parameters += ((name, repr(_finite_parameter(default))),)

    # ---- build the canonical manifest (identity CARRIED, not negotiated) ----
    base = FactorCandidateManifest(
        schema_version=_required_text(raw.get("schema_version") or "1.0.0", "schema_version"),
        candidate_id=candidate_id,
        submitted_at=submitted_at,
        submitted_by=submitted_by,
        generator_type=generator_type,
        generator_version=generator_version,
        market=market,
        frequency=frequency,
        formula_language=formula_language,
        factor_spec_uri=factor_spec_uri,
        factor_spec_sha256=content_hash_value,
        parent_factor_ids=parent_factor_ids,
        required_fields=required_fields,
        semantic_family_hint=semantic_family_hint,
        campaign_id=campaign_id,
        attempt_id=attempt_id,
    )
    final_content_hash = content_hash_value
    return NormalizedCandidate(
        **fields_dict(base),
        content_hash=final_content_hash,
        semantic_hash=raw_semantic,
        parameter_domain=parameters,
        formula_expression=_clean_text(
            raw.get("formula_expression") or raw.get("expression") or raw.get("formula") or "",
            "formula_expression",
        ),
        operator_semantics_version=generator_version,
    )


# bypass calling anon
def anon(x):
    return ""


def fields_dict(base: FactorCandidateManifest) -> dict[str, Any]:
    return {f.name: getattr(base, f.name) for f in fields(base)}


def reconcile_candidates(
    candidates: Sequence[Any],
    known_registry: Mapping[str, Any] | None = None,
) -> ReconcileReport:
    """Reconcile candidates against the known registry by dual key.

    ``known_registry`` is a mapping of *already-registered* candidates — either
    hash-keyed ``{content_hash: obj}`` / ``{semantic_hash: obj}`` (letters only
    — the keys are the 64-char hex hashes) or ``{candidate_id: obj}`` (any other
    string). Objects must be ``NormalizedCandidate``.

    Duplicate candidates within the candidates batch are rejected (they would
    double-ingest); a candidate that is simultaneously duplicate AND conflicting
    with different registry entries fails closed.
    """
    registry = {}
    if known_registry:
        for key, obj in known_registry.items():
            if isinstance(key, str):
                if len(key) == 64 and all(ch in "0123456789abcdef" for ch in key):
                    registry[key] = obj
                else:
                    registry[f"id:{key}"] = obj

    # normalize the registry objects to dict keyed by slot
    by_content: dict[str, Any] = {}
    by_semantic: dict[str, Any] = {}
    for key, obj in registry.items():
        if key.startswith("id:"):
            continue
        ch = len(key) == 64 and all(ch in "0123456789abcdef" for ch in key)
        if not ch:
            continue
        if isinstance(obj, NormalizedCandidate):
            by_content[obj.content_hash] = obj
            by_semantic[obj.semantic_hash] = obj

    # candidates from registry that were id-keyed: unify in the same way
    for obj in registry.values():
        if isinstance(obj, NormalizedCandidate):
            by_content.setdefault(obj.content_hash, obj)
            by_semantic.setdefault(obj.semantic_hash, obj)

    norm_candidates: list[Any] = list(candidates)
    seen_slots: set[tuple[str, str]] = set()
    new_entries: list[Any] = []
    duplicate_entries: list[Any] = []
    conflict_records: list[ConflictRecord] = []

    for candidate in norm_candidates:
        slot = _slot(candidate)
        if slot in seen_slots:
            # in-batch duplicate — fail closed (would double-ingest)
            conflict_records.append(
                ConflictRecord(candidate=candidate, registered=candidate, reason=RECONCILE_REASON_INVALID_DUPLICATE)
            )
            continue
        seen_slots.add(slot)
        reason, matched = _reconcile_identity(candidate, by_content, by_semantic)
        if reason == RECONCILE_REASON_NEW:
            new_entries.append(candidate)
        elif reason == RECONCILE_REASON_DUPLICATE:
            duplicate_entries.append(candidate)
        elif reason in (RECONCILE_REASON_CONFLICT_SEMANTIC, RECONCILE_REASON_CONFLICT_CONTENT):
            conflict_records.append(
                ConflictRecord(candidate=candidate, registered=matched, reason=reason)
            )

    fingerprint = batch_fingerprint(new_entries)
    return ReconcileReport(
        new=tuple(new_entries),
        duplicates=tuple(duplicate_entries),
        conflicts=tuple(conflict_records),
        batch_fingerprint=fingerprint,
    )


# --------------------------------------------------------------------------- #
# batch fingerprint
# --------------------------------------------------------------------------- #


def batch_fingerprint(candidates: Sequence[Any]) -> str:
    """Order-invariant batch Merkle fingerprint over CARRIED content hashes.

    Folds each candidate's carried ``content_hash`` with a per-candidate counter
    so a formula with a *repeated* content_hash still produces a nonzero
    contribution; sorting makes the fingerprint order-independent; the fold
    string itself is length-prefixed and canonicalized before hashing.

    This is a platform anti-replay TOKEN over carried digests — not a domain
    identity: no factor semantics enter it.
    """
    folds: list[str] = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        ch = candidate.content_hash
        if not isinstance(ch, str) or not ch:
            raise ValueError(f"candidate has no valid content_hash: {candidate!r}")
        counts[ch] = counts.get(ch, 0) + 1
    for ch in sorted(counts):
        folds.append(f"{ch}:{counts[ch]}")
    material = canonical_str(folds)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()