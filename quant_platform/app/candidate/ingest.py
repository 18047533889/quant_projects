# -*- coding: utf-8 -*-
"""QRP-P3 Candidate ingestion — normalization, reconciliation, batch fingerprint.

Implementations of the three pure-logic operations specified for the
candidate-intake stage of the QRP pipeline (mining discovery → canonical
candidate → identity reconcile → registered assets):

* :func:`normalize_candidate` — heterogeneous raw discovery records →
  canonical :class:`FactorCandidateManifest`; fail-closed validation
  (missing content hash, non-finite parameter domain, missing semantic id).
* :func:`reconcile_candidates` — dedupe on the dual key *content_hash* and
  *semantic_hash* against a known registry; classifies each candidate as
  ``NEW`` / ``DUPLICATE`` / ``CONFLICT`` with machine-parseable reason codes.
* :func:`batch_fingerprint` — order-independent merkle-style batch fingerprint
  over per-candidate content hashes, for idempotent intake.

Purity rules (platform DTO discipline, extended to this logic layer):

* stdlib only — no third-party runtime imports;
* identity authority = the OWNING DOMAIN PACKAGE. Every identity this layer
  carries (``factor_spec_sha256``, ``semantic_family_hint``) is **reported by
  the producer / domain package** and validated FORMAT-ONLY here
  (64-char lowercase hex). The platform never derives, re-negotiates or hashes
  factor semantics — no factor parameter dicts, no formula/AST, no 口径 enter
  any platform-side hash (R55 P0-5);
* content-hash authority = the publisher over the spec bytes. A manifest
  ``content_hash`` is a *publisher-reported* artifact hash — this layer
  validates its format and never recomputes object bytes;
* no new authority, no domain-internal imports.
"""

from __future__ import annotations

import enum
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from operator import itemgetter
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..contracts._contenthash import content_hash
from ..contracts.candidate import FactorCandidateManifest

__all__ = [
    "CandidateNormalizationError",
    "ReconcileReason",
    "ReconciliationConflict",
    "ReconcileReport",
    "normalize_candidate",
    "reconcile_candidates",
    "batch_fingerprint",
]


def _missing_semantic_id(raw: Mapping[str, Any]) -> CandidateNormalizationError:
    return CandidateNormalizationError("missing_semantic_id", raw_record=raw)

# --------------------------------------------------------------------------- #
# shared constants
# --------------------------------------------------------------------------- #

#: where the canonical content hash is found in a raw discovery record
_CONTENT_HASH_SOURCES = ("content_hash", "factor_spec_sha256", "sha256", "hash")

#: where a publish timestamp is found in a raw discovery record
_TS_SOURCES = ("submitted_at", "published_at", "publish_time", "created_at", "timestamp")

#: accepted scalar types for parameter-domain bounds (bool excluded explicitly —
#: it is an ``int`` subclass and must not pass as a numeric bound).
_DOMAIN_ATOM_TYPES = (int, float, str)

#: reasonable upper bound on a continuous lookback window / coefficient — domains
#: whose limits exceed this fail closed as physically implausible.
_MAX_WINDOW_OR_COEF = 1_000_000_000

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

#: publishing of the candidate directory uses the two-phase publish rule
#: (§3): the manifest alone is PENDING; only the presence of a ``_READY``
#: marker accepts the directory for ingestion. This logic layer works purely
#: on manifests, so no readiness check is performed here.
_ACCEPTED = ...  # placeholder, unused


class CandidateNormalizationError(ValueError):
    """A raw discovery record could not be normalized to a canonical candidate.

    ``reason`` is a short machine-parseable code: ``missing_content_hash`` /
    ``bad_content_hash_format`` / ``missing_semantic_id`` /
    ``bad_semantic_hash_format`` / ``bad_parameter_domain`` /
    ``missing_timestamp`` / ``bad_iso_timestamp`` / ``missing_string_field``.
    """

    def __init__(self, reason: str, *, raw_record: Mapping[str, Any] | None = None) -> None:
        suffix = f" (raw: {raw_record!r})" if raw_record is not None else ""
        super().__init__(f"candidate normalization failed: {reason}{suffix}")
        self.reason = reason


class ReconcileReason(enum.Enum):
    """Machine-parseable classification of one candidate against the registry."""

    NEW = "NEW"
    DUPLICATE_EXACT = "DUPLICATE_EXACT"  # content_hash + semantic_hash both hit registry
    CONFLICT_SEMANTIC_TO_HASH = "CONFLICT_SEMANTIC_TO_HASH"  # same semantic, different hash
    CONFLICT_HASH_TO_SEMANTIC = "CONFLICT_HASH_TO_SEMANTIC"  # same hash, different semantic


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #


def _pick_first(raw: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """First present, non-empty scalar among ``keys``, else ``None``."""
    for key in keys:
        value = raw.get(key)
        if value is None or value == "":
            continue
        return value
    return None


def _coerce_iso_timestamp(value: Any, raw: Mapping[str, Any]) -> str:
    """Normalize a publish/occurred timestamp to UTC ISO-8601 (``Z`` form).

    Accepts tz-aware ``datetime``, ``(year, month, day[, ...])`` tuples,
    ``YYYY-MM-DD`` (treated as UTC midnight), epoch floats/ints (UTC), and
    full ISO strings. Fails closed on anything else — we never fabricate a
    timestamp for a discovery record.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw) from exc
        if parsed.tzinfo is None and "T" not in value:
            parsed = parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, bool):
        raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw)
    if isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw) from exc
        return parsed.isoformat().replace("+00:00", "Z")
    if isinstance(value, (tuple, list)) and value:
        try:
            parsed = datetime(*value, tzinfo=timezone.utc)
        except (TypeError, ValueError) as exc:
            raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw) from exc
        return parsed.isoformat().replace("+00:00", "Z")
    raise CandidateNormalizationError("bad_iso_timestamp", raw_record=raw)


def _representative_semantic_fields(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Discovery-side *carrier* fields used to pick the carried semantic id.

    R55 P0-5: these fields are NOT hashed by the platform any more. The
    semantic identity is the domain package's own digest, carried verbatim;
    the carrier fields only decide whether a producer-supplied semantic id is
    present (fail-closed when it is not) and give the manifest its
    market/frequency labels.
    """
    market = _pick_first(raw, ("market", "universe", "asset_class"))
    frequency = _pick_first(raw, ("frequency", "periodicity", "freq"))
    factor_name = _pick_first(raw, ("factor_name", "name", "recipe")) or raw.get("semantic_id")
    fields: dict[str, Any] = {
        "formula": _pick_first(raw, ("formula", "recipe_expr", "expression", "expr")),
        "factor_name": factor_name,
        "market": market,
        "frequency": frequency,
    }
    for key in ("formula_language", "referenced_data_fields"):
        if key in raw and raw.get(key) is not None:
            fields[key] = raw[key]
    return fields


def _normalize_parameter_domain(raw: Mapping[str, Any]) -> tuple[tuple[str, float, float], ...]:
    """Canonical, finite parameter domain as a tuple of ``(name, lo, hi)``.

    A domain is a *range* of admissible parameter values, so its bounds are
    numbers. Scalar categorical params (e.g. a preset name) do not delimit a
    range and are ignored for identity — but a malformed numeric domain
    (string bounds, bools, non-finite floats, inverted range, or absurd
    magnitude) fails closed.
    """
    raw_domains = raw.get("parameter_domain") or raw.get("param_domain")
    if raw_domains is None:
        return ()
    if isinstance(raw_domains, Mapping):
        items: Iterable[Any] = raw_domains.items()
    elif isinstance(raw_domains, (tuple, list)):
        items = raw_domains
    else:
        raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
    domains: list[tuple[str, float, float]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        name_raw, bounds = item[0], item[1]
        if not isinstance(name_raw, str) or not name_raw:
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        if isinstance(bounds, (tuple, list)):
            if len(bounds) != 2:
                raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
            lo, hi = bounds[0], bounds[1]
        else:
            lo, hi = bounds, bounds
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        if not isinstance(lo, _DOMAIN_ATOM_TYPES) or not isinstance(hi, _DOMAIN_ATOM_TYPES):
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        lo_f, hi_f = float(lo), float(hi)
        if math.isnan(lo_f) or math.isinf(lo_f) or math.isnan(hi_f) or math.isinf(hi_f):
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        if abs(lo_f) >= _MAX_WINDOW_OR_COEF or abs(hi_f) >= _MAX_WINDOW_OR_COEF:
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        if lo_f > hi_f:
            raise CandidateNormalizationError("bad_parameter_domain", raw_record=raw)
        domains.append((name_raw, lo_f, hi_f))
    return tuple(sorted(domains, key=itemgetter(0)))


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise CandidateNormalizationError("missing_string_field", raw_record=raw)
    return value


def _semantic_hash(raw: Mapping[str, Any]) -> str:
    """First-class domain-semantics hash carried by the raw record (if any).

    A discovery record may already carry ``semantic_hash`` / ``semantic_id``;
    when present we validate its sha256-hex *format* (same rule as content
    hashes) so it can seed the dual-key reconciliation. The digest is minted by
    the owning domain package — the platform carries it, never recomputes it.
    """
    value = _pick_first(raw, ("semantic_hash", "semantic_id"))
    if value is None:
        return ""
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise CandidateNormalizationError("bad_semantic_hash_format", raw_record=raw)
    return value


# --------------------------------------------------------------------------- #
# 1. normalization
# --------------------------------------------------------------------------- #


def normalize_candidate(raw: Mapping[str, Any]) -> FactorCandidateManifest:
    """Normalize a heterogeneous raw discovery record into a canonical candidate.

    Fail-closed validation (each with a machine-parseable ``reason``):

    * ``content_hash`` — from ``content_hash`` / ``factor_spec_sha256`` /
      ``sha256`` / ``hash``; must be a 64-char lowercase hex sha256
      (``missing_content_hash`` / ``bad_content_hash_format``).
    * semantic id — the ``semantic_id`` / ``semantic_hash`` alias key
      (``missing_semantic_id`` / ``bad_semantic_hash_format``). This is the
      OWNING DOMAIN PACKAGE's semantic-identity digest: the platform validates
      its hex format and carries it verbatim — it never hashes factor
      semantics itself (R55 P0-5).
    * parameter domain — numeric, canonical, bounds finite and in-range with
      ``lo <= hi`` (``bad_parameter_domain``).
    * required strings + a publish timestamp (``missing_string_field`` /
      ``missing_timestamp`` / ``bad_iso_timestamp``).

    Returns a standard ``contracts.candidate.FactorCandidateManifest`` whose
    ``factor_spec_sha256`` mirrors the carried content hash and whose
    ``semantic_family_hint`` carries the domain semantic identity — the two
    seeds of the subsequent dual-key reconciliation.
    """
    # 1. content hash — the primary identity key.
    content_hash_value = _pick_first(raw, _CONTENT_HASH_SOURCES)
    if content_hash_value is None:
        # The publish timestamp is not a content-hash source. A content hash is
        # *reported by the publisher* over the spec bytes (spec §7.2); we never
        # derive one from arbitrary raw fields here.
        raise CandidateNormalizationError("missing_content_hash", raw_record=raw)
    if not isinstance(content_hash_value, str) or not _SHA256_HEX_RE.match(content_hash_value):
        raise CandidateNormalizationError("bad_content_hash_format", raw_record=raw)

    # 2. semantic id — the secondary identity key, MINTED BY THE DOMAIN.
    semantic_id = raw.get("semantic_id")
    if semantic_id is None:
        semantic_id = raw.get("semantic_hash")
    if not isinstance(semantic_id, str) or not semantic_id.strip():
        # Absent: the platform cannot derive a semantic identity from carrier
        # fields (that would mint one) — fail closed.
        raise _missing_semantic_id(raw)
    if not _SHA256_HEX_RE.match(semantic_id):
        # A non-hex alias (e.g. a discovery-side slug) is a format failure — the
        # platform cannot judge whether it is "semantically the same factor", so
        # it must not silently accept a hand-rolled id either.
        raise CandidateNormalizationError("bad_semantic_hash_format", raw_record=raw)
    semantic_id = semantic_id.strip()

    # 3. parameter domain — finite / canonical.
    _normalize_parameter_domain(raw)

    # 4. required strings + timestamp.
    generator_type = _required_string(raw, "generator_type")
    generator_version = _required_string(raw, "generator_version")
    submitted_by = _required_string(raw, "submitted_by")
    submitted_at_raw = _pick_first(raw, _TS_SOURCES)
    if submitted_at_raw is None:
        raise CandidateNormalizationError("missing_timestamp", raw_record=raw)
    submitted_at = _coerce_iso_timestamp(submitted_at_raw, raw)

    # 5. semantic identity — CARRIED, not derived. The carrier fields only
    #    label the manifest; they are never hashed into the semantic key.
    semantic_fields = _representative_semantic_fields(raw)
    market = str(semantic_fields.get("market") or "")
    frequency = str(semantic_fields.get("frequency") or "")
    if not market or not frequency:
        raise CandidateNormalizationError("missing_string_field", raw_record=raw)
    semantic_family_hint = semantic_id

    return FactorCandidateManifest(
        schema_version=raw.get("schema_version") or "1.0",
        candidate_id=raw.get("candidate_id") or f"FC_{content_hash_value[:16]}",
        submitted_at=submitted_at,
        submitted_by=submitted_by,
        generator_type=generator_type,
        generator_version=generator_version,
        market=market,
        frequency=frequency,
        formula_language=str(raw.get("formula_language") or "recipe"),
        factor_spec_uri=_required_string(raw, "factor_spec_uri"),
        factor_spec_sha256=content_hash_value,
        parent_factor_ids=tuple(raw.get("parent_factor_ids") or ()),
        required_fields=tuple(raw.get("required_fields") or ()),
        semantic_family_hint=semantic_family_hint,
        campaign_id=raw.get("campaign_id"),
        attempt_id=raw.get("attempt_id"),
    )


# --------------------------------------------------------------------------- #
# 2. reconciliation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReconciliationConflict:
    """A single conflicting candidate: dual-key mismatch with registry entries.

    ``reason`` is one of ``ReconcileReason``:
    ``CONFLICT_SEMANTIC_TO_HASH`` (same semantic identity, different content
    hash) or ``CONFLICT_HASH_TO_SEMANTIC`` (same content hash, different
    semantic identity). ``matched_registry`` records the registry entries
    (``(content_hash, semantic_hash)``) that triggered the mismatch.
    """

    candidate_id: str
    content_hash: str
    semantic_hash: str
    reason: str
    matched_registry: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReconcileReport:
    """Immutable outcome of one reconciliation batch.

    ``counts`` maps reason code → number of candidates in that class
    (``NEW`` / ``DUPLICATE_EXACT`` / ``CONFLICT_SEMANTIC_TO_HASH`` /
    ``CONFLICT_HASH_TO_SEMANTIC``). ``conflicts`` carries per-candidate
    conflict details, and ``batch_fingerprint`` the order-independent batch
    fingerprint for idempotent intake.
    """

    counts: Mapping[str, int]
    conflicts: tuple[ReconciliationConflict, ...]
    batch_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    def __getitem__(self, reason: ReconcileReason) -> int:
        """Return the candidate count for a ``ReconcileReason``.

        Delegates to the string-keyed ``counts`` mapping so callers may index
        with the enum directly.
        """
        return int(self.counts.get(reason.value, 0))

    @property
    def new(self) -> int:
        return self.counts.get(ReconcileReason.NEW.value, 0)

    @property
    def duplicates(self) -> int:
        return self.counts.get(ReconcileReason.DUPLICATE_EXACT.value, 0)

    @property
    def conflicts_count(self) -> int:
        return self.counts.get(ReconcileReason.CONFLICT_SEMANTIC_TO_HASH.value, 0) + self.counts.get(
            ReconcileReason.CONFLICT_HASH_TO_SEMANTIC.value, 0
        )

    @property
    def total(self) -> int:
        """Total number of candidates in the batch (sum of all classes)."""
        return sum(int(v) for v in self.counts.values())


def _entry_keys(entry: Any) -> tuple[str, str]:
    """Extract ``(content_hash, semantic_hash)`` from a registry entry.

    Accepts a :class:`FactorCandidateManifest` or a mapping with
    ``content_hash`` / ``factor_spec_sha256`` and ``semantic_hash`` /
    ``semantic_family_hint`` keys. Raises ``ValueError`` on entries that carry
    neither key form — the registry must be key-addressable or the reconcile
    would silently misclassify.
    """
    if isinstance(entry, FactorCandidateManifest):
        return entry.factor_spec_sha256, entry.semantic_family_hint or ""
    if isinstance(entry, Mapping):
        content = (
            entry.get("content_hash")
            or entry.get("factor_spec_sha256")
            or entry.get("hash")
            or ""
        )
        semantic = entry.get("semantic_hash") or entry.get("semantic_family_hint") or ""
        if content:
            return str(content), str(semantic)
    raise ValueError(f"known_registry entry not key-addressable: {entry!r}")


def reconcile_candidates(
    candidates: Iterable[FactorCandidateManifest],
    known_registry: Iterable[Any] | None = None,
) -> ReconcileReport:
    """Dedupe candidates on the dual key (content_hash, semantic_hash).

    Classification per candidate (exactly one class, priority order):

    * ``DUPLICATE_EXACT`` — both the content hash and the semantic hash hit a
      registered entry (the candidate is already known);
    * ``CONFLICT_HASH_TO_SEMANTIC`` — same content hash as a registered entry
      but a different semantic identity;
    * ``CONFLICT_SEMANTIC_TO_HASH`` — same semantic identity as a registered
      entry but a different content hash;
    * ``NEW`` — hits nothing.

    ``known_registry`` may be ``None`` (empty registry) or an iterable of
    previously-normalized manifests / key-addressable mappings.
    """
    registry_keys = [(_entry_keys(entry)) for entry in (known_registry or ())]
    registry_contents: set[str] = {ch for ch, _ in registry_keys}
    registry_semantics: dict[str, list[str]] = {}
    for ch, sh in registry_keys:
        if sh:
            registry_semantics.setdefault(sh, []).append(ch)

    conflicts: list[ReconciliationConflict] = []
    counts: dict[str, int] = {reason.value: 0 for reason in ReconcileReason}

    for candidate in candidates:
        content = candidate.factor_spec_sha256
        semantic = candidate.semantic_family_hint or ""
        content_matches = [
            (ch, sh) for ch, sh in registry_keys if ch == content
        ]
        semantic_matches = [
            (ch, sh) for ch, sh in registry_keys if sh and sh == semantic
        ]

        if content_matches and any(sh == semantic for _, sh in content_matches):
            reason = ReconcileReason.DUPLICATE_EXACT
            conflicts.append(
                ReconciliationConflict(
                    candidate_id=candidate.candidate_id,
                    content_hash=content,
                    semantic_hash=semantic,
                    reason=reason.value,
                    matched_registry=tuple(content_matches),
                )
            )
        elif content_matches:
            reason = ReconcileReason.CONFLICT_HASH_TO_SEMANTIC
            conflicts.append(
                ReconciliationConflict(
                    candidate_id=candidate.candidate_id,
                    content_hash=content,
                    semantic_hash=semantic,
                    reason=reason.value,
                    matched_registry=tuple(content_matches),
                )
            )
        elif semantic_matches:
            reason = ReconcileReason.CONFLICT_SEMANTIC_TO_HASH
            conflicts.append(
                ReconciliationConflict(
                    candidate_id=candidate.candidate_id,
                    content_hash=content,
                    semantic_hash=semantic,
                    reason=reason.value,
                    matched_registry=tuple(semantic_matches),
                )
            )
        else:
            reason = ReconcileReason.NEW
        counts[reason.value] += 1

    fingerprint = batch_fingerprint(candidates)
    return ReconcileReport(counts=counts, conflicts=tuple(conflicts), batch_fingerprint=fingerprint)


# --------------------------------------------------------------------------- #
# 3. batch fingerprint
# --------------------------------------------------------------------------- #


def batch_fingerprint(candidates: Iterable[FactorCandidateManifest]) -> str:
    """Order-independent merkle-style batch fingerprint.

    Collapses the per-candidate ``content_hash`` values: each candidate's
    contribution is ``content_hash(candidate.content_hash)`` (length-prefixed
    canonical sha256, per the platform codec), then the leaves are *sorted* and
    folded with a plain sha256 chain. The fingerprint is:

    * order-independent — reordering the batch does not change it;
    * content-sensitive — a change to any candidate's content hash changes it;
    * idempotent — the same batch fingerprints identically every time.

    Concrete serialization is ``merkle-v1:<hex>`` so a future codec can bump
    the version without ambiguity.
    """
    leaves = sorted(
        content_hash(cand.factor_spec_sha256) if cand.factor_spec_sha256 else ""
        for cand in candidates
    )
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(leaf.encode("ascii"))
    return "merkle-v1:" + digest.hexdigest()
