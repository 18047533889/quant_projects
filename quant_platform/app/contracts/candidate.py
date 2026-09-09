"""FactorCandidateManifest + _READY protocol.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.4 (spec §10.1,
§10.2). PURE stdlib frozen dataclass. The COS candidate directory is only
accepted by the Ingestion Service when ``_READY`` exists and the manifest
checksum verifies.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FactorCandidateManifest",
    "READY_MARKER_NAME",
    "is_ready_marker",
]

# spec §10.1: the readiness marker file must be named exactly "_READY" and be empty.
READY_MARKER_NAME = "_READY"


def is_ready_marker(filename: str) -> bool:
    """True iff ``filename`` is the readiness marker (exact ``_READY``)."""
    return filename == READY_MARKER_NAME


@dataclass(frozen=True)
class FactorCandidateManifest:
    """manifest.json fields (spec §10.2)."""

    schema_version: str
    candidate_id: str
    submitted_at: str
    submitted_by: str
    generator_type: str
    generator_version: str
    market: str
    frequency: str
    formula_language: str
    factor_spec_uri: str
    factor_spec_sha256: str
    parent_factor_ids: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    semantic_family_hint: str | None = None
    campaign_id: str | None = None
    attempt_id: str | None = None
    factor_definition_ref: str | None = None
    semantic_ref: str | None = None
    factor_value_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if not self.submitted_at:
            raise ValueError("submitted_at is required")
        if not self.submitted_by:
            raise ValueError("submitted_by is required")
        if not self.generator_type:
            raise ValueError("generator_type is required")
        if not self.generator_version:
            raise ValueError("generator_version is required")
        if not self.market:
            raise ValueError("market is required")
        if not self.frequency:
            raise ValueError("frequency is required")
        if not self.formula_language:
            raise ValueError("formula_language is required")
        if not self.factor_spec_uri:
            raise ValueError("factor_spec_uri is required")
        if not self.factor_spec_sha256:
            raise ValueError("factor_spec_sha256 is required")
