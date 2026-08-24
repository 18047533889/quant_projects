"""TrialValidator contracts with identity.

P0-11: production search must run behind a validator that carries a
machine-readable identity (id / version / implementation hash).  This module
defines that contract and a concrete grammar validator that wraps the
existing ``factor_optimizer.grammar.validation.MutationValidator``.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from factor_optimizer.contracts.candidate_mutation import CandidateMutation
from factor_optimizer.contracts.trial import Trial
from factor_optimizer.grammar.registry import MutationRegistry, get_mutation_registry
from factor_optimizer.grammar.validation import MutationValidator


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


@dataclass(frozen=True)
class TrialValidatorIdentity:
    """Immutable identity of a trial validator implementation.

    ``implementation_hash`` pins the exact validator code/grammar payload the
    identity claims; it must be a hex digest of at least 16 characters so a
    truncated/hand-wavy hash cannot pass as a production identity.
    """

    validator_id: str
    validator_version: str
    implementation_hash: str

    def __post_init__(self) -> None:
        for name in ("validator_id", "validator_version", "implementation_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.implementation_hash) < 16 or not _HEX_RE.match(
            self.implementation_hash
        ):
            raise ValueError(
                "implementation_hash must be a hex digest of at least 16 characters"
            )

    def as_dict(self) -> Dict[str, str]:
        return {
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "implementation_hash": self.implementation_hash,
        }


@runtime_checkable
class ProductionTrialValidator(Protocol):
    """A trial validator that carries a verifiable identity.

    ``validate`` must return a dict with a boolean ``is_legal`` and may carry
    ``research_only`` (default False), ``errors`` (list of str), and
    ``checks_passed`` (list of str).
    """

    identity: TrialValidatorIdentity

    def validate(self, trial: Any) -> Dict[str, Any]:
        ...


def _specs_canonical_json(registry: MutationRegistry) -> str:
    """Build a canonical JSON payload over the registry's mutation specs.

    Only semantically load-bearing spec fields are hashed so cosmetic
    description drift does not invalidate a deployed validator identity.
    """
    specs = []
    for spec in registry.list_specs():
        params = []
        for param in spec.parameters:
            params.append(
                {
                    "name": param.name,
                    "kind": param.kind.value,
                    "role": param.role.value,
                    "required": param.required,
                    "default": param.default,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "allowed_values": param.allowed_values,
                }
            )
        specs.append(
            {
                "mutation_type": spec.mutation_type,
                "version": spec.version,
                "parameters": params,
                "requires_single_parent": spec.requires_single_parent,
                "requires_multiple_parents": spec.requires_multiple_parents,
                "domains": sorted(spec.domains),
                "sources": sorted(spec.sources),
            }
        )
    payload = {
        "registry_version": registry.version(),
        "specs": specs,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class MutationGrammarValidator:
    """Concrete production validator wrapping the grammar ``MutationValidator``.

    Exposes a ``TrialValidatorIdentity`` derived from the mutation registry
    version plus a sha256 of the canonical registered mutation specs, and
    validates trials through the grammar validator's ``CandidateMutation``
    interface (adapting a plain ``Trial`` when no CandidateMutation is
    attached).
    """

    VALIDATOR_ID = "factor_optimizer.mutation_grammar"
    VALIDATOR_VERSION = "0.1.0"

    def __init__(
        self,
        registry: Optional[MutationRegistry] = None,
        implementation_hash: Optional[str] = None,
    ):
        self.registry = registry or get_mutation_registry()
        self._mutation_validator = MutationValidator(registry=self.registry)
        if implementation_hash is None:
            implementation_hash = hashlib.sha256(
                _specs_canonical_json(self.registry).encode("utf-8")
            ).hexdigest()
        self.identity = TrialValidatorIdentity(
            validator_id=self.VALIDATOR_ID,
            validator_version=self.VALIDATOR_VERSION,
            implementation_hash=implementation_hash,
        )

    def _to_candidate_mutation(self, trial: Trial) -> CandidateMutation:
        candidate = getattr(trial, "candidate_mutation", None)
        if isinstance(candidate, CandidateMutation):
            return candidate
        mutation_type = getattr(trial, "mutation_type", None)
        if mutation_type is None:
            # Denormalized trial: the mutation_id names the mutation; the
            # grammar type is carried in metadata when the caller does not
            # attach a full CandidateMutation.
            metadata = getattr(trial, "metadata", {}) or {}
            mutation_type = metadata.get("mutation_type", trial.mutation_id)
        params = (getattr(trial, "metadata", {}) or {}).get("params", {})
        if not isinstance(params, dict):
            params = {}
        return CandidateMutation(
            mutation_id=trial.mutation_id,
            mutation_spec_version="0.1.0",
            parent_factor_ids=list(getattr(trial, "parent_factor_ids", []) or []),
            mutation_type=mutation_type,
            parameters=dict(params),
            trial_ref=trial.trial_id,
        )

    def validate(self, trial: Trial) -> Dict[str, Any]:
        """Validate a trial and return a production-shaped legality dict."""
        if not isinstance(trial, Trial):
            return {
                "is_legal": False,
                "research_only": False,
                "validator_id": self.identity.validator_id,
                "validator_version": self.identity.validator_version,
                "implementation_hash": self.identity.implementation_hash,
                "errors": ["trial must be a Trial"],
                "checks_passed": [],
            }
        try:
            candidate = self._to_candidate_mutation(trial)
        except Exception as exc:
            return {
                "is_legal": False,
                "research_only": False,
                "validator_id": self.identity.validator_id,
                "validator_version": self.identity.validator_version,
                "implementation_hash": self.identity.implementation_hash,
                "errors": [f"trial cannot be mapped to a CandidateMutation: {exc}"],
                "checks_passed": [],
            }
        result = self._mutation_validator.validate(candidate)
        errors: List[str] = list(result.errors)
        checks_passed: List[str] = []
        if result.is_valid:
            checks_passed.append("grammar")
        if errors:
            checks_passed = []
        if result.warnings:
            checks_passed = list(result.warnings)
        return {
            "is_legal": result.is_valid,
            "research_only": False,
            "validator_id": self.identity.validator_id,
            "validator_version": self.identity.validator_version,
            "implementation_hash": self.identity.implementation_hash,
            "errors": errors,
            "checks_passed": checks_passed,
        }


__all__ = [
    "TrialValidatorIdentity",
    "ProductionTrialValidator",
    "MutationGrammarValidator",
    "_specs_canonical_json",
]
