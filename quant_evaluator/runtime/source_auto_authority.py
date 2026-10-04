"""Fail-closed policy gate for historical static factor-source auto routes."""
from __future__ import annotations

from quant_evaluator.contracts.errors import InvalidContractError


_SOURCE_AUTO_POLICIES = frozenset({"qualified_only", "legacy_measured"})
_LEGACY_MISS_REASONS = frozenset({
    "qualified_cache_miss",
    "qualified_profile_cache_miss",
})
_ABSENT_PROVIDER_STATUSES = frozenset({
    "provider_not_configured",
    "candidate_not_found",
})


def validate_source_auto_policy(value: object) -> str:
    """Accept only explicit public source-auto policy literals."""
    if type(value) is not str or value not in _SOURCE_AUTO_POLICIES:
        raise InvalidContractError(
            "source_auto_policy must be 'qualified_only' or 'legacy_measured'"
        )
    return value


def legacy_auto_authorized(
    *, policy: str | None = None,
    qualification_status: object = None,
    qualification_reason: object = None,
    provider_status: object = None,
    qualification_supplied: object = None,
) -> bool:
    """Allow static evidence only for an explicitly opted-in pure cache miss.

    Present or rejected evidence, provider failures, and malformed state all
    fail closed. The caller invokes this only after current qualification has
    failed to produce a route.
    """
    return (
        type(policy) is str
        and policy == "legacy_measured"
        and type(qualification_status) is str
        and qualification_status == "not_available_legacy_fallback"
        and type(qualification_reason) is str
        and qualification_reason in _LEGACY_MISS_REASONS
        and type(provider_status) is str
        and provider_status in _ABSENT_PROVIDER_STATUSES
        and qualification_supplied is False
    )


__all__ = ("legacy_auto_authorized", "validate_source_auto_policy")
