"""Contracts for the default-qualified versus opted-in legacy source auto policy."""

import pytest

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.source_auto_authority import (
    legacy_auto_authorized,
    validate_source_auto_policy,
)


def test_accepts_only_the_two_public_source_auto_policy_values():
    assert validate_source_auto_policy("qualified_only") == "qualified_only"
    assert validate_source_auto_policy("legacy_measured") == "legacy_measured"


@pytest.mark.parametrize("value", [None, True, False, 0, "", "qualified", "legacy", "QUALIFIED_ONLY"])
def test_rejects_invalid_source_auto_policy_before_source_access(value):
    # A permissive parser would silently re-enable static GPU routes through a typo or truthy value.
    with pytest.raises(InvalidContractError):
        validate_source_auto_policy(value)


def test_actual_f48_static_envelope_requires_explicit_legacy_authority():
    """An existing full-market envelope must not itself authorize default auto."""
    from quant_evaluator.runtime.source_auto_evidence import select_source_auto_route

    route = select_source_auto_route(
        shape=(2586, 5461, 48),
        metrics=("rank_ic", "quantile_spread", "factor_turnover_rate"),
        source_dtype="float64", label_dtype="float64", requested_tile_width=16,
    )
    assert route is not None
    assert route.evidence_id == "real_cos_f48_mixed_three_cap16_tile2"
    assert route.effective_tile_width == 2
    absence = dict(
        qualification_status="not_available_legacy_fallback",
        qualification_reason="qualified_cache_miss",
        provider_status="provider_not_configured", qualification_supplied=False,
    )
    assert legacy_auto_authorized(policy="qualified_only", **absence) is False
    assert legacy_auto_authorized(policy="legacy_measured", **absence) is True
    assert legacy_auto_authorized(
        policy="legacy_measured", **{**absence, "provider_status": "report_invalid"}
    ) is False


def test_only_opted_in_pure_qualification_absence_authorizes_legacy_auto():
    assert legacy_auto_authorized(
        policy="legacy_measured",
        qualification_status="not_available_legacy_fallback",
        qualification_reason="qualified_cache_miss",
        provider_status="provider_not_configured",
        qualification_supplied=False,
    ) is True
    assert legacy_auto_authorized(
        policy="legacy_measured",
        qualification_status="not_available_legacy_fallback",
        qualification_reason="qualified_profile_cache_miss",
        provider_status="candidate_not_found",
        qualification_supplied=False,
    ) is True


@pytest.mark.parametrize(
    "changes",
    [
        {"policy": "qualified_only"},
        {"qualification_supplied": True},
        {"qualification_supplied": 1},
        {"qualification_status": "rejected_legacy_fallback"},
        {"qualification_status": "guard_error_legacy_fallback"},
        {"qualification_status": "qualified_current_source"},
        {"qualification_reason": "qualified_receipt_mismatch"},
        {"provider_status": "report_invalid"},
        {"provider_status": "report_unreadable"},
        {"provider_status": "fingerprint_mismatch"},
        {"provider_status": "provider_error"},
        {"provider_status": "candidate_found"},
        {"provider_status": "candidate_rejected"},
        {"provider_status": None},
    ],
)
def test_legacy_optin_never_overrides_supplied_rejected_or_nonmiss_evidence(changes):
    # Any evidence-bearing, rejected, malformed, or error state must fail closed even with opt-in.
    args = {
        "policy": "legacy_measured",
        "qualification_status": "not_available_legacy_fallback",
        "qualification_reason": "qualified_cache_miss",
        "provider_status": "candidate_not_found",
        "qualification_supplied": False,
    }
    args.update(changes)
    assert legacy_auto_authorized(**args) is False


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"policy": "legacy_measured", "qualification_supplied": False},
        {"policy": "legacy_measured", "qualification_status": True,
         "qualification_reason": "qualified_cache_miss", "provider_status": "candidate_not_found",
         "qualification_supplied": False},
        {"policy": "legacy_measured", "qualification_status": "not_available_legacy_fallback",
         "qualification_reason": True, "provider_status": "candidate_not_found",
         "qualification_supplied": False},
    ],
)
def test_legacy_authority_rejects_missing_or_malformed_state(args):
    # Missing status fields or bool-shaped values must not turn an unknown state into authorization.
    assert legacy_auto_authorized(**args) is False
