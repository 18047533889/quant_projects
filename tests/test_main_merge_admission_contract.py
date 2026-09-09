"""Boundary tests for immutable policy and purpose-scoped authority."""
from dataclasses import replace
import pytest
from factor_assets.selection import MetricRule, SelectionPolicySpec, UtilityDirection
from quant_evaluator.contracts.qualification import NumericalQualificationReceipt, UseAdmissionReceipt


def _policy(**kwargs):
    rule = MetricRule("ic", UtilityDirection.HIGHER_IS_BETTER, 1., 0., 1.)
    return SelectionPolicySpec("policy", "v1", (rule,), {}, **kwargs)


def _admission(**kwargs):
    fields = dict(candidate_id="a", purpose="PRODUCTION",
        statistical_evidence_refs=("oos:a",), admission_policy_ref="policy:v1",
        issued_at="2026-09-09T08:00:00+08:00", valid_until="2026-09-10T00:00:00Z",
        revocation_epoch=1, evidence_bundle_ref="bundle:a")
    fields.update(kwargs)
    return UseAdmissionReceipt(**fields)


def test_policy_assertion_suite_is_hashed_and_deeply_frozen():
    assertions = ["golden"]
    policy = _policy(required_qualification_assertions=assertions)
    identity = policy.content_hash
    assertions.append("mutated_after_construction")
    assert policy.required_qualification_assertions == ("golden",)
    assert policy.content_hash == identity
    assert replace(policy, qualification_suite_id="new-suite").content_hash != identity
    assert replace(policy, required_qualification_assertions=("golden", "parity")).content_hash != identity
    assert replace(policy, minimum_gain=.01).content_hash != identity


def test_old_policy_positional_dimensions_and_receipt_evidence_level_keep_meaning():
    rule = MetricRule("ic", UtilityDirection.HIGHER_IS_BETTER, 1., 0., 1.)
    assert SelectionPolicySpec("p", "v", (rule,), {}, 0., 0., 0., ("quality",)).required_dimensions == ("quality",)
    receipt = NumericalQualificationReceipt("tree", "impl", "route", "cpu", "domain",
        "metric", "run", {"golden": "PASS"}, "NUMERICAL_VALIDATED")
    assert receipt.evidence_level == "NUMERICAL_VALIDATED"
    assert receipt.assertion_suite_id == "qe.standard.v1"


@pytest.mark.parametrize("bad", ["", None, 1, True])
def test_invalid_qualification_suite_is_not_an_authority(bad):
    with pytest.raises((ValueError, TypeError)):
        NumericalQualificationReceipt("tree", "impl", "route", "cpu", "domain",
            "metric", "run", {"golden": "PASS"}, assertion_suite_id=bad)


@pytest.mark.parametrize("bad", ["", None, 1, True])
def test_admission_requires_a_real_bundle_reference(bad):
    with pytest.raises((ValueError, TypeError)):
        _admission(evidence_bundle_ref=bad)


def test_admission_parses_offsets_instead_of_lexical_date_order():
    receipt = _admission()
    assert receipt.require_use_admission(purpose="PRODUCTION", candidate_id="a",
        evidence_bundle_ref="bundle:a", current_time="2026-09-09T01:00:00Z",
        current_revocation_epoch=1) is receipt


@pytest.mark.parametrize("change", [
    {"evidence_bundle_ref": "bundle:previous-state"},
    {"candidate_id": "other"}, {"purpose": "LIVE"},
    {"current_time": "2026-09-08T23:00:00Z"},
    {"current_time": "2026-09-10T00:00:01Z"},
    {"current_time": "not-a-time"},
    {"current_time": "2026-09-09T01:00:00"},
    {"current_revocation_epoch": True},
    {"current_revocation_epoch": 1.0},
    {"current_revocation_epoch": 2},
])
def test_admission_rejects_stale_bundle_time_purpose_or_epoch(change):
    args = dict(purpose="PRODUCTION", candidate_id="a", evidence_bundle_ref="bundle:a",
        current_time="2026-09-09T01:00:00Z", current_revocation_epoch=1)
    args.update(change)
    with pytest.raises((ValueError, TypeError)):
        _admission().require_use_admission(**args)


@pytest.mark.parametrize("change", [
    {"issued_at": "bad"}, {"valid_until": "2026-09-08T00:00:00Z"},
    {"revocation_epoch": True}, {"status": "REVOKED"},
])
def test_invalid_or_revoked_admission_never_grants_access(change):
    with pytest.raises((ValueError, TypeError)):
        _admission(**change).require_use_admission(purpose="PRODUCTION", candidate_id="a",
            evidence_bundle_ref="bundle:a", current_time="2026-09-09T01:00:00Z",
            current_revocation_epoch=1)
