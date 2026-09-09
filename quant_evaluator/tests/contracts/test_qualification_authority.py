import pytest

from quant_evaluator.contracts.qualification import (
    NumericalQualificationReceipt,
    UseAdmissionReceipt,
)


def test_numerical_qualification_is_execution_scoped_only():
    receipt = NumericalQualificationReceipt(
        "tree", "impl", "route", "cpu", "domain", "metric", "run",
        {"golden": "PASS"},
    )
    assert receipt.require_scope(source_tree_hash="tree", implementation_hash="impl",
        route="route", backend="cpu", parameter_domain_hash="domain",
        metric_instance_hash="metric") is receipt
    assert receipt.require_assertion_suite(suite_id="qe.standard.v1",required_assertions=("golden",)) is receipt
    with pytest.raises(ValueError,match="suite"):
        receipt.require_assertion_suite(suite_id="qe.standard.v1",required_assertions=("unexecuted",))
    assert not hasattr(receipt, "require_use_admission")


def test_use_admission_is_candidate_and_purpose_scoped_and_revocable():
    receipt = UseAdmissionReceipt("factor:a", "PRODUCTION", ("oos:1",), "policy:1",
        "2026-09-09T00:00:00Z", "2026-12-31T00:00:00Z", 3,evidence_bundle_ref="bundle:a")
    assert receipt.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
        evidence_bundle_ref="bundle:a",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=3) is receipt
    with pytest.raises(ValueError,match="scope"):
        receipt.require_use_admission(purpose="RESEARCH",candidate_id="factor:a",
            evidence_bundle_ref="bundle:a",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=3)
    with pytest.raises(ValueError,match="scope"):
        receipt.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
            evidence_bundle_ref="bundle:a",current_time="2027-01-01T00:00:00Z",current_revocation_epoch=3)
    with pytest.raises(ValueError,match="scope"):
        receipt.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
            evidence_bundle_ref="bundle:a",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=4)
    revoked = UseAdmissionReceipt("factor:a", "PRODUCTION", ("oos:1",), "policy:1",
        "2026-09-09T00:00:00Z", "2026-12-31T00:00:00Z", 4, status="REVOKED",evidence_bundle_ref="bundle:a")
    with pytest.raises(ValueError,match="scope"):
        revoked.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
            evidence_bundle_ref="bundle:a",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=4)
    with pytest.raises(ValueError,match="scope"):
        receipt.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
            evidence_bundle_ref="bundle:old",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=3)
    with pytest.raises(ValueError,match="scope"):
        receipt.require_use_admission(purpose="PRODUCTION",candidate_id="factor:a",
            evidence_bundle_ref="bundle:a",current_time="2026-09-09T00:00:00Z",current_revocation_epoch=True)
