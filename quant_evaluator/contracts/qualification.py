"""Execution-domain-specific numerical receipts; inventory is not authority.

Only a caller's trusted evidence resolver may supply these for admission.
Constructing a receipt or knowing its hash does not establish provenance.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Tuple
from .metric_artifacts import FrozenMapping
from ._hashutil import stable_content_hex


@dataclass(frozen=True)
class NumericalQualificationReceipt:
    source_tree_hash: str
    implementation_hash: str
    route: str
    backend: str
    parameter_domain_hash: str
    metric_instance_hash: str
    test_run_ref: str
    assertions: Mapping[str, str]
    evidence_level: str = 'NUMERICAL_VALIDATED'
    assertion_suite_id: str = "qe.standard.v1"

    def __post_init__(self):
        for name in ('source_tree_hash', 'implementation_hash', 'route', 'backend',
                     'parameter_domain_hash', 'metric_instance_hash', 'test_run_ref'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'qualification missing {name}')
        if not self.assertions or any(not isinstance(k, str) or not k for k in self.assertions):
            raise ValueError('executed assertion identities required')
        if any(v not in ('PASS', 'FAIL', 'SKIP', 'XFAIL', 'NOT_RUN') for v in self.assertions.values()):
            raise ValueError('unknown assertion outcome')
        if not isinstance(self.assertion_suite_id,str) or not self.assertion_suite_id:
            raise ValueError("qualification assertion_suite_id required")
        object.__setattr__(self, 'assertions', FrozenMapping(self.assertions))

    @property
    def content_hash(self):
        return stable_content_hex(tag='qe.numerical_qualification.v1', fields={
            **{k: getattr(self, k) for k in ('source_tree_hash', 'implementation_hash', 'route', 'backend',
                     'parameter_domain_hash', 'metric_instance_hash', 'test_run_ref', 'evidence_level')},
            'assertions': self.assertions, 'assertion_suite_id': self.assertion_suite_id})

    def require_scope(self, *, source_tree_hash, implementation_hash, route, backend,
                      parameter_domain_hash, metric_instance_hash):
        expected = locals().copy(); expected.pop('self')
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError('qualification execution domain mismatch')
        if self.evidence_level != 'NUMERICAL_VALIDATED' or any(v != 'PASS' for v in self.assertions.values()):
            raise ValueError('qualification requires executed PASS assertions, not inventory/heuristics')
        return self

    def require_assertion_suite(self, *, suite_id, required_assertions):
        if (not suite_id or suite_id != self.assertion_suite_id or not required_assertions
                or any(self.assertions.get(name) != "PASS" for name in required_assertions)):
            raise ValueError("qualification assertion suite mismatch")
        return self


@dataclass(frozen=True)
class UseAdmissionReceipt:
    """Separate, purpose-scoped authority; numerical correctness is insufficient."""
    candidate_id: str
    purpose: str
    statistical_evidence_refs: Tuple[str, ...]
    admission_policy_ref: str
    issued_at: str
    valid_until: str
    revocation_epoch: int
    status: str = "ADMITTED"
    evidence_bundle_ref: str = ""

    def __post_init__(self):
        for name in ("candidate_id","purpose","admission_policy_ref","issued_at","valid_until"):
            if not isinstance(getattr(self,name),str) or not getattr(self,name):
                raise ValueError(f"use admission missing {name}")
        if not isinstance(self.evidence_bundle_ref,str) or not self.evidence_bundle_ref:
            raise ValueError("use admission missing evidence_bundle_ref")
        if not self.statistical_evidence_refs or any(not isinstance(x,str) or not x for x in self.statistical_evidence_refs):
            raise ValueError("use admission requires statistical evidence")
        if isinstance(self.revocation_epoch,bool) or not isinstance(self.revocation_epoch,int) or self.revocation_epoch < 0:
            raise ValueError("invalid revocation epoch")
        if self.status not in ("ADMITTED","REVOKED","EXPIRED"):
            raise ValueError("unknown use admission status")
        issued,expires=self._time(self.issued_at),self._time(self.valid_until)
        if issued >= expires: raise ValueError("use admission validity interval is empty")
        object.__setattr__(self,"statistical_evidence_refs",tuple(self.statistical_evidence_refs))

    @property
    def content_hash(self):
        return stable_content_hex(tag="qe.use_admission.v1", fields={
            "candidate_id":self.candidate_id,"purpose":self.purpose,
            "statistical_evidence_refs":self.statistical_evidence_refs,
            "admission_policy_ref":self.admission_policy_ref,"issued_at":self.issued_at,
            "valid_until":self.valid_until,"revocation_epoch":self.revocation_epoch,
            "status":self.status,"evidence_bundle_ref":self.evidence_bundle_ref})

    def require_use_admission(self, *, purpose, candidate_id, evidence_bundle_ref,
                              current_time, current_revocation_epoch):
        now=self._time(current_time)
        if (self.status != "ADMITTED" or self.purpose != purpose or self.candidate_id != candidate_id
                or self.evidence_bundle_ref != evidence_bundle_ref
                or not self._time(self.issued_at) <= now <= self._time(self.valid_until)
                or isinstance(current_revocation_epoch,bool)
                or not isinstance(current_revocation_epoch,int)
                or current_revocation_epoch != self.revocation_epoch):
            raise ValueError("use admission scope mismatch")
        return self

    @staticmethod
    def _time(value):
        if isinstance(value,str): value=datetime.fromisoformat(value.replace("Z","+00:00"))
        if not isinstance(value,datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone-aware admission timestamp required")
        return value.astimezone(timezone.utc)
