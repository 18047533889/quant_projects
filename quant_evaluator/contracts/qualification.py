"""Execution-domain-specific numerical receipts; inventory is not authority.

Only a caller's trusted evidence resolver may supply these for admission.
Constructing a receipt or knowing its hash does not establish provenance.
"""
from dataclasses import dataclass
from typing import Mapping
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

    def __post_init__(self):
        for name in ('source_tree_hash', 'implementation_hash', 'route', 'backend',
                     'parameter_domain_hash', 'metric_instance_hash', 'test_run_ref'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'qualification missing {name}')
        if not self.assertions or any(not isinstance(k, str) or not k for k in self.assertions):
            raise ValueError('executed assertion identities required')
        if any(v not in ('PASS', 'FAIL', 'SKIP', 'XFAIL', 'NOT_RUN') for v in self.assertions.values()):
            raise ValueError('unknown assertion outcome')
        object.__setattr__(self, 'assertions', FrozenMapping(self.assertions))

    @property
    def content_hash(self):
        return stable_content_hex(tag='qe.numerical_qualification.v1', fields={
            **{k: getattr(self, k) for k in ('source_tree_hash', 'implementation_hash', 'route', 'backend',
                     'parameter_domain_hash', 'metric_instance_hash', 'test_run_ref', 'evidence_level')},
            'assertions': self.assertions})

    def require_scope(self, *, source_tree_hash, implementation_hash, route, backend,
                      parameter_domain_hash, metric_instance_hash):
        expected = locals().copy(); expected.pop('self')
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError('qualification execution domain mismatch')
        if self.evidence_level != 'NUMERICAL_VALIDATED' or any(v != 'PASS' for v in self.assertions.values()):
            raise ValueError('qualification requires executed PASS assertions, not inventory/heuristics')
        return self
