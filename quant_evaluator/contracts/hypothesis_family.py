"""Immutable complete confirmation-family input for QE correction authority."""
from dataclasses import dataclass
from typing import Mapping
import math
from .metric_artifacts import FrozenMapping
from ._hashutil import stable_content_hex


@dataclass(frozen=True)
class HypothesisFamilyArtifact:
    family_id: str
    campaign_id: str
    policy_hash: str
    comparison_context_hash: str
    members: tuple[Mapping, ...]
    ledger_head: str
    ledger_count: int
    correction_method: str = 'holm'
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.family_id, self.campaign_id, self.policy_hash, self.comparison_context_hash, self.ledger_head)):
            raise ValueError('complete family identity and ledger seal required')
        if type(self.ledger_count) is not int or self.ledger_count < 1:
            raise ValueError('ledger_count must be positive integer')
        if self.correction_method not in ('holm', 'bonferroni', 'sidak', 'bh', 'by'):
            raise ValueError('unsupported family correction')
        members = tuple(FrozenMapping(m) for m in self.members)
        ids = tuple(m.get('hypothesis_id') for m in members)
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('nonempty unique predeclared hypothesis identities required')
        for m in members:
            for key in ('effective_spec_hash', 'evaluation_intent_hash', 'horizon', 'label_ref', 'direction'):
                if not m.get(key):
                    raise ValueError(f'missing member {key}')
            if m.get('status') not in ('PENDING', 'COMPUTED', 'FAILED', 'NOT_TESTED'):
                raise ValueError('unknown member state')
            if m['status'] == 'COMPUTED':
                p = m.get('pvalue')
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 or not m.get('pvalue_ref'):
                    raise ValueError('computed member requires legal p and evidence ref')
            elif m.get('pvalue') is not None:
                raise ValueError('noncomputed member cannot claim a measured p-value')
        object.__setattr__(self, 'members', tuple(sorted(members, key=lambda m: m['hypothesis_id'])))
        object.__setattr__(self, 'assumptions', tuple(self.assumptions))
        if self.correction_method == 'sidak' and 'independent_tests' not in self.assumptions:
            raise ValueError('Sidak requires declared independent tests')
        if self.correction_method == 'bh' and not {'independent_tests', 'positive_dependence'}.intersection(self.assumptions):
            raise ValueError('BH requires independence or positive-dependence assumption')

    @property
    def content_hash(self):
        return stable_content_hex(tag="qe.hypothesis_family.v1", fields={"family_id": self.family_id, "campaign_id": self.campaign_id, "policy_hash": self.policy_hash,
            "comparison_context_hash": self.comparison_context_hash, "members": self.members, "ledger_head": self.ledger_head, "ledger_count": self.ledger_count,
            "correction_method": self.correction_method, "assumptions": self.assumptions})
