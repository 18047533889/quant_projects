"""Trusted-store binding counterexamples with a response frozen up front."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from factor_assets.selection import CandidateEvidence, DecisionProvider
from factor_assets.tests.test_v8_selection_decision import (
    Receipt,
    candidate,
    policy,
    request,
)


def _response(c, p):
    raw = c.raw_joint_metric_evidence
    return SimpleNamespace(
        candidate_id=c.candidate_id,
        raw_content_hash=raw.content_hash,
        recipe_hash=raw.recipe_hash,
        fitted_state_hash=raw.fitted_state_hash,
        data_snapshot_hash=raw.data_snapshot_hash,
        universe_hash=raw.universe_hash,
        label_hash=raw.label_hash,
        value_artifact_hash=raw.value_artifact_hash,
        dimensions=dict(c.dimensions),
        coverage=c.coverage,
        metric_instance_hash=raw.metric_instance_hash,
        parameter_domain_hash=raw.parameter_domain_hash,
        metric_directions={r.metric_id: r.direction.value for r in p.metric_rules},
    )


class FrozenTrustedStore:
    """Returns only responses captured before any attacked candidate exists."""

    def __init__(self, responses):
        self._responses = dict(responses)

    def resolve(self, ref):
        return Receipt()

    def resolve_evidence(self, ref, *, candidate):
        return self._responses[ref]


def _provider_and_baseline():
    p = policy()
    baseline = candidate("raw")
    store = FrozenTrustedStore({baseline.evidence_bundle_ref: _response(baseline, p)})
    return p, baseline, DecisionProvider(p, store, store)


def _copy(c, **changes):
    fields = dict(
        candidate_id=c.candidate_id, evidence_refs=c.evidence_refs,
        fidelity=c.fidelity, dimensions=c.dimensions, metrics=c.metrics,
        coverage=c.coverage, qualification_scope=c.qualification_scope,
        raw_joint_metric_evidence=c.raw_joint_metric_evidence,
        qualification_ref=c.qualification_ref,
        qualification_requirements=c.qualification_requirements,
        health_evidence_ref=c.health_evidence_ref,
        health_candidate_id=c.health_candidate_id,
        health_coverage=c.health_coverage,
        evidence_bundle_ref=c.evidence_bundle_ref,
    )
    fields.update(changes)
    return CandidateEvidence(**fields)


def _rehash(raw, **changes):
    """Return a self-consistent altered raw envelope under caller control."""
    get = lambda name: changes.get(name, getattr(raw, name))
    samples = get("samples")
    semantic = {
        "context": raw.comparison_context_hash, "plan": raw.resampling_plan_ref,
        "pairing": (raw.resampling_plan_content_hash, raw.sample_identity_hash,
                    raw.time_identity_hash, raw.common_mask_hash),
        "replicates": list(raw.replicate_ids), "metrics": list(raw.metric_ids),
        "metric_units": list(raw.metric_units), "window_ids": list(raw.window_ids),
        "scenario_ids": list(raw.scenario_ids),
        "samples": json.loads(json.dumps(samples)),
        "qualification": raw.qualification_scope,
        "execution": (raw.source_tree_hash, raw.implementation_hash, raw.route,
                      raw.backend, raw.parameter_domain_hash, raw.metric_instance_hash),
        "candidate": (get("candidate_id"), get("recipe_hash"),
                      get("fitted_state_hash"), get("data_snapshot_hash"),
                      get("universe_hash"), get("label_hash"),
                      get("value_artifact_hash")),
    }
    digest = hashlib.sha256(json.dumps(
        semantic, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return replace(raw, **changes, content_hash=digest)


def test_precommitted_trusted_response_accepts_untampered_candidate():
    p, baseline, provider = _provider_and_baseline()
    result = provider.decide(request(p, [baseline]))
    assert result.winner_id == "raw"
    assert result.eligibility["raw"] is True


def test_recomputed_caller_raw_hash_cannot_change_precommitted_store():
    p, baseline, provider = _provider_and_baseline()
    samples = ((((.95,),),), (((.99,),),))
    attacked_raw = _rehash(baseline.raw_joint_metric_evidence, samples=samples)
    attacked = _copy(
        baseline, raw_joint_metric_evidence=attacked_raw,
        metrics={"ic": .97},
    )
    result = provider.decide(request(p, [attacked]))
    assert result.eligibility["raw"] is False
    assert any(r.gate_id == "trusted_evidence_binding" for r in result.gate_receipts)


@pytest.mark.parametrize("field", ["fitted_state_hash", "value_artifact_hash"])
def test_same_candidate_changed_state_or_value_is_rejected(field):
    p, baseline, provider = _provider_and_baseline()
    attacked_raw = _rehash(
        baseline.raw_joint_metric_evidence, **{field: "attacker-controlled"}
    )
    result = provider.decide(request(p, [_copy(
        baseline, raw_joint_metric_evidence=attacked_raw
    )]))
    assert result.eligibility["raw"] is False
    assert any(r.gate_id == "trusted_evidence_binding" for r in result.gate_receipts)


def test_other_candidate_complete_raw_cannot_be_relabelled_as_baseline():
    p, baseline, provider = _provider_and_baseline()
    other = candidate("fixed").raw_joint_metric_evidence
    relabelled = _rehash(other, candidate_id="raw")
    result = provider.decide(request(p, [_copy(
        baseline, raw_joint_metric_evidence=relabelled
    )]))
    assert result.eligibility["raw"] is False
    assert any(r.gate_id == "trusted_evidence_binding" for r in result.gate_receipts)


@pytest.mark.parametrize(
    "changes",
    [
        {"dimensions": {"quality": "A"}},
        {"coverage": .9, "health_coverage": .9},
        {"health_candidate_id": "other"},
    ],
)
def test_health_or_coverage_tampering_cannot_change_precommitted_store(changes):
    p, baseline, provider = _provider_and_baseline()
    attacked = _copy(baseline, **changes)
    result = provider.decide(request(p, [attacked]))
    assert result.eligibility["raw"] is False
    assert any(
        r.gate_id in {"trusted_evidence_binding", "health_evidence_identity"}
        for r in result.gate_receipts
    )
