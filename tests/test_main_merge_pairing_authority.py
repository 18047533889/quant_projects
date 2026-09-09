"""Public decision rejects non-paired, independently valid trusted evidence."""
from dataclasses import replace
import pytest
from factor_assets.selection import DecisionProvider, MetricRule, SelectionPolicySpec, UtilityDirection
from factor_assets.tests.test_v8_selection_decision import _raw_candidate, request
from factor_assets.tests.test_main_merge_trusted_store import FrozenTrustedStore, _response, _rehash


def _decision(pairing=("plan-content", "samples", "times", "mask"), data_change=None):
    policy = SelectionPolicySpec("paired", "1",
        (MetricRule("ic", UtilityDirection.HIGHER_IS_BETTER, 1., 0., 1.),), {})
    baseline = _raw_candidate("raw", ("ic",), ("dimensionless",),
        ((((.6,),),), (((.6,),),)), pairing=("plan-content", "samples", "times", "mask"))
    variant = _raw_candidate("fixed", ("ic",), ("dimensionless",),
        ((((.8,),),), (((.8,),),)), pairing=pairing)
    if data_change:
        variant = replace(variant, raw_joint_metric_evidence=_rehash(
            variant.raw_joint_metric_evidence, **data_change))
    baseline = replace(baseline, evidence_refs=("bundle:raw",), evidence_bundle_ref="bundle:raw")
    variant = replace(variant, evidence_refs=("bundle:fixed",), evidence_bundle_ref="bundle:fixed")
    # Both records genuinely exist in this preseeded store. The problem is
    # incompatible comparison samples, not an absent or spoofed record.
    store = FrozenTrustedStore({c.evidence_bundle_ref: _response(c, policy)
                                for c in (baseline, variant)})
    return DecisionProvider(policy, store, store).decide(request(policy, (baseline, variant)))


def test_same_actual_samples_allow_a_truly_superior_variant():
    assert _decision().winner_id == "fixed"


@pytest.mark.parametrize("axis", range(4))
def test_same_plan_label_does_not_allow_other_draw_time_or_mask(axis):
    pairing = ["plan-content", "samples", "times", "mask"]
    pairing[axis] += ":different"
    result = _decision(tuple(pairing))
    assert result.eligibility["fixed"] is True
    assert result.winner_id == "raw"
    assert result.relationship["fixed"].value == "INCONCLUSIVE"


@pytest.mark.parametrize("field", ["data_snapshot_hash", "universe_hash", "label_hash"])
def test_same_context_label_does_not_allow_other_data_universe_or_label(field):
    result = _decision(data_change={field: "other-content"})
    assert result.eligibility["fixed"] is True
    assert result.winner_id == "raw"
