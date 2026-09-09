from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd

from factor_assets.selection import CandidateEvidence, DecisionProvider, RawJointMetricEvidence
from factor_optimizer.ports.factor_intelligence import require_bound_decision_receipt
from factor_preprocess.adapters.fitted_recipe import (
    FITTED_STANDARDIZE_NAME,
    FITTED_STANDARDIZE_VERSION,
    apply_frozen_fitted_recipe,
    fitted_standardize_implementation_hash,
)
from factor_preprocess.contracts.state import FittedState, StateKind
from factor_preprocess.contracts.treatment_recipe import RecipeStep, TreatmentRecipe
from quant_evaluator.contracts.resampling import ResamplingPlan
from quant_evaluator.metrics.robustness import compute_joint_block_bootstrap
from tests.search.test_v8_qe_fa_fo_integration import (
    _Resolver,
    _digest,
    _policy,
    _qualification,
    _request,
    _requirements,
)


def _candidates_from_qe_runs(context, qualification, plan, named_runs):
    """Canonical-align production QE outputs from arbitrary factor shards."""
    by_id = {}
    pairing_by_id = {}
    for values, candidate_ids in named_runs:
        artifact = compute_joint_block_bootstrap(values, plan, candidate_ids)
        assert artifact.provenance["resampling_plan_ref"] == plan.content_hash
        assert artifact.provenance["replicate_ids"] == plan.replicate_ids
        pairing = tuple(artifact.provenance[name] for name in (
            "resampling_plan_content_hash", "sample_identity_hash",
            "time_identity_hash", "common_mask_hash"))
        for column, candidate_id in enumerate(candidate_ids):
            by_id[candidate_id] = tuple(float(v) for v in artifact.samples[:, column])
            pairing_by_id[candidate_id] = pairing

    candidates = []
    for candidate_id in sorted(by_id):
        values = by_id[candidate_id]
        pairing = pairing_by_id[candidate_id]
        samples = tuple((((value,),),) for value in values)
        identity=(candidate_id,"recipe","state","data","universe","label","values:"+candidate_id)
        semantic = {
            "context": context,
            "plan": plan.content_hash,
            "pairing": pairing,
            "replicates": plan.replicate_ids,
            "metrics": ("alpha",),
            "metric_units": ("ratio",),
            "window_ids": ("window",),
            "scenario_ids": ("scenario",),
            "samples": json.loads(json.dumps(samples)),
            "qualification": "scope:v8",
            "execution": ("source", "impl", "qe.runtime", "cpu", "domain", "metric-instance"),
            "candidate": identity,
        }
        evidence = RawJointMetricEvidence(
            f"qe:{candidate_id}", _digest(semantic), context, plan.content_hash,
            plan.replicate_ids, ("alpha",), samples, "scope:v8",
            source_tree_hash="source", implementation_hash="impl", route="qe.runtime",
            backend="cpu", parameter_domain_hash="domain",
            metric_instance_hash="metric-instance", metric_units=("ratio",),
            candidate_id=candidate_id,recipe_hash="recipe",fitted_state_hash="state",data_snapshot_hash="data",
            universe_hash="universe",label_hash="label",value_artifact_hash="values:"+candidate_id,
            resampling_plan_content_hash=pairing[0], sample_identity_hash=pairing[1],
            time_identity_hash=pairing[2], common_mask_hash=pairing[3],
        )
        qe_ref = _digest({"candidate": candidate_id, "plan": plan.content_hash, "samples": values})
        candidates.append(CandidateEvidence(
            candidate_id, (qe_ref, evidence.content_hash, qualification.content_hash),
            "FULL_VALIDATION", {}, {"alpha": float(np.mean(values))}, 1.0,
            "scope:v8", qualification_ref=qualification.content_hash,
            qualification_requirements=_requirements(), raw_joint_metric_evidence=evidence,
            health_evidence_ref="health:"+candidate_id,health_candidate_id=candidate_id,health_coverage=1.0,evidence_bundle_ref=qe_ref,
        ))
    return tuple(candidates)


def _fo_decision(provider, policy, candidates, context):
    request = _request(
        policy, candidates, request_id="canonical-shards", baseline="RAW",
        original_raw="RAW", context=context,
    )
    return require_bound_decision_receipt(request, provider.decide(request))


def test_t26_qe_candidate_permutation_single_column_and_factor_shards_align_to_one_decision():
    context = "t26:frozen-context"
    qualification = _qualification()
    policy = _policy()
    provider = DecisionProvider(policy, _Resolver(qualification))
    plan = ResamplingPlan(tuple(range(48)), "synthetic-test-clock", 4, 24, 26027)
    matrix = np.column_stack((
        np.linspace(.30, .60, 48),
        np.linspace(.42, .78, 48),
        np.linspace(.36, .69, 48),
    ))

    permuted = _candidates_from_qe_runs(
        context, qualification, plan,
        ((matrix[:, [2, 0, 1]], ("ALT", "RAW", "VAR")),),
    )
    single_columns = _candidates_from_qe_runs(
        context, qualification, plan,
        tuple((matrix[:, [i]], (name,)) for i, name in enumerate(("RAW", "VAR", "ALT"))),
    )
    factor_shards = _candidates_from_qe_runs(
        context, qualification, plan,
        ((matrix[:, :2], ("RAW", "VAR")), (matrix[:, 2:], ("ALT",))),
    )

    receipts = tuple(
        _fo_decision(provider, policy, candidates, context)
        for candidates in (permuted, single_columns, factor_shards)
    )
    assert len({receipt.candidate_set_hash for receipt in receipts}) == 1
    assert len({receipt.decision_id for receipt in receipts}) == 1
    assert len({receipt.content_hash for receipt in receipts}) == 1


def _frozen_state_and_recipe():
    definition = hashlib.sha256(b"t27-factor-definition").hexdigest()
    state = FittedState(
        transform_name=FITTED_STANDARDIZE_NAME,
        transform_version=FITTED_STANDARDIZE_VERSION,
        fit_start_time=datetime(2023, 1, 1), fit_end_time=datetime(2023, 12, 31),
        state_kind=StateKind.FITTED, feature_ids=[definition], feature_order=[definition],
        learned_params={"mean": 0.25, "scale": 1.75},
        implementation_hash=fitted_standardize_implementation_hash(),
        data_snapshot_ref=_digest("train-snapshot"), split_ref=_digest("train-split"),
        universe_ref=_digest("universe"), calendar_ref=_digest("calendar"),
        fit_coordinate_hash=_digest("fit-coordinates"), policy_hash=_digest("fit-policy"),
        production=True,
    )
    recipe = TreatmentRecipe(
        recipe_id="t27-frozen-standardize", source_factor_definition_ref=definition,
        source_factor_value_ref=_digest("source-values"),
        ordered_steps=(RecipeStep(
            "standardize", "TRAIN_STANDARDIZE:population-v1", FITTED_STANDARDIZE_NAME,
            "representation", requires_fit=True, state_ref=state.state_id,
        ),),
    )
    return definition, state, recipe


def test_t27_future_tail_cannot_change_causal_fp_qe_evidence_or_fo_decision():
    context = "t27:frozen-fit-and-comparison-context"
    qualification = _qualification()
    policy = _policy()
    provider = DecisionProvider(policy, _Resolver(qualification))
    definition, state, recipe = _frozen_state_and_recipe()
    prefix = np.linspace(-1.0, 1.0, 48)
    original = np.concatenate((prefix, np.linspace(1.1, 2.0, 12)))
    perturbed = np.concatenate((prefix, np.linspace(-1000.0, 1000.0, 12)))

    def causal_prefix(values):
        frame = pd.DataFrame(values, index=pd.date_range("2024-01-02", periods=len(values)))
        transformed, state_refs = apply_frozen_fitted_recipe(
            frame, recipe=recipe, fitted_states={state.state_id: state},
            factor_definition_hash=definition, decision_start="2024-01-02",
        )
        assert state_refs == (state.state_id,)
        return transformed.iloc[:48, 0].to_numpy()

    transformed_a = causal_prefix(original)
    transformed_b = causal_prefix(perturbed)
    np.testing.assert_array_equal(transformed_a, transformed_b)

    plan = ResamplingPlan(tuple(range(48)), "synthetic-test-clock", 4, 24, 27027)
    raw = prefix
    candidates_a = _candidates_from_qe_runs(
        context, qualification, plan,
        ((np.column_stack((raw, transformed_a)), ("RAW", "VAR")),),
    )
    candidates_b = _candidates_from_qe_runs(
        context, qualification, plan,
        ((np.column_stack((raw, transformed_b)), ("RAW", "VAR")),),
    )
    receipt_a = _fo_decision(provider, policy, candidates_a, context)
    receipt_b = _fo_decision(provider, policy, candidates_b, context)
    assert candidates_a == candidates_b
    assert receipt_a.decision_id == receipt_b.decision_id
    assert receipt_a.content_hash == receipt_b.content_hash
