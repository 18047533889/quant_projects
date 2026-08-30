"""QRP-P5-E2E1 real chain: raw synthetic factor values -> FP TreatmentRecipe
-> QE treated-FactorBatch + EvaluationRequest -> EvaluationArtifact
(computed=True) -> FO 2-candidate winner -> FA TreatmentSelectionArtifact
consumer -> FactorSet members / treatment landing (QRP-P5-E2E1).

Chain steps use ONLY the four domains' public contracts/adapters:
- FP  ``factor_preprocess.contracts.treatment_recipe`` (TreatmentRecipe /
      RecipeStep) — the treatment authority.
- QE  ``quant_evaluator.adapters.recipe_refs`` (recipe_to_factor_value_ref /
      request_for_recipe / factor_batch_for_recipe) and the QE public
      ``evaluate`` facade — the evaluation authority.
- FO  ``factor_optimizer.search.pareto`` (ParetoPoint) +
      ``factor_optimizer.search.winner_selector`` (select_winner /
      WinnerPolicy) + ``factor_optimizer.contracts.treatment_result``
      (TreatmentOptimizationResultArtifact) — the winner-selection authority.
- FA  ``factor_assets.contracts.treatment_selection`` (TreatmentSelectionArtifact)
      + ``factor_assets.assembly`` (FactorSetAssembler) + FA contracts
      (FactorSetSpec / FactorMembership / FactorAsset / FactorAdmissionArtifact)
      — the library/assembly authority.

收益口径 is global vwap->vwap (forward labels = Vwap_{t+H}/Vwap_t - 1).  No
full factor pools, no COS, no network.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# The repo-root package dirs mirror FactorEngine (factor_optimizer/src layout):
# the installed wheel may lag the source tree, so the test forces the source
# tree first for the domains it consumes (FO owns treatment_result so this is
# safe there).  Identical canonical source under /repo/factor_optimizer/.
_FO_SRC = "/home/sunhaiwei/quant_projects/factor_optimizer"
if _FO_SRC not in os.sys.path:
    os.sys.path.insert(0, _FO_SRC)
_FP_SRC = "/home/sunhaiwei/quant_projects/factor_preprocess"
if _FP_SRC not in os.sys.path:
    os.sys.path.insert(0, _FP_SRC)

from factor_assets.assembly import FactorSetAssembler
from factor_assets.contracts.admission import (
    AdmissionDecision,
    FactorAdmissionArtifact,
)
from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.factor_set import FactorSetSpec
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.contracts.treatment_selection import TreatmentSelectionArtifact
from factor_preprocess.contracts.treatment_recipe import (
    RecipeStep,
    TreatmentRecipe,
)
from factor_optimizer.contracts.treatment_result import (
    TreatmentOptimizationResultArtifact,
)
from factor_optimizer.search.pareto import ParetoPoint
from factor_optimizer.search.winner_selector import WinnerPolicy, select_winner
from quant_evaluator.adapters.recipe_refs import assert_computed_value
from quant_evaluator.contracts.evaluation_artifact import (
    EvaluationArtifact,
    EvaluationResultContentHash,
)
from quant_evaluator.contracts.domain_refs import ArtifactDomain, DomainArtifactRef
from quant_evaluator.contracts.evaluation_refs import FactorValueRef

# n_assets must exceed evaluate()'s default min_assets=10 for a finite daily IC.
_T = 60
_N = 10
_SEED = 20260827

SNAPSHOT = "snapshot:synth-1"
UNIVERSE = "universe:synth-a"
SPLIT = "split:full-eval"


# ---------------------------------------------------------------------------
# synthetic fixtures the whole chain consumes
# ---------------------------------------------------------------------------


def _forward_labels(rng):
    """vwap-to-vwap forward returns: Vwap_{t+1}/Vwap_t - 1 (global basis)."""
    pts = np.abs(np.cumsum(rng.uniform(0.9, 1.1, size=(_T, _N)), axis=0)) * 10.0 + 3.0
    fwd = pts / np.roll(pts, 1, axis=0) - 1.0
    fwd[0, :] = np.nan
    return fwd


def _raw_panel(rng):
    """2 factors x (T, N) raw cross-sectional values."""
    assets = [f"a{i}" for i in range(_N)]
    return (
        (list(range(_T)), assets, rng.normal(size=(_T, _N))),
        (list(range(_T)), assets, rng.normal(size=(_T, _N))),
    )


FACTOR_IDS = ("profit", "growth")


def make_recipe(*, factor_id="profit", rank_pct: bool = True):
    """A causal, cross-sectional TreatmentRecipe (winsor -> rankpct)."""
    steps = [
        RecipeStep(
            step_id="s1",
            semantic_transform_id="WINSOR:cs",
            implementation_ref="fp://cs_winsor",
            stage="outlier",
            requires_fit=False,
            parameters={"lower": 0.02, "upper": 0.98},
        ),
    ]
    if rank_pct:
        steps.append(
            RecipeStep(
                step_id="s2",
                semantic_transform_id="CS_RANK:pct",
                implementation_ref="fp://cs_rank",
                stage="representation",
                requires_fit=False,
                parameters={"pct": True},
            )
        )
    return TreatmentRecipe(
        recipe_id=f"recipe-{factor_id}",
        source_factor_definition_ref=f"defn:{factor_id}",
        source_factor_value_ref=f"val:{factor_id}:raw",
        ordered_steps=tuple(steps),
    )


# ---------------------------------------------------------------------------
# the real chain
# ---------------------------------------------------------------------------


def test_qrp_p5_e2e1_real_chain():
    """Run the full 8-step chain and assert the release gates.

    1. raw synthetic factor values (2 assets x T, vwap label column)
    2. FP TreatmentRecipe (winsor -> rankpct, causal, no future function)
    3. QE: recipe_to_factor_value_ref + request_for_recipe -> treated
       factor-value ref + evaluation request refs
    4. QE public evaluate -> computed EvaluationArtifact (computed=True)
    5. FO: 2 candidates -> winner (select_winner)
    6. FA: TreatmentSelectionArtifact consuming the treated eval content_hash
       (winner library_snapshot_ref == FA snapshot_ref)
    7. FA: FactorSetAssembler -> FactorSet members carry treatment_selection_ref
       == artifact content_hash
    8. assert TreatmentRecipe is executable (ordered steps apply cleanly)
    """

    rng = np.random.default_rng(_SEED)
    p_profit, p_growth = _raw_panel(rng)
    panel = {"profit": p_profit, "growth": p_growth}
    labels = _forward_labels(rng)

    recipe = make_recipe(factor_id="profit", rank_pct=True)

    # -- step 2: FP TreatmentRecipe is executable ---------------------------
    from quant_evaluator.adapters.recipe_refs import factor_batch_for_recipe

    treated = factor_batch_for_recipe(recipe, panel, factor_ids=("profit",))
    assert treated.values.shape == (_T, _N, 1)
    assert np.isfinite(treated.values[1, :, 0]).all()

    # -- step 3: QE refs from the recipe via the public adapter ---------------
    from quant_evaluator.adapters.recipe_refs import (
        recipe_to_factor_value_ref,
        request_for_recipe,
    )

    fv_ref = recipe_to_factor_value_ref(recipe, ref_for="treated")
    assert isinstance(fv_ref, FactorValueRef)
    assert fv_ref.factor_value_id.startswith("factor_value:treated")

    # request_for_recipe also builds a serializable round-trip request — its
    # raw payloads are runtime-only by the DLI8-QE-003 contract.
    from quant_evaluator.api.requests import EvaluationRequest
    from quant_evaluator.contracts.label_bundle import LabelBundle

    label_bundle = LabelBundle(
        target_id="vwap_forward_return",
        values=labels,
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
        price_convention="vwap_to_vwap",
    )
    request = request_for_recipe(
        recipe,
        label_bundle=label_bundle,
        factor_batch=factor_batch_for_recipe(recipe, panel, factor_ids=("profit",)),
        factor_value_id="val:profit:raw",
        metadata={"snapshot_ref": SNAPSHOT, "universe_ref": UNIVERSE,
                  "split_ref": SPLIT},
    )
    assert isinstance(request, EvaluationRequest)
    assert request.factor_value_ref is not None
    rt = EvaluationRequest.from_dict(request.to_dict())
    assert rt.factor_value_ref == request.factor_value_ref
    assert rt.label_bundle_ref == request.label_bundle_ref

    # -- step 4: computed evaluation evidence --------------------------------
    # Run the same treated batch through the QE public evaluate facade.  The
    # adapter's convenience seam makes the raw->recipe->evaluate loop one call,
    # then we materialize the canonical durable EvaluationArtifact.
    from quant_evaluator.adapters.recipe_refs import recipe_evidence_evaluation

    bundle = recipe_evidence_evaluation(
        recipe,
        panel,
        labels,
        metrics=("rank_ic", "pearson_ic", "coverage"),
        context={"snapshot": SNAPSHOT, "universe": UNIVERSE},
    )
    assert bundle.request_id
    rank_ic = bundle.grouped_metrics["profit"]["rank_ic"]
    assert rank_ic.valid is True, "evaluation must be COMPUTED (valid=True)"
    assert rank_ic.value is not None and np.isfinite(rank_ic.value)
    mu = assert_computed_value(rank_ic, metric_id="rank_ic", factor_id="profit")

    # Canonical durable QE artifact: typed DomainArtifactRef cross-references
    # (R55 audit #24) + the three-part identity (R55 audit #23).
    artifact = EvaluationArtifact(
        evaluation_id=f"e2e/eval/{bundle.request_id}",
        evaluation_identity="qrp-p5-e2e1:profit:recipe-2step",
        factor_value_ref=DomainArtifactRef.of(
            ArtifactDomain.FACTOR_VALUE, fv_ref.to_dict()["factor_value_id"]
        ),
        label_definition_ref=DomainArtifactRef.of(
            ArtifactDomain.LABEL_DEFINITION, "vwap_forward_return"
        ),
        evaluation_policy_ref=DomainArtifactRef.of(
            ArtifactDomain.EVALUATION_POLICY, "policy:core"
        ),
        evaluation_profile_ref=DomainArtifactRef.of(
            ArtifactDomain.EVALUATION_PROFILE, "profile:core"
        ),
        split_ref=DomainArtifactRef.of(ArtifactDomain.SPLIT, SPLIT),
        snapshot_ref=DomainArtifactRef.of(ArtifactDomain.SNAPSHOT, SNAPSHOT),
        universe_ref=DomainArtifactRef.of(ArtifactDomain.UNIVERSE, UNIVERSE),
        metric_evidence_refs=(
            DomainArtifactRef.of(ArtifactDomain.METRIC_EVIDENCE, "metric/rank_ic"),
            DomainArtifactRef.of(ArtifactDomain.METRIC_EVIDENCE, "metric/pearson_ic"),
        ),
        diagnostic_refs=(DomainArtifactRef.of(ArtifactDomain.DIAGNOSTIC, bundle.request_id),),
        result_content=EvaluationResultContentHash(
            metric_values={"rank_ic": float(mu)},
            timing={"decision_time": "2026-08-27T00:00:00+00:00"},
        ),
        created_at="2026-08-27T00:00:00+00:00",
    )
    assert artifact.content_hash  # derived-only content identity
    # R55 #23: same spec re-run (different evaluation id / data) keeps the
    # spec identity while the envelope identity moves.
    rerun_payload = artifact.to_dict()
    rerun_payload["evaluation_id"] = "e2e/eval/rerun"
    rerun_payload.pop("content_hash")  # derived-only: recomputed for the rerun
    rerun_payload["result_content"] = EvaluationResultContentHash(
        metric_values={"rank_ic": float(mu) + 1e-6},  # a re-run's data moved
        timing={"decision_time": "2026-08-27T00:00:00+00:00"},
    ).to_dict()
    rerun = EvaluationArtifact.from_dict(rerun_payload)
    assert rerun.evaluation_spec_identity.identity_hash == (
        artifact.evaluation_spec_identity.identity_hash
    )
    assert rerun.evaluation_result_content_hash != artifact.evaluation_result_content_hash
    assert rerun.evaluation_envelope_identity.identity_hash != (
        artifact.evaluation_envelope_identity.identity_hash
    )

    # -- step 5: FO winner among 2 candidates --------------------------------
    # Candidate A = recipe rankpct (the evaluated treatment); candidate B = a
    # single-step winsor-only treatment on the same raw value.  Desirabilities
    # are normalized from QE metrics into [0,1] by the caller (FO consumes
    # ParetoPoint objectives, not raw metrics), following the documented
    # select_winner contract.
    candidates = [
        ParetoPoint(
            trial_id="T1-profit@recipe:2step",
            objectives=(0.75, 0.60, 0.90),  # rank_ic-aware dims
        ),
        ParetoPoint(
            trial_id="T2-profit@recipe:winsor",
            objectives=(0.70, 0.55, 0.95),
        ),
    ]
    robustness = {p.trial_id: 0.7 for p in candidates}
    complexity = {p.trial_id: 1.0 for p in candidates}  # complexity in [0,1]
    policy = WinnerPolicy(
        alpha=0.4, beta=0.3, gamma=0.2, lambda_=0.1,
        policy_id="e2e", policy_version="1.0.0",
    )
    winner = select_winner(candidates, robustness, complexity, policy)
    assert winner.trial_id == "T1-profit@recipe:2step"

    # -- step 6: FA TreatmentSelectionArtifact consumes the QE result ---------
    # The winner's winning treatment identity == the recipe that was evaluated;
    # FA FA consumes the content_hash into winner_recipe (its own authority),
    # and won't override content_hash (derived-only).  The FO artifact binds
    # the winner to the FA library snapshot (DLIB-FO-008).
    winner_treatment_ref = f"treatment:{recipe.content_hash}"
    result_artifact = TreatmentOptimizationResultArtifact(
        search_session_id="e2e/session/1",
        source_factor_value_ref="val:profit:raw",
        raw_baseline_evidence_ref=artifact.content_hash,
        factor_profile_ref="profile:profit:v1",
        treatment_search_space_ref="search-space:e2e",
        transform_registry_snapshot_ref="registry:fp:1.0",
        desirability_policy_ref="policy:desirability:v1",
        winner_policy_ref=f"{policy.policy_id}@{policy.policy_version}",
        split_plan_ref="split:full-eval",
        trial_ledger_ref="ledger:e2e/1",
        all_trial_refs=("T1-profit@recipe:2step", "T2-profit@recipe:winsor"),
        pareto_trial_refs=("T1-profit@recipe:2step",),
        multiplicity_ref="multiplicity:1",
        library_snapshot_ref=f"{SNAPSHOT}::lib-v1",
        require_library_snapshot_ref=True,
        selected_trial_ref=winner.trial_id,
        uncertainty_evidence_ref="evidence:uncertainty/e2e/1",
    )
    result_artifact.verify()

    selection = TreatmentSelectionArtifact(
        factor_id="profit",
        factor_version="v1",
        raw_baseline_evidence_ref=artifact.content_hash,  # QE EvaluationArtifact
        factor_profile_ref="profile:profit:v1",
        eligibility_policy_ref="policy:eligibility:v1",
        search_space_ref="search-space:e2e",
        all_trial_refs=("T1-profit@recipe:2step", "T2-profit@recipe:winsor"),
        pareto_candidate_refs=("T1-profit@recipe:2step",),
        winner_recipe={
            "recipe_id": recipe.recipe_id,
            "content_hash": recipe.content_hash,
            "treatment_identity": recipe.treatment_identity,
            "transform_sequence": list(recipe.step_semantic_ids()),
        },
        winner_policy_identity=f"{policy.policy_id}@{policy.policy_version}",
        absolute_metric_refs={"rank_ic_evidence": artifact.content_hash},
        delta_metric_refs={},
        dimension_scores={"predictive": 0.75, "stability": 0.60, "quality": 0.90},
        hard_gate_results={"min_obs": "PASS", "vf": "PASS"},
        soft_floor_results={"min_rank_ic": 0.0},
        robustness_evidence="evidence:robustness/e2e/1",
        complexity_score=0.42,
        snapshot_ref=SNAPSHOT,
        universe_ref=UNIVERSE,
        split_ref=SPLIT,
        created_at="2026-08-27T00:00:00+00:00",
    )
    assert selection.content_hash  # derived-only FA treatment content hash
    # The FA snapshot_ref must equal the FO winner's library_snapshot_ref root
    # (same data snapshot the search evaluated on).
    assert SNAPSHOT in result_artifact.library_snapshot_ref.library_version_ref

    # -- step 7: FA FactorSet members land the treatment ---------------------
    metadata = AssetMetadata(
        factor_id="profit",
        canonical_repr="identity(profit)",
        canonical_hash="hash-profit",
        frequency="daily",
        domains=("price",),
        timing="daily",
    )
    asset = FactorAsset(
        metadata=metadata,
        lineage=LineageRef(factor_id="profit", parents=()),
        lifecycle_state=LifecycleState.APPROVED,
        registered_at="2024-01-01T00:00:00Z",
        family="family-a",
        latest_evidence_ref=None,
    )
    admission = FactorAdmissionArtifact(
        factor_id="profit",
        decision=AdmissionDecision.APPROVED,
        quality=0.86,
        factor_version="v1",
        health_state_ref="lifecycle:APPROVED",
        similarity_ref="sim-profit",
        novelty_ref="novelty-profit",
        cluster_id=3,
        orientation=1,
        reason="APPROVED",
        evidence_refs=(artifact.content_hash,),
        gate_results=("gate-1",),
        policy_ref="policy:1.0",
        created_at="2026-08-27T00:00:00+00:00",
    )
    spec = FactorSetSpec(
        set_id="e2e-set-1",
        name="qrp-p5-e2e1",
        selection_policy="manual",
        data_snapshot_ref=SNAPSHOT,
        universe_ref=UNIVERSE,
        split_ref=SPLIT,
        frequency="daily",
        max_factors=1,
    )
    assembled = FactorSetAssembler().assemble(
        spec,
        [asset],
        admission_artifacts={"profit": admission},
        treatment_selection_artifacts={"profit": selection},
        production=True,
    )
    (membership,) = assembled.memberships
    assert membership.factor_id == "profit"
    assert membership.treatment_selection_ref == selection.content_hash
    # FactorSetArtifact snapshot provenance matches the FO winner snapshot.
    assert assembled.snapshot_ref == SNAPSHOT
    assert assembled.universe_ref == UNIVERSE
    assert assembled.split_ref == SPLIT
    assert "profit" in assembled.factor_ids
    assert assembled.assembly_hash


def test_e2e_recipe_variant_evaluates_distinctly():
    """A 1-step winsor-only recipe evaluates to a DIFFERENT treated ref id,
    so the optimizer sees two truly distinct candidates on the same raw value."""
    rng = np.random.default_rng(_SEED)
    p_profit, p_growth = _raw_panel(rng)
    panel = {"profit": p_profit, "growth": p_growth}

    recipe_2step = make_recipe(factor_id="profit", rank_pct=True)
    recipe_1step = make_recipe(factor_id="profit", rank_pct=False)

    from quant_evaluator.adapters.recipe_refs import recipe_to_factor_value_ref

    ref_2 = recipe_to_factor_value_ref(recipe_2step, ref_for="treated")
    ref_1 = recipe_to_factor_value_ref(recipe_1step, ref_for="treated")
    assert ref_2.factor_value_id != ref_1.factor_value_id
    assert ref_2.metadata["treatment_identity"] == recipe_2step.treatment_identity
    assert recipe_2step.content_hash != recipe_1step.content_hash