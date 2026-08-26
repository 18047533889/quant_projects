"""
Tests for treatment lineage contracts.

Key semantics:
- rank@pre_neutralization vs rank@post_neutralization are DIFFERENT semantics.
- Deduplication keys on semantic stage, not function name.
- already_industry_neutral prunes duplicate neutralization.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from factor_preprocess.contracts.treatment_lineage import (
    TransformStage,
    TransformSemanticID,
    TransformStep,
    TransformLineage,
    ExistingTreatmentSignature,
    map_fe_dsl_to_semantic,
)


def _rank_step(stage: TransformStage, name: str = "cs_rank") -> TransformStep:
    return TransformStep(
        semantic_id=TransformSemanticID("CS_RANK:pct"),
        stage=stage,
        name=name,
        parameters={"pct": True},
    )


def test_rank_pre_vs_post_neutralization_are_different_semantics():
    pre = _rank_step(TransformStage.PRE_NEUTRALIZATION)
    post = _rank_step(TransformStage.POST_NEUTRALIZATION)
    # Same function name, same semantic ID string, but different stages.
    assert pre.name == post.name
    assert pre.semantic_id == post.semantic_id
    # They must be treated as distinct treatments.
    lineage = TransformLineage((pre, post))
    # Dedupe by semantic stage: both are kept because stages differ.
    deduped = lineage.dedupe()
    assert len(deduped) == 2
    assert deduped.steps[0].stage == TransformStage.PRE_NEUTRALIZATION
    assert deduped.steps[1].stage == TransformStage.POST_NEUTRALIZATION


def test_dedupe_by_semantic_stage_not_function_name():
    # Two steps with the same stage but different function names are the same
    # treatment and must dedupe to one.
    a = _rank_step(TransformStage.POST_NEUTRALIZATION, name="cs_rank")
    b = _rank_step(TransformStage.POST_NEUTRALIZATION, name="rank")
    lineage = TransformLineage((a, b))
    deduped = lineage.dedupe()
    assert len(deduped) == 1
    assert deduped.steps[0].name == "cs_rank"  # first wins


def test_dedupe_keeps_distinct_semantic_ids():
    winsor = TransformStep(
        semantic_id=TransformSemanticID("WINSOR:q01_q99"),
        stage=TransformStage.OUTLIER,
        name="cs_winsor",
    )
    rank = _rank_step(TransformStage.POST_NEUTRALIZATION)
    lineage = TransformLineage((winsor, rank, winsor))
    deduped = lineage.dedupe()
    assert len(deduped) == 2


def test_already_industry_neutral_prunes_duplicate():
    sig = ExistingTreatmentSignature(industry_neutral=True, industry_schema="SW_L1")
    assert sig.industry_neutral is True
    assert sig.industry_schema == "SW_L1"
    # The eligibility engine must not re-add industry neutralization.
    from factor_preprocess.eligibility.engine import TreatmentEligibilityEngine
    from factor_preprocess.contracts.factor_profile import FactorProfileArtifact

    profile = FactorProfileArtifact(
        factor_id="f1",
        factor_version="1.0.0",
        semantic_family="PRICE_VOLUME",
        source_type="price",
        update_frequency="daily",
        natural_horizon=20,
    )
    engine = TreatmentEligibilityEngine()
    space = engine.build_search_space(profile, existing=sig)
    assert "industry_neutral" not in space.allowed_transform_ids
    assert "dual_neutral" not in space.allowed_transform_ids


def test_fe_dsl_mapping_known_and_unknown():
    assert map_fe_dsl_to_semantic("cs_winsor") == "WINSOR:cs"
    assert map_fe_dsl_to_semantic("cs_rank") == "CS_RANK:pct"
    assert map_fe_dsl_to_semantic("ewma") == "SMOOTH:ewma"
    assert map_fe_dsl_to_semantic("ols_neutralize") == "NEUTRAL:ols"
    # Unknown names map to None — the optimizer never guesses.
    assert map_fe_dsl_to_semantic("totally_unknown_transform") is None
