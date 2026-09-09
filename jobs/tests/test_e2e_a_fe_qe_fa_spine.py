import pandas as pd
import pytest

from jobs.e2e_a_fe_qe_fa_spine import (
    SyntheticSnapshotSource,
    materialized_factor_value_ref,
    run_e2e_a_oos_pit_rejection,
    run_e2e_a_fe_qe_fa_spine,
)
from quant_evaluator.runtime.evaluator import authoritative_array_hash


def test_actual_fe_catalog_snapshot_qe_fa_grading_spine():
    result = run_e2e_a_fe_qe_fa_spine()
    assert result.dsl == "rank(close)"
    assert result.snapshot_ref.startswith("synthetic-snapshot:")
    assert result.catalog_ref.startswith("semantic-catalog:")
    assert "PRICE" in result.taxonomy.data_domains
    assert result.factor_value_ref == materialized_factor_value_ref(
        values=result.factor_batch.values[..., 0],
        time_values=result.factor_batch.time_axis.values,
        asset_values=result.factor_batch.asset_axis.values,
        snapshot_ref=result.snapshot_ref,
        factor_definition_ref=result.factor_definition_ref,
    )
    assert result.evaluation_bundle.factor_ids == result.factor_batch.factor_ids
    provenance = result.evaluation_bundle.metadata["provenance"]
    assert provenance["time_coordinates"]["sha256"] == authoritative_array_hash(
        result.factor_batch.time_axis.values
    )
    assert provenance["asset_coordinates"]["sha256"] == authoritative_array_hash(
        result.factor_batch.asset_axis.values
    )
    grades = {grade.metric_id: grade for grade in result.metric_grades}
    assert grades["rank_ic"].value == 1.0
    assert grades["rank_ic"].production_provenance_complete is True
    assert grades["rank_ic"].factor_definition_id == result.factor_definition_ref
    assert grades["rank_ic"].factor_value_ref == result.factor_value_ref
    assert grades["coverage"].evidence_status == "COMPUTED"
    daily_q10 = result.evaluation_bundle.artifacts["quantile_returns_daily"]
    assert daily_q10.values.shape == (30, 10, 1)
    assert daily_q10.counts.shape == (30, 10, 1)
    assert daily_q10.provenance["config_hash"] == result.evaluation_bundle.config_hash
    assert result.label_fixture_kind == "CONTROLLED_SELF_CORRELATED_SYNTHETIC_NOT_OOS"
    assert result.evaluation_bundle.metadata["label_fixture_kind"] == result.label_fixture_kind


def test_missing_factor_identity_stays_explicitly_unknown_not_fake_complete():
    result = run_e2e_a_fe_qe_fa_spine(include_factor_provenance=False)
    assert result.snapshot_ref in result.factor_batch.context_refs.values()
    assert result.catalog_ref in result.factor_batch.context_refs.values()
    assert result.metric_grades
    assert all(grade.evidence_status == "UNKNOWN" for grade in result.metric_grades)
    assert all(grade.grade is None and grade.desirability is None
               for grade in result.metric_grades)
    assert all(not grade.production_provenance_complete for grade in result.metric_grades)


def test_synthetic_snapshot_source_isolated_and_loads_defensive_copies():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2), ["A", "B"]],
        names=("timestamp", "instrument"),
    )
    original = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    source = SyntheticSnapshotSource.from_data({"close": original})
    snapshot_ref = source.snapshot_ref
    original.iloc[0] = 999.0
    first = source.load_column("close")
    assert first.iloc[0] == 1.0
    first.iloc[0] = -1.0
    assert source.load_column("close").iloc[0] == 1.0
    assert source.snapshot_ref == snapshot_ref
    source._data["close"].iloc[0] = 7.0
    with pytest.raises(ValueError, match="mutated after snapshot"):
        source.load_column("close")


def test_materialization_identity_changes_with_axis_snapshot_or_definition():
    result = run_e2e_a_fe_qe_fa_spine()
    kwargs = dict(
        values=result.factor_batch.values[..., 0],
        time_values=result.factor_batch.time_axis.values,
        asset_values=result.factor_batch.asset_axis.values,
        snapshot_ref=result.snapshot_ref,
        factor_definition_ref=result.factor_definition_ref,
    )
    assert materialized_factor_value_ref(**kwargs) == result.factor_value_ref
    for change in (
        {"asset_values": result.factor_batch.asset_axis.values[::-1]},
        {"snapshot_ref": result.snapshot_ref + ":other"},
        {"factor_definition_ref": result.factor_definition_ref + ":other"},
    ):
        assert materialized_factor_value_ref(**(kwargs | change)) != result.factor_value_ref


def test_high_independent_oos_ic_is_rejected_by_invalid_pit():
    result = run_e2e_a_oos_pit_rejection()
    assert result.frozen_direction == 1
    assert result.training_rank_ic > .95
    assert result.oos_rank_ic > .95
    assert result.label_generation_rule.startswith("forward_return=0.02*asset_latent")
    assert result.oos_labels_ref.startswith("labels:")
    assert result.catalog_ref.startswith("semantic-catalog:")
    assert result.evaluation_bundle.metadata["provenance"]["context_refs"][
        "catalog_ref"
    ] == result.catalog_ref
    q10 = result.evaluation_bundle.artifacts["quantile_returns_daily"]
    assert q10.values.shape == (30, 10, 1)

    pit_gate = result.health_card.gate("pit_valid")
    assert pit_gate.passed is False
    assert pit_gate.evidence_ref == result.pit_validation.evidence_ref
    assert result.pit_validation.observed_knowledge_time == "none"
    assert result.health_card.admission_relevant.hard_gates_passed is False
    assert result.health_card.admission_relevant.admissible is False
    assert result.health_card.display_grade_is_display_only is True
    assert result.admission_decision.decision == "REJECT"
    assert result.admission_decision.reason_codes == ("pit_invalid",)
    assert result.admission_decision.health_card_ref.startswith("health-card-content:")
    assert result.promotion_attempted is False
    assert result.promotion_decision is None


def test_rejection_trace_records_real_refs_and_explicit_missing_stages():
    result = run_e2e_a_oos_pit_rejection()
    trace = result.trace
    assert trace.request_id == trace.evaluation_ref
    assert trace.factor_definition_ref == result.health_card.factor_definition_id
    assert trace.evaluation_ref == result.health_card.evaluation_ref
    assert trace.verdict_ref == (
        f"health-admission-decision:{result.admission_decision.content_hash}"
    )
    assert trace.health_policy_ref == (
        f"{result.health_card.health_policy_id}:{result.health_card.health_policy_version}"
    )
    assert trace.library_version_ref.startswith("library-version:")
    assert trace.parent_trial_ref is None
    assert trace.recipe_ref is None
    assert trace.feature_version_ref is None
    assert trace.failure_stage == "PIT_INTEGRITY_GATE"
