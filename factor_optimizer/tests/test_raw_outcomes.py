"""R61-FI-033 / plan §22: RAW treatment-outcome tests.

Covers (matrix E4):
1. All 8 canonical outcomes are enumerated and the enum vocabulary is closed.
2. RAW-win outcomes return the RAW factor body (winner_factor_ref ==
   raw_factor_ref) — never ``None``.
3. The full §22 ref-field set is present on the extended artifact.
4. Outcome/ref iron rules fail closed (RAW win without RAW refs, IMPROVED with
   a RAW winner ref, FAILED_RAW_INVALID with RAW refs, etc.).
5. Legacy (pre-§22) artifact construction and round-trips are unchanged.
"""

import dataclasses

import pytest

from factor_optimizer.contracts.library_snapshot_ref import LibrarySnapshotRef
from factor_optimizer.contracts.treatment_result import (
    OUTCOME_IS_RAW_WIN,
    OUTCOME_IS_SUCCESS,
    RAW_OUTCOME_VALUES,
    RawTreatmentOutcome,
    TreatmentOptimizationResultArtifact,
    treatment_result_from_selection,
    validate_outcome_refs,
)

#: The outcome enum carries exactly these 8 values.
EXPECTED_OUTCOMES = {
    "IMPROVED",
    "RAW_SELECTED_NO_IMPROVEMENT",
    "RAW_SELECTED_NEAR_EQUIVALENT",
    "RAW_SELECTED_CANDIDATES_FAILED",
    "NO_ELIGIBLE_REPAIR",
    "SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED",
    "FAILED_RAW_VALID",
    "FAILED_RAW_INVALID",
}

#: RAW-win outcomes: RAW is the returned production candidate.
RAW_WINS = {
    "RAW_SELECTED_NO_IMPROVEMENT",
    "RAW_SELECTED_NEAR_EQUIVALENT",
    "RAW_SELECTED_CANDIDATES_FAILED",
}

#: Runs that end with a winner candidate.
SUCCESS = {
    "IMPROVED",
    "RAW_SELECTED_NO_IMPROVEMENT",
    "RAW_SELECTED_NEAR_EQUIVALENT",
    "RAW_SELECTED_CANDIDATES_FAILED",
}


def _base(**overrides):
    """Common construction kwargs for a result artifact."""
    defaults = dict(
        search_session_id="s1",
        source_factor_value_ref="sfv-1",
        raw_baseline_evidence_ref="raw-ev-1",
        factor_profile_ref="fp-1",
        treatment_search_space_ref="tss-1",
        transform_registry_snapshot_ref="trs-1",
        desirability_policy_ref="dp-1",
        winner_policy_ref="wp-1",
        split_plan_ref="sp-1",
        trial_ledger_ref="tl-1",
        all_trial_refs=("RAW", "t1"),
        pareto_trial_refs=("RAW", "t1"),
        multiplicity_ref="mult-1",
        selected_trial_ref="t1",
        uncertainty_evidence_ref="ue-1",
    )
    defaults.update(overrides)
    return defaults


def _selection(outcome="IMPROVED", **overrides):
    kwargs = dict(
        search_session_id="s1",
        source_factor_value_ref="sfv-1",
        raw_baseline_evidence_ref="raw-ev-1",
        factor_profile_ref="fp-1",
        treatment_search_space_ref="tss-1",
        transform_registry_snapshot_ref="trs-1",
        desirability_policy_ref="dp-1",
        winner_policy_ref="wp-1",
        split_plan_ref="sp-1",
        trial_ledger_ref="tl-1",
        all_trial_refs=("RAW", "t1"),
        pareto_trial_refs=("RAW", "t1"),
        multiplicity_ref="mult-1",
        outcome=outcome,
        raw_factor_ref="F_RAW",
        raw_evaluation_ref="ev-raw",
        raw_health_ref="hc-raw",
        winner_factor_ref="F_RAW" if outcome in RAW_WINS else "F_T1",
        winner_evaluation_ref="ev-raw" if outcome in RAW_WINS else "ev-t1",
        winner_health_ref="hc-raw" if outcome in RAW_WINS else "hc-t1",
        candidate_trial_refs=("t1",),
        pareto_refs=("RAW", "t1"),
        multiplicity_family_ref="fam-1",
        selection_reason=(
            "no improvement over RAW" if outcome in RAW_WINS
            else "treatment beats RAW"
        ),
        uncertainty_evidence_ref="ue-1",
    )
    kwargs.update(overrides)
    return treatment_result_from_selection(**kwargs)


# ---------------------------------------------------------------------------
# 1. 8 canonical outcomes, closed enum
# ---------------------------------------------------------------------------


def test_all_8_outcomes_enumerated():
    assert RawTreatmentOutcome.__members__.keys() == EXPECTED_OUTCOMES
    assert tuple(o.value for o in RawTreatmentOutcome) == RAW_OUTCOME_VALUES


def test_outcome_from_value_fails_closed():
    assert RawTreatmentOutcome.from_value("IMPROVED") is RawTreatmentOutcome.IMPROVED
    assert (
        RawTreatmentOutcome.from_value(RawTreatmentOutcome.FAILED_RAW_INVALID)
        is RawTreatmentOutcome.FAILED_RAW_INVALID
    )
    with pytest.raises(ValueError, match="Unknown RawTreatmentOutcome"):
        RawTreatmentOutcome.from_value("NOT_A_STATUS")


def test_outcome_vocabulary_has_no_extra_members():
    for member in RawTreatmentOutcome:
        assert member.value in EXPECTED_OUTCOMES


# ---------------------------------------------------------------------------
# 2. RAW-win outcomes return the RAW factor body — never None
# ---------------------------------------------------------------------------


def test_raw_win_returns_raw_factor_body_not_none():
    """For every RAW-win outcome, winner_factor_ref == raw_factor_ref."""
    for token in sorted(RAW_WINS):
        result = _selection(outcome=token)
        assert result.raw_won is True, token
        assert OUTCOME_IS_RAW_WIN(token) is True
        assert result.winner_factor_ref == result.raw_factor_ref, token
        assert result.winner_factor_ref == "F_RAW", token
        assert result.winner_evaluation_ref == result.raw_evaluation_ref, token
        assert result.winner_health_ref == result.raw_health_ref, token
        # The produced candidate is not None and is the RAW body.
        assert result.produced_winner is True
        assert result.raw_factor_ref


def test_raw_win_rejects_missing_raw_factor_ref():
    for token in sorted(RAW_WINS):
        with pytest.raises(ValueError, match="RAW factor body|raw_factor_ref"):
            _selection(outcome=token, raw_factor_ref="")


def test_raw_win_rejects_divergent_winner_ref():
    # Caller claims a RAW win but asks for a different winner factor: the
    # artifact must fail closed (RAW is the only legal production candidate).
    with pytest.raises(ValueError, match="must equal raw_factor_ref"):
        _selection(
            outcome="RAW_SELECTED_NO_IMPROVEMENT",
            winner_factor_ref="F_OTHER",
        )


def test_raw_win_defaults_winner_refs_from_raw_refs():
    # The builder fills winner refs from the raw refs when not supplied.
    result = _selection(
        outcome="RAW_SELECTED_NEAR_EQUIVALENT",
        winner_factor_ref="",
        winner_evaluation_ref="",
        winner_health_ref="",
    )
    assert result.winner_factor_ref == result.raw_factor_ref == "F_RAW"
    assert result.winner_evaluation_ref == "ev-raw"
    assert result.winner_health_ref == "hc-raw"
    assert result.selected_trial_ref == "RAW"


def test_raw_win_serialization_roundtrip_preserves_raw_body():
    for token in sorted(RAW_WINS):
        result = _selection(outcome=token)
        restored = TreatmentOptimizationResultArtifact.from_dict(result.to_dict())
        assert restored.content_hash == result.content_hash
        assert restored.raw_won is True
        assert restored.winner_factor_ref == restored.raw_factor_ref


# ---------------------------------------------------------------------------
# 3. Full §22 ref field set on the extended artifact
# ---------------------------------------------------------------------------

#: Every §22 ref field the FA treatment-selection artifact consumes.
SECTION_22_FIELDS = {
    "raw_factor_ref",
    "raw_evaluation_ref",
    "raw_health_ref",
    "winner_factor_ref",
    "winner_evaluation_ref",
    "winner_health_ref",
    "candidate_trial_refs",
    "pareto_refs",
    "multiplicity_family_ref",
    "selection_reason",
    "failure_summary",
    "raw_outcome",
    "per_field_refs",
}


def test_extended_artifact_has_all_section22_ref_fields():
    names = {f.name for f in dataclasses.fields(TreatmentOptimizationResultArtifact)}
    assert SECTION_22_FIELDS <= names
    result = _selection(outcome="IMPROVED")
    for field_name in SECTION_22_FIELDS:
        assert hasattr(result, field_name), field_name


def test_improved_carries_all_refs():
    result = _selection(outcome="IMPROVED")
    assert result.raw_outcome is RawTreatmentOutcome.IMPROVED
    assert result.raw_factor_ref == "F_RAW"
    assert result.raw_evaluation_ref == "ev-raw"
    assert result.raw_health_ref == "hc-raw"
    assert result.winner_factor_ref == "F_T1"
    assert result.winner_evaluation_ref == "ev-t1"
    assert result.winner_health_ref == "hc-t1"
    assert result.candidate_trial_refs == ("t1",)
    assert result.pareto_refs == ("RAW", "t1")
    assert result.multiplicity_family_ref == "fam-1"
    assert result.selection_reason == "treatment beats RAW"
    assert result.failure_summary == ""
    assert result.produced_winner is True
    assert result.raw_won is False


def test_improved_rejects_winner_equaling_raw():
    with pytest.raises(ValueError, match="must be declared as a RAW_SELECTED"):
        _selection(outcome="IMPROVED", winner_factor_ref="F_RAW")


def test_improved_requires_raw_refs():
    with pytest.raises(ValueError, match="raw_factor_ref"):
        _selection(outcome="IMPROVED", raw_factor_ref="")


def test_failed_raw_valid_keeps_raw_but_no_winner():
    result = _selection(
        outcome="FAILED_RAW_VALID",
        winner_factor_ref="",
        winner_evaluation_ref="",
        winner_health_ref="",
        failure_summary="search crashed",
    )
    assert result.raw_factor_ref == "F_RAW"
    assert result.produced_winner is False
    assert result.raw_won is False
    assert result.failure_summary == "search crashed"


def test_failed_raw_invalid_has_no_raw_refs():
    result = _selection(
        outcome="FAILED_RAW_INVALID",
        raw_factor_ref="",
        raw_evaluation_ref="",
        raw_health_ref="",
        winner_factor_ref="",
        winner_evaluation_ref="",
        winner_health_ref="",
        failure_summary="RAW is invalid",
        all_trial_refs=("t1",),
        pareto_trial_refs=(),
        selected_trial_ref="t1",
    )
    assert result.raw_factor_ref == ""
    assert result.produced_winner is False
    assert result.raw_won is False
    assert result.failure_summary == "RAW is invalid"


def test_no_eligible_repair_keeps_raw_baseline():
    result = _selection(
        outcome="NO_ELIGIBLE_REPAIR",
        winner_factor_ref="",
        winner_evaluation_ref="",
        winner_health_ref="",
    )
    assert result.raw_factor_ref == "F_RAW"
    assert result.raw_evaluation_ref == "ev-raw"
    assert result.produced_winner is False


# ---------------------------------------------------------------------------
# 4. Iron-rule validation fails closed
# ---------------------------------------------------------------------------


def test_validate_outcome_refs_raw_win_requires_all_raw_refs():
    for missing in ("raw_factor_ref", "raw_evaluation_ref", "raw_health_ref"):
        kwargs = {
            "raw_factor_ref": "F",
            "raw_evaluation_ref": "E",
            "raw_health_ref": "H",
        }
        kwargs[missing] = ""
        with pytest.raises(ValueError, match=missing):
            validate_outcome_refs("RAW_SELECTED_NO_IMPROVEMENT", **kwargs)


def test_validate_outcome_refs_failed_raw_invalid_rejects_raw():
    with pytest.raises(ValueError, match="FAILED_RAW_INVALID requires"):
        validate_outcome_refs(
            "FAILED_RAW_INVALID",
            raw_factor_ref="F",
            raw_evaluation_ref="",
            raw_health_ref="",
        )


def test_validate_outcome_refs_failed_raw_valid_keeps_raw():
    validate_outcome_refs(
        "FAILED_RAW_VALID",
        raw_factor_ref="F",
        raw_evaluation_ref="",
        raw_health_ref="",
    )
    with pytest.raises(ValueError, match="raw_factor_ref"):
        validate_outcome_refs(
            "FAILED_RAW_VALID",
            raw_factor_ref="",
            raw_evaluation_ref="",
            raw_health_ref="",
        )


# ---------------------------------------------------------------------------
# 5. Legacy construction is unchanged
# ---------------------------------------------------------------------------


def test_legacy_construction_still_works():
    """Pre-§22 callers (existing tests) construct without the new fields."""
    artifact = TreatmentOptimizationResultArtifact(**_base())
    assert artifact.raw_outcome is RawTreatmentOutcome.IMPROVED
    assert artifact.raw_factor_ref == ""
    assert artifact.winner_factor_ref == ""
    assert artifact.selected_trial_ref == "t1"
    assert artifact.content_hash
    artifact.verify()


def test_legacy_roundtrip_and_tamper_detection():
    artifact = TreatmentOptimizationResultArtifact(**_base())
    restored = TreatmentOptimizationResultArtifact.from_dict(artifact.to_dict())
    assert restored.content_hash == artifact.content_hash
    data = artifact.to_dict()
    data["selected_trial_ref"] = "RAW"
    with pytest.raises(ValueError, match="content_hash"):
        TreatmentOptimizationResultArtifact.from_dict(data)


def test_new_fields_are_in_content_hash():
    """A tampered outcome or winner ref must be detected by the hash.

    The tampered outcome is chosen so the ref validation still passes (the
    IMPROVED result already carries all four refs), letting the tamper reach
    the content-hash check.
    """
    result = _selection(outcome="IMPROVED")
    data = result.to_dict()
    data["raw_outcome"] = "FAILED_RAW_VALID"
    with pytest.raises(ValueError, match="content_hash"):
        TreatmentOptimizationResultArtifact.from_dict(data)
    data = result.to_dict()
    data["raw_outcome"] = "NO_ELIGIBLE_REPAIR"
    with pytest.raises(ValueError, match="content_hash"):
        TreatmentOptimizationResultArtifact.from_dict(data)
    data = result.to_dict()
    data["winner_factor_ref"] = "F_TAMPERED"
    with pytest.raises(ValueError, match="content_hash"):
        TreatmentOptimizationResultArtifact.from_dict(data)


def test_success_and_rawwin_predicates():
    for token in EXPECTED_OUTCOMES:
        if token in RAW_WINS:
            assert OUTCOME_IS_RAW_WIN(token) is True
            assert OUTCOME_IS_SUCCESS(token) is True
        elif token in SUCCESS:
            assert OUTCOME_IS_RAW_WIN(token) is False
            assert OUTCOME_IS_SUCCESS(token) is True
        else:
            assert OUTCOME_IS_RAW_WIN(token) is False
            assert OUTCOME_IS_SUCCESS(token) is False


def test_library_snapshot_and_new_fields_coexist():
    result = _selection(
        outcome="RAW_SELECTED_CANDIDATES_FAILED",
        library_snapshot_ref="lib-v1",
        require_library_snapshot_ref=True,
    )
    assert isinstance(result.library_snapshot_ref, LibrarySnapshotRef)
    assert result.raw_won is True
    assert result.winner_factor_ref == result.raw_factor_ref == "F_RAW"
