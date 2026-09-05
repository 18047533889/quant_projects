"""Tests for the versioned repair-family registry (R61-FI-034, plan §20 E3)."""

import pytest

from factor_optimizer.policy.repair import DiagnosisKind, PLAN_E2_DIAGNOSIS_NAMES
from factor_optimizer.policy.repair_registry import (
    CausalityClass,
    ExecutionDomain,
    ParameterPrior,
    ParameterSchema,
    PLAN_E3_FAMILY_NAMES,
    RepairCandidateSlot,
    RepairFamily,
    RepairFamilyDeclaration,
    RepairFamilyRegistry,
)


def _required_fields() -> dict:
    return dict(
        family=RepairFamily.CAUSAL_SMOOTHING,
        eligible_diagnoses=frozenset({"HIGH_TURNOVER", "HIGH_VARIANCE"}),
        owner=ExecutionDomain.FP,
        parameter_schema=ParameterSchema(
            {
                "method": "choice:EWMA|KAMA",
                "natural_time_scale_relative": "float:0.1:2.0",
            }
        ),
        parameter_prior=ParameterPrior({"method": "EWMA"}),
        maximum_candidates=3,
        causality_class=CausalityClass.CAUSAL,
        complexity_cost=2,
        required_evidence_profile="FULL_VALIDATION_CN_1D",
    )


# ---------------------------------------------------------------------------
# 1. RepairFamily vocabulary
# ---------------------------------------------------------------------------


def test_repair_family_vocabulary_20_members():
    assert len(list(RepairFamily)) == 20
    assert RepairFamily.names() == PLAN_E3_FAMILY_NAMES


def test_plan_e3_family_names_are_upper_snake():
    for name in PLAN_E3_FAMILY_NAMES:
        assert name == name.upper()
        assert " " not in name


def test_repair_family_declaration_requires_known_diagnosis():
    with pytest.raises(ValueError, match="unknown eligible diagnosis"):
        RepairFamilyDeclaration(
            family=RepairFamily.ABANDON,
            eligible_diagnoses=frozenset({"NOT_A_DIAGNOSIS"}),
            owner=ExecutionDomain.FE,
        )


def test_repair_family_declaration_validates_owner_and_causality():
    with pytest.raises(ValueError, match="ExecutionDomain"):
        RepairFamilyDeclaration(
            family=RepairFamily.ABANDON,
            eligible_diagnoses=frozenset({"INTEGRITY_FAILURE"}),
            owner="FE",  # raw string rejected (enum required)
        )
    with pytest.raises(ValueError, match="CausalityClass"):
        RepairFamilyDeclaration(
            family=RepairFamily.ABANDON,
            eligible_diagnoses=frozenset({"INTEGRITY_FAILURE"}),
            owner=ExecutionDomain.FE,
            causality_class="causal",
        )


def test_parameter_schema_validation_fail_closed():
    with pytest.raises(ValueError, match="choice spec"):
        ParameterSchema({"method": "choice:"})
    with pytest.raises(ValueError, match="float spec"):
        ParameterSchema({"x": "float:1:1"})
    with pytest.raises(ValueError, match="unknown parameter spec"):
        ParameterSchema({"x": "whatever"})


def test_parameter_prior_must_be_jsonable_mapping():
    with pytest.raises(TypeError):
        ParameterPrior("not-a-mapping")
    assert ParameterPrior({"a": 1}).value_of("a") == 1
    assert ParameterPrior({"a": 1}).value_of("missing") is None


# ---------------------------------------------------------------------------
# 2. Canonical 20-family registry
# ---------------------------------------------------------------------------


def test_default_registry_has_20_families_with_all_declared_fields():
    reg = RepairFamilyRegistry.default()
    assert len(reg) == 20
    for name in PLAN_E3_FAMILY_NAMES:
        declaration = reg.get(name)
        assert declaration.family_name == name
        # every declaration declares the plan §20 E3 fields
        assert isinstance(declaration.eligible_diagnoses, frozenset)
        assert declaration.eligible_diagnoses, name
        assert isinstance(declaration.owner, ExecutionDomain)
        assert isinstance(declaration.parameter_schema, ParameterSchema)
        assert isinstance(declaration.parameter_prior, ParameterPrior)
        assert declaration.maximum_candidates >= 1
        assert isinstance(declaration.causality_class, CausalityClass)
        assert declaration.complexity_cost >= 0
        assert isinstance(declaration.required_evidence_profile, str)
        assert declaration.required_evidence_profile
        assert declaration.policy_id == "FO_REPAIR_FAMILY"
        assert declaration.policy_version == "1.0.0"


def test_default_registry_policy_id_versioned():
    reg = RepairFamilyRegistry.default()
    assert reg.policy_id == "FO_REPAIR_FAMILY"
    assert reg.policy_version == "1.0.0"


def test_default_registry_evidence_profiles_are_reference_strings():
    reg = RepairFamilyRegistry.default()
    for declaration in reg.declarations:
        # Plan §20 E3: required QE evidence profile referenced as a string id;
        # FO never imports QE.  All must be upper-case token ids.
        profile = declaration.required_evidence_profile
        assert isinstance(profile, str)
        assert profile.isupper(), profile
        assert profile in {
            "CHEAP_SCREEN_CN_1D",
            "SHAPE_DIAGNOSTIC_CN_1D",
            "FULL_VALIDATION_CN_1D",
            "EXPENSIVE_STATISTICAL_CN_1D",
            "MODEL_FEATURE_DIAGNOSTIC_CN_1D",
        }


def test_get_family_by_member_or_name():
    reg = RepairFamilyRegistry.default()
    assert reg.get(RepairFamily.CAUSAL_SMOOTHING).family is RepairFamily.CAUSAL_SMOOTHING
    assert reg.get("CAUSAL_SMOOTHING").family is RepairFamily.CAUSAL_SMOOTHING
    assert RepairFamily.CAUSAL_SMOOTHING in reg
    assert "CAUSAL_SMOOTHING" in reg
    assert "NOT_A_FAMILY" not in reg
    with pytest.raises(ValueError, match="unknown repair family"):
        reg.get("NOT_A_FAMILY")


def test_eligible_for_and_bidirectional_mapping():
    reg = RepairFamilyRegistry.default()
    # Every declaration's eligible diagnoses map back to that family.
    for declaration in reg.declarations:
        for diag_name in declaration.eligible_diagnoses:
            eligible = reg.eligible_families_for(diag_name)
            family_names = [d.family_name for d in eligible]
            assert declaration.family_name in family_names, (
                f"{diag_name} eligible families should include "
                f"{declaration.family_name}"
            )
    # A diagnosis with no eligible family yields an empty tuple.
    assert reg.eligible_families_for("INTEGRITY_FAILURE") != ()
    # Diagnoses without a family (not in any eligible set) — ABANDON covers
    # most; check the mapping is total for every canonical E2 name on at least
    # one family (ABANDON's eligible set covers hard rejects).
    for name in PLAN_E2_DIAGNOSIS_NAMES:
        assert reg.eligible_families_for(name), name


def test_registry_rejects_policy_mismatch_and_duplicates():
    decl = RepairFamilyDeclaration(
        family=RepairFamily.ABANDON,
        eligible_diagnoses=frozenset({"INTEGRITY_FAILURE"}),
        owner=ExecutionDomain.FE,
    )
    with pytest.raises(ValueError, match="policy_id"):
        RepairFamilyRegistry([decl], policy_id="OTHER_POLICY")
    dup = [
        RepairFamilyDeclaration(
            family=RepairFamily.ABANDON,
            eligible_diagnoses=frozenset({"INTEGRITY_FAILURE"}),
            owner=ExecutionDomain.FE,
        ),
        RepairFamilyDeclaration(
            family=RepairFamily.ABANDON,
            eligible_diagnoses=frozenset({"INTEGRITY_FAILURE"}),
            owner=ExecutionDomain.FE,
        ),
    ]
    with pytest.raises(ValueError, match="unique per family"):
        RepairFamilyRegistry(dup)


def test_registry_dict_round_trip():
    reg = RepairFamilyRegistry.default()
    data = reg.to_dict()
    assert data["policy_id"] == "FO_REPAIR_FAMILY"
    assert len(data["declarations"]) == 20
    # Rebuild a registry from serialized declarations and confirm equality.
    rebuilt = RepairFamilyRegistry(
        [
            RepairFamilyDeclaration(
                family=RepairFamily[d["family"]],
                eligible_diagnoses=frozenset(d["eligible_diagnoses"]),
                owner=ExecutionDomain(d["owner"]),
                parameter_schema=ParameterSchema(d["parameter_schema"]),
                parameter_prior=ParameterPrior(d["parameter_prior"]),
                maximum_candidates=d["maximum_candidates"],
                causality_class=CausalityClass(d["causality_class"]),
                complexity_cost=d["complexity_cost"],
                required_evidence_profile=d["required_evidence_profile"],
                policy_id=d["policy_id"],
                policy_version=d["policy_version"],
            )
            for d in data["declarations"]
        ]
    )
    assert rebuilt.family_names == reg.family_names


def test_declaration_family_name_and_eligible_for():
    d = RepairFamilyDeclaration(**_required_fields())
    assert d.family_name == "CAUSAL_SMOOTHING"
    assert d.eligible_for(DiagnosisKind.HIGH_TURNOVER)
    assert not d.eligible_for(DiagnosisKind.U_SHAPE)


def test_repair_candidate_slot_validation():
    slot = RepairCandidateSlot(diagnosis="HIGH_TURNOVER", family="CAUSAL_SMOOTHING", max_candidates=3)
    assert slot.max_candidates == 3
    with pytest.raises(ValueError, match="unknown repair family"):
        RepairCandidateSlot(diagnosis="HIGH_TURNOVER", family="NOT_A_FAMILY")
    with pytest.raises(ValueError, match="unknown diagnosis"):
        RepairCandidateSlot(diagnosis="NOT_A_DIAGNOSIS", family="CAUSAL_SMOOTHING")
    with pytest.raises(ValueError, match="max_candidates"):
        RepairCandidateSlot(diagnosis="HIGH_TURNOVER", family="CAUSAL_SMOOTHING", max_candidates=0)


def test_abandon_family_marks_hard_rejects():
    reg = RepairFamilyRegistry.default()
    abandon = reg.get(RepairFamily.ABANDON)
    # Integrity failures are hard rejects: no repair, ABANDON only.
    for name in ("INTEGRITY_FAILURE", "PIT_VIOLATION", "LABEL_TIMING_VIOLATION"):
        assert abandon.eligible_for(DiagnosisKind[name])
