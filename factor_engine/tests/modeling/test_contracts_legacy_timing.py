# -*- coding: utf-8 -*-
"""Model-layer redesign: contracts / legacy classification / decision-clock
tests (taskbook §1 / §5 / §7 / §11 / §14 / §16 / §35)."""
from __future__ import annotations

import numpy as np
import pytest

from modeling.contracts import (
    AFTER_CLOSE_TO_NEXT_VWAP,
    BEFORE_SAME_DAY_VWAP,
    DecisionClock,
    LabelContract,
    ModelExecutionClass,
    ModelOperatorSpec,
    ParamRole,
    ParameterSearchPolicy,
    SampleAdequacyContract,
    TimingKind,
    ashare_decision_clock,
    sample_adequacy_met,
    validate_model_operator_spec,
)
from modeling.legacy import (
    LEGACY_LOCAL_PREDICTIVE_CANONICALS,
    classify_execution_class,
    classification_of,
)
from modeling.timing import (
    check_same_day_target_gate,
    clock_compliant,
    vwap_to_vwap_label,
)
from modeling.presets import (
    MODEL_PARAM_RECOMMENDATIONS,
    PREDICTIVE_LINEAR_DEFAULT,
    PREDICTIVE_REGIME_DEFAULT,
    recommendation_map,
)


def test_execution_classes():
    assert {e.value for e in ModelExecutionClass} == {
        "local_rolling_estimator",
        "recursive_state_estimator",
        "same_time_cross_sectional",
        "predictive_supervised",
        "research_structural",
    }


def test_timing_kind_seven():
    # §35 requires seven timing kinds including FORWARD_LABEL_SUPERVISED.
    assert len(set(TimingKind)) == 7
    assert TimingKind.FORWARD_LABEL_SUPERVISED.value == "forward_label_supervised"


def test_rich_model_timing_validates_kind():
    from modeling.contracts import RichModelTiming

    RichModelTiming(timing_kind=TimingKind.FORWARD_LABEL_SUPERVISED.value)
    with pytest.raises(ValueError):
        RichModelTiming(timing_kind="not_a_kind")


def test_sample_adequacy_met():
    c = SampleAdequacyContract(
        min_raw_obs=10, min_effective_obs=5, min_unique_dates=3,
        min_unique_stocks=2, min_obs_per_parameter=2.0,
    )
    ok, fails = sample_adequacy_met(
        contract=c, raw_obs=100, effective_obs=80, unique_dates=50,
        unique_stocks=30, free_parameter_count=5,
    )
    assert ok and fails == []
    ok, fails = sample_adequacy_met(
        contract=c, raw_obs=5, effective_obs=80, unique_dates=50,
        unique_stocks=30, free_parameter_count=5,
    )
    assert not ok and any("raw_obs" in f for f in fails)


def test_regime_expert_contracts():
    c = SampleAdequacyContract(
        min_raw_obs=100, min_effective_obs=50, min_unique_dates=10,
        min_unique_stocks=5, min_obs_per_parameter=2.0,
        min_regime_obs=20, min_expert_obs=15,
    )
    ok, fails = sample_adequacy_met(
        contract=c, raw_obs=200, effective_obs=100, unique_dates=20,
        unique_stocks=10, free_parameter_count=3, regime_obs=10,
    )
    assert not ok and any("regime_obs" in f for f in fails)
    ok, _ = sample_adequacy_met(
        contract=c, raw_obs=200, effective_obs=100, unique_dates=20,
        unique_stocks=10, free_parameter_count=3, regime_obs=25, expert_obs=20,
    )
    assert ok


def test_label_contract_vwap_authority():
    # §8 VWAP-to-VWAP is the label authority.
    lc = vwap_to_vwap_label("vwap_20", 20)
    assert lc.return_basis == "vwap_to_vwap"
    assert lc.horizon_bars == 20
    assert lc.label_interval(5) == (5, 25)


def test_decision_clock_scenarios():
    after = ashare_decision_clock(AFTER_CLOSE_TO_NEXT_VWAP)
    before = ashare_decision_clock(BEFORE_SAME_DAY_VWAP)
    assert after.execution_at == "t+1 VWAP"
    assert before.execution_at == "t VWAP"
    # §62: AFTER_CLOSE may use close; BEFORE_SAME_DAY may not.
    assert clock_compliant(after, ["close", "volume"])[0] is True
    assert clock_compliant(before, ["close", "volume"])[0] is False
    assert clock_compliant(before, ["open", "pre_close"])[0] is True


def test_same_day_target_gate():
    ok, _ = check_same_day_target_gate(AFTER_CLOSE_TO_NEXT_VWAP, target_uses_full_day=True)
    assert ok
    ok, problems = check_same_day_target_gate(BEFORE_SAME_DAY_VWAP, target_uses_full_day=True)
    assert not ok and len(problems) >= 1
    ok, _ = check_same_day_target_gate(BEFORE_SAME_DAY_VWAP, features=["close", "volume"])
    assert not ok


def test_param_role_taxonomy():
    # §14 seven roles.
    assert {r.value for r in ParamRole} == {
        "economic_horizon", "model_complexity", "regularization",
        "estimator_resolution", "numerical_policy", "data_policy", "timing_policy",
    }


def test_parameter_search_policy_never_searchable_roles():
    with pytest.raises(ValueError):
        ParameterSearchPolicy(role=ParamRole.NUMERICAL_POLICY, searchable=True)
    with pytest.raises(ValueError):
        ParameterSearchPolicy(role=ParamRole.DATA_POLICY, searchable=True)
    # OK: regularisation tune-inside-validation, never searchable by miner.
    p = ParameterSearchPolicy(
        role=ParamRole.REGULARIZATION, searchable=False, tune_inside_validation=True
    )
    assert p.tune_inside_validation and not p.searchable


def test_model_operator_spec_validation():
    spec = ModelOperatorSpec(
        canonical="predictive_pcr", execution_class=ModelExecutionClass.PREDICTIVE_SUPERVISED,
        semantic_role="model_score", artifact_required=False,
    )
    assert len(validate_model_operator_spec(spec)) >= 1
    good = ModelOperatorSpec(
        canonical="predictive_pcr", execution_class=ModelExecutionClass.PREDICTIVE_SUPERVISED,
        semantic_role="model_score", artifact_required=True,
        training_lifecycle="artifact_walk_forward", default_searchable=False,
    )
    assert validate_model_operator_spec(good) == []


def test_legacy_classification():
    # §2.1 the five legacy local operators are LOCAL_ROLLING_ESTIMATOR + research.
    for canon in LEGACY_LOCAL_PREDICTIVE_CANONICALS:
        meta = classification_of(canon)
        assert meta is not None
        assert meta.execution_class == ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR
        assert meta.research_only is True
        assert meta.default_searchable is False
        assert classify_execution_class(canon) == ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR


def test_execution_class_heuristics():
    assert classify_execution_class("ts_kalman_level") == ModelExecutionClass.RECURSIVE_STATE_ESTIMATOR
    assert classify_execution_class("ts_dmd_trailing_eig") == ModelExecutionClass.RESEARCH_STRUCTURAL
    assert classify_execution_class("cs_knn_local_linear_residual") == ModelExecutionClass.SAME_TIME_CROSS_SECTIONAL
    assert classify_execution_class("ts_ar_coefficient") == ModelExecutionClass.LOCAL_ROLLING_ESTIMATOR


def test_presets_defaults():
    assert PREDICTIVE_LINEAR_DEFAULT.train_lookback_bars == 1000
    assert PREDICTIVE_LINEAR_DEFAULT.validation_bars == 252
    assert PREDICTIVE_REGIME_DEFAULT.train_mode == "decay_weighted_expanding"
    assert PREDICTIVE_REGIME_DEFAULT.embargo_bars == 5


def test_param_recommendation_table():
    recs = recommendation_map()
    assert ("elastic_net", "alpha") in recs
    assert recs[("elastic_net", "alpha")]["searchable_by_miner"] is False
    assert recs[("pcr", "n_components")]["searchable_by_miner"] is True
    # Numerical/resolution params are never miner-searchable (§15/§69).
    for family, param in (("dmd", "rank"), ("rqa", "eps"), ("te", "bins")):
        assert recs[(family, param)]["searchable_by_miner"] is False
