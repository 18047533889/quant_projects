import pytest, hashlib, json
from types import SimpleNamespace
from factor_assets.selection import (CandidateEvidence, DecisionProvider,
    DecisionRequest, DecisionStatus, MetricRule, Relationship,
    JointUtilityEvidence, RawJointMetricEvidence, ReplacementRoleRule,
    SelectionPolicySpec, UtilityDirection)
class Receipt:
    def require_scope(self, **kwargs): return None
    def require_assertion_suite(self, **kwargs): return None
class Resolver:
    def __init__(self, policy=None): self.policy=policy
    def resolve(self, ref): return Receipt()
    def resolve_evidence(self, ref, *, candidate):
        raw=candidate.raw_joint_metric_evidence
        return SimpleNamespace(candidate_id=candidate.candidate_id,raw_content_hash=raw.content_hash,
            recipe_hash=raw.recipe_hash,fitted_state_hash=raw.fitted_state_hash,data_snapshot_hash=raw.data_snapshot_hash,
            universe_hash=raw.universe_hash,label_hash=raw.label_hash,value_artifact_hash=raw.value_artifact_hash,
            dimensions=candidate.dimensions,coverage=candidate.coverage,metric_instance_hash=raw.metric_instance_hash,
            parameter_domain_hash=raw.parameter_domain_hash,
            metric_directions={r.metric_id:r.direction.value for r in self.policy.metric_rules if r.metric_id in raw.metric_ids})
def provider(p):
    resolver=Resolver(p); return DecisionProvider(p,resolver,resolver)

def policy():
    return SelectionPolicySpec("p","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,.1,hard_floor=0),),{"quality":"B"},minimum_coverage=.5,required_dimensions=("quality",))

def candidate(name, grade="B", coverage=.8, evidence=("ev",), utility=None, **kw):
    bump=.02 if name=="fixed" else 0
    samples=(((([.08+bump][0],),),),((([.06+bump][0],),),))
    execution=("s","i","r","cpu","p","m")
    identity=(name,"recipe","state","data","universe","label","values:"+name)
    pairing=("plan-content","samples","times","mask")
    semantic={"context":"ctx","plan":"plan","pairing":pairing,"replicates":["r1","r2"],"metrics":["ic"],"metric_units":["dimensionless"],"window_ids":["window"],"scenario_ids":["scenario"],"samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution,"candidate":identity}
    digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    raw=RawJointMetricEvidence("joint",digest,"ctx","plan",("r1","r2"),("ic",),samples,"research",source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",parameter_domain_hash="p",metric_instance_hash="m",metric_units=("dimensionless",),candidate_id=name,recipe_hash="recipe",fitted_state_hash="state",data_snapshot_hash="data",universe_hash="universe",label_hash="label",value_artifact_hash="values:"+name,resampling_plan_content_hash=pairing[0],sample_identity_hash=pairing[1],time_identity_hash=pairing[2],common_mask_hash=pairing[3])
    return CandidateEvidence(name,evidence,"FINAL",{"quality":grade},{"ic":.07+bump},coverage,"research",raw_joint_metric_evidence=raw,qualification_ref="q",qualification_requirements={"source_tree_hash":"s","implementation_hash":"i","route":"r","backend":"cpu","parameter_domain_hash":"p","metric_instance_hash":"m"},health_evidence_ref="health:"+name,health_candidate_id=name,health_coverage=coverage,evidence_bundle_ref=evidence[0] if evidence else None,**kw)

def request(p, candidates, request_id="r"):
    candidates=tuple(candidates)
    baseline="raw" if any(c.candidate_id=="raw" for c in candidates) else min(c.candidate_id for c in candidates)
    return DecisionRequest(request_id,p.policy_id,p.content_hash,"ctx","variant","final",baseline,candidates,"FINAL","family")

def test_hard_gates_precede_scoring_and_do_not_borrow_grade_or_coverage():
    p=policy(); result=provider(p).decide(request(p,[candidate("raw",utility=.1),candidate("d",grade="D",utility=.99),candidate("thin",coverage=.2,utility=.99)]))
    assert result.winner_id=="raw"
    assert result.point_utility["d"] is None and result.point_utility["thin"] is None

def test_policy_binding_and_order_invariant_semantic_identity():
    p=policy(); a,b=candidate("a",utility=.4),candidate("b",utility=.6)
    one=provider(p).decide(request(p,[a,b],"transport-1"))
    two=provider(p).decide(request(p,[b,a],"transport-2"))
    assert one.decision_id==two.decision_id and one.candidate_set_hash==two.candidate_set_hash
    bad=DecisionRequest("r","p","changed","ctx","variant","final","a",(a,b),"FINAL","family")
    with pytest.raises(ValueError,match="policy binding"): provider(p).decide(bad)

def test_paired_relationships_are_typed_and_missing_evidence_is_inconclusive():
    p=policy(); raw=candidate("raw",grade="D",utility=.1)
    repaired=candidate("fixed",utility=.5,paired_effect=.02,paired_interval=(.01,.03),paired_evidence_ref="paired")
    result=provider(p).decide(request(p,[raw,repaired]))
    assert result.relationship["fixed"] is Relationship.REPAIRED
    assert result.relationship["raw"] is Relationship.INCONCLUSIVE

def test_missing_evidence_and_qualification_are_unselectable():
    p=policy(); missing=candidate("raw",evidence=(),utility=.9)
    result=provider(p).decide(request(p,[missing]))
    assert result.status is DecisionStatus.WAIT and result.point_utility["raw"] is None
    unqualified=CandidateEvidence("x",("ev",),"FINAL",{"quality":"B"},{"ic":.05},.8,None)
    assert DecisionProvider(p).decide(request(p,[unqualified])).winner_id is None

def test_nonfinite_and_bool_metric_inputs_fail_closed_at_boundary():
    with pytest.raises((TypeError,ValueError)):
        CandidateEvidence("x",("ev",),"FINAL",{"quality":"B"},{"ic":float("inf")},.8,"research")

def test_identical_candidate_cannot_replace_legal_baseline_by_id_tiebreak():
    p=policy(); raw=candidate("raw"); challenger=candidate("zzz")
    result=provider(p).decide(request(p,[challenger,raw]))
    assert result.relationship["zzz"] is Relationship.EQUIVALENT
    assert result.winner_id=="raw"

def test_raw_joint_samples_are_deeply_frozen():
    c=candidate("raw"); before=c.raw_joint_metric_evidence.content_hash
    with pytest.raises(TypeError):
        c.raw_joint_metric_evidence.samples[0][0][0][0]=.99
    assert c.raw_joint_metric_evidence.content_hash==before

def test_t68_original_unit_cumulative_raw_guard_rejects_laundered_steps():
    p=SelectionPolicySpec("guard","1",(
        MetricRule("benefit",UtilityDirection.HIGHER_IS_BETTER,.8,0,1,unit="return",noninferiority_delta=.5),
        MetricRule("risk",UtilityDirection.LOWER_IS_BETTER,.2,1,0,unit="drawdown",noninferiority_delta=.05),
    ),{},block_weights={"P":1.0})
    def item(name, benefit, risk):
        samples=((((benefit,risk),),),(((benefit,risk),),))
        execution=("s","i","r","cpu","p","m")
        identity=(name,"recipe","state","data","universe","label","values:"+name)
        pairing=("plan-content","samples","times","mask")
        semantic={"context":"ctx","plan":"plan","pairing":pairing,"replicates":["r1","r2"],"metrics":["benefit","risk"],"metric_units":["return","drawdown"],"window_ids":["window"],"scenario_ids":["scenario"],"samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution,"candidate":identity}
        digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        joint=RawJointMetricEvidence("e:"+name,digest,"ctx","plan",("r1","r2"),("benefit","risk"),samples,"research",source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",parameter_domain_hash="p",metric_instance_hash="m",metric_units=("return","drawdown"),candidate_id=name,recipe_hash="recipe",fitted_state_hash="state",data_snapshot_hash="data",universe_hash="universe",label_hash="label",value_artifact_hash="values:"+name,resampling_plan_content_hash=pairing[0],sample_identity_hash=pairing[1],time_identity_hash=pairing[2],common_mask_hash=pairing[3])
        return CandidateEvidence(name,("ev",),"FINAL",{}, {"benefit":benefit,"risk":risk},1,"research",raw_joint_metric_evidence=joint)
    raw,a,b=item("raw",.2,.20),item("a",.5,.24),item("b",.8,.28)
    provider_=DecisionProvider(p,Resolver())
    assert provider_._passes_raw_guards(b,(a,)) is True
    assert provider_._passes_raw_guards(b,(a,raw)) is False

def _raw_candidate(name, metric_ids, metric_units, samples, dimensions=None, metrics=None,
                   window_ids=("window",), scenario_ids=("scenario",), plan="plan", context="ctx",
                   pairing=None, data_snapshot="data"):
    samples=tuple(samples); execution=("s","i","r","cpu","p","m")
    identity=(name,"recipe","state",data_snapshot,"universe","label","values:"+name)
    pairing=pairing or ("plan-content:"+plan,"samples","times","mask")
    semantic={"context":context,"plan":plan,"pairing":pairing,"replicates":["r1","r2"],
        "metrics":list(metric_ids),"metric_units":list(metric_units),
        "window_ids":list(window_ids),"scenario_ids":list(scenario_ids),
        "samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution,"candidate":identity}
    digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    raw=RawJointMetricEvidence("joint:"+name,digest,context,plan,("r1","r2"),tuple(metric_ids),
        samples,"research",window_ids=tuple(window_ids),scenario_ids=tuple(scenario_ids),
        source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",
        parameter_domain_hash="p",metric_instance_hash="m",metric_units=tuple(metric_units),
        candidate_id=name,recipe_hash="recipe",fitted_state_hash="state",data_snapshot_hash=data_snapshot,
        universe_hash="universe",label_hash="label",value_artifact_hash="values:"+name,
        resampling_plan_content_hash=pairing[0],sample_identity_hash=pairing[1],time_identity_hash=pairing[2],common_mask_hash=pairing[3])
    return CandidateEvidence(name,("ev",),"FINAL",dimensions or {},metrics or {},1.,"research",
        raw_joint_metric_evidence=raw,qualification_ref="q",evidence_bundle_ref="ev")

def test_t02_required_dimension_ids_are_unique_and_empty_means_no_dimension_gate():
    rule=MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1)
    with pytest.raises(ValueError,match="duplicate required dimensions"):
        SelectionPolicySpec("dup","1",(rule,),{},required_dimensions=("quality","quality"))
    p=SelectionPolicySpec("none","1",(rule,),{},required_dimensions=())
    c=_raw_candidate("raw",("ic",),("dimensionless",),((((.6,),),),(((.7,),),)))
    assert provider(p).decide(request(p,[c])).winner_id == "raw"

@pytest.mark.parametrize("metric_id",("shape_score","state_purity"))
def test_t10_policy_used_shape_or_state_raw_perturbation_changes_utility_and_gate(metric_id):
    p=SelectionPolicySpec("profile:"+metric_id,"1",(
        MetricRule(metric_id,UtilityDirection.HIGHER_IS_BETTER,1,0,1,hard_floor=.5),),{})
    good=_raw_candidate("raw",(metric_id,),("dimensionless",),((((.8,),),),(((.8,),),)))
    bad=_raw_candidate("bad",(metric_id,),("dimensionless",),((((.4,),),),(((.4,),),)))
    result=provider(p).decide(request(p,[good,bad]))
    assert result.point_utility["raw"] == pytest.approx(80.)
    assert result.point_utility["bad"] is None
    assert any(r.candidate_id=="bad" and r.gate_id==f"metric:{metric_id}:floor" for r in result.gate_receipts)

@pytest.mark.parametrize("role",("CHEAPER","RISK_CLEANER","SIMPLER"))
def test_t24_noninferior_candidate_records_proven_replacement_role(role):
    rules=(MetricRule("quality",UtilityDirection.HIGHER_IS_BETTER,1,0,1),
           MetricRule("cost",UtilityDirection.LOWER_IS_BETTER,0,10,0,unit="bps",noninferiority_delta=3))
    p=SelectionPolicySpec("role:"+role,"1",rules,{},replacement_role_rules=(
        ReplacementRoleRule(role,"cost",UtilityDirection.LOWER_IS_BETTER,1),))
    raw=_raw_candidate("raw",("quality","cost"),("dimensionless","bps"),((((.7,10),),),(((.7,10),),)))
    better=_raw_candidate("better",("quality","cost"),("dimensionless","bps"),((((.7,8),),),(((.7,8),),)))
    result=provider(p).decide(request(p,[raw,better]))
    assert result.relationship["better"] is Relationship.EQUIVALENT
    assert result.replacement_role["better"] == role
    assert result.winner_id == "better"

def test_t24_naked_scalar_or_subthreshold_raw_cannot_spoof_replacement_role():
    rules=(MetricRule("quality",UtilityDirection.HIGHER_IS_BETTER,1,0,1),
           MetricRule("cost",UtilityDirection.LOWER_IS_BETTER,0,10,0,unit="bps",noninferiority_delta=3))
    p=SelectionPolicySpec("role:spoof","1",rules,{},replacement_role_rules=(
        ReplacementRoleRule("CHEAPER","cost",UtilityDirection.LOWER_IS_BETTER,1),))
    raw=_raw_candidate("raw",("quality","cost"),("dimensionless","bps"),((((.7,10),),),(((.7,10),),)))
    spoof=_raw_candidate("spoof",("quality","cost"),("dimensionless","bps"),((((.7,9.5),),),(((.7,9.5),),)),metrics={"cost":0})
    result=provider(p).decide(request(p,[raw,spoof]))
    assert result.replacement_role["spoof"] is None
    assert result.winner_id == "raw"

def test_t24_replacement_role_direction_must_match_metric_semantics():
    with pytest.raises(ValueError,match="direction must match"):
        SelectionPolicySpec("role:direction","1",(
            MetricRule("cost",UtilityDirection.LOWER_IS_BETTER,1,10,0,unit="bps"),),{},
            replacement_role_rules=(ReplacementRoleRule("CHEAPER","cost",UtilityDirection.HIGHER_IS_BETTER,1),))

def test_t24_replacement_role_requires_policy_unit_on_both_raw_artifacts():
    rules=(MetricRule("quality",UtilityDirection.HIGHER_IS_BETTER,1,0,1),
           MetricRule("cost",UtilityDirection.LOWER_IS_BETTER,0,10,0,unit="bps",noninferiority_delta=3))
    p=SelectionPolicySpec("role:unit","1",rules,{},replacement_role_rules=(
        ReplacementRoleRule("CHEAPER","cost",UtilityDirection.LOWER_IS_BETTER,1),))
    raw=_raw_candidate("raw",("quality","cost"),("dimensionless","bps"),((((.7,10),),),(((.7,10),),)))
    wrong_unit=_raw_candidate("wrong",("quality","cost"),("dimensionless","percent"),((((.7,8),),),(((.7,8),),)))
    result=provider(p).decide(request(p,[raw,wrong_unit]))
    assert result.replacement_role["wrong"] is None
    assert result.winner_id == "raw"

def test_t69_equivalent_scenario_duplication_with_conserved_weight_is_invariant():
    rule=MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1)
    p1=SelectionPolicySpec("scenario:one","1",(rule,),{},scenario_ids=("event",),scenario_weights=(1.,))
    p2=SelectionPolicySpec("scenario:alias","1",(rule,),{},scenario_ids=("event:a","event:b"),scenario_weights=(.5,.5))
    one=_raw_candidate("raw",("ic",),("dimensionless",),((((.6,),),),(((.8,),),)),scenario_ids=("event",))
    two=_raw_candidate("raw",("ic",),("dimensionless",),((((.6,),(.6,)),),(((.8,),(.8,)),)),scenario_ids=("event:a","event:b"))
    left=provider(p1).decide(request(p1,[one])).point_utility["raw"]
    right=provider(p2).decide(request(p2,[two])).point_utility["raw"]
    assert left == pytest.approx(right)

def test_gate_statistic_is_explicit_hashed_and_does_not_equate_caller_scalar_to_bootstrap_mean():
    base=MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1,hard_floor=.5)
    assert base.gate_statistic == "joint_cell_mean"
    with pytest.raises(ValueError,match="unsupported hard-gate statistic"):
        MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1,gate_statistic="caller_scalar")
    p=SelectionPolicySpec("gate-stat","1",(base,),{})
    c=_raw_candidate("raw",("ic",),("dimensionless",),((((.6,),),),(((.8,),),)),metrics={"ic":999})
    assert provider(p).decide(request(p,[c])).winner_id == "raw"

def test_m05_swapped_candidate_bound_raw_evidence_is_rejected_before_scoring():
    p=SelectionPolicySpec("identity","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{})
    a=_raw_candidate("a",("ic",),("dimensionless",),((((.6,),),),(((.7,),),)))
    b=CandidateEvidence("b",a.evidence_refs,a.fidelity,a.dimensions,a.metrics,a.coverage,
        a.qualification_scope,raw_joint_metric_evidence=a.raw_joint_metric_evidence,qualification_ref="q")
    result=provider(p).decide(request(p,[a,b]))
    assert result.eligibility["b"] is False
    assert any(r.candidate_id=="b" and r.gate_id=="candidate_identity" for r in result.gate_receipts)

def test_m06_unit_mismatch_disqualifies_even_a_lone_baseline():
    p=SelectionPolicySpec("units","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1,unit="fraction"),),{})
    raw=_raw_candidate("raw",("ic",),("percent",),((((.6,),),),(((.7,),),)))
    result=provider(p).decide(request(p,[raw]))
    assert result.winner_id is None and not result.eligibility["raw"]

def test_m07_equivalence_uses_equivalence_band_not_ni_margin():
    p=SelectionPolicySpec("eq","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{},
        noninferiority_margin=.20,equivalence_margin=.01)
    raw=_raw_candidate("raw",("ic",),("dimensionless",),((((.7,),),),(((.7,),),)))
    down=_raw_candidate("down",("ic",),("dimensionless",),((((.6,),),),(((.65,),),)))
    result=provider(p).decide(request(p,[raw,down]))
    assert result.relationship["down"] is Relationship.NONINFERIOR

def test_m08_bad_named_axes_is_isolated_without_aborting_batch():
    p=SelectionPolicySpec("axes","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{})
    raw=_raw_candidate("raw",("ic",),("dimensionless",),((((.7,),),),(((.7,),),)))
    bad=_raw_candidate("bad",("ic",),("dimensionless",),((((.8,),),),(((.8,),),)),window_ids=("other",))
    result=provider(p).decide(request(p,[raw,bad]))
    assert result.winner_id=="raw" and result.point_utility["bad"] is None

def test_m09_original_raw_guard_requires_exact_context_and_plan():
    p=SelectionPolicySpec("pair","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1,noninferiority_delta=.2),),{})
    left=_raw_candidate("left",("ic",),("dimensionless",),((((.7,),),),(((.7,),),)),plan="P")
    other=_raw_candidate("other",("ic",),("dimensionless",),((((.6,),),),(((.6,),),)),plan="Q")
    assert provider(p)._passes_raw_guards(left,(other,)) is False

def test_m09_same_plan_ref_and_replicate_names_cannot_hide_different_plan_content():
    p=SelectionPolicySpec("pair-content","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{})
    samples=((((.7,),),),(((.7,),),))
    left=_raw_candidate("left",("ic",),("dimensionless",),samples,pairing=("content:P","samples","times","mask"))
    right=_raw_candidate("right",("ic",),("dimensionless",),samples,pairing=("content:Q","samples","times","mask"))
    assert provider(p)._passes_raw_guards(left,(right,)) is False

def test_m09_same_pair_draws_cannot_compare_different_data_snapshot():
    p=SelectionPolicySpec("pair-data","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),),{})
    samples=((((.7,),),),(((.7,),),))
    left=_raw_candidate("left",("ic",),("dimensionless",),samples,data_snapshot="data:A")
    right=_raw_candidate("right",("ic",),("dimensionless",),samples,data_snapshot="data:B")
    assert provider(p)._passes_raw_guards(left,(right,)) is False

def test_m10_raw_guard_uses_policy_weighted_effect_within_replicate():
    p=SelectionPolicySpec("estimand","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1,noninferiority_delta=0),),{},
        window_ids=("main","stress"),window_weights=(.99,.01))
    left=_raw_candidate("left",("ic",),("dimensionless",),
        ((((.02,),),((-1.,),)),(((.02,),),((-1.,),))),window_ids=("main","stress"))
    right=_raw_candidate("right",("ic",),("dimensionless",),
        ((((0.,),),((0.,),)),(((0.,),),((0.,),))),window_ids=("main","stress"))
    assert provider(p)._passes_raw_guards(left,(right,)) is True

def test_m11_required_grade_and_health_identity_are_evidence_bound():
    with pytest.raises(ValueError,match="unknown dimension grade"):
        CandidateEvidence("x",("e",),"FINAL",{"quality":"invented"},{},1,"research")
    p=policy(); raw=candidate("raw")
    spoof=CandidateEvidence(raw.candidate_id,raw.evidence_refs,raw.fidelity,raw.dimensions,raw.metrics,.99,
        raw.qualification_scope,raw_joint_metric_evidence=raw.raw_joint_metric_evidence,qualification_ref="q",
        health_evidence_ref="health",health_candidate_id="raw",health_coverage=.5)
    assert provider(p).decide(request(p,[spoof])).winner_id is None

def test_m12_qualification_ref_changes_request_identity_and_retained_is_not_eligible():
    p=policy(); raw=candidate("raw"); challenger=candidate("zzz")
    changed=CandidateEvidence(raw.candidate_id,raw.evidence_refs,raw.fidelity,raw.dimensions,raw.metrics,raw.coverage,
        raw.qualification_scope,raw_joint_metric_evidence=raw.raw_joint_metric_evidence,qualification_ref="q2",
        health_evidence_ref=raw.health_evidence_ref,health_candidate_id="raw",health_coverage=raw.health_coverage)
    assert request(p,[raw]).candidate_set_hash != request(p,[changed]).candidate_set_hash
    result=provider(p).decide(request(p,[raw,challenger]))
    assert result.eligible_ids == ("raw","zzz") and result.retained_ids == ("raw",)

def test_policy_hash_binds_frozen_estimand_and_qualification_suite():
    rule=(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,1),)
    assertions=["golden"]
    base=SelectionPolicySpec("hash","1",rule,{},required_qualification_assertions=assertions)
    before=base.content_hash
    assertions.append("late")
    assert base.required_qualification_assertions==("golden",) and base.content_hash==before
    suite=SelectionPolicySpec("hash","1",rule,{},qualification_suite_id="qe.other.v1")
    changed=SelectionPolicySpec("hash","1",rule,{},effect_estimand="weighted_window_scenario_mean_within_replicate",
        required_qualification_assertions=("golden","parity"))
    assert len({before,suite.content_hash,changed.content_hash})==3

def test_m12_production_requires_separate_purpose_scoped_use_admission():
    class Admission:
        content_hash="admission-hash"
        def require_use_admission(self, *, purpose, candidate_id, **kwargs):
            if (purpose,candidate_id)!=("PRODUCTION","raw"): raise ValueError("wrong scope")
    class ProductionResolver(Resolver):
        current_time="2026-09-09T00:00:00Z"
        current_revocation_epoch=1
        def resolve(self, ref): return Admission() if ref=="admit" else Receipt()
    p=policy(); raw=candidate("raw")
    production=DecisionRequest("prod",p.policy_id,p.content_hash,"ctx","PRODUCTION","final","raw",(raw,),"FINAL","family")
    assert DecisionProvider(p,ProductionResolver(p),ProductionResolver(p)).decide(production).winner_id is None
    admitted=CandidateEvidence(raw.candidate_id,raw.evidence_refs,raw.fidelity,raw.dimensions,raw.metrics,raw.coverage,
        raw.qualification_scope,raw_joint_metric_evidence=raw.raw_joint_metric_evidence,qualification_ref="q",
        health_evidence_ref=raw.health_evidence_ref,health_candidate_id="raw",health_coverage=raw.health_coverage,
        use_admission_ref="admit",evidence_bundle_ref=raw.evidence_bundle_ref)
    production=DecisionRequest("prod",p.policy_id,p.content_hash,"ctx","PRODUCTION","final","raw",(admitted,),"FINAL","family")
    resolver=ProductionResolver(p)
    assert DecisionProvider(p,resolver,resolver).decide(production).winner_id=="raw"
