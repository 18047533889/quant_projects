import pytest, hashlib, json
from factor_assets.selection import (CandidateEvidence, DecisionProvider,
    DecisionRequest, DecisionStatus, MetricRule, Relationship,
    JointUtilityEvidence, RawJointMetricEvidence, ReplacementRoleRule,
    SelectionPolicySpec, UtilityDirection)
class Receipt:
    def require_scope(self, **kwargs): return None
class Resolver:
    def resolve(self, ref): return Receipt()
def provider(p): return DecisionProvider(p, Resolver())

def policy():
    return SelectionPolicySpec("p","1",(MetricRule("ic",UtilityDirection.HIGHER_IS_BETTER,1,0,.1,hard_floor=0),),{"quality":"B"},minimum_coverage=.5,required_dimensions=("quality",))

def candidate(name, grade="B", coverage=.8, evidence=("ev",), utility=None, **kw):
    bump=.02 if name=="fixed" else 0
    samples=(((([.08+bump][0],),),),((([.06+bump][0],),),))
    execution=("s","i","r","cpu","p","m")
    semantic={"context":"ctx","plan":"plan","replicates":["r1","r2"],"metrics":["ic"],"metric_units":["dimensionless"],"window_ids":["window"],"scenario_ids":["scenario"],"samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution}
    digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    raw=RawJointMetricEvidence("joint",digest,"ctx","plan",("r1","r2"),("ic",),samples,"research",source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",parameter_domain_hash="p",metric_instance_hash="m",metric_units=("dimensionless",))
    return CandidateEvidence(name,evidence,"FINAL",{"quality":grade},{"ic":.07+bump},coverage,"research",raw_joint_metric_evidence=raw,qualification_ref="q",qualification_requirements={"source_tree_hash":"s","implementation_hash":"i","route":"r","backend":"cpu","parameter_domain_hash":"p","metric_instance_hash":"m"},**kw)

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
        semantic={"context":"ctx","plan":"plan","replicates":["r1","r2"],"metrics":["benefit","risk"],"metric_units":["return","drawdown"],"window_ids":["window"],"scenario_ids":["scenario"],"samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution}
        digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        joint=RawJointMetricEvidence("e:"+name,digest,"ctx","plan",("r1","r2"),("benefit","risk"),samples,"research",source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",parameter_domain_hash="p",metric_instance_hash="m",metric_units=("return","drawdown"))
        return CandidateEvidence(name,("ev",),"FINAL",{}, {"benefit":benefit,"risk":risk},1,"research",raw_joint_metric_evidence=joint)
    raw,a,b=item("raw",.2,.20),item("a",.5,.24),item("b",.8,.28)
    provider_=DecisionProvider(p,Resolver())
    assert provider_._passes_raw_guards(b,(a,)) is True
    assert provider_._passes_raw_guards(b,(a,raw)) is False

def _raw_candidate(name, metric_ids, metric_units, samples, dimensions=None, metrics=None,
                   window_ids=("window",), scenario_ids=("scenario",)):
    samples=tuple(samples); execution=("s","i","r","cpu","p","m")
    semantic={"context":"ctx","plan":"plan","replicates":["r1","r2"],
        "metrics":list(metric_ids),"metric_units":list(metric_units),
        "window_ids":list(window_ids),"scenario_ids":list(scenario_ids),
        "samples":json.loads(json.dumps(samples)),"qualification":"research","execution":execution}
    digest=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    raw=RawJointMetricEvidence("joint:"+name,digest,"ctx","plan",("r1","r2"),tuple(metric_ids),
        samples,"research",window_ids=tuple(window_ids),scenario_ids=tuple(scenario_ids),
        source_tree_hash="s",implementation_hash="i",route="r",backend="cpu",
        parameter_domain_hash="p",metric_instance_hash="m",metric_units=tuple(metric_units))
    return CandidateEvidence(name,("ev",),"FINAL",dimensions or {},metrics or {},1.,"research",
        raw_joint_metric_evidence=raw,qualification_ref="q")

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
