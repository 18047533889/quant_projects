"""Authoritative deterministic selection over pre-computed evidence."""
from dataclasses import dataclass, asdict
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional
import hashlib, json, math

def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a non-boolean number")
    value = float(value)
    if not math.isfinite(value): raise ValueError(f"{name} must be finite")
    return value

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _freeze(value):
    return tuple(_freeze(v) for v in value) if isinstance(value,(list,tuple)) else value
def _quantile(values, q):
    ordered=sorted(values); position=q*(len(ordered)-1); lo=math.floor(position); hi=math.ceil(position)
    return ordered[lo] if lo==hi else ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo])

class UtilityDirection(Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class DecisionStatus(Enum):
    SELECTED = "SELECTED"
    WAIT = "WAIT"
    INCOMPARABLE = "INCOMPARABLE"
    REJECTED = "REJECTED"

class Relationship(Enum):
    SUPERIOR = "SUPERIOR"
    NONINFERIOR = "NONINFERIOR"
    EQUIVALENT = "EQUIVALENT"
    INCONCLUSIVE = "INCONCLUSIVE"
    REPAIRED = "REPAIRED"

@dataclass(frozen=True)
class JointUtilityEvidence:
    """Joint same-plan desirabilities: replicate -> window -> scenario -> block."""
    evidence_id: str
    content_hash: str
    comparison_context_hash: str
    resampling_plan_ref: str
    replicate_ids: tuple[str,...]
    block_ids: tuple[str,...]
    block_weights: tuple[float,...]
    window_weights: tuple[float,...]
    scenario_weights: tuple[float,...]
    samples: tuple
    qualification_scope: str
    window_ids: tuple[str,...] = ("window",)
    scenario_ids: tuple[str,...] = ("scenario",)
    balanced_arithmetic_weight: float = .55
    scenario_mean_weight: float = .75
    window_mean_weight: float = .60
    lower_tail_mass: float = .20
    conservative_quantile: float = .10
    def __post_init__(self):
        object.__setattr__(self,"replicate_ids",tuple(self.replicate_ids)); object.__setattr__(self,"block_ids",tuple(self.block_ids)); object.__setattr__(self,"window_ids",tuple(self.window_ids)); object.__setattr__(self,"scenario_ids",tuple(self.scenario_ids)); object.__setattr__(self,"block_weights",tuple(self.block_weights)); object.__setattr__(self,"window_weights",tuple(self.window_weights)); object.__setattr__(self,"scenario_weights",tuple(self.scenario_weights))
        if not all((self.evidence_id,self.content_hash,self.comparison_context_hash,self.resampling_plan_ref,self.replicate_ids,self.qualification_scope)): raise ValueError("complete joint evidence provenance required")
        if len(self.replicate_ids)<2: raise ValueError("at least two joint replicates required")
        if len(set(self.replicate_ids))!=len(self.replicate_ids): raise ValueError("duplicate replicate ids")
        for weights,name in ((self.block_weights,"block"),(self.window_weights,"window"),(self.scenario_weights,"scenario")):
            if not weights or any(_number(w,name+" weight")<=0 for w in weights) or not math.isclose(sum(weights),1.,abs_tol=1e-12): raise ValueError(f"{name} weights must be positive and sum to one")
        if len(self.block_ids)!=len(self.block_weights) or len(self.samples)!=len(self.replicate_ids): raise ValueError("joint evidence axes mismatch")
        object.__setattr__(self,"samples",_freeze(self.samples)); canonical=json.loads(json.dumps(self.samples))
        if len(self.window_ids)!=len(self.window_weights) or len(self.scenario_ids)!=len(self.scenario_weights): raise ValueError("named utility axes mismatch")
        for rep in canonical:
            if len(rep)!=len(self.window_weights): raise ValueError("window axis mismatch")
            for window in rep:
                if len(window)!=len(self.scenario_weights): raise ValueError("scenario axis mismatch")
                for scenario in window:
                    if len(scenario)!=len(self.block_weights): raise ValueError("block axis mismatch")
                    if any(not 0<=_number(v,"desirability")<=1 for v in scenario): raise ValueError("desirability outside [0,1]")
        semantic={"context":self.comparison_context_hash,"plan":self.resampling_plan_ref,"replicates":self.replicate_ids,"blocks":self.block_ids,"block_weights":self.block_weights,"window_ids":self.window_ids,"window_weights":self.window_weights,"scenario_ids":self.scenario_ids,"scenario_weights":self.scenario_weights,"samples":canonical,"qualification":self.qualification_scope,"coefficients":(self.balanced_arithmetic_weight,self.scenario_mean_weight,self.window_mean_weight,self.lower_tail_mass,self.conservative_quantile)}
        if _digest(semantic)!=self.content_hash: raise ValueError("joint evidence content hash mismatch")

    def fitness(self):
        results=self.replicate_scores()
        q10=_quantile(results,self.conservative_quantile)
        return 100*sum(results)/len(results),100*q10
    def replicate_scores(self):
        results=[]
        for rep in self.samples:
            qs=[]
            for window in rep:
                us=[]
                for values in window:
                    arithmetic=sum(w*v for w,v in zip(self.block_weights,values))
                    geometric=0. if any(v==0 for v in values) else math.exp(sum(w*math.log(v) for w,v in zip(self.block_weights,values)))
                    us.append(self.balanced_arithmetic_weight*arithmetic+(1-self.balanced_arithmetic_weight)*geometric)
                qs.append(self.scenario_mean_weight*sum(w*u for w,u in zip(self.scenario_weights,us))+(1-self.scenario_mean_weight)*min(us))
            mean=sum(w*q for w,q in zip(self.window_weights,qs))
            remaining=self.lower_tail_mass; tail=0.
            for q,w in sorted(zip(qs,self.window_weights)):
                take=min(remaining,w); tail+=take*q; remaining-=take
                if remaining<=1e-15: break
            results.append(self.window_mean_weight*mean+(1-self.window_mean_weight)*(tail/self.lower_tail_mass))
        return tuple(results)

@dataclass(frozen=True)
class MetricRule:
    metric_id: str
    direction: UtilityDirection
    weight: float
    bad: float
    good: float
    hard_floor: Optional[float] = None
    hard_ceiling: Optional[float] = None
    block_id: str = "P"
    unit: str = "dimensionless"
    noninferiority_delta: float = 0.0
    required: bool = True
    gate_statistic: str = "joint_cell_mean"
    def __post_init__(self):
        if not self.metric_id or not isinstance(self.direction, UtilityDirection): raise ValueError("valid metric rule identity required")
        weight, bad, good = _number(self.weight,"weight"), _number(self.bad,"bad"), _number(self.good,"good")
        if weight < 0 or bad == good: raise ValueError("invalid weight or anchors")
        if (self.direction is UtilityDirection.HIGHER_IS_BETTER) != (good > bad): raise ValueError("anchors contradict direction")
        for name in ("hard_floor", "hard_ceiling"):
            if getattr(self, name) is not None: _number(getattr(self, name), name)
        if not isinstance(self.required,bool) or not self.unit or _number(self.noninferiority_delta,"noninferiority_delta") < 0:
            raise ValueError("metric unit and nonnegative NI delta required")
        if self.gate_statistic != "joint_cell_mean":
            raise ValueError("unsupported hard-gate statistic")

@dataclass(frozen=True)
class ReplacementRoleRule:
    role: str
    metric_id: str
    direction: UtilityDirection
    minimum_improvement: float
    def __post_init__(self):
        if self.role not in ("CHEAPER","RISK_CLEANER","SIMPLER") or not self.metric_id: raise ValueError("predeclared replacement role/metric required")
        if not isinstance(self.direction,UtilityDirection) or _number(self.minimum_improvement,"minimum_improvement")<=0: raise ValueError("role direction and positive threshold required")

@dataclass(frozen=True)
class SelectionPolicySpec:
    policy_id: str
    version: str
    metric_rules: tuple[MetricRule, ...]
    dimension_floors: Mapping[str, str]
    minimum_coverage: float = 0.0
    noninferiority_margin: float = 0.0
    equivalence_margin: float = 0.0
    required_dimensions: tuple[str, ...] = ()
    block_weights: Mapping[str,float] = None
    window_weights: tuple[float,...] = (1.0,)
    scenario_weights: tuple[float,...] = (1.0,)
    balanced_arithmetic_weight: float = .55
    scenario_mean_weight: float = .75
    window_mean_weight: float = .60
    lower_tail_mass: float = .20
    conservative_quantile: float = .10
    window_ids: tuple[str,...] = ("window",)
    scenario_ids: tuple[str,...] = ("scenario",)
    replacement_role_rules: tuple[ReplacementRoleRule,...] = ()
    minimum_gain: float = 0.0
    effect_estimand: str = "weighted_window_scenario_mean_within_replicate"
    qualification_suite_id: str = "qe.standard.v1"
    required_qualification_assertions: tuple[str,...] = ("golden",)
    def __post_init__(self):
        if not self.policy_id or not self.version or not self.metric_rules: raise ValueError("complete policy required")
        if len({r.metric_id for r in self.metric_rules}) != len(self.metric_rules): raise ValueError("duplicate metric rules")
        if len(set(self.required_dimensions)) != len(self.required_dimensions): raise ValueError("duplicate required dimensions")
        if sum(r.weight for r in self.metric_rules) <= 0: raise ValueError("positive total metric weight required")
        if not 0 <= _number(self.minimum_coverage,"minimum_coverage") <= 1: raise ValueError("coverage outside [0,1]")
        if (_number(self.noninferiority_margin,"noninferiority_margin") < 0
                or _number(self.equivalence_margin,"equivalence_margin") < 0
                or _number(self.minimum_gain,"minimum_gain") < 0): raise ValueError("margins and gain must be nonnegative")
        if self.effect_estimand != "weighted_window_scenario_mean_within_replicate": raise ValueError("unsupported effect estimand")
        if not self.qualification_suite_id or not self.required_qualification_assertions: raise ValueError("qualification suite policy required")
        assertions=tuple(self.required_qualification_assertions)
        if len(set(assertions))!=len(assertions) or any(not isinstance(x,str) or not x for x in assertions): raise ValueError("valid unique qualification assertions required")
        object.__setattr__(self,"required_qualification_assertions",assertions)
        for dimension in self.required_dimensions:
            if not isinstance(dimension,str) or not dimension: raise ValueError("valid required dimension ids required")
        for dimension,floor in self.dimension_floors.items():
            if floor not in self._valid_grades(): raise ValueError("unknown dimension floor")
        object.__setattr__(self,"metric_rules",tuple(self.metric_rules)); object.__setattr__(self,"required_dimensions",tuple(self.required_dimensions)); object.__setattr__(self,"window_weights",tuple(self.window_weights)); object.__setattr__(self,"scenario_weights",tuple(self.scenario_weights)); object.__setattr__(self,"window_ids",tuple(self.window_ids)); object.__setattr__(self,"scenario_ids",tuple(self.scenario_ids))
        object.__setattr__(self,"replacement_role_rules",tuple(self.replacement_role_rules))
        metric_by_id={m.metric_id:m for m in self.metric_rules}
        if any(r.metric_id not in metric_by_id for r in self.replacement_role_rules): raise ValueError("replacement role metric must be a policy metric")
        if any(r.direction is not metric_by_id[r.metric_id].direction for r in self.replacement_role_rules): raise ValueError("replacement role direction must match metric semantics")
        object.__setattr__(self,"dimension_floors",MappingProxyType(dict(self.dimension_floors)))
        blocks = dict(self.block_weights or {"P":1.0})
        if set(blocks) != {r.block_id for r in self.metric_rules}: raise ValueError("block weights must exactly cover metric-rule blocks")
        if any(not any(r.block_id==block and r.weight>0 for r in self.metric_rules) for block in blocks): raise ValueError("each block requires a positive-weight metric")
        if len(self.window_ids)!=len(self.window_weights) or len(self.scenario_ids)!=len(self.scenario_weights): raise ValueError("policy named axes mismatch")
        for values,name in ((tuple(blocks.values()),"block"),(self.window_weights,"window"),(self.scenario_weights,"scenario")):
            if any(_number(v,name+" weight")<=0 for v in values) or not math.isclose(sum(values),1.,abs_tol=1e-12): raise ValueError(f"{name} weights must sum to one")
        object.__setattr__(self,"block_weights",MappingProxyType(blocks))
        for name in ("balanced_arithmetic_weight","scenario_mean_weight","window_mean_weight","lower_tail_mass","conservative_quantile"):
            if not 0 < _number(getattr(self,name),name) < 1: raise ValueError(f"{name} must be in (0,1)")
    @property
    def content_hash(self):
        return _digest({"id":self.policy_id,"version":self.version,"rules":[{**asdict(r),"direction":r.direction.value} for r in self.metric_rules],"floors":dict(sorted(self.dimension_floors.items())),"coverage":self.minimum_coverage,"ni":self.noninferiority_margin,"eq":self.equivalence_margin,"gain":self.minimum_gain,"estimand":self.effect_estimand,"qualification_suite":(self.qualification_suite_id,self.required_qualification_assertions),"dimensions":sorted(self.required_dimensions),"blocks":dict(sorted(self.block_weights.items())),"windows":tuple(zip(self.window_ids,self.window_weights)),"scenarios":tuple(zip(self.scenario_ids,self.scenario_weights)),"formula":(self.balanced_arithmetic_weight,self.scenario_mean_weight,self.window_mean_weight,self.lower_tail_mass,self.conservative_quantile),"replacement_roles":[{**asdict(r),"direction":r.direction.value} for r in self.replacement_role_rules]})

    @staticmethod
    def _valid_grades():
        return frozenset(("NONE","NOT_APPLICABLE","NOT_READY","D","C","B","B+","A","A+","S","S+"))

@dataclass(frozen=True)
class RawJointMetricEvidence:
    evidence_id: str; content_hash: str; comparison_context_hash: str; resampling_plan_ref: str
    replicate_ids: tuple[str,...]; metric_ids: tuple[str,...]; samples: tuple; qualification_scope: str
    window_ids: tuple[str,...] = ("window",); scenario_ids: tuple[str,...] = ("scenario",)
    source_tree_hash: str = ""; implementation_hash: str = ""; route: str = ""; backend: str = ""
    parameter_domain_hash: str = ""; metric_instance_hash: str = ""
    metric_units: tuple[str,...] = ()
    candidate_id: str = ""; recipe_hash: str = ""; fitted_state_hash: str = ""
    data_snapshot_hash: str = ""; universe_hash: str = ""; label_hash: str = ""
    value_artifact_hash: str = ""
    resampling_plan_content_hash: str = ""
    sample_identity_hash: str = ""
    time_identity_hash: str = ""
    common_mask_hash: str = ""
    def __post_init__(self):
        object.__setattr__(self,"replicate_ids",tuple(self.replicate_ids)); object.__setattr__(self,"metric_ids",tuple(self.metric_ids)); object.__setattr__(self,"window_ids",tuple(self.window_ids)); object.__setattr__(self,"scenario_ids",tuple(self.scenario_ids))
        object.__setattr__(self,"metric_units",tuple(self.metric_units))
        if not all((self.evidence_id,self.content_hash,self.comparison_context_hash,self.resampling_plan_ref,self.replicate_ids,self.metric_ids,self.qualification_scope,self.source_tree_hash,self.implementation_hash,self.route,self.backend,self.parameter_domain_hash,self.metric_instance_hash,self.candidate_id,self.recipe_hash,self.fitted_state_hash,self.data_snapshot_hash,self.universe_hash,self.label_hash,self.value_artifact_hash,self.resampling_plan_content_hash,self.sample_identity_hash,self.time_identity_hash,self.common_mask_hash)): raise ValueError("complete raw joint evidence, candidate, pairing, data, and execution identity required")
        if len(set(self.replicate_ids))!=len(self.replicate_ids) or len(set(self.metric_ids))!=len(self.metric_ids): raise ValueError("duplicate raw joint axes")
        if len(self.metric_units)!=len(self.metric_ids) or any(not unit for unit in self.metric_units): raise ValueError("one metric unit per raw metric required")
        if len(self.replicate_ids)<2: raise ValueError("at least two raw joint replicates required")
        object.__setattr__(self,"samples",_freeze(self.samples)); canonical=json.loads(json.dumps(self.samples))
        if len(canonical)!=len(self.replicate_ids): raise ValueError("replicate axis mismatch")
        for rep in canonical:
            if len(rep)!=len(self.window_ids): raise ValueError("named window axis mismatch")
            for window in rep:
                if len(window)!=len(self.scenario_ids): raise ValueError("named scenario axis mismatch")
                for scenario in window:
                    if len(scenario)!=len(self.metric_ids): raise ValueError("metric axis mismatch")
                    for value in scenario: _number(value,"raw joint metric")
        semantic={"context":self.comparison_context_hash,"plan":self.resampling_plan_ref,"pairing":(self.resampling_plan_content_hash,self.sample_identity_hash,self.time_identity_hash,self.common_mask_hash),"replicates":self.replicate_ids,"metrics":self.metric_ids,"metric_units":self.metric_units,"window_ids":self.window_ids,"scenario_ids":self.scenario_ids,"samples":canonical,"qualification":self.qualification_scope,"execution":(self.source_tree_hash,self.implementation_hash,self.route,self.backend,self.parameter_domain_hash,self.metric_instance_hash),"candidate":(self.candidate_id,self.recipe_hash,self.fitted_state_hash,self.data_snapshot_hash,self.universe_hash,self.label_hash,self.value_artifact_hash)}
        if _digest(semantic)!=self.content_hash: raise ValueError("raw joint evidence content hash mismatch")

@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    evidence_refs: tuple[str, ...]
    fidelity: str
    dimensions: Mapping[str, str]
    metrics: Mapping[str, float]
    coverage: float
    qualification_scope: Optional[str]
    conservative_utility: Optional[float] = None
    paired_effect: Optional[float] = None
    paired_interval: Optional[tuple[float,float]] = None
    paired_evidence_ref: Optional[str] = None
    bootstrap_replicate_ids: tuple[str, ...] = ()
    joint_utility_evidence: Optional[JointUtilityEvidence] = None
    qualification_ref: Optional[str] = None
    qualification_requirements: Optional[Mapping[str,str]] = None
    raw_joint_metric_evidence: Optional[RawJointMetricEvidence] = None
    health_evidence_ref: Optional[str] = None
    health_candidate_id: Optional[str] = None
    health_coverage: Optional[float] = None
    use_admission_ref: Optional[str] = None
    evidence_bundle_ref: Optional[str] = None
    def __post_init__(self):
        if not self.candidate_id: raise ValueError("candidate_id required")
        object.__setattr__(self,"dimensions",MappingProxyType(dict(self.dimensions)))
        if any(value not in SelectionPolicySpec._valid_grades() for value in self.dimensions.values()): raise ValueError("unknown dimension grade")
        object.__setattr__(self,"metrics",MappingProxyType({key:_number(value,f"metric {key}") for key,value in self.metrics.items()}))
        if not 0 <= _number(self.coverage,"coverage") <= 1: raise ValueError("coverage outside [0,1]")
        if self.health_coverage is not None and not 0 <= _number(self.health_coverage,"health coverage") <= 1: raise ValueError("health coverage outside [0,1]")
        if self.conservative_utility is not None: _number(self.conservative_utility,"conservative utility")
        if self.paired_effect is not None: _number(self.paired_effect,"paired effect")
        if self.paired_interval is not None and _number(self.paired_interval[0],"interval") > _number(self.paired_interval[1],"interval"): raise ValueError("reversed paired interval")
        if self.qualification_requirements is not None:
            object.__setattr__(self,"qualification_requirements",MappingProxyType(dict(self.qualification_requirements)))

@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    policy_id: str
    policy_content_hash: str
    comparison_context_hash: str
    purpose: str
    decision_level: str
    baseline_ref: Optional[str]
    candidates: tuple[CandidateEvidence, ...]
    required_final_fidelity: str
    hypothesis_family_ref: str
    original_raw_ref: Optional[str] = None
    def __post_init__(self):
        if not all((self.request_id,self.policy_id,self.policy_content_hash,self.comparison_context_hash,self.purpose,self.decision_level,self.required_final_fidelity,self.hypothesis_family_ref)): raise ValueError("complete request identity/context required")
        ids=[c.candidate_id for c in self.candidates]
        if not ids or len(ids)!=len(set(ids)): raise ValueError("unique nonempty candidates required")
        if not self.baseline_ref or self.baseline_ref not in ids: raise ValueError("comparison baseline_ref must name one candidate")
        if self.original_raw_ref is not None and self.original_raw_ref not in ids: raise ValueError("original_raw_ref must name one candidate")
        object.__setattr__(self,"candidates",tuple(self.candidates))
    @property
    def candidate_set_hash(self):
        return _digest(sorted((c.candidate_id, c.raw_joint_metric_evidence.content_hash if c.raw_joint_metric_evidence else None,
                               sorted(c.evidence_refs), sorted(c.metrics.items()), sorted(c.dimensions.items()), c.coverage,
                               c.fidelity, c.qualification_scope,c.qualification_ref,
                               sorted((c.qualification_requirements or {}).items()),c.health_evidence_ref,
                               c.health_candidate_id,c.health_coverage,c.use_admission_ref,c.evidence_bundle_ref) for c in self.candidates))
    @property
    def content_hash(self):
        return _digest({"policy":self.policy_id,"policy_hash":self.policy_content_hash,"context":self.comparison_context_hash,"purpose":self.purpose,"level":self.decision_level,"baseline":self.baseline_ref,"original_raw":self.original_raw_ref,"candidate_set":self.candidate_set_hash,"fidelity":self.required_final_fidelity,"family":self.hypothesis_family_ref})

@dataclass(frozen=True)
class GateReceipt:
    candidate_id: str; gate_id: str; passed: bool; reason: str

@dataclass(frozen=True)
class DecisionArtifact:
    request_id: str; request_hash: str; policy_id: str; policy_content_hash: str; comparison_context_hash: str; candidate_set_hash: str
    decision_id: str; content_hash: str; status: DecisionStatus; eligibility: Mapping[str,bool]; gate_receipts: tuple[GateReceipt,...]
    point_utility: Mapping[str,Optional[float]]; conservative_utility: Mapping[str,Optional[float]]; relationship: Mapping[str,Relationship]
    effect_refs: Mapping[str,Optional[str]]; qualification_scope: Mapping[str,Optional[str]]; reasons: tuple[str,...]
    replacement_role: Mapping[str,Optional[str]]; winner_id: Optional[str]; retained_ids: tuple[str,...]; final_fidelity: str
    qualification_identities: Mapping[str,Optional[str]]
    @property
    def eligible_ids(self):
        return tuple(sorted(k for k,v in self.eligibility.items() if v))

class DecisionProvider:
    _GRADES={"NONE":-1,"D":0,"C":1,"B":2,"B+":3,"A":4,"A+":5,"S":6,"S+":7}
    _PURPOSES={"RESEARCH","VARIANT","VARIANT-SELECTION","REPRESENTATION","DIAGNOSTIC","PRODUCTION","LIVE","DEPLOY","PUBLISH","PRODUCTION_TRADING"}
    def __init__(self, policy, qualification_resolver=None, evidence_resolver=None):
        self.policy=policy
        self.qualification_resolver=qualification_resolver
        self.evidence_resolver=evidence_resolver or (qualification_resolver if hasattr(qualification_resolver,"resolve_evidence") else None)
    def decide(self, request):
        if request.policy_id!=self.policy.policy_id or request.policy_content_hash!=self.policy.content_hash: raise ValueError("request policy binding mismatch")
        if request.purpose.upper() not in self._PURPOSES: raise ValueError("unknown decision purpose")
        receipts=[]; eligible={}; point={}; conservative={}; mapped_evidence={}; qualification_ids={}
        for c in request.candidates:
            failures=[]
            if c.fidelity!=request.required_final_fidelity: failures.append("final_fidelity")
            if not c.evidence_refs: failures.append("evidence")
            if not c.evidence_bundle_ref or self.evidence_resolver is None:
                failures.append("trusted_evidence")
            else:
                try:
                    trusted=self.evidence_resolver.resolve_evidence(c.evidence_bundle_ref,candidate=c)
                    raw=c.raw_joint_metric_evidence
                    expected=(c.candidate_id,raw.content_hash if raw else None,raw.recipe_hash if raw else None,
                        raw.fitted_state_hash if raw else None,raw.data_snapshot_hash if raw else None,
                        raw.universe_hash if raw else None,raw.label_hash if raw else None,
                        raw.value_artifact_hash if raw else None)
                    actual=(trusted.candidate_id,trusted.raw_content_hash,trusted.recipe_hash,
                        trusted.fitted_state_hash,trusted.data_snapshot_hash,trusted.universe_hash,
                        trusted.label_hash,trusted.value_artifact_hash)
                    if (actual!=expected or dict(trusted.dimensions)!=dict(c.dimensions)
                            or not math.isclose(trusted.coverage,c.coverage,abs_tol=1e-12)
                            or trusted.metric_instance_hash!=raw.metric_instance_hash
                            or trusted.parameter_domain_hash!=raw.parameter_domain_hash
                            or dict(trusted.metric_directions)!={r.metric_id:r.direction.value for r in self.policy.metric_rules if r.metric_id in raw.metric_ids}):
                        failures.append("trusted_evidence_binding")
                except (KeyError,ValueError,TypeError,AttributeError):
                    failures.append("trusted_evidence")
            if not c.qualification_scope: failures.append("qualification_scope")
            if not c.qualification_ref or c.raw_joint_metric_evidence is None or self.qualification_resolver is None:
                failures.append("numerical_qualification")
            else:
                try:
                    receipt=self.qualification_resolver.resolve(c.qualification_ref)
                    raw=c.raw_joint_metric_evidence
                    receipt.require_scope(source_tree_hash=raw.source_tree_hash,
                        implementation_hash=raw.implementation_hash,route=raw.route,
                        backend=raw.backend,parameter_domain_hash=raw.parameter_domain_hash,
                        metric_instance_hash=raw.metric_instance_hash)
                    receipt.require_assertion_suite(suite_id=self.policy.qualification_suite_id,
                        required_assertions=self.policy.required_qualification_assertions)
                    identities=[getattr(receipt,"content_hash",c.qualification_ref)]
                    if self._requires_use_admission(request.purpose):
                        if not c.use_admission_ref: raise ValueError("purpose-scoped use admission required")
                        admission=self.qualification_resolver.resolve(c.use_admission_ref)
                        admission.require_use_admission(purpose=request.purpose,candidate_id=c.candidate_id,
                            evidence_bundle_ref=c.evidence_bundle_ref,
                            current_time=self.qualification_resolver.current_time,
                            current_revocation_epoch=self.qualification_resolver.current_revocation_epoch)
                        identities.append(getattr(admission,"content_hash",c.use_admission_ref))
                    qualification_ids[c.candidate_id]=_digest(identities)
                except (KeyError,ValueError,TypeError,AttributeError):
                    failures.append("numerical_qualification")
            if c.raw_joint_metric_evidence is None: failures.append("raw_joint_metric_evidence")
            elif c.raw_joint_metric_evidence.candidate_id != c.candidate_id: failures.append("candidate_identity")
            elif c.raw_joint_metric_evidence.comparison_context_hash != request.comparison_context_hash: failures.append("joint_context")
            elif (c.raw_joint_metric_evidence.window_ids != self.policy.window_ids
                  or c.raw_joint_metric_evidence.scenario_ids != self.policy.scenario_ids): failures.append("joint_named_axes")
            if c.raw_joint_metric_evidence is not None:
                raw_units=dict(zip(c.raw_joint_metric_evidence.metric_ids,c.raw_joint_metric_evidence.metric_units))
                allowed={r.metric_id for r in self.policy.metric_rules}
                required={r.metric_id for r in self.policy.metric_rules if r.required or r.weight>0}
                if not required.issubset(raw_units) or not set(raw_units).issubset(allowed): failures.append("joint_metric_axes")
                for rule in self.policy.metric_rules:
                    if rule.metric_id in raw_units and raw_units[rule.metric_id]!=rule.unit:
                        failures.append(f"metric:{rule.metric_id}:unit")
            if c.coverage<self.policy.minimum_coverage: failures.append("coverage")
            if self.policy.required_dimensions or self.policy.minimum_coverage:
                if (not c.health_evidence_ref or c.health_candidate_id!=c.candidate_id
                        or c.health_coverage is None or not math.isclose(c.coverage,c.health_coverage,abs_tol=1e-12)):
                    failures.append("health_evidence_identity")
            for d in self.policy.required_dimensions:
                if c.dimensions.get(d) in (None,"NONE","NOT_APPLICABLE","NOT_READY"): failures.append(f"dimension:{d}:missing")
            for d,floor in self.policy.dimension_floors.items():
                if self._GRADES.get(c.dimensions.get(d,"NONE"),-2)<self._GRADES.get(floor,99): failures.append(f"dimension:{d}:below_floor")
            for rule in self.policy.metric_rules:
                value=None
                raw=c.raw_joint_metric_evidence
                if raw is not None and rule.metric_id in raw.metric_ids:
                    idx=raw.metric_ids.index(rule.metric_id)
                    values=[scenario[idx] for rep in raw.samples for window in rep for scenario in window]
                    value=sum(values)/len(values)
                if value is None:
                    if rule.required: failures.append(f"metric:{rule.metric_id}:missing")
                    continue
                if rule.hard_floor is not None and value<rule.hard_floor: failures.append(f"metric:{rule.metric_id}:floor")
                if rule.hard_ceiling is not None and value>rule.hard_ceiling: failures.append(f"metric:{rule.metric_id}:ceiling")
            eligible[c.candidate_id]=not failures
            receipts.extend(GateReceipt(c.candidate_id,x,False,x) for x in failures)
            receipts.append(GateReceipt(c.candidate_id,"eligibility",not failures,"passed" if not failures else "hard gate failure"))
            integrity_failures={"evidence","trusted_evidence","trusted_evidence_binding","qualification_scope","numerical_qualification","joint_context","joint_named_axes","joint_metric_axes","raw_joint_metric_evidence","candidate_identity","final_fidelity","health_evidence_identity"}
            if c.raw_joint_metric_evidence is not None and not integrity_failures.intersection(failures) and not any(x.endswith(":unit") for x in failures):
                mapped_evidence[c.candidate_id]=self._map_raw_evidence(c.raw_joint_metric_evidence)
            if failures: point[c.candidate_id]=conservative[c.candidate_id]=None; continue
            mapped=mapped_evidence[c.candidate_id]
            point[c.candidate_id],conservative[c.candidate_id]=mapped.fitness()
        selected=[c for c in request.candidates if eligible[c.candidate_id]]
        baseline=next((c for c in request.candidates if c.candidate_id==request.baseline_ref),None)
        relation={}; effects={}; replacement_roles={}
        for c in request.candidates:
            effects[c.candidate_id]=None
            rel=Relationship.INCONCLUSIVE
            if (eligible.get(c.candidate_id,False) and baseline and c.candidate_id!=baseline.candidate_id and c.candidate_id in mapped_evidence
                    and baseline.candidate_id in mapped_evidence):
                left,right=mapped_evidence[c.candidate_id],mapped_evidence[baseline.candidate_id]
                if (set(left.replicate_ids) != set(right.replicate_ids) or left.resampling_plan_ref != right.resampling_plan_ref
                        or self._pairing_identity(c.raw_joint_metric_evidence)!=self._pairing_identity(baseline.raw_joint_metric_evidence)
                        or left.window_ids != right.window_ids or left.scenario_ids != right.scenario_ids):
                    # Individually valid evidence from another sample/plan is
                    # incomparable, not an exception that aborts valid peers.
                    receipts.append(GateReceipt(c.candidate_id, "paired_comparison", False,
                        "paired candidates require identical data, plan, samples and axes"))
                    relation[c.candidate_id] = Relationship.INCONCLUSIVE
                    replacement_roles[c.candidate_id] = None
                    continue
                left_scores=dict(zip(left.replicate_ids,left.replicate_scores()))
                right_scores=dict(zip(right.replicate_ids,right.replicate_scores()))
                differences=sorted(left_scores[rep]-right_scores[rep] for rep in left.replicate_ids)
                lo=_quantile(differences,.05)
                hi=_quantile(differences,.95)
                effects[c.candidate_id]=_digest({"candidate":left.content_hash,"baseline":right.content_hash,"effect_estimand":self.policy.effect_estimand,"replicate_differences":differences,"interval_quantiles":(.05,.95)})
                if lo>self.policy.minimum_gain: rel=Relationship.REPAIRED if not eligible.get(baseline.candidate_id,False) else Relationship.SUPERIOR
                elif lo>=-self.policy.equivalence_margin and hi<=self.policy.equivalence_margin: rel=Relationship.EQUIVALENT
                elif lo>=-self.policy.noninferiority_margin: rel=Relationship.NONINFERIOR
            relation[c.candidate_id]=rel
            replacement_roles[c.candidate_id]=self._replacement_role(c,baseline,rel) if baseline else None
        # Utility ranks eligible candidates, but cannot itself authorize a
        # replacement.  Retain a legal baseline unless paired evidence proves
        # superiority; a repaired candidate may replace only an illegal RAW.
        if baseline is not None and eligible.get(baseline.candidate_id,False):
            raw=next((c for c in request.candidates if c.candidate_id==(request.original_raw_ref or request.baseline_ref)),baseline)
            qualified=[c for c in selected if (relation.get(c.candidate_id) is Relationship.SUPERIOR or replacement_roles.get(c.candidate_id) is not None) and all(x.candidate_id in mapped_evidence for x in (baseline,raw)) and self._passes_raw_guards(c,(baseline,raw))]
            winner=max(qualified,key=lambda c:(conservative[c.candidate_id],c.candidate_id)) if qualified else baseline
        elif baseline is not None:
            raw=next((c for c in request.candidates if c.candidate_id==(request.original_raw_ref or request.baseline_ref)),baseline)
            qualified=[c for c in selected if relation.get(c.candidate_id) in (Relationship.SUPERIOR,Relationship.REPAIRED) and all(x.candidate_id in mapped_evidence for x in (baseline,raw)) and self._passes_raw_guards(c,(baseline,raw))]
            winner=max(qualified,key=lambda c:(conservative[c.candidate_id],c.candidate_id)) if qualified else None
        else:
            winner=max(selected,key=lambda c:(conservative[c.candidate_id],c.candidate_id)) if selected else None
        waiting=any(any(r.candidate_id==c.candidate_id and r.gate_id in ("evidence","trusted_evidence","trusted_evidence_binding","numerical_qualification","raw_joint_metric_evidence") for r in receipts) for c in request.candidates)
        status=DecisionStatus.SELECTED if winner else (DecisionStatus.WAIT if waiting else DecisionStatus.INCOMPARABLE)
        semantic={"request":request.content_hash,"policy":self.policy.content_hash,"context":request.comparison_context_hash,"candidates":request.candidate_set_hash,"status":status.value,"winner":winner.candidate_id if winner else None,"eligible":dict(sorted(eligible.items())),"point":dict(sorted(point.items())),"conservative":dict(sorted(conservative.items())),"relationship":{k:v.value for k,v in sorted(relation.items())},"replacement_role":dict(sorted(replacement_roles.items())),"qualification_identities":dict(sorted(qualification_ids.items()))}
        content_hash=_digest(semantic)
        return DecisionArtifact(request.request_id,request.content_hash,self.policy.policy_id,self.policy.content_hash,request.comparison_context_hash,request.candidate_set_hash,"decision:"+content_hash,content_hash,status,MappingProxyType(eligible),tuple(receipts),MappingProxyType(point),MappingProxyType(conservative),MappingProxyType(relation),MappingProxyType(effects),MappingProxyType({c.candidate_id:c.qualification_scope for c in request.candidates}),tuple(r.reason for r in receipts if not r.passed),MappingProxyType(replacement_roles),winner.candidate_id if winner else None,((winner.candidate_id,) if winner else ()),request.required_final_fidelity,MappingProxyType(qualification_ids))

    @staticmethod
    def _requires_use_admission(purpose):
        return purpose.upper() in {"PRODUCTION","LIVE","DEPLOY","PUBLISH","PRODUCTION_TRADING"}

    def _map_raw_evidence(self, evidence):
        positions={name:i for i,name in enumerate(evidence.metric_ids)}
        allowed={r.metric_id for r in self.policy.metric_rules}; required={r.metric_id for r in self.policy.metric_rules if r.required or r.weight>0}
        if not required.issubset(positions) or not set(positions).issubset(allowed): raise ValueError("raw metric axes do not match policy")
        if evidence.window_ids!=self.policy.window_ids or evidence.scenario_ids!=self.policy.scenario_ids: raise ValueError("raw evidence named axes do not match policy")
        blocks=tuple(self.policy.block_weights)
        samples=[]
        for rep in evidence.samples:
            windows=[]
            for window in rep:
                scenarios=[]
                for raw in window:
                    block_values=[]
                    for block in blocks:
                        rules=[r for r in self.policy.metric_rules if r.block_id==block and r.weight>0]
                        weights=sum(r.weight for r in rules); values=[]
                        for rule in rules:
                            value=raw[positions[rule.metric_id]]
                            scaled=(value-rule.bad)/(rule.good-rule.bad) if rule.direction is UtilityDirection.HIGHER_IS_BETTER else (rule.bad-value)/(rule.bad-rule.good)
                            values.append((rule,min(1.,max(0.,scaled))))
                        arithmetic=sum(r.weight*v for r,v in values)/weights
                        geometric=0. if any(v==0 for _,v in values) else math.exp(sum((r.weight/weights)*math.log(v) for r,v in values))
                        block_values.append(self.policy.balanced_arithmetic_weight*arithmetic+(1-self.policy.balanced_arithmetic_weight)*geometric)
                    scenarios.append(tuple(block_values))
                windows.append(tuple(scenarios))
            samples.append(tuple(windows))
        coefficients=(self.policy.balanced_arithmetic_weight,self.policy.scenario_mean_weight,self.policy.window_mean_weight,self.policy.lower_tail_mass,self.policy.conservative_quantile)
        semantic={"context":evidence.comparison_context_hash,"plan":evidence.resampling_plan_ref,"replicates":evidence.replicate_ids,"blocks":blocks,"block_weights":tuple(self.policy.block_weights[b] for b in blocks),"window_ids":evidence.window_ids,"window_weights":self.policy.window_weights,"scenario_ids":evidence.scenario_ids,"scenario_weights":self.policy.scenario_weights,"samples":json.loads(json.dumps(samples)),"qualification":evidence.qualification_scope,"coefficients":coefficients}
        return JointUtilityEvidence(evidence.evidence_id+":mapped",_digest(semantic),evidence.comparison_context_hash,evidence.resampling_plan_ref,evidence.replicate_ids,blocks,tuple(self.policy.block_weights[b] for b in blocks),self.policy.window_weights,self.policy.scenario_weights,tuple(samples),evidence.qualification_scope,window_ids=evidence.window_ids,scenario_ids=evidence.scenario_ids,balanced_arithmetic_weight=coefficients[0],scenario_mean_weight=coefficients[1],window_mean_weight=coefficients[2],lower_tail_mass=coefficients[3],conservative_quantile=coefficients[4])

    def _passes_raw_guards(self, candidate, references):
        left=candidate.raw_joint_metric_evidence
        if left is None: return False
        positions={m:i for i,m in enumerate(left.metric_ids)}
        for reference in references:
            right=reference.raw_joint_metric_evidence
            if right is None or left.comparison_context_hash!=right.comparison_context_hash or left.resampling_plan_ref!=right.resampling_plan_ref or self._pairing_identity(left)!=self._pairing_identity(right) or set(left.replicate_ids)!=set(right.replicate_ids) or (left.window_ids,left.scenario_ids)!=(right.window_ids,right.scenario_ids): return False
            if left.metric_ids!=right.metric_ids or left.metric_units!=right.metric_units: return False
            right_replicates=dict(zip(right.replicate_ids,right.samples))
            for rule in self.policy.metric_rules:
                if rule.metric_id not in positions: continue
                idx=positions[rule.metric_id]
                if left.metric_units[idx]!=rule.unit: return False
                diffs=[]
                for replicate_id,lr in zip(left.replicate_ids,left.samples):
                    rr=right_replicates[replicate_id]
                    effect=0.
                    for window_weight,lw,rw in zip(self.policy.window_weights,lr,rr):
                        for scenario_weight,ls,rs in zip(self.policy.scenario_weights,lw,rw):
                            delta=ls[idx]-rs[idx]
                            effect += window_weight*scenario_weight*(delta if rule.direction is UtilityDirection.HIGHER_IS_BETTER else -delta)
                    diffs.append(effect)
                if _quantile(diffs,.05) < -rule.noninferiority_delta: return False
        return True

    def _replacement_role(self, candidate, baseline, relationship):
        if relationship not in (Relationship.NONINFERIOR,Relationship.EQUIVALENT): return None
        left,right=candidate.raw_joint_metric_evidence,baseline.raw_joint_metric_evidence
        if left is None or right is None or left.metric_ids!=right.metric_ids: return None
        for role in self.policy.replacement_role_rules:
            idx=left.metric_ids.index(role.metric_id); diffs=[]
            rule=next(r for r in self.policy.metric_rules if r.metric_id==role.metric_id)
            if left.metric_units[idx]!=rule.unit or right.metric_units[idx]!=rule.unit: continue
            if (left.comparison_context_hash!=right.comparison_context_hash
                    or left.resampling_plan_ref!=right.resampling_plan_ref
                    or self._pairing_identity(left)!=self._pairing_identity(right)
                    or set(left.replicate_ids)!=set(right.replicate_ids)): continue
            right_replicates=dict(zip(right.replicate_ids,right.samples))
            for replicate_id,lr in zip(left.replicate_ids,left.samples):
                rr=right_replicates[replicate_id]; effect=0.
                for window_weight,lw,rw in zip(self.policy.window_weights,lr,rr):
                    for scenario_weight,ls,rs in zip(self.policy.scenario_weights,lw,rw):
                        delta=ls[idx]-rs[idx]
                        effect += window_weight*scenario_weight*(delta if role.direction is UtilityDirection.HIGHER_IS_BETTER else -delta)
                diffs.append(effect)
            if _quantile(diffs,.05)>=role.minimum_improvement: return role.role
        return None

    @staticmethod
    def _pairing_identity(raw):
        return (raw.resampling_plan_content_hash,raw.sample_identity_hash,raw.time_identity_hash,
            raw.common_mask_hash,raw.data_snapshot_hash,raw.universe_hash,raw.label_hash)
