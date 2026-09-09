from dataclasses import replace
import json
import numpy as np
import pytest

from quant_evaluator.api.requests import EvaluationRequest, EvaluationBundle
from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.tests.test_v3_public_artifacts import inputs


def test_aliases_and_explicit_defaults_have_same_instance_identity():
    from quant_evaluator.registry.metrics import CANONICAL_METRIC_ALIASES
    alias = next(k for k,v in CANONICAL_METRIC_ALIASES.items() if v == "rank_ic")
    item = MetricInstance("rank_ic")
    assert MetricInstance(alias).instance_id == item.instance_id
    assert MetricInstance("rank_ic", parameters=dict(item.parameters)).instance_id == item.instance_id
    assert MetricInstance("rank_ic", horizon=20).instance_id != item.instance_id
    assert MetricInstance.from_dict(json.loads(json.dumps(item.to_dict()))) == item


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_public_q10_q20_h10_h20_coexist_and_roundtrip(backend):
    batch, label = inputs()
    h10 = replace(label, horizon=10, label_end_time=tuple(i+11 for i in range(30)))
    h20 = replace(label, horizon=20, label_end_time=tuple(i+21 for i in range(30)), values=-label.values)
    items = tuple(MetricInstance("quantile_returns_full", parameters={"n_quantiles": q},
                                 scenario_id=f"H{h}", horizon=h)
                  for h in (10, 20) for q in (10, 20))
    request = EvaluationRequest(batch, h10, metric_ids=(), metric_instances=items, tier="research",
                                scenario_inputs={"H10": EvaluationScenario(h10), "H20": EvaluationScenario(h20)})
    transport = EvaluationRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert [x.instance_id for x in transport.metric_instances] == [x.instance_id for x in items]
    assert not transport.scenario_inputs
    result = evaluate(request, backend=backend)
    assert len(result.instance_results) == 4
    assert len({child.config_hash for child in result.instance_results.values()}) == 4
    for item in items:
        child = result.instance_results[result.metadata["requested_to_resolved_instances"][item.instance_id]]
        assert child.artifacts[item.metric_id].values.shape == (item.parameters["n_quantiles"], 2)
        expected = evaluate(batch, h10 if item.horizon == 10 else h20,
                            metrics=[item.metric_id], metric_parameters={item.metric_id: dict(item.parameters)}, backend=backend)
        np.testing.assert_allclose(child.artifacts[item.metric_id].values,
                                   expected.artifacts[item.metric_id].values, equal_nan=True)
    restored = EvaluationBundle.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.config_hash == result.config_hash
    assert set(restored.instance_results) == set(result.instance_results)


def test_missing_bindings_and_lying_horizons_fail_closed():
    batch, label = inputs()
    with pytest.raises(ValueError, match="horizon"):
        evaluate(EvaluationRequest(batch, label, metric_instances=(MetricInstance("rank_ic", horizon=20),)))
    with pytest.raises(ValueError, match="Missing runtime"):
        evaluate(EvaluationRequest(batch, label, metric_instances=(MetricInstance("rank_ic", scenario_id="missing"),)))
    with pytest.raises(ValueError, match="trajectory"):
        EvaluationScenario(label, cost_profile="net-base")


def test_budget_admits_whole_request_and_identical_aliases_deduplicate():
    batch, label = inputs()
    item = MetricInstance("rank_ic")
    result = evaluate(EvaluationRequest(batch, label, metric_instances=(item, item)))
    assert len(result.instance_results) == 1
    with pytest.raises(ValueError, match="cost_budget"):
        evaluate(EvaluationRequest(batch, label, metric_instances=(item,), cost_budget=0))
    tampered = result.to_dict()
    key = next(iter(tampered["instance_specs"]))
    tampered["instance_specs"][key]["horizon"] = 500
    with pytest.raises(ValueError, match="identity"):
        EvaluationBundle.from_dict(tampered)


def test_t01_t02_all_h_q_cost_variants_and_third_factor_remain_independent():
    from hashlib import sha256
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from vectorbt_qs.contracts.costs import CostScope
    from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_research_trajectory
    rng = np.random.default_rng(57)
    t,n=30,80
    x=rng.normal(size=(t,n))
    values=np.stack([x,-x,rng.normal(size=(t,n))],axis=-1)
    dates=tuple(str(np.datetime64("2024-01-01")+i) for i in range(t))
    def label(h):
        return LabelBundle(f"forward-H{h}",x if h==10 else -x,h,decision_time=dates,
            label_start_time=dates,label_end_time=tuple(str(np.datetime64(d)+h) for d in dates))
    labels={h:label(h) for h in (10,20)}
    gross=np.column_stack([np.tile([.02,-.005],15),np.tile([-.02,.005],15),rng.normal(0,.01,t)])
    factor_ids=("f","minus_f","unrelated")
    scenarios={}
    instances=[]
    for h in (10,20):
        sid=f"diagnostic-H{h}"
        scenarios[sid]=EvaluationScenario(labels[h])
        instances.extend(MetricInstance("quantile_returns_full",parameters={"n_quantiles":q},
                        scenario_id=sid,horizon=h) for q in (10,20))
        instances.append(MetricInstance("rank_ic",scenario_id=sid,horizon=h))
    for sid,scope,cost_profile,cost in (("gross",CostScope.GROSS_DIAGNOSTIC,"gross",0.),
                                      ("base",CostScope.NET_ASSUMED,"net-base",.001),
                                      ("stress",CostScope.NET_ASSUMED,"net-stress",.004)):
        trajectories={}
        for f,fid in enumerate(factor_ids):
            refs=TrajectoryRefs((fid,),sha256(gross[:,f].tobytes()).hexdigest(),f"synthetic-portfolio:{fid}",
                               f"fixed-cost:{cost}","zero-benchmark")
            trajectories[fid]=build_research_trajectory(scenario_id=sid,profile="LONG_ONLY_RESEARCH",
                dates=dates,gross_return=gross[:,f],benchmark_return=np.zeros(t),
                cost_contributions={"execution":np.full(t,cost)},refs=refs,scope=scope)
        scenarios[sid]=EvaluationScenario(labels[10],cost_profile=cost_profile,
            portfolio_profile="LONG_ONLY_RESEARCH",trajectory=trajectories)
        instances.append(MetricInstance("sharpe_ratio",scenario_id=sid,cost_profile=cost_profile,
                         portfolio_profile="LONG_ONLY_RESEARCH",horizon=10))
    def run(count):
        batch=FactorBatch(factor_ids[:count],AxisRef("time","date",t),AxisRef("asset","int",n),values[:,:,:count])
        bound={key: EvaluationScenario(value.label_bundle,cost_profile=value.cost_profile,
                portfolio_profile=value.portfolio_profile,trajectory={fid:value.trajectory[fid] for fid in factor_ids[:count]})
               if value.trajectory is not None else value for key,value in scenarios.items()}
        return evaluate(EvaluationRequest(batch,labels[10],metric_instances=tuple(instances),
                        scenario_inputs=bound,tier="research"))
    two,three=run(2),run(3)
    assert len(three.instance_results)==9
    assert len({child.config_hash for child in three.instance_results.values()})==9
    for key,child in two.instance_results.items():
        mid=two.instance_specs[key].metric_id
        np.testing.assert_allclose(child.artifacts[mid].values,
            three.instance_results[key].artifacts[mid].values[...,:2],equal_nan=True)
    by_scenario={item.scenario_id:three.instance_results[key] for key,item in three.instance_specs.items()
                 if item.metric_id=="sharpe_ratio"}
    assert by_scenario["gross"].get_metric("sharpe_ratio","f").value > 0
    assert by_scenario["gross"].get_metric("sharpe_ratio","minus_f").value < 0
    for fid in factor_ids:
        assert by_scenario["gross"].get_metric("sharpe_ratio",fid).value > by_scenario["base"].get_metric("sharpe_ratio",fid).value
        assert by_scenario["base"].get_metric("sharpe_ratio",fid).value > by_scenario["stress"].get_metric("sharpe_ratio",fid).value
    restored=EvaluationBundle.from_dict(json.loads(json.dumps(three.to_dict())))
    assert set(restored.instance_results)==set(three.instance_results)
