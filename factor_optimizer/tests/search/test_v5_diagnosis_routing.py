import pytest
import numpy as np
import pandas as pd

from factor_optimizer.search.diagnosis_routing import (RecipeKind, RoutingPolicy,
    compile_portfolio_recipe, execute_routed_portfolio_trial, route_diagnoses)
from factor_optimizer.search.runner import SearchRunner
from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.search.runner import SearchConfig
from factor_optimizer.contracts.treatment_integrity import build_integrity_evidence


def test_runner_portfolio_route_reaches_real_planner_ledger_and_public_qe():
    # Load the public execution module without importing the optional report UI
    # from engine.__init__ (the server test venv intentionally has no plotly).
    import importlib, sys, types
    engine_pkg = types.ModuleType("vectorbt_qs.mvp.engine")
    engine_pkg.__path__ = [str(__import__("pathlib").Path(__file__).parents[3] / "vectorbt_qs" / "mvp" / "engine")]
    sys.modules.setdefault("vectorbt_qs.mvp.engine", engine_pkg)
    from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact
    from quant_evaluator.api.requests import EvaluationRequest
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate as qe_evaluate
    execution = importlib.import_module("vectorbt_qs.mvp.engine.execution")
    ExecutionCosts, plan_ashare_orders = execution.ExecutionCosts, execution.plan_ashare_orders
    from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_trajectory_from_filled_ledger
    dates = pd.bdate_range("2026-01-05", periods=8); columns = ["A", "B", "C"]
    ranks = np.array([[.1,.5,.9],[.3,.1,.8],[.4,.2,.7],[.1,.6,.9],
                      [.3,.7,.2],[.8,.1,.4],[.2,.9,.3],[.7,.4,.1]])
    close = pd.DataFrame(10*np.cumprod(1+np.array([[.00,.002,.01],[.001,.002,.008],
        [.002,.001,.006],[-.001,.002,.01],[.001,.003,-.005],[.004,-.002,.001],
        [-.001,.006,.002],[.005,.001,-.003]]),axis=0),index=dates,columns=columns)
    zeros = pd.DataFrame(False,index=dates,columns=columns)
    high, low = close*1.2, close*.8
    outcomes = {}
    def economic_consumer(trial, target_weights, fidelity):
        targets = pd.DataFrame(target_weights,index=dates,columns=columns)
        plan = plan_ashare_orders(close,close,targets,zeros,high,low,init_cash=100_000.,
            costs=ExecutionCosts(minimum_commission=0.),lot_size=1,planner_engine="python")
        trajectory = build_trajectory_from_filled_ledger(plan=plan,order_price=close,
            valuation_price=close,initial_nav=100_000.,slippage=.0005,scenario_id=trial.trial_id,
            refs=TrajectoryRefs((trial.trial_id,),"snapshot:routing","portfolio:"+trial.trial_id,
                "cost:v5","benchmark:zero"),benchmark_return=(0.,)*len(dates))
        probe = trajectory_to_probe_artifact(trajectory,expected_portfolio_profile="LONG_ONLY_RESEARCH",
                                             expected_cost_profile="net-base")
        batch = FactorBatch((trial.trial_id,),AxisRef("time","date",len(dates)),AxisRef("asset","str",3),ranks[...,None])
        date_strings=tuple(str(x.date()) for x in dates); ends=tuple(str((x+pd.Timedelta(days=1)).date()) for x in dates)
        labels=LabelBundle("h1",ranks,1,decision_time=date_strings,label_start_time=date_strings,label_end_time=ends)
        bundle=qe_evaluate(EvaluationRequest(batch,labels,metric_ids=("sharpe_ratio",),tier="extended",
                                             portfolio_returns=probe,metric_parameters={"sharpe_ratio":{"min_periods":2}}))
        score=bundle.get_metric("sharpe_ratio",trial.trial_id).value; outcomes[trial.metadata["treatment"]]=score
        integrity=build_integrity_evidence(trial.trial_id,trial.metadata["treatment"],{},[1.],[1.])
        return {"evaluation_id":bundle.request_id,"score":score,"cost":1.,"treatment_integrity_evidence":integrity}
    def raw_evaluate(trial,fidelity):
        weights=np.full_like(ranks,1/3)
        return economic_consumer(trial,weights,fidelity)
    protocol=EvaluationProtocol(SplitPlan("route",[True],[False],[False],{}),raw_evaluate)
    planned=SearchRunner.plan_diagnosis_trials("f",["BOTTOM_EXCLUSION"])
    config=SearchConfig(budget=SearchBudget(max_trials=2,max_evaluations=2,max_cost_units=2),enable_multifidelity=False)
    SearchRunner.for_diagnoses(config,"f",["BOTTOM_EXCLUSION"],protocol,
        portfolio_recipe_context={"signal_ranks":ranks,"consumer":economic_consumer}).run("real-route")
    assert set(outcomes)=={"raw","bottom_exclusion"}
    assert outcomes["raw"] != outcomes["bottom_exclusion"]


def test_routes_are_bounded_raw_retaining_and_not_cartesian():
    trials = SearchRunner.plan_diagnosis_trials(
        "f", ["HIGH_TURNOVER", "STYLE_EXPOSURE", "SYMMETRIC_U"]
    )
    assert trials[0].is_raw_baseline
    assert len(trials) - 1 <= 8
    assert len({t.trial_id for t in trials}) == len(trials)
    assert all("+" not in t.treatment for t in trials)


def test_portfolio_recipe_is_not_factor_recipe_and_expansion_is_explicit():
    trials = route_diagnoses("f", ["BOTTOM_EXCLUSION", "HIGH_TURNOVER"], explicit_budget=12)
    bottom = next(t for t in trials if t.treatment == "bottom_exclusion")
    assert bottom.recipe_kind is RecipeKind.PORTFOLIO
    with pytest.raises(ValueError):
        route_diagnoses("f", ["HIGH_TURNOVER"], explicit_budget=13)
    ranks = [[.95, .85, .1], [.82, .91, .2], [.79, .88, .3]]
    weights = compile_portfolio_recipe("buy_hold_buffer").execute(ranks)
    assert weights.shape == (3, 3)
    assert weights[1, 0] > 0  # held through the frozen 90%/80% buffer
    assert weights[2, 0] == 0


def test_routing_identity_binds_diagnoses_policy_and_effective_budget():
    base = route_diagnoses("f", ["HIGH_TURNOVER"])
    changed_policy = route_diagnoses("f", ["HIGH_TURNOVER"],
                                     policy=RoutingPolicy(policy_version="V5.1"))
    changed_diagnosis = route_diagnoses("f", ["HIGH_TURNOVER", "BOTTOM_EXCLUSION"])
    changed_budget = route_diagnoses("f", ["HIGH_TURNOVER"], explicit_budget=12)
    assert len({x[0].routing_identity for x in (base, changed_policy, changed_diagnosis, changed_budget)}) == 4
    assert all(t.routing_identity == base[0].routing_identity for t in base)


def test_public_runner_executes_real_routed_trials_with_recipe_identity():
    planned = SearchRunner.plan_diagnosis_trials("f", ["BOTTOM_EXCLUSION"])
    config = SearchConfig(budget=SearchBudget(
        max_trials=len(planned), max_evaluations=len(planned), max_cost_units=10
    ), enable_multifidelity=False)

    def evaluate(trial, fidelity):
        raw = trial.metadata["is_raw_baseline"]
        integrity = build_integrity_evidence(
            trial.trial_id, "raw" if raw else trial.metadata["treatment"],
            {} if raw else {"enabled": True}, [1.0], [1.0] if raw else [2.0],
        )
        return {"evaluation_id": "qe:" + trial.trial_id, "score": 1.0,
                "cost": 1.0, "treatment_integrity_evidence": integrity}

    protocol = EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), evaluate
    )
    consumed = []
    def consume(trial, target_weights, fidelity):
        consumed.append(target_weights.copy())
        assert target_weights.shape == (2, 3)
        return evaluate(trial, fidelity)
    session = SearchRunner.for_diagnoses(
        config, "f", ["BOTTOM_EXCLUSION"], protocol,
        portfolio_recipe_context={"signal_ranks": np.array([[.1,.5,.9],[.3,.1,.8]]),
                                  "consumer": consume},
    ).run("v5-real-route")
    assert len(session.successful_trials()) == len(planned)
    assert {t.metadata["recipe_kind"] for t in session.successful_trials()} == {
        "FACTOR_RECIPE", "PORTFOLIO_RECIPE"
    }
    assert len(consumed) == 1 and consumed[0][0, 0] == 0


def test_unknown_diagnosis_and_mutable_recipe_parameters_fail_closed():
    with pytest.raises(ValueError, match="unknown diagnoses"):
        route_diagnoses("f", ["TYPO_DIAGNOSIS"])
    params = [["minimum_rank", .2]]
    recipe = compile_portfolio_recipe("bottom_exclusion")
    assert isinstance(recipe.parameters, tuple)
    trial = route_diagnoses("f", ["BOTTOM_EXCLUSION"])[1].to_trial()
    trial.metadata["portfolio_recipe_ref"] = "portfolio-recipe:wrong"
    with pytest.raises(ValueError, match="does not match"):
        execute_routed_portfolio_trial(trial, [[.1, .9]], lambda *_: {}, 1)
