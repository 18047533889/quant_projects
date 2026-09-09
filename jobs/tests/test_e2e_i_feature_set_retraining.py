import pytest

from jobs.e2e_i_feature_set_retraining import compatibility_from_set_trial
from modeling.feature_set_trials import FeatureSetAction
import numpy as np
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
from quant_evaluator.runtime.evaluator import evaluate
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import TrajectoryRefs, build_research_trajectory


def test_n14_rejects_duck_typed_or_unresolved_set_trial():
    action = FeatureSetAction("add", "ADD", ("a", "b"), "features@v2#hash")
    class Trial:
        accepted=True; evidence_ref="trial:evidence"; model_refs=("model:fold1",)
        baseline_model_refs=("model:base",); oof_row_ids=("row:1",)
    Trial.action = action
    with pytest.raises(TypeError, match="FeatureSetTrialEvidence"):
        compatibility_from_set_trial(Trial(), model_artifact_ref="model:replacement")


def test_n14_public_evaluate_keeps_gross_base_stress_as_distinct_scenarios():
    n = 30
    decision_times = tuple(str(np.datetime64("2024-01-01") + np.timedelta64(i, "D")) for i in range(n))
    dates = decision_times
    x = np.arange(n * 3, dtype=float).reshape(n, 3)
    batch = FactorBatch(("set-v2",), AxisRef("time", "date", n), AxisRef("asset", "str", 3), x[:, :, None])
    label = LabelBundle("forward", x, 1, decision_time=decision_times,
                        label_start_time=tuple(str(np.datetime64(x) + np.timedelta64(1, "D")) for x in decision_times),
                        label_end_time=tuple(str(np.datetime64(x) + np.timedelta64(2, "D")) for x in decision_times))
    refs = TrajectoryRefs(("set-v2",), "sha256:data", "portfolio:set-v2", "cost:v5", "benchmark:pit", "fills:set-v2")
    gross_ret = tuple([.01, -.005] * 15)
    scenarios = {}
    for sid, scope, cost_profile, cost in (
        ("gross", CostScope.GROSS_DIAGNOSTIC, "gross", 0.0),
        ("base", CostScope.NET_ASSUMED, "net-base", .001),
        ("stress", CostScope.NET_ASSUMED, "net-stress", .004),
    ):
        trajectory = build_research_trajectory(
            scenario_id=sid, profile="LONG_ONLY_RESEARCH", dates=dates,
            gross_return=gross_ret, benchmark_return=tuple(0.0 for _ in dates),
            cost_contributions={"execution": tuple(cost for _ in dates)}, refs=refs, scope=scope,
        )
        scenarios[sid] = EvaluationScenario(label, cost_profile=cost_profile,
                                             portfolio_profile="LONG_ONLY_RESEARCH", trajectory=trajectory)
    instances = tuple(MetricInstance("sharpe_ratio", scenario_id=sid,
                                     cost_profile=cp, portfolio_profile="LONG_ONLY_RESEARCH")
                      for sid, cp in (("gross", "gross"), ("base", "net-base"), ("stress", "net-stress")))
    out = evaluate(EvaluationRequest(batch, label, metric_instances=instances,
                                     scenario_inputs=scenarios, tier="extended"))
    values = {spec.scenario_id: out.instance_results[iid].get_metric("sharpe_ratio", "set-v2").value
              for iid, spec in out.instance_specs.items()}
    assert values["gross"] > values["base"] > values["stress"]
    assert len(out.instance_results) == 3
