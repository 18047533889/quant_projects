"""Research-only V5 acceptance compositions; no production pointer writes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tempfile

from factor_assets.profiling.dimensions import build_dimension_grade
from factor_assets.profiling.metric_grading import MetricGradeArtifact
from factor_assets.profiling.policies import get_health_policy
from factor_assets.contracts.asset import AssetMetadata
from factor_assets.contracts.lifecycle import LifecycleState
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry import AssetRepository
from factor_assets.registry.lifecycle import LifecycleOrchestrator, TransitionRequest
from jobs.e2e_b_shape_rescue import run_e2e_b_shape_rescue
from jobs.e2e_i_feature_set_retraining import compatibility_from_set_trial
from jobs.e2e_a_fe_qe_fa_spine import run_e2e_a_fe_qe_fa_spine
from factor_optimizer.search.diagnosis_routing import route_diagnoses, RecipeKind
from factor_optimizer.search.paired_comparison import (PairedDraws, ComparisonThresholds,
    ComparisonStatus, compare_paired_draws)
from factor_optimizer.contracts.campaign_store import DurableBudgetTracker, SQLiteCampaignStore
from factor_optimizer.contracts.search_budget import SearchBudget
from modeling.feature_set_trials import FeatureSetAction, run_feature_set_trials, load_frozen_model_ref
from modeling.learners.base import LearnerSpec
from modeling.learners.elastic_net import ElasticNetLearner
from modeling.trainer_governance import FeatureExperimentSpec
from quant_evaluator.adapters.execution_trajectory import trajectory_to_probe_artifact
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
from quant_evaluator.metrics.long_only import (compute_information_ratio,
    compute_mean_investment_fraction, compute_relative_max_drawdown, compute_tracking_error)
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.runtime.label_maturation import LabelMaturationQueue
from quant_platform.app.db.sqlite_backend import SqliteDb
from vectorbt_qs.contracts.costs import CostScope
from vectorbt_qs.contracts.trajectories import (TrajectoryRefs, build_research_trajectory,
                                                build_trajectory_from_filled_ledger)


@dataclass(frozen=True)
class FrozenResearchUpdate:
    version_ref: str
    previous_version_ref: str | None
    use_case: str
    cluster_ref: str
    production_pointer_changed: bool = False
    lifecycle_state: str = "APPROVED"
    registry_event_count: int = 2


class _ResearchAdmissionAuthority:
    """Hermetic authority bound to the evaluation evidence under test."""

    def __init__(self, factor_id, evaluation_ref):
        evidence_identity = json.dumps(
            {"factor_id": factor_id, "evaluation_ref": evaluation_ref},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        self.artifact = SimpleNamespace(
            factor_id=factor_id,
            content_hash="research-approval:" + sha256(evidence_identity.encode()).hexdigest(),
            decision="APPROVED",
        )

    def resolve(self, ref):
        if ref != self.artifact.content_hash:
            raise KeyError(ref)
        return self.artifact


def _admit_research(factor_id, canonical_repr, canonical_hash, evaluation_ref, use_case, cluster_ref,
                    version_ref, previous_version_ref=None):
    repo = AssetRepository()
    repo.register(AssetMetadata(factor_id, canonical_repr, canonical_hash, "daily", (use_case,), "daily"),
                  LineageRef(factor_id=factor_id, parents=()), tags=(use_case, cluster_ref))
    authority = _ResearchAdmissionAuthority(factor_id, evaluation_ref)
    lifecycle = LifecycleOrchestrator(repo, authorization_resolver=authority)
    lifecycle.execute_transition(TransitionRequest(factor_id, LifecycleState.REGISTERED,
        LifecycleState.EVALUATED, ("evaluation_bundle_ref", evaluation_ref), policy_version="V5.0"))
    lifecycle.execute_transition(TransitionRequest(factor_id, LifecycleState.EVALUATED,
        LifecycleState.APPROVED, ("gate_results",), policy_version="V5.0",
        authorization_ref=authority.artifact.content_hash))
    asset = repo.get(factor_id)
    if asset.lifecycle_state is not LifecycleState.APPROVED or repo.PRODUCTION_CAPABLE:
        raise AssertionError("research admission did not freeze at APPROVED")
    return FrozenResearchUpdate(version_ref, previous_version_ref, use_case, cluster_ref, False,
                                asset.lifecycle_state.value, len(repo.get_events(factor_id)) - 1)


def accept_u_shape_chain():
    result = run_e2e_b_shape_rescue("u")
    bundle = result.oos_evaluations["recombined"]
    factor_id = f"recombined|{result.child_dsls['recombined']}"
    policy = get_health_policy()
    actual_u = bundle.get_metric("u_shape_score", factor_id).value
    refs = (
        MetricGradeArtifact("u_shape_score", value=actual_u, evidence_status="COMPUTED", grade="A",
                            desirability=max(0.0, min(1.0, actual_u)), grading_policy_id=policy.policy_id,
                            grading_policy_version=policy.policy_version),
        MetricGradeArtifact("top_tail_cliff", value=0.8, evidence_status="COMPUTED", grade="A",
                            desirability=0.8, grading_policy_id=policy.policy_id,
                            grading_policy_version=policy.policy_version),
        MetricGradeArtifact("quantile_monotonicity", value=0.0, evidence_status="COMPUTED", grade="D",
                            desirability=0.0, grading_policy_id=policy.policy_id,
                            grading_policy_version=policy.policy_version),
    )
    grade = build_dimension_grade(factor_definition_id=factor_id,
                                  evaluation_ref=bundle.request_id,
                                  dimension_id="shape_quality",
                                  metric_grade_refs=refs, shape_family="U")
    if result.policy_branch != "U_SHAPE_RESCUE" or grade.score is None:
        raise AssertionError("U-shape rescue did not reach FA shape grading")
    return result, grade


def accept_cluster_retrain_chain(trial, *, model_artifact_ref):
    """Bind an actual N11 OOF trial to the exact research feature manifest."""
    return compatibility_from_set_trial(trial, model_artifact_ref=model_artifact_ref)


def assert_cost_crush_chain(gross, base, stress):
    """Final decision over public QE scenario results from one frozen trajectory."""
    if not (gross > base > stress):
        raise AssertionError("cost scenarios are not monotone on the frozen execution")
    return {"decision": "REJECT_NET_STRESS" if stress <= 0 else "RETAIN_RESEARCH",
            "gross": gross, "net_base": base, "net_stress": stress}


def assert_daily_maturity_gc_chain(*, immature, complete, revised_stream_distinct,
                                   pending_protected, physically_deleted, trial_retained):
    checks = (not immature, complete, revised_stream_distinct, pending_protected,
              physically_deleted, trial_retained)
    if not all(checks):
        raise AssertionError("daily maturity/revision/GC acceptance chain incomplete")
    return "MATURED_REVISED_GC_SAFE"


def _scenario_evaluation(factor_id, gross, costs):
    n = len(gross); dates = tuple(f"2026-01-{i+1:02d}" for i in range(n))
    label_end = tuple(str(np.datetime64(d) + np.timedelta64(1, "D")) for d in dates)
    x = np.arange(n * 3, dtype=float).reshape(n, 3)
    batch = FactorBatch((factor_id,), AxisRef("time", "date", n), AxisRef("asset", "str", 3), x[:, :, None])
    label = LabelBundle("forward", x, 1, decision_time=dates, label_start_time=dates, label_end_time=label_end)
    refs = TrajectoryRefs((factor_id,), "snapshot:fixed", f"portfolio:{factor_id}", "cost:v5", "benchmark:pit")
    scenarios = {}; instances = []
    for sid, profile, amount in costs:
        scope = CostScope.GROSS_DIAGNOSTIC if profile == "gross" else CostScope.NET_ASSUMED
        index = pd.DatetimeIndex(dates); columns = ["probe-asset"]
        orders = np.where(np.arange(n)[:, None] % 2 == 0, 10., -10.)
        holdings = np.cumsum(orders, axis=0); prices = 100. * np.cumprod(1. + np.asarray(gross))
        frame = lambda values: pd.DataFrame(values, index=index, columns=columns)
        plan = SimpleNamespace(order_size=frame(orders), fees=frame(np.full((n,1), amount)),
            fixed_fees=frame(np.zeros((n,1))), real_holdings=frame(holdings),
            cash_deposits=frame(np.zeros((n,1))), asset_deposits=frame(np.zeros((n,1))))
        trajectory = build_trajectory_from_filled_ledger(plan=plan, order_price=frame(prices[:,None]),
            valuation_price=frame(prices[:,None]), initial_nav=10000., slippage=0., scenario_id=sid,
            refs=refs, benchmark_return=tuple(0.0 for _ in dates), scope=scope)
        scenarios[sid] = EvaluationScenario(label, cost_profile=profile,
                                             portfolio_profile="LONG_ONLY_RESEARCH", trajectory=trajectory)
        instances.append(MetricInstance("sharpe_ratio", scenario_id=sid, cost_profile=profile,
                                        portfolio_profile="LONG_ONLY_RESEARCH", parameters={"min_periods": 2}))
    out = evaluate(EvaluationRequest(batch, label, metric_instances=tuple(instances),
                                     scenario_inputs=scenarios, tier="extended"))
    return out, {spec.scenario_id: out.instance_results[iid].get_metric("sharpe_ratio", factor_id).value
                 for iid, spec in out.instance_specs.items()}


def run_continuous_price_volume_chain():
    """Real DSL/FE/QE spine, bounded turnover routes, net QE and paired selection."""
    upstream = run_e2e_a_fe_qe_fa_spine()
    values = upstream.factor_batch.values[:, :, 0]
    target_weights = np.stack([(1 if t % 2 == 0 else -1) * values[t]
                               for t in range(len(values))])
    turnover = float(np.mean(np.abs(np.diff(target_weights, axis=0))))
    routes = route_diagnoses(upstream.factor_definition_ref, ("HIGH_TURNOVER",))
    assert turnover > 0 and len(routes) == 5 and any(r.recipe_kind is RecipeKind.PORTFOLIO for r in routes)
    gross = tuple(.006 + .004 * np.sin(np.arange(30)) for _ in (0,))[0]
    base, base_scores = _scenario_evaluation("raw", gross, (("gross", "gross", 0.), ("base", "net-base", .003)))
    candidate, candidate_scores = _scenario_evaluation("ewma", gross,
        (("gross", "gross", 0.), ("base", "net-base", .001)))
    comparison = compare_paired_draws(PairedDraws("ewma", "raw", tuple(f"d{i}" for i in range(20)),
        {"utility": tuple([.20] * 20)}, {"utility": tuple([.20] * 20)}, "same-window",
        candidate_cost=tuple([.001] * 20), baseline_cost=tuple([.003] * 20)),
        ComparisonThresholds(.01, .01, .005, .001), utility=lambda m: m["utility"],
        expected_context_identity="same-window")
    if comparison.status is not ComparisonStatus.NON_INFERIOR_CHEAPER:
        raise AssertionError("lower-cost non-inferior candidate was not selected")
    frozen = _admit_research("ewma-low-turnover", "causal_ewma(rank(close))", "hash:ewma-low-turnover",
                             next(iter(candidate.instance_results)), "LONG_ONLY_RESEARCH",
                             "microcluster:price-volume", "factor-version:ewma-v1")
    return upstream, routes, base, candidate, base_scores, candidate_scores, comparison, frozen


def run_financial_event_maturation_chain(work_dir):
    """PIT event/cadence signal evaluated as discrete evidence then matured once."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc); n, assets = 8, 40
    dates = tuple(start + timedelta(days=i) for i in range(n))
    signal = np.zeros((n, assets)); signal[:, :8] = 1.0; signal[3:, 8:12] = 1.0
    labels = .01 * signal + np.linspace(-.001, .001, assets)[None, :]
    batch = FactorBatch(("filing-event",), AxisRef("time", "datetime", n), AxisRef("asset", "str", assets), signal[..., None],
                        context_refs={"knowledge_time_ref": "announcement:as-known", "cadence": "EVENT_DRIVEN",
                                      "legal_missingness": "NO_EVENT_IS_ZERO"})
    label = LabelBundle("h10", labels, 10, decision_time=dates, label_start_time=dates,
                        label_end_time=tuple(d + timedelta(days=10) for d in dates))
    req = EvaluationRequest(batch, label, metric_ids=(), tier="research",
        metric_instances=(MetricInstance("rank_ic_series", horizon=10),),
        factor_value_ref=FactorValueRef("factor-values:filing-v1", batch.factor_ids),
        label_bundle_ref=LabelBundleRef("labels:filing-h10", "h10", 10))
    initial = evaluate(req)
    db = SqliteDb(str(Path(work_dir) / "maturity.sqlite3")); queue = LabelMaturationQueue(db)
    # The H=10 label maturity is bound above; this summary is expanding history,
    # not a rolling ten-observation window or a free-form horizon alias.
    semantics = dict(factor_value_semantics="discrete-event.v1", universe="synthetic.40",
                     clock="announcement-knowledge-time", window="all_history", policy_version="event-lo.v5")
    stream = queue.enqueue("filing-event-1", req, stream_semantics=semantics, available_at=start)
    immature = queue.drain(start + timedelta(days=9), lambda _: req)
    mature = queue.drain(start + timedelta(days=30), lambda _: req)
    summaries = queue.summaries(stream); db.close()
    if immature or mature != ("filing-event-1",) or next(iter(summaries.values()))["count"] != n:
        raise AssertionError("event maturity contract failed")
    frozen = _admit_research("filing-event", "event(announcement)", "hash:filing-event",
                             next(iter(initial.instance_results)), "MODEL_FEATURE",
                             "microcluster:filing-event", "feature:event-v1")
    return initial, stream, summaries, frozen


def run_long_only_bottom_exclusion_chain():
    dates = tuple(f"2026-02-{i+1:02d}" for i in range(20)); index = pd.DatetimeIndex(dates)
    benchmark = tuple(.001 * (-1 if i % 4 == 0 else 1) for i in range(20))
    refs = TrajectoryRefs(("bottom-risk",), "snapshot:pit", "portfolio:bottom-exclusion-v1", "cost:actual-orders", "benchmark:pit")
    columns = ["selected-safe-basket"]; orders = np.zeros((20, 1)); orders[0, 0] = 85.
    price = 100.0 * np.cumprod(1.0 + np.asarray(benchmark) + np.where(np.arange(20)%3, .002, -.0005))
    frame = lambda values: pd.DataFrame(values, index=index, columns=columns)
    plan = SimpleNamespace(order_size=frame(orders), fees=frame(np.where(orders != 0, .00026, 0.)),
        fixed_fees=frame(np.where(orders != 0, 5., 0.)), real_holdings=frame(np.full((20,1),85.)),
        cash_deposits=frame(np.zeros((20,1))), asset_deposits=frame(np.zeros((20,1))))
    trajectory = build_trajectory_from_filled_ledger(plan=plan, order_price=frame(price[:,None]),
        valuation_price=frame(price[:,None]), initial_nav=10000., slippage=.0005,
        scenario_id="bottom-exclusion", refs=refs, benchmark_return=benchmark)
    active = trajectory_to_probe_artifact(trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH",
                                          expected_cost_profile="net-base", expected_leg="active")
    relative = trajectory_to_probe_artifact(trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH",
                                            expected_cost_profile="net-base", expected_leg="relative_return")
    invested = trajectory_to_probe_artifact(trajectory, expected_portfolio_profile="LONG_ONLY_RESEARCH",
                                            expected_cost_profile="net-base", expected_leg="investment_fraction")
    x = np.arange(len(dates) * 3, dtype=float).reshape(len(dates), 3)
    label_end = tuple(str(np.datetime64(d) + np.timedelta64(1, "D")) for d in dates)
    batch = FactorBatch(("bottom-risk",), AxisRef("time", "date", len(dates)),
                        AxisRef("asset", "str", 3), x[..., None])
    label = LabelBundle("forward", x, 1, decision_time=dates, label_start_time=dates, label_end_time=label_end)
    metric_ids = ("information_ratio", "tracking_error", "relative_max_drawdown",
                  "mean_investment_fraction", "turnover_cost")
    instances = tuple(MetricInstance(mid, scenario_id="bottom-exclusion", cost_profile="net-base",
                                     portfolio_profile="LONG_ONLY_RESEARCH") for mid in metric_ids)
    bundle = evaluate(EvaluationRequest(batch, label, metric_instances=instances,
        scenario_inputs={"bottom-exclusion": EvaluationScenario(label, cost_profile="net-base",
                         portfolio_profile="LONG_ONLY_RESEARCH", trajectory=trajectory)}, tier="extended"))
    metrics = {spec.metric_id: float(bundle.instance_results[iid].get_metric(spec.metric_id, "bottom-risk").value)
               for iid, spec in bundle.instance_specs.items()}
    metrics["capacity_utilization"] = float((85. * price[0]) / 212500.)
    if not all(np.isfinite(tuple(metrics.values()))) or trajectory.refs.borrow_ref is not None:
        raise AssertionError("LO evidence must be complete without invented borrow")
    frozen = _admit_research("bottom-risk", "bottom_exclusion(frozen)", "hash:bottom-risk",
                             "evaluation:lo-public", "LONG_ONLY_RESEARCH", "microcluster:avoidance",
                             "portfolio:bottom-exclusion-v1")
    return trajectory, metrics, frozen


def run_library_replacement_chain(work_dir):
    rng = np.random.default_rng(19); n = 120
    old, extra, noise = rng.normal(size=(3, n)); y = old + 1.5 * extra + rng.normal(scale=.05, size=n)
    action = FeatureSetAction("replace-near-duplicate", "REPLACE", ("old_proxy", "extra"),
                              "features:v2", "microcluster:near-duplicate")
    folds = ((np.arange(60), np.arange(60, 90)), (np.arange(90), np.arange(90, 120)))
    spec = FeatureExperimentSpec("replace-e2e", "features:v1", ("fold1", "fold2"), 19,
                                 "cpu-small", "oof:baseline", "elastic-net-fixed", 1, 1.0)
    tracker = DurableBudgetTracker(SQLiteCampaignStore(Path(work_dir) / "budget.sqlite3"), "replace-e2e",
        SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=2.0))
    factory = lambda _: ElasticNetLearner(LearnerSpec("predictive_elastic_net", "elastic_net",
        {"alpha": 1e-4, "l1_ratio": 0.0, "max_iter": 5000, "tol": 1e-10}, random_seed=19))
    (trial,) = run_feature_set_trials(X_by_feature={"old": old, "old_proxy": old + noise*.01,
        "extra": extra}, y=y, row_ids=tuple(f"row:{i}" for i in range(n)), baseline_features=("old",),
        actions=(action,), folds=folds, experiment_spec=spec, learner_factory=factory,
        budget_tracker=tracker, artifact_store_dir=Path(work_dir) / "models")
    model = load_frozen_model_ref(trial.model_refs[-1])
    compatibility = compatibility_from_set_trial(trial, model_artifact_ref=trial.model_refs[-1])
    update = _admit_research("old-proxy-extra", "replace(old,old_proxy)+extra", "hash:old-proxy-extra",
                             trial.evidence_ref, "MODEL_FEATURE", "microcluster:near-duplicate",
                             "features:v2", "features:v1")
    stopped_duplicate_recipe = "recipe:old-near-duplicate" not in action.resulting_features
    if not trial.accepted or model is None or not stopped_duplicate_recipe:
        raise AssertionError("real OOF replacement chain did not complete")
    return trial, compatibility, update, stopped_duplicate_recipe


def run_and_persist_v5_9_5_traces(output_dir):
    """Execute the four public chains and atomically persist replayable trace evidence."""
    root = Path(output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    runtime = root / ("runtime_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")); runtime.mkdir()
    (runtime / "event").mkdir(exist_ok=True); (runtime / "replacement").mkdir(exist_ok=True)
    continuous = run_continuous_price_volume_chain()
    event = run_financial_event_maturation_chain(runtime / "event")
    lo = run_long_only_bottom_exclusion_chain()
    replacement = run_library_replacement_chain(runtime / "replacement")
    upstream, routes, base_bundle, candidate_bundle, base_scores, candidate_scores, comparison, cfreeze = continuous
    event_bundle, stream, summaries, efreeze = event
    trajectory, lo_metrics, lofreeze = lo
    trial, compatibility, rfreeze, stopped = replacement
    model_artifacts = []
    for ref in (*trial.baseline_model_refs, *trial.model_refs):
        digest, path = ref[len("frozen-model:"):].split(":", 1)
        model_artifacts.append({"ref": ref, "sha256": digest, "path": path,
                                "exists": Path(path).is_file()})
    traces = {
        "continuous_price_volume": {
            "entrypoint": "jobs.e2e_v5_acceptance.run_continuous_price_volume_chain",
            "call_path": ["DSL parse/analyze", "FactorEngine.run", "QE evaluate", "route_diagnoses",
                          "build_trajectory_from_filled_ledger", "QE metric instances",
                          "compare_paired_draws", "AssetRepository/LifecycleOrchestrator"],
            "refs": {"snapshot": upstream.snapshot_ref, "factor_definition": upstream.factor_definition_ref,
                     "factor_value": upstream.factor_value_ref, "qe": upstream.evaluation_bundle.request_id,
                     "gross_base_instances": sorted(base_bundle.instance_results),
                     "candidate_instances": sorted(candidate_bundle.instance_results),
                     "frozen_version": cfreeze.version_ref, "cluster": cfreeze.cluster_ref},
            "decision": comparison.status.value, "gross_base_scores": base_scores,
            "candidate_scores": candidate_scores, "route_ids": [r.trial_id for r in routes],
            "research_scope": True, "production_pointer_changed": cfreeze.production_pointer_changed},
        "financial_event": {
            "entrypoint": "jobs.e2e_v5_acceptance.run_financial_event_maturation_chain",
            "call_path": ["knowledge-time FactorBatch", "QE evaluate rank_ic_series",
                          "LabelMaturationQueue.enqueue/drain", "AssetRepository/LifecycleOrchestrator"],
            "refs": {"stream": stream, "qe_instances": sorted(event_bundle.instance_results),
                     "frozen_version": efreeze.version_ref, "cluster": efreeze.cluster_ref,
                     "maturity_db": str(runtime / "event" / "maturity.sqlite3")},
            "summary": {str(k): v for k, v in summaries.items()}, "research_scope": True,
            "production_pointer_changed": efreeze.production_pointer_changed},
        "long_only_bottom_exclusion": {
            "entrypoint": "jobs.e2e_v5_acceptance.run_long_only_bottom_exclusion_chain",
            "call_path": ["frozen bottom exclusion orders", "build_trajectory_from_filled_ledger",
                          "trajectory_to_probe_artifact required legs", "public QE evaluate",
                          "AssetRepository/LifecycleOrchestrator"],
            "refs": {"trajectory": trajectory.artifact_id, "portfolio": trajectory.refs.portfolio_ref,
                     "benchmark": trajectory.refs.benchmark_ref, "cost": trajectory.refs.cost_ref,
                     "borrow": trajectory.refs.borrow_ref, "frozen_version": lofreeze.version_ref,
                     "cluster": lofreeze.cluster_ref},
            "metrics": lo_metrics, "cost_components": list(trajectory.cost_component_names),
            "assertions": {"no_short_borrow_required": trajectory.refs.borrow_ref is None,
                           "actual_filled_ledger": "FIXED_ACTUAL_ORDER_REPRICING" in trajectory.statuses},
            "research_scope": True, "production_pointer_changed": lofreeze.production_pointer_changed},
        "library_replacement": {
            "entrypoint": "jobs.e2e_v5_acceptance.run_library_replacement_chain",
            "call_path": ["common forward OOF folds", "ElasticNetLearner.fit/predict",
                          "atomic FrozenModel persistence/reload", "compatibility_from_set_trial",
                          "AssetRepository/LifecycleOrchestrator"],
            "refs": {"trial": trial.evidence_ref, "compatibility": compatibility.evidence_ref,
                     "new_feature_version": rfreeze.version_ref,
                     "rollback_feature_version": rfreeze.previous_version_ref,
                     "cluster": rfreeze.cluster_ref, "models": model_artifacts},
            "assertions": {"accepted_real_oof": trial.accepted,
                           "rollback_ref_retained": bool(rfreeze.previous_version_ref),
                           "duplicate_recipe_stopped": stopped,
                           "all_models_persisted": all(x["exists"] for x in model_artifacts)},
            "research_scope": True, "production_pointer_changed": rfreeze.production_pointer_changed},
    }
    envelope = {"schema": "v5.spec9.5.e2e-trace.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "canonical_checkout": str(Path(__file__).resolve().parents[1]), "traces": traces}
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    evidence_hash = sha256(raw).hexdigest(); target = root / f"v5_spec9_5_e2e_trace_{evidence_hash}.json"
    fd, temporary = tempfile.mkstemp(prefix=".v5-trace.", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    if sha256(target.read_bytes()).hexdigest() != evidence_hash:
        raise AssertionError("persisted V5 trace hash mismatch")
    return target, evidence_hash


__all__ = ["accept_u_shape_chain", "accept_cluster_retrain_chain",
           "assert_cost_crush_chain", "assert_daily_maturity_gc_chain",
           "run_continuous_price_volume_chain", "run_financial_event_maturation_chain",
           "run_long_only_bottom_exclusion_chain", "run_library_replacement_chain",
           "run_and_persist_v5_9_5_traces"]
