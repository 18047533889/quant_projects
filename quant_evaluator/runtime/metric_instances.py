"""Compile variant requests onto the existing public evaluator and shared builders."""
from dataclasses import replace
import math

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.metric_instance import EvaluationScenario


def evaluate_instances(request, *, evaluator=None, backend=None, gpu_policy=None):
    from quant_evaluator.runtime.evaluator import evaluate, Evaluator, _compile_public_artifact_plan
    from quant_evaluator.registry.metrics import get_metric

    if request.slices is not None:
        raise ValueError("Metric instance slices are not supported")
    if request.metric_parameters or request.quantile_builder_parameters:
        raise ValueError("Instance parameters must be carried by each MetricInstance")
    scenarios = dict(request.scenario_inputs)
    if "default" not in scenarios and any(item.scenario_id == "default" for item in request.metric_instances):
        scenarios["default"] = EvaluationScenario(
            label_bundle=request.label_bundle, portfolio_returns=request.portfolio_returns,
            holding_returns=request.holding_returns, portfolio_spec=request.portfolio_spec,
            trade_eligibility=request.trade_eligibility, calendar_snapshot=request.calendar_snapshot,
            exposure_panel=request.exposure_panel)
    instances = {}
    requested_to_resolved = {}
    for requested in request.metric_instances:
        if requested.scenario_id not in scenarios:
            raise ValueError(f"Missing runtime scenario binding: {requested.scenario_id}")
        scenario = scenarios[requested.scenario_id]
        scenario.validate(requested)
        required_leg = get_metric(requested.metric_id).required_portfolio_leg
        item = replace(requested, horizon=requested.horizon or scenario.label_bundle.horizon,
                       leg=required_leg if requested.leg == "unspecified" and required_leg is not None else requested.leg,
                       price_convention=requested.price_convention or scenario.label_bundle.price_convention)
        instances[item.instance_id] = item
        requested_to_resolved[requested.instance_id] = item.instance_id
    groups = []
    tier_order = {"core": 0, "extended": 1, "research": 2}
    if request.tier not in tier_order:
        raise ValueError("Unknown evaluation tier")
    for item in instances.values():
        if item.scenario_id not in scenarios:
            raise ValueError(f"Missing runtime scenario binding: {item.scenario_id}")
        scenario = scenarios[item.scenario_id]
        scenario.validate(item)
        if get_metric(item.metric_id).required_portfolio_leg not in (None, item.leg):
            raise ValueError("Metric instance leg disagrees with its registry definition")
        if tier_order[get_metric(item.metric_id).tier.value] > tier_order[request.tier]:
            raise ValueError("Metric instance exceeds requested tier")
        for group in groups:
            other = group[0]
            if (item.scenario_id != other.scenario_id or
                    item.leg != other.leg or
                    dict(item.quantile_builder_parameters) != dict(other.quantile_builder_parameters)):
                continue
            compatible = True
            for prior in group:
                if prior.metric_id == item.metric_id and dict(prior.parameters) != dict(item.parameters):
                    compatible = False
                for name in set(prior.parameters) & set(item.parameters):
                    if stable_content_hex(tag=name, fields=prior.parameters[name]) != stable_content_hex(tag=name, fields=item.parameters[name]):
                        compatible = False
            if compatible:
                group.append(item)
                break
        else:
            groups.append([item])

    # Admit the entire request before any scenario is executed. Artifact builders
    # are counted once per compatible group, exactly as in the existing facade.
    planned_cost = 0.0
    for group in groups:
        scenario = scenarios[group[0].scenario_id]
        specs = {item.metric_id: get_metric(item.metric_id) for item in group}
        plan = _compile_public_artifact_plan(specs, portfolio_returns=scenario.portfolio_returns,
                                            holding_returns=scenario.holding_returns,
                                            exposure_panel=scenario.exposure_panel)
        planned_cost += len(specs) + sum(node["cost"] for node in plan)
    budget = request.cost_budget
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float))
                               or not math.isfinite(budget) or budget < planned_cost):
        raise ValueError(f"Metric instance plan cost {planned_cost} exceeds valid cost_budget {budget}")

    children = {}
    first = None
    runtime = evaluator or Evaluator()
    for group in groups:
        scenario = scenarios[group[0].scenario_id]
        fields = {name: getattr(scenario, name) for name in (
            "label_bundle", "portfolio_returns", "holding_returns", "portfolio_spec",
            "trade_eligibility", "calendar_snapshot", "exposure_panel")}
        fields["portfolio_returns"] = scenario.probe_for(group[0].leg, request.batch_or_factor_ids.factor_ids)
        bound = replace(request, **fields, metric_instances=(), scenario_inputs={},
                        metric_ids=tuple(dict.fromkeys(item.metric_id for item in group)),
                        metric_parameters={item.metric_id: dict(item.parameters) for item in group},
                        quantile_builder_parameters=dict(group[0].quantile_builder_parameters),
                        cost_budget=None)
        bundle = evaluate(bound, evaluator=runtime, backend=backend, gpu_policy=gpu_policy)
        first = first or bundle
        for item in group:
            mid = item.metric_id
            keep_artifacts = item.output_mode != "SUMMARY_ONLY"
            metadata = dict(bundle.metadata)
            metadata.update(metric_instance_id=item.instance_id, metric_instance=item.to_dict(),
                            scenario_label_content_hash=scenario.label_bundle.content_hash,
                            computation_status="COMPUTED")
            config = stable_content_hex(tag="MetricInstanceResult.v1", fields={
                "instance": item.instance_id, "source_config": bundle.config_hash,
                "labels": scenario.label_bundle.content_hash})
            children[item.instance_id] = replace(
                bundle, request_id=f"{bundle.request_id}:{item.instance_id}", config_hash=config,
                metric_values={k:v for k,v in bundle.metric_values.items() if k == mid},
                metric_versions={mid: bundle.metric_versions[mid]},
                grouped_metrics=None if bundle.grouped_metrics is None else {
                    fid: {k:v for k,v in values.items() if k == mid}
                    for fid, values in bundle.grouped_metrics.items()},
                artifacts={k:v for k,v in bundle.artifacts.items() if k == mid and keep_artifacts},
                factor_artifacts={fid: {k:v for k,v in values.items() if k == mid and keep_artifacts}
                                  for fid, values in bundle.factor_artifacts.items()},
                series_refs=None if bundle.series_refs is None else {
                    k:v for k,v in bundle.series_refs.items() if k == mid and keep_artifacts},
                metadata=metadata)
    if first is None:
        raise ValueError("Metric instance evaluation requires at least one instance")
    config = stable_content_hex(tag="MetricInstanceBundle.v1", fields={
        key: child.config_hash for key, child in children.items()})
    return replace(first, metric_values={}, metric_versions={}, artifacts={}, factor_artifacts={},
                   grouped_metrics=None, series_refs=None, label_id="MULTI_SCENARIO",
                   config_hash=config, instance_results=children, instance_specs=instances,
                   metadata={**dict(request.metadata), "planned_cost": planned_cost,
                             "requested_to_resolved_instances": requested_to_resolved,
                             "shared_evaluation_groups": len(groups),
                             "instance_statuses": {key: "COMPUTED" for key in children}})
