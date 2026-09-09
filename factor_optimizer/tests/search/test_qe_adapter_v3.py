import numpy as np
import pytest


def _fixtures(factor_ids=("a", "b", "c")):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle

    rng = np.random.default_rng(41)
    t, n = 36, 12
    values = rng.normal(size=(t, n, len(factor_ids)))
    labels = rng.normal(size=(t, n))
    batch = FactorBatch(
        factor_ids=tuple(factor_ids),
        time_axis=AxisRef("t", "int", t),
        asset_axis=AxisRef("a", "str", n),
        values=np.ascontiguousarray(values),
    )
    bundle = LabelBundle(
        target_id="forward_return",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
    )
    return batch, bundle


def test_real_public_batch_adapter_preserves_per_factor_typed_evidence():
    from factor_optimizer.adapters.quant_evaluator import (
        InMemoryEvidenceStore,
        create_qe_adapter,
    )

    batch, labels = _fixtures()
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    result = adapter.evaluate(batch, labels, metrics=["rank_ic", "coverage"])

    assert result["metrics"] == {}
    assert set(result["grouped_metrics"]) == {"a", "b", "c"}
    for factor_id in batch.factor_ids:
        for metric_id in ("rank_ic", "coverage"):
            metric = result["grouped_metrics"][factor_id][metric_id]
            assert metric["metric_id"] == metric_id
            assert isinstance(metric["valid"], bool)
            assert isinstance(metric["observation_count"], int)
            assert "metric_version" in metric
            assert "warnings" in metric

    stored = adapter.get_evidence(result["evaluation_id"])
    assert stored["grouped_metrics"] == result["grouped_metrics"]
    assert stored["factor_ids"] == ["a", "b", "c"]
    assert stored["content_identity"] == result["content_identity"]
    assert result["factor_ids"] == list(batch.factor_ids)
    assert set(result["metric_versions"]) == {"rank_ic", "coverage"}
    assert result["request_id"] == stored["request_id"]
    assert result["split_ref"] == stored["split_ref"]
    assert result["config_hash"] == stored["config_hash"]


def test_batch_and_single_factor_views_match_by_factor_id_not_position():
    from factor_optimizer.adapters.quant_evaluator import (
        InMemoryEvidenceStore,
        create_qe_adapter,
    )
    from quant_evaluator.contracts.factor_batch import FactorBatch

    batch, labels = _fixtures()
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    batched = adapter.evaluate(batch, labels, metrics=["rank_ic"])

    for index, factor_id in enumerate(batch.factor_ids):
        one = FactorBatch(
            factor_ids=(factor_id,),
            time_axis=batch.time_axis,
            asset_axis=batch.asset_axis,
            values=np.ascontiguousarray(batch.values[:, :, index:index + 1]),
        )
        single = adapter.evaluate(one, labels, metrics=["rank_ic"])
        single_metric = single["metrics"]["rank_ic"]
        batch_metric = batched["grouped_metrics"][factor_id]["rank_ic"]
        assert single_metric["valid"] == batch_metric["valid"]
        assert single_metric["observation_count"] == batch_metric["observation_count"]
        assert single_metric["value"] == pytest.approx(batch_metric["value"], abs=1e-15)


def test_invalid_metric_remains_invalid_typed_evidence():
    from factor_optimizer.adapters.quant_evaluator import (
        InMemoryEvidenceStore,
        create_qe_adapter,
    )

    batch, labels = _fixtures(("constant",))
    constant = type(batch)(
        factor_ids=batch.factor_ids,
        time_axis=batch.time_axis,
        asset_axis=batch.asset_axis,
        values=np.ones_like(batch.values),
    )
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    result = adapter.evaluate(constant, labels, metrics=["rank_ic"])
    evidence = result["metrics"]["rank_ic"]
    assert evidence["valid"] is False
    assert evidence["value"] is None


def test_tier_cost_and_metadata_reach_qe_typed_request():
    from factor_optimizer.adapters.quant_evaluator import InMemoryEvidenceStore, create_qe_adapter

    batch, labels = _fixtures(("a",))
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    result = adapter.evaluate(
        batch, labels, metrics=["coverage"], tier="core",
        cost_budget=7.5, request_metadata={"stage": "S1"}, backend="cpu",
    )
    stored = adapter.get_evidence(result["evaluation_id"])
    assert stored["metadata"]["tier"] == "core"
    assert stored["metadata"]["cost_budget"] == 7.5
    assert stored["metadata"]["stage"] == "S1"


def test_unknown_tier_cannot_bypass_qe_budget_policy():
    from factor_optimizer.adapters.quant_evaluator import InMemoryEvidenceStore, create_qe_adapter
    from quant_evaluator.contracts.errors import InvalidContractError
    batch, labels = _fixtures(("a",))
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    with pytest.raises(InvalidContractError, match="Unknown evaluation tier"):
        adapter.evaluate(batch, labels, metrics=["coverage"], tier="screen")


def test_metric_parameters_fail_closed_when_qe_contract_cannot_accept_them():
    from dataclasses import fields
    from factor_optimizer.adapters.quant_evaluator import InMemoryEvidenceStore, create_qe_adapter
    from quant_evaluator.api.requests import EvaluationRequest

    batch, labels = _fixtures(("a",))
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    if "metric_parameters" in {item.name for item in fields(EvaluationRequest)}:
        baseline = adapter.evaluate(batch, labels, metrics=["coverage"])
        result = adapter.evaluate(
            batch, labels, metrics=["coverage"],
            metric_parameters={"coverage": {"min_assets": 5}},
        )
        stored = adapter.get_evidence(result["evaluation_id"])
        baseline_stored = adapter.get_evidence(baseline["evaluation_id"])
        assert stored["config_hash"] != baseline_stored["config_hash"]
    else:
        with pytest.raises(ValueError, match="refusing to silently drop"):
            adapter.evaluate(
                batch, labels, metrics=["coverage"],
                metric_parameters={"coverage": {"minimum": 0.9}},
            )


def test_metric_instances_and_runtime_scenarios_reach_public_qe_and_return_children():
    from factor_optimizer.adapters.quant_evaluator import InMemoryEvidenceStore, create_qe_adapter
    from quant_evaluator.contracts.metric_instance import MetricInstance, EvaluationScenario

    batch, labels = _fixtures(("a",))
    instance = MetricInstance("coverage", scenario_id="gross_h1", horizon=1,
                              price_convention=labels.price_convention)
    scenario = EvaluationScenario(label_bundle=labels)
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    result = adapter.evaluate(
        batch, labels, metrics=["coverage"], metric_instances=(instance,),
        scenario_inputs={"gross_h1": scenario},
    )
    assert set(result["instance_specs"]) == {instance.instance_id}
    assert set(result["instance_results"]) == {instance.instance_id}
    child = result["instance_results"][instance.instance_id]
    assert child["metric_versions"] == {"coverage": instance.metric_version}


def test_real_public_adapter_cuda_batch_path():
    pytest.importorskip("cupy")
    from factor_optimizer.adapters.quant_evaluator import InMemoryEvidenceStore, create_qe_adapter

    batch, labels = _fixtures(("gpu-a", "gpu-b", "gpu-c"))
    adapter = create_qe_adapter(evidence_store=InMemoryEvidenceStore())
    result = adapter.evaluate(batch, labels, metrics=["rank_ic"], backend="cuda")
    assert set(result["grouped_metrics"]) == set(batch.factor_ids)
    for metric in result["grouped_metrics"].values():
        assert metric["rank_ic"]["observation_count"] > 0
        assert metric["rank_ic"]["metric_version"]
    assert adapter.get_evidence(result["evaluation_id"])["requested_backend"] == "cuda"
