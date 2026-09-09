import numpy as np
import pytest

from factor_assets.adapters.quant_evaluator import QEEvidenceProvider
from factor_assets.adapters.quant_evaluator import validate_policy_runtime_capability
from factor_assets.profiling.policies import AdmissionFloors, get_health_policy
from dataclasses import replace
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def _inputs(*, factor_id="factor-a", with_identity=True, validity=None,
            value_hash=None, time_values=None, asset_values=None):
    rng = np.random.default_rng(1701)
    times = np.arange(30) if time_values is None else np.asarray(time_values)
    assets = np.arange(40) if asset_values is None else np.asarray(asset_values)
    values = rng.normal(size=(30, 40))
    context = {}
    if with_identity:
        context = {
            "factor_definition_refs": {factor_id: "factor-definition:A"},
            "factor_value_ref": "factor-value:A@snapshot:1",
        }
    batch = FactorBatch(
        (factor_id,), AxisRef("time", "int64", 30, times),
        AxisRef("asset", "int64", 40, assets), values[..., None],
        validity=validity, context_refs=context, value_hash=value_hash,
    )
    if times.dtype.hasobject:
        label_start = tuple(f"{item}:1" for item in times)
        label_end = tuple(f"{item}:2" for item in times)
    else:
        label_start, label_end = tuple(times + 1), tuple(times + 2)
    label = LabelBundle(
        "forward", values * .01, 1, decision_time=tuple(times),
        label_start_time=label_start, label_end_time=label_end,
        asset_axis=batch.asset_axis,
    )
    request = EvaluationRequest(
        batch, label, metric_ids=("rank_ic", "coverage"),
        metadata={"request_id": "evaluation:e2e-a-typed"},
    )
    return batch, label, evaluate(request)


def test_real_qe_cpu_bundle_grades_metrics_with_complete_source_provenance():
    batch, _, bundle = _inputs()
    grades = QEEvidenceProvider().grade_typed_metrics(
        bundle, batch, expected_config_hash=bundle.config_hash,
        expected_evaluation_ref="evaluation:e2e-a-typed",
    )
    by_metric = {grade.metric_id: grade for grade in grades}
    assert set(by_metric) == {"coverage", "rank_ic"}
    assert by_metric["rank_ic"].value == pytest.approx(1.0)
    assert by_metric["rank_ic"].evidence_status == "COMPUTED"
    assert by_metric["rank_ic"].grade is not None
    assert by_metric["rank_ic"].factor_definition_id == "factor-definition:A"
    assert by_metric["rank_ic"].factor_value_ref == "factor-value:A@snapshot:1"
    assert by_metric["rank_ic"].factor_axis_ref.startswith("factor-axis:")
    assert by_metric["rank_ic"].config_hash == bundle.config_hash
    assert by_metric["rank_ic"].evaluation_ref == bundle.request_id
    assert by_metric["rank_ic"].production_provenance_complete is True


def test_swapped_factor_axis_values_and_config_are_rejected():
    batch, _, bundle = _inputs()
    provider = QEEvidenceProvider()
    other, _, _ = _inputs(factor_id="factor-b")
    with pytest.raises(ValueError, match="factor identity"):
        provider.grade_typed_metrics(bundle, other)

    swapped_axis = FactorBatch(
        batch.factor_ids, batch.time_axis,
        AxisRef("asset", "int64", 40, np.arange(40)[::-1]),
        batch.values, context_refs=batch.context_refs,
    )
    with pytest.raises(ValueError, match="asset_coordinates"):
        provider.grade_typed_metrics(bundle, swapped_axis)

    changed_values = np.array(batch.values, copy=True)
    changed_values[0, 0, 0] += 1
    changed_batch = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, changed_values,
        context_refs=batch.context_refs,
    )
    with pytest.raises(ValueError, match="factor values"):
        provider.grade_typed_metrics(bundle, changed_batch)
    with pytest.raises(ValueError, match="config_hash"):
        provider.grade_typed_metrics(bundle, batch, expected_config_hash="config:wrong")


def test_external_value_hash_cannot_hide_changed_values_or_validity():
    validity = np.ones((30, 40, 1), dtype=bool)
    batch, _, bundle = _inputs(validity=validity, value_hash="external:same")
    changed_values = np.array(batch.values, copy=True)
    changed_values[1, 1, 0] += 7
    forged = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, changed_values,
        validity=validity, context_refs=batch.context_refs,
        value_hash="external:same",
    )
    with pytest.raises(ValueError, match="factor values"):
        QEEvidenceProvider().grade_typed_metrics(bundle, forged)
    changed_validity = validity.copy()
    changed_validity[0, 0, 0] = False
    remasked = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, batch.values,
        validity=changed_validity, context_refs=batch.context_refs,
        value_hash="external:same",
    )
    with pytest.raises(ValueError, match="factor validity"):
        QEEvidenceProvider().grade_typed_metrics(bundle, remasked)


def test_object_axes_survive_bundle_json_roundtrip_and_empty_context_mismatch_fails():
    import json
    from quant_evaluator.api.requests import EvaluationBundle

    object_times = np.asarray([f"2026-01-{day:02d}" for day in range(1, 31)], dtype=object)
    object_assets = np.asarray([f"asset-{i}" for i in range(40)], dtype=object)
    batch, _, bundle = _inputs(time_values=object_times, asset_values=object_assets)
    restored = EvaluationBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
    grades = QEEvidenceProvider().grade_typed_metrics(restored, batch)
    assert all(grade.production_provenance_complete for grade in grades)
    no_context = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, batch.values,
        context_refs={},
    )
    with pytest.raises(ValueError, match="context refs"):
        QEEvidenceProvider().grade_typed_metrics(restored, no_context)


def test_absent_coordinates_cannot_produce_complete_axis_provenance():
    rng = np.random.default_rng(1702)
    values = rng.normal(size=(30, 40))
    batch = FactorBatch(
        ("factor-a",), AxisRef("time", "int64", 30),
        AxisRef("asset", "int64", 40), values[..., None],
        context_refs={
            "factor_definition_refs": {"factor-a": "factor-definition:A"},
            "factor_value_ref": "factor-value:A@snapshot:1",
        },
    )
    label = LabelBundle(
        "forward", values * .01, 1, decision_time=tuple(range(30)),
        label_start_time=tuple(range(1, 31)), label_end_time=tuple(range(2, 32)),
    )
    bundle = evaluate(EvaluationRequest(
        batch, label, metric_ids=("rank_ic",),
        metadata={"request_id": "evaluation:no-coordinates"},
    ))
    grade, = QEEvidenceProvider().grade_typed_metrics(bundle, batch)
    assert grade.factor_axis_ref == ""
    assert grade.evidence_status == "UNKNOWN"
    assert grade.production_provenance_complete is False


def test_missing_identity_is_explicitly_unknown_and_ungraded():
    batch, _, bundle = _inputs(with_identity=False)
    grades = QEEvidenceProvider().grade_typed_metrics(bundle, batch)
    assert grades
    assert all(grade.evidence_status == "UNKNOWN" for grade in grades)
    assert all(grade.grade is None and grade.desirability is None for grade in grades)
    assert all(not grade.production_provenance_complete for grade in grades)


def test_v6_exposure_semantics_version_reaches_fa_evidence_provenance():
    from quant_evaluator.metrics.exposure_evidence import ExposurePanel

    batch, label, _ = _inputs()
    values = np.asarray(batch.values[:, :, 0])
    panel = ExposurePanel(
        values[:, :, None], style_names=("size",),
        source_ref="risk:versioned-v6", provider="test-provider",
        date_index=tuple(label.decision_time),
        security_ids=tuple(batch.asset_axis.values.tolist()),
        factor_ids=batch.factor_ids, universe_snapshot_ref="universe:test",
    )
    bundle = evaluate(EvaluationRequest(
        batch, label, metric_ids=("size_exposure", "purity_ratio"),
        tier="extended",
        exposure_panel=panel,
        metadata={"request_id": "evaluation:v6-exposure-version"},
    ))
    grades = QEEvidenceProvider().grade_typed_metrics(bundle, batch)
    by_metric = {grade.metric_id: grade for grade in grades}
    assert by_metric["size_exposure"].metric_version == "4.0.0"
    assert by_metric["purity_ratio"].metric_version == "4.0.0"
    assert all(grade.production_provenance_complete for grade in grades)
# V5 D12: required policy metric -> QE alias -> executable sealed registry.
def test_v5_policy_required_metric_compiles_to_real_qe_runtime_capability():
    from quant_evaluator.registry.metrics import seal_metric_registry, registry_state
    if registry_state() != "sealed":
        seal_metric_registry()
    base = get_health_policy()
    predictive_only = replace(
        base,
        use_case_admission_floors={
            **dict(base.use_case_admission_floors),
            "CAPABILITY_TEST": AdmissionFloors(
                use_case="CAPABILITY_TEST", required_dimension_ids=("predictive_power",),
                require_all_dimensions_graded=False,
            ),
        },
    )
    compiled = validate_policy_runtime_capability(predictive_only, use_case="CAPABILITY_TEST")
    assert compiled["rank_ic"] == {
        "runtime_metric_id": "rank_ic", "metric_version": "3.0.0", "status": "stable"
    }
    from quant_evaluator.registry.metrics import resolve_alias, get_metric
    for alias in ("rank_icir_raw", "rank_ic_ir", "rankicir", "icir"):
        assert resolve_alias(alias) == "ic_ir"
        spec = get_metric(resolve_alias(alias))
        assert spec.compute_fn is not None and spec.metric_version == "3.0.0"


def test_v5_policy_required_unexecutable_metric_fails_closed_not_empty_gate():
    base = get_health_policy()
    unavailable = replace(
        base,
        metric_grade_rules={
            **dict(base.metric_grade_rules),
            "cost_drag": replace(
                base.metric_rule("cost_drag"),
                runtime_metric_id="definitely_not_registered",
            ),
        },
    )
    with pytest.raises(ValueError, match="NOT_REGISTERED"):
        validate_policy_runtime_capability(unavailable, use_case="LONG_ONLY_RESEARCH")


def _real_turnover_cost_bundle():
    from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact

    batch, label, _ = _inputs()
    costs = ProbePortfolioArtifact(
        np.full((30, 1), 0.0015),
        time_index=tuple(range(30)),
        factor_ids=batch.factor_ids,
        provenance={"leg": "cost_drag", "trajectory_artifact_id": "trajectory:real-cost"},
    )
    bundle = evaluate(EvaluationRequest(
        batch,
        label,
        metric_ids=("turnover_cost",),
        tier="extended",
        portfolio_returns=costs,
        metadata={"request_id": "evaluation:real-turnover-cost"},
    ))
    return batch, bundle


def test_d12_real_typed_turnover_cost_binds_fraction_and_explicit_budget_ratio():
    batch, bundle = _real_turnover_cost_bundle()
    assert bundle.metric_values["turnover_cost"].value == pytest.approx(15.0)
    assert bundle.metric_values["turnover_cost"].metric_version == "3.0.0"

    grades = QEEvidenceProvider().grade_typed_metrics(
        bundle,
        batch,
        expected_evaluation_ref="evaluation:real-turnover-cost",
        cost_budget_bps=20.0,
    )
    by_metric = {grade.metric_id: grade for grade in grades}
    assert set(by_metric) == {"cost_drag", "cost_budget_utilization"}
    assert by_metric["cost_drag"].value == pytest.approx(0.0015)
    assert by_metric["cost_budget_utilization"].value == pytest.approx(0.75)
    assert all(grade.production_provenance_complete for grade in grades)


def test_d12_cost_budget_utilization_is_ungraded_without_explicit_budget():
    batch, bundle = _real_turnover_cost_bundle()
    grades = QEEvidenceProvider().grade_typed_metrics(bundle, batch)
    assert len(grades) == 1
    assert grades[0].metric_id == "cost_drag"
    assert grades[0].value == pytest.approx(0.0015)
    for invalid in (0, -1, np.inf, True):
        with pytest.raises(ValueError, match="cost_budget_bps"):
            QEEvidenceProvider().grade_typed_metrics(
                bundle, batch, cost_budget_bps=invalid
            )
