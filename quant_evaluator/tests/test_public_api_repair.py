import inspect

import numpy as np


def _contracts():
    from quant_evaluator import AxisRef, FactorBatch, LabelBundle

    batch = FactorBatch(
        factor_ids=("f",),
        time_axis=AxisRef(name="time", dtype="int64", size=3, values=np.arange(3)),
        asset_axis=AxisRef(name="asset", dtype="int64", size=2, values=np.arange(2)),
        values=np.array([[[1.0], [2.0]], [[2.0], [4.0]], [[3.0], [6.0]]]),
    )
    labels = LabelBundle(
        target_id="ret",
        values=np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        horizon=1,
        decision_time=(1, 2, 3),
        label_start_time=(1, 2, 3),
        label_end_time=(2, 3, 4),
    )
    return batch, labels


def test_public_imports_and_signature():
    from quant_evaluator import EvaluationBundle, Evaluator, MetricValue, evaluate

    assert inspect.isfunction(evaluate)
    assert Evaluator is not None
    assert EvaluationBundle is not None
    assert MetricValue is not None
    assert "factors" in inspect.signature(evaluate).parameters
    assert "labels" in inspect.signature(evaluate).parameters


def test_public_evaluate_returns_canonical_bundle():
    from quant_evaluator import EvaluationBundle, MetricValue, evaluate

    batch, labels = _contracts()
    result = evaluate(batch, labels, metrics=["coverage"])

    assert isinstance(result, EvaluationBundle)
    metric = result.get_metric("coverage")
    assert isinstance(metric, MetricValue)
    assert metric.metric_id == "coverage"
    assert metric.valid
    assert metric.value == 1.0
    assert metric.observation_count == 6
    assert result.factor_ids == ("f",)
    assert result.label_id == "ret"
    assert result.get_diagnosis("f").coverage == 1.0


def test_public_evaluate_supports_multiple_factors_without_scalar_aggregation():
    from quant_evaluator import evaluate

    batch, labels = _contracts()
    factor_one = np.tile(np.arange(10, dtype=float)[None, :, None], (3, 1, 1))
    values = np.concatenate([factor_one, factor_one + 1.0], axis=2)
    label_values = np.tile(np.arange(10, dtype=float)[None, :], (3, 1))
    batch = type(batch)(
        factor_ids=("f", "g"),
        time_axis=batch.time_axis,
        asset_axis=type(batch.asset_axis)(
            name="asset", dtype="int64", size=10, values=np.arange(10)
        ),
        values=values,
    )
    labels = type(labels)(
        target_id=labels.target_id,
        values=label_values,
        horizon=labels.horizon,
        decision_time=labels.decision_time,
        label_start_time=labels.label_start_time,
        label_end_time=labels.label_end_time,
    )
    result = evaluate(batch, labels, metrics=["coverage", "rank_ic"])

    assert result.metric_values == {}
    assert set(result.grouped_metrics) == {"f", "g"}
    assert result.get_metric("coverage", factor_id="g").observation_count == 30
    assert result.get_metric("rank_ic", factor_id="f").valid


def test_public_evaluate_rejects_slices_and_preserves_request_metadata():
    import pytest
    from quant_evaluator import EvaluationRequest, evaluate
    from quant_evaluator.contracts.errors import UnsupportedMetricError

    batch, labels = _contracts()
    request = EvaluationRequest(
        batch_or_factor_ids=batch,
        label_bundle=labels,
        metric_ids=("coverage",),
        slices={"year": 2026},
        metadata={"request_id": "req-7", "caller": "test"},
    )
    with pytest.raises(UnsupportedMetricError, match="slices"):
        evaluate(request)

    request = EvaluationRequest(
        batch_or_factor_ids=batch,
        label_bundle=labels,
        metric_ids=("coverage",),
        slices={},
        metadata={"request_id": "req-7", "caller": "test"},
    )
    with pytest.raises(UnsupportedMetricError, match="slices"):
        evaluate(request)

    request = EvaluationRequest(
        batch_or_factor_ids=batch,
        label_bundle=labels,
        metric_ids=("coverage",),
        metadata={"request_id": "req-7", "caller": "test"},
    )
    result = evaluate(request)
    assert result.request_id == "req-7"
    assert result.metadata["caller"] == "test"


def test_public_evaluate_rejects_empty_and_array_where():
    import pytest
    from quant_evaluator import evaluate
    from quant_evaluator.contracts.errors import UnsupportedMetricError

    batch, labels = _contracts()
    for where in ({}, np.array([], dtype=int)):
        with pytest.raises(UnsupportedMetricError, match="where"):
            evaluate(batch, labels, where=where)


def test_public_evaluate_rejects_label_asset_shape_mismatch():
    import pytest
    from quant_evaluator import evaluate
    from quant_evaluator.contracts.errors import InvalidContractError

    batch, labels = _contracts()
    mismatched = type(labels)(
        target_id=labels.target_id,
        values=np.ones((3, 1)),
        horizon=labels.horizon,
        decision_time=labels.decision_time,
        label_start_time=labels.label_start_time,
        label_end_time=labels.label_end_time,
        validity=np.ones((3, 1), dtype=bool),
    )
    with pytest.raises(InvalidContractError, match="asset axis"):
        evaluate(batch, mismatched)


def test_runtime_preserves_entire_public_error_taxonomy():
    import pytest
    from quant_evaluator import Evaluator
    from quant_evaluator.contracts.errors import (
        CapabilityError,
        ContractError,
        DataError,
        QuantEvaluatorError,
        TimingContractError,
    )
    from quant_evaluator.planner.dependency_plan import MetricKind

    batch, labels = _contracts()
    for error_type in (QuantEvaluatorError, ContractError, DataError, CapabilityError, TimingContractError):
        evaluator = Evaluator()

        def raises(**kwargs):
            raise error_type("preserve me")

        evaluator.register_metric("boom", raises, MetricKind.CUSTOM)
        with pytest.raises(error_type, match="preserve me"):
            evaluator.evaluate(batch, labels, [{"metric_id": "boom", "metric_kind": "custom"}], use_chunking=False)


def test_public_unsupported_metric_is_not_wrapped_by_runtime():
    import pytest
    from quant_evaluator import evaluate
    from quant_evaluator.contracts.errors import UnsupportedMetricError

    batch, labels = _contracts()
    with pytest.raises(UnsupportedMetricError, match="unsupported_metric"):
        evaluate(batch, labels, metrics=["unsupported_metric"])
