import numpy as np

from quant_evaluator.api.batch_bundle import BatchEvaluationBundle
from quant_evaluator.api.evaluate_many import evaluate_many
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import (
    Evaluator,
    authoritative_array_hash,
    evaluate,
)


def _inputs():
    times = np.arange(30)
    assets = np.arange(24)
    batch = FactorBatch(
        ("f0", "f1"),
        AxisRef("time", "int64", len(times), times),
        AxisRef("asset", "int64", len(assets), assets),
        np.arange(len(times) * len(assets) * 2, dtype=np.float64).reshape(
            len(times), len(assets), 2,
        ),
    )
    base = np.sin(np.arange(len(times) * len(assets), dtype=np.float64)).reshape(
        len(times), len(assets),
    )
    labels = tuple(
        LabelBundle(
            f"h{horizon}", base + horizon, horizon,
            decision_time=tuple(times), observation_time=tuple(times),
            label_start_time=tuple(times + horizon),
            label_end_time=tuple(times + horizon + 1),
            asset_axis=batch.asset_axis,
        )
        for horizon in (1, 2)
    )
    return batch, labels


class _ProvenanceSpy(Evaluator):
    def __init__(self):
        super().__init__()
        self.provenance_calls = 0

    def _provenance(self, factor_batch, label_bundle):
        self.provenance_calls += 1
        return super()._provenance(factor_batch, label_bundle)


def _assert_same_scalar_artifacts(actual, expected):
    assert actual.config_hash == expected.config_hash
    assert actual.metadata["provenance"] == expected.metadata["provenance"]
    assert actual.metric_values.keys() == expected.metric_values.keys()
    for metric_id in expected.metric_values:
        left = actual.metric_values[metric_id]
        right = expected.metric_values[metric_id]
        assert (left.value, left.valid, left.observation_count,
                left.metric_version) == (
            right.value, right.valid, right.observation_count,
            right.metric_version,
        )
    assert actual.artifacts.keys() == expected.artifacts.keys()
    for metric_id in expected.artifacts:
        left = actual.artifacts[metric_id]
        right = expected.artifacts[metric_id]
        assert type(left) is type(right)
        np.testing.assert_equal(left.values, right.values)
        assert left.provenance == right.provenance


def test_cpu_facade_reuses_current_runtime_provenance_once(monkeypatch):
    batch, labels = _inputs()
    label = labels[0]
    expected = evaluate(batch, label, backend="cpu", metrics=("coverage", "rank_ic"))
    provenance_calls = []
    original_provenance = Evaluator._provenance

    def count_provenance(runtime, factor_batch, label_bundle):
        provenance_calls.append(label_bundle.target_id)
        return original_provenance(runtime, factor_batch, label_bundle)

    monkeypatch.setattr(Evaluator, "_provenance", count_provenance)
    actual = evaluate(
        batch, label, backend="cpu", metrics=("coverage", "rank_ic"),
    )
    assert provenance_calls == [label.target_id]
    _assert_same_scalar_artifacts(actual, expected)

    changed = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis,
        np.asarray(batch.values) + 1.0,
    )
    changed_result = evaluate(changed, label, backend="cpu", metrics=("coverage", "rank_ic"))
    assert provenance_calls == [label.target_id, label.target_id]
    assert (changed_result.metadata["provenance"]["factor_value_bytes_hash"]
            != actual.metadata["provenance"]["factor_value_bytes_hash"])


def test_gpu_override_still_uses_current_batch_provenance_once():
    batch, labels = _inputs()
    label = labels[0]
    fake_gpu = BatchEvaluationBundle(
        factor_ids=batch.factor_ids,
        label_id=label.target_id,
        scalar_metrics={"coverage": np.ones(batch.num_factors)},
        observation_counts={
            "coverage": np.full(batch.num_factors, batch.num_times * batch.num_assets),
        },
        metadata={"provenance": {"factor_value_bytes_hash": "forged"}},
    )
    runtime = _ProvenanceSpy()
    result = evaluate(
        batch, label, metrics=("coverage",), evaluator=runtime,
        _gpu_result_override=fake_gpu,
    )
    assert runtime.provenance_calls == 1
    assert (result.metadata["provenance"]["factor_value_bytes_hash"]
            == authoritative_array_hash(batch.values))
    assert result.metadata["provenance"]["factor_value_bytes_hash"] != "forged"


def test_injected_runtime_provenance_is_recomputed(monkeypatch):
    batch, labels = _inputs()
    label = labels[0]

    class ForgingEvaluator(_ProvenanceSpy):
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            result.provenance = {"factor_value_bytes_hash": "forged"}
            return result

    runtime = ForgingEvaluator()
    result = evaluate(batch, label, backend="cpu", metrics=("coverage",), evaluator=runtime)
    assert runtime.provenance_calls == 2
    assert (result.metadata["provenance"]["factor_value_bytes_hash"]
            == authoritative_array_hash(batch.values))
    assert result.metadata["provenance"]["factor_value_bytes_hash"] != "forged"


def test_evaluate_many_keeps_label_provenance_and_scalar_artifacts(monkeypatch):
    from quant_evaluator.api.evaluate_many import evaluate_many as run_many

    batch, labels = _inputs()
    expected = {
        label.target_id: evaluate(batch, label, backend="cpu", metrics=("coverage",))
        for label in labels
    }
    provenance_calls = []
    original_provenance = Evaluator._provenance

    def count_provenance(runtime, factor_batch, label_bundle):
        provenance_calls.append(label_bundle.target_id)
        return original_provenance(runtime, factor_batch, label_bundle)

    monkeypatch.setattr(Evaluator, "_provenance", count_provenance)
    actual = run_many(batch, labels, backend="cpu", metrics=("coverage",))

    assert sorted(provenance_calls) == sorted(label.target_id for label in labels)
    assert actual.keys() == expected.keys()
    value_hashes = set()
    label_hashes = set()
    for label in labels:
        result = actual[label.target_id]
        baseline = expected[label.target_id]
        _assert_same_scalar_artifacts(result, baseline)
        provenance = result.metadata["provenance"]
        value_hashes.add(provenance["factor_value_bytes_hash"])
        label_hashes.add(provenance["label_value_hash"])
    assert len(value_hashes) == 1
    assert len(label_hashes) == len(labels)
