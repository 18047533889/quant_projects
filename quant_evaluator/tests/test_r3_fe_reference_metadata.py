"""FE provenance transport, not real-data or GPU evaluation acceptance."""
import json
import pytest
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.api.requests import EvaluationRequest


@pytest.mark.parametrize("kind", [FactorValueRef, LabelBundleRef])
def test_nested_unverified_provenance_json_roundtrip_is_detached(kind):
    metadata = {"schema_version": "factor_engine.factor_value_ref.v1",
                "execution_purpose": {"purpose": "research_compute", "assurance": "UNVERIFIED",
                                      "publication_authorized": False},
                "run_identity": {"digest": "a" * 64}, "sources": [{"id": "approved"}]}
    ref = kind("artifact", metadata=metadata)
    encoded = ref.to_dict()
    assert kind.from_dict(json.loads(json.dumps(encoded))) == ref
    encoded["metadata"]["execution_purpose"]["assurance"] = "PASS"
    encoded["metadata"]["sources"][0]["id"] = "other"
    assert ref.metadata["execution_purpose"]["assurance"] == "UNVERIFIED"
    assert ref.metadata["sources"][0]["id"] == "approved"
    with pytest.raises(TypeError):
        ref.metadata["execution_purpose"]["assurance"] = "PASS"


def test_evaluation_request_preserves_nested_factor_reference():
    ref = FactorValueRef("artifact", ("factor",), metadata={
        "execution_purpose": {"purpose": "research_compute", "assurance": "UNVERIFIED"},
        "run_identity": {"digest": "b" * 64}})
    request = EvaluationRequest(batch_or_factor_ids=None, label_bundle=None,
                                factor_value_ref=ref, tier="research")
    restored = EvaluationRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert restored.factor_value_ref == ref


@pytest.mark.parametrize("instances", [False, True])
def test_real_cpu_evaluation_keeps_unverified_reference(instances, tmp_path):
    import numpy as np
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.contracts.metric_instance import MetricInstance
    from quant_evaluator.runtime.evaluator import evaluate
    from quant_evaluator.contracts.errors import InvalidContractError
    from dataclasses import replace
    import pandas as pd
    import hashlib
    from factor_engine.runtime.default_execution_policy import ExecutionPurpose, resolve_default_policy
    from factor_engine.runtime.durable_artifact_sink import write_verified_factor_artifact, read_verified_factor_artifact
    from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker
    values = np.random.default_rng(17).normal(size=(8, 40))
    # Real FE parquet/hash/readback with a synthetic budget broker. This does
    # not stand in for approved DataAccess input or process-family RSS proof.
    frame = pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=8),
                         columns=[f"A{i:02d}" for i in range(40)])
    frame.index.name, frame.columns.name = "timestamp", "instrument"
    series = frame.stack()
    purpose, policy = ExecutionPurpose(), resolve_default_policy()
    identity_digest = hashlib.sha256(b"synthetic-test-only").hexdigest()
    artifact = write_verified_factor_artifact(
        tmp_path, "a" * 32, 0, "f", series, policy=policy,
        budget_bytes=8 * 1024 * 1024, execution_purpose=purpose,
        run_identity_digest=identity_digest)
    restored = read_verified_factor_artifact(
        artifact, tmp_path, policy=policy, budget_bytes=8 * 1024 * 1024,
        broker=FakeBroker(), execution_purpose=purpose,
        run_identity_digest=identity_digest)
    pd.testing.assert_series_equal(restored["value"], series.rename("factor_value"))
    values = restored["value"].unstack().to_numpy()
    times = tuple(frame.index.to_numpy())
    batch = FactorBatch(("f",), AxisRef("time", str(frame.index.dtype), 8, frame.index.to_numpy()),
                        AxisRef("asset", "str", 40, np.asarray(frame.columns, dtype=str)),
                        values[:, :, None])
    labels = LabelBundle("synthetic", values * 2, 1, decision_time=times,
                         label_start_time=times, label_end_time=tuple(t + np.timedelta64(1, "D") for t in times),
                         asset_axis=batch.asset_axis)
    ref = FactorValueRef(restored["artifact_id"], ("f",),
                         metadata=restored["manifest_projection"])
    request = EvaluationRequest(batch, labels, tier="research", factor_value_ref=ref,
        metric_ids=() if instances else ("rank_ic_series",),
        metric_instances=(MetricInstance("rank_ic_series", horizon=1),) if instances else ())
    bundle = evaluate(request, backend="cpu")
    assert bundle.metadata["factor_value_ref"] == ref.to_dict()
    children = tuple(bundle.instance_results.values()) if instances else (bundle,)
    for child in children:
        assert child.metadata["factor_value_ref"]["metadata"]["execution_purpose"]["assurance"] == "UNVERIFIED"
        assert child.artifacts or child.factor_artifacts
        np.testing.assert_allclose(child.artifacts["rank_ic_series"].values,
                                   np.ones((8, 1)), rtol=1e-12, atol=1e-12)
    bad = replace(request, factor_value_ref=FactorValueRef("bad", ("other",)))
    with pytest.raises(InvalidContractError, match="factor axis"):
        evaluate(bad, backend="cpu")
