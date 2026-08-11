from __future__ import annotations

import numpy as np
import pytest

from modeling.benchmark import (
    PREDICTION_SIZES, RETRAIN_SIZES, RuntimeBoundary,
    deterministic_resource_parity, seed_stability, validate_benchmark_sizes,
)
from modeling.ledger import (
    ArtifactDependencyDAG, ArtifactSchemaMigration, FeatureBundle, FrozenModelSchema,
    ModelArtifactIdentity, ModelFitFingerprint, PredictionCacheIdentity, PredictionRow,
    RetrainFingerprint, RuntimeEnvironment, TrainingRun,
)
from modeling.monitoring import (
    DRIFT_DIMENSIONS, DataQualityCertificate, DriftContract, DriftThreshold,
    ModelCostContract, MonitorAction, RealisedMetric, production_drift_metrics,
)


def test_production_drift_contract_has_all_dimensions_and_distinct_actions():
    thresholds = {name: DriftThreshold(0.05, retrain=0.2, block=0.5) for name in DRIFT_DIMENSIONS}
    contract = DriftContract(thresholds)
    actions = contract.assess({name: (0.1 if name == "psi" else 0.3 if name == "ks" else 0.7)
                               for name in DRIFT_DIMENSIONS})
    assert actions["psi"] is MonitorAction.MONITOR
    assert actions["ks"] is MonitorAction.RETRAIN
    assert actions["wasserstein"] is MonitorAction.PROMOTION_BLOCK
    with pytest.raises(ValueError, match="missing drift metrics"):
        contract.assess({"psi": 0.0})


def test_drift_metric_scaffolding_computes_all_nine_metrics():
    ref = np.arange(20.0)
    got = production_drift_metrics(
        ref, ref + 2, reference_prediction=ref, current_prediction=ref[::-1],
        reference_coefficient=[1, 2], current_coefficient=[2, 3],
        reference_ic=.1, current_ic=.03, reference_coverage=.9, current_coverage=.7,
        reference_regime_occupancy=[.7, .3], current_regime_occupancy=[.4, .6],
        reference_missingness=.01, current_missingness=.05,
    )
    assert set(got) == set(DRIFT_DIMENSIONS)
    assert all(value >= 0 for value in got.values())


def test_realised_metric_is_unavailable_until_horizon_matures():
    metric = RealisedMetric("p1", "rank_ic", None, 20, "2026-01-01", "2026-02-01", "2026-01-10")
    with pytest.raises(ValueError, match="not PIT-available"):
        metric.require_available("2026-01-31")
    metric.require_available("2026-02-01")


def test_prediction_row_and_cache_bind_artifact_and_snapshots():
    row = PredictionRow("2026-01-01", "000001", .2, "a1", "v1", "2025-12-31",
                        "2026-01-01", "schema", "source", "feature-1", "feature-hash")
    row.validate()
    key1 = PredictionCacheIdentity("a1", "feature-1", "feature-hash", "source", row.date, row.stock).key()
    key2 = PredictionCacheIdentity("a2", "feature-1", "feature-hash", "source", row.date, row.stock).key()
    assert key1 != key2
    with pytest.raises(ValueError, match="missing identity"):
        PredictionRow(*((*row.__dict__.values(),) if False else (row.date, row.stock, row.prediction,
            row.artifact_id, row.model_version, row.training_cutoff, row.available_at,
            row.feature_schema_hash, row.source_snapshot_hash, "", row.feature_snapshot_hash))).validate()


def test_training_run_is_distinct_from_content_addressed_artifact_and_retrain_is_idempotent():
    fingerprint = RetrainFingerprint("enet-v1", "2026-01-01", "snap", "policy", "code")
    assert fingerprint.value() == RetrainFingerprint("enet-v1", "2026-01-01", "snap", "policy", "code").value()
    run1 = TrainingRun("run-1", fingerprint.value(), {"host": "a"}, "t0")
    run2 = TrainingRun("run-2", fingerprint.value(), {"host": "b"}, "t1")
    a1 = ModelArtifactIdentity.from_content({"weights": [1, 2]}, "fit", run1.run_id)
    a2 = ModelArtifactIdentity.from_content({"weights": [1, 2]}, "fit", run2.run_id)
    assert a1.artifact_id == a2.artifact_id
    assert a1.producing_run_id != a2.producing_run_id


def test_reproducibility_and_seed_stability_contracts():
    env = RuntimeEnvironment({"numpy": "2"}, "openblas", "cpu", ("deterministic",), 1)
    assert len(env.runtime_environment_hash) == 64
    fp = ModelFitFingerprint(*(str(i) for i in range(7)))
    assert fp.value() == ModelFitFingerprint(*(str(i) for i in range(7))).value()
    output = np.arange(4.0)
    deterministic_resource_parity({name: output.copy() for name in
                                   ("single_thread", "multi_thread", "batch_size", "shard_order")})
    with pytest.raises(AssertionError):
        deterministic_resource_parity({"single_thread": output, "multi_thread": output + 1,
                                       "batch_size": output, "shard_order": output})
    assert seed_stability({1: .10, 2: .11, 3: .09}, max_spread=.03)
    assert not seed_stability({1: .10}, max_spread=.03)


def test_cost_and_data_quality_gate_training_and_validation():
    cost = ModelCostContract(10, 1000, 100, .1, 100)
    assert cost.failures({"fit_seconds": 9, "peak_memory_bytes": 900, "artifact_bytes": 90,
                          "prediction_latency_seconds": .05, "rows_per_second": 200}) == []
    assert "fit_seconds" in cost.failures({"fit_seconds": 11, "peak_memory_bytes": 900,
        "artifact_bytes": 90, "prediction_latency_seconds": .05, "rows_per_second": 200})
    cert = DataQualityCertificate("c1", "validation", "f", "u", "s", True)
    cert.require_valid("validation")
    with pytest.raises(ValueError, match="data quality gate"):
        cert.require_valid("training")


def test_feature_bundle_dependency_invalidation_and_explicit_schema_migration():
    bundle = FeatureBundle("x", "2", {"w": 20}, "ast", "snapshot", "1.0.0")
    assert bundle.identity() != FeatureBundle("x", "3", {"w": 20}, "ast", "snapshot", "1.0.0").identity()
    dag = ArtifactDependencyDAG()
    dag.add("model", "feature", "dataset", "universe", "label", "run", "code")
    dag.add("ensemble", "model")
    assert {"model", "ensemble"} <= dag.invalidate({"feature"})
    migration = ArtifactSchemaMigration("1", "2", "model-v1-to-v2")
    assert migration.convert({"x": 1}, lambda p: {**p, "y": 2}) == {"x": 1, "y": 2}


def test_per_learner_frozen_schema_rejects_fields_types_and_shapes():
    schema = FrozenModelSchema("enet", "2", ("coef", "intercept"),
                               {"coef": np.ndarray, "intercept": float}, {"coef": (2,)})
    schema.validate({"coef": np.ones(2), "intercept": 0.0})
    with pytest.raises(ValueError, match="fields"):
        schema.validate({"coef": np.ones(2)})
    with pytest.raises(ValueError, match="shape"):
        schema.validate({"coef": np.ones(3), "intercept": 0.0})


def test_real_scale_benchmark_scaffold_and_runtime_training_boundary():
    assert PREDICTION_SIZES == (1_000, 3_000, 5_000, 10_000)
    assert RETRAIN_SIZES == (100_000, 500_000, 1_000_000, 5_000_000)
    validate_benchmark_sizes()
    RuntimeBoundary.require_runtime("frozen_score")
    for operation in RuntimeBoundary.TRAINING_OPERATIONS:
        with pytest.raises(RuntimeError, match="cannot perform"):
            RuntimeBoundary.require_runtime(operation)
