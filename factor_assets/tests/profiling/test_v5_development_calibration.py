import numpy as np
import pytest

from quant_evaluator.metrics.ic_summary import compute_icir
from factor_assets.profiling.calibration import (
    CalibrationObservation, DevelopmentCalibrationArtifact,
    build_development_calibration, build_policy_rescore_trace,
)
from factor_assets.profiling.metric_grading import grade_metric_evidence
from factor_assets.profiling.policies import get_health_policy


def _qe_observation(fid, family, mean, std, seed):
    rng = np.random.default_rng(seed)
    series = rng.normal(size=64)
    series = (series - series.mean()) / series.std(ddof=1) * std + mean
    raw = float(compute_icir(series[:, None], min_periods=20)[0])
    return CalibrationObservation(
        fid, family, f"qe:evaluation:{fid}", float(series.mean()), raw,
        split_role="DEVELOPMENT",
    )


def test_n06_real_qe_synthetic_family_aware_persistent_development_artifact(tmp_path):
    observations = [
        _qe_observation("mom-a", "momentum", .02, .08, 1),
        _qe_observation("mom-b", "momentum", -.002, .09, 2),  # failure retained
        _qe_observation("mom-c", "momentum", .01, .10, 3),
        _qe_observation("mom-d", "momentum", .04, .07, 4),  # family cap prevents crowding
        _qe_observation("val-a", "value", .005, .08, 5),
        _qe_observation("val-b", "value", -.005, .09, 6),
        _qe_observation("evt-a", "event", .015, .12, 7),
    ]
    artifact = build_development_calibration(
        observations, policy_id=get_health_policy().policy_id,
        cutoff_as_of="2026-01-01T00:00:00Z", synthetic_example=True,
        max_per_family=3,
    )
    assert artifact.synthetic_example and not artifact.admission_authority
    assert dict(artifact.family_counts)["momentum"] == 3
    assert artifact.observation_count == 6
    assert any(o.rank_ic < 0 for o in observations if o.evaluation_ref in artifact.source_evaluation_refs)
    path = tmp_path / "synthetic-development-calibration.json"
    artifact.persist(path)
    assert DevelopmentCalibrationArtifact.load(path) == artifact


def test_n06_rejects_holdout_and_policy_rescore_reuses_qe_raw_without_fe_materialization():
    with pytest.raises(ValueError, match="holdout"):
        CalibrationObservation("x", "f", "qe:x", .1, .2, split_role="TEST")
    raw = {"qe:evaluation:x": {"rank_ic": .02, "rank_icir_raw": .25}}
    trace = build_policy_rescore_trace(
        old_policy_ref="policy:1.0.0", new_policy_ref="policy:2.0.0", raw_metrics=raw,
    )
    old = grade_metric_evidence(metric_id="rank_ic", value=.02, evidence_status="computed",
                                policy=get_health_policy(policy_version="1.0.0"),
                                evaluation_ref="qe:evaluation:x")
    new = grade_metric_evidence(metric_id="rank_ic", value=.02, evidence_status="computed",
                                policy=get_health_policy(), evaluation_ref="qe:evaluation:x")
    assert old.evaluation_ref == new.evaluation_ref == trace.source_evaluation_refs[0]
    assert old.grade != new.grade
    assert trace.recomputed_metric_ids == trace.rematerialized_factor_refs == ()
