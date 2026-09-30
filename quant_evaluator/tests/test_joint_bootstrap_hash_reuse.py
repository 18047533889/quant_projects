import numpy as np

from quant_evaluator.contracts import resampling
from quant_evaluator.contracts.resampling import ResamplingPlan
from quant_evaluator.metrics.robustness import compute_joint_block_bootstrap


def test_joint_bootstrap_reuses_one_plan_digest_for_provenance_and_ids(monkeypatch):
    plan = ResamplingPlan(
        time_ids=tuple(range(20)),
        clock_ref="joint-bootstrap-hash-test",
        block_length=5,
        num_replicates=12,
        seed=19,
    )
    expected_hash = plan.content_hash
    expected_ids = plan.replicate_ids
    original = resampling.stable_content_hex
    calls = []

    def count_digest(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(resampling, "stable_content_hex", count_digest)
    values = np.arange(40.0).reshape(20, 2)
    artifact = compute_joint_block_bootstrap(values, plan, ("a", "b"))

    assert len(calls) == 1
    assert artifact.provenance["resampling_plan_ref"] == expected_hash
    assert artifact.provenance["resampling_plan_content_hash"] == expected_hash
    assert artifact.provenance["replicate_ids"] == expected_ids
