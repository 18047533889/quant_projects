import json
import time
from pathlib import Path

import pytest

import factor_engine.runtime.bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.run_deadline import restore_run_deadline
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeBroker, FakeEngine


@pytest.mark.parametrize("input_budget", [5, 300])
def test_original_deadline_is_durable_before_ingestion_and_used_for_stage(
    tmp_path, monkeypatch, input_budget,
):
    class StopIngestion(RuntimeError):
        pass

    policy = resolve_default_policy({"job_unknown_seconds": 20, "job_min_seconds": 1,
                                     "job_max_seconds": 30,
                                     "input_ingestion_deadline_seconds": input_budget})
    before = time.monotonic_ns()

    def inspect_ingestion(factors, path, **kwargs):
        identity = json.loads(Path(path).parent.joinpath("identity.json").read_text())
        record = identity["job_deadline"]
        assert record["duration_ns"] == 20_000_000_000
        assert before <= record["start_monotonic_ns"] <= time.monotonic_ns()
        restored = restore_run_deadline(
            record, expected_run_id=identity["run_id"], expected_policy_digest=policy.digest,
        )
        admission = kwargs["serialization_broker"]
        assert admission._job_deadline == restored.deadline_monotonic
        assert kwargs["deadline_monotonic"] == min(
            restored.deadline_monotonic, admission._input_deadline,
        )
        assert 0 < kwargs["deadline_seconds"] <= min(20, input_budget)
        assert kwargs["deadline_monotonic"] <= restored.deadline_monotonic
        if input_budget == 300:
            assert kwargs["deadline_monotonic"] == restored.deadline_monotonic
        else:
            assert kwargs["deadline_monotonic"] < restored.deadline_monotonic
        assert 0 < restored.remaining_seconds <= 20
        # Exercise enforcement of the persisted deadline, not a now-unused
        # helper spy. No renewed full job budget may enter admission.
        with monkeypatch.context() as clock_patch:
            clock_patch.setattr(pipeline.time, "monotonic", lambda: restored.deadline_monotonic + 1)
            with pytest.raises(pipeline.JobDeadlineExceeded):
                admission._check_deadlines()
        raise StopIngestion()

    monkeypatch.setattr(pipeline.FiniteFactorManifest, "ingest_supervised", inspect_ingestion)
    with pytest.raises(StopIngestion):
        pipeline.execute_run_many_durable(
            FakeEngine(), [], policy=policy, artifact_root=tmp_path,
            run_kwargs={"broker": FakeBroker()},
        )


def test_completed_revalidation_never_rewrites_original_deadline(tmp_path):
    policy = resolve_default_policy()
    kwargs = dict(policy=policy, artifact_root=tmp_path, run_kwargs={"broker": FakeBroker()})
    receipt = pipeline.execute_run_many_durable(FakeEngine(), [], **kwargs)
    identity = Path(receipt["identity_path"])
    original = identity.read_bytes()
    record = json.loads(original)["job_deadline"]
    first = restore_run_deadline(record, expected_run_id=receipt["run_id"],
                                 expected_policy_digest=policy.digest)
    resumed = pipeline.execute_run_many_durable(
        FakeEngine(), (), resume_run_id=receipt["run_id"], **kwargs,
    )
    second = restore_run_deadline(record, expected_run_id=receipt["run_id"],
                                  expected_policy_digest=policy.digest)
    assert resumed["run_id"] == receipt["run_id"]
    assert identity.read_bytes() == original
    assert second.deadline_monotonic_ns == first.deadline_monotonic_ns
    assert second.remaining_ns < first.remaining_ns
