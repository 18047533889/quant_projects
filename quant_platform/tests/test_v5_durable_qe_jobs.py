from quant_platform.app.contracts import ErrorClass, JobResult, JobSpec, JobStatus
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.worker.durable_jobs import DurableJobStore
from quant_platform.app.worker.jobs import JobError, JobRunner


def test_retry_restart_and_success_are_exact_once(tmp_path):
    db = SqliteDb(str(tmp_path / "jobs.db"))
    spec = JobSpec("qe_metric_instance_shard", "same", max_retries=1)
    first = JobRunner("w1", DurableJobStore(db)).execute(
        spec, lambda _: (_ for _ in ()).throw(JobError(ErrorClass.RETRYABLE_STORAGE))
    )
    assert first.status is JobStatus.FAILED_RETRYABLE
    calls = {"n": 0}
    def success(_):
        calls["n"] += 1
        return JobResult(("result:a",))
    second = JobRunner("w2", DurableJobStore(db)).execute(spec, success)
    third = JobRunner("w3", DurableJobStore(db)).execute(spec, success)
    assert second.status is third.status is JobStatus.SUCCEEDED
    assert second.attempt_count == 2 and calls["n"] == 1
    assert second.result.output_artifact_refs == ("result:a",)


def test_two_workers_compete_and_only_claimant_executes(tmp_path):
    import threading
    path = str(tmp_path / "race.db"); SqliteDb(path).close()
    entered = threading.Event(); release = threading.Event(); calls = []
    spec = JobSpec("qe_metric_instance_shard", "race", timeout_seconds=30)
    def handler(_):
        calls.append(1); entered.set(); release.wait(2); return JobResult(("one",))
    results = []
    def run():
        db = SqliteDb(path); results.append(JobRunner("w", DurableJobStore(db)).execute(spec, handler)); db.close()
    first = threading.Thread(target=run); first.start(); assert entered.wait(2)
    second = threading.Thread(target=run); second.start(); second.join(); release.set(); first.join()
    assert len(calls) == 1
    db = SqliteDb(path); final = DurableJobStore(db).load("race"); db.close()
    assert final.status is JobStatus.SUCCEEDED and final.result.output_artifact_refs == ("one",)


def test_metadata_submission_is_bounded_at_1k_10k_100k():
    from quant_evaluator.jobs.refs import shard_factor_refs
    for count in (1_000, 10_000, 100_000):
        shards = shard_factor_refs((f"ref:{i}" for i in range(count)))
        assert sum(map(len, shards)) == count
        assert max(map(len, shards)) <= 256
        assert len(shards) == (count + 255) // 256


def test_real_public_qe_multi_instance_shard_writes_bounded_results(tmp_path):
    import numpy as np
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.contracts.metric_instance import EvaluationScenario, MetricInstance
    from quant_evaluator.jobs.handler import QEJobHandler
    from quant_evaluator.jobs.refs import QEJobShard, write_manifest

    rng = np.random.default_rng(7); t, n = 36, 8
    batch = FactorBatch(("f",), AxisRef("t", "int", t), AxisRef("a", "str", n),
                        np.ascontiguousarray(rng.normal(size=(t, n, 1))))
    labels = LabelBundle(
        target_id="return", values=np.ascontiguousarray(rng.normal(size=(t, n))),
        horizon=1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)), label_end_time=tuple(range(1, t+1)),
    )
    instances = (MetricInstance("coverage", scenario_id="s", horizon=1),
                 MetricInstance("rank_ic", scenario_id="s", horizon=1))
    scenario = EvaluationScenario(labels)
    shard = QEJobShard(("factor:f",), "label:h1", tuple(x.to_dict() for x in instances),
                       (("s", "scenario:s"),), str(tmp_path / "sink"), 0)
    manifest = write_manifest(tmp_path / "manifests", shard)
    handler = QEJobHandler(factor_batch_resolver=lambda _: batch,
                           label_resolver=lambda _: labels,
                           scenario_resolver=lambda _: scenario)
    spec = JobSpec("qe_metric_instance_shard", "qe-real", max_retries=1,
                   input_artifact_refs=(manifest,))
    record = JobRunner().execute(spec, handler)
    assert record.status is JobStatus.SUCCEEDED
    assert len(record.result.output_artifact_refs) == 2
    assert all(__import__("pathlib").Path(ref).stat().st_size > 0
               for ref in record.result.output_artifact_refs)
