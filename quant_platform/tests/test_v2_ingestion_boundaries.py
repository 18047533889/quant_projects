"""V2 PL-01/PL-05: bad records cannot abort their healthy batch neighbours."""
import ast
from pathlib import Path

import pytest

from quant_platform.app.candidate.ingest import CandidateNormalizationError, normalize_candidate
from quant_platform.app.orchestrator import Pipeline


def _raw(**changes):
    result = dict(content_hash="a" * 64, semantic_id="b" * 64,
                  factor_definition_ref="c" * 64, candidate_id="candidate-1",
                  generator_type="test", generator_version="1", submitted_by="test",
                  market="ashare", frequency="1d", factor_spec_uri="recipe://test",
                  published_at="2026-09-01T00:00:00Z")
    result.update(changes)
    return result


@pytest.mark.parametrize("domain", [{"window": ["bad", "30"]}, {"window": [10**1000, 10**1001]}])
def test_bad_numeric_domain_does_not_abort_good_neighbour(domain):
    calls = []
    pipeline = Pipeline(evaluate_candidate=lambda candidate, context: calls.append(candidate))
    report = pipeline.run([_raw(parameter_domain=domain), _raw()], library_snapshot={})
    assert report.num_normalization_failed == 1
    assert report.num_consumed == 1
    assert len(calls) == 1


@pytest.mark.parametrize("durable", [False, True])
def test_changed_evaluation_context_reassesses_same_spec_and_restart_replays(tmp_path, durable):
    from quant_platform.app.db.durable_store import SqliteRunStateStore
    store = SqliteRunStateStore(str(tmp_path / "run.db")) if durable else None
    calls = []
    def evaluate(candidate, context):
        calls.append(context["evaluation_context_ref"])
        return None
    pipeline = Pipeline(run_storage=store, evaluate_candidate=evaluate)
    first = pipeline.run([_raw()], library_snapshot={}, evaluation_context_ref="data:1-policy:1")
    assert first.num_consumed == 1
    if durable:
        pipeline = Pipeline(run_storage=store, evaluate_candidate=evaluate)
    replay = pipeline.run([_raw()], library_snapshot={}, evaluation_context_ref="data:1-policy:1")
    assert replay.num_replayed == 1
    for context in ("data:2-policy:1", "data:2-policy:2"):
        assert pipeline.run([_raw()], library_snapshot={}, evaluation_context_ref=context).num_consumed == 1
    assert calls == ["data:1-policy:1", "data:2-policy:1", "data:2-policy:2"]
    if store is not None:
        store.close()


@pytest.mark.parametrize("raw", [None, [], "broken"])
def test_non_mapping_input_has_typed_normalization_failure(raw):
    with pytest.raises(CandidateNormalizationError, match="bad_record_type"):
        normalize_candidate(raw)


@pytest.mark.parametrize("raw", [None, [], "broken"])
def test_pipeline_records_non_mapping_failure_without_fabricating_hash(raw):
    report = Pipeline().run([raw], library_snapshot={})
    assert report.num_normalization_failed == 1
    assert all(item.content_hash is None for item in report.states)


def test_pipeline_has_no_shadowed_class_methods():
    import quant_platform.app.orchestrator as module
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names = [child.name for child in node.body
                     if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
            assert len(names) == len(set(names)), node.name


def test_two_contexts_remain_consumed_in_mixed_batch_after_restart(tmp_path):
    from quant_platform.app.db.durable_store import SqliteRunStateStore
    store = SqliteRunStateStore(str(tmp_path / "run.db"))
    calls = []
    def evaluate(candidate, context):
        calls.append(candidate.candidate_id)
    pipeline = Pipeline(run_storage=store, evaluate_candidate=evaluate)
    for context in ("data:1", "data:2"):
        pipeline.run([_raw()], library_snapshot={}, evaluation_context_ref=context)
    restarted = Pipeline(run_storage=store, evaluate_candidate=evaluate)
    report = restarted.run([_raw(), _raw(candidate_id="candidate-2", content_hash="d"*64,
                                        semantic_id="e"*64)],
                           library_snapshot={}, evaluation_context_ref="data:2")
    assert report.num_consumed == 1
    assert report.num_duplicates == 1
    assert calls == ["candidate-1", "candidate-1", "candidate-2"]
    store.close()


@pytest.mark.parametrize("context", [None, "data:1"])
def test_restart_retains_full_semantic_conflict_identity(tmp_path, context):
    from quant_platform.app.db.durable_store import SqliteRunStateStore
    store = SqliteRunStateStore(str(tmp_path / "run.db"))
    Pipeline(run_storage=store).run([_raw()], library_snapshot={}, evaluation_context_ref=context)
    report = Pipeline(run_storage=store).run([_raw(content_hash="d"*64)],
                                            library_snapshot={}, evaluation_context_ref=context)
    assert report.num_conflicts == 1
    assert report.num_consumed == 0
    store.close()


def test_durable_job_retry_survives_process_counter_reset(tmp_path):
    from quant_platform.app.db.durable_store import SqliteRunStateStore
    from quant_platform.app.db.sqlite_backend import SqliteDb
    from quant_platform.app.worker.durable_jobs import DurableJobStore
    from quant_platform.app.worker.jobs import JobRunner
    run_store = SqliteRunStateStore(str(tmp_path / "run.db"))
    job_db = SqliteDb(str(tmp_path / "jobs.db"))
    calls = []
    def evaluate(candidate, context):
        calls.append(candidate.candidate_id)
        if len(calls) == 1:
            raise ConnectionError("temporary outage")
        return None
    def pipeline():
        return Pipeline(run_storage=run_store, evaluate_candidate=evaluate,
                        runner=JobRunner("test-worker", DurableJobStore(job_db)))
    assert pipeline().run([_raw()], library_snapshot={}).num_evaluation_failed == 1
    recovered = pipeline().run([_raw()], library_snapshot={})
    assert recovered.num_evaluation_failed == 0
    assert recovered.num_rejected == 1
    assert len(calls) == 2
    job_db.close()
    run_store.close()


def test_immature_label_waits_instead_of_permanent_consumption(tmp_path):
    from quant_platform.app.orchestrator import build_with_evidence
    from quant_platform.app.db.durable_store import SqliteRunStateStore
    store = SqliteRunStateStore(str(tmp_path / "run.db"))
    mature = [False]
    def evaluate(candidate, context):
        return build_with_evidence(dict(candidate_ref=candidate.candidate_id,
            label_maturity=mature[0], evidence_status="computed", return_basis="vwap_to_vwap"))
    failed = Pipeline(run_storage=store, evaluate_candidate=evaluate).run([_raw()], library_snapshot={})
    assert failed.num_evaluation_failed == 1
    assert not store.consumed_hashes()
    assert not store.consumed_fingerprints()
    mature[0] = True
    report = Pipeline(run_storage=store, evaluate_candidate=evaluate).run([_raw()], library_snapshot={})
    assert report.num_evaluation_failed == 0
    assert report.num_consumed == 1
    store.close()


def test_maturity_string_cannot_masquerade_as_true():
    from quant_platform.app.orchestrator import build_with_evidence
    with pytest.raises(ValueError, match="must be a bool"):
        build_with_evidence(dict(candidate_ref="candidate-1", label_maturity="false",
                                evidence_status="computed", return_basis="vwap_to_vwap"))
