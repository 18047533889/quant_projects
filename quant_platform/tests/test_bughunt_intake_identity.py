"""Counterexamples for operational replay identity and durable intake."""
from datetime import datetime, timezone
import pytest
from quant_platform.app.candidate.ingest import batch_fingerprint, normalize_candidate, reconcile_candidates
from quant_platform.app.db.durable_store import SqliteRunStateStore
from quant_platform.app.orchestrator import Pipeline
from quant_platform.tests.test_v2_ingestion_boundaries import _raw


def test_one_shot_reconciliation_preserves_batch_fingerprint():
    items = [normalize_candidate(_raw())]
    assert reconcile_candidates(iter(items)).batch_fingerprint == batch_fingerprint(items)


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("mutation", [{"semantic_id": "e"*64}, {"content_hash": "d"*64}])
def test_same_batch_conflicting_carried_identities_never_evaluate(tmp_path, durable, mutation):
    store = SqliteRunStateStore(str(tmp_path/"run.db")) if durable else None
    calls = []
    p = Pipeline(run_storage=store, evaluate_candidate=lambda *args: calls.append(args))
    report = p.run([_raw(), _raw(**mutation)], library_snapshot={})
    assert report.num_conflicts == 2
    assert not calls
    if store:
        store.close()


@pytest.mark.parametrize("durable", [False, True])
def test_same_hash_changed_semantic_is_conflict_not_replay(tmp_path, durable):
    store = SqliteRunStateStore(str(tmp_path/"run.db")) if durable else None
    p = Pipeline(run_storage=store)
    p.run([_raw()], library_snapshot={})
    if store:
        p = Pipeline(run_storage=store)
    report = p.run([_raw(semantic_id="e"*64)], library_snapshot={})
    assert report.num_replayed == 0
    assert report.num_conflicts == 1
    if store:
        store.close()


def test_durable_intake_accepts_supported_datetime_without_raw_json_failure(tmp_path):
    store = SqliteRunStateStore(str(tmp_path/"run.db"))
    result = Pipeline(run_storage=store).run(
        [_raw(published_at=datetime(2026,9,1,tzinfo=timezone.utc))], library_snapshot={})
    assert result.num_consumed == 1
    store.close()


def test_reconcile_exact_duplicate_keeps_first_record_new():
    candidate = normalize_candidate(_raw())
    result = reconcile_candidates([candidate, candidate])
    assert result.new == 1
    assert result.duplicates == 1


@pytest.mark.parametrize("changes", [
    {"candidate_id": 123}, {"factor_value_ref": {"bad": "ref"}},
    {"parent_factor_ids": "not-a-sequence-of-refs"},
    {"required_fields": [{"bad": "field"}]},
])
@pytest.mark.parametrize("durable", [False, True])
def test_malformed_carrier_fields_cannot_crash_healthy_neighbours(tmp_path, changes, durable):
    store = SqliteRunStateStore(str(tmp_path/"run.db")) if durable else None
    report = Pipeline(run_storage=store).run([_raw(**changes), _raw()], library_snapshot={})
    assert report.num_normalization_failed == 1
    assert report.num_consumed == 1
    if store:
        store.close()


def test_admission_carriers_cannot_mutate_through_nested_external_aliases():
    from quant_platform.app.contracts.admission import AdmissionRequest, AdmissionVerdict
    payload = {"refs": ["original"]}
    request = AdmissionRequest("candidate", "a"*64, context=payload)
    verdict = AdmissionVerdict("APPROVED", detail=payload)
    payload["refs"].append("tampered")
    assert tuple(request.context["refs"]) == ("original",)
    assert tuple(verdict.detail["refs"]) == ("original",)
    with pytest.raises(TypeError):
        verdict.detail["refs"] = ()
    with pytest.raises(TypeError):
        request.context["refs"] = ()
