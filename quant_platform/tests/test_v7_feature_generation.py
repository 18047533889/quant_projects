import json
import threading
import time
from dataclasses import replace
import pytest
from quant_platform.app.contracts import SecurityClassification
from quant_platform.app.orchestrator import Pipeline, PipelineConfig
from quant_platform.app.db.sqlite_backend import SqliteDb
from quant_platform.app.db.durable_store import SqliteRunStateStore
from quant_platform.app.storage.generation_coordinator import DurableGenerationCoordinator
from quant_platform.tests.test_durable_store import _raw_candidate, _library_snapshot, _evaluate, _ApproveAuthority
from quant_platform.tests.test_generation_coordinator import Publisher, ref

def pipeline(db, publisher, store=None, **kwargs):
    return Pipeline(evaluate_candidate=_evaluate, admission_authority=_ApproveAuthority(),
        run_storage=store, generation_coordinator=DurableGenerationCoordinator(db,publisher), **kwargs)

def test_restart_merge_retains_prior_members(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); publisher=Publisher()
    store=SqliteRunStateStore(str(tmp_path/'run.db'))
    p=pipeline(db,publisher,store)
    p.run([_raw_candidate(seed=s, factor_name=s, formula=f'sma(close, {i+1})')
           for i,s in enumerate(['a','b'])],library_snapshot=_library_snapshot())
    previous=p.feature_snapshots()[-1]
    restarted=pipeline(db,publisher,store)
    restarted.run([_raw_candidate(seed='c',factor_name='c',formula='sma(close, 3)')],
                  library_snapshot=_library_snapshot())
    latest=restarted.feature_snapshots()[-1]
    assert len(latest.ordered_members)==3
    assert latest.version != previous.version
    active=restarted.generation_coordinator.resolve_active('feature-set:'+p.config.feature_set_id)
    assert active and active['status']=='COMPLETE'
    payload=json.loads(bytes.fromhex(active['payload_hex']))
    assert len(payload['version']['ordered_members'])==3

def test_stale_parent_compare_and_swap_rejects_before_stage(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); c=DurableGenerationCoordinator(db,Publisher())
    first=c.stage(ref(b'first',artifact_id='fs'),b'first',expected_parent_generation=None)
    with pytest.raises(ValueError,match='parent|stale|CAS'):
        c.stage(ref(b'second',artifact_id='fs'),b'second',expected_parent_generation=None)
    assert len(db.query('SELECT * FROM artifact_generations'))==1
    assert c.stage(ref(b'first',artifact_id='fs'),b'first',expected_parent_generation=None)==first

def test_crash_after_feature_publication_replays_without_new_generation(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); publisher=Publisher()
    store=SqliteRunStateStore(str(tmp_path/'run.db'))
    def fail(stage, artifact):
        if stage=='feature_set_complete_before_consumption':
            raise RuntimeError('injected crash')
    p=pipeline(db,publisher,store,failure_injector=fail)
    candidate=_raw_candidate(seed='a')
    with pytest.raises(RuntimeError,match='injected crash'):
        p.run([candidate],library_snapshot=_library_snapshot())
    assert not store.consumed_hashes()
    restarted=pipeline(db,publisher,store)
    result=restarted.run([candidate],library_snapshot=_library_snapshot())
    assert result.num_approved==1
    assert restarted.feature_snapshots()[-1].version=='v1'
    rows=db.query("SELECT * FROM artifact_generations WHERE artifact_id LIKE 'feature-set:%'")
    assert len(rows)==1
    assert store.consumed_hashes()

def test_replace_explicitly_replaces_members(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); publisher=Publisher()
    p=pipeline(db,publisher)
    p.run([_raw_candidate(seed='a',factor_name='a')],library_snapshot=_library_snapshot())
    replacement=pipeline(db,publisher,config=PipelineConfig(feature_set_update_mode='REPLACE'))
    replacement.run([_raw_candidate(seed='b',factor_name='b')],library_snapshot=_library_snapshot())
    members=replacement.feature_snapshots()[-1].ordered_members
    assert len(members)==1
    assert members[0].factor_definition_ref != p.feature_snapshots()[-1].ordered_members[0].factor_definition_ref

def test_merge_same_definition_replaces_changed_provenance(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); publisher=Publisher()
    p=pipeline(db,publisher)
    first=_raw_candidate(seed='old-bytes',factor_name='same',formula='sma(close, 5)')
    second=_raw_candidate(seed='new-bytes',factor_name='same',formula='sma(close, 5)')
    p.run([first],library_snapshot=_library_snapshot())
    old=p.feature_snapshots()[-1].ordered_members[0]
    restarted=p=pipeline(db,publisher)
    restarted.run([second],library_snapshot=_library_snapshot())
    latest=restarted.feature_snapshots()[-1]
    assert len(latest.ordered_members)==1
    assert latest.ordered_members[0].factor_definition_ref == old.factor_definition_ref
    assert latest.ordered_members[0].raw_value_ref != old.raw_value_ref
    assert latest.ordered_members[0].source_artifact_id != old.source_artifact_id

def test_idempotent_stage_rejects_changed_parent_and_serializes_enum(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); c=DurableGenerationCoordinator(db,Publisher())
    artifact=replace(ref(b'payload',artifact_id='enum'),
        security_classification=SecurityClassification.INTERNAL_RESEARCH)
    generation=c.stage(artifact,b'payload',expected_parent_generation=None)
    with pytest.raises(ValueError,match='changed CAS parent'):
        c.stage(artifact,b'payload',expected_parent_generation='gen:wrong')
    c.outbox.publish_pending()
    assert c.resolve_active('enum')['generation_id']==generation

def test_reservation_fence_blocks_expired_lease_takeover(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import quant_platform.app.db.durable_store as durable_store
    clock = [100.0]
    monkeypatch.setattr(durable_store, 'time', SimpleNamespace(time=lambda: clock[0]))
    path=str(tmp_path/'run.db'); owner=SqliteRunStateStore(path)
    owner.reserve_candidates({'h':'{}'},'old',lease_seconds=1)
    entered=threading.Event(); acquired=threading.Event()
    attempt=threading.Event(); attempting=threading.Event()
    def takeover():
        contender=SqliteRunStateStore(path)
        entered.set()
        assert attempt.wait(5)
        attempting.set()
        contender.reserve_candidates({'h':'{}'},'new',lease_seconds=1)
        acquired.set(); contender.close()
    worker=threading.Thread(target=takeover); worker.start()
    assert entered.wait(5)
    try:
        with owner.fence_reservations(('h',),'old'):
            # Expire only after entering the fence: machine scheduling cannot
            # turn the setup into an already-expired-owner rejection test.
            clock[0] = 102.0
            attempt.set()
            assert attempting.wait(5)
            assert not acquired.wait(.05)
    finally:
        attempt.set()
        worker.join(timeout=5)
        owner.close()
    assert not worker.is_alive()
    assert acquired.is_set()

def test_live_pipeline_reads_consumption_written_after_construction(tmp_path):
    db=SqliteDb(str(tmp_path/'metadata.db')); publisher=Publisher()
    store=SqliteRunStateStore(str(tmp_path/'run.db'))
    stale=pipeline(db,publisher,store)
    writer=pipeline(db,publisher,store)
    candidate=_raw_candidate(seed='cross-instance',factor_name='cross')
    writer.run([candidate],library_snapshot=_library_snapshot())
    report=stale.run([candidate,_raw_candidate(seed='fresh',factor_name='fresh')],
                     library_snapshot=_library_snapshot())
    assert report.num_duplicates==1
    assert report.num_approved==1
