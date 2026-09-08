import json
import os
from pathlib import Path

import pytest

from factor_engine.runtime import bounded_pipeline as pipeline
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor


def _missing_compile(engine, factors):
    return {}


class OverlapEngine(FakeEngine):
    def run_many_parallel(self, factors, **kwargs):
        output = super().run_many_parallel(factors, **kwargs)
        output['physical_preflight_errors'] = {
            factors[0].name: {'code': 'UNKNOWN_FIELD', 'message': 'overlap'}
        }
        return output


@pytest.mark.parametrize('fault', ['missing_compile', 'overlapping_result'])
def test_protocol_corruption_aborts_without_artifacts(tmp_path, monkeypatch, fault):
    if fault == 'missing_compile':
        monkeypatch.setattr(pipeline, '_compile_wave', _missing_compile)
    engine = OverlapEngine() if fault == 'overlapping_result' else FakeEngine()
    with pytest.raises(RuntimeError, match='protocol') as caught:
        pipeline.execute_run_many_durable(
            engine, [FakeFactor('a')], policy=resolve_default_policy(), artifact_root=tmp_path)
    receipt = json.loads(Path(caught.value.receipt_path).read_text())
    assert receipt['status'] == 'ABORTED'
    assert sum(receipt['counts'].values()) == 1
    assert not list(tmp_path.rglob('*.parquet'))


@pytest.mark.parametrize('mapping', [{}, {'a': None, 'extra': None}, {'a': False}, {'a': ''}, []])
def test_compile_protocol_rejects_missing_extra_and_untyped(mapping):
    with pytest.raises(pipeline.WorkerProtocolError):
        pipeline._validate_compile_results(mapping, {'a'})


@pytest.mark.parametrize('envelope', [
    {'result_blobs': {}, 'transport_errors': {}, 'factor_errors': {}},
    {'result_blobs': {'a': None}, 'transport_errors': {}, 'factor_errors': {}},
    {'result_blobs': {}, 'transport_errors': {'a': 'bad'}, 'factor_errors': {}},
    {'result_blobs': {'a': b'x'}, 'transport_errors': {}, 'factor_errors': {}, 'extra': {}},
])
def test_result_protocol_rejects_malformed_records(envelope):
    with pytest.raises(pipeline.WorkerProtocolError):
        pipeline._validate_result_envelope(envelope, {'a'})


def test_control_receipt_sync_order_and_atomic_visibility(tmp_path, monkeypatch):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    target = run_dir / 'receipt.json'
    target.write_text('{"old": true}')
    events = []
    real_sync, real_replace = os.fsync, os.replace

    def synced(fd):
        events.append(('sync', Path(os.readlink(f'/proc/self/fd/{fd}')).name))
        real_sync(fd)

    def replaced(source, destination):
        assert json.loads(target.read_text()) == {'old': True}
        assert json.loads(Path(source).read_text()) == {'new': True}
        events.append(('replace', Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(os, 'replace', replaced)
    pipeline._write_control_receipt(target, {'new': True})
    assert json.loads(target.read_text()) == {'new': True}
    assert [e[0] for e in events] == ['sync', 'replace', 'sync', 'sync']
    assert events[2:] == [('sync', 'run'), ('sync', tmp_path.name)]


def test_failed_control_file_sync_does_not_replace_previous_receipt(tmp_path, monkeypatch):
    target = tmp_path / 'receipt.json'
    target.write_text('{"old": true}')

    def fail(fd):
        raise OSError('injected file sync failure')

    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError, match='injected'):
        pipeline._write_control_receipt(target, {'new': True})
    assert json.loads(target.read_text()) == {'old': True}
