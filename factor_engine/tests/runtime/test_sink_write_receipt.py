import numpy as np
import pytest
import hashlib
import json

from factor_engine.runtime.materialize_batch import WriteReceipt, WriteItemReceipt, WriteState
from factor_engine.runtime.streaming_result_sink import StreamingResultSink


def receipt(states):
    payload = b"fixture"
    inventory = [{"path": "fixture.parquet", "rows": 1, "bytes": len(payload),
                  "sha256": hashlib.sha256(payload).hexdigest()}]
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return WriteReceipt("generation-1", tuple(states),
                        {name: WriteItemReceipt(name, state, rows=1, run_id="test-run",
                            inventory_digest=digest, inventory=inventory)
                         for name, state in states.items()},
                        manifest_digest="digest", idempotency_key="key")


def run_sink(writer, names=("a", "b"), **kwargs):
    sink = StreamingResultSink(writer=writer, queue_bytes=1024, batch_size=2,
                               require_write_receipt=True, **kwargs)
    for name in names:
        assert sink.submit(name, np.array([1.0]))
    sink.start()
    return sink


def test_complete_typed_receipt_commits():
    sink = run_sink(lambda _: receipt({"a": WriteState.COMMITTED, "b": WriteState.PUBLISHED}))
    sink.finish()
    assert sink.summary()["committed"] == 2
    assert sink.queue.current_bytes == 0


@pytest.mark.parametrize("state", [WriteState.FAILED, WriteState.IN_DOUBT, WriteState.STAGED, WriteState.CREATED])
def test_partial_receipt_preserves_commits_without_replay(state):
    calls = []
    def writer(_):
        calls.append(1)
        return receipt({"a": WriteState.COMMITTED, "b": state})
    sink = run_sink(writer)
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["committed"] == 1
    assert sink.summary()["retried"] == 0
    assert len(calls) == 1
    assert sink.queue.current_bytes == 0


@pytest.mark.parametrize("output", [None, True, False, {"materializations": {"a": {"error": "disk full"}}}])
def test_strict_writer_rejects_unproven_success(output):
    sink = run_sink(lambda _: output)
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["committed"] == 0


def test_exception_receipt_does_not_retry_transient_message():
    calls = []
    def writer(_):
        calls.append(1)
        exc = ConnectionError("remote retryable")
        exc.materialization = {"_write_receipt": receipt({"a": WriteState.COMMITTED, "b": WriteState.IN_DOUBT}).to_dict()}
        raise exc
    sink = run_sink(writer)
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["committed"] == 1
    assert sink.summary()["in_doubt"] == 1
    assert len(calls) == 1


def test_custom_receipt_ids():
    sink = run_sink(lambda _: {"receipt": receipt({"id-a": WriteState.COMMITTED, "id-b": WriteState.COMMITTED})},
                    receipt_item_key=lambda item: "id-" + item.name)
    sink.finish()
    assert sink.summary()["committed"] == 2


def test_mismatched_receipt_fails_closed():
    sink = run_sink(lambda _: receipt({"a": WriteState.COMMITTED, "c": WriteState.COMMITTED}))
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["committed"] == 0


def test_strict_unknown_commit_response_not_retried():
    calls = []
    def writer(_):
        calls.append(1)
        raise TimeoutError("network response lost")
    sink = run_sink(writer)
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["in_doubt"] == 2
    assert len(calls) == 1
