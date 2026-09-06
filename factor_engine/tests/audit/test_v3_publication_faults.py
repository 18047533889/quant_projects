from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from factor_engine.runtime import atomic_commit as ac
from factor_engine.runtime.materialize_batch import WriteItemReceipt, WriteReceipt, WriteState
from factor_engine.runtime.streaming_result_sink import StreamingResultSink


def _watermarks() -> ac.WatermarkSet:
    return ac.WatermarkSet("2026-09-07", "2026-09-07", "2026-09-07")


def _committed(root, payload=b"old") -> str:
    tx = ac.IncrementalCommitTransaction(root)
    tx.stage(factor_parts={"factor.bin": payload}, watermarks=_watermarks())
    return tx.commit().generation


@pytest.mark.parametrize("fail_destination", ["factor.bin", "state.bin"])
def test_first_or_middle_part_failure_keeps_old_current_and_hides_partial_generation(
    tmp_path, monkeypatch, fail_destination
):
    old_generation = _committed(tmp_path)
    tx = ac.IncrementalCommitTransaction(tmp_path)
    tx.stage(
        factor_parts={"factor.bin": b"new-factor"},
        state_parts={"state.bin": b"new-state"},
        watermarks=_watermarks(),
    )
    new_generation = tx.generation
    original_replace = ac.os.replace

    def fail_selected(source, destination):
        if str(destination).endswith(fail_destination):
            raise OSError(f"fault at {fail_destination}")
        return original_replace(source, destination)

    monkeypatch.setattr(ac.os, "replace", fail_selected)
    with pytest.raises(OSError, match="fault"):
        tx.commit()
    assert ac.current_generation(tmp_path) == old_generation
    assert ac.load_generation(tmp_path, new_generation) is None
    assert ac.list_generations(tmp_path) == [old_generation]
    quarantined = list((tmp_path / ".quarantine").glob(f"incomplete-{new_generation}-*"))
    assert len(quarantined) == 1


def _receipt(states: dict[str, WriteState]) -> WriteReceipt:
    inventory = [{"path": "p", "rows": 1, "bytes": 1, "sha256": "0" * 64}]
    digest = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return WriteReceipt(
        generation_id="g", expected_items=tuple(states),
        items={name: WriteItemReceipt(
            name, state=state, rows=1, run_id="r",
            inventory_digest=digest, inventory=inventory,
        ) for name, state in states.items()},
        manifest_digest="m", idempotency_key="g",
    )


def test_partial_receipt_never_becomes_complete_or_replays_committed_item():
    calls = 0

    def writer(_items):
        nonlocal calls
        calls += 1
        return _receipt({"a": WriteState.COMMITTED, "b": WriteState.IN_DOUBT})

    sink = StreamingResultSink(
        writer=writer, queue_bytes=1024, batch_size=2, require_write_receipt=True,
    )
    assert sink.submit("a", np.array([1.0]))
    assert sink.submit("b", np.array([2.0]))
    sink.start()
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    summary = sink.summary()
    assert summary["committed"] == 1
    assert summary["in_doubt"] == 1
    assert calls == 1


def test_unproven_writer_return_is_not_reported_as_success():
    sink = StreamingResultSink(
        writer=lambda _items: True,
        queue_bytes=1024,
        require_write_receipt=True,
    )
    assert sink.submit("a", np.array([1.0]))
    sink.start()
    with pytest.raises(RuntimeError, match="fatal"):
        sink.finish()
    assert sink.summary()["committed"] == 0


def test_exception_after_current_flip_is_verified_as_committed_not_replayed(
    tmp_path, monkeypatch
):
    tx = ac.IncrementalCommitTransaction(tmp_path)
    tx.stage(factor_parts={"factor.bin": b"value"}, watermarks=_watermarks())
    generation = tx.generation
    original_replace = ac.os.replace
    injected = False

    def replace_then_report_lost_ack(source, destination):
        nonlocal injected
        result = original_replace(source, destination)
        if str(destination).endswith("CURRENT") and not injected:
            injected = True
            raise OSError("lost acknowledgement after CURRENT flip")
        return result

    monkeypatch.setattr(ac.os, "replace", replace_then_report_lost_ack)
    committed = tx.commit()
    assert committed.generation == generation
    assert ac.current_generation(tmp_path) == generation
    assert ac.load_generation(tmp_path, generation) == committed
    assert tx.commit() == committed


def test_generation_discovery_rejects_stale_mismatched_manifest(tmp_path):
    generation_dir = tmp_path / "generation=directory-generation"
    generation_dir.mkdir()
    manifest = ac.IncrementalExecutionGeneration(
        generation="stale-manifest-generation", source_snapshot_id="s",
        execution_id="e", factor_parts=(), state_parts=(),
    )
    (generation_dir / "manifest.json").write_text(json.dumps(manifest.to_manifest()))
    assert ac.list_generations(tmp_path) == []


def test_corrupt_part_after_current_flip_is_in_doubt_not_optimistic_success(
    tmp_path, monkeypatch
):
    tx = ac.IncrementalCommitTransaction(tmp_path)
    tx.stage(factor_parts={"factor.bin": b"complete-value"}, watermarks=_watermarks())
    generation = tx.generation
    original_replace = ac.os.replace

    def flip_corrupt_then_raise(source, destination):
        result = original_replace(source, destination)
        if str(destination).endswith("CURRENT"):
            (tmp_path / f"generation={generation}" / "factor.bin").write_bytes(b"truncated")
            raise OSError("lost acknowledgement with corrupt bytes")
        return result

    monkeypatch.setattr(ac.os, "replace", flip_corrupt_then_raise)
    with pytest.raises(ac.IncrementalCommitInDoubtError, match="completeness is unproven"):
        tx.commit()
    assert ac.current_generation(tmp_path) == generation
    assert (tmp_path / f"generation={generation}" / "factor.bin").read_bytes() == b"truncated"
    assert ac.list_generations(tmp_path) == []


def test_generation_discovery_requires_all_declared_parts_and_valid_hashes(tmp_path):
    generation = _committed(tmp_path)
    gen_dir = tmp_path / f"generation={generation}"
    manifest_path = gen_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    (gen_dir / "factor.bin").unlink()
    assert ac.list_generations(tmp_path) == []
    (gen_dir / "factor.bin").write_bytes(b"wrong-bytes")
    assert ac.list_generations(tmp_path) == []
    manifest["factor_parts"] = ["../escape"]
    manifest["part_sha256"] = {"../escape": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    assert ac.list_generations(tmp_path) == []


def test_unreadable_current_after_failure_preserves_transaction_bytes_as_in_doubt(
    tmp_path, monkeypatch
):
    _committed(tmp_path)
    tx = ac.IncrementalCommitTransaction(tmp_path)
    tx.stage(
        factor_parts={"factor.bin": b"new-factor"},
        state_parts={"state.bin": b"new-state"},
        watermarks=_watermarks(),
    )
    generation = tx.generation
    original_replace = ac.os.replace

    def fail_midway(source, destination):
        if str(destination).endswith("state.bin"):
            raise OSError("midway")
        return original_replace(source, destination)

    monkeypatch.setattr(ac.os, "replace", fail_midway)
    monkeypatch.setattr(ac, "current_generation", lambda _root: (_ for _ in ()).throw(OSError("pointer unreadable")))
    with pytest.raises(ac.IncrementalCommitInDoubtError, match="cannot read CURRENT"):
        tx.commit()
    assert (tmp_path / f"generation={generation}" / "factor.bin").is_file()
    assert tx._staging is not None and (tx._staging / "state.bin.state.part.tmp").is_file()


def test_generation_verification_streams_hash_without_read_bytes(tmp_path, monkeypatch):
    generation = _committed(tmp_path, payload=b"x" * 1024)

    def forbidden_read_bytes(_self):
        raise AssertionError("verification must not allocate the whole part")

    monkeypatch.setattr(ac.Path, "read_bytes", forbidden_read_bytes)
    assert ac.list_generations(tmp_path) == [generation]


@pytest.mark.parametrize("manifest_payload", [[], "text", 7, None])
def test_generation_discovery_rejects_non_mapping_manifest(tmp_path, manifest_payload):
    generation_dir = tmp_path / "generation=g"
    generation_dir.mkdir()
    (generation_dir / "manifest.json").write_text(json.dumps(manifest_payload))
    assert ac.list_generations(tmp_path) == []


def test_generation_discovery_rejects_symlink_part(tmp_path):
    generation = _committed(tmp_path)
    gen_dir = tmp_path / f"generation={generation}"
    part = gen_dir / "factor.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(part.read_bytes())
    part.unlink()
    part.symlink_to(outside)
    assert ac.list_generations(tmp_path) == []
