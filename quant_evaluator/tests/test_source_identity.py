import os
import time

import pytest

import quant_evaluator.runtime.source_identity as source_identity_module
from quant_evaluator.runtime.source_identity import (
    ProcessSourceIdentity,
    SourceIdentityError,
    SourceIdentityRaceError,
    UnsupportedSourceLayout,
)


def _write(path, content=b"x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _replace(path, content):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def test_stat_guard_cache_hit_skips_source_reads(tmp_path):
    _write(tmp_path / "a.py", b"alpha\n")
    _write(tmp_path / "sub" / "b.py", b"beta\n")
    identity = ProcessSourceIdentity(tmp_path)

    initial = identity.identify()
    cached = identity.identify()

    assert initial.strategy == "process_stat_guarded_changed_file_hash"
    assert initial.full_content_checked is True
    assert initial.stat_guard_fields == ("device", "inode", "mode", "size", "mtime_ns", "ctime_ns")
    assert initial.source_size_bytes == len(b"alpha\n") + len(b"beta\n")
    assert initial.record_cap == identity.max_records
    assert initial.byte_cap == identity.max_bytes
    assert initial.drift_path_cap == identity.max_drift_paths
    assert initial.retry_cap == identity.max_retries
    assert initial.directory_changes == ()
    assert initial.files_hashed == 2
    assert initial.bytes_hashed == len(b"alpha\n") + len(b"beta\n")
    assert cached.digest == initial.digest
    assert cached.stat_guard_verified is True
    assert cached.full_content_checked is False
    assert cached.files_hashed == cached.bytes_hashed == 0
    assert cached.changed_paths == ()
    assert cached.process_nonce == initial.process_nonce


def test_addition_and_deletion_change_identity(tmp_path):
    _write(tmp_path / "a.py", b"alpha\n")
    identity = ProcessSourceIdentity(tmp_path)
    original = identity.identify()

    _write(tmp_path / "nested" / "b.py", b"beta\n")
    added = identity.identify()
    assert added.drifted is True
    assert "nested/b.py" in added.changed_paths
    assert added.files_hashed == 1
    assert added.digest != original.digest

    (tmp_path / "a.py").unlink()
    deleted = identity.identify()
    assert deleted.drifted is True
    assert "a.py" in deleted.changed_paths
    assert deleted.files_hashed == 0
    assert deleted.digest != added.digest


def test_same_size_edit_is_seen_even_when_mtime_is_restored(tmp_path):
    path = tmp_path / "a.py"
    _write(path, b"old\n")
    identity = ProcessSourceIdentity(tmp_path)
    first = identity.identify()
    before = path.stat()

    time.sleep(0.01)
    _write(path, b"new\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns

    changed = identity.identify()
    assert changed.drifted is True
    assert changed.changed_paths == ("a.py",)
    assert changed.files_hashed == 1
    assert changed.digest != first.digest


def test_read_race_retries_once_then_commits_stable_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "a.py"
    _write(path, b"before\n")
    identity = ProcessSourceIdentity(tmp_path, max_retries=1)
    original_hash = identity._hash_file
    calls = []

    def mutate_after_read(relative, guard):
        result = original_hash(relative, guard)
        calls.append(relative)
        if len(calls) == 1:
            _replace(path, b"after!\n")
        return result

    monkeypatch.setattr(identity, "_hash_file", mutate_after_read)
    receipt = identity.identify()
    assert calls == ["a.py", "a.py"]
    assert receipt.attempts == 2
    assert receipt.files_hashed == 2
    assert receipt.digest == identity.identify().digest


def test_continuous_read_race_fails_after_bounded_retries(tmp_path, monkeypatch):
    path = tmp_path / "a.py"
    _write(path, b"zero\n")
    identity = ProcessSourceIdentity(tmp_path, max_retries=1)
    original_hash = identity._hash_file
    calls = []

    def mutate_every_read(relative, guard):
        result = original_hash(relative, guard)
        calls.append(relative)
        _replace(path, (b"one!\n" if len(calls) % 2 else b"two!\n"))
        return result

    monkeypatch.setattr(identity, "_hash_file", mutate_every_read)
    with pytest.raises(SourceIdentityRaceError, match="remained unstable"):
        identity.identify()
    assert calls == ["a.py", "a.py"]
    assert identity._snapshot is None


def test_strict_mode_rehashes_unchanged_files_every_call(tmp_path):
    _write(tmp_path / "a.py", b"alpha\n")
    identity = ProcessSourceIdentity(tmp_path, strict_full_content=True)
    first = identity.identify()
    second = identity.identify()
    assert first.strategy == "strict_full_content"
    assert second.full_content_checked is True
    assert second.files_hashed == 1
    assert second.bytes_hashed == len(b"alpha\n")
    assert second.digest == first.digest


def test_source_size_and_drift_caps_fail_closed(tmp_path):
    _write(tmp_path / "a.py", b"12345")
    with pytest.raises(SourceIdentityError, match="byte cap"):
        ProcessSourceIdentity(tmp_path, max_bytes=4).identify()

    identity = ProcessSourceIdentity(tmp_path, max_drift_paths=1)
    identity.identify()
    _write(tmp_path / "nested" / "b.py", b"2")
    _write(tmp_path / "nested" / "c.py", b"3")
    with pytest.raises(SourceIdentityError, match="drift cap"):
        identity.identify()



def test_scandir_stops_materializing_at_record_cap(tmp_path, monkeypatch):
    for index in range(20):
        _write(tmp_path / f"f{index}.py", b"x")
    identity = ProcessSourceIdentity(tmp_path, max_records=2)
    real_scandir = os.scandir
    consumed = []

    class CountedScanner:
        def __init__(self, path):
            self._scanner = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._scanner.close()

        def __iter__(self):
            for entry in self._scanner:
                consumed.append(entry.name)
                yield entry

    monkeypatch.setattr(source_identity_module.os, "scandir", CountedScanner)
    with pytest.raises(SourceIdentityError, match="entry cap"):
        identity._scan()
    assert len(consumed) == 3


def test_strict_mode_detects_content_change_under_same_stat_guard(tmp_path, monkeypatch):
    path = tmp_path / "a.py"
    _write(path, b"old!\n")
    identity = ProcessSourceIdentity(tmp_path, strict_full_content=True)
    first = identity.identify()
    old_file_guard = identity._snapshot["a.py"]
    original_signature = source_identity_module._stat_signature

    def same_signature_for_target(value):
        if value.st_ino == old_file_guard.target_stat[1]:
            return old_file_guard.link_stat
        return original_signature(value)

    monkeypatch.setattr(source_identity_module, "_stat_signature", same_signature_for_target)
    _write(path, b"new!\n")
    second = identity.identify()
    assert second.changed_paths == ("a.py",)
    assert second.drifted is True
    assert second.digest != first.digest
    assert second.full_content_checked is True


def test_non_python_entry_changes_are_reported_separately(tmp_path):
    _write(tmp_path / "a.py", b"alpha\n")
    identity = ProcessSourceIdentity(tmp_path)
    identity.identify()
    root_guard = identity._snapshot["."]
    _write(tmp_path / "README.txt", b"documentation\n")
    os.utime(tmp_path, ns=(root_guard.link_stat[4], root_guard.link_stat[4] + 1_000_000))
    receipt = identity.identify()
    assert receipt.directory_changes == (".",)
    assert receipt.changed_paths == ()
    assert receipt.drifted is False
    assert receipt.files_hashed == 0

def test_symlink_directories_and_external_source_links_fail_closed(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    _write(outside / "outside.py", b"safe\n")

    directory_tree = tmp_path / "directory_case"
    directory_tree.mkdir()
    (directory_tree / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsupportedSourceLayout, match="symlink directory"):
        ProcessSourceIdentity(directory_tree).identify()

    file_tree = tmp_path / "file_case"
    file_tree.mkdir()
    (file_tree / "external.py").symlink_to(outside / "outside.py")
    with pytest.raises(UnsupportedSourceLayout, match="escapes root"):
        ProcessSourceIdentity(file_tree).identify()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_does_not_inherit_a_lock_held_by_another_thread(tmp_path):
    import multiprocessing
    import threading

    _write(tmp_path / "a.py")
    identity = ProcessSourceIdentity(tmp_path)
    parent = identity.identify()
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with identity._lock:
            acquired.set()
            release.wait(5)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert acquired.wait(2)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def identify_in_child():
        receipt = identity.identify()
        sender.send((receipt.process_nonce, receipt.process_id, receipt.digest))
        sender.close()

    child = context.Process(target=identify_in_child)
    try:
        child.start()
        assert receiver.poll(3), "child blocked on an inherited source-identity lock"
        nonce, pid, digest = receiver.recv()
        assert nonce != parent.process_nonce and pid != parent.process_id
        assert digest == parent.digest
        child.join(1)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(1)
        release.set()
        thread.join(1)
        receiver.close()
        sender.close()
