import os
from pathlib import Path

import pytest

from factor_engine.runtime.resume_validation import _read_json, ResumeIdentityError


def test_control_record_swap_at_open_is_rejected(tmp_path, monkeypatch):
    record = tmp_path / "identity.json"
    other = tmp_path / "other.json"
    record.write_text('{"run":"original"}')
    other.write_text('{"run":"substituted"}')
    original_path_open, original_os_open = Path.open, os.open
    swapped = []

    def swap(path):
        if Path(path) == record and not swapped:
            record.unlink()
            record.symlink_to(other)
            swapped.append(True)

    def path_open(path, *args, **kwargs):
        swap(path)
        return original_path_open(path, *args, **kwargs)

    def os_open(path, *args, **kwargs):
        swap(path)
        return original_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(os, "open", os_open)
    with pytest.raises(ResumeIdentityError, match="control record"):
        _read_json(record)
    assert swapped == [True]


@pytest.mark.parametrize("kind", ["fifo", "directory", "oversized"])
def test_control_record_rejects_nonregular_or_oversized(tmp_path, kind):
    record = tmp_path / "identity.json"
    if kind == "fifo":
        os.mkfifo(record)
    elif kind == "directory":
        record.mkdir()
    else:
        record.write_bytes(b" " * 33)
    with pytest.raises(ResumeIdentityError, match="control record"):
        _read_json(record, maximum_bytes=32)


def test_control_record_reads_and_hashes_same_bytes(tmp_path):
    import hashlib
    record = tmp_path / "identity.json"
    payload = b'{"run":"original"}\n'
    record.write_bytes(payload)
    assert _read_json(record) == ({"run": "original"}, hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("failure", ["fstat", "read"])
def test_control_record_closes_fd_on_io_failure(tmp_path, monkeypatch, failure):
    import errno
    record = tmp_path / "identity.json"
    record.write_text('{}')
    opened = []
    original_open, original_fstat = os.open, os.fstat

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def broken_fstat(fd):
        raise OSError("injected fstat failure")

    class BrokenRead:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, *_):
            raise OSError("injected read failure")

    monkeypatch.setattr(os, "open", tracked_open)
    if failure == "fstat":
        monkeypatch.setattr(os, "fstat", broken_fstat)
    else:
        monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: BrokenRead())
    with pytest.raises(ResumeIdentityError, match="control record"):
        _read_json(record)
    assert len(opened) == 1
    with pytest.raises(OSError) as caught:
        original_fstat(opened[0])
    assert caught.value.errno == errno.EBADF


def test_control_record_path_replaced_after_open_keeps_fd_authority(tmp_path, monkeypatch):
    import hashlib
    record = tmp_path / "identity.json"
    payload = b'{"run":"original"}'
    record.write_bytes(payload)
    original_open = os.open
    swaps = []

    def open_then_swap(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if Path(path) == record:
            record.unlink()
            record.write_text('{"run":"substituted"}')
            swaps.append(True)
        return fd

    monkeypatch.setattr(os, "open", open_then_swap)
    assert _read_json(record) == ({"run": "original"}, hashlib.sha256(payload).hexdigest())
    assert swaps == [True]
