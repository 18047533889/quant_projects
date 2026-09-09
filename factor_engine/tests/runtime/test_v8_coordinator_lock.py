"""Independent ownership oracles: real processes, stable lock inode."""
import multiprocessing
import os

import pytest

from factor_engine.runtime.bounded_pipeline import _RunCoordinatorLock


def _try_lock(directory, output, release):
    lock = _RunCoordinatorLock(directory)
    try:
        lock.acquire(allow_dead_owner=True)
    except RuntimeError:
        output.put("blocked")
        return
    output.put("acquired")
    release.wait(10)
    lock.release()


def test_release_preserves_inode_and_excludes_other_open_descriptors(tmp_path):
    first = _RunCoordinatorLock(tmp_path)
    second = _RunCoordinatorLock(tmp_path)
    first.acquire(allow_dead_owner=True)
    inode = first.path.stat().st_ino
    try:
        with pytest.raises(RuntimeError, match="coordinator"):
            second.acquire(allow_dead_owner=True)
    finally:
        first.release()
    assert first.path.stat().st_ino == inode
    second.acquire(allow_dead_owner=True)
    second.release()
    assert first.path.stat().st_ino == inode


def test_cross_process_contender_is_excluded_then_crashed_owner_recovers(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    output, release = ctx.Queue(), ctx.Event()
    owner = ctx.Process(target=_try_lock, args=(tmp_path, output, release))
    contender = None
    owner.start()
    try:
        assert output.get(timeout=20) == "acquired"
        inode = (tmp_path / "coordinator.lock").stat().st_ino
        contender = ctx.Process(target=_try_lock, args=(tmp_path, output, release))
        contender.start()
        assert output.get(timeout=20) == "blocked"
        contender.join(10)
        assert contender.exitcode == 0
        owner.kill()
        owner.join(10)
        assert not owner.is_alive()
        recovered = _RunCoordinatorLock(tmp_path)
        recovered.acquire(allow_dead_owner=True)
        recovered.release()
        assert (tmp_path / "coordinator.lock").stat().st_ino == inode
    finally:
        # A killed Event waiter may leave its semaphore locked. Never touch
        # that synchronization primitive again after deliberate crash injection.
        for process in (owner, contender):
            if process is not None:
                if process.is_alive():
                    process.kill()
                process.join(10)
        output.close()
        output.join_thread()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="Linux lock safety")
def test_symlink_lock_does_not_overwrite_target(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (tmp_path / "coordinator.lock").symlink_to(target)
    with pytest.raises(RuntimeError, match="safely openable"):
        _RunCoordinatorLock(tmp_path).acquire(allow_dead_owner=True)
    assert target.read_text() == "preserve"


def test_acquire_unexpected_flock_error_closes_open_descriptor(tmp_path, monkeypatch):
    import errno
    import fcntl

    descriptors = []

    def fail(fd, flags):
        descriptors.append(fd)
        raise OSError(errno.EIO, "injected lock I/O failure")

    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError):
        _RunCoordinatorLock(tmp_path).acquire(allow_dead_owner=True)
    assert len(descriptors) == 1
    with pytest.raises(OSError) as caught:
        os.fstat(descriptors[0])
    assert caught.value.errno == errno.EBADF


def test_unlock_failure_still_closes_descriptor(tmp_path, monkeypatch):
    import errno
    import fcntl

    lock = _RunCoordinatorLock(tmp_path)
    lock.acquire(allow_dead_owner=True)
    descriptor = lock._fd

    def fail(fd, flags):
        raise OSError(errno.EIO, "injected unlock failure")

    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError):
        lock.release()
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF
    assert not lock.owned
