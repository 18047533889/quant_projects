import gc
import multiprocessing
import os
import threading
import weakref

import pytest

from quant_evaluator.runtime.backend_calibration import BoundedCalibrationCache


def test_cache_registry_does_not_keep_instances_alive():
    cache = BoundedCalibrationCache()
    reference = weakref.ref(cache)
    del cache

    gc.collect()

    assert reference() is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_resets_held_cache_lock_and_clears_child_records():
    cache = BoundedCalibrationCache()
    cache.put("parent", {"route": "cpu"})
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with cache._lock:
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert acquired.wait(2)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def exercise_child_cache():
        inherited_empty = cache.get("parent") is None and len(cache) == 0
        cache.put("child", {"route": "gpu"})
        inherited_cache_usable = cache.get("child") == {"route": "gpu"}
        fresh_cache = BoundedCalibrationCache()
        fresh_cache.put("fresh", {"route": "cpu"})
        fresh_cache_usable = fresh_cache.get("fresh") == {"route": "cpu"}
        sender.send((inherited_empty, inherited_cache_usable, fresh_cache_usable))
        sender.close()

    child = context.Process(target=exercise_child_cache)
    try:
        child.start()
        assert receiver.poll(3), "child blocked on an inherited calibration-cache lock"
        assert receiver.recv() == (True, True, True)
        child.join(1)
        assert child.exitcode == 0
        release.set()
        assert cache.get("parent") == {"route": "cpu"}
        assert len(cache) == 1
    finally:
        if child.is_alive():
            child.terminate()
            child.join(1)
        release.set()
        holder.join(1)
        receiver.close()
        sender.close()
