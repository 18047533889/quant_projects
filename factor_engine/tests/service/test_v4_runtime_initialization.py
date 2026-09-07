from types import SimpleNamespace

import pytest

from factor_engine.service import app


@pytest.fixture
def isolated_runtime(monkeypatch):
    monkeypatch.setattr(app, "_RUNTIME_CONSTRUCTED", False)
    monkeypatch.setattr(app, "_RUNTIME_ACTIVE", False)
    monkeypatch.setattr(app, "_RUNTIME_OWNED", frozenset())
    monkeypatch.setattr(app, "_RUNTIME_START_FAILURE", None)
    for name in ("STORE", "EXECUTOR", "QUEUE", "FEATURE_POLICY", "SOURCE_POLICY"):
        monkeypatch.setattr(app, name, app._LazyRuntimeProxy(name))
    made = []

    class Store:
        def __init__(self):
            self.closed = False
            made.append(self)
        def close(self):
            self.closed = True

    class Executor:
        def __init__(self, **kwargs):
            self.closed = False
            made.append(self)
        def shutdown(self, **kwargs):
            self.closed = True

    class Queue:
        def __init__(self, **kwargs):
            self._started = False
            self.starts = 0
            self.stopped = False
            made.append(self)
        def start(self, store):
            self.starts += 1
            self._started = True
        def stop(self):
            self.stopped = True
        def drain(self, **kwargs):
            self.stopped = True

    monkeypatch.setattr(app, "JobStore", Store)
    monkeypatch.setattr(app, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(app, "BoundedJobQueue", Queue)
    monkeypatch.setattr(app.RuntimeFeaturePolicy, "from_env", lambda: SimpleNamespace())
    monkeypatch.setattr(app.ApprovedSourcePolicy, "from_env", lambda: SimpleNamespace())
    return made


def test_construct_without_start_then_start_once(isolated_runtime):
    app._ensure_runtime(start=False)
    assert app._RUNTIME_CONSTRUCTED
    assert not app.QUEUE._started
    app._ensure_runtime(start=True)
    app._ensure_runtime(start=True)
    assert app.QUEUE._started and app.QUEUE.starts == 1


def test_policy_failure_does_not_poison_runtime(isolated_runtime, monkeypatch):
    def fail():
        raise ValueError("invalid source policy")
    monkeypatch.setattr(app.ApprovedSourcePolicy, "from_env", fail)
    with pytest.raises(ValueError, match="invalid source policy"):
        app._ensure_runtime()
    assert not app._RUNTIME_CONSTRUCTED
    assert not isolated_runtime, "validate policy before creating resources"
    monkeypatch.setattr(app.ApprovedSourcePolicy, "from_env", lambda: SimpleNamespace())
    app._ensure_runtime()
    assert app._RUNTIME_CONSTRUCTED and app.QUEUE._started


def test_queue_construction_failure_releases_owned_resources(isolated_runtime, monkeypatch):
    class BrokenQueue:
        def __init__(self, **kwargs):
            raise RuntimeError("queue construction failed")
    monkeypatch.setattr(app, "BoundedJobQueue", BrokenQueue)
    with pytest.raises(RuntimeError, match="queue construction failed"):
        app._ensure_runtime()
    assert not app._RUNTIME_CONSTRUCTED
    assert all(item.closed for item in isolated_runtime)
    assert isinstance(app.STORE, app._LazyRuntimeProxy)
    assert isinstance(app.EXECUTOR, app._LazyRuntimeProxy)


def test_failed_preflight_does_not_start_runtime(isolated_runtime, monkeypatch):
    import asyncio
    from factor_engine.service import preflight
    monkeypatch.setattr(preflight, "production_preflight", lambda: {"ok": False})
    monkeypatch.setattr(app, "resolve_ambient_run_mode", lambda: "production")
    service = app.create_app()
    async def enter():
        async with service.router.lifespan_context(service):
            pytest.fail("failed preflight must not serve requests")
    with pytest.raises(RuntimeError, match="production preflight failed"):
        asyncio.run(enter())
    assert not isolated_runtime


def test_lifespan_error_still_closes_runtime(isolated_runtime, monkeypatch):
    import asyncio
    from factor_engine.service import preflight
    monkeypatch.setattr(preflight, "production_preflight", lambda: {"ok": True})
    monkeypatch.setattr(app, "resolve_ambient_run_mode", lambda: "research")
    service = app.create_app()
    async def enter():
        async with service.router.lifespan_context(service):
            raise RuntimeError("lifespan body error")
    with pytest.raises(RuntimeError, match="lifespan body error"):
        asyncio.run(enter())
    assert all(getattr(item, "closed", False) or getattr(item, "stopped", False)
               for item in isolated_runtime)
    assert not app._RUNTIME_CONSTRUCTED
    assert not app._RUNTIME_ACTIVE


def test_lifespan_restart_constructs_fresh_runtime(isolated_runtime, monkeypatch):
    import asyncio
    from factor_engine.service import preflight
    monkeypatch.setattr(preflight, "production_preflight", lambda: {"ok": True})
    monkeypatch.setattr(app, "resolve_ambient_run_mode", lambda: "research")
    service = app.create_app()

    async def enter_twice():
        async with service.router.lifespan_context(service):
            first = (app.STORE, app.EXECUTOR, app.QUEUE)
            assert app.QUEUE._started
        async with service.router.lifespan_context(service):
            second = (app.STORE, app.EXECUTOR, app.QUEUE)
            assert app.QUEUE._started
            assert all(new is not old for new, old in zip(second, first))

    asyncio.run(enter_twice())
    assert len(isolated_runtime) == 6


def test_concurrent_lifespan_is_rejected(isolated_runtime, monkeypatch):
    import asyncio
    from factor_engine.service import preflight
    monkeypatch.setattr(preflight, "production_preflight", lambda: {"ok": True})
    monkeypatch.setattr(app, "resolve_ambient_run_mode", lambda: "research")
    service = app.create_app()

    async def nest():
        async with service.router.lifespan_context(service):
            with pytest.raises(RuntimeError, match="active lifespan"):
                async with service.router.lifespan_context(service):
                    pytest.fail("concurrent lifespan must not reuse runtime")

    asyncio.run(nest())


def test_partial_queue_start_failure_preserves_primary(isolated_runtime, monkeypatch):
    class BrokenQueue(app.BoundedJobQueue):
        def start(self, store):
            self._started = True
            raise LookupError("queue start failed")
        def stop(self):
            raise RuntimeError("rollback failed")

    monkeypatch.setattr(app, "BoundedJobQueue", BrokenQueue)
    with pytest.raises(LookupError, match="queue start failed"):
        app._ensure_runtime()
    assert not app._RUNTIME_CONSTRUCTED
    assert isinstance(app.STORE, app._LazyRuntimeProxy)
    assert isinstance(app.EXECUTOR, app._LazyRuntimeProxy)
    assert isinstance(app.QUEUE, app._LazyRuntimeProxy)


def test_delayed_owned_queue_start_failure_discards_and_retries(isolated_runtime, monkeypatch):
    app._ensure_runtime(start=False)
    first = (app.STORE, app.EXECUTOR, app.QUEUE)
    queue_type = type(app.QUEUE)
    original_start = queue_type.start
    calls = 0

    def fail_once(self, store):
        nonlocal calls
        calls += 1
        self._started = True
        if calls == 1:
            raise LookupError("delayed queue start failed")
        return original_start(self, store)

    monkeypatch.setattr(queue_type, "start", fail_once)
    with pytest.raises(LookupError, match="delayed queue start failed"):
        app._ensure_runtime(start=True)
    assert not app._RUNTIME_CONSTRUCTED
    assert isinstance(app.STORE, app._LazyRuntimeProxy)
    assert isinstance(app.EXECUTOR, app._LazyRuntimeProxy)
    assert isinstance(app.QUEUE, app._LazyRuntimeProxy)
    assert first[0].closed and first[1].closed and first[2].stopped

    app._ensure_runtime(start=True)
    assert app._RUNTIME_CONSTRUCTED and app.QUEUE._started
    assert all(new is not old for new, old in zip((app.STORE, app.EXECUTOR, app.QUEUE), first))


def test_preflight_accepts_not_yet_created_service_root(tmp_path, monkeypatch):
    import shutil
    from factor_engine.service import preflight

    configured = tmp_path / "new" / "service-root"
    seen = []
    original = shutil.disk_usage

    def disk_usage(path):
        seen.append(path)
        return original(path)

    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(configured))
    monkeypatch.setenv("FACTOR_ENGINE_MIN_FREE_DISK_MB", "0")
    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    result = preflight.production_preflight()
    assert result["checks"]["disk_space"]["ok"]
    assert seen == [tmp_path]
    assert not configured.exists()


def test_preflight_rejects_existing_file_as_service_root(tmp_path, monkeypatch):
    from factor_engine.service import preflight

    configured = tmp_path / "not-a-directory"
    configured.write_text("x")
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(configured))
    result = preflight.production_preflight()
    check = result["checks"]["disk_space"]
    assert not check["ok"]
    assert "not a directory" in check["error"]


def test_preflight_rejects_existing_file_ancestor(tmp_path, monkeypatch):
    from factor_engine.service import preflight

    ancestor = tmp_path / "not-a-directory"
    ancestor.write_text("x")
    monkeypatch.setenv("FACTOR_ENGINE_SERVICE_ROOT", str(ancestor / "child"))
    result = preflight.production_preflight()
    check = result["checks"]["disk_space"]
    assert not check["ok"]
    assert "ancestor is not a directory" in check["error"]
