import contextlib
import os
import signal
from types import SimpleNamespace

import pytest
import smoke_catalog as smoke


def valid_scope(monkeypatch):
    monkeypatch.setenv("FACTOR_CATALOG_EXTERNAL_WATCHDOG", "1")
    monkeypatch.setenv("FACTOR_CATALOG_WATCHDOG_PARENT_PID", str(os.getppid()))
    monkeypatch.setenv("FACTOR_CATALOG_WATCHDOG_TOKEN", "a" * 32)


def test_external_watchdog_mode_rejects_missing_scope(monkeypatch):
    for name in smoke.WATCHDOG_SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="external watchdog"):
        smoke.validate_external_watchdog_scope()


def test_external_watchdog_deadline_never_installs_sigalrm(monkeypatch):
    valid_scope(monkeypatch)
    calls = []
    monkeypatch.setattr(signal, "signal", lambda *a: calls.append(("signal", a)))
    monkeypatch.setattr(signal, "setitimer", lambda *a: calls.append(("timer", a)))
    with smoke.execution_deadline(1, "external-watchdog"):
        pass
    assert calls == []


def test_legacy_cooperative_is_the_only_mode_using_factor_deadline(monkeypatch):
    entered = []
    @contextlib.contextmanager
    def fake_deadline(seconds):
        entered.append(seconds)
        yield
    monkeypatch.setattr(smoke, "factor_deadline", fake_deadline)
    with smoke.execution_deadline(7, "legacy-cooperative"):
        pass
    assert entered == [7]


def test_preflight_external_mode_does_not_use_cooperative_alarm(monkeypatch):
    valid_scope(monkeypatch)
    monkeypatch.setattr(smoke, "factor_deadline", lambda _seconds: pytest.fail("SIGALRM path used"))
    factor = SimpleNamespace(name="ok")
    rows = {"ok": {}}
    class Engine:
        def compile(self, _factor):
            return None
    ready = smoke.preflight_factors(
        Engine(), [factor], rows, timeout_seconds=1, deadline_mode="external-watchdog"
    )
    assert ready == [factor]


def test_backend_factory_receives_requested_backend(monkeypatch):
    sentinel = object()
    calls = []
    import factor_engine.backend.factory as factory
    monkeypatch.setattr(factory, "build_backend", lambda name: calls.append(name) or sentinel)
    assert smoke.build_execution_backend("auto") is sentinel
    assert calls == ["auto"]


def test_execution_scope_fields_are_per_row_audit_evidence():
    assert smoke.execution_scope_fields("2025-01-01", "2026-04-30", ["a", "b"]) == {
        "execution_window": ["2025-01-01", "2026-04-30"],
        "execution_symbol_count": 2,
    }
