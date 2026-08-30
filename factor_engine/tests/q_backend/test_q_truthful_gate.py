"""Q/KDB truthful fail-closed gate (backlog task #60).

The q backend must have a REAL integration path — an actual q connection OR an
honest "not available in this env" fail-closed gate — NOT a fake success.

This environment has no q/kdb+ binary, no pykx/qpython/pyq client, and no
Q_LICENSED env.  These tests lock in the honest behavior:

  1. ``QProcessManager.check_availability()`` reports a non-AVAILABLE status
     (never a fabricated AVAILABLE), with a clear error message.
  2. Constructing a production ``QBackend`` raises ``BackendUnavailableError``
     instead of silently pretending a q runtime exists.
  3. The executor's readiness check honestly reports not-ready.
  4. ``Q_ENABLE`` alone (no binary / no licensed pykx) does NOT manufacture an
     available q runtime — the gate stays fail-closed.
"""
from __future__ import annotations

import importlib.util
import os
import shutil

import pytest

from factor_engine.backend.q_backend.q_backend import QBackend
from factor_engine.backend.q_backend.q_executor import get_q_executor
from factor_engine.backend.q_backend.q_process_manager import (
    QAvailabilityStatus,
    QProcessManager,
    get_q_process_manager,
)
from factor_engine.backend.operator_capability import BackendUnavailableError


def _has_real_q_evidence() -> bool:
    """True when this environment genuinely has a q integration path."""
    if shutil.which("q") is not None or shutil.which("kdb") is not None:
        return True
    if importlib.util.find_spec("pykx") is not None:
        return True
    if importlib.util.find_spec("qpython") is not None:
        return True
    if importlib.util.find_spec("pyq") is not None:
        return True
    return False


def test_availability_is_honest_in_this_env():
    """The availability gate must reflect reality, never fabricate success."""
    manager = QProcessManager()
    info = manager.check_availability()
    # If the environment genuinely has q, the gate must say AVAILABLE (real
    # integration path).  If it does not, the gate must fail closed with a
    # clear status and message — never a silent AVAILABLE.
    if _has_real_q_evidence():
        assert info.status == QAvailabilityStatus.AVAILABLE
    else:
        assert info.status in {
            QAvailabilityStatus.UNAVAILABLE,
            QAvailabilityStatus.LICENSE_MISSING,
            QAvailabilityStatus.PROCESS_FAILED,
        }
        assert info.error_message, "fail-closed status must carry a clear error message"


def test_production_qbackend_fails_closed_when_no_q():
    """Production QBackend must NOT silently construct without a q runtime."""
    if _has_real_q_evidence():
        # Real integration path: construction is allowed; skip the fail-closed
        # assertion (the executor gate will verify a real connection).
        pytest.skip("real q integration path present in this env")
    with pytest.raises(BackendUnavailableError) as exc:
        QBackend(fallback_to_pandas=False, production_mode=True)
    message = str(exc.value)
    assert "q" in message.lower()
    assert "not" in message.lower() or "unavailable" in message.lower()


def test_executor_readiness_is_honest():
    """Executor readiness must reflect the real gate, not claim a fake runtime."""
    ready, message = get_q_executor().check_execution_readiness()
    if _has_real_q_evidence():
        # Real integration path: executor must be ready.
        assert ready
    else:
        assert ready is False
        assert message and "q unavailable" in message


def test_q_enable_alone_does_not_fabricate_availability(monkeypatch):
    """Q_ENABLE is an override, not a fake-success switch."""
    monkeypatch.setenv("Q_ENABLE", "1")
    # Ensure no binary and no client module is actually present, regardless of
    # the host environment, so this test asserts the gate's fail-closed logic.
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    manager = QProcessManager()
    info = manager.check_availability()
    assert info.status != QAvailabilityStatus.AVAILABLE
    assert info.error_message and "fail-closed" in info.error_message
