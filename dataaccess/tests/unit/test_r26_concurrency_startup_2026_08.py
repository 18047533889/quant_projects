# -*- coding: utf-8 -*-
"""R26 —— 并发身份隔离 + startup gate / ready 测试。

T-R26-SEC-007：并发两个 API principal 1000 次 nested reads，不串身份。
T-R26-SVC-001..005：credential invalid / legacy root strict / engine probe /
critical calendar / snapshot provider → startup gate / ready 行为。
"""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from data_access.core.exceptions import AccessDeniedError
from data_access.registry import load_registry
from data_access.core.engine import DuckDBEngine
from data_access.store import DataAccessStore


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch):
    from data_access.security.policy import set_authorizer
    from data_access.security.api_principals import reset_api_principal_registry
    from data_access.runtime.resource_governor import reset_global_governor

    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("DATA_ACCESS_PRINCIPAL_ID", raising=False)
    monkeypatch.delenv("DATA_ACCESS_ALLOWED_DATASETS", raising=False)
    monkeypatch.delenv("DATA_ACCESS_API_PRINCIPALS", raising=False)
    set_authorizer(None)
    reset_api_principal_registry()
    reset_global_governor()
    yield
    set_authorizer(None)
    reset_api_principal_registry()
    reset_global_governor()


def _base_store():
    registry = load_registry()
    engine = DuckDBEngine(threads=2, enable_object_cache=False)
    return DataAccessStore(registry=registry, engine=engine)


def test_tr26_sec007_concurrent_principals_no_crosstalk(tmp_path):
    """R26-P0-005：并发 API principal A/B 1000 次 nested reads，不串身份。

    A（basic）只能读 basic dataset；B（premium）可读 premium。
    A 永远不能读 premium；B 不被 A 的权限降级。
    """
    from data_access.security.execution_context import (
        DataAccessExecutionContext,
        execution_scope,
    )
    from data_access.security.principal import AccessPolicy, DataPrincipal
    from data_access.security.policy import DefaultAuthorizer

    store = _base_store()
    basic_policy = AccessPolicy(
        allowed_datasets=frozenset({"ashare_stock_daily"}),
        allowed_actions=frozenset({"dataset:read", "metadata:read"}),
    )
    premium_policy = AccessPolicy(
        allowed_datasets=frozenset({"ashare_stock_daily", "us_stock_daily"}),
        allowed_actions=frozenset({"dataset:read", "metadata:read"}),
    )
    basic_principal = DataPrincipal(principal_id="basic", server_id="basic")
    premium_principal = DataPrincipal(principal_id="premium", server_id="premium")

    errors: list[Exception] = []
    counts = {"basic_ok": 0, "basic_denied": 0, "premium_ok": 0, "premium_denied": 0}

    def _read_for(ctx, dataset):
        try:
            with execution_scope(ctx):
                store.authorize_dataset(dataset)
            return True
        except AccessDeniedError:
            return False

    def thread_a():
        ctx = DataAccessExecutionContext(
            principal=basic_principal,
            authorizer=DefaultAuthorizer(
                policy=basic_policy, principal=basic_principal, strict_default_deny=True
            ),
            access_policy=basic_policy,
            request_id="req-A",
        )
        try:
            for _ in range(500):
                assert _read_for(ctx, "ashare_stock_daily") is True
                counts["basic_ok"] += 1
                assert _read_for(ctx, "us_stock_daily") is False
                counts["basic_denied"] += 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def thread_b():
        ctx = DataAccessExecutionContext(
            principal=premium_principal,
            authorizer=DefaultAuthorizer(
                policy=premium_policy,
                principal=premium_principal,
                strict_default_deny=True,
            ),
            access_policy=premium_policy,
            request_id="req-B",
        )
        try:
            for _ in range(500):
                assert _read_for(ctx, "ashare_stock_daily") is True
                counts["premium_ok"] += 1
                assert _read_for(ctx, "us_stock_daily") is True
                counts["premium_ok"] += 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, errors
    # A 永远不能读 premium。
    assert counts["basic_ok"] == 500
    assert counts["basic_denied"] == 500
    # B 不被 A 的权限降级——B 的 premium 读全部成功。
    assert counts["premium_ok"] == 1000


# ---------------------------------------------------------------------------
# Startup gate / ready（T-R26-SVC）
# ---------------------------------------------------------------------------
def test_tr26_svc001_startup_gate_production_blocks_missing_security():
    """R26-P0-007/020：production 无显式 policy → startup gate fail。"""
    from data_access.runtime.startup_gate import run_startup_gate

    store = MagicMock()
    with patch("data_access.security.policy.production_security_configured", return_value=False):
        with patch("data_access.read.query_budget.is_strict_semantics", return_value=True):
            with pytest.raises(RuntimeError, match="startup gate failed"):
                run_startup_gate(store, production=True)


def _http_client(store=None):
    from fastapi.testclient import TestClient

    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    with patch("data_access.service.app.get_store", return_value=store or _base_store()):
        app = create_app(
            ServiceSettings(api_key="testkey", production_mode=False, allow_open=False)
        )
    return TestClient(app)


def _http_client_with_store(store):
    """创建 client，get_store patch 在请求期间保持生效。"""
    from fastapi.testclient import TestClient

    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    app = create_app(
        ServiceSettings(api_key="testkey", production_mode=False, allow_open=False)
    )
    return TestClient(app), store


def test_tr26_svc003_ready_uses_real_engine_api():
    """R26-P0-021：/ready 用真实 public engine API（execute_arrow），不是 .execute()。"""
    client, store = _http_client_with_store(_base_store())
    with patch("data_access.service.app.get_store", return_value=store):
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["engine"] == "ok"


def test_tr26_svc_ready_engine_probe_fails_503():
    """R26-P0-021：engine 探测失败 → 503（不返回 ready）。"""
    store = _base_store()
    client, _ = _http_client_with_store(store)
    with patch("data_access.service.app.get_store", return_value=store):
        with patch.object(
            store._engine, "execute_arrow", side_effect=RuntimeError("boom")
        ):
            resp = client.get("/ready")
    assert resp.status_code == 503


def test_ready_rejects_degraded_certificate_even_when_passed():
    """Readiness is fail-closed for non-production DEGRADED startup results."""
    from fastapi.testclient import TestClient
    from data_access.runtime.startup_gate import build_startup_certificate
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    store = _base_store()
    store._startup_certificate = build_startup_certificate(True, ["degraded"])
    app = create_app(ServiceSettings(api_key="testkey"))
    with patch("data_access.service.app.get_store", return_value=store):
        with TestClient(app) as client:
            response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Service not ready (startup checks failed)"


def test_lifespan_runs_gate_and_publishes_certificate():
    """ASGI startup runs the gate before serving and caches its certificate."""
    from fastapi.testclient import TestClient
    from data_access.runtime.startup_gate import build_startup_certificate
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    store = _base_store()
    certificate = build_startup_certificate(True, [])
    app = create_app(ServiceSettings(api_key="testkey", production_mode=True))
    with patch("data_access.service.app.get_store", return_value=store), patch(
        "data_access.runtime.startup_gate.run_startup_gate",
        return_value=certificate,
    ) as gate:
        with TestClient(app) as client:
            assert client.get("/ready").status_code == 200
    gate.assert_called_once_with(store, production=True)
    assert store._startup_certificate is certificate


def test_ready_fails_closed_when_current_startup_identity_cannot_be_built():
    """A certificate cannot mask an unavailable current environment identity."""
    from fastapi.testclient import TestClient
    from data_access.runtime.startup_gate import build_startup_certificate
    from data_access.service.app import create_app
    from data_access.service.config import ServiceSettings

    store = _base_store()
    store._startup_certificate = build_startup_certificate(True, [])
    app = create_app(ServiceSettings(api_key="testkey"))
    with patch("data_access.service.app.get_store", return_value=store), patch(
        "data_access.runtime.startup_subject.build_startup_subject_digest",
        side_effect=RuntimeError("identity unavailable"),
    ):
        with TestClient(app) as client:
            response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Service not ready (startup identity unavailable)"


def test_require_startup_certificate_reuses_valid_subject_bound_cache():
    """A valid, currently bound cache is reused without rerunning the gate."""
    from data_access.runtime.startup_gate import (
        build_startup_certificate,
        require_startup_certificate,
    )

    store = _base_store()
    certificate = build_startup_certificate(True, [], subject_digest="current")
    store._startup_certificate = certificate
    subject = Mock()
    subject.to_digest.return_value = "current"
    with patch(
        "data_access.runtime.startup_subject.build_startup_subject_digest",
        return_value=subject,
    ), patch("data_access.runtime.startup_gate.run_startup_gate") as gate:
        assert require_startup_certificate(store) is certificate
    gate.assert_not_called()


def test_require_startup_certificate_reruns_gate_when_subject_unavailable():
    """An unavailable current identity must not make a cached certificate valid."""
    from data_access.runtime.startup_gate import (
        build_startup_certificate,
        require_startup_certificate,
    )

    store = _base_store()
    cached = build_startup_certificate(True, [], subject_digest="current")
    replacement = build_startup_certificate(True, [], subject_digest="replacement")
    store._startup_certificate = cached
    with patch(
        "data_access.runtime.startup_subject.build_startup_subject_digest",
        side_effect=RuntimeError("identity unavailable"),
    ), patch(
        "data_access.runtime.startup_gate.run_startup_gate",
        return_value=replacement,
    ) as gate:
        assert require_startup_certificate(store) is replacement
    gate.assert_called_once_with(store, production=None, checks=None)
    assert store._startup_certificate is replacement
