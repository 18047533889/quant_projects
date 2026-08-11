# -*- coding: utf-8 -*-
"""R40 #140/#141: execution-context-scoped modin selection (no os.environ
mutation) and the clickhouse_sql backend-chain certificate.

``PandasBackend()`` construction registers thousands of kernels and is far too
slow for a unit test, so #140 is tested against a lightweight stub that records
the ``use_modin_pandas`` argument."""
from __future__ import annotations

import os

import pytest


class TestPandasModinContextScoped:
    def test_pandas_modin_choice_context_scoped_not_env(self, monkeypatch):
        import backend.factory as factory
        import backend.pandas_compat as pc

        calls = {}

        class _StubPandasBackend:
            def __init__(self, use_modin_pandas=None):
                calls["use_modin_pandas"] = use_modin_pandas

        monkeypatch.setattr(factory, "PandasBackend", _StubPandasBackend)

        prev_env = os.environ.get("FACTOR_ENGINE_USE_MODIN")
        os.environ.pop("FACTOR_ENGINE_USE_MODIN", None)
        pc._MODIN_ENABLED.set(False)
        try:
            backend = factory.build_backend("pandas_modin")
            # build_backend must NOT mutate process-global env.
            assert os.environ.get("FACTOR_ENGINE_USE_MODIN") is None
            # The modin choice is passed to the backend (context-scoped), not env.
            assert calls["use_modin_pandas"] is True
            assert backend is not None
            # The ContextVar remains untouched by the factory itself (the real
            # PandasBackend sets it via set_modin_enabled — tested separately).
            assert pc._MODIN_ENABLED.get() is False
        finally:
            if prev_env is not None:
                os.environ["FACTOR_ENGINE_USE_MODIN"] = prev_env
            pc._MODIN_ENABLED.set(False)

    def test_pandas_plain_does_not_enable_modin(self, monkeypatch):
        import backend.factory as factory
        import backend.pandas_compat as pc

        calls = {}

        class _StubPandasBackend:
            def __init__(self, use_modin_pandas=None):
                calls["use_modin_pandas"] = use_modin_pandas

        monkeypatch.setattr(factory, "PandasBackend", _StubPandasBackend)
        pc._MODIN_ENABLED.set(False)
        try:
            factory.build_backend("pandas")
            assert calls["use_modin_pandas"] is None
            assert pc._MODIN_ENABLED.get() is False
        finally:
            pc._MODIN_ENABLED.set(False)

    def test_set_modin_enabled_uses_contextvar_not_env(self):
        import backend.pandas_compat as pc

        os.environ.pop("FACTOR_ENGINE_USE_MODIN", None)
        pc.set_modin_enabled(True)
        assert pc._MODIN_ENABLED.get() is True
        assert os.environ.get("FACTOR_ENGINE_USE_MODIN") is None
        pc.set_modin_enabled(False)
        assert pc._MODIN_ENABLED.get() is False


class TestClickhouseCertificateChain:
    def test_clickhouse_sql_certificate_records_backend_chain(self, monkeypatch):
        from backend.factory import build_backend, build_backend_execution_certificate
        from runtime.production_execution_certificate import (
            ProductionExecutionCertificate,
        )

        # Stub the DuckDBPushdownBackend to avoid heavy construction. The lazy
        # ``from .duckdb_pushdown_backend import DuckDBPushdownBackend`` inside
        # build_backend picks up the stub from the module attribute.
        class _StubBackend:
            pass

        import backend.duckdb_pushdown_backend as dpb

        monkeypatch.setattr(dpb, "DuckDBPushdownBackend", _StubBackend)
        backend = build_backend("clickhouse_sql")
        assert backend._requested_backend == "clickhouse_sql"
        assert backend._resolved_dialect == "clickhouse"

        cert = build_backend_execution_certificate(
            backend,
            structural_hash="abc123",
            bound_ops=["ts_mean"],
            backend_eligibility=["clickhouse_sql"],
            output_shape_hash="shape1",
        )
        assert isinstance(cert, ProductionExecutionCertificate)
        assert cert.requested_backend == "clickhouse_sql"
        assert cert.resolved_dialect == "clickhouse"
        # certificate hash binds the backend chain — changing the dialect changes it.
        cert2 = ProductionExecutionCertificate.build(
            structural_hash="abc123",
            bound_ops=["ts_mean"],
            backend_eligibility=["clickhouse_sql"],
            output_shape_hash="shape1",
            requested_backend="clickhouse_sql",
            resolved_dialect="duckdb",
        )
        assert cert.certificate_hash != cert2.certificate_hash
        # integrity + backend event validation still O(1)
        assert cert.validate({"backend": "clickhouse_sql", "no_fallback": True}) is True
