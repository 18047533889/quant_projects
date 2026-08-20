# -*- coding: utf-8 -*-
"""R40 bootstrap items #210/#215."""
from __future__ import annotations

import pytest


class TestBackendReplacementAuditTrail:
    def test_replace_backend_records_audit_trail(self, writable_global_registry):
        from cleaned_operators import (
            _BACKEND_REPLACEMENT_AUDIT,
            backend_replacement_audit,
            replace_backend,
        )
        from cleaned_operators.registry import OperatorRegistry

        # replace_backend 会从 runtime 移除该 backend——保留原实现用于 finally
        # 恢复（避免在全局 registry 留下 runtime/catalog 分裂状态）。
        impl = OperatorRegistry._operators["period_lag"]["pandas_numpy"]
        baseline = len(backend_replacement_audit())
        try:
            migration = replace_backend(
                "period_lag",
                "pandas_numpy",
                reason="test migration",
                source="test.fiscal_strict",
            )
            assert isinstance(migration, str) and len(migration) >= 8
            rows = backend_replacement_audit()
            assert len(rows) == baseline + 1
            row = rows[-1]
            assert row["migration_id"] == migration
            assert row["canonical"] == "period_lag"
            assert row["reason"] == "test migration"
            # old implementation was recorded (may be empty if not registered)
            assert "old_implementation_hash" in row
        finally:
            # 恢复被移除的 backend，保持 catalog/runtime 一致（R4-102）。
            backends = OperatorRegistry._operators.get("period_lag")
            if backends is not None and "pandas_numpy" not in backends:
                backends["pandas_numpy"] = impl

    def test_backend_replacement_audit_readonly(self):
        from cleaned_operators import (
            _BACKEND_REPLACEMENT_AUDIT,
            backend_replacement_audit,
        )

        snapshot = tuple(dict(r) for r in _BACKEND_REPLACEMENT_AUDIT)
        rows = backend_replacement_audit()
        if rows:
            rows[0]["canonical"] = "mutated"  # mutating the view is a no-op
        assert tuple(dict(r) for r in _BACKEND_REPLACEMENT_AUDIT) == snapshot


class TestSignatureAuthority:
    def test_signature_authority_import_failure_fails_production(self, monkeypatch):
        from cleaned_operators import check_signature_authority

        # Simulate a missing signature authority.
        monkeypatch.setattr("cleaned_operators._SIGNATURE_AUTHORITY_AVAILABLE", False)
        with pytest.raises(RuntimeError):
            check_signature_authority(production=True)
        # research degrades
        check_signature_authority(production=False)
