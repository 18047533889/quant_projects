# -*- coding: utf-8 -*-
"""Regression tests for R21-IMPL-CLOSURE-HASH.

Verifies that PhysicalImplementationSpec treats implementation_closure_hash as a
required SHA-256 binding and that blank / stale values are rejected at the
contract boundary.
"""
from __future__ import annotations

import pytest

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

_VALID_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
_INVALID_HASH = "ts_mean:v2"


def _spec(**overrides) -> PhysicalImplementationSpec:
    base = dict(
        canonical="ts_corr",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        implementation_source_hash=_VALID_HASH,
        emitter_identity="polars.expr:v1",
        parameter_domain_hash=_VALID_HASH,
        semantic_contract_hash=_VALID_HASH,
        implementation_closure_hash=_VALID_HASH,
    )
    base.update(overrides)
    return PhysicalImplementationSpec(**base)


class TestImplementationClosureHashContract:
    """R21-IMPL-CLOSURE-HASH: closure-hash contract enforcement."""

    def test_blank_closure_hash_rejects_production_eligibility(self) -> None:
        spec = _spec(implementation_closure_hash="")
        assert not spec.is_production_eligible()
        assert spec.physical_implementation_id is None
        assert any("implementation_closure_hash" in e for e in spec.validation_errors())

    def test_invalid_closure_hash_rejects_production_eligibility(self) -> None:
        spec = _spec(implementation_closure_hash=_INVALID_HASH)
        assert not spec.is_production_eligible()
        assert spec.physical_implementation_id is None
        errors = spec.validation_errors()
        assert any("implementation_closure_hash" in e and "invalid" in e.lower() for e in errors)

    def test_valid_closure_hash_included_in_physical_id(self) -> None:
        base = _spec()
        changed = _spec(implementation_closure_hash="b" * 64)
        assert base.physical_implementation_id is not None
        assert changed.physical_implementation_id is not None
        assert base.physical_implementation_id != changed.physical_implementation_id
        assert str(base.physical_implementation_id).startswith("pi:v3:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
