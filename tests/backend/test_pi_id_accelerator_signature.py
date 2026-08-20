# -*- coding: utf-8 -*-
"""Regression tests for R21-PI-ID-ACCELERATOR-FIX.

Verifies that PhysicalImplementationID hash includes accelerator and
kernel_signature, so switching these fields produces a different ID.
"""
from __future__ import annotations

import pytest

from backend.contracts import (
    Accelerator,
    ExecutionKind,
    PhysicalImplementationSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KW = dict(
    canonical="ts_mean",
    backend="polars",
    execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
    supports_lazy=True,
    supports_streaming=True,
    supports_nulls=True,
    implementation_source_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    emitter_identity="test.polars.expr:v1",
    parameter_domain_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    semantic_contract_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    implementation_closure_hash="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
)


def _spec(**overrides) -> PhysicalImplementationSpec:
    """Return a fully valid spec with optional overrides."""
    kw = {**_BASE_KW, **overrides}
    return PhysicalImplementationSpec(**kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPIIDAcceleratorSignature:
    """R21-PI-ID-ACCELERATOR-FIX: accelerator + kernel_signature in payload."""

    def test_different_accelerator_yields_different_hash(self) -> None:
        """Switching accelerator (NONE -> NUMBA_CPU) must change the PI ID."""
        spec_none = _spec(
            accelerator=Accelerator.NONE,
            kernel_signature="",
        )
        spec_numba = _spec(
            accelerator=Accelerator.NUMBA_CPU,
            kernel_signature="",
        )
        id_none = spec_none.physical_implementation_id
        id_numba = spec_numba.physical_implementation_id

        assert id_none is not None
        assert id_numba is not None
        assert id_none != id_numba, "Different accelerator must yield different PI ID"

    def test_different_kernel_signature_yields_different_hash(self) -> None:
        """Switching kernel_signature must change the PI ID."""
        spec_a = _spec(
            accelerator=Accelerator.NONE,
            kernel_signature="kernel_a_v1",
        )
        spec_b = _spec(
            accelerator=Accelerator.NONE,
            kernel_signature="kernel_b_v2",
        )
        id_a = spec_a.physical_implementation_id
        id_b = spec_b.physical_implementation_id

        assert id_a is not None
        assert id_b is not None
        assert id_a != id_b, "Different kernel_signature must yield different PI ID"

    def test_same_spec_yields_same_hash(self) -> None:
        """Two identical specs must produce the same PI ID (stability)."""
        spec_1 = _spec(
            accelerator=Accelerator.NUMBA_CPU,
            kernel_signature="stable_kernel_v3",
        )
        spec_2 = _spec(
            accelerator=Accelerator.NUMBA_CPU,
            kernel_signature="stable_kernel_v3",
        )
        id_1 = spec_1.physical_implementation_id
        id_2 = spec_2.physical_implementation_id

        assert id_1 is not None
        assert id_2 is not None
        assert id_1 == id_2, "Identical specs must yield the same PI ID"

    def test_version_prefix_is_v3(self) -> None:
        """New payloads must use the pi:v3: prefix."""
        spec = _spec()
        pid = spec.physical_implementation_id
        assert pid is not None
        assert str(pid).startswith("pi:v3:"), f"Expected pi:v3: prefix, got {pid}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
