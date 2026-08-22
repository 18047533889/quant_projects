# -*- coding: utf-8 -*-
"""Audit: _polars_status must never classify delegates as production_safe.

Polars delegates (polars_udf_pandas_delegate) round-trip through to_pandas(),
materialize the full panel, and execute the pandas reference kernel. They are
NOT native Polars implementations and must never be classified as production_safe,
even if they appear in POLARS_REFERENCE_PARITY_VERIFIED evidence sets.

The evidence sets certify that *some* polars implementation produces correct
results, but they do not distinguish native vs delegate execution. The capability
system must consult polars_backend_kind to reject delegates from production_safe.

Regression tests for the fix applied to operator_capability.py::_polars_status.
"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")

from backend.operator_capability import _polars_status, backend_status
from backend.polars_backend_kind import (
    PolarsImplementationKind,
    canonical_polars_is_delegate,
    canonical_polars_kind,
)
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def loaded():
    from cleaned_operators import load_all
    load_all()
    yield


def test_polars_status_rejects_delegates_from_production_safe(loaded):
    """Delegates must never reach production_safe, even with evidence."""
    from backend.primitive_evidence import (
        POLARS_REFERENCE_PARITY_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
    )

    # Check all delegates
    delegates = []
    for canon in OperatorRegistry.list_canonical():
        if "polars" not in OperatorRegistry.backends_for(canon):
            continue
        if canonical_polars_is_delegate(canon):
            status = _polars_status(canon)
            delegates.append((canon, status))
            # Core assertion: no delegate is production_safe
            assert status != "production_safe", (
                f"{canon} is a polars_udf_pandas_delegate but _polars_status "
                f"returned 'production_safe'. Delegates round-trip through "
                f"to_pandas() and must never be production-certified."
            )

    assert len(delegates) > 0, "Test requires at least one delegate to validate"


def test_polars_status_allows_native_production_safe(loaded):
    """Native implementations can be production_safe with evidence."""
    from backend.primitive_evidence import (
        POLARS_REFERENCE_PARITY_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
    )

    production_evidence = (
        POLARS_REFERENCE_PARITY_VERIFIED
        & POLARS_EDGE_VERIFIED
        & POLARS_NO_FALLBACK_VERIFIED
    )

    # Check that native operators with evidence CAN be production_safe
    native_production = []
    for canon in production_evidence:
        if "polars" not in OperatorRegistry.backends_for(canon):
            continue
        if not canonical_polars_is_delegate(canon):
            status = _polars_status(canon)
            if status == "production_safe":
                native_production.append(canon)

    # At least some native operators should be production_safe
    assert len(native_production) > 0 or len(production_evidence) == 0, (
        "Native operators with evidence should be production_safe"
    )


def test_delegate_classification_examples(loaded):
    """Verify known delegates are correctly identified."""
    # These are gap-coverage operators registered by polars_udf
    known_delegates = ["ALMA", "HMA", "CoppockCurve"]

    for canon in known_delegates:
        if "polars" not in OperatorRegistry.backends_for(canon):
            continue

        kind = canonical_polars_kind(canon)
        # Delegates may be classified as unsupported if they lack PhysicalImplementationSpec
        # The key assertion is that they are NOT polars_native
        assert kind != PolarsImplementationKind.POLARS_NATIVE, (
            f"{canon} should NOT be classified as POLARS_NATIVE"
        )

        status = _polars_status(canon)
        # Delegates should NOT be production_safe
        assert status != "production_safe", (
            f"{canon} (delegate) must not be production_safe, got {status}"
        )


def test_native_vs_delegate_execution_kind_metadata(loaded):
    """Verify backend_meta execution_kind for delegates vs native."""
    delegates = []
    natives = []

    for canon in OperatorRegistry.list_canonical():
        if "polars" not in OperatorRegistry.backends_for(canon):
            continue

        meta = (
            (OperatorRegistry._catalog.get(canon, {}) or {})
            .get("backend_meta", {})
            .get("polars", {})
        )
        execution_kind = meta.get("execution_kind", "")

        if canonical_polars_is_delegate(canon):
            delegates.append((canon, execution_kind))
            # Delegates should have delegate execution_kind
            if execution_kind and execution_kind not in {
                "polars_udf_pandas_delegate",
                "pandas_fallback",
                "pandas_materialization_fallback",
            }:
                pytest.fail(
                    f"{canon} is a delegate but has execution_kind={execution_kind}"
                )
        else:
            natives.append((canon, execution_kind))

    assert len(delegates) > 0, "Should have some delegates"
    assert len(natives) > 0, "Should have some native implementations"
