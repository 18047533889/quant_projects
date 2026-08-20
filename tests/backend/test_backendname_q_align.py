# -*- coding: utf-8 -*-
"""Regression tests for R21-BACKENDNAME-Q-ALIGN.

Covers the BackendName/BackendFamily/PhysicalBackend alignment for q_kdb in
backend/operator_capability.py:
1. _REGISTRY_TO_CAPABILITY maps the "q_kdb" registry slot to the "q_kdb"
   BackendName (and all enum families agree on the literal).
2. backend_status / OperatorCapabilitySummary expose a q_kdb column whose
   status is derived from the q physical-implementation evidence authority
   (fail-closed to "unsupported").
3. get_best_backend(prefer="q_kdb") routes through the q evidence authority:
   eligible status is required (fail-closed rejection otherwise) and a q
   operator is only returned when a q slot is actually registered.
"""
from __future__ import annotations

import typing

import pytest

from backend.contracts import BackendFamily, BackendKind
from backend.operator_capability import (
    _REGISTRY_TO_CAPABILITY,
    OperatorCapabilitySummary,
    UnsupportedOperatorBackendError,
    backend_status,
    capability_for,
    get_best_backend,
    summarize_operator,
)


def test_registry_to_capability_maps_q_kdb_and_family_literals_agree():
    """_REGISTRY_TO_CAPABILITY must map the q_kdb slot; enums share the literal."""
    assert _REGISTRY_TO_CAPABILITY.get("q_kdb") == "q_kdb"
    # The mapped name must be a member of the BackendName Literal.
    assert "q_kdb" in typing.get_args(
        __import__("backend.operator_capability", fromlist=["BackendName"]).BackendName
    )
    # BackendFamily (contracts) and BackendKind (contracts) agree on the literal.
    assert BackendFamily.Q_KDB.value == "q_kdb"
    assert BackendKind.Q_KDB.value == "q_kdb"
    # planner.PhysicalBackend agrees too (no import cycle: planner imports backend).
    from backend.operator_capability import _OperatorBackendRegistryStub
    from planner.backend_region import PhysicalBackend, normalize_backend_name

    assert PhysicalBackend.Q_KDB.value == "q_kdb"
    for alias in ("q", "q_kdb", "kdb"):
        assert normalize_backend_name(alias) is PhysicalBackend.Q_KDB

    # get_best_backend respects an injectable registry's canonical resolver.
    registry = _OperatorBackendRegistryStub(
        backends_for={"ts_mean": ["q_kdb"]},
        get_impl=lambda canonical, backend: f"{canonical}:{backend}",
        resolve_canonical=lambda name: name.upper(),
    )
    # Production must reject q_kdb when evidence authority does not certify it.
    with pytest.raises(UnsupportedOperatorBackendError):
        get_best_backend(
            "ts_mean",
            mode="production",
            prefer="q_kdb",
            registry=registry,  # type: ignore[arg-type]
        )

    op, backend = get_best_backend(
        "ts_mean",
        mode="research",
        prefer="q_kdb",
        allow_unverified_backend=True,
        registry=registry,  # type: ignore[arg-type]
    )
    assert backend == "q_kdb"
    assert op == "TS_MEAN:q_kdb"


def test_summary_and_status_expose_q_kdb_column_from_evidence_authority():
    """OperatorCapabilitySummary must carry a q_kdb column backed by evidence."""
    from backend.contracts import BackendKind
    from backend.operator_capability import BackendCapabilityRegistry

    fields = OperatorCapabilitySummary.__dataclass_fields__
    assert "q_kdb" in fields, "OperatorCapabilitySummary missing q_kdb column"

    summary = summarize_operator("ts_mean")
    # ts_mean is a declared q target with a lowering => at least implemented
    # (never production_safe until the q registry certifies it).
    assert summary.q_kdb in {"implemented", "parity_verified", "production_safe"}
    assert summary.q_kdb == backend_status("ts_mean", "q_kdb")

    # garch is not a declared q target => fail-closed unsupported.
    assert summarize_operator("garch").q_kdb == "unsupported"
    assert backend_status("garch", "q_kdb") == "unsupported"

    # capability_for keeps BackendKind.Q_KDB identity for the q backend.
    cap = capability_for("ts_mean", "q_kdb")
    assert cap.backend == BackendKind.Q_KDB

    # BackendCapabilityRegistry.query() must accept both string and enum forms.
    query_str = BackendCapabilityRegistry.query("ts_mean", "q_kdb")
    query_enum = BackendCapabilityRegistry.query("ts_mean", BackendKind.Q_KDB)
    assert (query_str.supported, query_str.production_safe) == (
        query_enum.supported,
        query_enum.production_safe,
    )


def test_get_best_backend_routes_q_kdb_prefer_fail_closed(monkeypatch):
    """prefer='q_kdb' routes via the q evidence authority, fail-closed."""
    import backend.operator_capability as oc

    class _FakeRegistry:
        def backends_for(self, canonical):
            return ["pandas_numpy", "q_kdb"]

        def get(self, canonical, backend):
            if backend == "q_kdb":
                return object()
            return None

    monkeypatch.setattr(
        "cleaned_operators.registry.OperatorRegistry.resolve_canonical",
        staticmethod(lambda name: name),
    )
    monkeypatch.setattr(
        oc, "resolve_canonical", lambda name: name, raising=False
    )
    monkeypatch.setattr(
        "backend.operator_capability._pandas_status",
        lambda canon: "unsupported",
        raising=False,
    )

    # 1) Eligible q status + registered q slot => routed to q_kdb.
    monkeypatch.setattr(
        "backend.operator_capability._q_status",
        lambda canon: "production_safe",
        raising=False,
    )
    op, backend = get_best_backend(
        "ts_mean",
        mode="production",
        prefer="q_kdb",
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )
    assert backend == "q_kdb"
    assert op is not None

    # 2) Ineligible q status (evidence denies) => explicit fail-closed error.
    monkeypatch.setattr(
        "backend.operator_capability._q_status",
        lambda canon: "implemented",
        raising=False,
    )
    with pytest.raises(UnsupportedOperatorBackendError):
        get_best_backend(
            "ts_mean",
            mode="production",
            prefer="q_kdb",
            registry=_FakeRegistry(),  # type: ignore[arg-type]
        )

    # 3) Eligible status but no registered q slot => fail-closed error.
    class _NoQSlotRegistry(_FakeRegistry):
        def get(self, canonical, backend):
            return None

    monkeypatch.setattr(
        "backend.operator_capability._q_status",
        lambda canon: "production_safe",
        raising=False,
    )
    with pytest.raises(UnsupportedOperatorBackendError):
        get_best_backend(
            "ts_mean",
            mode="production",
            prefer="q_kdb",
            registry=_NoQSlotRegistry(),  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
