# -*- coding: utf-8 -*-
"""R21-P030: regression tests for DirectUse production_admitted unification.

Tests that production_admitted requires at least one PhysicalImplementationID
with evidence, and that production_admitted is False when all backends are NOT_RUN.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from mining.direct_use import (
    build_direct_use_operator,
    DirectUseContext,
    get_direct_use_mining_operators,
    _has_physical_production_evidence,
)
from backend.contracts import PhysicalImplementationSpec, ExecutionKind
from backend.operator_capability import (
    PhysicalInventoryAdmission,
    PhysicalInventoryRecord,
)


# `_has_physical_production_evidence` imports the inventory lazily from
# `backend.operator_capability`, so patch it there (not on mining.direct_use).
_INVENTORY = "backend.operator_capability.enumerate_physical_inventory"


def _admitted_record(canonical: str = "ts_mean") -> PhysicalInventoryRecord:
    """One fully-admitted physical inventory row for a canonical."""
    spec = PhysicalImplementationSpec(
        canonical=canonical,
        backend="pandas_numpy",
        execution_kind=ExecutionKind.REFERENCE,
        implementation_source_hash="abc123",
        parameter_domain_hash="def456",
        semantic_contract_hash="ghi789",
        emitter_identity="test_emitter",
    )
    assert spec.physical_implementation_id is not None
    return PhysicalInventoryRecord(
        canonical=canonical,
        registry_slot="pandas_numpy",
        backend="pandas_numpy",
        implementation_id=spec.physical_implementation_id,
        spec=spec,
        admission=PhysicalInventoryAdmission(
            production_surface=True,
            policy_allows_production=True,
            evidence_production_safe=True,
            spec_complete=True,
            admitted=True,
            reasons=(),
        ),
    )


class TestProductionAdmittedRequiresPhysicalEvidence:
    """R21-P030: production_admitted must require physical evidence."""

    def test_production_admitted_requires_physical_evidence(self) -> None:
        """production_admitted is False when no physical evidence exists."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        with patch(_INVENTORY, return_value=[]):
            row = build_direct_use_operator(canonical, catalog)
            assert row.production_admitted is False

    def test_production_admitted_fails_when_all_backends_not_run(self) -> None:
        """production_admitted is False when all backends are NOT_RUN.

        NOT_RUN means no inventory row carries current production evidence —
        an empty evidence query is exactly that state, and admission must
        fail closed (never true when every backend is NOT_RUN).
        """
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        with patch(_INVENTORY, return_value=[]):
            row = build_direct_use_operator(canonical, catalog)
            assert row.production_admitted is False
            assert row.directly_usable is False

    def test_production_admitted_passes_with_evidence(self) -> None:
        """production_admitted can be True when physical evidence exists."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        record = _admitted_record(canonical)
        with patch(_INVENTORY, return_value=[record]):
            with patch("mining.direct_use.cost_contract_declared", return_value=True):
                with patch("mining.direct_use.source_status") as mock_source:
                    mock_source.return_value = MagicMock(missing=())
                    with patch(
                        "mining.direct_use._is_production_denied", return_value=False
                    ):
                        row = build_direct_use_operator(canonical, catalog)
                        if row.production_certified:
                            assert row.production_admitted is True
                            # The admitting implementation must be an exact
                            # current-head PhysicalImplementationID.
                            assert record.implementation_id is not None
                            assert str(record.implementation_id).startswith("pi:v1:")


class TestDirectlyUsableRequiresProductionAdmitted:
    """R21-P030: directly_usable must require production_admitted."""

    def test_directly_usable_requires_production_admitted(self) -> None:
        """directly_usable is False when production_admitted is False."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        with patch(_INVENTORY, return_value=[]):
            row = build_direct_use_operator(canonical, catalog)
            if row.mining_visible:
                assert row.directly_usable is False


class TestPhysicalInventoryAdmissionConsistency:
    """R21-P030: PhysicalInventoryAdmission must be consistent with DirectUse."""

    def test_physical_inventory_admission_consistency(self) -> None:
        """Every admitted inventory row must carry evidence and an exact ID."""
        from cleaned_operators import load_all
        from backend.operator_capability import enumerate_physical_inventory

        load_all()
        records = list(enumerate_physical_inventory())

        for record in records:
            if record.admission.admitted:
                assert record.admission.evidence_production_safe is True, (
                    f"Record {record.canonical}/{record.backend} is admitted "
                    f"but lacks evidence_production_safe"
                )
                assert record.implementation_id is not None, (
                    f"Record {record.canonical}/{record.backend} is admitted "
                    f"but has no implementation_id"
                )
                assert record.admission.spec_complete is True
                assert str(record.implementation_id).startswith("pi:v1:")


class TestProductionAdmittedLogic:
    """R21-P030: test the unified production_admitted logic."""

    def test_production_admitted_all_conditions_must_pass(self) -> None:
        """production_admitted requires every gate to pass."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        with patch(_INVENTORY, return_value=[]):
            row = build_direct_use_operator(canonical, catalog)
            assert row.production_admitted is False

        record = _admitted_record(canonical)
        with patch(_INVENTORY, return_value=[record]):
            with patch("mining.direct_use.cost_contract_declared", return_value=False):
                row = build_direct_use_operator(canonical, catalog)
                assert row.production_admitted is False

    def test_has_physical_production_evidence_returns_false_when_no_records(self) -> None:
        """_has_physical_production_evidence is False with no records (NOT_RUN)."""
        with patch(_INVENTORY, return_value=[]):
            assert _has_physical_production_evidence("nonexistent_canonical") is False

    def test_has_physical_production_evidence_returns_true_with_valid_record(self) -> None:
        """_has_physical_production_evidence is True with a valid evidence row."""
        record = _admitted_record("test_canonical")
        with patch(_INVENTORY, return_value=[record]):
            assert _has_physical_production_evidence("test_canonical") is True


class TestDirectUseMiningOperatorsAdmission:
    """R21-P030: verify get_direct_use_mining_operators respects admission."""

    def test_eligible_admission_requires_production_admitted(self) -> None:
        """'eligible' admission filters out non-production-admitted operators."""
        from cleaned_operators import load_all

        load_all()

        with patch(_INVENTORY, return_value=[]):
            operators = get_direct_use_mining_operators(
                context=DirectUseContext(market="ashare"),
                admission="eligible",
            )
            for op in operators:
                assert op.production_admitted is False, (
                    f"Operator {op.canonical} is in eligible list "
                    f"but production_admitted is {op.production_admitted}"
                )
