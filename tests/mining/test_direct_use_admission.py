# -*- coding: utf-8 -*-
"""R21-P030: regression tests for DirectUse production_admitted unification.

Tests that production_admitted requires at least one PhysicalImplementationID
with evidence, and that production_admitted is False when all backends are NOT_RUN.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Any

from mining.direct_use import (
    build_direct_use_operator,
    DirectUseStatus,
    DirectUseContext,
    get_direct_use_mining_operators,
)
from backend.contracts import (
    PhysicalImplementationID,
    PhysicalImplementationSpec,
    ExecutionKind,
    BackendKind,
)
from backend.operator_capability import (
    PhysicalInventoryAdmission,
    PhysicalInventoryRecord,
)


class TestProductionAdmittedRequiresPhysicalEvidence:
    """R21-P030: production_admitted must require physical evidence."""

    def test_production_admitted_requires_physical_evidence(self) -> None:
        """Verify production_admitted is False when no physical evidence exists."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock _has_physical_production_evidence to return False
        with patch("mining.direct_use._has_physical_production_evidence", return_value=False):
            row = build_direct_use_operator(canonical, catalog)
            # production_admitted should be False even if other conditions pass
            assert row.production_admitted is False

    def test_production_admitted_fails_when_all_backends_not_run(self) -> None:
        """Verify production_admitted is False when all backends are NOT_RUN."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock enumerate_physical_inventory to return empty (no evidence)
        with patch("mining.direct_use.enumerate_physical_inventory", return_value=[]):
            row = build_direct_use_operator(canonical, catalog)
            # production_admitted must be False when no physical inventory
            assert row.production_admitted is False

    def test_production_admitted_passes_with_evidence(self) -> None:
        """Verify production_admitted can be True when physical evidence exists."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Create a mock physical inventory record with evidence
        mock_spec = PhysicalImplementationSpec(
            canonical=canonical,
            backend="pandas_numpy",
            execution_kind=ExecutionKind.REFERENCE,
            implementation_source_hash="abc123",
            parameter_domain_hash="def456",
            semantic_contract_hash="ghi789",
            emitter_identity="test_emitter",
        )
        mock_record = PhysicalInventoryRecord(
            canonical=canonical,
            registry_slot="pandas_numpy",
            backend="pandas_numpy",
            implementation_id=mock_spec.physical_implementation_id,
            spec=mock_spec,
            admission=PhysicalInventoryAdmission(
                production_surface=True,
                policy_allows_production=True,
                evidence_production_safe=True,
                spec_complete=True,
                admitted=True,
                reasons=(),
            ),
        )

        # Mock to return our record
        with patch("mining.direct_use.enumerate_physical_inventory", return_value=[mock_record]):
            # Mock other dependencies to pass
            with patch("mining.direct_use.cost_contract_declared", return_value=True):
                with patch("mining.direct_use.source_status") as mock_source:
                    mock_source.return_value = MagicMock(missing=False)
                    with patch("mining.direct_use._is_production_denied", return_value=False):
                        row = build_direct_use_operator(canonical, catalog)
                        # If catalog has production_certified=True, this should pass
                        if row.production_certified:
                            assert row.production_admitted is True


class TestDirectlyUsableRequiresProductionAdmitted:
    """R21-P030: directly_usable must require production_admitted."""

    def test_directly_usable_requires_production_admitted(self) -> None:
        """Verify directly_usable is False when production_admitted is False."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Ensure production_admitted is False
        with patch("mining.direct_use._has_physical_production_evidence", return_value=False):
            row = build_direct_use_operator(canonical, catalog)
            # directly_usable = mining_visible AND production_admitted
            if row.mining_visible:
                assert row.directly_usable is False


class TestPhysicalInventoryAdmissionConsistency:
    """R21-P030: PhysicalInventoryAdmission must be consistent with DirectUse."""

    def test_physical_inventory_admission_consistency(self) -> None:
        """Verify PhysicalInventoryAdmission.admitted requires evidence_production_safe."""
        from cleaned_operators import load_all
        from backend.operator_capability import enumerate_physical_inventory

        load_all()
        records = list(enumerate_physical_inventory())

        # Every admitted record must have evidence_production_safe=True
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


class TestProductionAdmittedLogic:
    """R21-P030: test the unified production_admitted logic."""

    def test_production_admitted_all_conditions_must_pass(self) -> None:
        """Verify production_admitted requires all conditions to pass."""
        from cleaned_operators import load_all
        from cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Test with physical evidence False
        with patch("mining.direct_use._has_physical_production_evidence", return_value=False):
            row = build_direct_use_operator(canonical, catalog)
            assert row.production_admitted is False

        # Test with physical evidence True but cost_contract_declared False
        with patch("mining.direct_use._has_physical_production_evidence", return_value=True):
            with patch("mining.direct_use.cost_contract_declared", return_value=False):
                row = build_direct_use_operator(canonical, catalog)
                assert row.production_admitted is False

    def test_has_physical_production_evidence_returns_false_when_no_records(self) -> None:
        """Verify _has_physical_production_evidence returns False when no records."""
        from mining.direct_use import _has_physical_production_evidence

        with patch("mining.direct_use.enumerate_physical_inventory", return_value=[]):
            result = _has_physical_production_evidence("nonexistent_canonical")
            assert result is False

    def test_has_physical_production_evidence_returns_true_with_valid_record(self) -> None:
        """Verify _has_physical_production_evidence returns True with valid record."""
        from mining.direct_use import _has_physical_production_evidence

        mock_record = MagicMock()
        mock_record.canonical = "test_canonical"
        mock_record.implementation_id = MagicMock()
        mock_record.admission.spec_complete = True
        mock_record.admission.evidence_production_safe = True

        with patch("mining.direct_use.enumerate_physical_inventory", return_value=[mock_record]):
            result = _has_physical_production_evidence("test_canonical")
            assert result is True


class TestDirectUseMiningOperatorsAdmission:
    """R21-P030: verify get_direct_use_mining_operators respects admission."""

    def test_eligible_admission_requires_production_admitted(self) -> None:
        """Verify 'eligible' admission filters out non-production-admitted operators."""
        from cleaned_operators import load_all

        load_all()

        # Mock _has_physical_production_evidence to return False
        with patch("mining.direct_use._has_physical_production_evidence", return_value=False):
            operators = get_direct_use_mining_operators(
                context=DirectUseContext(market="ashare"),
                admission="eligible",
            )
            # All operators should have production_admitted=False
            for op in operators:
                assert op.production_admitted is False, (
                    f"Operator {op.canonical} is in eligible list "
                    f"but production_admitted is {op.production_admitted}"
                )
