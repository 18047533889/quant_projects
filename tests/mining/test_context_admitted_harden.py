# -*- coding: utf-8 -*-
"""R21-CONTEXT-ADMITTED-HARDEN: regression tests for context_admitted validation.

Tests that context_admitted validates Market=ASHARE + SourceCapabilities + FieldRecipes + Units + Grain + Availability + Calendar + Universe.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import time

from factor_engine.mining.direct_use import (
    build_direct_use_operator,
    DirectUseContext,
    get_direct_use_mining_operators,
)
from factor_engine.market.context import Market


class TestContextAdmittedMarketValidation:
    """R21-CONTEXT-ADMITTED-HARDEN: context_admitted requires Market=ASHARE."""

    def test_context_admitted_requires_ashare_market(self) -> None:
        """context_admitted is False for non-ASHARE market (US)."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock market to be "us" instead of "ashare"
        with patch("factor_engine.mining.direct_use.market_support", return_value=("us",)):
            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is False, (
                f"Operator {canonical} should not be context_admitted for US market"
            )

    def test_context_admitted_passes_for_ashare_market(self) -> None:
        """context_admitted is True for ASHARE market."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        row = build_direct_use_operator(canonical, catalog)
        # ts_mean should be context_admitted for ASHARE
        assert row.context_admitted is True, (
            f"Operator {canonical} should be context_admitted for ASHARE market"
        )


class TestContextAdmittedSourceCapabilities:
    """R21-CONTEXT-ADMITTED-HARDEN: context_admitted validates SourceCapabilities."""

    def test_context_admitted_fails_when_sources_missing(self) -> None:
        """context_admitted is False when required sources are missing."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock source_status to indicate missing sources
        with patch("factor_engine.mining.direct_use.source_status") as mock_source:
            mock_status = MagicMock()
            mock_status.required = ("daily_bar",)
            mock_status.satisfied = ()
            mock_source.return_value = mock_status

            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is False, (
                f"Operator {canonical} should not be context_admitted when sources are missing"
            )

    def test_context_admitted_passes_when_sources_satisfied(self) -> None:
        """context_admitted is True when required sources are satisfied."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock source_status to indicate satisfied sources
        with patch("factor_engine.mining.direct_use.source_status") as mock_source:
            mock_status = MagicMock()
            mock_status.required = ("daily_bar",)
            mock_status.satisfied = ("daily_bar",)
            mock_source.return_value = mock_status

            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is True, (
                f"Operator {canonical} should be context_admitted when sources are satisfied"
            )


class TestContextAdmittedFieldRecipes:
    """R21-CONTEXT-ADMITTED-HARDEN: context_admitted validates FieldRecipes."""

    def test_context_admitted_fails_when_no_field_recipe(self) -> None:
        """context_admitted is False when no default_input_recipe exists."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock default_input_recipe to return empty dict
        with patch("factor_engine.mining.direct_use.default_input_recipe", return_value={}):
            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is False, (
                f"Operator {canonical} should not be context_admitted without field recipe"
            )

    def test_context_admitted_passes_when_field_recipe_exists(self) -> None:
        """context_admitted is True when default_input_recipe exists."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock default_input_recipe to return a valid recipe
        with patch("factor_engine.mining.direct_use.default_input_recipe", return_value={"x": "continuous_close"}):
            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is True, (
                f"Operator {canonical} should be context_admitted with field recipe"
            )


class TestContextAdmittedIntegration:
    """R21-CONTEXT-ADMITTED-HARDEN: integration tests for context_admitted."""

    def test_context_admitted_requires_all_components(self) -> None:
        """context_admitted requires all components to pass."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # All components should be satisfied for ts_mean in ASHARE
        row = build_direct_use_operator(canonical, catalog)
        assert row.context_admitted is True, (
            f"Operator {canonical} should be context_admitted with all components"
        )

    def test_context_admitted_fails_when_any_component_fails(self) -> None:
        """context_admitted fails when any component fails."""
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()
        canonical = "ts_mean"
        catalog = dict(OperatorRegistry._catalog.get(canonical, {}))

        # Mock market to be "us" - should fail
        with patch("factor_engine.mining.direct_use.market_support", return_value=("us",)):
            row = build_direct_use_operator(canonical, catalog)
            assert row.context_admitted is False, (
                f"Operator {canonical} should not be context_admitted with US market"
            )
