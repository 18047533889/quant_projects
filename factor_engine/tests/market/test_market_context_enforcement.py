"""R21-P033: Regression tests for explicit market enum enforcement.

Production interfaces must require Market enum; market=None must hard fail.
"""
from __future__ import annotations

import pytest

from factor_engine.api.mining_integration import export_dsl_allowlist_json, validate_production_dsl
from factor_engine.mining.direct_use import DirectUseContext


class TestValidateProductionDslRequiresMarket:
    """validate_production_dsl must not accept market=None."""

    def test_validate_production_dsl_requires_market(self) -> None:
        with pytest.raises(TypeError, match="market"):
            validate_production_dsl("rank(ts_mean(col('close'), 20))")  # type: ignore[call-arg]

    def test_validate_production_dsl_rejects_none_explicit(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            validate_production_dsl("rank(ts_mean(col('close'), 20))", market=None)  # type: ignore[arg-type]


class TestExportDslAllowlistJsonRequiresMarket:
    """export_dsl_allowlist_json must not accept market=None."""

    def test_export_dsl_allowlist_json_requires_market(self) -> None:
        with pytest.raises(TypeError, match="market"):
            export_dsl_allowlist_json()  # type: ignore[call-arg]

    def test_export_dsl_allowlist_json_rejects_none_explicit(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            export_dsl_allowlist_json(market=None)  # type: ignore[arg-type]


class TestDirectUseContextRequiresMarket:
    """DirectUseContext must not accept market=None."""

    def test_direct_use_context_requires_market(self) -> None:
        with pytest.raises(TypeError, match="market"):
            DirectUseContext()  # type: ignore[call-arg]

    def test_direct_use_context_rejects_none_explicit(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            DirectUseContext(market=None)  # type: ignore[arg-type]

    def test_direct_use_context_accepts_explicit_market(self) -> None:
        from factor_engine.market.context import Market

        ctx = DirectUseContext(market=Market.ASHARE)
        assert ctx.market == Market.ASHARE
