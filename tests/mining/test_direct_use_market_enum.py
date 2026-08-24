# -*- coding: utf-8 -*-
"""R21-MARKET-ENUM-FIX: Regression tests for Market enum enforcement.

DirectUseContext.market must be a Market enum, not an arbitrary string.
Hard fail on None or arbitrary string.
"""
from __future__ import annotations

import pytest

from factor_engine.market.context import Market
from factor_engine.mining.direct_use import DirectUseContext


class TestDirectUseContextMarketEnum:
    """DirectUseContext.market must accept only Market enum values."""

    def test_market_enum_ashare(self) -> None:
        """DirectUseContext accepts Market.ASHARE."""
        ctx = DirectUseContext(market=Market.ASHARE)
        assert ctx.market == Market.ASHARE
        assert ctx.market.value == "ashare"
        # Verify it works with string comparisons (backward compatibility)
        assert ctx.market == "ashare"

    def test_market_enum_us(self) -> None:
        """DirectUseContext accepts Market.US."""
        ctx = DirectUseContext(market=Market.US)
        assert ctx.market == Market.US
        assert ctx.market.value == "us"
        # Verify it works with string comparisons (backward compatibility)
        assert ctx.market == "us"

    def test_market_enum_in_tuple(self) -> None:
        """Market enum works with tuple membership checks."""
        ctx = DirectUseContext(market=Market.ASHARE)
        # This pattern is used in get_direct_use_mining_operators
        assert ctx.market in ("ashare", "us")
        assert ctx.market in (Market.ASHARE, Market.US)

    def test_market_enum_not_none(self) -> None:
        """DirectUseContext rejects None market."""
        with pytest.raises((TypeError, ValueError)):
            DirectUseContext(market=None)  # type: ignore[arg-type]

    def test_market_enum_not_empty_string(self) -> None:
        """DirectUseContext rejects empty string market."""
        with pytest.raises(ValueError, match="market is required"):
            DirectUseContext(market="")  # type: ignore[arg-type]

    def test_market_enum_rejects_arbitrary_string(self) -> None:
        """DirectUseContext rejects arbitrary string market."""
        # Python's type system should catch this at runtime with str enum
        with pytest.raises((TypeError, ValueError)):
            DirectUseContext(market="invalid_market")  # type: ignore[arg-type]

    def test_market_enum_comparison_with_string(self) -> None:
        """Market enum values can be compared with strings."""
        ctx = DirectUseContext(market=Market.ASHARE)
        # This is the pattern used in capability_resolver.py
        assert ctx.market == "ashare"
        assert "ashare" == ctx.market
