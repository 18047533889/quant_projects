"""Cold-start seed for liquidity names that differ from physical data fields."""
from __future__ import annotations

from .model import ColdStartFactor


def liquidity_naming_seeds(market: str) -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    return (
        ColdStartFactor(
            factor_id=f"{market}_liquidity_ts_average_volume",
            market=market,
            surface="extended",
            formula="ts_average_volume(volume,20)",
            family="liquidity",
            subfamily="activity",
            horizon=20,
            complexity="basic",
            availability_tier="core",
            rationale=(
                "Rolling average volume operator uses a collision-free ts_ name; "
                "average_volume remains reserved for the authoritative data field."
            ),
            direction_hint="unknown",
            metadata={
                "source": "liquidity_naming_seeds",
                "operator": "ts_average_volume",
                "field_collision_resolved": True,
            },
        ),
    )
