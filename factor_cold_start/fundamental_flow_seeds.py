"""Cold-start coverage for explicit quarterly/cumulative filing semantics."""
from __future__ import annotations

from .model import ColdStartFactor

_FORMULAS = {
    "fin_ttm_quarterly": "fin_ttm_quarterly(fundamental_x,period_id,4)",
    "fin_quarter_from_cumulative": (
        "fin_quarter_from_cumulative(fundamental_x,period_id,fiscal_quarter)"
    ),
    "fin_ttm_cumulative": (
        "fin_ttm_cumulative(fundamental_x,period_id,fiscal_quarter,4)"
    ),
}


def fundamental_flow_seeds(market: str) -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    return tuple(
        ColdStartFactor(
            factor_id=f"{market}_fundamental_flow_{name}",
            market=market,
            surface="extended",
            formula=formula,
            family="fundamental_transform",
            subfamily="flow_semantics",
            horizon=None,
            complexity="moderate",
            availability_tier="fundamental",
            rationale=f"Explicit PIT filing-flow transform {name}.",
            direction_hint="unknown",
            metadata={
                "source": "fundamental_flow_seeds",
                "operator": name,
                "requires_period_id": True,
                "requires_fiscal_quarter": "cumulative" in name,
            },
        )
        for name, formula in _FORMULAS.items()
    )
