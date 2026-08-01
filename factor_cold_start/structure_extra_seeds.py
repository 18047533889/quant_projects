"""Supplemental cold-start seeds for advanced bounded chart patterns."""
from __future__ import annotations
from .model import ColdStartFactor

_FORMULAS={
"pattern_triple_top":"pattern_triple_top(high,low,3,3,120,0.03,0.05,5,60)",
"pattern_triple_bottom":"pattern_triple_bottom(high,low,3,3,120,0.03,0.05,5,60)",
"pattern_123_bull":"pattern_123_bull(high,low,3,3,100,0.05)",
"pattern_123_bear":"pattern_123_bear(high,low,3,3,100,0.05)",
"pattern_rounding_bottom":"pattern_rounding_bottom(close,60,0.02)",
"pattern_rounding_top":"pattern_rounding_top(close,60,0.02)",
"pattern_cup":"pattern_cup(close,80,0.10,0.05,0.02)",
"pattern_cup_handle":"pattern_cup_handle(close,high,low,80,15,0.10,0.05,0.02,0.15)",
"pattern_bull_pennant":"pattern_bull_pennant(close,high,low,volume,20,12,0.08,0.12,0.0)",
"pattern_bear_pennant":"pattern_bear_pennant(close,high,low,volume,20,12,0.08,0.12,0.0)",
"pattern_breakout_retest":"pattern_breakout_retest(close,60,10,0.02)",
"pattern_breakdown_retest":"pattern_breakdown_retest(close,60,10,0.02)",
}
def structure_extra_seeds(market:str):
    return tuple(ColdStartFactor(
        factor_id=f"{market}_structure_extra_{name}",market=market,surface="extended",formula=formula,
        family="technical_structure",subfamily="advanced_pattern",horizon=120,complexity="composite",availability_tier="core",
        rationale=f"Bounded causal chart-pattern seed for {name}.",direction_hint="unknown",
        metadata={"source":"structure_extra_seeds","operator":name,"causal":True},
    ) for name,formula in _FORMULAS.items())
