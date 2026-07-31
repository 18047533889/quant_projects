"""Cold-start coverage for generic analyst-expectation operators."""
from __future__ import annotations
from .model import ColdStartFactor
_FORMULAS={
"fin_surprise":"fin_surprise(actual,expected,scale_base)",
"fin_surprise_zscore":"fin_surprise_zscore(actual,expected,scale_base,252)",
"fin_expectation_revision":"fin_expectation_revision(expected,target_period_id)",
"fin_expectation_revision_pct":"fin_expectation_revision_pct(expected,target_period_id)",
"fin_expectation_revision_speed":"fin_expectation_revision_speed(expected,target_period_id,60)",
"fin_expectation_dispersion":"fin_expectation_dispersion(expected_std,expected_mean)",
"fin_actual_expectation_divergence":"fin_actual_expectation_divergence(actual,expected,scale_base)",
"fin_beat_streak":"fin_beat_streak(actual,expected,period_id,8)",
"fin_miss_streak":"fin_miss_streak(actual,expected,period_id,8)",
}
def expectation_seeds(market:str):
    return tuple(ColdStartFactor(
        factor_id=f"{market}_expectation_{name}",market=market,surface="extended",formula=formula,
        family="analyst_expectation",subfamily="generic_expectation",horizon=None,complexity="moderate",availability_tier="analyst",
        rationale=f"Generic PIT analyst-expectation transform {name}.",direction_hint="unknown",
        metadata={"source":"expectation_seeds","operator":name,"requires_analyst_data":True},
    ) for name,formula in _FORMULAS.items())
