"""Cold-start seeds for multi-bar and adaptive candlestick operators."""
from __future__ import annotations
from .model import ColdStartFactor

_NAMES=(
"cdl_dragonfly_doji","cdl_gravestone_doji","cdl_hanging_man","cdl_harami","cdl_harami_cross","cdl_piercing","cdl_dark_cloud_cover","cdl_morning_star","cdl_evening_star","cdl_three_white_soldiers","cdl_three_black_crows","cdl_tweezer_top","cdl_tweezer_bottom",
)
def candle_pattern_seeds(market:str):
    if market not in {"ashare","us"}:raise ValueError("market must be ashare or us")
    prefix="cn" if market=="ashare" else "us";rows=[]
    for i,name in enumerate(_NAMES,start=1):
        bars=3 if name in {"cdl_morning_star","cdl_evening_star","cdl_three_white_soldiers","cdl_three_black_crows"} else (1 if name in {"cdl_dragonfly_doji","cdl_gravestone_doji","cdl_hanging_man"} else 2)
        rows.append(ColdStartFactor(factor_id=f"{prefix}_cdl_ext_{i:03d}",market=market,surface="extended",formula=f"{name}(open,high,low,close)",family="candlestick_pattern",subfamily=name.removeprefix("cdl_"),horizon=bars,complexity="basic" if bars<=2 else "moderate",availability_tier="core",rationale="Raw causal candlestick geometry; combine with trend/location/volume context downstream.",direction_hint="unknown",metadata={"causal":True,"frequency":"1d","bars":bars,"production_seed":True}))
    rows.append(ColdStartFactor(
        factor_id=f"{prefix}_cdl_adaptive_engine",market=market,surface="extended",
        formula="candlestick_pattern(open,high,low,close,'high_wave',10,10,0.3)",
        family="candlestick_pattern",subfamily="adaptive_engine",horizon=10,complexity="moderate",availability_tier="core",
        rationale="Adaptive candlestick semantic engine with explicit body/shadow history windows.",direction_hint="unknown",
        metadata={"causal":True,"frequency":"1d","production_seed":True,"parameterized":True},
    ))
    return tuple(rows)
