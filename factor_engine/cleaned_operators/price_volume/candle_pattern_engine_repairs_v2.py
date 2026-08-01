# -*- coding: utf-8 -*-
"""Complete two adaptive pattern cases used by the recipe catalog."""
from __future__ import annotations
import numpy as np
import pandas as pd
from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.price_volume.candle_pattern_engine_v2 import _engine as _base_engine,_pi


def _engine(o,h,l,c,pattern,body_window,shadow_window,penetration):
    p=str(pattern).strip().lower().removeprefix("cdl_")
    if p not in {"2_crows","evening_doji_star"}:
        return _base_engine(o,h,l,c,pattern,body_window,shadow_window,penetration)
    bw=_pi(body_window,"body_window",2);body=(c-o).abs();bavg=body.shift(1).rolling(bw,min_periods=bw).mean();tol=0.1*(h-l).shift(1).rolling(_pi(shadow_window,"shadow_window",2),min_periods=_pi(shadow_window,"shadow_window",2)).mean()
    po,pc,ph,pl=o.shift(1),c.shift(1),h.shift(1),l.shift(1);o2,c2,h2,l2=o.shift(2),c.shift(2),h.shift(2),l.shift(2);bull2=c2>o2;pbear=pc<po;bear=c<o;doji1=(pc-po).abs()<=0.1*bavg.shift(1)
    if p=="2_crows":
        mask=bull2&pbear&bear&(pl>h2)&(o>po)&(c<c2)&(c>o2);return pd.DataFrame(np.where(mask,-1.0,0.0),index=o.index,columns=o.columns)
    pen=float(penetration);mid=o2+(c2-o2)*(1.0-pen);mask=bull2&doji1&(pl>h2)&bear&(c<mid);return pd.DataFrame(np.where(mask,-1.0,0.0),index=o.index,columns=o.columns)

class CandlestickPatternEngineV2(SeriesOperator):
    metadata=OperatorMetadata(name="candlestick_pattern",category="candle_pattern",description="Adaptive bounded Japanese-candlestick pattern engine.",param_names=["open","high","low","close","pattern","body_window","shadow_window","penetration"],return_type="series",tags=["pit_safe","causal","bounded_history","production_repair"])
    def _calculate_series(self,open,high,low,close,pattern,body_window=10,shadow_window=10,penetration=0.3,**kwargs):return _engine(open,high,low,close,pattern,body_window,shadow_window,penetration)
register_operator(name="candlestick_pattern",category="candle_pattern",business_category="technical_extension",canonical="candlestick_pattern",source="candle_pattern_engine_repairs_v2",backend="pandas_numpy",status="production")(CandlestickPatternEngineV2)
