"""Stateful, checkpointable past-only correlation regime detector."""
from dataclasses import dataclass,field
from typing import Optional,List
import numpy as np
import pandas as pd
from factor_preprocess.regime.detector import RegimeState

@dataclass
class CausalRegimeDetector:
    window:int; n_regimes:int=2; percentiles:Optional[List[float]]=None; min_periods:Optional[int]=None; time_col:str="date"; value_cols:Optional[List[str]]=None
    _fitted:bool=field(default=False,init=False); _boundaries:np.ndarray=field(default=None,init=False); _value_cols:List[str]=field(default=None,init=False); _history:pd.DataFrame=field(default=None,init=False); _fit_end:object=field(default=None,init=False); _last_label:Optional[int]=field(default=None,init=False)
    def __post_init__(self):
        if isinstance(self.window,bool) or self.window<2: raise ValueError("window must be >=2")
        if self.n_regimes<2: raise ValueError("n_regimes must be >=2")
        self.min_periods=self.window if self.min_periods is None else self.min_periods
        if not 2<=self.min_periods<=self.window: raise ValueError("min_periods must be in [2, window]")
        self.percentiles=self.percentiles or [i/self.n_regimes for i in range(1,self.n_regimes)]
        if len(self.percentiles)!=self.n_regimes-1 or any(not 0<p<1 for p in self.percentiles): raise ValueError("invalid percentiles")
    def _cols(self,df):
        cols=list(self.value_cols) if self.value_cols is not None else [c for c in df if c!=self.time_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(cols)<2 or any(c not in df for c in cols): raise ValueError("at least two valid value columns required")
        return cols
    def fit(self,train):
        if len(train)==0 or not train[self.time_col].is_monotonic_increasing or train[self.time_col].duplicated().any(): raise ValueError("train times must be nonempty, sorted and unique")
        cols=self._cols(train); corr=_rolling(train[cols].to_numpy(float),self.window,self.min_periods); finite=corr[np.isfinite(corr)]
        if len(finite)<2: raise ValueError("insufficient usable correlation points")
        boundaries=np.quantile(finite,self.percentiles)
        if not np.all(np.diff(boundaries)>0): raise ValueError("degenerate regime boundaries")
        self._boundaries=np.asarray(boundaries); self._value_cols=cols; self._history=train[[self.time_col]+cols].tail(self.window).copy(); self._fit_end=train[self.time_col].iloc[-1]; self._last_label=None; self._fitted=True; return self
    def detect(self,data,*,allow_historical_replay=False):
        if not self._fitted: raise RuntimeError("detect requires fit first")
        if len(data)==0: return RegimeState(pd.Series(dtype=float,index=data.index),pd.Series(dtype=float,index=data.index),pd.Series(dtype=bool,index=data.index))
        if not data[self.time_col].is_monotonic_increasing or data[self.time_col].duplicated().any(): raise ValueError("detect times must be sorted and unique")
        replay=data[self.time_col].iloc[0]<=self._fit_end
        if replay and not allow_historical_replay: raise ValueError("inference at/before fit_end requires explicit historical replay")
        if not replay and data[self.time_col].iloc[0]<=self._history[self.time_col].iloc[-1]: raise ValueError("inference must be strictly after checkpoint history")
        joined=data[[self.time_col]+self._value_cols].copy() if replay else pd.concat([self._history,data[[self.time_col]+self._value_cols]],ignore_index=True)
        corr=_rolling(joined[self._value_cols].to_numpy(float),self.window,self.min_periods)[-len(data):]
        labels=np.full(len(data),np.nan); good=np.isfinite(corr); labels[good]=np.digitize(corr[good],self._boundaries); strength=np.full(len(data),np.nan); trans=np.zeros(len(data),bool); prev=self._last_label
        for i in range(len(data)):
            if good[i]:
                label=int(labels[i]); strength[i]=_strength(corr[i],label,self._boundaries); trans[i]=prev is not None and prev!=label; prev=label
        if not replay:
            self._last_label=prev; self._history=joined.tail(self.window).copy()
        return RegimeState(pd.Series(labels,index=data.index),pd.Series(strength,index=data.index),pd.Series(trans,index=data.index))
    def checkpoint(self):
        if not self._fitted: raise RuntimeError("not fitted")
        return {"version":1,"window":self.window,"n_regimes":self.n_regimes,"percentiles":list(self.percentiles),"min_periods":self.min_periods,"time_col":self.time_col,"value_cols":list(self._value_cols),"boundaries":self._boundaries.tolist(),"fit_end":self._fit_end,"history":self._history.to_dict("list"),"last_label":self._last_label}
    @classmethod
    def from_checkpoint(cls,p):
        if p.get("version")!=1: raise ValueError("unsupported checkpoint")
        o=cls(p["window"],p["n_regimes"],p["percentiles"],p["min_periods"],p["time_col"],p["value_cols"]); o._boundaries=np.array(p["boundaries"],float); o._value_cols=list(p["value_cols"]); o._fit_end=p["fit_end"]; o._history=pd.DataFrame(p["history"]); o._last_label=p["last_label"]; o._fitted=True; return o

def _rolling(a,window,min_periods):
    out=np.full(len(a),np.nan)
    for i in range(len(a)):
        z=a[max(0,i-window):i]; vals=[]
        for x in range(a.shape[1]):
            for y in range(x+1,a.shape[1]):
                ok=np.isfinite(z[:,x])&np.isfinite(z[:,y])
                if ok.sum()>=min_periods and np.std(z[ok,x])>0 and np.std(z[ok,y])>0: vals.append(np.corrcoef(z[ok,x],z[ok,y])[0,1])
        if vals: out[i]=np.mean(vals)
    return out
def _strength(v,label,b):
    lo=-1. if label==0 else b[label-1]; hi=1. if label==len(b) else b[label]; width=hi-lo
    if width<=0: return 0.
    if label==0: raw=(hi-v)/width
    elif label==len(b): raw=(v-lo)/width
    else: raw=1.-abs(v-(lo+hi)/2)/(width/2)
    return float(np.clip(raw,0,1))
