"""Independent, bounded formula oracle for representative R30 operators."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

OUT = Path("evidence/r30_operator_campaign_windows/oracle_results.json")
WINDOW = 5

def rolling(values, fn):
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            out[row, col] = fn(values[max(0, row-WINDOW+1):row+1, col])
    return out

def conditional(values, conditions, kind):
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            start = max(0, row-WINDOW+1)
            x, c = values[start:row+1, col], conditions[start:row+1, col]
            selected = x[np.isfinite(x) & np.isfinite(c) & (c == 1.0)]
            if kind == "last":
                if selected.size: out[row, col] = selected[-1]
            elif kind == "std":
                if selected.size >= 2: out[row, col] = selected.std(ddof=1)
            elif selected.size:
                out[row, col] = {"sum":np.sum,"mean":np.mean,"max":np.max,"min":np.min}[kind](selected)
    return out

def count_if(conditions):
    def one(chunk):
        valid = np.isfinite(chunk)
        return float(np.sum(chunk[valid] == 1.0)) if valid.any() else np.nan
    return rolling(conditions, one)

def tail_mean(values):
    def one(chunk):
        x = chunk[np.isfinite(chunk)]
        if x.size < 2: return np.nan
        threshold = np.quantile(x, 0.25)
        return float(np.mean(x[x <= threshold]))
    return rolling(values, one)

def downside(values):
    def one(chunk):
        x = chunk[np.isfinite(chunk)]
        if x.size < 2: return np.nan
        return float(np.sqrt(np.mean(np.minimum(x-0.1, 0.0)**2)))
    return rolling(values, one)

def abs_distribution(values, kind):
    def one(chunk):
        x = np.abs(chunk[np.isfinite(chunk)])
        if not x.size or x.sum() == 0: return np.nan
        p = x/x.sum()
        if kind == "concentration": return float(np.sum(p**2))
        entropy = float(-np.sum(p*np.log(p+1e-300)))
        return entropy/np.log(p.size) if kind == "normalized" and p.size > 1 else entropy
    return rolling(values, one)

def ema(values, span):
    alpha = 2.0/(span+1.0)
    out = np.empty_like(values); out[0] = values[0]
    for row in range(1, values.shape[0]): out[row] = alpha*values[row] + (1-alpha)*out[row-1]
    return out

def actual(name, backend, frames, params):
    op = OperatorRegistry.get(name, backend, mode="any")
    assert op is not None, (name, backend)
    args = [pl.from_pandas(f) for f in frames] if backend == "polars" else frames
    result = op.calculate(*args, **params)
    return result.to_numpy() if isinstance(result, pl.DataFrame) else result.to_numpy(dtype=float)

def main():
    base = np.array([[-1,2],[.5,-3],[np.nan,1],[2,0],[-.5,np.nan],[4,-2],[1.5,3],[-2.5,.5],[0,-1],[3.5,4],[-4,2.5],[2.5,-.5]],dtype=float)
    cond = np.array([[1,0],[0,1],[1,1],[np.nan,0],[1,np.nan],[0,1],[1,0],[1,1],[0,1],[1,0],[0,1],[1,1]],dtype=float)
    frame = pd.DataFrame(base,columns=["A","B"]); condition = pd.DataFrame(cond,columns=frame.columns)
    clean = pd.DataFrame(np.where(np.isfinite(base),base,.25),columns=frame.columns)
    cases = {
      "ts_count_if":([condition],{"window":WINDOW,"min_periods":1},count_if(cond)),
      "ts_sum_if":([frame,condition],{"window":WINDOW,"min_periods":1},conditional(base,cond,"sum")),
      "ts_mean_if":([frame,condition],{"window":WINDOW,"min_periods":1},conditional(base,cond,"mean")),
      "ts_std_if":([frame,condition],{"window":WINDOW,"min_periods":2,"ddof":1},conditional(base,cond,"std")),
      "ts_last_if":([frame,condition],{"window":WINDOW},conditional(base,cond,"last")),
      "ts_max_if":([frame,condition],{"window":WINDOW,"min_periods":1},conditional(base,cond,"max")),
      "ts_min_if":([frame,condition],{"window":WINDOW,"min_periods":1},conditional(base,cond,"min")),
      "ts_tail_mean":([frame],{"window":WINDOW,"q":.25,"side":"lower","min_periods":2},tail_mean(base)),
      "ts_downside_deviation":([frame],{"window":WINDOW,"target":.1,"min_periods":2},downside(base)),
      "ts_abs_concentration":([frame],{"window":WINDOW,"min_periods":1},abs_distribution(base,"concentration")),
      "ts_abs_entropy":([frame],{"window":WINDOW,"normalize":True,"min_periods":1},abs_distribution(base,"normalized")),
      "ts_abs_entropy_normalized":([frame],{"window":WINDOW,"min_periods":1},abs_distribution(base,"normalized")),
      "ts_abs_entropy_nats":([frame],{"window":WINDOW,"min_periods":1},abs_distribution(base,"nats")),
      "ts_ema":([clean],{"span":4},ema(clean.to_numpy(),4)),
    }
    load_all(); rows=[]
    for name,(frames,params,expected) in cases.items():
        for backend in ("pandas_numpy","polars"):
            observed=actual(name,backend,frames,params)
            error=float(np.nanmax(np.abs(observed-expected)))
            passed=bool(np.allclose(observed,expected,equal_nan=True,rtol=1e-11,atol=1e-12))
            rows.append({"canonical":name,"backend":backend,"passed":passed,"max_abs_error":error})
            if not passed: raise AssertionError(f"{name}/{backend}: max_abs_error={error}")
    payload={"status":"PASS","cases":len(cases),"backend_checks":len(rows),"results":rows}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("status","cases","backend_checks")}))
if __name__ == "__main__": main()
