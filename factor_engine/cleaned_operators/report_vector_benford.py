"""Six-item same-report first-digit distance, research descriptor only.

This is not the rolling ten-observation Benford estimator and does not infer
fraud probabilities. All six contemporaneously available monetary items must
be finite and nonzero; incomplete vectors produce null, not fabricated digits.
"""
import numpy as np
import pandas as pd
from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.advanced_information import _metadata

@register_operator(name="report_vector_benford_js_divergence",
                   canonical="report_vector_benford_js_divergence",
                   backend="pandas_numpy", source="report_vector_benford",
                   status="experimental")
class ReportVectorBenfordJsDivergence(SeriesOperator):
    metadata = _metadata("report_vector_benford_js_divergence",
        "Six contemporaneous report amounts: normalized first-digit Jensen-Shannon distance; research only, not fraud probability.",
        ["x1","x2","x3","x4","x5","x6"], domain="fundamental",unit="ratio",cost=2)
    def _calculate_series(self,x1,x2,x3,x4,x5,x6):
        frames=(x1,x2,x3,x4,x5,x6)
        if not all(isinstance(x,pd.DataFrame) for x in frames):
            raise TypeError("report vector inputs must be pandas panels")
        if any(not x.index.equals(x1.index) or not x.columns.equals(x1.columns) for x in frames[1:]):
            raise ValueError("report vector inputs must share exact observation identity")
        values=np.stack([x.to_numpy(dtype=float) for x in frames])
        valid=np.all(np.isfinite(values)&(values!=0),axis=0)
        # Mask invalid cells before logarithms, and use wide precision for subnormals.
        absolute=np.abs(np.where(np.isfinite(values)&(values!=0),values,1)).astype(np.longdouble)
        digits=np.floor(absolute/np.power(np.longdouble(10),np.floor(np.log10(absolute))))
        digits=np.clip(digits,1,9).astype(np.int8)
        p=np.stack([(digits==d).mean(axis=0) for d in range(1,10)])
        b=np.log10(1+1/np.arange(1.,10.))[:,None,None]
        m=(p+b)/2
        rp=np.divide(p,m,out=np.ones_like(p),where=p>0)
        js=.5*np.sum(p*np.log(rp)+b*np.log(b/m),axis=0)
        result=np.where(valid,np.sqrt(np.maximum(js,0)/np.log(2)),np.nan)
        out=pd.DataFrame(result,index=x1.index,columns=x1.columns)
        out.attrs=dict(x1.attrs)
        return out
