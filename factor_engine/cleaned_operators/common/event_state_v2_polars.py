"""Polars event/state alpha kernels: strict domains, full windows, NumPy CPU."""
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import OperatorMetadata,SeriesOperator,PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.parameter_validation import strict_integer
from factor_engine.backend.contracts import ExecutionKind,PhysicalImplementationSpec

def calculate(name,params,args,kwargs):
    values=dict(zip(params,args))
    values.update(kwargs)
    frame=values[params[0]]
    w=strict_integer(values["window"],"window",minimum=3 if name=="event_recency_z" else 2)
    rw=strict_integer(values["rate_window"],"rate_window",minimum=2) if name=="event_cluster_score" else w
    if name=="event_cluster_score" and rw<=w:
        raise ValueError("rate_window must be > window")
    columns=[c for c in frame.columns if c not in PANEL_SKIP_COLUMNS]
    # Validate the atomic input panel before calculating any output column.
    arrays={c:frame[c].to_numpy().astype(float) for c in columns}
    for arr in arrays.values():
        legal=np.isnan(arr)|(arr==0.)|(arr==1.) if name.startswith("event_") else np.isnan(arr)|np.isfinite(arr)
        if not legal.all():
            raise ValueError("event panel must be strict EventBool {0, 1, NaN}" if name.startswith("event_") else "state panel must contain finite state codes or NaN")
    outputs=[]
    for col,arr in arrays.items():
        out=np.full(arr.size,np.nan)
        for t in range(rw-1,arr.size):
            history=arr[t-rw+1:t+1]
            if np.isnan(history).any():
                continue
            chunk=arr[t-w+1:t+1]
            if name=="event_rate_pct":
                out[t]=np.count_nonzero(chunk==1.)/w
            elif name=="event_cluster_score":
                expected=np.count_nonzero(history==1.)/rw*w
                if expected>1e-12:
                    out[t]=(np.count_nonzero(chunk==1.)-expected)/np.sqrt(expected)
            elif name=="event_recency_z":
                pos=np.flatnonzero(chunk==1.)
                if pos.size<3:
                    continue
                gaps=np.diff(pos).astype(float)
                sd=float(np.std(gaps,ddof=1))
                if sd>1e-12:
                    out[t]=(w-1-pos[-1]-float(np.mean(gaps)))/sd
            elif name=="state_dwell_pct":
                out[t]=np.count_nonzero(chunk==chunk[-1])/w
            elif name=="state_transition_surprise":
                m=np.unique(chunk).size
                if m<2:
                    continue
                p=1.-1./m
                variance=(w-1)*p*(1.-p)
                if variance>1e-12:
                    out[t]=(np.count_nonzero(chunk[1:]!=chunk[:-1])-(w-1)*p)/np.sqrt(variance)
            else:
                raise ValueError("unknown event/state operator")
        outputs.append(pl.Series(col,out,dtype=pl.Float64))
    return frame.with_columns(outputs)

def make(name,meta,reference):
    from factor_engine.cleaned_operators.technical import event_state_v2
    m=OperatorMetadata(name=name,category=meta.category,description=meta.description,
        param_names=list(meta.param_names),panel_params=meta.panel_params,scalar_params=meta.scalar_params,
        param_specs=dict(meta.param_specs),relational_specs=list(meta.relational_specs),
        tags=[*meta.tags,"numpy_kernel"])
    physical=PhysicalImplementationSpec(canonical=name,backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
        supports_nulls=True,supports_nan=True,supports_inf=False,
        implementation_source_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(event_state_v2.__file__).read_bytes()).hexdigest(),
        kernel_identity="event_state_v2."+name,notes="Atomic domain validation, full-window NumPy CPU; no Pandas-panel conversion.")
    def _calculate_series(self,*args,**kwargs):
        return calculate(name,m.param_names,args,kwargs)
    return type("EventStateV2Polars_"+name,(SeriesOperator,),dict(metadata=m,
        _physical_spec=physical,_contract_callable=staticmethod(reference),
        _calculate_series=_calculate_series,__module__=__name__))()
