"""Pandas-free Polars-boundary PIT kernels for research-quality metrics."""
from __future__ import annotations
import copy, hashlib
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base_polars import Operator
from factor_engine.cleaned_operators.fiscal_strict import period_ordinal

SOURCE="factor_engine.cleaned_operators.polars_native.research_quality_pit_native_20260930"
NAMES=("accounting_comparability_score","fiscal_asymmetric_timeliness")
AXIS={"__fe_time__","date","timestamp","trade_date","datetime","stock_code","instrument","symbol","session","__fe_instrument__"}
EPS=1e-12

def _positive(v,n):
    if isinstance(v,(bool,np.bool_)): raise TypeError(f"{n} must be an integer, not bool")
    try: r=int(v)
    except (TypeError,ValueError,OverflowError) as e: raise TypeError(f"{n} must be a positive integer") from e
    if r<1 or float(v)!=r: raise ValueError(f"{n} must be a positive integer")
    return r
def _policy(v):
    p=str(v).lower()
    if p not in {"latest_available","first_available"}: raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return p
def _frame(v,n):
    if not isinstance(v,pl.DataFrame): raise TypeError(f"{n} must be a Polars DataFrame")
    return v
def _columns(panels):
    base=panels[0]
    if any(p.height!=base.height or p.columns!=base.columns for p in panels[1:]): raise ValueError("research quality Polars panels must be exactly aligned")
    for p in panels[1:]:
        if any(not base[n].equals(p[n]) for n in base.columns if n in AXIS or n.startswith("__")): raise ValueError("research quality Polars axes must be exactly aligned")
    return [n for n in base.columns if n not in AXIS and not n.startswith("__")]
def _ord(periods):
    out=np.full(periods.shape,np.nan)
    for i,p in enumerate(periods):
        o=period_ordinal(p)
        if o is not None: out[i]=o
    return out
def _snap(values,ords,policy):
    out=[];state={};first=set()
    for value,o in zip(values,ords):
        if np.isfinite(o):
            try: v=float(value)
            except (TypeError,ValueError,OverflowError): v=np.nan
            if np.isfinite(v):
                k=int(o)
                if policy=="latest_available" or k not in first: state[k]=v
                first.add(k)
        out.append(dict(state))
    return out
def _hist(state,current,consecutive=True):
    if not np.isfinite(current): return {}
    h={k:v for k,v in state.items() if k<=int(current)};keys=sorted(h)
    if consecutive and keys:
        i=len(keys)-1
        while i and keys[i]-keys[i-1]==1: i-=1
        h={k:h[k] for k in keys[i:]}
    return h
def _finite(v):
    try: return bool(np.isfinite(float(v)))
    except (TypeError,ValueError,OverflowError): return False

def _timeliness(e,r,p,periods,min_periods,policy):
    ords=_ord(p);es=_snap(e,ords,policy);rs=_snap(r,ords,policy);out=np.full(len(e),np.nan)
    for row,current in enumerate(ords):
        eh=_hist(es[row],current);rh=_hist(rs[row],current);keys=sorted(set(eh)&set(rh))[-periods:]
        if len(keys)<min_periods: continue
        y=np.asarray([eh[k] for k in keys]);x=np.asarray([rh[k] for k in keys]);valid=np.isfinite(y)&np.isfinite(x);y,x=y[valid],x[valid]
        if len(y)<min_periods: continue
        bad=(x<0).astype(float);X=np.column_stack((np.ones(len(x)),x,bad,x*bad))
        if np.linalg.matrix_rank(X)!=X.shape[1]: continue
        try:
            b=np.linalg.lstsq(X,y,rcond=None)[0][3]
            if np.isfinite(b): out[row]=float(b)
        except (np.linalg.LinAlgError,ValueError): pass
    return out

def _comparability(e,r,g,p,periods,min_periods,min_peers,policy):
    rows,cols=e.shape;ords=np.full(p.shape,np.nan)
    for i in range(rows):
        for j in range(cols):
            o=period_ordinal(p[i,j])
            if o is not None: ords[i,j]=o
    es=[{} for _ in range(cols)];rs=[{} for _ in range(cols)];gs=[{} for _ in range(cols)];first=[set() for _ in range(cols)];out=np.full((rows,cols),np.nan)
    for i in range(rows):
        for j in range(cols):
            o=ords[i,j];y=e[i,j];x=r[i,j];group=g[i,j]
            if not np.isfinite(o) or not _finite(y) or not _finite(x): continue
            k=int(o)
            if policy=="latest_available" or k not in first[j]: es[j][k]=float(y);rs[j][k]=float(x);gs[j][k]=group
            first[j].add(k)
        for j in range(cols):
            cur=ords[i,j]
            if not np.isfinite(cur): continue
            k=int(cur);keys=sorted(q for q in es[j] if q<=k)[-periods:]
            if len(keys)<min_periods: continue
            group=gs[j].get(k)
            try:
                if group is None or bool(np.asarray(group!=group).item()): continue
            except (TypeError,ValueError): pass
            y=np.asarray([es[j][q] for q in keys]);x=np.asarray([rs[j][q] for q in keys]);ok=np.isfinite(y)&np.isfinite(x)
            if int(ok.sum())<min_periods: continue
            x,y=x[ok],y[ok]
            if np.std(x)<=EPS: continue
            X=np.column_stack((np.ones(len(x)),x))
            if np.linalg.matrix_rank(X)!=2: continue
            beta=np.linalg.lstsq(X,y,rcond=None)[0][1];errors=[]
            for peer in range(cols):
                if peer==j or gs[peer].get(k)!=group: continue
                pk=[q for q in es[peer] if q<=k and q in keys]
                if len(pk)<min_periods: continue
                py=np.asarray([es[peer][q] for q in pk]);px=np.asarray([rs[peer][q] for q in pk]);valid=np.isfinite(py)&np.isfinite(px)
                if int(valid.sum())<min_periods: continue
                errors.extend(np.abs(beta*px[valid]-py[valid]).tolist())
            if len(errors)>=min_peers: out[i,j]=-float(np.mean(errors))
    return out

def _compute(b,name):
    keys=("scaled_earnings","report_return","industry","period_id") if name==NAMES[0] else ("scaled_earnings","report_return","period_id")
    panels=[_frame(b[k],k) for k in keys];cols=_columns(panels);base=panels[0]
    n=_positive(b.get("periods",16),"periods");m=_positive(b.get("min_periods",12),"min_periods")
    if m>n: raise ValueError("min_periods must not exceed periods")
    policy=_policy(b.get("revision_policy","latest_available"))
    if name==NAMES[0]:
        peers=_positive(b.get("min_peers",5),"min_peers");e,x,g,p=[q.select(cols).to_numpy() for q in panels]
        result=_comparability(e.astype(float),x.astype(float),g,p,n,m,peers,policy)
    else:
        e,x,p=[q.select(cols).to_numpy() for q in panels]
        result=np.column_stack([_timeliness(e[:,j],x[:,j],p[:,j],n,m,policy) for j in range(len(cols))]) if cols else np.empty((base.height,0))
    return base.with_columns([pl.Series(c,result[:,j]) for j,c in enumerate(cols)])

class _Operator(Operator):
    _HANDLES_CALL_CONTRACT=True
    def __init__(self,n,m,s): self.name,self.metadata,self._physical_spec=n,m,s
    def calculate(self,*args,**kwargs):
        args,kwargs=self._prepare_call(args,kwargs);b=dict(zip(self.metadata.param_names,args));b.update(kwargs);return _compute(b,self.name)
    _calculate_series=calculate

def register_research_quality_pit_native_20260930():
    from factor_engine.backend.polars_backend_kind import canonical_polars_is_delegate
    from factor_engine.cleaned_operators import record_backend_replacement_after,replace_backend
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    source_root = Path(__file__).resolve().parents[1]
    source_files = (Path(__file__), source_root / "fiscal_strict.py", source_root / "fundamental" / "research_quality.py")
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()
    done=[]
    for n in NAMES:
        if not canonical_polars_is_delegate(n,production_mode=False): continue
        ref=OperatorRegistry.get(n,"pandas_numpy",mode="any")
        if ref is None: continue
        m=copy.deepcopy(ref.metadata);m.tags=list(m.tags or [])+["preserve_panel_time_coordinate"]
        spec=PhysicalImplementationSpec(canonical=n,backend="polars",execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,supports_lazy=False,supports_streaming=False,materializes_full_panel=True,supports_nulls=True,supports_nan=True,supports_inf=True,implementation_source_hash=digest,emitter_identity=SOURCE+":numpy-panel-kernel:v1",kernel_identity=SOURCE+":"+n+":pit-v1",parameter_domain_hash=hashlib.sha256(repr((m.param_names,m.param_specs)).encode()).hexdigest(),semantic_contract_hash=hashlib.sha256((n+":research_quality_pandas-authority:v1").encode()).hexdigest(),notes="Pandas-free Polars boundary using NumPy fiscal PIT state and OLS kernels.")
        migration=replace_backend(n,"polars",reason="replace inaccurate Polars delegate with PIT research-quality kernel",source=SOURCE)
        OperatorRegistry.register(_Operator(n,m,spec),canonical=n,backend="polars",source=SOURCE)
        record_backend_replacement_after(migration,n,"polars",source=SOURCE);done.append(n)
    return tuple(done)
__all__=["register_research_quality_pit_native_20260930"]
