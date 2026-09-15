"""Exact multi-component CPU backend with explicit optional slots and time identity."""
from __future__ import annotations
import copy
import hashlib
import inspect
from pathlib import Path
import numpy as np
import polars as pl
from factor_engine.cleaned_operators.base_polars import SeriesOperator, PANEL_SKIP_COLUMNS, register_operator
from factor_engine.cleaned_operators.fundamental import component_score as reference
from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
MAX_COMPONENTS=reference.MAX_COMPONENTS
_score_component=reference._score_component

def fin_component_score(*panels, component_directions=None, score_weights=None, missing_policy="score_available"):
    from factor_engine.cleaned_operators.common._polars_bridge import verify_frames_share_identity
    provided=[p for p in panels if p is not None]
    if not provided or len(panels)>MAX_COMPONENTS:
        raise ValueError("fin_component_score requires 1..8 component panels")
    if any(not isinstance(p,pl.DataFrame) for p in provided):
        raise TypeError("components must be Polars DataFrames")
    verify_frames_share_identity("fin_component_score",*provided)
    base=provided[0]
    cols=[c for c in base.columns if c not in PANEL_SKIP_COLUMNS]
    count=len(provided)
    directions=([component_directions]*count if isinstance(component_directions,str)
        else ["up"]*count if component_directions is None else list(component_directions))
    weights=[1.]*count if score_weights is None else list(score_weights)
    if len(weights)!=count or len(directions)!=count:
        raise ValueError("weights/directions length must match provided component panels")
    if any(isinstance(w,(bool,np.bool_)) or not isinstance(w,(int,float,np.number))
        or not np.isfinite(w) for w in weights):
        raise ValueError("score_weights must be a finite numeric sequence")
    if missing_policy not in ("score_available","require_full"):
        raise ValueError("invalid missing_policy")
    result=[]
    for c in cols:
        terms=np.stack([_score_component(p[c].to_numpy().astype(float),d).astype(np.longdouble)*np.longdouble(w)
            for p,d,w in zip(provided,directions,weights)],axis=1)
        support=np.isfinite(terms)
        score=np.nansum(terms,axis=1)
        valid=support.all(axis=1) if missing_policy=="require_full" else support.any(axis=1)
        score=np.where(valid & np.isfinite(score) & (abs(score)<=np.finfo(float).max),score,np.nan)
        result.append(pl.Series(c,score.astype(float),dtype=pl.Float64))
    return base.with_columns(result)

metadata=copy.deepcopy(reference.FinComponentScore.metadata)
metadata.tags=[*metadata.tags,"numpy_kernel"]

class PolarsComponentScore(SeriesOperator):
    metadata=metadata
    @property
    def _contract_callable(self):
        return reference.FinComponentScore()._calculate_series
    def _calculate_series(self,*args,**kwargs):
        bound=inspect.signature(self._contract_callable).bind(*args,**kwargs)
        bound.apply_defaults()
        values=bound.arguments
        if values.get("kwargs"):
            raise ValueError("unknown component score parameters")
        return fin_component_score(*(values[name] for name in reference._COMPONENT_PARAMS),
            component_directions=values["component_directions"],
            score_weights=values["score_weights"],missing_policy=values["missing_policy"])
    def physical_spec(self):
        digest=hashlib.sha256(Path(reference.__file__).read_bytes()+Path(__file__).read_bytes()).hexdigest()
        return PhysicalImplementationSpec(canonical="fin_component_score",backend="polars",
            execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,materializes_full_panel=True,
            supports_nulls=True,supports_nan=True,supports_inf=True,
            implementation_source_hash=digest,kernel_identity="component_score.fin_component_score",
            notes="Per-column NumPy CPU; no Pandas panel conversion/GPU. Pandas attrs are not a Polars receipt.")

cls=PolarsComponentScore
register_operator(name="fin_component_score",category="fundamental_period",
    business_category="fundamental",canonical="fin_component_score",source="polars_component_score",
    backend="polars",status="implemented")(cls)
