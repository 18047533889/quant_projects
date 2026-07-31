# -*- coding: utf-8 -*-
"""Production recipe certification beyond the portable three-backend subset."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from backend.context import ExecutionContext
from backend.factory import build_backend
from factor_recipes.planner_bridge import compile_recipe_plans
from tests.helpers import InMemorySeriesSource

SCALARS={
    "window":3,"smooth_window":2,"width":2.0,"ann_factor":252.0,
    "periods":1,"periods_per_year":4,"body_window":3,"shadow_window":3,
    "penetration":0.3,"market_cap_window":3,
}


def _bindings(recipe):
    out={}
    for p in recipe.parameters:
        out[p]=SCALARS[p] if p in SCALARS else p
    return out


def _source(recipes):
    ts=pd.date_range("2024-01-01",periods=24,freq="B")
    assets=["A","B"]
    idx=pd.MultiIndex.from_product([ts,assets],names=["timestamp","instrument"])
    n=len(idx);base=20.0+np.arange(n,dtype=float)*0.03
    scalar_names=set(SCALARS)
    fields={p for r in recipes for p in r.parameters if p not in scalar_names}
    data={}
    for i,name in enumerate(sorted(fields)):
        if name=="period_id":
            period=np.repeat(np.arange(len(ts))//3, len(assets))
            data[name]=pd.Series(period.astype(object),index=idx)
        elif name in {"group","industry","sector"}:
            vals=np.tile(np.array(["G0","G1"],dtype=object),len(ts))
            data[name]=pd.Series(vals,index=idx)
        else:
            vals=base*(1.0+0.01*i)+0.1*np.sin(np.arange(n)/3+i)
            if name in {"high"}: vals=vals+1.0
            if name in {"low"}: vals=vals-1.0
            if name in {"open"}: vals=vals-0.2
            if name in {"volume","market_cap","total_assets","total_equity","current_assets","current_liabilities","inventory","receivables","cash","short_debt","long_debt","invested_capital","fundamental_scale"}: vals=np.abs(vals)*1e6+1e6
            data[name]=pd.Series(vals,index=idx)
    # Common symbols can occur in expanded expressions without being explicit recipe parameters.
    for name in ("close","open","high","low","volume"):
        if name not in data:
            vals=base.copy()
            if name=="high":vals+=1
            if name=="low":vals-=1
            if name=="open":vals-=0.2
            if name=="volume":vals=np.abs(vals)*1e6
            data[name]=pd.Series(vals,index=idx)
    return InMemorySeriesSource(data=data)


def test_every_production_recipe_executes_on_certified_reference_path():
    os.environ["FACTOR_ENGINE_CERTIFY_RECIPE_EVIDENCE"]="1"
    from factor_recipes import FactorRecipeRegistry
    recipes=[FactorRecipeRegistry.get(name) for name in FactorRecipeRegistry.list_names(status="production")]
    recipes=[r for r in recipes if r is not None]
    source=_source(recipes)
    backend=build_backend("pandas")
    for recipe in recipes:
        request={recipe.name:(recipe.name,_bindings(recipe))}
        batch=compile_recipe_plans(request)
        plan=batch.plans[recipe.name]
        out=backend.execute(plan,ExecutionContext(data_source=source,run_mode="research"))
        assert isinstance(out,pd.Series),recipe.name
        assert out.index.equals(source.data["close"].index),recipe.name


def test_production_recipe_names_are_unique_and_registered():
    from factor_recipes import FactorRecipeRegistry
    names=FactorRecipeRegistry.list_names(status="production")
    assert len(names)==len(set(names))
    assert names
