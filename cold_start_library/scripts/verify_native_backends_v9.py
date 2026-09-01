#!/usr/bin/env python3
"""Run real Pandas / Polars / DuckDB parity for V9.

Requires a real FactorEngine checkout and the packages pinned in
requirements-native-validation.txt. Import-only stubs are explicitly rejected.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

def require_real(name):
    mod=__import__(name)
    path=str(getattr(mod,'__file__',''))
    version=getattr(mod,'__version__',None)
    if not version or 'fake_deps' in path or 'stub' in (getattr(mod,'__doc__','') or '').lower():
        raise RuntimeError(f'{name} is not a real installed runtime: file={path!r} version={version!r}')
    return mod

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--factor-engine-root',required=True);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int);ap.add_argument('--rtol',type=float,default=1e-5);ap.add_argument('--atol',type=float,default=1e-7);args=ap.parse_args()
    pl=require_real('polars'); duckdb=require_real('duckdb'); require_real('pyarrow')
    fe=Path(args.factor_engine_root).resolve();sys.path.insert(0,str(fe));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from api.dsl_parser import parse_factor,parse_expr
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from ir.analyzer import Analyzer
    from planner.lowerer import Lowerer
    from planner.optimizer import Optimizer
    from backend.sql_pushdown.emitter import compile_plan_to_sql
    from v9_validation_data import Source,synthetic_panel,long_dataframe
    lib=Path(__file__).resolve().parents[1]/'library'/'three_backend_static_path_v9.json'; rows=json.loads(lib.read_text()); rows=rows[args.start:args.end]
    data=synthetic_panel(); src=Source(data); pdeng=FactorEngine(build_backend('pandas'),src,run_mode='research'); pleng=FactorEngine(build_backend('polars'),src,run_mode='research')
    long=long_dataframe(data); con=duckdb.connect(':memory:'); con.register('daily_frame',long); con.execute('CREATE TABLE daily AS SELECT * FROM daily_frame')
    failures=[]
    for i,r in enumerate(rows,args.start):
        try:
            fac=parse_factor(r['formula_v9'],name=r['factor_id'],surface='compat')
            a=pdeng.run(fac)['result'].sort_index(); po=pleng.run(fac); b=po['result'].sort_index()
            av=pd.to_numeric(a,errors='coerce').to_numpy(float); bv=pd.to_numeric(b,errors='coerce').to_numpy(float)
            if not np.array_equal(np.isnan(av),np.isnan(bv)) or not np.allclose(av,bv,equal_nan=True,rtol=args.rtol,atol=args.atol): raise AssertionError('Pandas/Polars mismatch')
            expr=parse_expr(r['formula_v9'],surface='compat'); ar=Analyzer().lower(expr); plan=Optimizer().optimize(Lowerer().to_logical_plan(ar.ir),production=False)
            comp=compile_plan_to_sql(plan,dataset='daily',time_column='timestamp',instrument_column='instrument'); q=comp.query.replace('{{daily}}','daily')
            sq=con.execute(q).fetchdf(); sq['ts']=pd.to_datetime(sq['ts']); s=pd.Series(sq['value'].to_numpy(float),index=pd.MultiIndex.from_arrays([sq['ts'],sq['inst']],names=['timestamp','instrument'])).sort_index(); sv=s.reindex(a.index).to_numpy(float)
            if not np.array_equal(np.isnan(av),np.isnan(sv)) or not np.allclose(av,sv,equal_nan=True,rtol=args.rtol,atol=args.atol): raise AssertionError('Pandas/DuckDB mismatch')
        except Exception as exc: failures.append({'index':i,'factor_id':r['factor_id'],'error':f'{type(exc).__name__}: {exc}'})
        if (i-args.start+1)%50==0: print(i-args.start+1,'/',len(rows),'failures',len(failures),flush=True)
    out=Path('native_backend_results.json');out.write_text(json.dumps({'tested':len(rows),'failures':failures},ensure_ascii=False,indent=2));print(out.resolve());raise SystemExit(1 if failures else 0)
if __name__=='__main__':main()
