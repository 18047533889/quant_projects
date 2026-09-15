"""Bounded registry-wide diagnostic: actual DuckDB SQL vs reference kernels.

This is an audit, not a certification. Unsupported signatures/emission and
timeouts are recorded explicitly, never counted as numerical passes.
Run as a module from the project root; JSON is written to the supplied path.
"""
import inspect
import json
import signal
import sys
import numpy as np
import pandas as pd
import duckdb
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.sql_pushdown.emitter import _compile_layer, SqlDialect, _SQL_WINDOW_REPAIR_FALLBACKS
from factor_engine.planner.logical_plan import PlanNode


def alarm(*_):
    raise TimeoutError('per-operator 5 second bound')


def audit():
    load_all()
    signal.signal(signal.SIGALRM, alarm)
    rows=[]
    rng=np.random.default_rng(914)
    index=pd.date_range('2024-01-01', periods=72)
    for name in OperatorRegistry.list_canonical():
        record={'operator':name}
        if name=='ADX':
            record.update(status='requires_cse_harness',reason='Expanded-layer probe exceeded its bound; recursive ADX requires the production CSE harness, not this direct-SQL adapter.')
            rows.append(record)
            continue
        try:
            obj=OperatorRegistry.get(name, mode='research')
            if obj is None:
                record['status']='no_pandas_kernel';rows.append(record);continue
            fn=getattr(obj,'_calculate_series',None) or getattr(obj,'_fn',None)
            if fn is None:
                record['status']='unsupported_kernel_adapter';rows.append(record);continue
            record['reference']=inspect.getsourcefile(fn)
            sig=inspect.signature(fn)
            meta=obj.metadata
            series_names=list(getattr(meta,'series_params',()) or ())
            scalars=list(getattr(meta,'scalar_params',()) or ())
            window_names=[n for n in ('window','d','period','lookback') if n in sig.parameters and n not in series_names]
            if not window_names:
                record['status']='no_explicit_window_signature';rows.append(record);continue
            signal.alarm(5)
            kwargs={};inputs=[];data={}
            for j,(arg,param) in enumerate(sig.parameters.items()):
                if param.kind in (param.VAR_KEYWORD,param.VAR_POSITIONAL):continue
                if arg not in window_names and param.default is param.empty and param.annotation in (int,float,bool,str,'int','float','bool','str'):
                    annotation=param.annotation
                    kwargs[arg]=2 if annotation in (int,'int') else 0.05 if annotation in (float,'float') else True if annotation in (bool,'bool') else 'default'
                    continue
                if arg in series_names or arg in ('group','group_id') or (param.default is param.empty and arg not in scalars and arg not in window_names):
                    values=100+rng.normal(0,2,(72,2)).cumsum(axis=0)
                    if arg in ('condition','cond','mask','flag'):
                        values=(values>100).astype(float)
                    if arg in ('group','group_id'):
                        values=np.tile([1.0,1.0],(72,1))
                    values[27+j%3,0]=np.nan
                    values[:4,1]=np.nan
                    frame=pd.DataFrame(values,index=index,columns=['A','B'])
                    kwargs[arg]=frame;data[arg]=frame
                    inputs.append(PlanNode(op='column',attrs={'name':arg}))
                elif arg in window_names:kwargs[arg]=20
                elif param.default is not param.empty:kwargs[arg]=param.default
                else:raise ValueError('unbound scalar '+arg)
            attrs={k:v for k,v in kwargs.items() if k not in data}
            node=PlanNode(op=name,inputs=inputs,attrs=attrs)
            layer=_compile_layer(node,dialect=SqlDialect.DUCKDB)
            if layer is None:
                record['status']='sql_disabled_semantics' if name in _SQL_WINDOW_REPAIR_FALLBACKS else 'sql_not_emitted'
                if name in _SQL_WINDOW_REPAIR_FALLBACKS:record['reason']=_SQL_WINDOW_REPAIR_FALLBACKS[name]
                rows.append(record);continue
            expected=fn(**kwargs)
            if not isinstance(expected,pd.DataFrame):raise TypeError('non-frame reference')
            base=pd.concat([pd.DataFrame({'ts':index,'inst':asset,**{k:v[asset].values for k,v in data.items()}}) for asset in ['A','B']],ignore_index=True)
            with duckdb.connect(config={'threads':1}) as con:
                con.register('base',base)
                actual=con.execute(layer.sql).df().pivot(index='ts',columns='inst',values='_v').reindex(index=index,columns=['A','B'])
            record['attrs']=attrs
            record['missing_mask_differences']=int((actual.isna()!=expected.isna()).to_numpy().sum())
            record['premature_values']=int((actual.notna() & expected.isna()).to_numpy().sum())
            record['status']='pass' if np.allclose(actual.to_numpy(dtype=float,na_value=np.nan),expected.to_numpy(dtype=float,na_value=np.nan),equal_nan=True,rtol=1e-7,atol=1e-9) else 'mismatch'
        except Exception as exc:
            record.update(status='error',error=f'{type(exc).__name__}: {exc}'[:500])
        finally:signal.alarm(0)
        rows.append(record)
    return rows


if __name__=='__main__':
    result=audit()
    with open(sys.argv[1],'w') as out:json.dump(result,out,indent=2,default=str)
    from collections import Counter
    print(dict(Counter(r['status'] for r in result)))
    for row in result:
        if row['status'] in ('error','mismatch'):print(row)
