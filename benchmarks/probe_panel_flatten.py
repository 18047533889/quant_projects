"""Prototype timing/oracle only. Does not modify production conversion code."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

def original(panel,target=None):
    out=panel.stack(future_stack=True)
    out.index.names=['timestamp','instrument']
    return out if target is None or out.index.equals(target) else out.reindex(target)

def proposed(panel,target=None):
    # Candidate scope: nonempty homogeneous float64, simple unique axes only.
    index=pd.MultiIndex.from_product([panel.index,panel.columns],names=['timestamp','instrument'])
    out=pd.Series(panel.to_numpy(copy=False).reshape(-1,order='C').copy(),index=index)
    return out if target is None or out.index.equals(target) else out.reindex(target)

fixtures=[]
for tz in (None,'Asia/Hong_Kong'):
    for reverse_rows in (False,True):
        index=pd.date_range('2024-01-01',periods=8,tz=tz)
        if reverse_rows:index=index[::-1]
        values=np.arange(256,dtype=float).reshape(8,32)
        values[0,0]=np.nan; values[1,1]=np.inf
        panel=pd.DataFrame(values,index=index,columns=[f'a{i}' for i in reversed(range(32))])
        for target in (None,original(panel).index[::-1],original(panel).index[::2]):
            expected=original(panel,target); actual=proposed(panel,target)
            pd.testing.assert_series_equal(actual,expected)
            assert not np.shares_memory(actual.to_numpy(),panel.to_numpy())
            fixtures.append(True)
measurements={}
for name,fn in [('pandas_future_stack',original),('float64_flatten_copy',proposed)]:
    start=time.perf_counter()
    for _ in range(1000): fn(panel)
    measurements[name]=time.perf_counter()-start
result={'fixtures':len(fixtures),'parity':True,'shared_memory':False,'iterations':1000,
    'seconds':measurements,'scope':'prototype only; no production file edited',
    'guards_required':['float64 homogeneous','nonempty','single-level unique index/columns'],
    'limitations':'MultiIndex axes, duplicate axes, extension/mixed dtypes remain on pandas stack. No zero-copy because returned results must not alias cached panels.'}
Path('evidence/r2/20260906_panel_flatten_proposal.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
