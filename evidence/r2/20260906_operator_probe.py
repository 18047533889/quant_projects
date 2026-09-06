import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry as R
from factor_engine.backend.polars_backend_kind import canonical_polars_kind
load_all()
rows=[]
for name in R.list_canonical():
    ops=R._operators.get(name,{})
    rows.append({'canonical':name,'backends':list(ops),'polars_kind':canonical_polars_kind(name).value,
        'pandas_class':str(type(ops.get('pandas_numpy'))),'polars_class':str(type(ops.get('polars')))})
Path('evidence/r2/20260906_operator_inventory.json').write_text(json.dumps(rows,indent=2))
print('COUNTS',len(rows),Counter(r['polars_kind'] for r in rows),flush=True)
print('NO_POLARS',[r['canonical'] for r in rows if 'polars' not in r['backends']],flush=True)
data=pd.DataFrame({'a':[-8.,-1.,0.,1.,8.,np.nan,np.inf,-np.inf,1e-10], 'b':[1.,2.,3.,4.,np.nan,6.,7.,8.,9.]})
for name in ['abs','signed_sqrt','signed_log','sqrt_abs','cbrt','sigmoid','log_abs','exp_neg','truncate','round','expanding_mean','expanding_std','expanding_zscore','ts_delta','ts_pct','ts_sum','ts_count','ts_var']:
    p=R.get(name,'pandas_numpy',mode='any'); q=R.get(name,'polars',mode='any')
    print('OPS',name,type(p),type(q),flush=True)
    try:
        a=p.calculate(data).to_numpy(); b=q.calculate(pl.from_pandas(data)).to_numpy()
        print('PARITY',name,np.allclose(a,b,equal_nan=True),a.tolist() if not np.allclose(a,b,equal_nan=True) else '',b.tolist() if not np.allclose(a,b,equal_nan=True) else '',flush=True)
    except Exception as e: print('ERROR',name,type(e).__name__,str(e),flush=True)
