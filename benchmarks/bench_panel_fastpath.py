import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from factor_engine.backend.cleaned_bridge import panel_to_series
frame=pd.DataFrame(np.arange(256,dtype=float).reshape(8,32),
    index=pd.date_range('2024-01-01',periods=8),columns=[f'a{i}' for i in range(32)])
ctx=SimpleNamespace(timestamp_col='timestamp',instrument_col='instrument')
def baseline():
    result=frame.stack(future_stack=True); result.index.names=['timestamp','instrument']; return result
expected=baseline(); pd.testing.assert_series_equal(panel_to_series(frame,ctx),expected)
seconds={}
for name,fn in [('reference_stack',baseline),('production_fastpath',lambda:panel_to_series(frame,ctx))]:
    t=time.perf_counter()
    for _ in range(1000):fn()
    seconds[name]=time.perf_counter()-t
report={'iterations':1000,'panel_shape':[8,32],'seconds':seconds,'oracle_parity':True,
        'speedup':seconds['reference_stack']/seconds['production_fastpath'],'note':'microbenchmark; no end-to-end claim'}
Path('evidence/r2/20260906_panel_fastpath_benchmark.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
