"""Read-only small-panel stage timing; no operator execution or market data."""
import cProfile
import json
import os
import time
from pathlib import Path
import psutil
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import numpy as np
import pandas as pd
from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.planner.physical_lowerer import lower_batch_dag
from factor_engine.planner.native_fusion import plan_native_fusion_groups, native_fusion_capability_map

index=pd.MultiIndex.from_product([pd.date_range('2024-01-01',periods=8),list('ABCD')], names=['timestamp','instrument'])
engine=FactorEngine(backend=PandasBackend(),data_source=Source(pd.Series(np.arange(32,dtype=float),index=index)))
perf=PerfConfig(max_workers=4,memory_limit_bytes=1024**3,native_fusion=True)
ctx=engine._make_context(perf=perf)
reports=[]
for count in (100,256,512):
    row={'roots':count}
    factors=[Factor(name=f'f{count}_{i}',expr=col('close')+float(i)) for i in range(count)]
    profiler=cProfile.Profile(); profiler.enable()
    t=time.perf_counter(); dag,analyses=engine._dag_from_factors(factors,perf=perf); row['compile_dag_seconds']=time.perf_counter()-t
    t=time.perf_counter(); physical=lower_batch_dag(dag,analyses=analyses,ctx=ctx,rows=32,instruments=4); row['physical_lower_seconds']=time.perf_counter()-t
    tasks=[physical.tasks[key] for key in physical.roots]
    t=time.perf_counter(); groups=plan_native_fusion_groups(tasks,backend_capability=native_fusion_capability_map(ctx)); row['native_fusion_plan_seconds']=time.perf_counter()-t
    profiler.disable(); profiler.dump_stats(f'evidence/r2/20260906_planning_{count}.prof')
    row.update({'tasks':len(physical.tasks),'fusion_groups':len(groups),'rss_bytes':psutil.Process().memory_info().rss})
    print(json.dumps(row),flush=True); reports.append(row)
    if row['rss_bytes']>1024**3: raise RuntimeError('synthetic profile exceeded 1 GiB RSS budget')
Path('evidence/r2/20260906_planning_scaling.json').write_text(json.dumps(reports,indent=2))
