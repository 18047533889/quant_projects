import json
import time
import types
from pathlib import Path
from factor_engine.planner.read_wave_planner import ReadWavePlanner
from factor_engine.tests.r39.test_read_wave_marginal_cache import legacy_best

def planner(n):
    p=ReadWavePlanner(wave_memory_budget=1024**3,rows_estimate=256)
    for i in range(n):
        p.register_scan_task(str(i),dataset='d',source_scope='s',snapshot_id='v',time_range=None,columns=['close'])
    return p

results=[]
for count in (256,512,1024):
    fast=planner(count); t=time.perf_counter(); actual=fast.plan(); elapsed=time.perf_counter()-t
    row={'roots':count,'optimized_seconds':elapsed,'waves':len(actual.waves)}
    if count<=512:
        slow=planner(count); slow._best_marginal=types.MethodType(legacy_best,slow)
        t=time.perf_counter(); expected=slow.plan(); row['baseline_seconds']=time.perf_counter()-t
        assert actual==expected; row['exact_plan_parity']=True
    else: row['baseline_status']='NOT_RUN; bounded baseline at 256 and 512'
    results.append(row); print(json.dumps(row),flush=True)
Path('evidence/r2/20260906_read_wave_marginal_benchmark.json').write_text(json.dumps(results,indent=2))
