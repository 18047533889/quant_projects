"""Bounded pivot-query microbenchmark; immutable confirmation/supersession parity."""
import json
from statistics import median
from time import perf_counter
import numpy as np
from factor_engine.cleaned_operators.common._pivot_ledger import confirmed_pivot_events

def original_query(ledger,t,w):
    out=[];i0=t-w+1
    for ev in ledger.events:
        if ev.confirmed_at>t:
            break
        if ev.pivot_at<i0:
            continue
        effective=ledger._superseder_at.get(id(ev))
        if effective is not None and effective<=t:
            continue
        out.append(ev)
    return tuple(out)

rows=[]
for n in (1000,5000):
    ledger=confirmed_pivot_events(np.resize([100.,110.,100.,90.],n),1,.02,positive_only=True)
    targets=range(max(0,n-1000),n);times=[]
    for iteration in range(4):
        t=perf_counter();expected=[original_query(ledger,row,120) for row in targets];old=perf_counter()-t
        t=perf_counter();actual=[ledger.active_at(row,window=120) for row in targets];new=perf_counter()-t
        assert actual==expected
        if iteration:
            times.append((old,new))
    old,new=(median(t[i] for t in times) for i in range(2))
    rows.append(dict(history_rows=n,queries=len(targets),window=120,
                     old_scan_seconds=old,bounded_index_seconds=new,query_speedup=old/new))
print(json.dumps(dict(scope="CPU pivot-query-only microbenchmark; not complete operator/run_many/GPU evidence",
    repetitions=3,warmup=1,conversions="Already-built in-memory ledger; no conversion or ledger construction timed",
    preservation="All returned event tuples match original temporal scan exactly",
    results=rows),indent=2))
