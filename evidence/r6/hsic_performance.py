"""Bounded HSIC complexity evidence: equivalent formulas, one BLAS thread."""
import json
from statistics import median
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from factor_engine.cleaned_operators.dependence_ext import _hsic

def dense_reference(x,y):
    def kernel(v):
        d=np.abs(v[:,None]-v[None,:])
        positive=d[np.triu_indices(len(v),1)]
        sigma=np.median(positive[positive>0])
        return np.exp(-d*d/(2*sigma*sigma))
    n=len(x);H=np.eye(n)-np.ones((n,n))/n
    K,L=kernel(x),kernel(y)
    return np.trace(K@H@L@H)/np.sqrt(np.trace(K@H@K@H)*np.trace(L@H@L@H))

rng=np.random.default_rng(170)
rows=[]
with threadpool_limits(limits=1):
    for n in (60,120,240):
        x,y=rng.normal(size=(2,n))
        times=[]
        for i in range(4):
            t=perf_counter();expected=dense_reference(x,y);old=perf_counter()-t
            t=perf_counter();got=_hsic(x,y);new=perf_counter()-t
            np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)
            if i:
                times.append((old,new))
        old,new=(median(t[i] for t in times) for i in range(2))
        rows.append(dict(window=n,dense_seconds=old,centered_seconds=new,
                         kernel_speedup=old/new,abs_difference=abs(got-expected)))
print(json.dumps(dict(scope="HSIC kernel-only CPU microbenchmark, not run_many or GPU evidence",
    blas_threads=1,warmup=1,repetitions=3,conversions="NumPy arrays already resident; no conversion timed",
    complexity="dense reference O(n^3); mean-centered O(n^2); both O(n^2) scratch",
    results=rows),indent=2))
