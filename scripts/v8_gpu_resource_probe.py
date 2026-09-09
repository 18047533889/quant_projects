"""Isolated real CUDA probe; records allocator-observed peaks, not estimates."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cupy as cp
from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('quant_evaluator/kernels/gpu/quantile.py', 'quant_evaluator/kernels/gpu/rank.py')

def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}

def main():
    before = hashes()
    rng = np.random.default_rng(808)
    shape = (32, 8, 5000)
    x = rng.normal(size=shape).astype(np.float32)
    r = rng.normal(0, .01, size=(shape[0], shape[2]))
    x[0, 0, 0] = np.inf
    rows = []
    for q in (10, 20, 40):
        pool = cp.cuda.MemoryPool()
        peak = {'active': 0, 'reserved': 0}
        def allocate(n):
            result = pool.malloc(n)
            peak['active'] = max(peak['active'], pool.used_bytes())
            peak['reserved'] = max(peak['reserved'], pool.total_bytes())
            return result
        with cp.cuda.using_allocator(allocate):
            xd, rd = cp.asarray(x), cp.asarray(r)
            resident = pool.used_bytes()
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            out, counts = batched_quantile_returns(xd, rd, q, min_assets=1,
                return_counts=True, workspace_bytes=8 << 20)
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start
            host, host_counts = cp.asnumpy(out), cp.asnumpy(counts)
            for t, f in ((0, 0), (10, 3), (31, 7)):
                finite = np.isfinite(x[t, f])
                bounds = np.quantile(x[t, f, finite], np.arange(1, q) / q)
                buckets = np.searchsorted(bounds, x[t, f], side='right')
                for bucket in range(q):
                    mask = finite & (buckets == bucket)
                    np.testing.assert_allclose(host[t, bucket, f], r[t, mask].mean(), rtol=1e-10, atol=1e-12)
                    assert host_counts[t, bucket, f] == mask.sum()
            output_bytes = out.nbytes + counts.nbytes
            assert peak['active'] <= resident + output_bytes + (8 << 20)
            rows.append(dict(shape_TFN=shape, Q=q, method='max', precision='factor-float32/return-float64',
                workspace_budget_bytes=8 << 20, resident_input_bytes=resident,
                output_bytes=output_bytes, observed_peak_active_bytes=peak['active'],
                observed_peak_reserved_bytes=peak['reserved'], seconds=elapsed,
                independent_numpy_rows_verified=3, assertions='PASS'))
            del xd, rd, out, counts
        pool.free_all_blocks()
    assert hashes() == before
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    result = dict(device=str(props['name']), cupy=cp.__version__, numpy=np.__version__,
        source_hashes=before, source_unchanged=True, runs=rows,
        scope='Isolated allocator-observed active/reserved peaks. Excludes CUDA context/other processes. Cold per-run pool; kernels may be JIT-cached. Not 100k-factor certification; no async/multi-GPU claim.')
    (ROOT / 'evidence/v8/gpu_resource_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
