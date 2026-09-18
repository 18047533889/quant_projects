"""Bounded synthetic comparison, no production data and no output files."""
import json
import time
import numpy as np

def old(vals, codes):
    out = np.empty_like(vals)
    for code in np.unique(codes):
        members = codes == code
        x = vals[members]
        scale = np.max(np.abs(x))
        mean = np.mean(x / scale) * scale if scale else 0.0
        out[members] = x - mean
    return out

def candidate(vals, codes):
    scales = np.zeros(int(codes.max()) + 1)
    np.maximum.at(scales, codes, np.abs(vals))
    scaled = np.divide(vals, scales[codes], out=np.zeros_like(vals), where=scales[codes] > 0)
    counts = np.bincount(codes)
    sums = np.bincount(codes, weights=scaled)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0) * scales
    return vals - means[codes]

rng = np.random.default_rng(32)
results = []
for n, groups in [(5000, 50), (5000, 500), (10000, 1000)]:
    codes = rng.integers(0, groups, n)
    vals = rng.normal(size=n)
    expected = old(vals, codes)
    got = candidate(vals, codes)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    timings = {}
    for label, fn in [("group_loop", old), ("scaled_bincount", candidate)]:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for __ in range(10):
                fn(vals, codes)
            samples.append((time.perf_counter() - start) / 10)
        timings[label] = float(np.median(samples))
    results.append({"members": n, "groups": groups, **timings,
                    "speedup": timings["group_loop"] / timings["scaled_bincount"]})
for magnitude in [1e308, 1e-308, 0.0]:
    vals = magnitude * np.array([1., 1., 1., -1.])
    codes = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(candidate(vals, codes), old(vals, codes))
print(json.dumps({"bounded_synthetic_only": True, "parity": True, "results": results}))
