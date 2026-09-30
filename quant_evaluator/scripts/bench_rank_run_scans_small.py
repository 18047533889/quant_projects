"""Bounded microbenchmark for old bincount and current sorted-run GPU ranks.

Defaults to eight rows. The explicit rows <= 16 and columns <= 10000 guards
keep generated inputs and the former bincount workspace small. Pool totals are
retained allocator bytes after a call, not measured peak live memory.
"""

from __future__ import annotations

import argparse
import json
import statistics

import cupy as cp
import numpy as np

from quant_evaluator.kernels.gpu.rank import batched_rank


def _old_bincount_rank(values):
    flat = values.reshape(-1, values.shape[-1])
    mask = ~cp.isfinite(flat)
    safe = cp.where(mask, cp.inf, flat)
    order = cp.argsort(safe, axis=1, kind="stable")
    sorted_values = cp.take_along_axis(safe, order, axis=1)
    finite = sorted_values != cp.inf
    starts = cp.zeros_like(sorted_values, dtype=cp.bool_)
    starts[:, 0] = True
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_id = cp.cumsum(starts, axis=1, dtype=cp.int32) - 1
    pos = cp.arange(1, flat.shape[1] + 1, dtype=cp.float64)[None, :]
    weights = cp.where(finite, pos, 0.0)
    max_groups = int(group_id.max().item()) + 1
    keys = (cp.arange(flat.shape[0], dtype=cp.int64)[:, None] * max_groups + group_id).ravel()
    bins = flat.shape[0] * max_groups
    sums = cp.bincount(keys, weights=weights.ravel(), minlength=bins)
    counts = cp.bincount(keys, minlength=bins).astype(cp.float64)
    sorted_ranks = (sums / cp.maximum(counts, 1.0))[keys].reshape(flat.shape)
    ranks = cp.empty_like(sorted_ranks)
    ranks[cp.arange(flat.shape[0])[:, None], order] = sorted_ranks
    return cp.where(mask, cp.nan, ranks).reshape(values.shape)


def _measure(fn, values, repeats):
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=1 << 30)
    samples = []
    max_used_after_return = 0
    try:
        with cp.cuda.using_allocator(pool.malloc):
            warm = fn(values)
            cp.cuda.Device().synchronize()
            del warm
            cp.cuda.Device().synchronize()
            pool.free_all_blocks()
            for _ in range(repeats):
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                result = fn(values)
                end.record()
                end.synchronize()
                samples.append(cp.cuda.get_elapsed_time(start, end))
                max_used_after_return = max(max_used_after_return, pool.used_bytes())
                if _ + 1 == repeats:
                    host_result = cp.asnumpy(result)
                del result
                cp.cuda.Device().synchronize()
            retained_bytes = pool.total_bytes()
    except cp.cuda.memory.OutOfMemoryError as exc:
        pool.free_all_blocks()
        raise RuntimeError("budget_rejected: CuPy pool exceeded its 1 GiB limit") from exc
    pool.free_all_blocks()
    return statistics.median(samples), retained_bytes, max_used_after_return, host_result


def _inputs(rows, columns, dtype, distribution, seed):
    rng = np.random.default_rng(seed)
    if distribution == "unique":
        data = rng.standard_normal((rows, columns))
    elif distribution == "ties17":
        data = rng.integers(-8, 9, size=(rows, columns))
    else:
        data = np.full((rows, columns), 7)
    return data.astype(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--columns", type=int, default=5461)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.rows <= 16 or not 1 <= args.columns <= 10000:
        parser.error("rows must be 1..16 and columns 1..10000")
    if not 1 <= args.repeats <= 9:
        parser.error("repeats must be 1..9")

    output = {
        "shape": [args.rows, args.columns],
        "repeats": args.repeats,
        "pool_limit_bytes": 1 << 30,
        "memory_note": "pool retained bytes after calls; not exact peak live VRAM",
        "results": [],
    }
    for dtype in (np.float32, np.float64):
        for distribution in ("unique", "ties17", "constant"):
            host = _inputs(args.rows, args.columns, dtype, distribution, 20261001)
            values = cp.asarray(host)
            cp.cuda.Device().synchronize()
            try:
                new_ms, new_reserved, new_used, new_host = _measure(batched_rank, values, args.repeats)
                old_ms, old_reserved, old_used, old_host = _measure(_old_bincount_rank, values, args.repeats)
            except RuntimeError as exc:
                output["results"].append({
                    "dtype": np.dtype(dtype).name,
                    "distribution": distribution,
                    "input_bytes": int(host.nbytes),
                    "status": "budget_rejected",
                    "reason": str(exc),
                    "parity": "not_checked",
                })
                del values, host
                cp.cuda.Device().synchronize()
                print(json.dumps(output, indent=2, sort_keys=True))
                return
            np.testing.assert_allclose(new_host, old_host, rtol=0, atol=0, equal_nan=True)
            output["results"].append({
                "dtype": np.dtype(dtype).name,
                "distribution": distribution,
                "input_bytes": int(host.nbytes),
                "status": "complete",
                "parity": "exact",
                "new_event_median_ms": new_ms,
                "old_event_median_ms": old_ms,
                "new_pool_reserved_bytes_after_calls": new_reserved,
                "old_pool_reserved_bytes_after_calls": old_reserved,
                "new_max_pool_used_bytes_after_return": new_used,
                "old_max_pool_used_bytes_after_return": old_used,
            })
            del values, host
            cp.cuda.Device().synchronize()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
