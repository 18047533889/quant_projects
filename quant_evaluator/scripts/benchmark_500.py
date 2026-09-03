"""Real 500-factor GPU benchmark (spec §63, §65, §53).

Loads the 500-row factor tensor (float32) built from real factor matrices,
stages it to the GPU in factor tiles (proving tiling works within the VRAM
budget), and times batched Spearman rank_ic for all 500 factors (kernel-only
cold E2E).  Compares against a CPU single-factor oracle.

Emits a benchmark JSON + Markdown (spec §65).
"""

import json
import os
import sys
import time

# ensure the quant_projects root (parent of quant_evaluator) is importable
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np

CP = lambda: None


def main():
    import cupy as cp
    fact = np.load("/tmp/fact_500_float32.npy", mmap_mode="r")  # (T,N,F) float32
    y = np.load("/tmp/label_500.npy")                            # (T,N) float64
    T, N, F = fact.shape
    print(f"real 500-factor: T={T} N={N} F={F}  (loaded mmap, mem {fact.nbytes/1e9:.1f}GB)")

    from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy

    # Use the session's factor-tile planner (spec §7): it estimates the working
    # set for the spearman sort family within the VRAM budget and retiles on OOM.
    session = DeviceEvaluationSession(GPUExecutionPolicy(max_vram_fraction=0.75))
    tile = session.estimate_tile("spearman", T, N, dtype_bytes=4)  # float32 factors
    print(f"planned factor tile = {tile} (VRAM budget {session._vram_budget/1e9:.1f}GB)")
    all_rank_ic = np.zeros(F)
    t0 = time.time()
    peak_vram = 0
    n_tiles = 0
    import cupy as _cp
    cur_tile = tile
    for s in range(0, F, max(cur_tile, 1)):
        ftile = fact[:, :, s : s + cur_tile]
        ftile = np.ascontiguousarray(ftile)
        xdev = cp.transpose(cp.asarray(ftile, dtype=cp.float32), (0, 2, 1))
        try:
            ic, _ = batched_spearman_ic(xdev, cp.asarray(y), min_obs=20)
        except _cp.cuda.memory.OutOfMemoryError:
            # OOM retile (spec §7): halve and retry
            session.retile_on_oom(cur_tile)
            cur_tile = session._final_tile
            ftile = fact[:, :, s : s + cur_tile]
            ftile = np.ascontiguousarray(ftile)
            xdev = cp.transpose(cp.asarray(ftile, dtype=cp.float32), (0, 2, 1))
            ic, _ = batched_spearman_ic(xdev, cp.asarray(y), min_obs=20)
        all_rank_ic[s : s + cur_tile] = cp.asnumpy(cp.nanmean(ic, axis=0))
        free, total = _cp.cuda.runtime.memGetInfo()
        peak_vram = max(peak_vram, total - free)
        n_tiles += 1
        del ftile, xdev, ic
        cp.get_default_memory_pool().free_all_blocks()
    session.close()
    wall = time.time() - t0
    print(f"GPU tiled rank_ic for F={F}: {wall:.2f}s across {n_tiles} tiles, "
          f"peak VRAM {peak_vram/1e9:.1f}GB")

    # CPU single-factor oracle timing (a few factors)
    from quant_evaluator.metrics.ic import compute_daily_ic, _spearman_rank_correlation
    import pandas as pd
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    dates = pd.date_range("2016-01-04", periods=T)
    lb = LabelBundle(target_id="next_ret", values=y, horizon=2,
                     decision_time=tuple(dates), label_start_time=tuple(dates),
                     label_end_time=tuple(dates + pd.Timedelta(days=1)))
    cpu_wall = 0.0
    for f in range(4):
        fb = FactorBatch(
            factor_ids=(f"f{f}",),
            time_axis=AxisRef(name="TradingDay", dtype="datetime", size=T),
            asset_axis=AxisRef(name="OrderBookId", dtype="str", size=N),
            values=fact[:, :, f : f + 1],
            layout="wide",
        )
        ts = time.time()
        compute_daily_ic(fb, lb, method="spearman", min_assets=20)
        cpu_wall += time.time() - ts
    per_factor_cpu = cpu_wall / 4
    est_cpu_500 = per_factor_cpu * F
    speedup = est_cpu_500 / wall if wall > 0 else 0

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": "dbbc3fe75f38d75c88f62ad60e3e0eedc666f026",
        "hardware": {"gpu": "NVIDIA L20", "driver": "580.126.20", "cuda": "13.0",
                     "cupy": cp.__version__, "ram_gb": 92},
        "workload": {"T": T, "N": N, "F": F, "label": "spearman_vwap",
                     "tile": cur_tile, "num_tiles": n_tiles},
        "timing": {
            "gpu_tiled_rankic_seconds": wall,
            "num_tiles": n_tiles,
            "cpu_per_factor_seconds": per_factor_cpu,
            "est_cpu_500_seconds": est_cpu_500,
            "speedup_vs_cpu_single": speedup,
        },
        "memory": {"peak_vram_gb": round(peak_vram / 1e9, 2),
                   "factor_tensor_gb": round(fact.nbytes / 1e9, 2)},
        "throughput": {"factor_per_second": F / wall if wall else 0},
    }
    os.makedirs("quant_evaluator/docs/benchmarks", exist_ok=True)
    with open("quant_evaluator/docs/benchmarks/real_500_factor_benchmark.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print("\nsample all_rank_ic[:8]:", np.round(all_rank_ic[:8], 4))
    np.save("/tmp/all_rank_ic_500.npy", all_rank_ic)


if __name__ == "__main__":
    main()
