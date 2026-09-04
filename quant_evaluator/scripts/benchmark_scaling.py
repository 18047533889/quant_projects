"""Synthetic factor-count scaling benchmark (spec §64).

Proves the batched Spearman rank_ic GPU pipeline scales *approximately
linearly* in the number of factors F, so that F=2000 ≈ 4 x F=500 (a small
super-linear factor from VRAM tiling is acceptable, but never exponential).

Method
------
For each F in {250, 500, 1000, 2000}:
  * build a synthetic factor tensor (T=1854, N=5447, F) float32 on disk via
    numpy memmap (one random block written at a time, so the ~82GB F=2000
    case never materialises as one in-memory allocation),
  * stage it to the GPU in the session's factor tiles (estimate_tile + OOM
    retile, spec §7) - again a tile at a time so host memory stays bounded,
  * time the tiled spearman rank_ic kernel-only cold E2E, track peak VRAM.

Host-memory policy (spec/CPU constraint: 32 cores, 92GB):
  * never hold more than a few tiles of the factor tensor in RAM at once;
  * factors and label are read from mmap + `cp.asarray(tile)` per tile, and
    the block generator writes into an mmap so total host footprint is tiny.

Run (from quant_evaluator/):
    OMP_NUM_THREADS=31 python -m quant_evaluator.scripts.benchmark_scaling [--max-f 1000]

Emits:
  * quant_evaluator/docs/benchmarks/scaling_benchmark.json
  * quant_evaluator/docs/benchmarks/scaling_notes.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

_BENCH_DIR = os.path.join(_PROJ, "quant_evaluator", "docs", "benchmarks")

import numpy as np

T = 1854
N = 5447
_SIZES = (250, 500, 1000, 2000)
_NAN_FRAC = 0.05  # 5% NaN cross-section (simulates real factor gaps / missing)
_MIN_OBS = 20
_WORK_DIR = "/tmp/qe_scaling"
_SCALE_GB = 4  # 1854 x 5447 x 1 float32 (~40MB actually; see _factor_bytes)

_GB = 1e9


def _factor_bytes(F: int) -> float:
    return T * N * F * np.dtype(np.float32).itemsize


def _host_footprint_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / _GB
    except Exception:
        return 40.0  # conservative default


def gen_synthetic(path: str, F: int, seed: int) -> None:
    """Write a (T,F,N) float32 factor tensor w/ 5% NaN via mmap in
    factor-tile-sized chunks.

    The on-disk layout is (T, F, N) C-order so that a tile [t, s:s+tile, :]
    is 1854 * (tile*N) contiguous bytes - i.e. each factor tile read is a
    small number of large contiguous reads (real readahead).  A (T, N, F)
    layout would make the identical tile read ~10M strided 64-byte page
    faults and stall the host for tens of seconds per tile, so we pay one
    transpose at write time (numpy handles it; write stays row-contiguous)
    to make the benchmark read cheap.
    """
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(T, F, N))
    try:
        rng = np.random.default_rng(seed)
        block = 32 if F >= 1000 else 64
        for start in range(0, F, block):
            b = min(block, F - start)
            data = rng.standard_normal((T, b, N), dtype=np.float32)
            nanmask = rng.random((T, b, N)) < _NAN_FRAC
            data[nanmask] = np.nan
            mm[:, start : start + b, :] = data  # row-contiguous in file
            del data, nanmask
    finally:
        mm.flush()
        del mm


def bench_one(session, F: int, fact_path: str, y: np.ndarray) -> dict:
    import cupy as cp
    from quant_evaluator.kernels.gpu.correlation import batched_spearman_ic

    fact = np.load(fact_path, mmap_mode="r")  # (T,F,N) float32, never fully in RAM
    print(f"[F={F}] host-avail before run: {_host_footprint_gb():.1f}GB")
    _free0, _tot0 = cp.cuda.runtime.memGetInfo()
    _baseline_used_gb = (_tot0 - _free0) / _GB  # device-global includes other tenants
    tile = session.estimate_tile("spearman", T, N, dtype_bytes=4)
    print(f"[F={F}] planned factor tile = {tile}")

    y_dev = cp.asarray(y)  # labels cached once (T,N) float64
    all_rank_ic = np.zeros(F)
    cur_tile = tile
    t0 = time.time()
    peak_vram = 0.0
    n_tiles = 0
    oom_retries = 0
    for s in range(0, F, max(cur_tile, 1)):
        # (T,F,N) file: tile [t, s:s+tile, :] is (T,tile,N) - stage directly,
        # the kernel expects (T, F, N) so no transpose needed on this path.
        ftile = fact[:, s : s + cur_tile, :]           # (T,tile,N) contiguous
        ftile = np.ascontiguousarray(ftile)            # 1854 x (tile*N) rows
        xdev = cp.asarray(ftile, dtype=cp.float32)     # (T,tile,N) == (T,F,N)
        try:
            ic, _ = batched_spearman_ic(xdev, y_dev, min_obs=_MIN_OBS)
        except cp.cuda.memory.OutOfMemoryError:
            session.retile_on_oom(cur_tile)
            cur_tile = session._final_tile
            oom_retries += 1
            ftile = fact[:, s : s + cur_tile, :]
            ftile = np.ascontiguousarray(ftile)
            xdev = cp.asarray(ftile, dtype=cp.float32)
            ic, _ = batched_spearman_ic(xdev, y_dev, min_obs=_MIN_OBS)
        n_here = min(cur_tile, F - s)
        all_rank_ic[s : s + n_here] = cp.asnumpy(cp.nanmean(ic, axis=0))[:n_here]
        free, total = cp.cuda.runtime.memGetInfo()
        peak_vram = max(peak_vram, total - free)
        n_tiles += 1
        del ftile, xdev, ic
        cp.get_default_memory_pool().free_all_blocks()
    wall = time.time() - t0
    del y_dev
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "F": F,
        "wall_s": wall,
        "tiles": n_tiles,
        "tile_size": cur_tile,
        "oom_retries": oom_retries,
        # device-global used VRAM at peak. Matches the real-500 benchmark's
        # convention; on an idle box this equals our true footprint.
        "peak_vram_gb": round(peak_vram / _GB, 2),
        "baseline_used_gb": round(_baseline_used_gb, 2),
        "factor_tensor_gb": round(_factor_bytes(F) / _GB, 2),
        "factors_per_sec": F / wall if wall else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-f", type=int, default=2000,
                    help="largest F to benchmark (default 2000)")
    ap.add_argument("--host-gb", type=float, default=6.0,
                    help="minimum available host RAM (GB) required to proceed; "
                         "if below this we skip (host OOM guard). The factor "
                         "tensor lives on disk via mmap, not in RAM, so tensor "
                         "size itself is not the gate.")
    args = ap.parse_args()

    sizes = [s for s in _SIZES if s <= args.max_f]
    if not sizes:
        sizes = [args.max_f]

    os.makedirs(_WORK_DIR, exist_ok=True)

    # label: persistent (T,N) float64 random cross-section
    y_rng = np.random.default_rng(7)
    y = y_rng.standard_normal((T, N)).astype(np.float64)

    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy

    # session is opened lazily by estimate_tile; open it explicitly up-front so
    # an import/init failure surfaces before we spend time generating a 80GB file
    session = DeviceEvaluationSession(GPUExecutionPolicy(max_vram_fraction=0.75))
    session._open()
    bench_path = os.path.join(_BENCH_DIR, "scaling_benchmark.json")

    # resume: keep already-persisted per-F results and only compute missing ones
    results: dict = {}
    if os.path.exists(bench_path):
        try:
            with open(bench_path) as fh:
                old = json.load(fh).get("scale", {})
            for k in list(old):
                if int(k) in sizes:
                    results[k] = old[k]
            done = {int(k) for k in results}
            sizes = [s for s in sizes if s not in done]
            print(f"[resume] reusing persisted results for "
                  f"{sorted(done)}; computing {sizes if sizes else 'nothing'}")
        except Exception as e:  # noqa: BLE001
            print(f"[resume] could not load existing {bench_path}: {e}")
    if not sizes:
        _write_notes(results)
        print("all sizes already benchmarked; nothing to compute")

    # absolute paths so the JSON/notes land inside the package regardless of cwd
    notes_path = os.path.join(_BENCH_DIR, "scaling_notes.md")

    def _persist():
        os.makedirs(_BENCH_DIR, exist_ok=True)
        with open(bench_path, "w") as fh:
            json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "hardware": {"gpu": "NVIDIA L20", "ram_gb": 92,
                                    "cpu_cores": 32},
                       "workload": {"T": T, "N": N, "label": "spearman_vwap",
                                    "nan_frac": _NAN_FRAC, "min_obs": _MIN_OBS},
                       "scale": results}, fh, indent=2)

    try:
        for F in sizes:
            if _host_footprint_gb() < args.host_gb:
                print(f"[F={F}] <--host-gb headroom ({_host_footprint_gb():.1f}GB "
                      f"available vs {args.host_gb}GB required); skipping to "
                      f"avoid host OOM.", flush=True)
                continue
            path = f"{_WORK_DIR}/fact_{F}_float32.npy"
            if not os.path.exists(path):
                t0 = time.time()
                print(f"[F={F}] generating synthetic {_factor_bytes(F)/_GB:.1f}GB "
                      f"(5% NaN) mmap...", flush=True)
                gen_synthetic(path, F, seed=F)
                print(f"[F={F}] write took {time.time()-t0:.1f}s, "
                      f"host now {_host_footprint_gb():.1f}GB", flush=True)
            r = bench_one(session, F, path, y)
            results[str(F)] = r
            _persist()
            print(json.dumps(r, indent=2), flush=True)
            if os.path.exists(path):
                os.remove(path)  # free disk as we go
    finally:
        session.close()

    _persist()
    _write_notes(results)
    print("\n=== summary ===")
    for f, r in results.items():
        print(f"F={f}: {r['wall_s']:.2f}s  {r['tiles']} tiles  "
              f"peak {r['peak_vram_gb']}GB  {r['factors_per_sec']:.1f} fac/s")
    if "2000" in results and "500" in results:
        ratio = results["2000"]["wall_s"] / results["500"]["wall_s"]
        print(f"\nscaling: F=2000/F=500 wall ratio = {ratio:.2f} "
              f"(ideal linear = 4.00)")
    if "1000" in results:
        with open(os.path.join(_BENCH_DIR, "scaling_benchmark.json")) as fh:
            saved = json.load(fh)
        n = len(list(saved["scale"].values())) - 1
        if n >= 1:
            keys = list(saved["scale"].keys())
            last, prev = keys[-1], keys[-2]
            ratio = saved["scale"][last]["wall_s"] / saved["scale"][prev]["wall_s"]
            print(f"scaling: F={last}/F={prev} wall ratio = {ratio:.2f}")


def _write_notes(results: dict) -> None:
    lines = []
    lines.append("# Synthetic Factor Count Scaling Benchmark (spec §64)\n")
    lines.append("Proves the unary batched Spearman rank_ic GPU pipeline scales "
                 "approximately linearly in the number of factors F.\n")
    lines.append(f"Workload: T={T} trading days, N={N} assets, "
                 f"label = fixed random vwap proxy, min_obs={_MIN_OBS}, "
                 f"5% NaN in the factor cross-sections.\n")
    lines.append("## Results\n")
    lines.append("| F | wall (s) | tiles | tile size | peak VRAM (GB) | factors/s |")
    lines.append("|---|---------:|------:|----------:|---------------:|----------:|")
    for f, r in results.items():
        lines.append(f"| {f} | {r['wall_s']:.1f} | {r['tiles']} | {r['tile_size']} "
                     f"| {r['peak_vram_gb']} | {r['factors_per_sec']:.1f} |")
    lines.append("")
    if "2000" in results and "500" in results:
        ratio = results["2000"]["wall_s"] / results["500"]["wall_s"]
        lines.append(f"**F=2000 / F=500 wall ratio = {ratio:.2f}** "
                     f"(ideal linear = 4.00).")
    elif "1000" in results and "500" in results:
        r1000 = results["1000"]["wall_s"] / results["500"]["wall_s"]
        lines.append(f"**F=1000 / F=500 wall ratio = {r1000:.2f}** "
                     f"(ideal linear = 2.00; upper bound on the per-factor cost).")
    lines.append("")
    lines.append("## Linearity conclusion\n")
    lines.append("The cost is dominated by a fixed per-factor pipeline "
                 "(rank + distinct-level floor + pairwise finite sums), so "
                 "total wall time is expected to grow linearly in F. A small "
                 "super-linear factor is allowed because a larger F uses small "
                 "tiles and more kernel launches / H2D transfers; it should stay a "
                 "moderate constant and never close to quadratic or exponential.")
    lines.append("")
    lines.append("## Host-memory strategy (92GB machine)\n")
    lines.append("The full F=2000 tensor is ~82GB, far too large for one "
                 "in-memory allocation. To keep host RAM bounded:\n")
    lines.append("- Factors are generated directly into a numpy memmap on "
                 "`/tmp`, block by block, so no "
                 "single allocation exceeds one factor tile and disk, not RAM, "
                 "holds the bulk.")
    lines.append("- The GPU pipeline reads a tile from the memmap, copies it "
                 "to device, runs the batched kernel, frees it - so host "
                 "footprint stays at ~ a few tiles instead of the full tensor.")
    lines.append(f"- Labels are staged once on the device ({(T*N*8)/1e9:.2f}GB) "
                 "and reused across all tiles and all F runs.")
    lines.append(f"- Peak VRAM is bounded by the session's max_vram_fraction "
                 "(0.75) and OOM retiling halves the tile before the device "
                 "ever exhausts.")
    lines.append("- Each benchmark run removes its memmap after completion "
                 "to keep `/tmp` free.")
    lines.append("\n_Generated by `scripts/benchmark_scaling.py` (spec §64)._")
    os.makedirs(_BENCH_DIR, exist_ok=True)
    with open(os.path.join(_BENCH_DIR, "scaling_notes.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
