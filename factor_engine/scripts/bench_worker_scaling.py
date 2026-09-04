# -*- coding: utf-8 -*-
"""100k GO §100/§52 worker scaling 实测（1/2/4/8）。

目的：用 CPU-bound 合成 workload（Pandas 向量化 ts_mean/ts_std 因子批跑，
GIL 相关 → 进程级并行才能扩吞吐）实测 worker 数 1/2/4/8 的 throughput 曲线，
定位「出现拐点后禁止继续加 worker」的拐点，并验证多 worker 资源治理
（``multiworker_governance.per_process_cpu_budget``：``workers×OMP<=floor(31/N)``）。

worker 启动机制：每个 worker 是一个独立 ``python3`` 子进程（``--mode worker``），
父进程在 spawn 前为其设置 ``OMP_NUM_THREADS`` 等 BLAS/OpenMP 线程数 env →
保证 OpenMP 运行时在解释器启动前即收到正确配额（fork 继承会丢失该设置）。

运行：
    python3 scripts/bench_worker_scaling.py                # 跑 1/2/4/8 + 超卖档
    python3 scripts/bench_worker_scaling.py --rerun        # 复用已有 JSON 只出 MD

输出：
    /tmp/bench_worker_scaling.json
    factor_engine/artifacts/perf_vec/MULTIWORKER_SCALING.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_FE_ROOT = Path(__file__).resolve().parents[1]   # .../factor_engine
_PARENT = _FE_ROOT.parent                        # .../quant_projects
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

_TOTAL_CORES = 31  # 机器 32 核，OOM 守卫：子进程合计最多 31 核

# 每 worker 批跑的因子数（固定，测「加 worker 能换多少总吞吐」）。
FACTORS_PER_WORKER = int(os.environ.get("BENCH_FACS_PER_WORKER", "60"))
STOCKS = int(os.environ.get("BENCH_STOCKS", "2000"))
DAYS = int(os.environ.get("BENCH_DAYS", "400"))
SEED = int(os.environ.get("BENCH_SEED", "42"))


# --------------------------------------------------------------------------- #
# worker 子进程负载：构造面板 + 批跑因子，输出 {"wall_s", "rss_bytes"}
# --------------------------------------------------------------------------- #
def _build_source(close: "object") -> "object":
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.backend.pandas_backend import PandasBackend
    from tests.helpers import InMemorySeriesSource
    return FactorEngine(data_source=InMemorySeriesSource({"close": close}),
                        backend=PandasBackend())


def _build_factors(n: int) -> list:
    from factor_engine.api.factor import Factor
    from factor_engine.api import ts_mean, ts_std, ts_sum, ts_max
    from factor_engine.api.columns import col
    ops = [
        (ts_mean, {"window": 5}),
        (ts_mean, {"window": 20}),
        (ts_std, {"window": 10}),
        (ts_std, {"window": 20}),
        (ts_sum, {"window": 5}),
        (ts_max, {"window": 10}),
    ]
    out = []
    for i in range(n):
        op, kw = ops[i % len(ops)]
        out.append(Factor(name=f"w{op.__name__}_{i}", expr=op(col("close"), **kw)))
    return out


def _run_worker(oargs: argparse.Namespace) -> int:
    """子进程：算 FACTORS_PER_WORKER 个时序因子，打印 JSON 单行。"""
    import numpy as np
    import pandas as pd
    import psutil
    from factor_engine.runtime.perf_config import PerfConfig
    PerfConfig.invalidate_env_cache()

    # 固定 seed 可复跑
    rng = np.random.RandomState(oargs.seed + oargs.worker_id)
    dts = pd.bdate_range("2020-01-01", periods=oargs.days)
    idx = pd.MultiIndex.from_product(
        [dts, [f"s{i:04d}" for i in range(oargs.stocks)]],
        names=["timestamp", "instrument"],
    )
    n = len(idx)
    close = pd.Series(100.0 + np.cumsum(rng.randn(n) * 0.5), index=idx)

    engine = _build_source(close)
    factors = _build_factors(oargs.factors_per_worker)

    proc = psutil.Process()
    t0 = time.monotonic()
    engine.run_many(factors, enable_cse=True)
    wall = time.monotonic() - t0
    rss = proc.memory_info().rss
    print(json.dumps({"wall_s": round(wall, 6), "rss_bytes": rss}))
    return 0


# --------------------------------------------------------------------------- #
# 父进程：调度 worker 子进程、采集 CPU%/RSS、算 throughput
# --------------------------------------------------------------------------- #
def _measure_cpu(proc_hdl, step: float = 0.05):
    import psutil
    samples = []
    while any(p.poll() is None for p in proc_hdl):
        samples.append(psutil.cpu_percent(interval=step))
    return sum(samples) / max(1, len(samples)) if samples else 0.0


def _spawn_workers(worker_count: int, omp: int, factors_per_worker: int,
                   env_extra: dict, oargs: argparse.Namespace):
    cmd = [sys.executable, os.path.abspath(__file__), "--mode", "worker",
           "--worker-id", str(worker_count), "--omp", str(omp),
           "--stocks", str(oargs.stocks), "--days", str(oargs.days),
           "--factors-per-worker", str(factors_per_worker),
           "--seed", str(oargs.seed)]
    return [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env_extra)
            for _ in range(worker_count)]


def _collect(procs):
    import psutil
    walls = []
    rss = 0
    errs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        if p.returncode != 0:
            errs.append((p.returncode, err[-300:]))
            continue
        for line in out.strip().splitlines():
            try:
                d = json.loads(line)
                walls.append(d["wall_s"])
                rss += d["rss_bytes"]
            except json.JSONDecodeError:
                continue
        # 已退出子进程的 RSS 计入（峰值由 parent 统计近似）
    return walls, rss, errs


def _run_oargs(p: argparse.ArgumentParser) -> argparse.Namespace:
    p.add_argument("--mode", choices=["bench", "worker"], default="bench")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--omp", type=int, default=None)
    p.add_argument("--stocks", type=int, default=STOCKS)
    p.add_argument("--days", type=int, default=DAYS)
    p.add_argument("--factors-per-worker", type=int, default=FACTORS_PER_WORKER)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--json", default="/tmp/bench_worker_scaling.json")
    p.add_argument("--rerun", action="store_true",
                   help="复用已有 JSON 只重生成 MD")
    return p.parse_args()


def _baseline_env() -> dict:
    env = dict(os.environ)
    # 每个 worker 子进程内锁线程数到配额（BLAS/OpenMP/numpy）。
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.pop(key, None)
    return env


def _worker_env(env: dict, omp: int) -> dict:
    e = dict(env)
    # 每 worker 是独立 OS 进程：并行度来自进程数（不共享 GIL）。进程内部锁线程
    # 到配额（BLAS/OpenMP/numpy 向量化），避免单进程内 OpenMP 抢到超过配额。
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        e[key] = str(max(1, omp))
    # 强制内部 scheduler 走线程池：本进程的 CPU-bound 并行由外层多进程承接，
    # 避免 FE 内部 process pool 在受限环境 pickle 失败，也避免单进程内重复
    # 起 process pool 造成 N×M 线程/进程放大。
    e["FACTOR_ENGINE_SCHEDULER"] = "thread"
    return e


def run_bench(oargs: argparse.Namespace) -> dict:
    import psutil
    env = _baseline_env()
    # worker_count -> (omp, label)。solo=31；sharedN=floor(31/N)。
    scaling = [
        (1, _TOTAL_CORES),
        (2, _TOTAL_CORES // 2),
        (4, _TOTAL_CORES // 4),
        (8, _TOTAL_CORES // 8),
    ]
    results: dict = {}
    wall1 = None
    for count, omp in scaling:
        procs = _spawn_workers(count, omp, oargs.factors_per_worker,
                               _worker_env(env, omp), oargs)
        cpu = _measure_cpu(procs)
        walls, rss, errs = _collect(procs)
        if not walls:
            results[f"w{count}"] = {"error": errs[:3]}
            continue
        wall = max(walls)  # 整批 wall = 最慢 worker（barrier）
        total_factors = count * oargs.factors_per_worker
        thr = total_factors / wall
        results[f"w{count}"] = {
            "workers": count,
            "omp_per_worker": omp,
            "total_omp_threads_max": count * omp,
            "factor_budget_rule": f"workers×omp={count}×{omp}={count*omp} <= {_TOTAL_CORES}",
            "factors_per_worker": oargs.factors_per_worker,
            "total_factors": total_factors,
            "wall_s": round(wall, 4),
            "throughput_factors_per_s": round(thr, 2),
            "cpu_percent_avg": round(cpu, 1),
            "rss_total_bytes": rss,
            "rss_total_mb": round(rss / 1024**2, 1),
        }
        if count == 1:
            wall1 = wall
        # 父进程也受 quota（避免父进程抢核导致测量失真不大；solo 用主进程 31）
        time.sleep(0.3)

    # 计算 scaling ratio（相对 1 worker throughput）
    base = results.get("w1", {}).get("throughput_factors_per_s")
    for res in results.values():
        if base and res.get("throughput_factors_per_s"):
            res["scaling_ratio_vs_1"] = round(
                res["throughput_factors_per_s"] / base, 3)
        else:
            res["scaling_ratio_vs_1"] = None

    # ---- 超卖场景：8 workers × 8 threads = 64 > 31（验证 governor 必要性）----
    over_count, over_omp = 8, 8
    procs = _spawn_workers(over_count, over_omp, oargs.factors_per_worker,
                           _worker_env(env, over_omp), oargs)
    cpu = _measure_cpu(procs)
    walls, rss, errs = _collect(procs)
    wall = max(walls) if walls else float("nan")
    total_factors = over_count * oargs.factors_per_worker
    thr = (total_factors / wall) if walls else float("nan")
    base_thr = results.get("w1", {}).get("throughput_factors_per_s")
    results["oversub_8x8"] = {
        "workers": over_count,
        "omp_per_worker": over_omp,
        "total_omp_threads_max": over_count * over_omp,
        "factor_budget_rule": "VIOLATES workers×omp<=31",
        "factors_per_worker": oargs.factors_per_worker,
        "total_factors": total_factors,
        "wall_s": round(wall, 4),
        "throughput_factors_per_s": round(thr, 2) if walls else None,
        "cpu_percent_avg": round(cpu, 1),
        "rss_total_bytes": rss,
        "rss_total_mb": round(rss / 1024**2, 1),
        "scaling_ratio_vs_1": round(thr / base_thr, 3) if (walls and base_thr) else None,
        "error": errs[:3] if errs else None,
    }

    return {
        "scale": {
            "stocks": oargs.stocks,
            "days": oargs.days,
            "factors_per_worker": oargs.factors_per_worker,
            "total_cores_guard": _TOTAL_CORES,
        },
        "results": results,
        "hardware": {"cpu_count": os.cpu_count(),
                     "model": "AMD EPYC 9K84 96-Core (32 vCPU seen)",
                     "ram_gb": None},
    }


def write_json(data: dict, path: str) -> str:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return path


def build_md(data: dict) -> str:
    lines = [
        "# MULTIWORKER_SCALING — worker 1/2/4/8 throughput（100k GO §52）",
        "",
        "## 结论（拐点）",
        _inflection(data),
        "",
        "## 硬件环境",
        f"- CPU: {data['hardware']['cpu_count']} vCPU（AMD EPYC 9K84，32 核 92G 实例）",
        "- OOM 守卫：所有子进程合计最多 31 核（`multiworker_governance.DEFAULT_TOTAL_CORES`）",
        "- 合成面板：%d 股 × %d 日；每 worker 批跑 %d 个 ts_mean/ts_std/ts_sum/ts_max 因子"
        % (data["scale"]["stocks"], data["scale"]["days"], data["scale"]["factors_per_worker"]),
        "- seed 固定可复跑；后端 PandasBackend（CPU-bound、GIL 相关）",
        "",
        "## 1/2/4/8 worker 实测表格",
        "",
        "| worker | OMP/worker | 线程上限(workers×OMP) | 因子数 | wall(s) | throughput(f/s) | scaling vs 1 | CPU% | RSS(MB) |",
        "| ------ | ---------- | -------------------- | ------ | ------- | --------------- | ------------ | ---- | ------- |",
    ]
    for k in ("w1", "w2", "w4", "w8"):
        r = data["results"].get(k) or {}
        if "error" in r and not r.get("throughput_factors_per_s"):
            lines.append(f"| {k} | error → {r['error']} |")
            continue
        lines.append(
            f"| {r['workers']} | {r['omp_per_worker']} | {r['total_omp_threads_max']} | "
            f"{r['total_factors']} | {r['wall_s']:.3f} | {r['throughput_factors_per_s']:.1f} | "
            f"{r.get('scaling_ratio_vs_1') or '-'} | {r['cpu_percent_avg']:.0f} | {r['rss_total_mb']:.0f} |"
        )

    lines += [
        "",
        "## 超卖场景（验证 governor 必要性）",
        "",
        "| 配置 | OMP/worker | 线程上限 | 因子数 | wall(s) | throughput(f/s) | scaling vs 1 | CPU% |",
        "| ---- | ---------- | -------- | ------ | ------- | --------------- | ------------ | ---- |",
    ]
    r = data["results"].get("oversub_8x8")
    if r:
        lines.append(
            f"| 8×8 | {r['omp_per_worker']} | {r['total_omp_threads_max']}（>31 违反治理） | "
            f"{r['total_factors']} | {r['wall_s']:.3f} | "
            f"{r['throughput_factors_per_s'] if r.get('throughput_factors_per_s') else 'N/A'} | "
            f"{r.get('scaling_ratio_vs_1') or '-'} | {r['cpu_percent_avg']:.0f} |"
        )

    lines += [
        "",
        "## 与 multiworker_governance 配额规则一致性",
        "",
        "`workers × OMP <= floor(31 / coexist_count)` 在 1/2/4/8 各档均满足：",
        "- w1: 1×31=31 ✅",
        "- w2: 2×15=30 ✅（coexist=2 → floor(31/2)=15）",
        "- w4: 4×7=28 ✅（coexist=4 → floor(31/4)=7）",
        "- w8: 8×3=24 ✅（coexist=8 → floor(31/8)=3）",
        "- 超卖 8×8=64 > 31 ❌：violates governor，用于量化不加治理的劣化。",
        "",
        "## 治理决策",
        "默认推荐：单主进程内部并发（solo 31 核，`FACTOR_ENGINE_RESOURCE_PROFILE=solo`）。",
        "多进程多 worker 必须显式 `FACTOR_ENGINE_COEXIST=N` 让系统按 `floor(31/N)` 分桶。",
        "",
        "原始 JSON：`/tmp/bench_worker_scaling.json`",
    ]
    return "\n".join(lines)


def _inflection(data: dict) -> str:
    res = data["results"]
    thr = {k: res[k]["throughput_factors_per_s"] for k in ("w1", "w2", "w4", "w8")
           if res.get(k) and res[k].get("throughput_factors_per_s")}
    if not thr:
        return "（无有效档位数据，无法判定拐点。）"
    order = [thr[k] for k in ("w1", "w2", "w4", "w8") if k in thr]
    # 拐点 = 首次 throughput 增速降到 < 某阈值（如 <10% 增量即视为到顶）
    last, peak, peak_key = None, None, None
    pts = [k for k in ("w1", "w2", "w4", "w8") if k in thr]
    for i, k in enumerate(pts):
        v = thr[k]
        if last is None:
            last, peak, peak_key = v, v, k
            continue
        gain = (v - last) / max(1e-9, last)
        if peak is None or v > peak:
            peak, peak_key = v, k
        if gain < 0.10:  # 增量不足 10% → 拐点在上一档到本档之间
            return (f"拐点判定：throughput 在 {k} 档增速仅 {gain*100:.0f}%"
                    f"（<10%），峰值 {peak:.1f} f/s 出现在 {peak_key}"
                    f"。GO §52：达到拐点后禁止继续加 worker。")
        last = v
    return (f"四档均保持增长（末档增速 >10%），尚未出现明显拐点；"
            f"峰值 {peak:.1f} f/s 在 {peak_key}。建议延伸 16 worker 复核上限。"
            f"（若机器 32 核，8 档可能已是物理上限附近）")


def main() -> int:
    p = argparse.ArgumentParser()
    oargs = _run_oargs(p)

    if oargs.mode == "worker":
        return _run_worker(oargs)

    json_path = Path(oargs.json)
    if oargs.rerun and json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = run_bench(oargs)
        write_json(data, str(json_path))

    md_dir = _FE_ROOT / "artifacts" / "perf_vec"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / "MULTIWORKER_SCALING.md"
    md_path.write_text(build_md(data), encoding="utf-8")

    print("=" * 66)
    print("worker scaling benchmark done")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print("=" * 66)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
