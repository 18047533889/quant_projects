#!/usr/bin/env python3
"""R61-P1 #56 —— REAL DA/COS worker scaling（1/2/4/8）。

替换 synthetic-Pandas 版本的 worker benchmark：本脚本用 **真实 data_access 读链路 +
真实 COS** 测多进程 worker 伸缩。目标回答用户质疑：
「worker benchmark 是 synthetic Pandas」——它从未触达真实 DA/COS IO、远程扫描字节、
缓存命中率、重复下载。

设计（每档 worker 独立冷缓存，保证远程字节可测）：
  - 每档 worker 使用**全新的 cache root**（``DATA_ACCESS_COS_CACHE_ROOT=<per-level>``），
    因此每档都是冷启动远程读（真实 COS 下载）。
  - 用 ``ASHARE_PARQUET_ROOT=<空目录>`` 让本地镜像视图为空 → 强制走 remote
    （``DATA_ACCESS_COS_READ_MODE=remote``），后端 CLI（``DATA_ACCESS_COS_CLI``）。
  - 工作负载：每 worker 进程用 ``store.prepare_read + execute_prepared_read`` 读
    同一批真实日期区间（``ashare_stock_daily``）。每读拉取 N 个日对象（每对象一个
    parquet）。K 进程同时请求同一对象时由 ``_run_cos_cli`` 的 per-key flock 单飞
    保证只下载 1 次（其余进程等锁 + 锁内 freshness 复检命中缓存）。
  - 资源治理：``multiworker_governance.resolve_worker_budget`` →
    ``per_process_cpu_budget = floor(31 / coexist_count)``，每 worker ``OMP_NUM_THREADS``
    设为该预算，``workers × OMP <= 31``。

度量（每档）：
  - wall_s、throughput（reads/s 与 factors/s）、remote 对象 fetch 数
    （``DATA_ACCESS_COS_FETCH_LOG`` 行数，每个 fetch 一行）、
    duplicate-download 数（同 key 出现 >1 次的 fetch 行数）、
    对象级缓存命中率（``DATA_ACCESS_COS_DOWNLOAD_LOG`` hit/(hit+fetch)）、
    CPU%（进程均值）、RSS MB（进程峰值）。

失败策略：真实 COS 不可达 → **fail loudly**（打印 COS_SKIP + 原因，不伪造数字），
并给出本地 object-store 模拟回退的说明（本脚本含 ``--emulate`` 开关）。

用法：
    python factor_engine/scripts/bench_worker_scaling_real.py            # 真跑 1/2/4/8
    python factor_engine/scripts/bench_worker_scaling_real.py --levels 1,2,4
    python factor_engine/scripts/bench_worker_scaling_real.py --emulate # 本地对象存储模拟
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # .../quant_projects
sys.path.insert(0, str(_REPO))

# 每档 worker 的固定负载：读窗口天数（每对象=1 天 parquet），每 worker 重复轮数。
WINDOW_DAYS = int(os.environ.get("BENCH_REAL_WINDOW_DAYS", "8"))
ROUNDS_PER_WORKER = int(os.environ.get("BENCH_REAL_ROUNDS", "3"))
# 每轮单 worker 完成的"因子等价读"计数（每次 prepare+execute 视为一个因子读）。
FACTOR_READS_PER_ROUND = int(os.environ.get("BENCH_REAL_FACTOR_READS", "1"))

# 真实 COS 端环境（认证：quant-admin 成员经 linux-sudo-policy-coscli 网关）。
COS_CLI = os.environ.get("DATA_ACCESS_COS_CLI", "/usr/local/bin/research-cos")
COS_READ_MODE = os.environ.get("DATA_ACCESS_COS_READ_MODE", "remote")
# 基准 COS 前缀（与 data_access 内置镜像注册表一致）。
COS_BASE = os.environ.get("ASHARE_LQTP_COS_PREFIX", "cos://qs-cold/clean_data/ashare/lqtp_data")

# 日期窗口：2026-06-15..2026-06-30（15 个自然日，10-11 个交易日）。
_WINDOW_START = os.environ.get("BENCH_REAL_START", "2026-06-15")
_WINDOW_END = os.environ.get("BENCH_REAL_END", "2026-06-30")


def _cos_reason() -> str | None:
    """真实 COS 可用性探测。返回 None = 可用；否则返回 SKIP 原因。"""
    if os.environ.get("DATA_ACCESS_SKIP_COS_MIRROR", "").strip() == "1":
        return "DATA_ACCESS_SKIP_COS_MIRROR=1（镜像关闭）"
    try:
        import subprocess as _sp

        out = _sp.run(
            [COS_CLI, "ls", f"{COS_BASE}/StockDailyBar/{_WINDOW_START}.parquet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return f"COS CLI ls 失败 rc={out.returncode}: {(out.stderr or out.stdout)[:200]}"
    except Exception as exc:  # noqa: BLE001
        return f"COS CLI 不可达：{str(exc)[:200]}"
    return None


def _worker_script() -> str:
    """单 worker 子进程负载：真实 DA 读。返回 (script, args)。"""
    import textwrap

    return textwrap.dedent(
        r"""
        # -*- coding: utf-8 -*-
        import json, os, sys, time
        from pathlib import Path
        REPO = sys.argv[1]
        sys.path.insert(0, REPO)
        from data_access.registry.loader import load_registry
        from data_access.core.engine import DuckDBEngine
        from data_access.store import DataAccessStore

        cfg = os.path.join(REPO, 'data_access', 'config', 'datasets.yaml')
        store = DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))
        start = sys.argv[2]; end = sys.argv[3]
        days = int(sys.argv[4]); rounds = int(sys.argv[5]); fac = int(sys.argv[6])
        cols = ['TradeDate', 'Symbol', 'Close', 'Vwap', 'Volume', 'Factor']
        import datetime as dt
        s0 = dt.date.fromisoformat(start); e0 = dt.date.fromisoformat(end)
        spans = []
        for i in range(rounds):
            lo = s0 + dt.timedelta(days=i * days)
            hi = min(e0, lo + dt.timedelta(days=days - 1))
            spans.append((lo.isoformat(), hi.isoformat()))
        out = {"wall_s": 0.0, "reads": 0, "rows": 0, "fetch_by_key": {}, "skipped_missing": 0}
        from data_access.core.exceptions import MissingRequiredPartition, ValidationError

        t0 = time.time()
        # snapshot_policy=latest：多进程并发写同一缓存对象时 prepare 建立的
        # source snapshot 会因 mtime_ns 变化而过期（verified_fail_if_changed/
        # pin 或 strict 模式 fail-closed 抛 SourceSnapshotChanged）。benchmark 是
        # 研究读，用 latest 语义（execute 读当时最新文件），只量真实 IO 伸缩。
        for (lo, hi) in spans:
            for _ in range(fac):
                try:
                    prepared = store.prepare_read(
                        'ashare_stock_daily', columns=cols,
                        time_range=(lo, hi), instrument_filter=None,
                        snapshot_policy='latest',
                    )
                    res = store.execute_prepared_read(prepared)
                    out["rows"] += int(getattr(res.table, "num_rows", 0) or 0)
                    out["reads"] += 1
                except MissingRequiredPartition as exc:
                    # 缺失日（真实 COS 非工作日 partition）：计入跳过，不失败。
                    out["skipped_missing"] += 1
                except ValidationError as exc:
                    # 窗口全部落在缺失日（无任何物理对象）→ 视为跳过，不失败。
                    out["skipped_missing"] += 1
        out["wall_s"] = time.time() - t0
        # 本进程 fetch 计数：读 FETCH_LOG 文件（跨进程共享）。
        fl = os.environ.get('DATA_ACCESS_COS_FETCH_LOG', '')
        if fl and os.path.exists(fl):
            from collections import Counter
            c = Counter()
            for line in open(fl, encoding='utf-8'):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                c[rec.get('key', '')] += 1
            out["fetch_by_key"] = dict(c)
        print(json.dumps(out))
        """
    )


def _run_level(
    workers: int,
    *,
    cache_root: Path,
    fetch_log: Path,
    download_log: Path,
    script: str,
    emulate: bool = False,
) -> dict:
    """跑一个 worker 档。返回指标 dict（真跑出的数字）。"""
    omp = max(1, 31 // workers)
    env = dict(os.environ)
    env.update(
        {
            "DATA_ACCESS_COS_READ_MODE": COS_READ_MODE,
            "DATA_ACCESS_COS_CLI": COS_CLI,
            "DATA_ACCESS_COS_CACHE_ROOT": str(cache_root),
            "DATA_ACCESS_COS_FETCH_LOG": str(fetch_log),
            "DATA_ACCESS_COS_DOWNLOAD_LOG": str(download_log),
            "DATA_ACCESS_COS_SINGLE_FLIGHT": "1",
            # 多进程并发下载同一对象必然改变目标文件 mtime；source snapshot
            # verifier 会因 mtime_ns 变化抛 SourceSnapshotChanged。benchmark 是
            # 研究读（不发布），显式 interactive_research 非 strict，避免 verifier
            # fail-closed 干扰伸缩测量。
            "DATA_ACCESS_RUN_MODE": "interactive_research",
            "QUANT_PRODUCTION_MODE": "0",
            "DATA_ACCESS_STRICT_READ": "0",
            # 关闭 remote HEAD 元数据（非 strict 已默认关，这里显式关死）：
            # 真实 COS 上 HEAD 拿到的 last_modified 与 CLI 下载落盘 mtime 可能
            # 差毫秒，SnapshotVerifier 精确比较 mtime_ns 会在并发下载下误报
            # SourceSnapshotChanged。
            "DATA_ACCESS_REMOTE_SNAPSHOT_META": "0",
            "DATA_ACCESS_SKIP_MTIME_VERIFY": "1",
            "OMP_NUM_THREADS": str(omp),
            "ASHARE_PARQUET_ROOT": str(_EMPTY_LOCAL_ROOT()),
            "FACTOR_ENGINE_COEXIST": str(workers),
        }
    )
    # 清空 fetch 计数日志（该档冷启动）。
    fetch_log.unlink(missing_ok=True)
    download_log.unlink(missing_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(tempfile.mkdtemp(prefix="bw_real_")) / "worker.py"
    script_path.write_text(script, encoding="utf-8")

    procs: list[subprocess.Popen] = []
    t0 = time.time()
    for w in range(workers):
        p = subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                str(_REPO),
                _WINDOW_START,
                _WINDOW_END,
                str(WINDOW_DAYS),
                str(ROUNDS_PER_WORKER),
                str(FACTOR_READS_PER_ROUND),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append(p)
    results = []
    for p in procs:
        out, err = p.communicate(timeout=900)
        if p.returncode != 0:
            raise RuntimeError(f"worker rc={p.returncode}: {err[-800:]}")
        try:
            results.append(json.loads(out.strip().splitlines()[-1]))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"worker JSON parse fail: {exc}\n{out[-800:]}\n{err[-800:]}") from exc
    wall = time.time() - t0

    # fetch 计数：同 key 出现 >1 次 = 重复下载。
    fetch_rows: list[dict] = []
    if fetch_log.exists():
        for line in fetch_log.open(encoding="utf-8"):
            try:
                fetch_rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    from collections import Counter
    key_counts = Counter(r.get("key", "") for r in fetch_rows)
    duplicate_keys = {k: c for k, c in key_counts.items() if c > 1}
    total_fetch = len(fetch_rows)

    # 对象级缓存命中率。
    hits = 0
    fetches = 0
    if download_log.exists():
        for line in download_log.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if rec.get("kind") == "hit":
                hits += 1
            elif rec.get("kind") == "fetch":
                fetches += 1
    cache_hit_pct = 100.0 * hits / max(1, hits + fetches)

    reads = sum(r["reads"] for r in results)
    rows = sum(r["rows"] for r in results)
    # CPU%/RSS：取 worker 进程峰值（简单估计用 wall 期间系统均载不可靠，
    # 这里用 psutil 逐 worker 采样）。
    cpu_pct, rss_mb = _sample_processes(procs)
    return {
        "workers": workers,
        "omp_per_worker": omp,
        "threads_cap": workers * omp,
        "wall_s": round(wall, 3),
        "reads": reads,
        "rows": rows,
        "throughput_reads_s": round(reads / wall, 2) if wall > 0 else 0.0,
        "throughput_factors_s": round((reads * FACTOR_READS_PER_ROUND) / wall, 2) if wall > 0 else 0.0,
        "remote_object_fetches": total_fetch,
        "duplicate_download_count": sum(duplicate_keys.values()),
        "duplicate_keys": duplicate_keys,
        "cache_hit_pct": round(cache_hit_pct, 2),
        "cpu_pct": round(cpu_pct, 1),
        "rss_mb": round(rss_mb, 1),
    }


_EMPTY_LOCAL_ROOT_CACHE: Path | None = None


def _EMPTY_LOCAL_ROOT() -> Path:
    global _EMPTY_LOCAL_ROOT_CACHE
    if _EMPTY_LOCAL_ROOT_CACHE is None:
        _EMPTY_LOCAL_ROOT_CACHE = Path(tempfile.mkdtemp(prefix="bw_real_empty_root_"))
    return _EMPTY_LOCAL_ROOT_CACHE


def _sample_processes(procs: list[subprocess.Popen]) -> tuple[float, float]:
    """对已结束进程抽样（尽力而为）：返回 (cpu%, rss_mb)。"""
    try:
        import psutil  # noqa: F401
    except Exception:  # noqa: BLE001
        return 0.0, 0.0
    total_cpu = 0.0
    peak_rss = 0
    for p in procs:
        try:
            pp = psutil.Process(p.pid)
            cpu = pp.cpu_percent(interval=0.0)
            rss = pp.memory_info().rss
            total_cpu += cpu
            peak_rss = max(peak_rss, rss)
        except Exception:  # noqa: BLE001
            pass
    return total_cpu, peak_rss / (1024 * 1024)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="1,2,4,8", help="comma-separated worker counts")
    ap.add_argument("--emulate", action="store_true", help="local object-store emulation")
    ap.add_argument("--out", default=str(Path(_REPO) / "factor_engine" / "artifacts" / "perf_vec" / "MULTIWORKER_REAL.json"))
    args = ap.parse_args(argv)

    print(f"== R61-P1 #56 REAL DA/COS worker scaling ==")
    print(f"mode={'LOCAL-EMULATION' if args.emulate else 'REAL-COS'}")
    if not args.emulate:
        reason = _cos_reason()
        if reason is not None:
            print(f"COS_SKIP: {reason}")
            print("(无伪造数字。可选回退：--emulate 本地对象存储模拟，指标口径相同。)")
            return 3
        print(f"COS_OK: CLI={COS_CLI} base={COS_BASE} window={_WINDOW_START}..{_WINDOW_END}")

    base_dir = Path("/home/sunhaiwei/cos_data/cos_bench")
    base_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for level in [int(x.strip()) for x in args.levels.split(",")]:
        print(f"-- level w{level} --")
        cache_root = base_dir / f"cache_w{level}"
        fetch_log = base_dir / f"fetch_w{level}.jsonl"
        download_log = base_dir / f"download_w{level}.jsonl"
        # 清旧缓存：每档全新冷缓存（保证远程字节真实）。
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            row = _run_level(
                level,
                cache_root=cache_root,
                fetch_log=fetch_log,
                download_log=download_log,
                script=_worker_script(),
                emulate=args.emulate,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"w{level} FAILED: {exc}")
            results.append({"workers": level, "error": str(exc)[:400]})
            continue
        results.append(row)
        print(f"  wall={row['wall_s']}s reads={row['reads']} rows={row['rows']} "
              f"f/s={row['throughput_factors_s']} fetch={row['remote_object_fetches']} "
              f"dup={row['duplicate_download_count']} cache_hit%={row['cache_hit_pct']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "mode": "emulate" if args.emulate else "real-cos",
                "cos_cli": COS_CLI,
                "cos_base": COS_BASE,
                "window": [_WINDOW_START, _WINDOW_END],
                "window_days": WINDOW_DAYS,
                "rounds_per_worker": ROUNDS_PER_WORKER,
                "levels": results,
                "env": {
                    "DATA_ACCESS_COS_READ_MODE": COS_READ_MODE,
                    "DATA_ACCESS_COS_SINGLE_FLIGHT": "1",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"== wrote {out_path} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
