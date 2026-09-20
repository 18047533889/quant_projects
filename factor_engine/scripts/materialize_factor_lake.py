#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""materialize_factor_lake.py — 并行把一批因子值落进 factor lake（staging → 可选 upload COS）。

设计（R58-LAKE-PARALLEL）：
  1. 时间分片并行：把 [start, end] 切成 window-days 的块，每块附带 warmup-days
     的历史加载窗口；块内 run_many + CSE 一次求值整批因子，结果 trim 回块窗口。
  2. 写入独占：每个 (factor_id, year) 分区目录只被一个 worker 持有，
     store.write_arrow(mode="overwrite") 走候选目录 + 原子 rename 的崩溃安全
     写，mutation_lock 按 target_dir 划分 → 进程间零冲突、可任意重跑幂等。
  3. 多机扩展：--shard-index i --shard-count n 把块轮转分给 n 台机器，
     COS 前缀按 (factor_id, year) 天然不相交，各机独立 upload 后即合并。
  4. 全历史因子（expanding/stateful）不可时间分片：自动识别后放进单独的
     连续历史 worker（整段一个块），与分片因子并行互不干扰。

用法：
  python3 factor_engine/scripts/materialize_factor_lake.py \
      --factors-file factors.json --start 2019-01-02 --end 2026-08-24 \
      --workers 4 --upload

factors.json 两种写法：
  [{"name": "f_x", "expr": "ts_mean(AdjClose,10)"}                       # native DSL
   {"name": "f_y", "kind": "python", "python": "op('zscore')(col('Volume'))"}]
"""
from __future__ import annotations

import os

# 在任何重型 import 之前钉环境：子进程 spawn 继承。
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")
_HOME = os.path.expanduser("~")
os.environ.setdefault("ASHARE_PARQUET_ROOT", os.path.join(_HOME, "cos_data"))
os.environ.setdefault(
    "QUANTSOCIETY_WORKSPACE_DATA_ROOT", os.path.join(_HOME, "quant_projects/data/factors")
)
# 显式钉 namespace：兜底 namespace 是按 PID 生成的（anon__host__main__<pid>），
# 多进程下每个 worker 的授权白名单各不相同，write_dir 会跨进程校验失败。
os.environ.setdefault("QUANT_RUN_NAMESPACE", "lake_materialize")

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT = os.path.join(_HOME, "quant_projects")
for _p in (PROJECT, os.path.join(PROJECT, "quant_evaluator"),
           os.path.join(PROJECT, "factor_preprocess"), os.path.join(PROJECT, "vectorbt_qs"),
           os.path.join(PROJECT, "data_access"), os.path.join(PROJECT, "jobs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--factors-file", required=True)
    p.add_argument("--dataset", default="ashare_stock_daily_adj")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--warmup-days", type=int, default=550,
                   help="每块向前多加载的历史天数（供 rolling/stateful 预热，结果会 trim 掉）")
    p.add_argument("--window-days", type=int, default=366,
                   help="时间分片：<=0 整段单块；>0 按日历年对齐分块（每块 ≤ 该天数，边界对齐年份保证分区独占）")
    p.add_argument("--workers", type=int, default=0,
                   help="时间分片进程数；0 = 自动（min(块数, cpu/2, 8)）")
    p.add_argument("--shard-index", type=int, default=0,
                   help="多机分片：本机取块 i %% shard_count == shard_index")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--backend", default="auto")
    p.add_argument("--staging-root", default="",
                   help="覆盖 staging 根（默认由 QUANTSOCIETY_WORKSPACE_DATA_ROOT 推导）")
    p.add_argument("--publish", action="store_true",
                   help="落完后把 staging 分区拷进 published factor_lake 根")
    p.add_argument("--upload", action="store_true",
                   help="落完后用 coscli 并行上传到 --cos-prefix")
    p.add_argument("--cos-prefix", default="cos://factors/data/value/factor_lake",
                   help="因子值 COS 前缀（factors 桶的 data/value/ 前缀可经 factor-admin-cos 写入）")
    p.add_argument("--upload-workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="只打印分块计划，不计算")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 分块计划
# --------------------------------------------------------------------------- #

def build_blocks(start, end, window_days, warmup_days, shard_index, shard_count):
    """返回 [(load_start, block_start, block_end)]，已按 shard 过滤。

    块边界**必须按日历年对齐**：每个 (factor, year) 分区只被一个块持有，
    这是多进程 overwrite 写互不覆盖的前提（块内 trim 到 [b_start, b_end]，
    块与块的 trim 范围不共享任何年份）。
    """
    import pandas as pd
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        raise SystemExit(f"--end {end} 早于 --start {start}")
    if window_days <= 0:
        return [(start - pd.Timedelta(days=warmup_days), start, end)]
    blocks = []
    for y in range(start.year, end.year + 1):
        b_start = max(start, pd.Timestamp(year=y, month=1, day=1))
        b_end = min(end, pd.Timestamp(year=y, month=12, day=31))
        if b_start > b_end:
            continue
        blocks.append((b_start - pd.Timedelta(days=warmup_days), b_start, b_end))
    return [b for i, b in enumerate(blocks) if i % shard_count == shard_index]


# --------------------------------------------------------------------------- #
# 因子构造与全历史检测
# --------------------------------------------------------------------------- #

def build_factor(fp):
    """由 payload 构造 Factor：kind="dsl"（默认，走 parse_factor）或 kind="python"
    （表达式里可用 col('AdjClose') / op('ts_mean')(x, 20)）。"""
    from factor_engine.api.factor import Factor
    if fp.get("kind", "dsl") == "dsl":
        from factor_engine.api.dsl_parser import parse_factor
        return parse_factor(fp["expr"], name=fp["name"])
    from factor_engine.api.columns import col
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory

    def op(n):
        return make_cleaned_call_factory(n)

    expr = eval(fp["python"], {"col": col, "op": op})  # noqa: S307 - 受控输入
    return Factor(name=fp["name"], expr=expr)


def split_full_history(parsed_factors):
    """返回 (分片安全因子, 需连续历史的因子)。识别逻辑与 intake 作业一致。"""
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.runtime.execution_contract import (
        factor_history_requirement, is_full_history_lookback,
    )
    an = Analyzer(production=False)
    sharded, continuous = [], []
    for f in parsed_factors:
        analysis = an.lower(f.expr)
        requirement = factor_history_requirement(getattr(analysis, "ir", None))
        full = bool(getattr(analysis, "requires_full_history", False)
                    or is_full_history_lookback(getattr(analysis, "lookback", 0))
                    or requirement.is_full_history)
        (continuous if full else sharded).append(f)
    return sharded, continuous


# --------------------------------------------------------------------------- #
# 单块执行（worker 进程入口）
# --------------------------------------------------------------------------- #

def run_block(payload):
    """执行一个时间块：run_many(sink=...)，结果按 (factor, year) 分区落 staging。"""
    import pandas as pd
    import pyarrow as pa

    (block, factor_payloads, dataset, backend, staging_root) = payload
    load_start, b_start, b_end = block

    # 延迟重型 import：每个 worker 进程首次任务时才加载
    import factor_engine.cleaned_operators  # noqa: F401
    from factor_engine.backend.factory import build_backend
    from factor_engine.storage.factory import build_data_source
    from factor_engine.runtime.engine import FactorEngine
    from data_access import get_store
    from data_access.runtime.mode_identity import (
        set_runtime_mode_identity, reset_runtime_mode_identity,
    )

    factors = [build_factor(fp) for fp in factor_payloads]
    spec = {"type": "data_access", "dataset": dataset,
            "start_date": str(load_start.date()), "end_date": str(b_end.date())}
    snapshot_id = hashlib.sha256(
        f"{dataset}|{load_start.date()}|{b_end.date()}".encode()
    ).hexdigest()[:16]
    calc_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store = get_store()

    written = {}
    failed = {}

    def sink(name, value):
        s = value.get("result") if isinstance(value, dict) else value
        if not isinstance(s, pd.Series) or not isinstance(s.index, pd.MultiIndex):
            raise TypeError(f"run_many result for {name!r} must be a MultiIndex Series")
        idx = s.index
        dates = pd.to_datetime(idx.get_level_values(0))
        m = (dates >= b_start) & (dates <= b_end)
        if not m.any():
            written[name] = {"rows": 0, "note": "empty-after-trim"}
            return
        assets = idx.get_level_values(1)[m]
        df = pd.DataFrame({
            "datetime": dates[m],
            "asset": assets.astype(str) if hasattr(assets, "astype") else [str(a) for a in assets],
            "value": pd.to_numeric(s[m], errors="coerce").astype("float64").values,
            "calc_time": calc_time,
            "factor_version": "v1",
            "data_snapshot_id": snapshot_id,
            "is_valid": 1,
            "invalid_reason": "",
        })
        years = sorted({int(d.year) for d in dates[m]})
        rows = 0
        for y in years:
            my = df[[int(d.year) == y for d in dates[m]]]
            target = Path(staging_root) / "factors" / name / f"year={y}"
            tbl_y = pa.Table.from_pandas(my, preserve_index=False)
            # 分区目录独占 + overwrite 崩溃安全，可任意重跑幂等
            r = store.write_arrow("factor_lake_staging", tbl_y, factor_id=name,
                                  write_dir=str(target), mode="overwrite")
            rows += int(r.get("rows", 0))
        written[name] = {"rows": rows, "years": years}

    tok = set_runtime_mode_identity("interactive_research", source="lake-materialize")
    t0 = time.time()
    try:
        ds = build_data_source(spec)
        try:
            eng = FactorEngine(build_backend(backend), ds, run_mode="research")
            eng.run_many(factors, market="ashare", enable_cse=True,
                         auto_warmup=True, trim_warmup=True,
                         result_policy="sink", sink=sink)
        finally:
            try:
                ds.close()
            except Exception:
                pass
    except Exception as exc:
        failed["__block__"] = f"{type(exc).__name__}: {exc}"
        failed["__traceback__"] = traceback.format_exc()
    finally:
        reset_runtime_mode_identity(tok)
    return {"block": [str(load_start.date()), str(b_start.date()), str(b_end.date())],
            "elapsed_s": round(time.time() - t0, 1),
            "written": written, "failed": failed}


# 连续历史块：复用 run_block，整段一个块
def run_continuous(payload):
    (factor_payloads, dataset, backend, staging_root, start, end, warmup_days) = payload
    import pandas as pd
    block = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days),
             pd.Timestamp(start), pd.Timestamp(end))
    return run_block((block, factor_payloads, dataset, backend, staging_root))


# --------------------------------------------------------------------------- #
# 上传 / 发布
# --------------------------------------------------------------------------- #

def iter_year_partitions(staging_root, names):
    root = Path(staging_root) / "factors"
    for name in names:
        fdir = root / name
        if not fdir.is_dir():
            continue
        for ydir in sorted(fdir.glob("year=*")):
            if ydir.is_dir():
                yield name, ydir


def _cos_cp(src: Path, dest: str):
    """上传一个本地分区目录到 COS 前缀。

    走 quantsociety COS 网关（qs_cos_gateway.py 包装的 coscli）：位置参数必须在
    flag 之前，且网关按文件校验本地根所有权，因此按文件逐个 cp。
    """
    cli = os.environ.get("FACTOR_COS_CLI", "factor-admin-cos")
    base = dest.rstrip("/")
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        subprocess.run([cli, "cp", str(p), f"{base}/{p.name}"],
                       check=True, capture_output=True, text=True, timeout=600)


def upload_to_cos(staging_root, names, cos_prefix, workers):
    cmds = [(ydir, f"{cos_prefix.rstrip('/')}/factors/{name}/{ydir.name}/")
            for name, ydir in iter_year_partitions(staging_root, names)]
    ok, bad = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cos_cp, src, dest): (src, dest) for src, dest in cmds}
        for fut in as_completed(futs):
            src, dest = futs[fut]
            try:
                fut.result()
                ok += 1
                print(f"  [cos] ok  {src.parent.name}/{src.name} -> {dest}", flush=True)
            except Exception as exc:
                bad.append((str(src), str(exc)))
                print(f"  [cos] FAIL {src}: {exc}", flush=True)
    return ok, bad


def publish(staging_root, names, published_root):
    import shutil
    ok = 0
    for name, ydir in iter_year_partitions(staging_root, names):
        dest = Path(published_root) / "factors" / name / ydir.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(ydir, tmp)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
        ok += 1
    return ok


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def resolve_staging_root(override):
    if override:
        return override
    from data_access import get_store
    store = get_store()
    ds = store._registry.get("factor_lake_staging")
    paths = ds.resolve_paths(factor_id="_probe")
    p = Path(paths[0])
    parts = list(p.parts)
    # glob 形如 .../factor_lake/factors/_probe/year=*/data.parquet；
    # staging 根取 .../factor_lake（命名空间敏感，不能按首个 "factors" 段截）。
    if "factor_lake" in parts:
        return str(Path(*parts[: parts.index("factor_lake") + 1]))
    return str(p.parents[2])


def main(argv=None):
    args = parse_args(argv)
    t_all = time.time()

    factor_defs = json.loads(Path(args.factors_file).read_text(encoding="utf-8"))
    names = [d["name"] for d in factor_defs]
    if len(set(names)) != len(names):
        raise SystemExit("factor names 必须唯一")

    blocks = build_blocks(args.start, args.end, args.window_days,
                          args.warmup_days, args.shard_index, args.shard_count)
    if not blocks:
        raise SystemExit("shard 过滤后没有块：检查 --shard-index/--shard-count")

    # 全历史因子拆分（只用主进程的分析器，不触发数据读取）
    parsed = [build_factor(d) for d in factor_defs]
    sharded_f, cont_f = split_full_history(parsed)
    cont_names = {f.name for f in cont_f}

    cpu = os.cpu_count() or 4
    n_workers = args.workers or max(1, min(len(blocks) + (1 if cont_f else 0),
                                           cpu // 2, 8))
    os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, cpu // max(1, n_workers))))
    os.environ.setdefault("OMP_NUM_THREADS", os.environ["POLARS_MAX_THREADS"])

    staging_root = args.staging_root or resolve_staging_root(None)
    print(f"[plan] factors={len(names)} (continuous={len(cont_names)}) "
          f"blocks={len(blocks)} workers={n_workers} "
          f"POLARS_MAX_THREADS={os.environ['POLARS_MAX_THREADS']}", flush=True)
    print(f"[plan] staging_root={staging_root}", flush=True)
    for i, (ls, bs, be) in enumerate(blocks):
        print(f"  block[{i}] load {ls.date()} .. {be.date()} "
              f"(trim to {bs.date()} .. {be.date()})", flush=True)
    if cont_f:
        print(f"  continuous: {sorted(cont_names)}（整段单块，不可时间分片）", flush=True)

    if args.dry_run:
        return 0

    jobs = [(b, [d for d in factor_defs if d["name"] not in cont_names],
             args.dataset, args.backend, staging_root) for b in blocks]
    if cont_f:
        jobs.append(([d for d in factor_defs if d["name"] in cont_names],
                     args.dataset, args.backend, staging_root,
                     args.start, args.end, args.warmup_days))

    agg_written, agg_failed = {}, {}
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(run_block if len(j) == 5 else run_continuous, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:
                agg_failed["__worker__"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                continue
            for k, v in r["written"].items():
                agg_written.setdefault(k, {"rows": 0, "years": []})
                agg_written[k]["rows"] += v.get("rows", 0)
                agg_written[k]["years"] += v.get("years", [])
            for k, v in r["failed"].items():
                if k == "__traceback__":
                    agg_failed.setdefault("__tracebacks__", []).append(v)
                else:
                    agg_failed[k] = v
            print(f"[block {r['block'][1]}..{r['block'][2]}] {r['elapsed_s']}s "
                  f"written={len(r['written'])} failed={len(r['failed'])}", flush=True)

    print("\n=== per-factor totals ===", flush=True)
    for k in sorted(agg_written):
        v = agg_written[k]
        print(f"  {k:32s} rows={v['rows']:>9} years={sorted(set(v['years']))}", flush=True)

    ok = True
    if agg_failed:
        ok = False
        print("\n=== FAILURES ===", flush=True)
        for k, v in agg_failed.items():
            if k == "__tracebacks__":
                for tb in v[-3:]:
                    print(tb, flush=True)
            else:
                print(f"  {k}: {v}", flush=True)

    missing = [n for n in names if n not in agg_written]
    if missing:
        ok = False
        print("  MISSING factors:", missing, flush=True)

    if ok and args.publish:
        published_root = os.environ.get(
            "FACTOR_LAKE_ROOT", os.path.join(_HOME, "quant_projects/data/factors/lake"))
        n = publish(staging_root, names, published_root)
        print(f"[publish] {n} partitions -> {published_root}", flush=True)

    if ok and args.upload:
        up_ok, up_bad = upload_to_cos(staging_root, names, args.cos_prefix,
                                      args.upload_workers)
        print(f"[upload] ok={up_ok} bad={len(up_bad)}", flush=True)
        if up_bad:
            ok = False

    print(f"\n[done] total {time.time() - t_all:.1f}s verdict={'OK' if ok else 'FAILED'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
