# -*- coding: utf-8 -*-
"""用本地 DSL 求值器 + 表内 Factor 复权数据，批量重算 LQTP 因子到 data/factor_pools/lqtp/。

口径（与平台 functions.yaml 一致，已核验）：
  close=Close*Factor, open/high/low/pre_close/vwap=...*Factor,
  volume=Volume/Factor, amount=Amount, ret=Return, factor=Factor
后复权面板来自 data/build/ohlcv_adj.parquet（build_ohlcv_adj_from_table.py 已用表内 Factor 落盘）。

默认模式（无参数）幂等：只重算还未被覆盖为后复权的池子因子（按 mtime 阈值判断），
单进程顺序执行 + 定期 gc，避免并发 5460x2585 面板爆内存。

用法: python3.12 recompute_lqtp_adj.py [--name NAME] [--limit N] [--retry-names a,b,c] [--threads N]
"""
import os, sys, json, time, argparse, gc
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lqtp_dsl_eval import build_panels, eval_formula, OPS

OHLCV = f"{ROOT}/data/build/ohlcv_adj.parquet"
POOL = f"{ROOT}/data/factor_pools/lqtp"
FORMULA_MAP = f"{ROOT}/data/build/lqtp_formula_map.json"
LOG = "/tmp/recompute_lqtp_adj.log"

# 2026-08-27T12:00 本地表内 Factor 复权运行起点；mtime 早于此的池子文件视为未后复权
CUTOFF_MTIME = 1787803200.0

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def need_recompute(name):
    p = os.path.join(POOL, f"{name}.parquet")
    if not os.path.exists(p):
        return True
    return os.path.getmtime(p) < CUTOFF_MTIME

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 个")
    ap.add_argument("--name", default="", help="只算指定名字")
    ap.add_argument("--retry-names", default="", help="重算逗号分隔名单(仅这些失败过的)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--assets", type=int, default=0, help=">0 时只算前 N 只股票(调试)")
    args = ap.parse_args()

    t0 = time.time()
    # 面板
    ohlcv = pd.read_parquet(OHLCV)
    ohlcv_index = ohlcv.index
    P = build_panels(ohlcv)
    if args.assets:
        sub = list(P["close"].columns[:args.assets])
        for k in P:
            if isinstance(P[k], pd.DataFrame):
                P[k] = P[k].reindex(columns=sub)
    else:
        # 默认: 与特征矩阵对齐, 只算 tradable (297 只), 内存小 20x
        trad = list(pd.read_parquet("/tmp/tradable_assets.parquet")["asset"])
        for k in P:
            if isinstance(P[k], pd.DataFrame):
                P[k] = P[k].reindex(columns=trad)
    all_assets = list(P["close"].columns)
    plog(f"面板 assets={len(all_assets)} dates={P['close'].shape[0]}")

    formula_map = json.load(open(FORMULA_MAP))

    if args.retry_names:
        names = [n.strip() for n in args.retry_names.split(",") if n.strip()]
    elif args.name:
        names = [args.name]
    else:
        # 默认：只算池子中尚未被后复权覆盖的
        names = [n for n in formula_map.keys() if need_recompute(n)]
    if args.limit:
        names = names[:args.limit]
    plog(f"待重算: {len(names)} (默认模式已跳过 cutoff 之后覆盖过的池子因子)")

    if len(all_assets) not in (5460, len(list(pd.read_parquet("/tmp/tradable_assets.parquet")["asset"]))):
        plog(f"警告: 面板资产数={len(all_assets)} != 5460，输出列宽与池子网格不一致，不再继续")
        return

    ok = []; failed = {}

    def work(name):
        try:
            out = eval_formula(formula_map[name]["formula"], P, OPS)
            if isinstance(out, np.ndarray):
                out = pd.DataFrame(out, index=ohlcv_index, columns=all_assets)
            if not hasattr(out, "shape") or out.shape[1] != len(all_assets):
                return name, ("empty")
            return name, out
        except Exception as e:
            return name, f"{type(e).__name__}: {str(e)[:80]}"

    if args.threads > 1:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = {ex.submit(work, n): n for n in names}
            for fut in cf.as_completed(futs):
                name, res = fut.result()
                if isinstance(res, pd.DataFrame):
                    res.to_parquet(os.path.join(POOL, f"{name}.parquet"))
                    ok.append(name)
                else:
                    failed[name] = res
                if len(ok)+len(failed) % 50 == 0:
                    plog(f"  {len(ok)+len(failed)}/{len(names)} ok={len(ok)} fail={len(failed)} {time.time()-t0:.0f}s")
    else:
        # 顺序执行 + 每 50 个 gc，单面板常驻内存不该凭空爆炸
        for i, name in enumerate(names, 1):
            name_, res = work(name)
            if isinstance(res, pd.DataFrame):
                res.to_parquet(os.path.join(POOL, f"{name}.parquet"))
                ok.append(name)
            else:
                failed[name] = res
            if i % 50 == 0:
                gc.collect()
                plog(f"  {i}/{len(names)} ok={len(ok)} fail={len(failed)} {time.time()-t0:.0f}s")

    plog(f"完成: ok={len(ok)} fail={len(failed)} 总耗时{time.time()-t0:.0f}s")
    for n, e in failed.items():
        plog(f"  FAIL {n}: {e}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()