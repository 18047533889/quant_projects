# -*- coding: utf-8 -*-
"""分片批量重算 LQTP 因子（内存安全版）。从 SHARD 序号开始跑到 end。

用法（每进程跑一段，多个并行）:
  python3.12 recompute_lqtp_adj_batch.py --start 0 --end 250
  ... --start 250 --end 500 ...
用 nohup 多开几个进程分别跑，每片独立日志。
"""
import os, sys, json, time, argparse, glob
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lqtp_dsl_eval import build_panels, eval_formula, OPS

OHLCV = f"{ROOT}/data/build/ohlcv_adj.parquet"
POOL = f"{ROOT}/data/factor_pools/lqtp"
FORMULA_MAP = f"{ROOT}/data/build/lqtp_formula_map.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=250)
    ap.add_argument("--log", default=f"/tmp/recompute_shard.log")
    args = ap.parse_args()
    LOG = args.log
    plog = lambda *a: (print(*a, flush=True), open(LOG, "a").write(" ".join(map(str, a)) + "\n"))

    t0 = time.time()
    plog(f"[{time.strftime('%H:%M:%S')}] load panels...")
    ohlcv = pd.read_parquet(OHLCV)
    P = build_panels(ohlcv)
    # 只保留池子原列的 297+ 资产? 直接用全部资产(5460)算, 落盘时对齐池列
    # 为了内存, 限到池子有值的 297? 池 wide 是 297 列
    trad = pd.read_parquet('/tmp/tradable_assets.parquet')['0'].tolist() if False else list(pd.read_parquet('/tmp/tradable_assets.parquet').iloc[:,0])
    for k in P:
        if isinstance(P[k], pd.DataFrame):
            P[k] = P[k].reindex(columns=trad)
    plog(f"  restricted to {len(trad)} tradable assets")
    formula_map = json.load(open(FORMULA_MAP))
    names = list(formula_map.keys())[args.start:args.end]
    plog(f"shard {args.start}:{args.end} n={len(names)}")

    ok = []; failed = {}
    for i, name in enumerate(names):
        try:
            out = eval_formula(formula_map[name]["formula"], P, OPS)
            if not hasattr(out, "shape") or out.shape[1] != len(P["close"].columns):
                # result was a Series (e.g. wzr_006 cs_regression) -> 填 NaN??
                failed[name] = "not-wide"
                continue
            out.to_parquet(os.path.join(POOL, f"{name}.parquet"))
            ok.append(name)
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {str(e)[:80]}"
        if (i + 1) % 20 == 0:
            plog(f"  {i+1}/{len(names)} ok={len(ok)} fail={len(failed)} {time.time()-t0:.0f}s")
    plog(f"shard done ok={len(ok)} fail={len(failed)}")
    for n, e in failed.items():
        plog(f"  FAIL {n}: {e}")

if __name__ == "__main__":
    main()