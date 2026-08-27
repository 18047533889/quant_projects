# -*- coding: utf-8 -*-
"""并发分片重算 LQTP 池因子（每片独立 python 进程，各自 fork 共享面板）。

用法: python3.12 scripts/recompute_chunk.py <chunk_json> <log>
每个分片按名字列表重算，池文件 mtime 更新后即可被后续读取。输出 5460 列宽面板。
"""
import os, sys, json, time
import pandas as pd
import numpy as np

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lqtp_dsl_eval import build_panels, eval_formula, OPS

OHLCV = os.path.join(ROOT, "data/build/ohlcv_adj.parquet")
POOL = os.path.join(ROOT, "data/factor_pools/lqtp")
FMAP = os.path.join(ROOT, "data/build/lqtp_formula_map.json")


def main():
    chunk_f = sys.argv[1]
    log = sys.argv[2]
    names = json.load(open(chunk_f))
    formula_map = json.load(open(FMAP))
    fh = open(log, "a")

    def plog(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        fh.write(line + "\n")

    t0 = time.time()
    ohlcv = pd.read_parquet(OHLCV)
    P = build_panels(ohlcv)
    plog(f"[{os.getpid()}] panels {P['close'].shape} n={len(names)} t={time.time()-t0:.0f}s")
    ok = 0
    for i, name in enumerate(names, 1):
        t1 = time.time()
        try:
            out = eval_formula(formula_map[name]["formula"], P, OPS)
            if not hasattr(out, "shape") or out.shape[1] != P["close"].shape[1]:
                plog(f"  FAIL {name} not-wide")
                continue
            out.to_parquet(os.path.join(POOL, f"{name}.parquet"))
            ok += 1
        except Exception as e:
            plog(f"  FAIL {name} {type(e).__name__}: {str(e)[:80]}")
        if i % 10 == 0 or (time.time() - t1) > 300:
            plog(f"  [{os.getpid()}] {i}/{len(names)} ok={ok} t={time.time()-t0:.0f}s last={time.time()-t1:.0f}s")
    plog(f"[{os.getpid()}] DONE ok={ok} total={time.time()-t0:.0f}s")
    fh.close()


if __name__ == "__main__":
    main()
