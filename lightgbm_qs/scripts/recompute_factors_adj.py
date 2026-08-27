# -*- coding: utf-8 -*-
"""用后复权 OHLCV 重算 456 公式中可重算的纯价格因子(238个)。

输入: data/build/ohlcv_adj.parquet (后复权 Open/High/Low/Close/Vwap + 不复权 Volume + 后复权 Amount)
公式: /home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json (456 个, 含 code)
输出: data/build/factors_recomputed/ 每因子一个 wide parquet (date x asset)

只重算"纯价格"公式(只用 open/high/low/close/volume/amount/vwap/ret/range/up/down/abs_ret/TradingDay,
不依赖基本面列 pb_lf/pe_ttm/ps_ttm/mkt_cap/debttoassets 等, 不依赖 talib/alpha_tools 库)。
需基本面/需库的公式跳过(保留原值)。

用法: python3.12 recompute_factors_adj.py
"""
import os, json, time, glob
import numpy as np
import pandas as pd

ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
OHLCV = os.path.join(ROOT, "data/build/ohlcv_adj.parquet")
FORMULA = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
OUT_DIR = os.path.join(ROOT, "data/build/factors_recomputed")
LOG = "/tmp/recompute_factors_adj.log"

# 纯价格公式允许的 df 列(不含基本面)
PRICE_COLS = {"open","high","low","close","volume","amount","vwap",
              "ret","range","up","down","abs_ret","TradingDay"}

def plog(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def build_asset_df(ohlcv, asset):
    sub = ohlcv[ohlcv.asset == asset].pivot_table(index="date", columns="field", values="value")
    sub = sub.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close",
                              "Volume":"volume","Amount":"amount","Vwap":"vwap"})
    sub["ret"] = sub["close"].pct_change()
    sub["range"] = sub["high"] - sub["low"]
    sub["up"] = (sub["close"] >= sub["open"]).astype(float)
    sub["down"] = (sub["close"] < sub["open"]).astype(float)
    sub["abs_ret"] = sub["ret"].abs()
    sub["TradingDay"] = sub.index
    return sub

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    plog(f"[{time.strftime('%H:%M:%S')}] 加载后复权 OHLCV ...")
    ohlcv = pd.read_parquet(OHLCV)
    assets = sorted(ohlcv["asset"].unique())
    plog(f"  assets={len(assets)}")

    d = json.load(open(FORMULA))
    plog(f"  公式总数={len(d)}")

    # 预编译每个公式, 判断是否纯价格
    compiled = []  # (page_name, factor_name, fn)
    for r in d:
        code = r.get("code")
        if not code:
            continue
        # 提取 code 用到的 df 列
        import re
        used = set()
        for m in re.finditer(r"df_copy\['(\w+)'\]|df\['(\w+)'\]", code):
            used.add(m.group(1) or m.group(2))
        # 是否依赖基本面列
        if not used.issubset(PRICE_COLS):
            continue
        # 是否依赖外部库
        if "talib" in code or "alpha_tools" in code:
            continue
        try:
            ns = {"np": np, "pd": pd}
            exec(code, ns)
            fn = ns.get(r["factor_name"])
            if fn is None:
                continue
            compiled.append((r["page_name"], r["factor_name"], fn))
        except Exception:
            continue
    plog(f"  可重算纯价格公式: {len(compiled)}")

    # 逐公式逐资产计算
    done = 0
    for page, fname, fn in compiled:
        frames = []
        for asset in assets:
            try:
                sub = build_asset_df(ohlcv, asset)
                out = fn(sub)
                if out is None:
                    continue
                s = pd.Series(out, index=sub.index, name=asset)
                frames.append(s)
            except Exception:
                continue
        if not frames:
            continue
        wide = pd.concat(frames, axis=1)
        wide.to_parquet(os.path.join(OUT_DIR, f"{page}.parquet"))
        done += 1
        if done % 20 == 0:
            plog(f"  {done}/{len(compiled)} 公式完成 {time.time()-t0:.0f}s")
    plog(f"[{time.strftime('%H:%M:%S')}] DONE 重算 {done} 个纯价格因子 total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    main()
