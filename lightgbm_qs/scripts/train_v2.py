# -*- coding: utf-8 -*-
"""并行滚动前向 LightGBM v2 — 4 worker, 训练期 4 年, early-stopping, 用补全后的 158 因子。
用法: python train_v2.py <slice 0..3>; 主进程(无参)合并 predictions.parquet。
- 首切 = 2016 + 48 个月 = 2020-01-04（训练期 4 年）
- 每折: expanding window, 用该折训练集末段 10% 做 early-stopping 验证
- 参数: leaves 127, lr 0.03, min_data 30, feature_fraction 0.8, bagging 0.8, up to 600 rounds
"""
import os, sys, time, gc, pandas as pd, numpy as np, lightgbm as lgb

ROOT="/home/sunhaiwei/quant_projects/lightgbm_qs"
N_WORKERS=4; ROLL_MONTHS=3; TRAIN_MONTHS=48; MAX_ROUND=600
PARAMS=dict(objective="regression",metric="l2",learning_rate=0.03,num_leaves=127,
            min_data_in_leaf=30,feature_fraction=0.8,bagging_fraction=0.8,
            bagging_freq=1,verbosity=-1,device="cpu",num_threads=8)

def log(m): print(m,flush=True)

def load_data():
    df=pd.read_parquet(f"{ROOT}/data/build/features_full_filled.parquet")
    df["date"]=pd.to_datetime(df["date"])
    fwd=pd.read_parquet(f"{ROOT}/data/panel/fwd_ret10.parquet"); fwd.index=pd.to_datetime(fwd.index).date
    fl=fwd.stack().rename("fwd").reset_index(); fl.columns=["date","asset","fwd"]
    df["date"]=df["date"].dt.date
    df=df.merge(fl,on=["date","asset"],how="left"); df["date"]=pd.to_datetime(df["date"])
    df=df.sort_values(["date","asset"]).reset_index(drop=True)
    FEAT=[c for c in df.columns if c not in ("date","asset","fwd")]
    return df,FEAT

def rolls():
    START=pd.Timestamp("2016-01-04"); fc=START+pd.DateOffset(months=TRAIN_MONTHS)
    r=[]; c=fc
    while c<=pd.Timestamp("2026-08-10"): r.append(c); c+=pd.DateOffset(months=ROLL_MONTHS)
    return r

def run_slice(sl):
    df,FEAT=load_data(); rs=rolls(); dates=df["date"]
    lo=(sl*len(rs))//N_WORKERS; hi=((sl+1)*len(rs))//N_WORKERS
    out=[]
    for i in range(lo,hi):
        cut=rs[i]; nxt=cut+pd.DateOffset(months=ROLL_MONTHS)
        train=df[dates<cut]; oos=df[(dates>=cut)&(dates<nxt)]
        if len(oos)<100: continue
        Xt=train[FEAT].values.astype(np.float32); yt=train["fwd"].values.astype(np.float32)
        k=~np.isnan(yt); Xt,yt=Xt[k],yt[k]
        if len(Xt)<5000: continue
        # early-stopping: 训练集末 10% 做验证
        nv=max(2000,int(len(Xt)*0.1)); Xv,yv=Xt[-nv:],yt[-nv:]; Xt,yt=Xt[:-nv],yt[:-nv]
        t0=time.time()
        m=lgb.train(PARAMS,lgb.Dataset(Xt,yt),num_boost_round=MAX_ROUND,
                    valid_sets=[lgb.Dataset(Xv,yv)],callbacks=[lgb.early_stopping(30,verbose=False)])
        Xo=oos[FEAT].values.astype(np.float32)
        pred=m.predict(Xo,num_iteration=m.best_iteration or 100)
        o=oos[["date","asset"]].copy(); o["pred"]=pred; out.append(o)
        log(f"[w{sl}][{i+1}/{len(rs)}] cut={cut.date()} train={len(Xt)} oos={len(o)} iters={m.best_iteration or 100} {time.time()-t0:.0f}s")
        del Xt,yt,Xv,yv,Xo,m,o; gc.collect()
    if out:
        res=pd.concat(out,ignore_index=True)
        res.to_parquet(f"{ROOT}/data/build/preds_v2_slice_{sl}.parquet")
        log(f"[w{sl}] done {len(res)} -> preds_v2_slice_{sl}.parquet")

if __name__=="__main__":
    if len(sys.argv)>1: run_slice(int(sys.argv[1]))
    else:
        import glob
        parts=[pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/build/preds_v2_slice_*.parquet"))]
        pred=pd.concat(parts,ignore_index=True)
        pred.to_parquet(f"{ROOT}/data/build/predictions.parquet")
        log(f"merged {len(pred)} -> predictions.parquet  {pred.date.min().date()}..{pred.date.max().date()}")
