#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量盘点：177 候选 × (campaign, 状态, 不可用原因原文)"""
import json, os, re, glob
from collections import Counter, defaultdict
import pandas as pd

WORK = "/home/sunhaiwei/quant_projects/work/newmining_20260919"
cands = json.load(open(f"{WORK}/candidates.json"))["candidates"]
state = json.load(open(f"{WORK}/state.json"))["factors"]

def manifest(name):
    for pat in [f"{WORK}/factors/{name}/report_manifest.json",
                f"{WORK}/shards/*/factors/{name}/report_manifest.json",
                f"{WORK}/waves/*/*/{name}/report_manifest.json"]:
        hits = glob.glob(pat)
        if hits:
            try:
                return json.load(open(hits[0]))
            except Exception:
                return None
    return None

rows = []
for c in cands:
    n = c["page_name"]
    st = state.get(n, {}) if isinstance(state.get(n), dict) else {}
    m = manifest(n)
    finfo = (m or {}).get("factors", {}).get(n, {})
    reason = finfo.get("reason") or st.get("reason") or ""
    rows.append(dict(
        name=n,
        campaign=c.get("campaign", ""),
        mined_by=c.get("mined_by", ""),
        fe=c.get("fe_formula", ""),
        status=finfo.get("status") or st.get("status") or "not_landed",
        reason=str(reason),
        usable_days=(re.search(r"usable_days=(\d+)", str(reason)) or [None, None])[1],
    ))

df = pd.DataFrame(rows)
print("=== 按 campaign × status ===")
print(pd.crosstab(df["campaign"], df["status"]))
print()
print("=== campaign 汇总（生成时间） ===")
for camp, g in df.groupby("campaign"):
    d = camp[:8]
    print(f"  {camp} (20{d[:4]}-{d[4:6]}-{d[6:8]})  n={len(g)}  status={dict(Counter(g['status']))}")

print("\n=== 不可用原因归类 ===")
def klass(r):
    r = str(r)
    if not r:
        return "no-reason"
    if "broken landing, not warm-up" in r:
        return "leading-gap"
    if "usable_days=" in r:
        return "coverage:" + (re.search(r"usable_days=(\d+)", r) or ["","?"])[1]
    if "ResourceBudgetExceeded" in r:
        return "resource"
    if "PlanParamError" in r:
        return re.sub(r"[0-9.]+", "N", r[:110])
    return re.sub(r"[0-9]+", "N", r[:110])

df["klass"] = df["reason"].map(klass)
print(df["klass"].value_counts().to_string())

print("\n=== 明细（非 evaluated/below_gate） ===")
bad = df[~df["status"].isin(["evaluated", "below_gate"])]
for _, r in bad.sort_values(["klass", "campaign"]).iterrows():
    print(f"{r['status'][:11]:11s} {r['klass'][:52]:52s} {r['name'][-30:]:30s} {r['fe'][:70]}")
print("\n无可 user:")
print(df["status"].value_counts().to_string())
df.to_csv(f"{WORK}/inventory.csv", index=False)
print(f"\n-> {WORK}/inventory.csv")
