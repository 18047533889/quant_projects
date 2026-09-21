#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 state.json 里挖历史 repair 记录，回答：从 2015-01-05 重落值到底能不能修好 warmup。"""
import json, os, glob, re
from collections import Counter

WORK = "/home/sunhaiwei/quant_projects/work/newmining_20260919"
state = json.load(open(f"{WORK}/state.json"))["factors"]

reps = {k: v["repair"] for k, v in state.items() if isinstance(v, dict) and v.get("repair")}
print("有 repair 记录的因子数:", len(reps))
print("kind 分布:", Counter(r.get("kind") for r in reps.values()))
print("improved 分布:", Counter(bool(r.get("improved")) for r in reps.values()))
print()
print("=== kind=warmup 的 repair 结果 ===")
for k, r in reps.items():
    if r.get("kind") == "warmup":
        st = state[k]
        print(f"  {k[-24:]:24s} round={r.get('round')} improved={r.get('improved')} status={st.get('status')}")
        print(f"      reason={str(st.get('reason'))[:200]}")
print()
print("=== 现在还有 warmup 类不可用的（读 manifest） ===")
def finfo(name):
    for p in [f"{WORK}/factors/{name}/report_manifest.json"] + \
             glob.glob(f"{WORK}/shards/*/factors/{name}/report_manifest.json"):
        if os.path.exists(p):
            try:
                return json.load(open(p)).get("factors", {}).get(name, {})
            except Exception:
                pass
    return {}

warm, other = [], []
for k, v in state.items():
    if not isinstance(v, dict) or v.get("status") != "unavailable":
        continue
    f = finfo(k)
    reason = str(f.get("reason") or v.get("reason") or "")
    m = re.search(r"usable=(\d{4}-\d{2}-\d{2})", reason)
    if m:
        warm.append((k, m.group(1), reason))
    else:
        other.append((k, reason))
print(f"warmup 型 n={len(warm)}")
for k, d, r in warm:
    print(f"  {k[-24:]:24s} usable_start={d}  repair={state[k].get('repair')}")
print(f"\n其余 unavailable n={len(other)}")
for k, r in other:
    print(f"  {k[-24:]:24s} {r[:140]}")
