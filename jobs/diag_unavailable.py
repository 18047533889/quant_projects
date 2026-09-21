#!/usr/bin/env python3
"""诊断 61 个 unavailable 因子的精确原因：分类 + 公式 + 报文。"""
import json, os, re, sys, glob
from collections import Counter, defaultdict

WORK = "/home/sunhaiwei/quant_projects/work/newmining_20260919"

state = json.load(open(os.path.join(WORK, "state.json")))
factors = state["factors"]

def manifest_paths():
    out = []
    for p in glob.glob(os.path.join(WORK, "factors", "*", "report_manifest.json")):
        out.append(p)
    for p in glob.glob(os.path.join(WORK, "shards", "*", "factors", "*", "report_manifest.json")):
        out.append(p)
    for p in glob.glob(os.path.join(WORK, "waves", "*", "*", "report_manifest.json")):
        out.append(p)
    return out

def short(msg):
    if not msg:
        return ""
    m = str(msg)
    m = m.replace("\n", " | ")
    return m[:400]

buckets = defaultdict(list)
for name, entry in sorted(factors.items()):
    if not isinstance(entry, dict):
        continue
    status = entry.get("status")
    if status != "unavailable":
        continue
    reason = str(entry.get("reason") or entry.get("unavailable_reason") or "")
    msg = short(entry.get("message") or entry.get("error") or "")
    formula = entry.get("formula") or entry.get("expression") or ""
    buckets[reason].append((name, formula, msg))

print("=== unavailable 分类（按 reason 字段） ===")
for r, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"\n## {r}  n={len(items)}")
    for name, formula, msg in items[:200]:
        print(f"  - {name}\n      formula: {formula[:220]}\n      msg: {msg[:300]}")

print("\n=== 全部 status 统计 ===")
print(Counter(str(e.get("status")) for e in factors.values() if isinstance(e, dict)))

# 打印一个 entry 全字段样例
for name, entry in factors.items():
    if isinstance(entry, dict) and entry.get("status") == "unavailable":
        print("\n=== 样例 entry 字段 ===")
        print(name, json.dumps(entry, ensure_ascii=False, indent=2)[:2000])
        break
