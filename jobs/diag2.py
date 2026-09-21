#!/usr/bin/env python3
import json, os, re, glob
from collections import Counter, defaultdict

WORK = "/home/sunhaiwei/quant_projects/work/newmining_20260919"
state = json.load(open(os.path.join(WORK, "state.json")))["factors"]

un = {k: v for k, v in state.items() if isinstance(v, dict) and v.get("status") == "unavailable"}
fail = {k: v for k, v in state.items() if isinstance(v, dict) and v.get("status") == "failed"}
print("unavailable n=", len(un))
print("reason ->", Counter(str(v.get("reason"))[:80] for v in un.values()))
print()
for k, v in list(un.items())[:2]:
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "metrics"}, ensure_ascii=False)[:700])
print()
print("failed n=", len(fail))
print("failed reason ->", Counter(str(v.get("reason"))[:80] for v in fail.values()))
for k, v in list(fail.items())[:2]:
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "metrics"}, ensure_ascii=False)[:700])

# 扫描 run.log 提取错误行
print("\n=== unavailable run.log 错误模式 ===")
pat = Counter()
samples = defaultdict(list)
for k, v in un.items():
    rl = v.get("run_log")
    if not rl or not os.path.exists(rl):
        pat["<no run.log>"] += 1
        continue
    txt = open(rl, errors="replace").read()
    hits = []
    for line in txt.splitlines():
        low = line.lower()
        if any(s in low for s in ["error", "exception", "usable_days", "warmup", "exceed", "budget", "degrad", "raise"]):
            hits.append(line.strip()[:240])
    sig = None
    for h in hits:
        if "UsableDays" in h or "usable_days" in h:
            sig = "warmup:" + re.sub(r"\d+", "N", h)[:160]
            break
    if sig is None and hits:
        sig = re.sub(r"\d+", "N", hits[-1])[:160]
    if sig is None:
        sig = "<no error line>"
    pat[sig] += 1
    if len(samples[sig]) < 2:
        samples[sig].append((k, hits[-3:]))
for sig, n in pat.most_common(40):
    print(f"\n[{n}] {sig}")
    for k, hits in samples[sig]:
        print("   ", k)
        for h in hits:
            print("      ", h[:230])
