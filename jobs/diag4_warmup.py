#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1) 打印 warmup 类完整公式；2) 抽 3 个 usable_start 异常晚的，看矩阵 NaN 结构。"""
import json, os, re, glob
import numpy as np
import pandas as pd

WORK = "/home/sunhaiwei/quant_projects/work/newmining_20260919"
cands = {c["page_name"]: c for c in json.load(open(f"{WORK}/candidates.json"))["candidates"]}
state = json.load(open(f"{WORK}/state.json"))["factors"]

def finfo(name):
    for p in [f"{WORK}/factors/{name}/report_manifest.json"] + \
             glob.glob(f"{WORK}/shards/*/factors/{name}/report_manifest.json"):
        if os.path.exists(p):
            try:
                return json.load(open(p)).get("factors", {}).get(name, {})
            except Exception:
                pass
    return {}

targets = []
for k, v in state.items():
    if not isinstance(v, dict) or v.get("status") != "unavailable":
        continue
    f = finfo(k)
    r = str(f.get("reason") or v.get("reason") or "")
    m = re.search(r"usable=(\d{4})-(\d{2})-(\d{2})", r)
    if m:
        off = (pd.Timestamp(m.group(0)[7:]) - pd.Timestamp("2016-01-04")).days
        targets.append((off, k, str(cands.get(k, {}).get("fe_formula", ""))[:160]))

targets.sort()
print("=== 预热类按 usable_start 偏移天数排序 ===")
for off, k, fe in targets:
    flag = "  <<< 疑似非预热（真稀疏）" if off > 200 else ""
    print(f"{off:5d}d  {k[-24:]:24s} {fe}{flag}")

print("\n=== PlanParamError 明细 ===")
for k, v in state.items():
    if not isinstance(v, dict):
        continue
    f = finfo(k)
    r = str(f.get("reason") or "")
    if "PlanParamError" in r:
        print(f"  {k}\n     fe={cands.get(k,{}).get('fe_formula')}\n     lqtp={str(cands.get(k,{}).get('lqtp_formula'))[:200]}\n     reason={r}")

print("\n=== 退化样本（真坏） ===")
for k, v in state.items():
    if not isinstance(v, dict):
        continue
    f = finfo(k)
    r = str(f.get("reason") or "")
    if "未产生任何有效 RankIC" in r or "usable_days=0" in r:
        print(f"  {k}\n     fe={cands.get(k,{}).get('fe_formula')}\n     reason={r[:200]}")

print("\n=== 抽 3 个晚起矩阵看 NaN 结构 ===")
import sys
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
from factor_report_sources import resolve_raw_matrix, RAW_MATRIX_DIRS, _paths
for name in ["alphasage_20260911220613_bc370a5c", "alphasage_20260909062100_fe73d8fa",
             "alphasage_20260909062020_36a849b7", "alphasage_20260910155000_56c66ceb"]:
    print(f"\n--- {name}")
    for directory, path in _paths(name):
        d = pd.read_parquet(path)
        arr = d.to_numpy(dtype=float)
        finite = np.isfinite(arr).sum(axis=1)
        idx = pd.to_datetime(d.index)
        first = idx[np.argmax(finite >= 30)] if (finite >= 30).any() else None
        print(f"   dir={directory} shape={d.shape} idx={idx.min().date()}..{idx.max().date()} "
              f"first_row_ge30={first.date() if first is not None else None} "
              f"row0_finite={finite[0]} row100_finite={finite[100] if len(finite)>100 else '-'}")
    src = resolve_raw_matrix(name)
    print(f"   -> 选中 {src.source} full={src.is_full_window}")
