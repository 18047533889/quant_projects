# -*- coding: utf-8 -*-
"""Merge partial walk-forward manifest JSONs (one per factor subset) into the final
data/build/walkforward_selection_flip.json. Per-fold factor lists are unioned.
"""
import os, sys, glob, json
ROOT = "/home/sunhaiwei/quant_projects/lightgbm_qs"
BUILD = os.path.join(ROOT, "data/build")
parts = sorted(glob.glob(os.path.join(BUILD, "walkforward_selection_flip_part_*.json")))
if not parts:
    raise SystemExit("no part files")
merged = json.load(open(parts[0]))
cuts = merged["cuts"]
for p in parts[1:]:
    d = json.load(open(p))
    for cut, info in d["cuts"].items():
        cur = set(cuts.get(cut, {}).get("factors", []))
        cur |= set(info.get("factors", []))
        cuts.setdefault(cut, info)["factors"] = sorted(cur)
        # keep the densest sel info
        if info.get("n_sel_dates", 0) > cuts[cut].get("n_sel_dates", 0):
            cuts[cut]["sel_start"] = info["sel_start"]
            cuts[cut]["sel_end"] = info["sel_end"]
            cuts[cut]["n_sel_dates"] = info["n_sel_dates"]
out = os.path.join(BUILD, "walkforward_selection_flip.json")
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"merged {len(parts)} parts -> {out}, cuts={len(cuts)}")
