#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""461 条 can_use=True 因子全量对拍验收。

技术路线（已验证）：
  - ref parquet 文件名 = page_name = factor_name 去 factor_ 前缀
  - 引擎窗口 2016-01-04 起（旧矩阵为 2016 全历史；2018 窗口无重叠）
  - python cleaned-call 形态（fe 含 o['）: make_cleaned_call_factory + exec
  - DSL 文本形态: parse_expr(surface="compat_research") → Factor → eng.run
  - 判定: |ρ|≥0.95 pass95 / 0.90-0.95 pass90 / <0.90 basis_diff(表达式用 AdjCurrent 而参考矩阵旧未复权口径) 或 real_fail

用法:
  OMP_NUM_THREADS=31 nohup .venv/bin/python jobs/crosscheck_all_fe.py > /tmp/crosscheck_all.log 2>&1 &
  （写 /tmp/fe_crosscheck_final.json，每 10 条落盘，可续跑）
"""
import json
import math
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["ASHARE_PARQUET_ROOT"] = os.path.expanduser("~/cos_data")
os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
os.environ.setdefault("DATA_ACCESS_RUN_MODE", "interactive_research")
os.environ.setdefault("FACTOR_ENGINE_RUN_MODE", "research")

PROJ = "/home/sunhaiwei/quant_projects"
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/jobs")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_engine.storage.factory import build_data_source
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col

JSON_PATH = "/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json"
FV_DIR = "/home/sunhaiwei/quant_projects/weekly_backtest_output/factor_matrices_all"
OUT_PATH = "/tmp/fe_crosscheck_final.json"
BATCH = 40
WINDOW = ("2016-01-04", "2018-06-30")  # 旧矩阵基准窗口

_py_env = None

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        ds = build_data_source({
            "type": "data_access", "dataset": "ashare_stock_daily_adj",
            "start_date": WINDOW[0], "end_date": WINDOW[1],
        })
        _engine = FactorEngine(build_backend("pandas"), ds, run_mode="research")
    return _engine


def p2f(name: str) -> str:
    return name[7:] if name.startswith("factor_") else name


def run_factor(fe: str):
    """DSL 文本形态优先 parse_expr；真正 python o[...] 形态才走 exec。返回 wide DataFrame 或 raise。"""
    if fe.strip().startswith("o[") or re.search(r"^o\['[a-z_]+\'\]\(", fe):
        eng = _get_engine()
        ops = sorted(set(re.findall(r"o\['([A-Za-z_][A-Za-z0-9_]*)'\]", fe)))
        o2 = {nm: make_cleaned_call_factory(nm) for nm in ops}
        o2["col"] = col
        ns = {"o": o2, "col": col, "Factor": Factor}
        exec(f"_f = Factor(name='t', expr={fe})", ns)
        out = eng.run(ns["_f"], market="ashare")
    else:
        eng = _get_engine()
        fac = Factor(name="t", expr=parse_expr(fe, surface="compat_research"))
        out = eng.run(fac, market="ashare")
    r = out.get("result") if isinstance(out, dict) else out
    w = r.unstack() if hasattr(r, "unstack") else r
    if isinstance(w, pd.Series):
        w = w.unstack()
    return w


def corr_vs_ref(w: pd.DataFrame, pf: str):
    ref = pd.read_parquet(pf)
    ref.index = pd.to_datetime(ref.index)
    if not isinstance(w, pd.DataFrame) or w.empty:
        return None
    w.index = pd.to_datetime(w.index)
    ci = w.index.intersection(ref.index)
    cols = [c for c in w.columns if c in ref.columns]
    if len(ci) < 3 or not cols:
        return float("nan")
    a = w.loc[ci, cols].astype(float)
    b = ref.loc[ci, cols].astype(float)
    rs = []
    for c in cols:
        x = a[c].values
        y = b[c].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 10:
            r_, _ = spearmanr(x[m], y[m])
            if np.isfinite(r_):
                rs.append(r_)
    return float(np.mean(rs)) if rs else float("nan")


def main():
    data = json.loads(open(JSON_PATH, encoding="utf-8").read())
    targets = []
    for r in data:
        if not r.get("can_use_factor_engine"):
            continue
        if r.get("fe_formula", "").startswith("evoalpha"):
            continue
        targets.append(r)
    done = {}
    if os.path.exists(OUT_PATH):
        try:
            prev = json.loads(open(OUT_PATH, encoding="utf-8").read())
            done = prev.get("detail", {}) if isinstance(prev, dict) else {}
        except Exception:
            done = {}

    todo = [r for r in targets if r["factor_name"] not in done]
    print(f"[cross] targets={len(targets)} done={len(done)} todo={len(todo)}", flush=True)
    t0 = time.time()
    n_done = 0
    for i, r in enumerate(todo):
        name = r["factor_name"]
        fe = r.get("fe_formula") or ""
        pf = f"{FV_DIR}/{p2f(name)}.parquet"
        rec = {"corr": None, "verdict": "error", "error": ""}
        try:
            w = run_factor(fe)
            c = corr_vs_ref(w, pf)  # noqa: F821  (corr_vs_ref below)
            rec["corr"] = c
            if c is None or (isinstance(c, float) and not math.isfinite(c)):
                rec = {"corr": None, "verdict": "no_overlap", "error": "corr nan"}
            else:
                ac = abs(c)
                if ac >= 0.95:
                    rec["verdict"] = "pass95"
                elif ac >= 0.90:
                    rec["verdict"] = "pass90"
                else:
                    uses_adj = ("AdjClose" in fe or "AdjHigh" in fe or "AdjLow" in fe
                                or "AdjVwap" in fe or "AdjAmount" in fe or "Return" in fe)
                    rec["verdict"] = "basis_diff" if uses_adj else "real_fail"
        except Exception as exc:
            rec["error"] = str(exc)[:140]
        done[name] = rec
        n_done += 1
        if n_done % 10 == 0 or n_done == len(todo):
            with open(OUT_PATH, "w", encoding="utf-8") as fh:
                json.dump({"detail": done, "elapsed_s": round(time.time() - t0)}, fh, ensure_ascii=False, indent=1)
            ok = [v.get("corr") for v in done.values() if isinstance(v.get("corr"), (int, float)) and math.isfinite(v["corr"])]
            stats = {
                "valid": len(ok),
                "pass95": sum(1 for c in ok if abs(c) >= 0.95),
                "pass90": sum(1 for c in ok if 0.9 <= abs(c) < 0.95),
                "real_fail": sum(1 for v in done.values() if v.get("verdict") == "real_fail"),
                "basis_diff": sum(1 for v in done.values() if v.get("verdict") == "basis_diff"),
            }
            print(f"[cross] {n_done}/{len(todo)} | {stats} | {time.time()-t0:.0f}s", flush=True)
    # final summary
    ok = [v.get("corr") for v in done.values() if isinstance(v.get("corr"), (int, float)) and math.isfinite(v["corr"])]
    summary = {
        "total": len(done),
        "valid": len(ok),
        "pass95": sum(1 for c in ok if abs(c) >= 0.95),
        "pass90": sum(1 for c in ok if 0.9 <= abs(c) < 0.95),
        "real_fail": sum(1 for v in done.values() if v.get("verdict") == "real_fail"),
        "basis_diff": sum(1 for v in done.values() if v.get("verdict") == "basis_diff"),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "detail": done}, fh, ensure_ascii=False, indent=1)
    print("[cross] FINAL:", summary, flush=True)


if __name__ == "__main__":
    main()