# -*- coding: utf-8 -*-
"""Step 2: frequency-weighted backend coverage worklist.

Joins the backend readiness matrix with the R57c operator frequency and produces
evidence/factor_catalog_20260916/backend_coverage_worklist.json:

  operator | rows (catalog rows containing it) | current backend class
           | convertible? | basis (evidence) | planned conversion

Convertibility is decided from EVIDENCE, not name heuristics:
  * the registered polars slot's ``source`` (registry backend_meta)
  * a RUNTIME marshal probe: does executing the polars kernel call
    ``pl.DataFrame.to_pandas``?  (patched counter, real data) -> a kernel that
    never marshals has a genuine polars/numpy implementation.

Coverage is reported two ways:
  * occurrence-weighted: cumulative operator-occurrences / total occurrences
  * row-gated (the metric that matters): fraction of the 113893 factor rows all
    of whose operators are covered -> 100% only when the whole row can run.
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import statistics
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

REPO = "/home/sunhaiwei/quant_projects"
CATALOG = f"{REPO}/evidence/factor_catalog_20260916/r57c_20260919_final/factor_catalog_review_r57c_final.csv.gz"
FREQ = f"{REPO}/evidence/factor_catalog_20260916/r57c_operator_frequency.json"
DATA = os.path.expanduser("~/cos_data/StockDailyBarAdj")
OUT = f"{REPO}/evidence/factor_catalog_20260916/backend_coverage_worklist.json"
PROBE_TOP_N = int(os.environ.get("PROBE_TOP_N", "60"))
PROBE_SYMS = int(os.environ.get("PROBE_SYMS", "10"))
PROBE_DAYS = os.environ.get("PROBE_DAYS", "2021-01-01:2022-06-30")

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.backend.polars_backend_kind import (  # noqa: E402
    get_physical_spec,
    _kernel_is_delegate,
    _module_is_delegate,
    _source_is_delegate,
)
from factor_engine.backend.contracts import ExecutionKind  # noqa: E402

load_all()

EK = {
    ExecutionKind.POLARS_PANDAS_DELEGATE.value: "polars_pandas_delegate",
    ExecutionKind.POLARS_NUMPY_KERNEL.value: "polars_numpy_kernel",
    ExecutionKind.POLARS_NATIVE_EXPR.value: "polars_native_expr",
    ExecutionKind.DUCKDB_NATIVE_SQL.value: "sql_native",
    ExecutionKind.SQL_PYTHON_UDF.value: "sql_python_udf",
    ExecutionKind.PANDAS_REFERENCE.value: "pandas_reference",
    ExecutionKind.UNSUPPORTED.value: "unsupported",
}
ALREADY_REAL = {"polars_native_expr", "polars_numpy_kernel", "sql_native"}

CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


# --------------------------------------------------------------------------
# per-factor-row operator sets
# --------------------------------------------------------------------------
def parse_rows():
    df = pd.read_csv(CATALOG, usecols=["id", "r57_formula", "ops_call_count",
                                       "compile_status"])
    rows = []
    occ = collections.Counter()
    for fid, formula, n_call in zip(df["id"], df["r57_formula"], df["ops_call_count"]):
        ops = set(CALL_RE.findall(str(formula)))
        rows.append((fid, ops))
        for o in ops:
            occ[o] += 1
    return df, rows, occ


# --------------------------------------------------------------------------
# runtime marshal probe
# --------------------------------------------------------------------------
_PROBE = {"to_pandas_calls": 0}


def install_probe():
    orig = pl.DataFrame.to_pandas

    def counting(self, *a, **k):
        _PROBE["to_pandas_calls"] += 1
        return orig(self, *a, **k)

    pl.DataFrame.to_pandas = counting
    return orig


POOL = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "Volume", "AdjAmount",
        "Return", "AdjVwap"]
NAME_MAP = {
    "x": "AdjClose", "y": "AdjClose", "z": "AdjClose", "price": "AdjClose",
    "close": "AdjClose", "feature": "AdjClose", "series": "AdjClose",
    "signal": "AdjClose", "value": "AdjClose", "ret": "Return",
    "open": "AdjOpen", "high": "AdjHigh", "low": "AdjLow",
    "activity": "Volume", "volume": "Volume", "vol": "Volume",
    "amount": "AdjAmount", "vwap": "AdjVwap",
}
SECOND_EXPLANATORY = {"cs_quantile_resid", "cs_spline_resid", "ts_beta",
                      "cs_isotonic_residual"}


def load_probe_panels():
    lo, hi = PROBE_DAYS.split(":")
    files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    files = [f for f in files if lo <= os.path.basename(f)[:10] <= hi]
    syms = (pl.read_parquet(files[-1]).sort("AdjAmount", descending=True)
              .head(PROBE_SYMS)["Symbol"].to_list())
    long = (pl.scan_parquet(files).select(["TradeDate", "Symbol", *POOL])
              .filter(pl.col("Symbol").is_in(syms)).collect())
    p = {}
    for c in POOL:
        w = (long.select(["TradeDate", "Symbol", c])
                 .pivot(values=c, index="TradeDate", on="Symbol").sort("TradeDate"))
        pdf = w.to_pandas().set_index("TradeDate")
        pdf.index.name = "date"
        p[c] = pdf
    return p


def bind(canonical, md, panels):
    pn = list(getattr(md, "param_names", []) or [])
    ps = getattr(md, "param_specs", None) or {}
    frames = []
    for nm in pn:
        if nm in ps:
            break
        frames.append(nm)
    if not frames:
        return None
    if canonical == "composition_normalized_entropy":
        return [panels[c] for c in POOL[:8]]
    if canonical in SECOND_EXPLANATORY and len(frames) >= 2:
        return [panels["AdjClose"], panels["Volume"]]
    out = []
    for i, nm in enumerate(frames):
        if nm in NAME_MAP:
            col = NAME_MAP[nm]
        elif nm in ("period_end", "period_id", "report_date"):
            col = "AdjClose"
        elif nm.startswith("x") and nm[1:].isdigit():
            col = POOL[(int(nm[1:]) - 1) % len(POOL)]
        else:
            col = POOL[i % len(POOL)]
        out.append(panels[col])
    return out


def fill_kwargs(ps):
    ov = {"order": 3, "delay": 1, "normalize": True,
          "require_consecutive": True, "min_patterns": 20}
    MISS = object()
    try:
        from factor_engine.backend.contracts import MISSING as MISS  # type: ignore
    except Exception:
        pass
    kw = {}
    for name, spec in (ps or {}).items():
        if name in ov:
            kw[name] = ov[name]
            continue
        d = getattr(spec, "default", MISS)
        if d is not MISS and d is not None and not isinstance(d, type(MISS)):
            kw[name] = d
            continue
        if d is not MISS and d is not None:
            kw[name] = d
            continue
        dt, lo = getattr(spec, "dtype", None), (getattr(spec, "min", None) or 1)
        ch = getattr(spec, "choices", None)
        if ch:
            kw[name] = ch[0]
        elif dt is bool:
            kw[name] = True
        elif dt is int:
            kw[name] = max(int(lo), 20)
        elif dt is float:
            kw[name] = max(float(lo), 0.5)
    return kw


def main():
    print("parsing catalog rows ...", flush=True)
    df, rows, occ = parse_rows()
    freq_list = json.load(open(FREQ))
    freq = {d["operator"]: d["rows"] for d in freq_list}
    total_rows = len(rows)
    total_occ = sum(occ.values())
    print(f"catalog rows={total_rows} distinct_ops_in_formula={len(occ)} "
          f"total_operator_occurrences={total_occ}", flush=True)

    # validate that the frequency file counts the same thing we do
    common = [o for o in freq if o in occ]
    agree = sum(1 for o in common if freq[o] == occ[o])
    exact = [(o, freq[o], occ[o]) for o in common if freq[o] != occ[o]]
    val = {"freelist_ops": len(freq), "parsed_ops": len(occ),
           "common": len(common), "identical_counts": agree,
           "mismatch_examples": exact[:10], "mismatch_n": len(exact),
           "freq_sum": sum(freq.values()), "parsed_sum": total_occ,
           "parsed_in_freqlist_sum": sum(occ[o] for o in common)}
    print("VALIDATION", json.dumps(val), flush=True)

    # classification of every operator that appears in the catalog
    cls = {}
    evidence = {}
    cat = OperatorRegistry._catalog
    for op in sorted(occ):
        canonical = op
        if canonical not in cat:
            cls[canonical] = "not_in_catalog"
            evidence[canonical] = {"reason": "operator name not registered"}
            continue
        entry = cat[canonical]
        backends = tuple(sorted(map(str, entry.get("backends") or [])))
        bmeta = dict(entry.get("backend_meta") or {})
        pm = dict(bmeta.get("polars") or {})
        src = str(pm.get("source") or "")
        ev = {"backends": backends, "polars_source": src}
        if "polars" not in backends:
            cls[canonical] = "no_polars_slot"
            evidence[canonical] = {**ev, "reason": "registry has no polars backend"}
            continue
        p = OperatorRegistry.get(canonical, "polars", mode="any")
        if p is None:
            cls[canonical] = "polars_slot_unresolvable"
            evidence[canonical] = {**ev, "reason": "polars backend not resolvable"}
            continue
        spec = get_physical_spec(p)
        if spec is None:
            cls[canonical] = "no_explicit_physical_spec"
        else:
            ek = getattr(spec.execution_kind, "value", spec.execution_kind)
            cls[canonical] = EK.get(ek, "other:" + str(ek))
        try:
            ev["kernel_roundtrips_pandas"] = bool(_kernel_is_delegate(p))
        except Exception as e:
            ev["kernel_roundtrips_pandas"] = None
            ev["kernel_probe_error"] = f"{type(e).__name__}"
        ev["source_marker_is_delegate"] = bool(_source_is_delegate(src))
        ev["module_marker_is_delegate"] = bool(_module_is_delegate(p))
        ev["declared_execution_kind"] = (
            getattr(spec.execution_kind, "value", None) if spec is not None else None)
        evidence[canonical] = ev

    # ---- runtime marshal probe over the highest-frequency ops ----
    print("loading probe panels ...", flush=True)
    panels = load_probe_panels()
    top_for_probe = [o for o, _ in sorted(occ.items(), key=lambda x: -x[1])[:PROBE_TOP_N]]
    orig_to_pandas = install_probe()
    try:
        probe_res = {}
        for op in top_for_probe:
            r = {"probe": "not_attempted"}
            p = OperatorRegistry.get(op, "polars", mode="any")
            if p is None:
                probe_res[op] = {"probe": "no_polars_slot"}
                continue
            md = getattr(p, "metadata", None)
            frames = bind(op, md, panels)
            if not frames:
                probe_res[op] = {"probe": "unbindable"}
                continue
            kw = fill_kwargs(getattr(md, "param_specs", None) or {})
            try:
                pl_args = []
                for f in frames:
                    plf = pl.from_pandas(f)
                    f2 = f.copy()
                    f2.index.name = "date"
                    pl_args.append(pl.from_pandas(f2))
                _PROBE["to_pandas_calls"] = 0
                t0 = time.perf_counter()
                out = p._calculate_series(*pl_args, **kw)
                dt = time.perf_counter() - t0
                r = {"probe": "ok", "to_pandas_calls": _PROBE["to_pandas_calls"],
                     "seconds": dt, "n_frames": len(frames),
                     "out_rows": int(len(out)) if hasattr(out, "__len__") else None}
                if isinstance(out, pl.DataFrame):
                    num = out.select([c for c in out.columns if c != "date"]).to_numpy()
                    r["finite_frac"] = float(np.isfinite(num.astype(float)).mean())
            except Exception as e:
                r = {"probe": "error", "error": f"{type(e).__name__}: {str(e)[:180]}"}
            probe_res[op] = r
    finally:
        pl.DataFrame.to_pandas = orig_to_pandas

    # ---------------- verdicts ----------------
    def verdict(op):
        c = cls.get(op, "not_in_catalog")
        ev = evidence.get(op, {})
        probe = probe_res.get(op, {})
        rt = ev.get("kernel_roundtrips_pandas")
        pcall = probe.get("to_pandas_calls")
        if c in ALREADY_REAL:
            return ("already_real", "already_real",
                    "已声明真 polars/numpy/SQL 执行类型 (execution_kind=%s)" % c)
        if c in ("no_polars_slot", "polars_slot_unresolvable", "not_in_catalog"):
            return ("needs_inspection", "blocked",
                    "注册表无可用 polars 槽位 (source=%s)" % ev.get("polars_source"))
        if c == "no_explicit_physical_spec":
            if rt is False and ev.get("source_marker_is_delegate") is False \
                    and ev.get("module_marker_is_delegate") is False:
                note = "静态检查: 内核无 to_pandas()/_pl_to_pd 往返"
                if pcall == 0:
                    note += "; 运行时探针: 执行零次 pl.to_pandas 调用"
                elif pcall is not None:
                    note += "; 运行时探针: %d 次 to_pandas 调用(仍有往返)" % pcall
                return ("declare_spec_only", "native_expr_or_numpy_kernel",
                        note + " -> 只需补 _physical_spec 声明即可接入生产 polars_long")
            if pcall == 0:
                return ("declare_spec_only", "native_expr_or_numpy_kernel",
                        "运行时探针: 零次 pl.to_pandas 调用 -> 真实现, 补声明即可")
            return ("rewrite_needed", "pandas_free_rewrite",
                    "内核经 pandas 往返 (%s) -> 需改写为 polars expr 或 numpy 列内核"
                    % ev.get("declared_execution_kind"))
        if c in ("polars_pandas_delegate", "other:delegate_python"):
            return ("rewrite_needed", "pandas_free_rewrite",
                    "当前 polars 槽为 pl->pd->pl 委托 (source=%s); 已实测委托倍率见报告"
                    % ev.get("polars_source"))
        return ("needs_inspection", "unknown",
                "未识别的执行类型 %s" % c)

    probs = []
    for op, cnt in occ.items():
        v, way, basis = verdict(op)
        probs.append({
            "operator": op,
            "rows": cnt,
            "rows_in_freq_file": freq.get(op),
            "category": cls.get(op, "not_in_catalog"),
            "convertible_to": v,
            "planned_way": way,
            "basis": basis,
            "evidence": evidence.get(op, {}),
            "runtime_probe": probe_res.get(op),
        })
    probs.sort(key=lambda x: (-x["rows"], x["operator"]))

    # ---------------- coverage curves ----------------
    # conversion work = everything not already real
    work = [p for p in probs if p["convertible_to"] != "already_real"]
    work.sort(key=lambda x: (-x["rows"], x["operator"]))

    cum = 0
    occ_curve = []
    for i, p in enumerate(work, 1):
        cum += p["rows"]
        if i in (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000,
                 len(work)):
            occ_curve.append({"n_operators": i, "cum_rows": cum,
                              "pct_of_total_occurrences": round(100.0 * cum / total_occ, 2)})

    # row-gated: a factor row runs only if ALL of its operators are covered
    # baseline = operators already real
    def gate(covered_ops):
        ok = 0
        for _fid, ops in rows:
            if ops and ops <= covered_ops:
                ok += 1
        return ok

    already = {p["operator"] for p in probs if p["convertible_to"] == "already_real"}
    baseline_ok = gate(already)
    row_curve = [{"n_operators_converted": 0, "rows_that_run": baseline_ok,
                  "pct_of_catalog_rows": round(100.0 * baseline_ok / total_rows, 2)}]
    cov = set(already)
    for i, p in enumerate(work, 1):
        cov.add(p["operator"])
        if i in (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000,
                 len(work)):
            k = gate(cov)
            row_curve.append({"n_operators_converted": i, "rows_that_run": k,
                              "pct_of_catalog_rows": round(100.0 * k / total_rows, 2)})
    # full gate at the end
    if row_curve[-1]["n_operators_converted"] != len(work):
        k = gate(cov)
        row_curve.append({"n_operators_converted": len(work), "rows_that_run": k,
                          "pct_of_catalog_rows": round(100.0 * k / total_rows, 2)})

    # where does 50/80/90/95% of rows-that-run land?
    cov_all = set(already)
    landmarks = {}
    for i, p in enumerate(work, 1):
        cov_all.add(p["operator"])
        if i in (50, 100, 200, 300, 400, 500, 600, 700, 800) or i == len(work):
            k = gate(cov_all)
            landmarks[i] = round(100.0 * k / total_rows, 2)

    by_cat = collections.Counter(p["category"] for p in probs)
    by_conv = collections.Counter(p["convertible_to"] for p in probs)
    work_rows = sum(p["rows"] for p in work)

    # ---- counterfactual: what does "delete the delegate slots" actually cost? ----
    # If supports_polars becomes False for every delegate op, a factor that uses
    # ANY of them cannot run on the polars_long backend at all.
    delegate_ops = {p["operator"] for p in probs
                    if p["category"] in ("polars_pandas_delegate",
                                         "other:delegate_python")}
    rows_with_delegate = sum(1 for _fid, ops in rows if ops & delegate_ops)
    rows_blocked_by_delegate_only = sum(
        1 for _fid, ops in rows
        if (ops & delegate_ops) and (ops - delegate_ops) <= already)
    counterfactual = {
        "delegate_operators_in_catalog": len(delegate_ops),
        "delegate_occurrences": sum(p["rows"] for p in probs
                                    if p["operator"] in delegate_ops),
        "factor_rows_containing_a_delegate_op": rows_with_delegate,
        "pct_factor_rows_containing_a_delegate_op": round(
            100.0 * rows_with_delegate / total_rows, 2),
        "factor_rows_that_would_become_unsupported_on_polars_long": rows_with_delegate,
        "note": "这些行本身不因删除而错误, 但在 polars_long 上会 UNSUPPORTED -> "
                "回退 pandas 参考路径, 丢掉 polars_long 的整体加速",
    }
    print("COUNTERFACTUAL", json.dumps(counterfactual), flush=True)

    out = {
        "generated_from": {
            "catalog": CATALOG, "frequency": FREQ,
            "classification": "runtime registry + PhysicalImplementationSpec",
            "probe_panel": {"days": PROBE_DAYS, "symbols": PROBE_SYMS},
        },
        "totals": {
            "catalog_factor_rows": total_rows,
            "distinct_operators_in_formulas": len(occ),
            "total_operator_occurrences": total_occ,
            "operators_already_real": len(already),
            "operators_needing_work": len(work),
            "occurrences_already_real": sum(p["rows"] for p in probs
                                            if p["convertible_to"] == "already_real"),
            "occurrences_needing_work": work_rows,
            "factor_rows_that_run_today": baseline_ok,
            "pct_factor_rows_that_run_today": round(100.0 * baseline_ok / total_rows, 2),
        },
        "frequency_file_validation": val,
        "category_counts": dict(by_cat),
        "convertibility_counts": dict(by_conv),
        "delegate_deletion_counterfactual": counterfactual,
        "coverage_occurrence_weighted": occ_curve,
        "coverage_row_gated": row_curve,
        "row_gated_landmarks": landmarks,
        "operators": probs,
        "workqueue_by_frequency": [
            {k: p[k] for k in ("operator", "rows", "category", "convertible_to",
                               "planned_way", "basis")} for p in work],
    }
    with gzip.open(OUT + ".gz", "wt") if False else open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print(json.dumps({k: out[k] for k in
                      ("totals", "category_counts", "convertibility_counts",
                       "row_gated_landmarks")}, indent=1), flush=True)
    print("ROWS THAT RUN TODAY (all ops real):", baseline_ok, "/", total_rows,
          "=", round(100.0 * baseline_ok / total_rows, 2), "%", flush=True)
    print("\nOCCURRENCE-WEIGHTED coverage after converting top N:", flush=True)
    for c in occ_curve:
        print(f"   top {c['n_operators']:5d} ops -> {c['pct_of_total_occurrences']:6.2f}% "
              f"of occurrences", flush=True)
    print("\nROW-GATED coverage (rows that fully run) after converting top N:",
          flush=True)
    for c in row_curve:
        print(f"   top {c['n_operators_converted']:5d} ops -> "
              f"{c['pct_of_catalog_rows']:6.2f}% of catalog rows "
              f"({c['rows_that_run']})", flush=True)
    print("\nTOP 30 WORK ITEMS:", flush=True)
    for p in work[:30]:
        print(f"   {p['operator']:44s} rows={p['rows']:6d} {p['category']:28s} "
              f"{p['convertible_to']}", flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
