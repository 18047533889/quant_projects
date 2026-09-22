# -*- coding: utf-8 -*-
"""Full-operator matrix with fork-isolated chunk workers.

Each chunk runs in a forked child; segfaults and hangs kill only the child,
the parent bisects the chunk down to single-op isolation. Records stream to
matrix_stream.jsonl. Baseline arrays (pandas) live in the parent and are
inherited by children via fork COW.
"""
import warnings, json, sys, os, time, hashlib
import multiprocessing as mp
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
import os as _pkg_os
_PKG_DIR = _pkg_os.path.dirname(_pkg_os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
import factor_engine.cleaned_operators  # noqa
from factor_engine.cleaned_operators import load_all
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    load_all()

# R65 harness note: the WIP resource broker's family-RSS walk hangs on this
# box; swap the function body in place so from-importers see the fast version.
import resource as _rusage
import factor_engine.runtime.resource_broker as _rbroker


def _fast_family_rss(pss: bool = False):
    # body must be self-contained: after the __code__ swap it executes with the
    # resource_broker module's globals, so resolve everything locally
    import resource as _r
    return (_r.getrusage(_r.RUSAGE_SELF).ru_maxrss * 1024, True)


_fam_attr = next((n for n in dir(_rbroker)
                  if n.startswith("_process_family_") and n.endswith("_rss")), None)
assert _fam_attr, "family rss fn missing"
_orig = getattr(_rbroker, _fam_attr)
try:
    _orig.__code__ = _fast_family_rss.__code__
    _orig.__defaults__ = (False,)
except (AttributeError, TypeError):
    setattr(_rbroker, _fam_attr, _fast_family_rss)

# R65 harness note: batch_service starts a process-level autopilot thread on
# the first run_many; forked children then inherit locks held by that thread
# and deadlock. Stub it out -- value tests do not need resource adaptation.
class _NoAutopilot:
    started = False

    def summary(self):
        return {"mode": "harness-disabled"}

    def last_decision(self):
        return None


def _noop_start_autopilot(broker=None, **kwargs):
    return _NoAutopilot()


import factor_engine.runtime.resource_autopilot_service as _apsvc

_apsvc.start_resource_autopilot = _noop_start_autopilot

import numpy as np
from factor_engine.backend.factory import build_backend
from factor_engine.api.dsl_parser import parse_factor
from factor_engine.runtime.engine import FactorEngine
from synthetic_source import SyntheticPanelSource

MANIFEST = json.load(open("/tmp/r65/op_manifest.json"))
BACKENDS = ("pandas", "polars", "duckdb_sql", "auto")
OUT = "/tmp/r65/matrix_stream.jsonl"
OP_TIMEOUT = 90
RTOL, ATOL = 1e-7, 1e-9

_BASE_ARRAYS = {}          # parent-held pandas baseline (inherited by fork)
_CURRENT = {"be": None}


def _extract(v):
    if hasattr(v, "to_numpy"):
        return v.to_numpy(dtype="float64", na_value=np.nan)
    if hasattr(v, "values"):
        return np.asarray(v, dtype="float64")
    return np.asarray(getattr(v, "data", None) or v, dtype="float64")


def _checksum(a):
    a = np.ascontiguousarray(a, dtype="float64")
    fin = np.isfinite(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(fin.tobytes())
    h.update(a[fin].tobytes())
    return h.hexdigest()[:24]


def _parity(base, other):
    if base.shape != other.shape:
        return f"shape{base.shape}vs{other.shape}"
    nb, no = np.isnan(base), np.isnan(other)
    if not np.array_equal(nb, no):
        return f"nanpat {int((nb ^ no).sum())}"
    m = ~nb
    if not m.any():
        return None
    d = np.abs(base[m] - other[m])
    sc = np.maximum(np.abs(base[m]), 1.0)
    bad = d > ATOL + RTOL * sc
    return f"maxabs {d.max():.2e} x{int(bad.sum())}" if bad.any() else None


def _child_run(op, out_path, pkl_path=None):
    """Runs in a forked child: execute ONE op, write the record file, exit."""
    be = _CURRENT["be"]
    rec = {"be": be, "op": op, "ok": False, "err": None}
    try:
        eng = FactorEngine(build_backend(be), SyntheticPanelSource(), run_mode="research")
        f = parse_factor(MANIFEST[op]["expr"], name=op, surface="all")
        t0 = time.perf_counter()
        res = eng.run_many([f], market="ashare", result_policy="return")
        dt = time.perf_counter() - t0
        results = res.get("results") if isinstance(res, dict) else res
        results = results if isinstance(results, dict) else {}
        v = results.get(op)
        if v is None:
            rec["err"] = "no-result"
        else:
            arr = _extract(v)
            fin = float(np.isfinite(arr).mean()) if arr.size else 0.0
            rec.update(ok=True, finite=round(fin, 4), sum=_checksum(arr),
                       n=int(arr.size), secs=round(dt, 4))
            if be != "pandas":
                base = _BASE_ARRAYS.get(op)
                rec["par"] = _parity(base, arr) if base is not None else None
            if be == "pandas" and pkl_path:
                import pickle
                pickle.dump(arr, open(pkl_path, "wb"))
    except Exception as e:
        rec["err"] = f"{type(e).__name__}:{str(e)[:130]}"
    with open(out_path, "w") as fh:
        json.dump(rec, fh)




def _run_ops_forked(be, ops):
    """Fork-isolated per-op execution; returns {op: record}."""
    import os as _os
    rec_dir = "/tmp/r65_records"
    _os.makedirs(rec_dir, exist_ok=True)
    ctx = mp.get_context("fork")
    records = {}
    WINDOW = 32
    for k in range(0, len(ops), WINDOW):
        group = ops[k:k + WINDOW]
        pending = {}
        for op in group:
            rp = f"{rec_dir}/{be}_{op}.json"
            p = ctx.Process(target=_child_run, args=(op, rp))
            p.start()
            pending[op] = [p, rp]
        for op in group:
            p, rp = pending[op]
            p.join(OP_TIMEOUT)
            if p.is_alive():
                p.terminate()
                p.join(3)
                records[op] = {"be": be, "op": op, "ok": False, "err": "TIMEOUT"}
            elif p.exitcode != 0:
                records[op] = {"be": be, "op": op, "ok": False,
                               "err": f"CRASH:exit{p.exitcode}"}
            else:
                try:
                    records[op] = json.load(open(rp))
                except Exception:
                    records[op] = {"be": be, "op": op, "ok": False, "err": "NO_RECORD"}
    return records


def _build_engine(backend):
    return FactorEngine(build_backend(backend), SyntheticPanelSource(), run_mode="research")
