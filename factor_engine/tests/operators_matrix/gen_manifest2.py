# -*- coding: utf-8 -*-
"""Generate one DSL expression per canonical operator from catalog metadata.

Frozen output: op_manifest.json next to this file.
skip reasons: not_dsl_exposed | dsl_budget | undeclared_required_param
"""
import warnings, json, os, sys, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
import factor_engine.cleaned_operators  # noqa
from factor_engine.cleaned_operators import load_all

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    load_all()
from factor_engine.cleaned_operators.registry import OperatorRegistry as R, MISSING

cat = R.catalog()

NUMERIC_HINTS = [
    ("window", 20), ("win", 20), ("lag", 5), ("period", 20), ("left_window", 10),
    ("right_window", 10), ("min_periods", 1), ("d", 20), ("n", 20), ("k", 5),
    ("q", 0.5), ("p", 0.5), ("alpha", 0.5), ("beta", 0.5), ("gamma", 0.5),
    ("threshold", 0.0), ("level", 0.0), ("span", 20), ("order", 2), ("degree", 2),
    ("bins", 10), ("quantile", 0.5), ("prob", 0.5), ("rate", 0.1), ("shift", 5),
    ("offset", 5), ("horizon", 10), ("length", 20), ("size", 20),
]


def param_value(ps, name):
    if ps is not None:
        choices = getattr(ps, "choices", None)
        if choices:
            return choices[0]
        default = getattr(ps, "default", None)
        if default is not None and default is not MISSING:
            return None  # has default -> omit from expr
        dt = getattr(ps, "dtype", None)
        lo = getattr(ps, "min", None)
        hi = getattr(ps, "max", None)
        ln = str(name).lower()
        if dt is str:
            return "industry"
        if dt is int or (dt is None and any(h in ln for h in ("window", "period", "lag", "horizon", "shift", "order", "degree", "bins"))):
            # relational params: fast/short must stay below slow/long
            v = 20
            if "fast" in ln or "short" in ln:
                v = 5
            elif "slow" in ln or "long" in ln:
                v = 40
            elif ln.startswith("min_") or "min_periods" in ln:
                v = 1
            else:
                for hint, hv in NUMERIC_HINTS:
                    if hint in ln:
                        v = hv
                        break
            if lo is not None and v < lo:
                v = int(lo)
            if hi is not None and v > hi:
                v = int(hi)
            return v
        v = 0.5
        if ("maximum" in ln or ln.endswith("_max") or "ceiling" in ln
                or "cap" in ln or "upper" in ln):
            v = 0.5
        elif ("acceleration" in ln or "step" in ln or "increment" in ln
              or "epsilon" in ln or "tol" in ln or "rate" in ln):
            v = 0.1
        else:
            for hint, hv in NUMERIC_HINTS:
                if hint in ln:
                    v = hv
                    break
        if dt is int:
            if lo is not None and v < lo:
                v = int(lo)
            if hi is not None and v > hi:
                v = int(hi)
            return int(round(v))
        if lo is not None and v < lo:
            v = float(lo)
        if hi is not None and v > hi:
            v = float(hi)
        return float(v)
    ln = str(name).lower()
    if any(h in ln for h in ("ratio", "weight", "share", "pct", "prob", "quantile")):
        return 0.5
    return 20


def build_expr(name, e):
    specs = e.get("param_specs") or {}
    pnames = list(e.get("param_names") or [])
    inputs = [p for p in pnames if p not in specs]
    required = []
    for p in pnames:
        if p in inputs:
            continue
        ps = specs.get(p)
        if ps is None or getattr(ps, "default", None) is None or getattr(ps, "default", None) is MISSING:
            required.append((p, ps))
    if any(p == "..." or "kwargs" in p.lower() for p, _ in required):
        return None, "undeclared_required_param", len(inputs), [p for p, _ in required]
    arity = e.get("input_arity")
    if arity is None:
        arity = len(inputs) if inputs else 1
    if isinstance(arity, str):
        try:
            arity = int(arity)
        except Exception:
            arity = 1
    FIELD_LADDER = ["close", "volume", "returns", "high", "low", "open",
                    "vwap", "is_suspended"]
    BOOLISH = ("suspend", "tradable", "valid", "event", "flag", "mask",
               "is_", "limit_up", "limit_down", "st_", "halt")

    def pick_field(pname, idx):
        ln = str(pname).lower()
        if any(b in ln for b in BOOLISH):
            return "close"   # dataset-resolving fields hang offline; expect fast contract fail instead
        if "industry" in ln or "group" in ln or "sector" in ln:
            return "close"
        if "return" in ln or ln == "ret" or "pct" in ln:
            return "close"
        if "volume" in ln or "share" in ln or ln == "vol":
            return "volume"
        if "amount" in ln or "turnover" in ln:
            return "close"
        if "vwap" in ln:
            return "close"
        if "high" in ln:
            return "high"
        if "low" in ln:
            return "low"
        if "open" in ln:
            return "open"
        if "close" in ln or "price" in ln:
            return "close"
        return FIELD_LADDER[idx % len(FIELD_LADDER)]

    if arity == 0:
        in_args = []
    elif arity == 1:
        in_args = [pick_field(inputs[0] if inputs else "x", 0)]
    else:
        in_args = [pick_field(inputs[i] if i < len(inputs) else f"f{i+1}", i)
                   for i in range(arity)]
    kw = []
    for p, ps in required:
        v = param_value(ps, p)
        if v is None:
            v = 20
        kw.append(f"{p}={v!r}")
    return f"{name}({', '.join(in_args + kw)})", None, arity, [p for p, _ in required]


manifest = {}
for name, e in cat.items():
    expr, skip, arity, required = build_expr(name, e)
    manifest[name] = {
        "expr": expr,
        "backends": sorted(e.get("backends") or []),
        "production_policy": e.get("production_policy"),
        "stateful": bool(e.get("stateful")),
        "input_arity": arity,
        "required_params": required,
        "skip": skip,
    }

# classify: parse with surface="all"; unknown-operator -> not_dsl_exposed,
# arity/budget errors -> retry with inputs capped at 2, else dsl_budget
from factor_engine.api.dsl_parser import parse_expr

for name, entry in manifest.items():
    if entry["skip"]:
        continue
    try:
        parse_expr(entry["expr"], surface="all")
        continue
    except Exception as ex:
        msg = str(ex)
    if "Unsupported function" in msg:
        entry["skip"] = "not_dsl_exposed"
        continue
    name_part = entry["expr"].split("(", 1)[0]
    rest = entry["expr"].split("(", 1)[1] if "(" in entry["expr"] else ""
    kw = ",".join(t for t in rest.split(",") if "=" in t) if rest else ""
    capped = f"{name_part}(close, volume{(',' + kw) if kw else ''})"
    try:
        parse_expr(capped, surface="all")
        entry["expr"] = capped
        continue
    except Exception as ex2:
        msg2 = str(ex2)
    if "Unsupported function" in msg2:
        entry["skip"] = "not_dsl_exposed"
    else:
        entry["skip"] = "dsl_budget"

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_manifest.json")
json.dump(manifest, open(out, "w"), indent=0, sort_keys=True)
from collections import Counter
sk = Counter(v["skip"] for v in manifest.values())
print("N", len(manifest), "skips:", dict(sk))
print("parsable:", sum(1 for v in manifest.values() if v["expr"] and not v["skip"]))
be = Counter(tuple(v["backends"]) for v in manifest.values())
print("backend combos:", dict(sorted(be.items(), key=lambda x: -x[1])[:6]))
