# -*- coding: utf-8 -*-
"""Scan operator impl sources for row-wise iteration markers; freeze baseline.

Scans the pandas/polars/sql impl source files of every manifest op via the
registry, emits vectorization_baseline.json {op: {backend: [hits]}}.
"""
import warnings, json, os, sys, ast, importlib.util, inspect
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_engine.cleaned_operators  # noqa
from factor_engine.cleaned_operators import load_all
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    load_all()

MANIFEST = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_manifest.json")))

MARKERS = ("iterrows", "itertuples", "applymap", "swifter", "progress_apply")


def scan_text(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["syntax-error"]
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in MARKERS:
                hits.append(f"call:{name}:L{getattr(node, 'lineno', '?')}")
            elif name == "apply":
                for kw in node.keywords:
                    if kw.arg == "axis" and getattr(getattr(kw, "value", None), "value", None) == 1:
                        hits.append(f"apply-axis1:L{getattr(node, 'lineno', '?')}")
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            fn = getattr(node.iter, "func", None)
            if getattr(fn, "id", "") == "range" and node.iter.args:
                a0 = node.iter.args[0]
                if isinstance(a0, ast.Call) and getattr(a0.func, "id", "") == "len":
                    hits.append(f"range-len-loop:L{getattr(node, 'lineno', '?')}")
    return hits


def impl_module_path(op_name, backend):
    """Resolve to the implementation's module FILE (snippet parsing is unreliable)."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry as R
    try:
        op = R.get(op_name, backend=backend, mode="production")
    except Exception:
        return None
    if op is None:
        return None
    fn = getattr(op, "run", None) or getattr(op, "call", None) or getattr(op, "__call__", None)
    target = fn if fn is not None else op
    mod = getattr(type(target), "__module__", None) if not inspect.isfunction(target) else target.__module__
    if mod is None:
        return None
    try:
        spec = importlib.util.find_spec(mod)
    except Exception:
        return None
    origin = getattr(spec, "origin", None)
    if origin and os.path.exists(origin):
        return origin
    return None


baseline = {}
for name, v in MANIFEST.items():
    if v["skip"] or not v["expr"]:
        continue
    for be in v["backends"]:
        path = impl_module_path(name, be)
        if path is None:
            continue
        try:
            src = open(path).read()
        except Exception:
            continue
        hits = scan_text(src)
        if hits:
            baseline.setdefault(path, {"hits": hits[:12],
                                       "ops": sorted(set(
                                           baseline.get(path, {}).get("ops", []) + [name]))})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorization_baseline.json")
json.dump(baseline, open(out, "w"), indent=1, sort_keys=True)
print("SCAN_DONE modules_with_hits:", len(baseline))
from collections import Counter
c = Counter(len(v) for v in baseline.values())
print("backends_per_op dist:", dict(sorted(c.items())))
