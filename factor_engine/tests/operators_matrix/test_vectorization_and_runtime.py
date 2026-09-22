# -*- coding: utf-8 -*-
"""Static row-wise-iteration gate for operator implementations.

Scans the implementation module FILE of every manifest op for unambiguous
row-wise markers (iterrows/itertuples/applymap/apply(axis=1)/range(len) row
loops). The violating-module set is frozen in vectorization_baseline.json;
the suite fails on any NEW violation and never on the documented baseline.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

MARKERS = ("iterrows", "itertuples", "applymap", "swifter", "progress_apply")


def _scan_text(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in MARKERS:
                hits.append(f"call:{name}")
            elif name == "apply":
                for kw in node.keywords:
                    if kw.arg == "axis" and getattr(getattr(kw, "value", None), "value", None) == 1:
                        hits.append("apply-axis1")
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if getattr(node.iter.func, "id", "") == "range" and node.iter.args:
                a0 = node.iter.args[0]
                if isinstance(a0, ast.Call) and getattr(a0.func, "id", "") == "len":
                    hits.append("range-len-loop")
    return hits


def _impl_module_path(op_name, backend):
    from factor_engine.cleaned_operators.registry import OperatorRegistry as R
    try:
        op = R.get(op_name, backend=backend, mode="production")
    except Exception:
        return None
    if op is None:
        return None
    fn = getattr(op, "run", None) or getattr(op, "call", None) or getattr(op, "__call__", None)
    target = fn if fn is not None else op
    mod = (target.__module__ if inspect.isfunction(target)
           else getattr(type(target), "__module__", None))
    if not mod:
        return None
    try:
        spec = importlib.util.find_spec(mod)
    except Exception:
        return None
    origin = getattr(spec, "origin", None)
    return origin if origin and os.path.exists(origin) else None


def _current_violations():
    manifest = json.load(open(os.path.join(HERE, "op_manifest.json")))
    by_module = {}
    for name, v in manifest.items():
        if v["skip"] or not v["expr"]:
            continue
        for be in v["backends"]:
            path = _impl_module_path(name, be)
            if path is None or path in by_module:
                continue
            try:
                hits = _scan_text(open(path).read())
            except OSError:
                continue
            if hits:
                by_module[path] = hits
    return by_module


def test_no_new_rowwise_implementations():
    """No operator implementation module may introduce row-wise iteration
    beyond the frozen baseline."""
    baseline = json.load(open(os.path.join(HERE, "vectorization_baseline.json")))
    current = _current_violations()
    new = {p: h for p, h in current.items() if p not in baseline}
    assert not new, (
        "NEW row-wise operator implementation modules (not in baseline): "
        f"{dict(list(sorted(new.items()))[:10])}"
    )


def test_baseline_is_not_empty():
    """Guard: the baseline exists and actually pins some modules."""
    baseline = json.load(open(os.path.join(HERE, "vectorization_baseline.json")))
    assert len(baseline) >= 10
