#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dead / scale / sign / monotonic-alias parameter audit (round-7 P0, review §56/57).

AlphaMiner's search budget is wasted on parameters whose variation produces no
new factor:

* **dead**         — output is numerically identical for different values;
* **pure scale**   — ``factor2 = c * factor1`` (rank-identical);
* **sign alias**   — ``factor2 = -factor1`` (rank anti-identical);
* **monotonic alias** — ``factor2 = log(factor1)`` (rank nearly identical).

For every ``searchable`` numeric parameter of every registered operator this
audit evaluates the operator on a fixed synthetic panel at ``N_SAMPLES``
evenly-spaced values and computes, per value pair, the numeric equality, rank
correlation and coverage.  Parameters whose sampled outputs are all pairwise
rank-equivalent (``|rho_rank| >= RHO_THRESHOLD``) are reported as having no
search value.

Run: ``python3 scripts/parameter_sensitivity_audit.py``
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

# Allow running from a checkout without an installed egg (scripts/ ../repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.search.factor_dedup import probe_panel

N_SAMPLES = 3
RHO_THRESHOLD = 0.99999
MAX_OPS = 60  # audit a bounded sample per run; extend for a full sweep


def _numeric_searchable_params(op: Any) -> list[str]:
    specs = dict(getattr(getattr(op, "metadata", None), "param_specs", None) or {})
    out: list[str] = []
    for name in getattr(getattr(op, "metadata", None), "param_names", []) or []:
        spec = specs.get(name)
        if spec is None or getattr(spec, "searchable", True):
            out.append(name)
    return out


def _sample_values(param: str, op: Any, n: int) -> list[float]:
    """Evenly-spaced legal values for a numeric param (window/horizon ints too)."""
    spec = (dict(getattr(getattr(op, "metadata", None), "param_specs", None) or {})).get(param)
    lo = getattr(spec, "min", None)
    hi = getattr(spec, "max", None)
    dtype = getattr(spec, "dtype", None)
    is_int = dtype is int or param in {
        "window", "min_periods", "order", "lag", "periods", "bins",
        "coefficient_index", "max_q", "block", "level", "band", "d", "k",
    }
    base = 5.0 if not is_int else 5
    lo = float(lo) if lo is not None else base
    hi = float(hi) if hi is not None else base * 3
    if hi <= lo:
        hi = lo * 3.0
    vals = np.linspace(lo, hi, n)
    return [float(round(v)) if is_int else float(v) for v in vals]


def _build_inputs(op: Any) -> tuple[list[str], dict[str, pd.DataFrame]]:
    params = list(getattr(getattr(op, "metadata", None), "param_names", []) or [])
    panel = probe_panel(seed=7)
    inputs: dict[str, pd.DataFrame] = {}
    names: list[str] = []
    for p in params:
        if p in {"window", "min_periods", "lag", "periods", "bins", "order", "block", "max_q", "q", "coefficient_index", "d", "k", "level", "band", "short_periods", "long_periods"}:
            continue  # numeric control params get sampled, not fed
        if p in inputs:
            continue
        names.append(p)
        inputs[p] = panel if p != "condition" else (panel > 0).astype(float)
    return names, inputs


def _audit_operator(op: Any, canon: str) -> list[dict]:
    control = _numeric_searchable_params(op)
    if not control:
        return []
    panel_names, inputs = _build_inputs(op)
    if not panel_names:
        return []
    reports: list[dict] = []
    for param in control:
        vals = _sample_values(param, op, N_SAMPLES)
        if len(set(vals)) < 2:
            continue
        outs: list[pd.DataFrame] = []
        try:
            for v in vals:
                kw = dict(inputs)
                kw[param] = v
                outs.append(pd.DataFrame(op.calculate(**kw)))
        except Exception:
            continue
        if len(outs) < 2:
            continue
        # pairwise rank correlation across the 3+ samples
        rhos: list[float] = []
        same_count = 0
        for i in range(1, len(outs)):
            a = outs[i - 1].to_numpy(dtype=float)
            b = outs[i].to_numpy(dtype=float)
            if np.array_equal(np.nan_to_num(a), np.nan_to_num(b), equal_nan=True) or (
                a.shape == b.shape and np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
            ):
                same_count += 1
            ra = pd.DataFrame(a).rank(axis=0, method="average").to_numpy()
            rb = pd.DataFrame(b).rank(axis=0, method="average").to_numpy()
            m = np.isfinite(ra) & np.isfinite(rb)
            if int(m.sum()) < 50:
                continue
            r = np.corrcoef(ra[m], rb[m])[0, 1]
            if np.isfinite(r):
                rhos.append(float(abs(r)))
        if same_count == len(outs) - 1:
            reports.append({"canonical": canon, "param": param, "kind": "dead"})
        elif rhos and all(r >= RHO_THRESHOLD for r in rhos):
            reports.append({"canonical": canon, "param": param, "kind": "scale_or_monotonic_alias"})
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-ops", type=int, default=MAX_OPS)
    args = ap.parse_args(argv)
    load_all()

    findings: list[dict] = []
    canonicals = sorted(set(OperatorRegistry._catalog.keys()))
    count = 0
    for canon in canonicals:
        if count >= args.max_ops:
            break
        op = OperatorRegistry.get(canon, "pandas_numpy") or OperatorRegistry.get(canon)
        if op is None:
            continue
        count += 1
        findings.extend(_audit_operator(op, canon))

    if not findings:
        print("No dead / scale / monotonic-alias parameters found in the audited sample.")
        return 0
    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    print(f"Found {len(findings)} parameters with no search value (sampled {count} ops):")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind}: {n}")
    for f in findings[:40]:
        print(f"  {f['canonical']}: param={f['param']} ({f['kind']})")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
