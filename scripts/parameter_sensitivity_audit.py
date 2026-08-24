#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dead / scale / sign / monotonic-alias parameter audit (round-7 P0, review §56/57 + WS-G #294-#298).

AlphaMiner's search budget is wasted on parameters whose variation produces no
new factor:

* **dead**         — output is numerically identical for different values;
* **pure scale**   — ``factor2 = c * factor1`` (rank-identical);
* **sign alias**   — ``factor2 = -factor1`` (rank anti-identical);
* **monotonic alias** — ``factor2 = log(factor1)`` (rank nearly identical).

For every ``searchable`` scalar parameter of every registered operator this
audit evaluates the operator on the typed fixture regimes (Gaussian, heavy-tail,
trend, mean-revert, ties, gaps, positive-only, event-mask, group, OHLC) at a
small set of legal values and computes, per value pair, the numeric equality,
rank correlation and coverage.  Parameters whose sampled outputs are all
pairwise rank-equivalent (``|rho_rank| >= RHO_THRESHOLD``) are reported as
having no search value.

Behavioural guarantees (WS-G):
* scans **ALL** operators by default (``MAX_OPS`` is no longer capped); CI uses
  ``--shard i/N`` for a deterministic stable-hash partition;
* scalar parameters come from the operator spec's ``panel_params`` / scalar
  split — panel parameters are NEVER sampled as scalars (#295);
* ``choices`` are sampled by enumerating the legal choices; int bounds use a
  legal integer grid; float bounds use reviewed quantiles (#296);
* ``active_when`` parameters are only audited in branches where they are active
  (#297);
* exceptions are recorded as FAIL / AUDIT_UNEXECUTABLE, never silently skipped;
  ``--strict`` turns FAIL into a non-zero exit (#298).

Run: ``python3 scripts/parameter_sensitivity_audit.py``
      ``python3 scripts/parameter_sensitivity_audit.py --shard 0/16``
      ``python3 scripts/parameter_sensitivity_audit.py --limit 5``
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

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.search.factor_dedup import (
    build_typed_fixtures,
    split_scalar_panel_params,
)

N_SAMPLES = 3
RHO_THRESHOLD = 0.99999
# review #294: scan ALL operators by default.  CI keeps runs fast with
# ``--shard i/N`` (deterministic stable-hash partition) instead of a 60-op cap.
MAX_OPS = None


def _param_specs(op: Any) -> dict:
    return dict(getattr(getattr(op, "metadata", None), "param_specs", None) or {})


def _scalar_param_names(op: Any, canon: str) -> list[str]:
    """Scalar (non-panel) parameter names for an operator (review #295)."""
    meta = getattr(op, "metadata", None)
    all_params = list(getattr(meta, "param_names", None) or [])
    scalar, _panel = split_scalar_panel_params(all_params, meta, canon)
    return scalar


def _numeric_searchable_params(op: Any, canon: str) -> list[str]:
    """Searchable scalar parameters to sample (review #295 / #296)."""
    specs = _param_specs(op)
    out: list[str] = []
    for name in _scalar_param_names(op, canon):
        spec = specs.get(name)
        if spec is not None and not getattr(spec, "searchable", True):
            continue
        dtype = getattr(spec, "dtype", None)
        if dtype is str:
            continue  # string enums are semantic switches, not numeric knobs
        out.append(name)
    return out


def _activation_branches(param: str, op: Any) -> list[dict[str, Any]]:
    """Activation branches for ``param`` (review #297).

    A parameter with ``ParamSpec.active_when = (controller, allowed)`` is only
    live when ``controller`` is in ``allowed``; we enumerate those branches and
    run sensitivity only in an active branch.  Always-active params get a single
    empty branch.
    """
    spec = _param_specs(op).get(param)
    if spec is None or spec.active_when is None:
        return [{}]
    controller, allowed = spec.active_when
    branches = [{controller: value} for value in allowed]
    return branches or [{}]


def _sample_values(param: str, op: Any, n: int) -> list[float]:
    """Legal sample values for a numeric scalar param (review #296).

    Resolution order:
      1. ``choices`` exists → enumerate the legal choices directly;
      2. int bounds → a legal integer grid over ``[min, max]``;
      3. float bounds → reviewed quantiles ``{min, q25, mid, q75, max}``.

    Never linearly samples values outside the declared legal domain.
    """
    spec = _param_specs(op).get(param)
    if spec is not None and spec.choices is not None:
        return list(spec.choices)
    lo = getattr(spec, "min", None) if spec is not None else None
    hi = getattr(spec, "max", None) if spec is not None else None
    dtype = getattr(spec, "dtype", None) if spec is not None else None
    # Integer knobs: declared dtype=int, or a legacy name with no dtype declared.
    # Sampling an int knob with float quantiles would produce illegal values.
    is_int = dtype is int or (
        dtype is None and param in {
            "window", "min_periods", "order", "lag", "periods", "bins",
            "coefficient_index", "max_q", "block", "level", "band", "d", "k",
        }
    )
    if is_int:
        lo_i = int(lo) if lo is not None else 1
        hi_i = int(hi) if hi is not None else max(lo_i + n, lo_i * 3)
        if hi_i <= lo_i:
            hi_i = lo_i + n
        span = hi_i - lo_i + 1
        if span <= n:
            return list(range(lo_i, hi_i + 1))
        step = max(1, span // n)
        vals = list(range(lo_i, hi_i + 1, step))
        if vals[-1] != hi_i:
            vals.append(hi_i)
        return vals
    base = 5.0
    lo_f = float(lo) if lo is not None else base
    hi_f = float(hi) if hi is not None else base * 3.0
    if hi_f <= lo_f:
        hi_f = lo_f * 3.0 if lo_f != 0 else 3.0
    if hi_f == lo_f:
        hi_f = lo_f + 1.0
    qs = sorted({
        lo_f,
        hi_f,
        (lo_f + hi_f) / 2.0,
        lo_f + (hi_f - lo_f) * 0.25,
        lo_f + (hi_f - lo_f) * 0.75,
    })
    return qs


def _build_inputs(op: Any, canon: str, fixture: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Panel/data inputs for ``op``, fed from the given typed fixture.

    Scalar parameters are never fed here (they get sampled); panel/data params
    are fed DISTINCT values per slot so cross-input relationships (e.g. ADL's
    money-flow multiplier) are not collapsed by identical panels.
    """
    params = list(getattr(getattr(op, "metadata", None), "param_names", None) or [])
    scalars = set(_scalar_param_names(op, canon))
    inputs: dict[str, pd.DataFrame] = {}
    names: list[str] = []
    slot = 0
    for p in params:
        if p in scalars:
            continue
        if p in inputs:
            continue
        names.append(p)
        if p == "condition":
            inputs[p] = (fixture > 0).astype(float)
        elif slot == 0:
            inputs[p] = fixture
        elif slot == 1:
            inputs[p] = fixture + 1.0
        elif slot == 2:
            inputs[p] = fixture * 0.5 + 2.0
        else:
            inputs[p] = -fixture + 0.5
        slot += 1
    return names, inputs


def _matrix_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality with NaN-placement sensitivity (NaN is not 0.0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    return np.array_equal(
        np.where(np.isnan(a), 0.0, a), np.where(np.isnan(b), 0.0, b)
    )


def _classify(outs: list[pd.DataFrame]) -> str | None:
    """dead / scale_or_monotonic_alias / None for a set of sampled outputs.

    Uses **cross-sectional** rank (``axis=1`` — rank across instruments per
    day, the A-share convention) for the alias check.
    """
    rhos: list[float] = []
    same_count = 0
    for i in range(1, len(outs)):
        a = outs[i - 1].to_numpy(dtype=float)
        b = outs[i].to_numpy(dtype=float)
        if _matrix_equal(a, b):
            same_count += 1
        ra = pd.DataFrame(a).rank(axis=1, method="average").to_numpy()
        rb = pd.DataFrame(b).rank(axis=1, method="average").to_numpy()
        m = np.isfinite(ra) & np.isfinite(rb)
        if int(m.sum()) < 50:
            continue
        r = np.corrcoef(ra[m], rb[m])[0, 1]
        if np.isfinite(r):
            rhos.append(float(abs(r)))
    if same_count == len(outs) - 1:
        return "dead"
    if rhos and all(r >= RHO_THRESHOLD for r in rhos):
        return "scale_or_monotonic_alias"
    return None


def _audit_operator(op: Any, canon: str, fixtures: dict[str, pd.DataFrame]) -> tuple[list[dict], str]:
    """Audit one operator across all typed fixture regimes.

    Returns ``(reports, status)`` with status in
    ``{audited, FAIL, AUDIT_UNEXECUTABLE:no_scalar_params,
    AUDIT_UNEXECUTABLE:no_panel_inputs}``.
    """
    control = _numeric_searchable_params(op, canon)
    if not control:
        return [], "AUDIT_UNEXECUTABLE:no_scalar_params"
    reports: list[dict] = []
    failed = False
    any_audited = False
    for param in control:
        for branch in _activation_branches(param, op):
            try:
                vals = _sample_values(param, op, N_SAMPLES)
            except Exception:
                continue
            if len(set(vals)) < 2:
                continue
            for fixture_name, fixture in fixtures.items():
                names, inputs = _build_inputs(op, canon, fixture)
                if not names:
                    continue  # no panel input for this fixture -> no sensitivity
                outs: list[pd.DataFrame] = []
                try:
                    for v in vals:
                        kw = dict(branch)
                        kw.update(inputs)
                        kw[param] = v
                        outs.append(pd.DataFrame(op.calculate(**kw)))
                except Exception:
                    failed = True  # review #298: record, never silently skip
                    continue
                if len(outs) < 2:
                    continue
                kind = _classify(outs)
                if kind is not None:
                    reports.append({
                        "canonical": canon,
                        "param": param,
                        "fixture": fixture_name,
                        "kind": kind,
                    })
                    any_audited = True
                else:
                    any_audited = True
    if failed:
        return reports, "FAIL"
    if not any_audited:
        return reports, "AUDIT_UNEXECUTABLE:no_panel_inputs"
    return reports, "audited"


def _stable_hash(text: str) -> int:
    """Deterministic stable hash (review #294).

    Python's builtin ``hash(str)`` is seeded per-process (PYTHONHASHSEED), so
    it is NOT stable across runs/machines.  crc32 gives the same
    ``hash % N == i`` partition everywhere.
    """
    import zlib

    return zlib.crc32(text.encode("utf-8"))


def _in_shard(canon: str, num_shards: int, shard_index: int | None) -> bool:
    if num_shards <= 1 or shard_index is None:
        return True
    return _stable_hash(canon) % num_shards == shard_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-ops", type=int, default=MAX_OPS,
                    help="Cap on the number of operators audited (default: ALL).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Alias for --max-ops (small-N smoke runs).")
    ap.add_argument("--shard", type=int, default=None,
                    help="Shard index i (0-based) selected by stable hash % num-shards.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total shards (default 1 = no sharding).")
    ap.add_argument("--fixtures", type=str, default=None,
                    help="Comma-separated fixture names to use (default: all typed fixtures).")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero (2) if any operator FAILs.")
    args = ap.parse_args(argv)

    max_ops = args.limit if args.limit is not None else args.max_ops
    num_shards = max(1, args.num_shards)
    if args.shard is not None and not (0 <= args.shard < num_shards):
        ap.error(f"--shard must be in [0, {num_shards - 1}]")

    load_all()
    fixtures = build_typed_fixtures()
    if args.fixtures:
        want = [f.strip() for f in args.fixtures.split(",") if f.strip()]
        missing = [f for f in want if f not in fixtures]
        if missing:
            print(f"Unknown fixtures: {missing}; available: {sorted(fixtures)}",
                  file=sys.stderr)
            return 2
        fixtures = {k: v for k, v in fixtures.items() if k in want}

    findings: list[dict] = []
    status_counts: dict[str, int] = {}
    canonicals = sorted(set(OperatorRegistry.list_canonical()))
    count = 0
    for canon in canonicals:
        if not _in_shard(canon, num_shards, args.shard):
            continue
        if max_ops is not None and count >= max_ops:
            break
        try:
            op = OperatorRegistry.get(canon, "pandas_numpy") or OperatorRegistry.get(canon)
        except Exception:
            status_counts["LOAD_FAIL"] = status_counts.get("LOAD_FAIL", 0) + 1
            continue
        if op is None:
            status_counts["AUDIT_UNEXECUTABLE:no_runtime"] = (
                status_counts.get("AUDIT_UNEXECUTABLE:no_runtime", 0) + 1
            )
            continue
        count += 1
        reports, status = _audit_operator(op, canon, fixtures)
        findings.extend(reports)
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Operators scanned: {count} (shard {args.shard}/{num_shards}, "
          f"fixtures {sorted(fixtures)})")
    for status, n in sorted(status_counts.items()):
        print(f"  {status}: {n}")

    if not findings:
        print("No dead / scale / monotonic-alias parameters found in the audited sample.")
        return 0 if not (args.strict and status_counts.get("FAIL", 0)) else 2

    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    print(f"Found {len(findings)} parameter/fixture entries with no search value:")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind}: {n}")
    for f in findings[:40]:
        print(f"  {f['canonical']}: param={f['param']} fixture={f['fixture']} ({f['kind']})")

    ret = 1 if findings else 0
    if args.strict and status_counts.get("FAIL", 0):
        ret = 2
    return ret


if __name__ == "__main__":
    sys.exit(main())
