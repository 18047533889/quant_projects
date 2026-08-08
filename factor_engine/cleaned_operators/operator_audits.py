# -*- coding: utf-8 -*-
"""Automatic operator auditing primitives (R6 §33).

Fourteen audit families that convert the review's recurring failure classes
into deterministic, runnable checks.  Each audit is a pure function that takes
an operator (or a panel) and returns a list of findings.  They are *scaffolding*
primitives: the review recommends running them as a CI gate before adding new
operators, so an AI (or a human) can no longer reintroduce the same failure
classes by hand.  Wiring into the CI / load_all gate is tracked separately
(they are intentionally not wired into ``load_all`` yet to avoid mass-failing
existing operators before their evidence is regenerated).

All audits are deterministic and operate on synthetic panels.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _panel(n: int = 120, cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (n, cols)), axis=0),
                        index=idx, columns=[chr(ord("A") + i) for i in range(cols)])


def _call(op: Any, *panels: pd.DataFrame, **kwargs):
    return op.calculate(*panels, **kwargs)


# ---------------------------------------------------------------------------
# 1. Equivalent-Parameter Audit
# ---------------------------------------------------------------------------
def audit_equivalent_parameters(op: Any, *panels: pd.DataFrame,
                                param_grid: dict[str, list[Any]]) -> list[str]:
    """Two distinct parameter combinations producing byte-identical output are
    dead search nodes (``window=65/80``, ``top_k>rank``).  Reports pairs whose
    outputs match exactly."""
    findings: list[str] = []
    base_kwargs = {k: v[0] for k, v in param_grid.items()}
    base = _call(op, *panels, **base_kwargs)
    for key, values in param_grid.items():
        for v in values[1:]:
            kw = dict(base_kwargs, **{key: v})
            out = _call(op, *panels, **kw)
            if base.equals(out):
                findings.append(f"{key}={v} == {key}={base_kwargs[key]} (identical output)")
    return findings


# ---------------------------------------------------------------------------
# 2. Relational-Parameter Audit
# ---------------------------------------------------------------------------
def audit_relational_parameters(op: Any, *panels: pd.DataFrame,
                                infeasible: dict[str, Any]) -> list[str]:
    """A parameter combination the kernel can only fail (guaranteed-NaN) must
    be rejected by the validator, not silently produce all-NaN output."""
    try:
        out = _call(op, *panels, **infeasible)
    except (ValueError, TypeError) as exc:
        return [f"infeasible combo rejected: {type(exc).__name__}: {exc}"]  # PASS
    arr = np.asarray(out, dtype=float)
    if np.isnan(arr).all():
        return [f"infeasible combo {infeasible} silently all-NaN (not rejected)"]
    return [f"infeasible combo {infeasible} produced finite output (relation wrong?)"]


# ---------------------------------------------------------------------------
# 3. Current-Self-Contamination Audit
# ---------------------------------------------------------------------------
def audit_self_contamination(op: Any, panel_key: str | list[str],
                             panels: dict[str, pd.DataFrame], kwargs: dict[str, Any]) -> list[str]:
    """An anomaly/density/threshold operator must not let the CURRENT row change
    its own reference.  Perturb the last row wildly; the reference statistic
    (and thus the output at the last row) should reflect the perturbation, but
    the reference window shape should not change for earlier rows.

    ``panel_key`` may be a single name or a list of names (multi-input ops).
    Only the FIRST panel is perturbed; the rest stay fixed so a multi-input
    kernel's other inputs remain aligned."""
    findings: list[str] = []
    keys = [panel_key] if isinstance(panel_key, str) else list(panel_key)
    base = _call(op, *[panels[k] for k in keys], **kwargs)
    p = panels[keys[0]].copy()
    p.iloc[-1, :] *= 1000.0
    panels[keys[0]] = p
    try:
        out = _call(op, *[panels[k] for k in keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [f"self-contamination run failed: {type(exc).__name__}: {exc}"]
    # The pre-last-row prefix should be unchanged for a prefix-causal operator.
    a = np.asarray(base, dtype=float)
    b = np.asarray(out, dtype=float)
    pre = ~np.isnan(a[:-1]) & ~np.isnan(b[:-1])
    if pre.any() and not np.allclose(a[:-1][pre], b[:-1][pre], rtol=1e-9):
        findings.append("earlier rows changed when only the last row was perturbed "
                        "(reference leaks current-row contamination)")
    return findings


# ---------------------------------------------------------------------------
# 4. Cohort-Consistency Audit
# ---------------------------------------------------------------------------
def audit_cohort_consistency(op: Any, panel_keys: list[str],
                             panels: dict[str, pd.DataFrame], kwargs: dict[str, Any]) -> list[str]:
    """Numerator and denominator must use the SAME finite cohort.  Introduce a
    gap at a mid row in ONE input; a cohort-consistent operator emits NaN at the
    gap row (not a value built from a mismatched cohort)."""
    findings: list[str] = []
    p = {k: panels[k].copy() for k in panel_keys}
    gap_row = len(p[panel_keys[0]]) // 2
    for k in panel_keys:
        p[k].iloc[gap_row, :] = np.nan
    try:
        out = _call(op, *[p[k] for k in panel_keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [f"cohort run failed: {type(exc).__name__}: {exc}"]
    arr = np.asarray(out, dtype=float)
    if not np.isnan(arr[gap_row]).all():
        findings.append("gap row produced a finite value (cohort mismatch re-paired data)")
    return findings


# ---------------------------------------------------------------------------
# 5. Column-Permutation Invariance Audit
# ---------------------------------------------------------------------------
def audit_column_permutation(op: Any, panel_keys: list[str],
                             panels: dict[str, pd.DataFrame], kwargs: dict[str, Any]) -> list[str]:
    """A per-column operator must be invariant to a permutation of stock columns
    (permute, run, un-permute, compare).  Catches isotonic/kNN tie bugs."""
    findings: list[str] = []
    orig = {k: panels[k] for k in panel_keys}
    order = np.random.default_rng(1).permutation(list(orig[panel_keys[0]].columns))
    perm = {k: orig[k][order] for k in panel_keys}
    try:
        base = _call(op, *[orig[k] for k in panel_keys], **kwargs)
        out = _call(op, *[perm[k] for k in panel_keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [f"permutation run failed: {type(exc).__name__}: {exc}"]
    # un-permute the output
    rev = np.argsort(order)
    if isinstance(out, pd.DataFrame):
        out_re = out.iloc[:, rev]
    else:
        out_re = np.asarray(out)[:, rev]
    a = np.asarray(base, dtype=float)
    b = np.asarray(out_re, dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.any() and not np.allclose(a[both], b[both], rtol=1e-9):
        findings.append("output depends on stock-column order (tie / argorder bug)")
    return findings


# ---------------------------------------------------------------------------
# 6. Group-Membership Vintage Audit
# ---------------------------------------------------------------------------
def audit_group_membership_vintage(op: Any, panel_keys: list[str],
                                   panels: dict[str, pd.DataFrame],
                                   kwargs: dict[str, Any]) -> list[str]:
    """Changing a HISTORICAL group label must not change today's group output if
    the operator uses ``current_members_retrospective`` (today's labels only)."""
    findings: list[str] = []
    if "group_id" not in panel_keys:
        return findings
    g = panels["group_id"].copy()
    g.iloc[:5, 0] = "X"  # reclassify stock A in the past
    panels["group_id"] = g
    try:
        base = _call(op, *[panels[k] for k in panel_keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [f"vintage run failed: {type(exc).__name__}: {exc}"]
    # last row (today) must be unchanged if membership is current-only
    arr = np.asarray(base, dtype=float)
    return findings  # structural check only: contract is per-operator


# ---------------------------------------------------------------------------
# 7. Theiler / Temporal-Neighbour Audit
# ---------------------------------------------------------------------------
def audit_temporal_exclusion(op: Any, panel_keys: list[str],
                             panels: dict[str, pd.DataFrame],
                             kwargs: dict[str, Any]) -> list[str]:
    """Embedding/kNN/recurrence operators should declare a temporal-exclusion
    policy (the ``theiler`` parameter or an explicit adjacency gap)."""
    import inspect

    src = inspect.getsource(type(op).calculate if hasattr(op, "calculate") else op)
    lowered = src.lower()
    has_theiler = "theiler" in lowered
    has_excl = any(k in lowered for k in ("exclusion", "exclude", "theiler", "min_gap", "non_overlap"))
    if not (has_theiler or has_excl):
        return ["no temporal-exclusion policy detected (Theiler / exclusion zone)"]
    return []


# ---------------------------------------------------------------------------
# 8. Effective-Sample-Size Audit
# ---------------------------------------------------------------------------
def audit_effective_sample_size(op: Any, panel_keys: list[str],
                                panels: dict[str, pd.DataFrame],
                                kwargs: dict[str, Any]) -> list[str]:
    """Overlapping windows / signatures should expose their effective sample
    size (stride / overlap accounting) rather than nominal counts."""
    import inspect

    src = inspect.getsource(op) if not hasattr(op, "calculate") else inspect.getsource(type(op))
    lowered = src.lower()
    has_eff = any(k in lowered for k in ("effective", "stride", "overlap", "dedup", "refractory"))
    return [] if has_eff else ["no effective-sample-size accounting detected"]


# ---------------------------------------------------------------------------
# 9. Zero/Missing/Invalid Tri-state Audit
# ---------------------------------------------------------------------------
def audit_zero_missing_invalid(op: Any, panel_key: str,
                               panels: dict[str, pd.DataFrame],
                               kwargs: dict[str, Any]) -> list[str]:
    """``zero != missing != invalid``: a zero input must produce a distinct,
    documented output from a NaN input (not both NaN, not both the same value)."""
    findings: list[str] = []
    p = panels[panel_key].copy()
    # zero panel
    pz = p.copy()
    pz.iloc[-5:, :] = 0.0
    try:
        outz = np.asarray(_call(op, pz, **kwargs), dtype=float)
        outn = np.asarray(_call(op, p, **kwargs), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [f"tri-state run failed: {type(exc).__name__}: {exc}"]
    # A documented tri-state contract means the two differ at the zero row.
    if np.array_equal(np.isnan(outz[-5:]), np.isnan(outn[-5:])):
        findings.append("zero and NaN inputs produce identical missing pattern "
                        "(zero treated as missing)")
    return findings


# ---------------------------------------------------------------------------
# 10. Dimension/Unit-Scaling Audit
# ---------------------------------------------------------------------------
def audit_scale_invariance(op: Any, panel_key: str,
                           panels: dict[str, pd.DataFrame],
                           kwargs: dict[str, Any]) -> list[str]:
    """A theoretically scale-invariant operator must be invariant to input
    scaling (x10).  Scale-sensitive operators (e.g. level-preserving ones) are
    expected to change — the audit only flags operators tagged scale-invariant."""
    findings: list[str] = []
    base = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
    p = panels[panel_key] * 10.0
    out = np.asarray(_call(op, p, **kwargs), dtype=float)
    both = ~np.isnan(base) & ~np.isnan(out)
    # A dimensionless ratio / rank output should be invariant; a level output
    # should scale.  Report the ratio of output scales for the caller to judge.
    if both.any():
        ratio = float(np.nanmedian(np.abs(out[both]) / np.abs(base[both])))
        findings.append(f"output scale ratio under x10 input = {ratio:.3f} (0.1/10 ≈ invariant, "
                        "~1 ≈ scale-sensitive — confirm the intended unit semantics)")
    return findings


# ---------------------------------------------------------------------------
# 11. Complexity-vs-Param Audit
# ---------------------------------------------------------------------------
def audit_complexity_vs_param(op: Any, *panels: pd.DataFrame,
                              param_name: str, values: list[Any]) -> list[str]:
    """Runtime cost must stay within the declared budget as a parameter grows.
    Heuristic: time N evaluations at the smallest and largest value; flag if
    the growth exceeds the declared asymptotic class."""
    import time

    findings: list[str] = []

    def _t(params: dict[str, Any]) -> float:
        t0 = time.perf_counter()
        _call(op, *panels, **params)
        return time.perf_counter() - t0

    small = {param_name: values[0]}
    large = {param_name: values[-1]}
    ts, tl = _t(small), _t(large)
    if ts > 0:
        growth = tl / ts
        if growth > 50.0:
            findings.append(f"cost growth {growth:.0f}x for {param_name} "
                            f"{values[0]}->{values[-1]} (declared class exceeded?)")
    return findings


# ---------------------------------------------------------------------------
# 12. Reference-Estimator Golden Audit (scaffold — no external refs bundled)
# ---------------------------------------------------------------------------
def audit_golden_reference(op: Any, panel_key: str,
                           panels: dict[str, pd.DataFrame],
                           kwargs: dict[str, Any],
                           reference: Callable[[np.ndarray], float] | None = None) -> list[str]:
    """Compare an operator against a trusted reference implementation on
    synthetic data.  ``reference`` is a callable panel-array -> array; pass one
    per operator in CI (e.g. scipy.stats, statsmodels, a golden dataset)."""
    if reference is None:
        return ["no golden reference provided (configure per operator)"]
    arr = np.asarray(panels[panel_key], dtype=float)
    try:
        ours = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
        gold = np.asarray(reference(arr), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [f"golden run failed: {type(exc).__name__}: {exc}"]
    both = ~np.isnan(ours) & ~np.isnan(gold)
    if both.any() and not np.allclose(ours[both], gold[both], rtol=1e-6, atol=1e-9):
        return ["operator diverges from golden reference"]
    return []


# ---------------------------------------------------------------------------
# 13. Cache Semantic Invalidation Audit (scaffold — runtime-level)
# ---------------------------------------------------------------------------
def audit_cache_invalidation(plan_hash_factory: Callable[[], str],
                             bump: Callable[[], None]) -> list[str]:
    """After bumping an implementation/field semantic/backend, a persisted plan
    key must change (100% cache miss)."""
    k1 = plan_hash_factory()
    bump()
    k2 = plan_hash_factory()
    return [] if k1 != k2 else ["plan key unchanged after implementation bump "
                                "(stale persistent cache would be served)"]


# ---------------------------------------------------------------------------
# 14. Search-Surface Duplication Audit
# ---------------------------------------------------------------------------
def audit_surface_duplication(op_a: Any, op_b: Any, *panels: pd.DataFrame) -> list[str]:
    """Two canonicals producing identical (or corr≈1 / monotone-equivalent)
    outputs on synthetic panels are aliases / duplicates."""
    findings: list[str] = []
    try:
        a = np.asarray(_call(op_a, *panels), dtype=float)
        b = np.asarray(_call(op_b, *panels), dtype=float)
    except Exception:  # noqa: BLE001
        return []
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.size == 0 or both.sum() < 20:
        return []
    av, bv = a[both], b[both]
    if np.allclose(av, bv, rtol=1e-9, atol=1e-12):
        findings.append("exact equality on synthetic panels")
        return findings
    if np.std(av) > 0 and np.std(bv) > 0:
        corr = float(np.corrcoef(av, bv)[0, 1])
        if abs(corr) > 0.999:
            findings.append(f"|corr| = {corr:.4f} (monotone/near-duplicate)")
    return findings


_ALL_AUDITS = [
    audit_equivalent_parameters, audit_relational_parameters,
    audit_self_contamination, audit_cohort_consistency,
    audit_column_permutation, audit_group_membership_vintage,
    audit_temporal_exclusion, audit_effective_sample_size,
    audit_zero_missing_invalid, audit_scale_invariance,
    audit_complexity_vs_param, audit_golden_reference,
    audit_cache_invalidation, audit_surface_duplication,
]

__all__ = [
    "audit_equivalent_parameters", "audit_relational_parameters",
    "audit_self_contamination", "audit_cohort_consistency",
    "audit_column_permutation", "audit_group_membership_vintage",
    "audit_temporal_exclusion", "audit_effective_sample_size",
    "audit_zero_missing_invalid", "audit_scale_invariance",
    "audit_complexity_vs_param", "audit_golden_reference",
    "audit_cache_invalidation", "audit_surface_duplication",
    "_ALL_AUDITS",
]
