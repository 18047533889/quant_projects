# -*- coding: utf-8 -*-
"""Automatic operator auditing primitives (R6 §33, R9 behavioral rework).

Fourteen audit families that convert the review's recurring failure classes
into deterministic, runnable checks.  Each audit is a pure function that takes
an operator (or a panel) and returns a list of :class:`AuditResult` records.
They are *scaffolding* primitives: the review recommends running them as a CI
gate before adding new operators, so an AI (or a human) can no longer reintroduce
the same failure classes by hand.  Wiring into the CI / load_all gate is tracked
separately (they are intentionally not wired into ``load_all`` yet to avoid
mass-failing existing operators before their evidence is regenerated).

R9 behavioral rework (R9-P0-028..032): the cohort, group-membership-vintage,
zero-vs-missing, scale-invariance and temporal-exclusion / effective-sample-size
audits are now behavior-based.  They run the operator on constructed panels and
report the observed metric; a "fail" corresponds to a real behavioral
observation (a NaN region the operator re-paired, a vintage leak, a zero-vs-NaN
conflation, non-scale-invariance, a missing warmup exclusion / absent
effective-sample gate), never to a source-text grep.

All audits are deterministic and operate on synthetic panels.  Every audit
returns a non-empty list of :class:`AuditResult`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class AuditResult:
    """Structured result of a single audit check.

    ``status`` is one of ``"pass"`` | ``"fail"`` | ``"info"`` |
    ``"inconclusive"`` (R10 #42: an audit that cannot be assessed — e.g. no
    comparable cells — is never ``"pass"``).  ``metric`` names the observable
    measured, ``expected`` states the contract value and ``observed`` records
    what the operator actually produced.
    """

    status: str
    test_name: str
    canonical: str
    metric: str
    expected: float | str
    observed: float | str


def _panel(n: int = 120, cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (n, cols)), axis=0),
                        index=idx, columns=[chr(ord("A") + i) for i in range(cols)])


def _call(op: Any, *panels: pd.DataFrame, **kwargs):
    return op.calculate(*panels, **kwargs)


def _canonical(op: Any) -> str:
    md = getattr(op, "metadata", None)
    name = getattr(md, "name", None)
    return str(name) if name else type(op).__name__


def _first_key(panel_keys: str | list[str]) -> str:
    return panel_keys[0] if isinstance(panel_keys, (list, tuple)) else panel_keys


# ---------------------------------------------------------------------------
# Declared-contract helpers (R10 #36-#42)
#
# The audits below must consume machine-readable contracts instead of assuming
# a single semantic for every operator.  The declared fields live on the
# operator's ``OperatorMetadata``, as direct attributes on the operator, or as
# ``name:value`` tags, so a real operator declares them in its metadata and a
# synthetic test operator declares them on the class.  Integration of the new
# fields into ``base.OperatorMetadata`` is deferred (R10 #40 note); every helper
# uses ``getattr`` so it works with or without the metadata fields present.
# ---------------------------------------------------------------------------

# R10 #40: declared metamorphic contract (scale / translation / sign / none).
class MetamorphicContract:
    """Machine-readable metamorphic-test contract an operator declares.

    ``scale_invariant``        output unchanged under x10 input scaling
    ``translation_invariant``  output unchanged under +C input shift
    ``sign_equivariant``       out(-x) == -out(x)
    ``unit_covariant``         output scales by the SAME factor as the input
    ``none``                   no metamorphic property asserted (the scale
                               audit must NOT assert scale invariance)
    """

    SCALE_INVARIANT = "scale_invariant"
    TRANSLATION_INVARIANT = "translation_invariant"
    SIGN_EQUIVARIANT = "sign_equivariant"
    UNIT_COVARIANT = "unit_covariant"
    NONE = "none"


# canonical -> declared metamorphic contract.  Seeded conservatively; the
# authoritative declaration is the operator's own metadata / attribute / tag,
# so this registry is only a fallback.  ``base.OperatorMetadata`` integration
# (a ``metamorphic_contract`` field) is deferred and tracked separately.
_METAMORPHIC_CONTRACTS: dict[str, str] = {
    # e.g. "cs_rank": MetamorphicContract.SCALE_INVARIANT,
}


def metamorphic_contract(canonical: str) -> str:
    """Declared metamorphic contract for ``canonical``, or ``"none"`` by
    default.  Consumers should prefer the operator's own metadata/attribute
    declaration and use this registry only as a fallback."""
    return _METAMORPHIC_CONTRACTS.get(canonical, MetamorphicContract.NONE)


def _declared_metamorphic(op: Any) -> tuple[bool, str]:
    """``(declared, contract)`` for an operator.

    ``declared=False`` means the operator carries NO contract — the scale audit
    keeps its historical scale-invariance assertion (backward compatible).
    ``declared=True`` with ``contract='none'`` means scale invariance must NOT
    be asserted.
    """
    direct = getattr(op, "metamorphic_contract", None)
    md = getattr(op, "metadata", None)
    if direct is None and md is not None:
        direct = getattr(md, "metamorphic_contract", None)
    if direct is not None:
        return True, str(direct).strip().lower()
    tags = list(getattr(md, "tags", None) or [])
    for t in tags:
        t = str(t)
        if t.startswith("metamorphic:"):
            return True, t.split(":", 1)[1].strip().lower()
    canon = _canonical(op)
    registered = _METAMORPHIC_CONTRACTS.get(canon)
    if registered is not None:
        return True, registered
    return False, MetamorphicContract.NONE


# R10 #37: group-membership-semantics contract.
_MEMBERSHIP_CURRENT = "current_members_retrospective"
_MEMBERSHIP_HISTORICAL = "historical_membership"


def _membership_semantics(op: Any) -> str | None:
    """Declared group-membership semantics, or ``None`` when undeclared.

    ``"current_members_retrospective"`` — today's labels only; historical
    membership changes must NOT affect today's group output.
    ``"historical_membership"`` — the operator is *supposed* to reflect
    historical membership; a reclassified history SHOULD change the output.
    """
    direct = getattr(op, "membership_semantics", None)
    md = getattr(op, "metadata", None)
    if direct is None and md is not None:
        direct = getattr(md, "membership_semantics", None)
    if direct is not None:
        s = str(direct).strip().lower()
        if s in ("current_members_retrospective", "current", "current_members"):
            return _MEMBERSHIP_CURRENT
        if s in ("historical_membership", "historical", "historical_members"):
            return _MEMBERSHIP_HISTORICAL
    tags = list(getattr(md, "tags", None) or [])
    for t in tags:
        t = str(t)
        low = t.strip().lower()
        if "current_members_retrospective" in low or low == "current_members":
            return _MEMBERSHIP_CURRENT
        if "historical_membership" in low or low == "historical_members":
            return _MEMBERSHIP_HISTORICAL
        if low.startswith("membership:"):
            val = t.split(":", 1)[1].strip().lower()
            if "current" in val:
                return _MEMBERSHIP_CURRENT
            if "historical" in val:
                return _MEMBERSHIP_HISTORICAL
    return None


# R10 #38: missing-data policy contract.
def _missing_policy(op: Any, kwargs: dict[str, Any]) -> str | None:
    """Declared missing-data policy (``"carry"`` / ``"break"`` / ...), or
    ``None`` when the operator declares none.  ``"carry"`` means a missing input
    carries the previous state instead of producing NaN, so a temporal-exclusion
    audit must NOT require NaN in a missing-input region for such an operator."""
    if "missing_policy" in kwargs:
        return str(kwargs["missing_policy"]).strip().lower()
    direct = getattr(op, "missing_policy", None)
    md = getattr(op, "metadata", None)
    if direct is None and md is not None:
        direct = getattr(md, "missing_policy", None)
    if direct is not None:
        return str(direct).strip().lower()
    import inspect

    sig = inspect.signature(getattr(op, "calculate", None))
    p = sig.parameters.get("missing_policy")
    if p is not None and p.default is not inspect.Parameter.empty:
        return str(p.default).strip().lower()
    tags = list(getattr(md, "tags", None) or [])
    for t in tags:
        t = str(t)
        if t.startswith("missing_policy:"):
            return t.split(":", 1)[1].strip().lower()
    return None


# R10 #39: effective-sample (DOF) contract.
def _dof_contract(op: Any) -> tuple[int | None, int | None]:
    """``(required_n, estimator_dof)`` declared by the operator, or ``None``s.

    ``required_n`` — the minimum number of observations needed before an
    estimate may be emitted.  ``estimator_dof`` — degrees of freedom consumed by
    the estimator (an estimate is only valid once ``effective_n > estimator_dof``).
    """
    md = getattr(op, "metadata", None)
    required_n = getattr(op, "required_n", None)
    if required_n is None and md is not None:
        required_n = getattr(md, "required_n", None)
    estimator_dof = getattr(op, "estimator_dof", None)
    if estimator_dof is None and md is not None:
        estimator_dof = getattr(md, "estimator_dof", None)
    tags = list(getattr(md, "tags", None) or [])
    for t in tags:
        t = str(t)
        if t.startswith("required_n:"):
            try:
                required_n = int(t.split(":", 1)[1])
            except (TypeError, ValueError):
                pass
        if t.startswith("estimator_dof:"):
            try:
                estimator_dof = int(t.split(":", 1)[1])
            except (TypeError, ValueError):
                pass
    return required_n, estimator_dof


# R10 #41: declared complexity class.
def _complexity_class(op: Any) -> str | None:
    md = getattr(op, "metadata", None)
    cls = getattr(op, "complexity", None)
    if cls is None and md is not None:
        cls = getattr(md, "complexity", None)
    if cls is not None:
        return str(cls).strip().lower()
    tags = list(getattr(md, "tags", None) or [])
    for t in tags:
        t = str(t)
        if t.startswith("complexity:"):
            return t.split(":", 1)[1].strip().lower()
    return None


# ---------------------------------------------------------------------------
# 1. Equivalent-Parameter Audit
# ---------------------------------------------------------------------------
def audit_equivalent_parameters(op: Any, *panels: pd.DataFrame,
                                param_grid: dict[str, list[Any]]) -> list[AuditResult]:
    """Two distinct parameter combinations producing byte-identical output are
    dead search nodes (``window=65/80``, ``top_k>rank``).  Reports pairs whose
    outputs match exactly."""
    canon = _canonical(op)
    base_kwargs = {k: v[0] for k, v in param_grid.items()}
    base = _call(op, *panels, **base_kwargs)
    results: list[AuditResult] = []
    for key, values in param_grid.items():
        for v in values[1:]:
            kw = dict(base_kwargs, **{key: v})
            out = _call(op, *panels, **kw)
            try:
                identical = base.equals(out)
            except AttributeError:  # scalar output
                identical = np.array_equal(np.asarray(base), np.asarray(out),
                                           equal_nan=True)
            if identical:
                results.append(AuditResult(
                    "fail", "audit_equivalent_parameters", canon,
                    f"identical_output({key}={v} vs {key}={base_kwargs[key]})",
                    "distinct outputs", "byte-identical output"))
    if not results:
        results.append(AuditResult(
            "pass", "audit_equivalent_parameters", canon,
            "equivalent_pairs_found", 0, 0))
    return results


# ---------------------------------------------------------------------------
# 2. Relational-Parameter Audit
# ---------------------------------------------------------------------------
def audit_relational_parameters(op: Any, *panels: pd.DataFrame,
                                infeasible: dict[str, Any]) -> list[AuditResult]:
    """A parameter combination the kernel can only fail (guaranteed-NaN) must
    be rejected by the validator, not silently produce all-NaN output."""
    canon = _canonical(op)
    try:
        out = _call(op, *panels, **infeasible)
    except (ValueError, TypeError) as exc:
        return [AuditResult(
            "pass", "audit_relational_parameters", canon,
            "infeasible_rejection", f"rejected {infeasible}",
            f"{type(exc).__name__}: {exc}")]
    arr = np.asarray(out, dtype=float)
    if np.isnan(arr).all():
        return [AuditResult(
            "fail", "audit_relational_parameters", canon,
            "infeasible_behavior", f"rejected {infeasible}",
            f"silently all-NaN for {infeasible}")]
    return [AuditResult(
        "fail", "audit_relational_parameters", canon,
        "infeasible_behavior", f"rejected {infeasible}",
        f"finite output for infeasible {infeasible}")]


# ---------------------------------------------------------------------------
# 3. Current-Self-Contamination Audit
# ---------------------------------------------------------------------------
def audit_self_contamination(op: Any, panel_key: str | list[str],
                             panels: dict[str, pd.DataFrame],
                             kwargs: dict[str, Any]) -> list[AuditResult]:
    """An anomaly/density/threshold operator must not let the CURRENT row change
    its own reference.  Perturb the last row wildly; the reference statistic
    (and thus the output at the last row) should reflect the perturbation, but
    the reference window shape should not change for earlier rows.

    ``panel_key`` may be a single name or a list of names (multi-input ops).
    Only the FIRST panel is perturbed; the rest stay fixed so a multi-input
    kernel's other inputs remain aligned."""
    canon = _canonical(op)
    keys = [panel_key] if isinstance(panel_key, str) else list(panel_key)
    local = dict(panels)
    try:
        base = _call(op, *[local[k] for k in keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_self_contamination", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    p = local[keys[0]].copy()
    p.iloc[-1, :] *= 1000.0
    local[keys[0]] = p
    try:
        out = _call(op, *[local[k] for k in keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_self_contamination", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    a = np.asarray(base, dtype=float)
    b = np.asarray(out, dtype=float)
    pre = ~np.isnan(a[:-1]) & ~np.isnan(b[:-1])
    diff = float(np.nanmax(np.abs(a[:-1][pre] - b[:-1][pre]))) if pre.any() else 0.0
    if pre.any() and not np.allclose(a[:-1][pre], b[:-1][pre], rtol=1e-9):
        return [AuditResult(
            "fail", "audit_self_contamination", canon,
            "prefix_max_abs_diff_after_last_row_perturb", 0.0, diff)]
    return [AuditResult(
        "pass", "audit_self_contamination", canon,
        "prefix_max_abs_diff_after_last_row_perturb", 0.0, diff)]


# ---------------------------------------------------------------------------
# 4. Cohort-Consistency Audit
# ---------------------------------------------------------------------------
def audit_cohort_consistency(op: Any, panel_keys: list[str],
                             panels: dict[str, pd.DataFrame],
                             kwargs: dict[str, Any]) -> list[AuditResult]:
    """Numerator and denominator must use the SAME finite cohort.

    Behavior-based (R9-P0-028): two scenarios are run on constructed gaps.

    * same-row gap — every input is NaN on the SAME rows; a cohort-consistent
      operator emits NaN on those rows (no value built from a mismatched cohort).
    * non-overlapping gap — input A is NaN on rows ``[base, base+3)`` while
      input B is NaN on ``[base+3, base+6)`` (DIFFERENT rows); a cohort-
      consistent operator must emit NaN on the UNION of both gap regions.  If it
      produces a finite value on any cell of the union it has re-paired data
      across the two cohorts (multi-input misalignment).

    Gap rows are chosen inside the operator's own all-finite (post-warmup)
    region so an operator warmup cannot mask the finding."""
    canon = _canonical(op)
    if len(panel_keys) < 2:
        return [AuditResult(
            "info", "audit_cohort_consistency", canon, "input_count",
            ">= 2 panels", f"{len(panel_keys)}")]
    n = len(panels[panel_keys[0]])
    if n < 12:
        return [AuditResult(
            "info", "audit_cohort_consistency", canon, "panel_length",
            ">= 12 rows", f"{n}")]
    try:
        clean = np.asarray(
            _call(op, *[panels[k] for k in panel_keys], **kwargs), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_cohort_consistency", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    stable = np.where(np.isfinite(clean).all(axis=1))[0]
    if stable.size < 6:
        return [AuditResult(
            "info", "audit_cohort_consistency", canon, "stable_rows",
            ">= 6 all-finite rows", f"{stable.size}")]
    base = stable[0]
    same_rows = list(range(base, base + 3))
    a_rows = list(range(base, base + 3))
    b_rows = list(range(base + 3, base + 6))

    results: list[AuditResult] = []

    # Case 1: same-row gap in every input.
    ps = {k: panels[k].copy() for k in panel_keys}
    for k in panel_keys:
        ps[k].iloc[same_rows, :] = np.nan
    try:
        outs = np.asarray(_call(op, *[ps[k] for k in panel_keys], **kwargs),
                          dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_cohort_consistency", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    cov_s = float(np.isnan(outs[same_rows, :]).mean())
    results.append(AuditResult(
        "pass" if cov_s >= 1.0 - 1e-9 else "fail",
        "audit_cohort_consistency", canon,
        "same_row_gap_nan_coverage", 1.0, cov_s))

    # Case 2: DIFFERENT rows — input[0] gap at a_rows, input[1] gap at b_rows.
    pd_ = {k: panels[k].copy() for k in panel_keys}
    pd_[panel_keys[0]].iloc[a_rows, :] = np.nan
    pd_[panel_keys[1]].iloc[b_rows, :] = np.nan
    try:
        outd = np.asarray(_call(op, *[pd_[k] for k in panel_keys], **kwargs),
                          dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_cohort_consistency", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    cov_d = float(np.isnan(outd[a_rows + b_rows, :]).mean())
    results.append(AuditResult(
        "pass" if cov_d >= 1.0 - 1e-9 else "fail",
        "audit_cohort_consistency", canon,
        "non_overlap_gap_nan_coverage", 1.0, cov_d))
    return results


# ---------------------------------------------------------------------------
# 5. Column-Permutation Invariance Audit
# ---------------------------------------------------------------------------
def audit_column_permutation(op: Any, panel_keys: list[str],
                             panels: dict[str, pd.DataFrame],
                             kwargs: dict[str, Any]) -> list[AuditResult]:
    """A per-column operator must be invariant to a permutation of stock columns
    (permute, run, un-permute, compare).  Catches isotonic/kNN tie bugs."""
    canon = _canonical(op)
    orig = {k: panels[k] for k in panel_keys}
    order = np.random.default_rng(1).permutation(list(orig[panel_keys[0]].columns))
    perm = {k: orig[k][order] for k in panel_keys}
    try:
        base = _call(op, *[orig[k] for k in panel_keys], **kwargs)
        out = _call(op, *[perm[k] for k in panel_keys], **kwargs)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_column_permutation", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    # R10 #36: the inverse permutation must restore the ORIGINAL column order,
    # NOT ``np.argsort(order)``.  ``order`` is a permutation of column LABELS;
    # argsorting labels sorts them lexicographically, which is the correct
    # inverse only when the original columns were already sorted.  For
    # arbitrary unordered labels (e.g. ['ZZZ','AAA','MNO']) it silently produces
    # a wrong alignment and can mask a real permutation-sensitivity bug.
    original_columns = list(orig[panel_keys[0]].columns)
    if isinstance(out, pd.DataFrame):
        out_re = out.reindex(columns=original_columns)
    else:
        rev_pos = [list(order).index(c) for c in original_columns]
        out_re = np.asarray(out)[:, rev_pos]
    a = np.asarray(base, dtype=float)
    b = np.asarray(out_re, dtype=float)
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0
    if both.any() and not np.allclose(a[both], b[both], rtol=1e-9):
        return [AuditResult(
            "fail", "audit_column_permutation", canon,
            "max_abs_diff_after_column_permutation", 0.0, diff)]
    return [AuditResult(
        "pass", "audit_column_permutation", canon,
        "max_abs_diff_after_column_permutation", 0.0, diff)]


# ---------------------------------------------------------------------------
# 6. Group-Membership Vintage Audit
# ---------------------------------------------------------------------------
def audit_group_membership_vintage(op: Any, panel_keys: list[str],
                                   panels: dict[str, pd.DataFrame],
                                   kwargs: dict[str, Any]) -> list[AuditResult]:
    """Changing a HISTORICAL group label must not change today's group output if
    the operator uses ``current_members_retrospective`` (today's labels only).

    Behavior-based (R9-P0-029): run the operator under a baseline (constant /
    uniform group) and under a reclassified history (stock C0 moved to another
    group for the FIRST half of the history, today's label untouched).  A
    current-membership operator produces the same factor in both runs (pass);
    an operator that leaks historical membership produces a materially different
    factor (fail)."""
    canon = _canonical(op)
    if "group_id" not in panel_keys:
        return [AuditResult(
            "info", "audit_group_membership_vintage", canon, "group_id_input",
            "group_id present", "missing")]
    value_keys = [k for k in panel_keys if k != "group_id"]
    if not value_keys:
        return [AuditResult(
            "info", "audit_group_membership_vintage", canon, "value_input",
            ">= 1 value panel", "none")]
    n = len(panels[value_keys[0]])
    if n < 4:
        return [AuditResult(
            "info", "audit_group_membership_vintage", canon, "panel_length",
            ">= 4 rows", f"{n}")]

    g_uniform = panels["group_id"].copy()
    g_uniform.iloc[:, :] = "G0"
    base_panels = {k: panels[k].copy() for k in panel_keys}
    base_panels["group_id"] = g_uniform
    try:
        out_base = np.asarray(
            _call(op, *[base_panels[k] for k in panel_keys], **kwargs),
            dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_group_membership_vintage", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]

    g_changed = g_uniform.copy()
    hist = slice(0, max(1, n // 2))
    g_changed.iloc[hist, 0] = "G1"  # reclassify stock C0 in the past only
    chg_panels = {k: panels[k].copy() for k in panel_keys}
    chg_panels["group_id"] = g_changed
    try:
        out_chg = np.asarray(
            _call(op, *[chg_panels[k] for k in panel_keys], **kwargs),
            dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_group_membership_vintage", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]

    both = np.isfinite(out_base) & np.isfinite(out_chg)
    if not both.any():
        return [AuditResult(
            "info", "audit_group_membership_vintage", canon, "finite_overlap",
            ">= 1 finite cell", "none")]
    scale = float(np.nanmax(np.abs(out_base[both])))
    diff = float(np.nanmax(np.abs(out_base[both] - out_chg[both])))
    tol = max(1e-8, 1e-6 * scale)
    # R10 #37: the audit must respect the operator's declared membership
    # semantics.  ``current_members_retrospective`` (today's labels only) is the
    # default assumption: reclassifying HISTORY must not change today's output
    # (a leak is a fail).  ``historical_membership`` is the OPPOSITE contract:
    # the operator is SUPPOSED to reflect historical membership, so a
    # reclassified history SHOULD change the output — the audit must NOT flag
    # that behavior; it flags only an operator that ignores its own declared
    # historical-membership contract (output unchanged).
    semantics = _membership_semantics(op)
    if semantics == _MEMBERSHIP_HISTORICAL:
        if diff <= tol:
            return [AuditResult(
                "fail", "audit_group_membership_vintage", canon,
                "historical_membership_honored", f"reclassified history changes "
                f"output (declared {_MEMBERSHIP_HISTORICAL})", diff)]
        return [AuditResult(
            "pass", "audit_group_membership_vintage", canon,
            "max_abs_diff_uniform_vs_reclassified_hist",
            f"> 0.0 (declared {_MEMBERSHIP_HISTORICAL})", diff)]
    return [AuditResult(
        "pass" if diff <= tol else "fail",
        "audit_group_membership_vintage", canon,
        "max_abs_diff_uniform_vs_reclassified_hist", 0.0, diff)]


# ---------------------------------------------------------------------------
# 7. Theiler / Temporal-Neighbour Audit
# ---------------------------------------------------------------------------
def audit_temporal_exclusion(op: Any, panel_keys: str | list[str],
                             panels: dict[str, pd.DataFrame],
                             kwargs: dict[str, Any]) -> list[AuditResult]:
    """Embedding/kNN/recurrence operators should keep the warmup / missing-data
    region NaN instead of fabricating a value from it.

    Behavior-based (R9-P0-031): the head of the first input panel is forced to
    NaN (a missing-history warmup region); a proper temporal-exclusion /
    fail-closed operator keeps that region NaN in its output.  A finite value
    in the forced-NaN region means the operator re-paired or fabricated data
    across the gap (no exclusion zone / no warmup handling).

    R10 #38: the audit honours the operator's declared MissingPolicy.  For a
    stateful operator whose policy is ``carry`` (missing input => carry previous
    state), a missing input after valid observations must NOT be audited as
    "should output NaN".  The leading warmup (no prior state) is still asserted
    NaN; an interior gap is reported as info — never as a fail."""
    canon = _canonical(op)
    key = _first_key(panel_keys)
    n = len(panels[key])
    if n < 4:
        return [AuditResult(
            "info", "audit_temporal_exclusion", canon, "panel_length",
            ">= 4 rows", f"{n}")]
    k = max(1, n // 6)
    keys = [key] if isinstance(panel_keys, str) else list(panel_keys)
    carry = (_missing_policy(op, kwargs) == "carry")

    # Leading warmup: force the head k rows NaN.  There is no prior state before
    # the head, so even a carry-state operator must keep this region NaN.
    pw = panels[key].copy()
    pw.iloc[:k, :] = np.nan
    local = {k_: (pw if k_ == key else panels[k_]) for k_ in panels}
    try:
        out = np.asarray(_call(op, *[local[k_] for k_ in keys], **kwargs),
                         dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_temporal_exclusion", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    cov = float(np.isnan(out[:k, :]).mean())
    results = [AuditResult(
        "pass" if cov >= 1.0 - 1e-9 else "fail",
        "audit_temporal_exclusion", canon,
        "warmup_missing_region_nan_coverage", 1.0, cov)]

    if carry:
        # Interior gap AFTER valid observations: a carry-state operator bridges
        # it with the last carried state, which is legitimate — report, don't
        # flag.  Only the leading warmup is fail-closed.
        stable = np.where(np.isfinite(out).all(axis=1))[0]
        base = int(stable[0]) if stable.size else k
        base = max(base, n // 2)
        gap = 2
        if 0 < base + gap <= n:
            pg = panels[key].copy()
            pg.iloc[base:base + gap, :] = np.nan
            local_g = {k_: (pg if k_ == key else panels[k_]) for k_ in panels}
            try:
                out_g = np.asarray(
                    _call(op, *[local_g[k_] for k_ in keys], **kwargs), dtype=float)
            except Exception as exc:  # noqa: BLE001
                return results + [AuditResult(
                    "fail", "audit_temporal_exclusion", canon, "run",
                    "finite output", f"{type(exc).__name__}: {exc}")]
            cov_g = float(np.isnan(out_g[base:base + gap, :]).mean())
            results.append(AuditResult(
                "info", "audit_temporal_exclusion", canon,
                "interior_gap_nan_coverage_carry_policy",
                "carry bridges the gap (NaN not required)", cov_g))
    return results


# ---------------------------------------------------------------------------
# 8. Effective-Sample-Size Audit
# ---------------------------------------------------------------------------
def audit_effective_sample_size(op: Any, panel_keys: str | list[str],
                                panels: dict[str, pd.DataFrame],
                                kwargs: dict[str, Any]) -> list[AuditResult]:
    """Overlapping windows / signatures should fail closed when the effective
    sample size is too small instead of reporting a nominal count.

    Behavior-based (R9-P0-031): run the operator on the full panel and on a
    data-starved (short) panel.  An effective-sample gate makes the short panel
    produce strictly MORE NaN (fewer finite cells) than the full panel; an
    operator without a gate emits the same fraction of finite cells regardless
    of sample size."""
    canon = _canonical(op)
    key = _first_key(panel_keys)
    n = len(panels[key])
    if n < 8:
        return [AuditResult(
            "info", "audit_effective_sample_size", canon, "panel_length",
            ">= 8 rows", f"{n}")]
    short_n = max(3, n // 6)
    short_panels = {
        k_: (panels[k_].iloc[:short_n].copy() if k_ == key else panels[k_].iloc[:short_n])
        for k_ in panels
    }
    keys = [key] if isinstance(panel_keys, str) else list(panel_keys)
    try:
        out_full = np.asarray(_call(op, *[panels[k_] for k_ in keys], **kwargs),
                              dtype=float)
        out_short = np.asarray(
            _call(op, *[short_panels[k_] for k_ in keys], **kwargs), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_effective_sample_size", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    full_fin = float(np.isfinite(out_full).mean())
    short_fin = float(np.isfinite(out_short).mean())
    if full_fin == 0.0:
        return [AuditResult(
            "info", "audit_effective_sample_size", canon, "full_panel_finite",
            "> 0", "0 — cannot assess the gate")]
    results = [AuditResult(
        "pass" if short_fin < full_fin else "fail",
        "audit_effective_sample_size", canon,
        "short_panel_finite_fraction",
        f"< {full_fin:.3f} (full-panel finite fraction)", short_fin)]
    # R10 #39: the finite-ratio check alone lets a short panel pass "just via
    # warmup" — an operator with a tiny warmup on a short panel has a high
    # finite ratio.  A DOF-aware gate is added whenever the operator declares an
    # effective-sample contract (``required_n`` / ``estimator_dof``): when the
    # short panel's effective sample falls below the declared requirement the
    # operator MUST NOT emit a finite estimate (fail-closed).
    required_n, estimator_dof = _dof_contract(op)
    if required_n is None and estimator_dof is None:
        return results
    eff = np.asarray(short_panels[key], dtype=float)
    effective_n = int(np.nanmin(np.isfinite(eff).sum(axis=0)))
    below = (required_n is not None and effective_n < required_n) or \
            (estimator_dof is not None and effective_n <= estimator_dof)
    if below:
        expected = "no finite output (effective sample below declared requirement)"
    else:
        expected = "finite output permitted (effective sample meets requirement)"
    dof_status = "fail" if (below and short_fin > 0.0) else "pass"
    results.append(AuditResult(
        dof_status, "audit_effective_sample_size", canon,
        "short_panel_dof_gate",
        f"effective_n={effective_n} required_n={required_n} "
        f"estimator_dof={estimator_dof} — {expected}",
        f"short_fin={short_fin:.3f}"))
    return results


# ---------------------------------------------------------------------------
# 9. Zero/Missing/Invalid Tri-state Audit
# ---------------------------------------------------------------------------
def audit_zero_missing_invalid(op: Any, panel_key: str,
                               panels: dict[str, pd.DataFrame],
                               kwargs: dict[str, Any]) -> list[AuditResult]:
    """``zero != missing != invalid``: a zero input must produce a distinct,
    documented output from a NaN input (not both NaN, not both the same value).

    Behavior-based (R9-P0-030): the last five rows are set to 0 in one panel and
    to NaN in another.  A correct tri-state operator must NOT produce the same
    output in the NaN region as in the zero region — a NaN input must never be
    treated as 0.  If the outputs are identical cell-for-cell, the operator
    conflates zero and missing."""
    canon = _canonical(op)
    p = panels[panel_key].copy()
    n = len(p)
    if n < 6:
        return [AuditResult(
            "info", "audit_zero_missing_invalid", canon, "panel_length",
            ">= 6 rows", f"{n}")]
    pz = p.copy()
    pz.iloc[-5:, :] = 0.0
    pn = p.copy()
    pn.iloc[-5:, :] = np.nan
    try:
        outz = np.asarray(_call(op, pz, **kwargs), dtype=float)
        outn = np.asarray(_call(op, pn, **kwargs), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_zero_missing_invalid", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    z = outz[-5:, :]
    m = outn[-5:, :]
    fin = np.isfinite(z)
    if not fin.any():
        return [AuditResult(
            "info", "audit_zero_missing_invalid", canon, "zero_region_finite",
            ">= 1 finite zero-region cell", "none — cannot assess")]
    identical = float(np.isclose(z[fin], m[fin], equal_nan=True).mean())
    return [AuditResult(
        "pass" if identical < 1.0 else "fail",
        "audit_zero_missing_invalid", canon,
        "zero_vs_nan_identical_fraction", 0.0, identical)]


# ---------------------------------------------------------------------------
# 10. Dimension/Unit-Scaling Audit
# ---------------------------------------------------------------------------
def audit_scale_invariance(op: Any, panel_key: str,
                           panels: dict[str, pd.DataFrame],
                           kwargs: dict[str, Any]) -> list[AuditResult]:
    """A theoretically scale-invariant operator must be invariant to input
    scaling (x10).  Scale-sensitive operators (e.g. level-preserving ones) are
    expected to change — the audit reports the output-scale ratio.

    R10 #40: the audit MUST NOT assume every operator is scale-invariant.  It
    reads the operator's declared :class:`MetamorphicContract`
    (``scale_invariant`` | ``translation_invariant`` | ``sign_equivariant`` |
    ``unit_covariant`` | ``none``).  For a declared ``none`` contract the audit
    does not assert scale invariance at all.  Undeclared operators keep the
    historical scale-invariance assertion (backward compatible).

    Behavior-based (R9-P0-030): ``input*10 -> output unchanged`` means the ratio
    of output scales is ~1 (scale-invariant).  A linear operator has ratio ~10
    (NOT invariant) and is flagged unless it declares ``unit_covariant``."""
    canon = _canonical(op)
    declared, contract = _declared_metamorphic(op)

    if declared and contract == MetamorphicContract.NONE:
        return [AuditResult(
            "info", "audit_scale_invariance", canon,
            "metamorphic_contract",
            "none — scale invariance NOT asserted", "skipped")]

    # --- translation-invariant contract -------------------------------
    if declared and contract == MetamorphicContract.TRANSLATION_INVARIANT:
        try:
            base = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
            shifted = panels[panel_key] + 100.0
            out_t = np.asarray(_call(op, shifted, **kwargs), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return [AuditResult(
                "fail", "audit_scale_invariance", canon, "run",
                "finite output", f"{type(exc).__name__}: {exc}")]
        both = np.isfinite(base) & np.isfinite(out_t)
        if not both.any():
            return [AuditResult(
                "info", "audit_scale_invariance", canon, "finite_cells",
                ">= 1 finite cell", "none")]
        diff = float(np.nanmax(np.abs(base[both] - out_t[both])))
        return [AuditResult(
            "pass" if diff <= 1e-8 else "fail",
            "audit_scale_invariance", canon,
            "max_abs_diff_after_translation", "~0.0 (translation-invariant)", diff)]

    # --- sign-equivariant contract ------------------------------------
    if declared and contract == MetamorphicContract.SIGN_EQUIVARIANT:
        try:
            out_pos = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
            out_neg = np.asarray(_call(op, -panels[panel_key], **kwargs), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return [AuditResult(
                "fail", "audit_scale_invariance", canon, "run",
                "finite output", f"{type(exc).__name__}: {exc}")]
        both = np.isfinite(out_pos) & np.isfinite(out_neg)
        if not both.any():
            return [AuditResult(
                "info", "audit_scale_invariance", canon, "finite_cells",
                ">= 1 finite cell", "none")]
        err = float(np.nanmax(np.abs(out_pos[both] + out_neg[both])))
        return [AuditResult(
            "pass" if err <= 1e-8 else "fail",
            "audit_scale_invariance", canon,
            "sign_equivariance_error", "~0.0 (out(-x) == -out(x))", err)]

    # --- unit-covariant contract --------------------------------------
    if declared and contract == MetamorphicContract.UNIT_COVARIANT:
        try:
            base = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
            out10 = np.asarray(_call(op, panels[panel_key] * 10.0, **kwargs),
                               dtype=float)
        except Exception as exc:  # noqa: BLE001
            return [AuditResult(
                "fail", "audit_scale_invariance", canon, "run",
                "finite output", f"{type(exc).__name__}: {exc}")]
        a = np.abs(base)
        b = np.abs(out10)
        both = np.isfinite(base) & np.isfinite(out10) & (a > 1e-12)
        if not both.any():
            return [AuditResult(
                "info", "audit_scale_invariance", canon, "finite_scale_cells",
                ">= 1 finite cell", "none")]
        ratio = float(np.nanmedian(b[both] / a[both]))
        return [AuditResult(
            "pass" if abs(ratio - 10.0) <= 2.5 else "fail",
            "audit_scale_invariance", canon,
            "output_scale_ratio_under_x10", "~10.0 (unit-covariant)", ratio)]

    # --- scale-invariant contract (declared or undeclared legacy) -----
    try:
        base = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
        p10 = panels[panel_key] * 10.0
        out10 = np.asarray(_call(op, p10, **kwargs), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_scale_invariance", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    a = np.abs(base)
    b = np.abs(out10)
    both = np.isfinite(base) & np.isfinite(out10) & (a > 1e-12)
    if not both.any():
        return [AuditResult(
            "info", "audit_scale_invariance", canon, "finite_scale_cells",
            ">= 1 finite cell", "none")]
    ratio = float(np.nanmedian(b[both] / a[both]))
    # invariant op: output unchanged under x10 input => ratio ~ 1
    metric = "output_scale_ratio_under_x10"
    expected = "~1.0 (scale-invariant)"
    if abs(ratio - 1.0) <= 0.25:
        status = "pass"
    else:
        status = "fail"
    return [AuditResult(status, "audit_scale_invariance", canon, metric,
                        expected, ratio)]


# ---------------------------------------------------------------------------
# 11. Complexity-vs-Param Audit
# ---------------------------------------------------------------------------
def audit_complexity_vs_param(op: Any, *panels: pd.DataFrame,
                              param_name: str, values: list[Any],
                              budget_multiplier: float | None = None) -> list[AuditResult]:
    """(a) DETERMINISTIC complexity-model audit (R10 #41).

    Runtime cost is NOT gated on a single wall-clock with a fixed multiplier.
    Instead the audit uses the operator's DECLARED cost contract — its own
    ``metadata.cost_model`` (R9-P1-047) or the deterministic
    ``operator_cost_model.runtime_cost`` prefix table — to compute the
    theoretical cost growth between the smallest and largest parameter values.
    When a complexity class is declared (``linear``/``quadratic``/``cubic``) the
    growth must match it; without a declared class the ratio is reported as
    info (no assertion).  ``budget_multiplier``, when passed explicitly, is the
    only hard ceiling used."""
    canon = _canonical(op)
    md = getattr(op, "metadata", None)
    declared_cm = getattr(md, "cost_model", None) if md is not None else None
    complexity = _complexity_class(op)
    shape = (len(panels[0]), len(panels[0].columns)) if panels else None

    def _theoretical_cost(params: dict[str, Any]) -> float | None:
        if callable(declared_cm):
            try:
                rt, _mem = declared_cm(params, shape)
                return max(1.0, float(rt))
            except Exception:  # noqa: BLE001
                return None
        try:
            import cleaned_operators.operator_cost_model as _ocm
            return max(1.0, float(_ocm.runtime_cost(canon, params)))
        except Exception:  # noqa: BLE001
            return None

    small_cost = _theoretical_cost({param_name: values[0]})
    large_cost = _theoretical_cost({param_name: values[-1]})
    if small_cost is None or large_cost is None or small_cost <= 0:
        return [AuditResult(
            "info", "audit_complexity_vs_param", canon, "cost_model",
            "deterministic cost model available", "unavailable")]
    ratio = large_cost / small_cost
    param_growth = float(values[-1] / values[0]) if values[0] else 1.0
    if complexity is None and budget_multiplier is None:
        return [AuditResult(
            "info", "audit_complexity_vs_param", canon,
            f"deterministic_cost_growth({values[0]}->{values[-1]})",
            "declared complexity class", f"{ratio:.2f}x")]
    if complexity is not None:
        exponent = {"linear": 1, "quadratic": 2, "cubic": 3}.get(complexity, 1)
        expected = (param_growth ** exponent) if param_growth >= 1 else 1.0
        bound = expected * 1.5  # 50% slack on the declared class
        expected_txt = f"<= {expected:.2f}x ({complexity})"
    else:
        bound = float(budget_multiplier)
        expected_txt = f"<= {bound:.0f}x (explicit budget)"
    return [AuditResult(
        "pass" if ratio <= bound else "fail",
        "audit_complexity_vs_param", canon,
        f"deterministic_cost_growth({values[0]}->{values[-1]})",
        expected_txt, f"{ratio:.2f}x")]


def audit_performance_benchmark(op: Any, *panels: pd.DataFrame,
                                params: dict[str, Any] | None = None,
                                max_seconds: float | None = None,
                                repeat: int = 3) -> list[AuditResult]:
    """(b) Wall-clock performance benchmark (R10 #41).

    Separate from the deterministic complexity-model audit.  Reports the best
    observed wall-clock time; flags ONLY when an explicit ``max_seconds`` budget
    is exceeded (passed in, or declared as ``op.max_benchmark_seconds`` /
    ``metadata`` field / ``benchmark_max_seconds:`` tag).  There is NO fixed
    multiplier gate: a single noisy wall-clock reading never decides pass/fail.
    """
    import time

    canon = _canonical(op)
    params = params or {}
    md = getattr(op, "metadata", None)
    budget = max_seconds
    if budget is None:
        budget = getattr(op, "max_benchmark_seconds", None)
        if budget is None and md is not None:
            budget = getattr(md, "max_benchmark_seconds", None)
        tags = list(getattr(md, "tags", None) or [])
        for t in tags:
            if str(t).startswith("benchmark_max_seconds:"):
                try:
                    budget = float(str(t).split(":", 1)[1])
                except (TypeError, ValueError):
                    pass
                break
    times: list[float] = []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        try:
            _call(op, *panels, **params)
        except Exception as exc:  # noqa: BLE001
            return [AuditResult(
                "fail", "audit_performance_benchmark", canon, "run",
                "finite output", f"{type(exc).__name__}: {exc}")]
        times.append(time.perf_counter() - t0)
    best = float(min(times))
    if budget is None:
        return [AuditResult(
            "info", "audit_performance_benchmark", canon,
            "wall_clock_seconds", "<= declared budget", f"{best:.4f}s")]
    return [AuditResult(
        "pass" if best <= budget else "fail",
        "audit_performance_benchmark", canon,
        "wall_clock_seconds", f"<= {budget:.3f}s", f"{best:.4f}s")]


# ---------------------------------------------------------------------------
# 12. Reference-Estimator Golden Audit (scaffold — no external refs bundled)
# ---------------------------------------------------------------------------
def audit_golden_reference(op: Any, panel_key: str,
                           panels: dict[str, pd.DataFrame],
                           kwargs: dict[str, Any],
                           reference: Callable[[np.ndarray], float] | None = None) -> list[AuditResult]:
    """Compare an operator against a trusted reference implementation on
    synthetic data.  ``reference`` is a callable panel-array -> array; pass one
    per operator in CI (e.g. scipy.stats, statsmodels, a golden dataset)."""
    canon = _canonical(op)
    if reference is None:
        return [AuditResult(
            "info", "audit_golden_reference", canon, "reference",
            "callable provided", "none (configure per operator)")]
    arr = np.asarray(panels[panel_key], dtype=float)
    try:
        ours = np.asarray(_call(op, panels[panel_key], **kwargs), dtype=float)
        gold = np.asarray(reference(arr), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return [AuditResult(
            "fail", "audit_golden_reference", canon, "run",
            "finite output", f"{type(exc).__name__}: {exc}")]
    both = ~np.isnan(ours) & ~np.isnan(gold)
    n_cmp = int(both.sum())
    if n_cmp == 0:
        # R10 #42: with zero comparable cells (no finite overlap between the
        # computed output and the golden reference) the result must be FAIL or
        # INCONCLUSIVE — NEVER PASS.  We have no evidence the operator is right.
        return [AuditResult(
            "inconclusive", "audit_golden_reference", canon,
            "comparable_cells", ">= 1 finite overlap", "0 — inconclusive")]
    diff = float(np.nanmax(np.abs(ours[both] - gold[both])))
    if not np.allclose(ours[both], gold[both], rtol=1e-6, atol=1e-9):
        return [AuditResult(
            "fail", "audit_golden_reference", canon,
            "max_abs_diff_vs_reference", 0.0, diff)]
    return [AuditResult(
        "pass", "audit_golden_reference", canon,
        "max_abs_diff_vs_reference", 0.0, diff)]


# ---------------------------------------------------------------------------
# 13. Cache Semantic Invalidation Audit (scaffold — runtime-level)
# ---------------------------------------------------------------------------
def audit_cache_invalidation(plan_hash_factory: Callable[[], str],
                             bump: Callable[[], None]) -> list[AuditResult]:
    """After bumping an implementation/field semantic/backend, a persisted plan
    key must change (100% cache miss)."""
    k1 = plan_hash_factory()
    bump()
    k2 = plan_hash_factory()
    if k1 != k2:
        return [AuditResult(
            "pass", "audit_cache_invalidation", "-", "plan_key_after_bump",
            "changed", "changed")]
    return [AuditResult(
        "fail", "audit_cache_invalidation", "-", "plan_key_after_bump",
        "changed", "unchanged (stale persistent cache would be served)")]


# ---------------------------------------------------------------------------
# 14. Search-Surface Duplication Audit
# ---------------------------------------------------------------------------
def audit_surface_duplication(op_a: Any, op_b: Any, *panels: pd.DataFrame) -> list[AuditResult]:
    """Two canonicals producing identical (or corr≈1 / monotone-equivalent)
    outputs on synthetic panels are aliases / duplicates."""
    canon = f"{_canonical(op_a)} vs {_canonical(op_b)}"
    try:
        a = np.asarray(_call(op_a, *panels), dtype=float)
        b = np.asarray(_call(op_b, *panels), dtype=float)
    except Exception:  # noqa: BLE001
        return [AuditResult(
            "info", "audit_surface_duplication", canon, "run",
            "finite output", "operator call failed")]
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.size == 0 or both.sum() < 20:
        return [AuditResult(
            "info", "audit_surface_duplication", canon, "finite_overlap",
            ">= 20 cells", f"{int(both.sum())}")]
    av, bv = a[both], b[both]
    if np.allclose(av, bv, rtol=1e-9, atol=1e-12):
        return [AuditResult(
            "fail", "audit_surface_duplication", canon, "output_equality",
            "distinct outputs", "exact equality on synthetic panels")]
    if np.std(av) > 0 and np.std(bv) > 0:
        corr = float(np.corrcoef(av, bv)[0, 1])
        if abs(corr) > 0.999:
            return [AuditResult(
                "fail", "audit_surface_duplication", canon, "output_corr",
                "< 0.999", f"{corr:.4f}")]
    return [AuditResult(
        "pass", "audit_surface_duplication", canon, "output_corr",
        "distinct", "no near-duplicate detected")]


def _legal_probe_values(spec: Any, current: Any) -> list[Any]:
    """Legal probe alternatives for a ParamSpec, or ``[]`` when none exist.

    R9-P0-035/§55: probes must come from the declared contract — never an
    arbitrary ``2*default`` that violates ``dtype=bool`` / ``choices`` / bounds.
    """
    if spec is None:
        return []
    choices = getattr(spec, "choices", None)
    if choices:
        return list(choices)
    dtype = getattr(spec, "dtype", None)
    lo = getattr(spec, "min", None)
    hi = getattr(spec, "max", None)
    if dtype is bool or dtype is bool.__class__ or str(getattr(dtype, "__name__", "")) == "bool":
        return [False, True] if isinstance(current, (bool, np.bool_)) else []
    if dtype is int or str(getattr(dtype, "__name__", "")) == "int":
        base = int(current) if isinstance(current, (int, float)) else 10
        if lo is not None and hi is not None and hi - lo >= 2:
            return [lo, (lo + hi) // 2, hi]
        return list({base, base + 1}) if isinstance(current, (int, float)) else []
    return []


def _output_signature(arr: np.ndarray) -> tuple[str, str, str]:
    """Multi-level output signature (R9-P1-039): finite-mask, rank-vector and
    quantized-normalised-value hashes — two different factors that happen to
    share mean/std/count no longer collide."""
    a = np.asarray(arr, dtype=float)
    mask = np.isfinite(a).astype(np.uint8)
    mask_hash = mask.tobytes().hex()
    flat = a.ravel()
    fin = flat[np.isfinite(flat)]
    if fin.size < 2:
        rank_hash = "none"
    else:
        order = np.argsort(fin, kind="stable")
        ranks = np.empty(fin.size, dtype=np.int64)
        ranks[order] = np.arange(fin.size)
        rank_hash = ranks.tobytes().hex()
    if fin.size < 2 or float(fin.max() - fin.min()) <= 1e-12:
        val_hash = "const"
    else:
        q = np.clip(np.rint(255.0 * (fin - fin.min()) / (fin.max() - fin.min())), 0, 255)
        val_hash = q.astype(np.uint8).tobytes().hex()
    return mask_hash, rank_hash, val_hash


def audit_searchable_params(op: Any, *panels: pd.DataFrame) -> list[AuditResult]:
    """R9-P0-035 / §55: every SEARCHABLE parameter needs three proofs.

    1. ``feasibility_verified`` — a legal value yields finite output;
    2. ``sensitivity_verified`` — changing the parameter changes the factor
       (multi-level output signature differs);
    3. ``non_equivalence_verified`` — different legal values are NOT the same
       factor (distinct rank / finite-mask / normalised-value signatures).

    Probes are drawn from the declared ParamSpec contract (choices / int grid /
    bool toggle) so the audit never evaluates an illegal value.  Returns one
    AuditResult per (param, proof) — never a bare bool.
    """
    md = getattr(op, "metadata", None)
    canon = _canonical(op)
    results: list[AuditResult] = []
    specs = dict(getattr(md, "param_specs", None) or {})
    searchable: list[str] = []
    for name, spec in specs.items():
        if getattr(spec, "searchable", True) is False:
            continue
        searchable.append(name)
    if not searchable:
        return [AuditResult("info", "audit_searchable_params", canon,
                            "searchable_params", ">= 1 declared",
                            "none declared")]
    import inspect

    sig = inspect.signature(getattr(op, "calculate", None))
    default_params = {p: (sig.parameters[p].default if sig.parameters[p].default is not inspect.Parameter.empty else None)
                      for p in list(sig.parameters)[1:]}  # skip self
    for name in searchable:
        current = default_params.get(name)
        probes = _legal_probe_values(specs.get(name), current)
        if len(probes) < 2:
            results.append(AuditResult(
                "info", "audit_searchable_params", canon, f"{name}.feasibility",
                ">= 2 legal probes", f"{len(probes)} probe(s)"))
            continue
        outputs: list[np.ndarray] = []
        for pv in probes:
            kwargs = {k: v for k, v in default_params.items() if v is not None}
            kwargs[name] = pv
            try:
                out = np.asarray(_call(op, *panels, **kwargs), dtype=float)
            except Exception:  # noqa: BLE001
                outputs.append(np.full(_panel().shape, np.nan))
                continue
            outputs.append(out)
        # feasibility: any legal probe yields any finite cell
        any_finite = any(np.isfinite(o).any() for o in outputs)
        results.append(AuditResult(
            "pass" if any_finite else "fail",
            "audit_searchable_params", canon, f"{name}.feasibility_verified",
            "legal value produces finite output", "finite output" if any_finite else "all NaN"))
        # sensitivity: at least two probes differ on the multi-level signature
        sigs = {_output_signature(o) for o in outputs}
        sensitive = len(sigs) >= 2
        results.append(AuditResult(
            "pass" if sensitive else "fail",
            "audit_searchable_params", canon, f"{name}.sensitivity_verified",
            "different params change the factor", f"{len(sigs)} distinct signature(s)"))
        # non-equivalence: differing signatures are not rank-identical
        rank_only = len({s[1] for s in sigs}) == 1 and len(sigs) > 1
        results.append(AuditResult(
            "fail" if rank_only else "pass",
            "audit_searchable_params", canon, f"{name}.non_equivalence_verified",
            "params not merely rank-equivalent",
            "rank-equivalent only" if rank_only else f"{len(sigs)} distinct signature(s)"))
    return results


_ALL_AUDITS = [
    audit_equivalent_parameters, audit_relational_parameters,
    audit_self_contamination, audit_cohort_consistency,
    audit_column_permutation, audit_group_membership_vintage,
    audit_temporal_exclusion, audit_effective_sample_size,
    audit_zero_missing_invalid, audit_scale_invariance,
    audit_complexity_vs_param, audit_performance_benchmark,
    audit_golden_reference, audit_cache_invalidation,
    audit_surface_duplication, audit_searchable_params,
]

__all__ = [
    "AuditResult",
    "MetamorphicContract",
    "metamorphic_contract",
    "audit_equivalent_parameters", "audit_relational_parameters",
    "audit_self_contamination", "audit_cohort_consistency",
    "audit_column_permutation", "audit_group_membership_vintage",
    "audit_temporal_exclusion", "audit_effective_sample_size",
    "audit_zero_missing_invalid", "audit_scale_invariance",
    "audit_complexity_vs_param", "audit_performance_benchmark",
    "audit_golden_reference", "audit_cache_invalidation",
    "audit_surface_duplication", "audit_searchable_params",
    "_ALL_AUDITS",
]
