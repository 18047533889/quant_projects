#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""promote_or_delete audit for every registered FactorEngine canonical.

For each canonical decide exactly one of the eight cases (AI plan §三十八):

  case 1  math correct + PIT + data available          -> certification (mine)
  case 2  math correct + high compute cost             -> advanced mining lane
  case 3  cannot be a standalone alpha                 -> condition/state/group/global role
  case 4  underlying math capability                   -> internal helper / recipe internal
  case 5  a better replacement exists                  -> delete old public canonical
  case 6  current data can never support it            -> delete canonical (no-data)
  case 7  future / random / non-causal                 -> permanently delete from public
  case 8  role could not be resolved                   -> UNKNOWN bucket (never default ALPHA)

Emits the final A–J (+K) table and verifies the §四十二 invariants.

R15 changes vs R12:

* Exit code is honest: any non-empty hard release invariant returns 1
  (R15-INC-042/258) — CI can no longer be faked green.
* The three hardcoded-empty invariants are now REAL detectors
  (R15-INC-043/257):
    - ``CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`` is a set difference against the
      current-context manifest;
    - ``SEMANTIC_DUPLICATE_CANONICALS`` is a real alias/name/AST candidate graph;
    - ``DEAD_SEARCHABLE_PARAMS`` is a bounded parameter-injectivity probe.
  A detector that did not run reports its coverage honestly instead of claiming
  PASS with ``[]``.
* ``UNRESOLVED`` roles land in a new UNKNOWN bucket (R15-INC-044/259); release
  requires UNKNOWN == 0.  Nothing falls back to optimistic case-1 ALPHA.
* ``mining usable`` is split into static-certified and per-context usable
  (R15-INC-046/248).
* A research candidate is reported separately from production-usable ALPHA
  (R15-INC-045).

R19 changes (R19-009..015):

* R19-009: ``main()`` gates on the single module-level
  ``_RELEASE_BLOCKING_INVARIANTS`` set — there is no second ``_HARD_INVARIANTS``
  list.
* R19-010: duplicate pairs are tagged by ``reason`` and split into four disjoint
  invariants — ``SEMANTIC_DUPLICATE_CANDIDATES`` (advisory),
  ``CONFIRMED_EXACT_DUPLICATES`` (the only release-blocking bucket: a direct
  alias edge), ``MONOTONIC_EQUIVALENT_PAIRS`` (mining dedup) and
  ``RELATED_FAMILY_PAIRS`` (non-blocking).
* R19-011: the dead-param probe uses REAL declared ``choices`` for a choices
  parameter; a parameter with no alternative distinct from its base is marked
  not-probeable (reason recorded) instead of inventing ``lo + 1``.
* R19-012: ``MISSING`` is not treated as a declared default (the sentinel is
  never fed into a kernel); ``None`` remains a legitimate declared default.
* R19-013: dead-param coverage is reported at PARAMETER level
  (``total_searchable_params`` / ``successfully_probed_params`` /
  ``dead_params`` / ``probe_error_params`` / ``untested_params``); release
  requires ``probe_error_params == 0`` and ``untested_params == 0``.
* R19-014: the behavior fingerprint canonicalizes non-finite values (NaN->0,
  ±Inf->fixed sentinels) before hashing so semantically-identical NaN payloads
  hash the same; the NaN / ±Inf masks are hashed separately.
* R19-015: ``CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`` is split into an
  intrinsic check (full known-source universe, no market filter) and a per-context
  check (``CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST[market,source_context]``) —
  a context-free query is never used to approximate a runtime environment.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mining.operator_catalog import MiningRole, RoleSource, assign_mining_role, assign_mining_role_ex, mining_eligible

# R19-012: ``MISSING`` is the "no declared default" sentinel — DISTINCT from
# ``None`` which is a legitimate declared default.  The old probe checked
# ``spec.default is not None`` which treated MISSING as a real default and fed
# the sentinel object into the kernel.  Imported with a defensive fallback so
# the detector degrades to the old ``is not None`` check if the sentinel is
# ever unavailable.
try:
    from cleaned_operators.base import MISSING as _MISSING_SENTINEL
except Exception:  # pragma: no cover - defensive fallback
    _MISSING_SENTINEL = None

_BUCKET_LABELS = {
    "A": "deleted permanently-unusable public operators",
    "B": "internal helper / raw math capability",
    "C": "deleted old diagnostic/legacy duplicate",
    "D": "ALPHA mineable",
    "E": "STATE/CONDITION/EVENT mineable",
    "F": "GROUP/GLOBAL state mineable (interaction only)",
    "G": "INTRADAY_EOD mineable",
    "H": "ADVANCED/HIGH_COST mineable",
    "I": "SOURCE_TRANSFORM",
    "J": "deleted due to missing data",
    "K": "UNRESOLVED role (R15 fail-closed: never default ALPHA)",
}

# R16-008 / R19-009: release-critical invariants, marked EXPLICITLY.  This is
# the SINGLE source of truth the CLI ``main()`` gates on — there is no second
# ``_HARD_INVARIANTS`` list to keep in sync.  ``COMPAT_ALIAS_AS_CANONICAL`` is
# advisory (an alias-only compat surface is informational).  R19-010: only the
# confirmed exact-duplicate bucket gates release; name-family / monotonic /
# advisory candidates never do.  R19-015: the INTRINSIC manifest invariant
# gates; the per-context one is informational for the operator's own environment.
_RELEASE_BLOCKING_INVARIANTS = frozenset({
    "UNCLASSIFIED_FACTOR_CANONICALS",
    "MINING_ROLE_UNRESOLVED",
    "UNUSED_PUBLIC_CANONICALS",
    "FACTOR_SHAPED_RESEARCH_ONLY",
    "MINING_ELIGIBLE_WITHOUT_CERTIFICATION",
    "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY",
    "CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST",
    "CONFIRMED_EXACT_DUPLICATES",
    "DEAD_SEARCHABLE_PARAMS",
})


@dataclass
class Verdict:
    canonical: str
    case: int
    bucket: str
    role: str
    role_source: str
    mining_eligible: bool
    blockers: list[str]
    recommended_action: str


def _case_and_bucket(canonical: str, role: MiningRole, catalog: dict[str, Any]) -> tuple[int, str]:
    from cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS
    from cleaned_operators.production_hardening import (
        NON_FACTOR_PRODUCTION_CANONICALS,
        SOURCE_BLOCKED_CANONICALS,
    )

    # R15-INC-044/259: an unresolved role is its own bucket, never optimistic D.
    if role is MiningRole.UNRESOLVED:
        return 8, "K"
    if role is MiningRole.DENIED:
        if canonical in {
            "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
            "interpolate", "dropna", "shuffle", "sample",
        } or canonical.startswith("rand_"):
            return 7, "A"
        if canonical in {
            "norm", "norm_l1", "norm_linf", "fft", "ifft", "wavelet", "convolve",
            "correlate", "mat_inverse", "eig", "svd", "pca",
        }:
            return 4, "B"
        if canonical in {"arg", "tan", "cot", "sec", "csc", "cosh", "sinh"}:
            return 7, "A"
        return 7, "A"
    if role in (MiningRole.INTERNAL, MiningRole.RECIPE_INTERNAL):
        if canonical in NON_FACTOR_PRODUCTION_CANONICALS:
            return 7, "A"
        return 4, "B"
    if role in (MiningRole.DIAGNOSTIC, MiningRole.LEGACY):
        return 5, "C"
    if role is MiningRole.SOURCE_TRANSFORM:
        return 4, "I"
    # Source-blocked wins over the role's default bucket: an operator with no
    # physical field/vintage cannot be mined regardless of its factor shape.
    if canonical in SOURCE_BLOCKED_CANONICALS:
        return 6, "J"
    if role is MiningRole.RESEARCH:
        # factor-shaped research: case 1 (needs certification pipeline)
        if catalog.get("diagnostic_only") or catalog.get("benchmark_only"):
            return 5, "C"
        return 1, "D"
    if role is MiningRole.GLOBAL_STATE:
        return 3, "F"
    if role is MiningRole.GROUP_STATE:
        return 3, "F"
    if role in (MiningRole.STATE, MiningRole.CONDITION, MiningRole.EVENT):
        return 3, "E"
    if role is MiningRole.INTRADAY_EOD:
        return 1, "G"
    if role is MiningRole.FUNDAMENTAL_PIT:
        return 1, "D"
    if role is MiningRole.ALPHA_HIGH_COST:
        return 2, "H"
    if role is MiningRole.ALPHA:
        return 1, "D"
    return 8, "K"


# ---------------------------------------------------------------------------
# Real invariant detectors (R15-INC-043/049/050/051/257/262/263)
# ---------------------------------------------------------------------------

def _alias_edges(catalog: dict[str, Any]) -> list[tuple[str, str]]:
    """(alias, canonical) edges from the registry."""
    from cleaned_operators.registry import OperatorRegistry

    edges: list[tuple[str, str]] = []
    aliases = getattr(OperatorRegistry, "_aliases", {}) or {}
    for alias, target in aliases.items():
        edges.append((str(alias), str(target)))
    for canon, entry in catalog.items():
        for alias in (entry.get("aliases") or ()):
            edges.append((str(alias), str(canon)))
    return edges


# R19-010: duplicate pairs are classified by ``reason`` into four DISJOINT
# buckets.  Only ``CONFIRMED_EXACT_DUPLICATES`` (a direct alias edge — the two
# names are literally the same operator object) blocks release.  Alias / name-
# family / monotonic / advisory-candidate signals used to be lumped together in
# one release-blocking ``SEMANTIC_DUPLICATE_CANONICALS`` list; they are now
# reported separately so a mere name-family or advisory candidate can never
# gate a release.
_DUPLICATE_REASON_BUCKET: dict[str, str] = {
    "alias": "CONFIRMED_EXACT_DUPLICATES",
    "name_family": "RELATED_FAMILY_PAIRS",
    "monotonic": "MONOTONIC_EQUIVALENT_PAIRS",
    "candidate": "SEMANTIC_DUPLICATE_CANDIDATES",
}

# R18-021 strict-monotonic re-encoding classes (``mining/direct_use``): members
# of the SAME class are interchangeable in a search space (x / exp(x) /
# sigmoid(x) / rank(x) are one capability family).  ``monotonic_piecewise`` is
# NOT strict (``square`` is not monotonic on the negative domain) and never
# pairs.
_STRICT_MONOTONIC_CLASSES = frozenset(
    {"monotonic_increasing", "monotonic_equiv", "affine_equiv"}
)


def _bucket_duplicate_pairs(pairs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Split reason-tagged duplicate pairs into the four release invariants."""
    buckets: dict[str, list[str]] = {b: [] for b in set(_DUPLICATE_REASON_BUCKET.values())}
    for p in pairs:
        bucket = _DUPLICATE_REASON_BUCKET.get(p.get("reason"), "SEMANTIC_DUPLICATE_CANDIDATES")
        buckets[bucket].append(f"{p['left']} <-> {p['right']} ({p['reason']})")
    return buckets


def detect_semantic_duplicate_candidates(
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    """R15-INC-049/262: candidate duplicate public canonicals.

    A real detector combines (a) the alias graph and (b) canonical-name
    families (``MACD`` vs ``MACD_line``).  R16-010: an identical
    ``(param_names, output_unit, input_grain)`` contract is NO LONGER treated as
    a duplicate — ``ts_mean`` and ``ts_std`` share the exact same interface yet
    are mathematically different.  Identical interface is only candidate material;
    a true duplicate requires implementation/AST fingerprint + golden behavioural
    + metamorphic equivalence (the machine-audit layer, R16-225).  Candidates
    require human confirmation — the detector never silently deletes.

    R19-010: every pair is tagged with a ``reason`` — ``alias`` (proven exact:
    one name resolves to the other), ``name_family`` (word-boundary name
    variant), ``candidate`` (the base is a monotonic-transform canonical whose
    longer variant is a possibly-composite family member — monotonic equivalence
    cannot be proven statically), and ``monotonic`` (both canonicals are the same
    strict-monotonic re-encoding family).  Only the ``alias`` bucket blocks
    release.
    """
    candidates: list[dict[str, Any]] = []
    from cleaned_operators.operator_surface import classify_canonical

    try:
        from mining.direct_use import MONOTONIC_TRANSFORM_CLASS
    except Exception:  # pragma: no cover - read-only optional import
        MONOTONIC_TRANSFORM_CLASS = {}

    names = sorted(
        c for c in catalog
        if classify_canonical(c) in ("daily", "extended")
        and not catalog[c].get("compatibility_only")
    )
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = (a, b)
            if key in seen:
                continue
            # alias edge in either direction
            if (a in _ALIAS_LOOKUP and _ALIAS_LOOKUP[a] == b) or (
                b in _ALIAS_LOOKUP and _ALIAS_LOOKUP[b] == a
            ):
                seen.add(key)
                candidates.append(
                    {"left": a, "right": b, "reason": "alias", "confidence": "high"}
                )
                continue
            # name-family: one name is the other with a suffix/prefix variant
            # (``MACD`` in ``MACD_line``).  The shorter name must match at a
            # ``_`` / start / end WORD BOUNDARY so 2-letter operators (``lt``,
            # ``ne``) and mid-word substrings (``sign`` inside ``MACD_signal``)
            # never fire.
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            boundary = re.compile(
                r"(?:^|_)" + re.escape(shorter) + r"(?=_|$)"
            )
            if boundary.search(longer):
                seen.add(key)
                if MONOTONIC_TRANSFORM_CLASS.get(shorter):
                    # The base name is a monotonic transform canonical; the
                    # longer name is a possibly-COMPOSITE family member
                    # (``log`` vs ``ts_log_return``).  Monotonic equivalence is
                    # not provable statically -> advisory candidate.
                    reason = "candidate"
                else:
                    reason = "name_family"
                candidates.append(
                    {"left": a, "right": b, "reason": reason, "confidence": "candidate"}
                )
                continue
    # R19-010: MONOTONIC_EQUIVALENT_PAIRS — two canonicals that are BOTH the same
    # strict-monotonic re-encoding family are interchangeable in a search space
    # (mining dedup) even though neither name contains the other.
    if MONOTONIC_TRANSFORM_CLASS:
        by_class: dict[str, list[str]] = {}
        for n in names:
            cls = MONOTONIC_TRANSFORM_CLASS.get(n)
            if cls and cls in _STRICT_MONOTONIC_CLASSES:
                by_class.setdefault(cls, []).append(n)
        for members in by_class.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (members[i], members[j])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {"left": members[i], "right": members[j],
                         "reason": "monotonic", "confidence": "candidate"}
                    )
    return candidates


def _panel_arg_names(operator: Any) -> list[str]:
    """Typed panel-parameter names for a kernel (R16-005).

    The dead-param runner must bind the FULL declared panel inputs through the
    typed contract — never ``calculate(panel['x'], ...)`` for a binary/ternary/
    variadic operator (which throws, gets swallowed, and the operator silently
    leaves the audit).  ``metadata.panel_params`` is authoritative; a signature
    heuristic (leading no-default positional params, excluding declared scalars)
    covers operators that declare panels only in the kernel.
    """
    md = getattr(operator, "metadata", None)
    pp = getattr(md, "panel_params", None)
    if pp:
        return [p for p in pp if p not in ("", "_fn")]
    scalar_names = set(getattr(md, "scalar_params", None) or ())
    fn = getattr(operator, "_calculate_series", None) or getattr(operator, "calculate", None)
    if fn is not None:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            panels: list[str] = []
            for name, param in sig.parameters.items():
                if name in ("self", "_fn", "kwargs") or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if name in scalar_names:
                    continue
                if param.default is not inspect.Parameter.empty:
                    # first defaulted param ends the positional panel prefix
                    break
                panels.append(name)
            if panels:
                return panels
    return ["x"]


def _synthetic_panels(operator: Any) -> dict[str, Any]:
    """Deterministic synthetic panels keyed by panel-param name (R16-005).

    Every declared panel slot gets its OWN data so a parameter that changes
    behavior only through a second/third input is not masked by feeding the
    same array everywhere.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    rows, cols = 80, 5
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    columns = [f"s{i}" for i in range(cols)]
    panels: dict[str, Any] = {}
    for name in _panel_arg_names(operator):
        panels[name] = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=columns)
    if not panels:
        panels["x"] = pd.DataFrame(rng.normal(size=(rows, cols)), index=idx, columns=columns)
    return panels


def _behavior_fingerprint(out: Any) -> str:
    """Full behavioral fingerprint (R16-006).

    Comparing only finite values let two different missing-masks hash equal — a
    parameter whose change only altered missingness was misjudged equivalent.
    The fingerprint hashes shape / index / columns / raw values / NaN mask /
    ±Inf mask, so ANY behavioral difference (including NaN topology) changes it.

    R19-014: the value bytes are CANONICALIZED before hashing — NaN -> 0.0,
    +Inf -> a fixed sentinel, -Inf -> the negative sentinel — so two results
    that differ only in the NaN PAYLOAD bits (signaling vs quiet NaN, signed
    NaN) but are semantically identical hash to the same fingerprint.  The
    non-finite masks are hashed separately, so missingness / ±Inf topology still
    changes the hash.
    """
    import hashlib
    import json

    import numpy as np

    h = hashlib.sha256()
    h.update(f"{out.shape[0]}x{out.shape[1]}".encode("utf-8"))
    idx_head = [str(i) for i in (out.index[:40] if len(out.index) > 40 else out.index)]
    h.update(json.dumps(idx_head, ensure_ascii=False).encode("utf-8"))
    col_head = [str(c) for c in (out.columns[:40] if len(out.columns) > 40 else out.columns)]
    h.update(json.dumps(col_head, ensure_ascii=False).encode("utf-8"))
    arr = np.asarray(out, dtype=np.float64)
    nan_mask = np.isnan(arr)
    posinf_mask = np.isposinf(arr)
    neginf_mask = np.isneginf(arr)
    canonical = arr.copy()
    canonical[nan_mask] = 0.0
    canonical[posinf_mask] = np.finfo(np.float64).max
    canonical[neginf_mask] = -np.finfo(np.float64).max
    h.update(canonical.tobytes())
    h.update(nan_mask.tobytes())
    h.update(posinf_mask.tobytes())
    h.update(neginf_mask.tobytes())
    return h.hexdigest()[:16]


def _param_spec(operator: Any, name: str) -> Any:
    """Look up a searchable parameter's ParamSpec (best-effort)."""
    try:
        return operator.metadata.param_specs.get(name)
    except Exception:
        return None


def _base_param_value(operator: Any, name: str, spec: Any) -> Any:
    """Base (default) value for a searchable parameter.

    R19-012: ``spec.default is not MISSING`` — the MISSING sentinel means "no
    default declared" and must NOT be fed into the kernel.  ``None`` is a
    legitimate declared default and IS used.  Without a declared default, fall
    back to a type-consistent synthetic value.
    """
    if spec is not None and spec.default is not _MISSING_SENTINEL:
        return spec.default
    try:
        declared_type = operator.metadata.param_types.get(name)
    except Exception:
        declared_type = None
    if declared_type is int:
        return 10
    if declared_type is float:
        return 0.5
    return 10


def _param_alternatives(spec: Any, base_value: Any) -> tuple[list[Any], str | None]:
    """Legal probe values distinct from ``base_value`` for an injectivity check.

    R19-011: when ``spec.choices`` is declared, EVERY real choice other than the
    base is probed — the old code invented ``lo + 1`` which is not a legal
    choice (the kernel would raise or silently clamp, poisoning the probe).  A
    parameter with no alternative distinct from the base is NOT probeable: it is
    neither dead nor an error — the caller records ``reason`` and counts it as
    an untested parameter (R19-013: release requires zero untested).

    Returns ``(alternatives, reason)``; ``reason`` is ``None`` when probeable.
    """
    if spec is None:
        return [], "no ParamSpec for searchable param"
    if spec.choices:
        alternatives = [c for c in spec.choices if c != base_value]
        if not alternatives:
            return [], "no choice distinct from base"
        return alternatives, None
    lo = spec.min if spec.min is not None else 1
    hi = spec.max if spec.max is not None else (lo + 2)
    if hi < lo:
        hi = lo + 1
    for value in (lo + 1, hi, lo, hi + 1):
        in_min = spec.min is None or value >= spec.min
        in_max = spec.max is None or value <= spec.max
        if value != base_value and in_min and in_max:
            return [value], None
    return [], "no distinct legal value"


def _run_op_hash(
    operator: Any,
    panels: dict[str, Any],
    searchable: list[str],
    *,
    override: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Run the kernel on the full panel binding and fingerprint the output.

    Returns ``(fingerprint, error_summary)``.  ``override`` supplies an exact
    alternative value for a parameter (the 'different value' side of
    injectivity); the caller computes legal alternatives via
    :func:`_param_alternatives`.  R16-007: an exception is NEVER silently
    skipped — it is returned as the ``error_summary`` so the caller records an
    AUDIT_ERROR for the canonical.
    """
    error: str | None = None
    try:
        kwargs: dict[str, Any] = {}
        for name in searchable:
            spec = _param_spec(operator, name)
            if name in (override or ()):
                kwargs[name] = override[name]
            else:
                kwargs[name] = _base_param_value(operator, name, spec)
        # R16-005: bind the FULL declared panel inputs, in declared order.
        panel_names = _panel_arg_names(operator)
        missing = [p for p in panel_names if p not in panels]
        if missing:
            return None, f"missing synthetic panels for {missing}"
        args = [panels[p] for p in panel_names]
        out = operator.calculate(*args, **kwargs)
        if out is None:
            return None, "kernel returned None"
        return _behavior_fingerprint(out), None
    except Exception as exc:  # noqa: BLE001 — the error becomes an AUDIT_ERROR
        error = f"{type(exc).__name__}: {exc}"
    return None, error


def detect_dead_searchable_params_detail(catalog: dict[str, Any]) -> dict[str, Any]:
    """R16-004 + R19-013: FULL parameter-injectivity probe — no sampling.

    EVERY ALPHA/factor operator with a searchable param gets a machine outcome:
    run the kernel on a fixed synthetic panel (full typed panel binding, R16-005)
    and change ONE searchable parameter across legal values; identical full
    behavior fingerprint (incl. NaN topology, R16-006) => dead candidate.
    An operator that could not be constructed/executed is reported as
    AUDIT_ERROR (R16-007), never silently dropped.

    R19-011: a parameter with a declared ``choices`` tuple is probed with every
    REAL choice distinct from the base — never an invented ``lo + 1``.  A
    parameter with no alternative distinct from the base is recorded in
    ``not_probeable`` (reason) and counted under ``untested_params``.

    R19-012: a MISSING default is treated as "no default declared" — the
    sentinel object is never fed into the kernel.

    R19-013: coverage is reported at PARAMETER level
    (``total_searchable_params`` / ``successfully_probed_params`` /
    ``dead_params`` / ``probe_error_params`` / ``untested_params``) in addition
    to the operator-level ``audited_operators`` / ``total_operators``.  Release
    requires ``probe_error_params == 0`` and ``untested_params == 0``.
    """
    dead: list[dict[str, Any]] = []
    audit_errors: list[dict[str, Any]] = []
    not_probeable: list[dict[str, Any]] = []
    audited_operators = 0
    total_operators = 0
    total_searchable_params = 0
    successfully_probed_params = 0
    dead_params = 0
    probe_error_params = 0
    untested_params = 0
    try:
        from cleaned_operators.base import searchable_param_names
        from cleaned_operators.registry import OperatorRegistry
    except Exception as exc:
        audit_errors.append({"canonical": "<import>", "param": "*", "error": str(exc)})
        return {
            "dead": dead,
            "audit_errors": audit_errors,
            "not_probeable": not_probeable,
            "audited_operators": audited_operators,
            "total_operators": total_operators,
            "total_searchable_params": total_searchable_params,
            "successfully_probed_params": successfully_probed_params,
            "dead_params": dead_params,
            "probe_error_params": probe_error_params,
            "untested_params": untested_params,
        }

    for canonical in sorted(catalog):
        entry = catalog[canonical]
        if entry.get("compatibility_only") or entry.get("diagnostic_only"):
            continue
        if assign_mining_role(canonical, entry) not in (
            MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
            MiningRole.FUNDAMENTAL_PIT, MiningRole.INTRADAY_EOD,
        ):
            continue
        try:
            operator = OperatorRegistry.get(canonical, "pandas_numpy")
            grades = searchable_param_names(getattr(operator, "metadata", None))
        except Exception as exc:
            audit_errors.append({"canonical": canonical, "param": "*", "error": str(exc)})
            continue
        searchable = sorted(set(grades.get("full", ())) | set(grades.get("coarse", ())))
        if not searchable:
            continue
        total_operators += 1
        total_searchable_params += len(searchable)
        panels = _synthetic_panels(operator)
        base, base_err = _run_op_hash(operator, panels, searchable)
        if base_err is not None:
            audit_errors.append({"canonical": canonical, "param": "*", "error": base_err})
            untested_params += len(searchable)
            continue
        audited_operators += 1
        for name in searchable:
            spec = _param_spec(operator, name)
            base_value = _base_param_value(operator, name, spec)
            alternatives, reason = _param_alternatives(spec, base_value)
            if reason is not None:
                not_probeable.append({"canonical": canonical, "param": name, "reason": reason})
                untested_params += 1
                continue
            changed = False
            probed_any = False
            alt_err_summary: str | None = None
            for alt in alternatives:
                alt_fp, alt_err = _run_op_hash(
                    operator, panels, searchable, override={name: alt}
                )
                if alt_err is not None:
                    alt_err_summary = alt_err
                    break
                probed_any = True
                if alt_fp != base:
                    changed = True
                    break
            if alt_err_summary is not None:
                audit_errors.append({"canonical": canonical, "param": name, "error": alt_err_summary})
                probe_error_params += 1
                continue
            if not probed_any:
                # Defensive: no alternative actually executed.
                not_probeable.append({"canonical": canonical, "param": name, "reason": "no probe executed"})
                untested_params += 1
                continue
            successfully_probed_params += 1
            if not changed:
                dead.append({"canonical": canonical, "param": name, "evidence": "injectivity"})
                dead_params += 1
    return {
        "dead": dead,
        "audit_errors": audit_errors,
        "not_probeable": not_probeable,
        "audited_operators": audited_operators,
        "total_operators": total_operators,
        "total_searchable_params": total_searchable_params,
        "successfully_probed_params": successfully_probed_params,
        "dead_params": dead_params,
        "probe_error_params": probe_error_params,
        "untested_params": untested_params,
    }


def detect_dead_searchable_params(
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    """R16-004 backward-compatible ``(dead, audited, total, audit_errors)``.

    Kept as a thin wrapper so callers of the pre-R19 4-tuple contract (regression
    tests) keep working; the audit uses :func:`detect_dead_searchable_params_detail`
    which also reports parameter-level coverage.
    """
    detail = detect_dead_searchable_params_detail(catalog)
    return (
        detail["dead"],
        detail["audited_operators"],
        detail["total_operators"],
        detail["audit_errors"],
    )


def detect_manifest_gap(catalog: dict[str, Any]) -> list[str]:
    """R15-INC-043/051 backward-compatible context-free eligible-manifest gap.

    .. note:: R19-015 — this is kept ONLY as a backward-compat shim for the R16
       regression suite.  The release gate no longer consumes it: a context-free
       ``available_sources=None`` query fails closed on every source-requiring
       operator (R15-INC-004) and therefore reports a near-empty manifest (false
       gaps).  The audit uses :func:`detect_manifest_gap_intrinsic` and
       :func:`detect_manifest_gap_contextual` instead.
    """
    try:
        from mining.operator_catalog import get_mining_operators
    except Exception as exc:
        # A detector that cannot run must not claim PASS with [] — the caller
        # records this as AUDIT_ERROR coverage.
        raise RuntimeError(f"detect_manifest_gap: mining catalog unavailable: {exc}") from exc
    manifest_eligible = {op.canonical for op in get_mining_operators(admission="eligible")}
    missing: list[str] = []
    for canonical, entry in sorted(catalog.items()):
        if entry.get("production_certified"):
            role = assign_mining_role(canonical, entry)
            if role in (MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
                        MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT):
                if canonical not in manifest_eligible:
                    missing.append(canonical)
    return missing


# R19-015: fallback source universe if ``mining.operator_catalog._KNOWN_SOURCES``
# is unavailable (read-only import — never a hard dependency).
_KNOWN_SOURCES_FALLBACK = frozenset({
    "daily_bar", "minute_bar", "fundamental_pit", "shareholder_pit",
    "relation_pit", "event_pit", "index_pit",
})


def _known_source_universe() -> tuple[str, ...]:
    try:
        from mining.operator_catalog import _KNOWN_SOURCES
        if _KNOWN_SOURCES:
            return tuple(sorted(_KNOWN_SOURCES))
    except Exception:
        pass
    return tuple(sorted(_KNOWN_SOURCES_FALLBACK))


def _certified_factor_canonicals(catalog: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for canonical, entry in sorted(catalog.items()):
        if entry.get("production_certified"):
            role = assign_mining_role(canonical, entry)
            if role in (MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
                        MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT):
                out.append((canonical, entry))
    return out


def detect_manifest_gap_intrinsic(catalog: dict[str, Any]) -> list[str]:
    """R19-015: ``CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST``.

    Every production-certified factor operator must be reachable in the eligible
    mining manifest when ALL KNOWN sources are hypothetically available and no
    market filter is applied — i.e. an INTRINSIC capability check (certified +
    role-admissible + cost-declared + source kinds satisfiable in principle).
    This replaces the old context-free query, which failed closed on sources and
    produced false ``CERTIFIED_FACTOR_NOT_IN_MINING_MANIFEST`` reports.
    """
    try:
        from mining.operator_catalog import get_mining_operators
    except Exception as exc:
        raise RuntimeError(
            f"detect_manifest_gap_intrinsic: mining catalog unavailable: {exc}"
        ) from exc
    manifest_eligible = {op.canonical for op in get_mining_operators(
        admission="eligible", available_sources=_known_source_universe())}
    return [canonical for canonical, _entry in _certified_factor_canonicals(catalog)
            if canonical not in manifest_eligible]


def detect_manifest_gap_contextual(catalog: dict[str, Any]) -> dict[str, list[str]]:
    """R19-015: ``CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST[market,source_context]``.

    A certified factor is also checked in ITS OWN declared context — the markets
    it supports (``market_support``) with exactly the sources it requires.  A
    factor that is intrinsically eligible yet absent from the eligible manifest
    for its own environment is a contextual gap.  The result maps a context key
    (``"market=<m>,sources=<s1,s2,...>"``) to the canonical names missing there.
    The runtime environment is never approximated by a context-free query.
    """
    try:
        from mining.operator_catalog import get_mining_operators, market_support
    except Exception as exc:
        raise RuntimeError(
            f"detect_manifest_gap_contextual: mining catalog unavailable: {exc}"
        ) from exc
    gaps: dict[str, list[str]] = {}
    # Batch certified factors by their exact (market, required-sources) context
    # so the (potentially expensive) eligible-manifest query runs ONCE per
    # distinct context instead of once per canonical.
    contexts: dict[tuple[str | None, tuple[str, ...]], list[str]] = {}
    for canonical, entry in _certified_factor_canonicals(catalog):
        required: list[str] = []
        for s in (entry.get("source_requirements") or ()):
            sid = getattr(s, "source_id", None)
            if sid:
                required.append(str(sid))
        req_sources = tuple(sorted(required))
        try:
            markets = tuple(market_support(canonical, entry))
        except Exception:
            markets = ()
        for market in markets if markets else (None,):
            contexts.setdefault((market, req_sources), []).append(canonical)
    for (market, req_sources), canonicals in contexts.items():
        manifest_eligible = {op.canonical for op in get_mining_operators(
            admission="eligible", market=market, available_sources=list(req_sources))}
        missing = sorted(c for c in canonicals if c not in manifest_eligible)
        if missing:
            sources_key = ",".join(req_sources) or "none"
            gaps[f"market={market},sources={sources_key}"] = missing
    return gaps


_ALIAS_LOOKUP: dict[str, str] = {}


def audit_all_registered_operators() -> dict[str, Any]:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    from cleaned_operators.operator_surface import classify_canonical, unclassified_canonicals
    from research_tools.registry import ResearchToolRegistry

    catalog = OperatorRegistry._catalog
    research_tools = set(ResearchToolRegistry.list_canonical())
    global _ALIAS_LOOKUP
    _ALIAS_LOOKUP = {
        a: t for a, t in _alias_edges(catalog)
    }

    verdicts: list[Verdict] = []
    for canonical in sorted(catalog):
        if canonical in research_tools:
            continue
        cat = catalog[canonical]
        role, role_source = assign_mining_role_ex(canonical, cat)
        case, bucket = _case_and_bucket(canonical, role, cat)
        eligible = mining_eligible(canonical, catalog=cat, role=role)
        from audit.operator_admission_matrix import _recommended_action, _record

        rec = _record(canonical, cat)
        verdicts.append(
            Verdict(
                canonical=canonical,
                case=case,
                bucket=bucket,
                role=role.value,
                role_source=role_source.value,
                mining_eligible=eligible,
                blockers=rec.blocker_codes,
                recommended_action=rec.recommended_action,
            )
        )

    buckets: dict[str, list[str]] = defaultdict(list)
    for v in verdicts:
        buckets[v.bucket].append(v.canonical)
    for key in buckets:
        buckets[key].sort()

    # ---- invariants (§四十二 + R15) ----
    forbidden_public = sorted(
        c for c in catalog
        if (cat := catalog[c]).get("surface") == "unsafe"
        or c in {
            "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate",
            "interpolate", "dropna", "shuffle", "sample",
        } or c.startswith("rand_")
    )
    factor_shaped_research = sorted(
        c for c in catalog if classify_canonical(c) == "research"
        and not (catalog[c].get("diagnostic_only") or catalog[c].get("benchmark_only"))
    )
    eligible_without_cert = sorted(
        v.canonical for v in verdicts if v.mining_eligible
        and not catalog[v.canonical].get("production_certified")
    )
    unused = sorted(
        c for c in catalog
        if c not in research_tools
        and assign_mining_role(c, catalog[c]) in (MiningRole.RESEARCH, MiningRole.LEGACY)
        and not (catalog[c].get("compatibility_only") or catalog[c].get("diagnostic_only"))
    )
    unresolved_roles = sorted(
        v.canonical for v in verdicts if v.role == "unresolved"
    )

    # real detectors (never hardcoded [])
    # R19-015: the release gate uses the intrinsic + per-context manifest gap;
    # the context-free ``detect_manifest_gap`` shim is NOT the runtime env.
    manifest_gap_intrinsic = detect_manifest_gap_intrinsic(catalog)
    manifest_gap_contextual = detect_manifest_gap_contextual(catalog)
    duplicate_candidates = detect_semantic_duplicate_candidates(catalog)
    duplicate_buckets = _bucket_duplicate_pairs(duplicate_candidates)
    # R19-013: parameter-level dead-param coverage.
    dead_detail = detect_dead_searchable_params_detail(catalog)
    dead_audit_errors = dead_detail["audit_errors"]
    dead_coverage = (
        f"audited={dead_detail['audited_operators']}/{dead_detail['total_operators']}"
        if dead_detail["total_operators"] else "no searchable-param factor operators"
    )

    # R15-INC-047: TWO independent unclassified invariants — the authoring
    # surface classification and the mining-role resolution.
    authoring_unclassified = sorted(unclassified_canonicals(catalog))

    invariants = {
        "UNCLASSIFIED_FACTOR_CANONICALS": authoring_unclassified,
        "MINING_ROLE_UNRESOLVED": unresolved_roles,
        "UNUSED_PUBLIC_CANONICALS": unused,
        "FACTOR_SHAPED_RESEARCH_ONLY": factor_shaped_research,
        "MINING_ELIGIBLE_WITHOUT_CERTIFICATION": eligible_without_cert,
        "PERMANENTLY_FORBIDDEN_IN_PUBLIC_REGISTRY": forbidden_public,
        "CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST": manifest_gap_intrinsic,
        "CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST": manifest_gap_contextual,
        "COMPAT_ALIAS_AS_CANONICAL": sorted(
            c for c in catalog if catalog[c].get("compatibility_only")
        ),
        # R19-010: four disjoint duplicate invariants — only the confirmed
        # exact-duplicate bucket blocks release.
        "SEMANTIC_DUPLICATE_CANDIDATES": duplicate_buckets["SEMANTIC_DUPLICATE_CANDIDATES"],
        "CONFIRMED_EXACT_DUPLICATES": duplicate_buckets["CONFIRMED_EXACT_DUPLICATES"],
        "MONOTONIC_EQUIVALENT_PAIRS": duplicate_buckets["MONOTONIC_EQUIVALENT_PAIRS"],
        "RELATED_FAMILY_PAIRS": duplicate_buckets["RELATED_FAMILY_PAIRS"],
        "DEAD_SEARCHABLE_PARAMS": [
            f"{d['canonical']}.{d['param']}"
            for d in dead_detail["dead"]
        ],
    }
    # R16-008 / R19-009: release-critical invariants are EXPLICITLY marked
    # ``release_blocking`` (single source of truth, module-level).  R19-010: only
    # CONFIRMED_EXACT_DUPLICATES gates; name-family / monotonic / advisory
    # candidates never do.  R19-015: the intrinsic manifest invariant gates; the
    # per-context one is informational for the operator's own environment.
    release_blocking = _RELEASE_BLOCKING_INVARIANTS
    # R15-INC-264: detectors that did not run are reported as UNKNOWN coverage,
    # never silently PASS.  R16-004: dead-param coverage is FULL (no sampling);
    # any AUDIT_ERROR on a dead-param probe is itself a release blocker.
    # R19-013: parameter-level dead-param coverage is reported and gated
    # (``probe_error_params == 0`` and ``untested_params == 0`` required).
    detector_coverage = {
        "CERTIFIED_FACTOR_NOT_INTRINSIC_MANIFEST": {
            "ran": True,
            "coverage": "all certified factor operators (intrinsic eligible manifest, full known-source universe)",
        },
        "CERTIFIED_FACTOR_NOT_CONTEXTUAL_MANIFEST": {
            "ran": True,
            "coverage": "certified factor operators checked in their own market/source context",
        },
        "SEMANTIC_DUPLICATE_DETECTOR": {
            "ran": True,
            "coverage": (
                f"{len(duplicate_candidates)} candidate pairs "
                f"(alias={len(duplicate_buckets['CONFIRMED_EXACT_DUPLICATES'])}, "
                f"name-family={len(duplicate_buckets['RELATED_FAMILY_PAIRS'])}, "
                f"monotonic={len(duplicate_buckets['MONOTONIC_EQUIVALENT_PAIRS'])}, "
                f"candidate={len(duplicate_buckets['SEMANTIC_DUPLICATE_CANDIDATES'])})"
            ),
        },
        "DEAD_SEARCHABLE_PARAMS": {
            "ran": dead_detail["total_operators"] > 0,
            "coverage": dead_coverage,
            "audit_errors": dead_audit_errors,
            "unrun": (
                dead_detail["total_operators"] - dead_detail["audited_operators"]
                if dead_detail["total_operators"] else 0
            ),
            "total_searchable_params": dead_detail["total_searchable_params"],
            "successfully_probed_params": dead_detail["successfully_probed_params"],
            "dead_params": dead_detail["dead_params"],
            "probe_error_params": dead_detail["probe_error_params"],
            "untested_params": dead_detail["untested_params"],
            "not_probeable": dead_detail["not_probeable"],
        },
    }

    # R15-INC-046/248: per-context usable (source/market) vs static certified.
    certified_factor_count = sum(
        1 for c, e in catalog.items()
        if e.get("production_certified")
        and assign_mining_role(c, e) in (
            MiningRole.ALPHA, MiningRole.ALPHA_HIGH_COST,
            MiningRole.INTRADAY_EOD, MiningRole.FUNDAMENTAL_PIT,
        )
    )

    counts = {
        "registered public total": len(verdicts),
        "static certified usable total": certified_factor_count,
        "mining usable total": sum(1 for v in verdicts if v.mining_eligible),
        "internal total": len(buckets["B"]),
        "source transform total": len(buckets["I"]),
        "deleted total": len(buckets["A"]) + len(buckets["C"]) + len(buckets["J"]),
        "unclassified total": len(authoring_unclassified),
        "unresolved role total": len(unresolved_roles),
        "unused total": len(invariants["UNUSED_PUBLIC_CANONICALS"]),
    }
    for label, keys in _BUCKET_LABELS.items():
        counts[f"bucket {label} ({keys})"] = len(buckets[label])

    return {
        "verdicts": [v.__dict__ for v in verdicts],
        "buckets": {k: list(v) for k, v in buckets.items()},
        "invariants": invariants,
        "release_blocking": sorted(release_blocking),
        "detector_coverage": detector_coverage,
        "counts": counts,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FactorEngine promote_or_delete audit",
        "",
        f"- registered public total = {result['counts']['registered public total']}",
        f"- static certified usable total = {result['counts']['static certified usable total']}",
        f"- mining usable total = {result['counts']['mining usable total']}",
        f"- internal total = {result['counts']['internal total']}",
        f"- source transform total = {result['counts']['source transform total']}",
        f"- deleted total = {result['counts']['deleted total']}",
        f"- unclassified total = {result['counts']['unclassified total']}",
        f"- unresolved role total = {result['counts']['unresolved role total']}",
        f"- unused total = {result['counts']['unused total']}",
        "",
        "## Buckets (A–K)",
        "",
    ]
    for label, keys in _BUCKET_LABELS.items():
        members = result["buckets"].get(label, [])
        lines.append(f"### {label} — {keys} ({len(members)})")
        for c in members:
            v = next((v for v in result["verdicts"] if v["canonical"] == c), None)
            if v:
                lines.append(
                    f"- `{c}` [case {v['case']}] blockers={','.join(v['blockers']) or '-'} → {v['recommended_action']}"
                )
        lines.append("")
    lines += ["## Invariants", ""]
    for name, members in result["invariants"].items():
        if isinstance(members, dict):
            # R19-015: per-context manifest gaps render as ``context: canonicals``.
            rendered = "; ".join(
                f"{ctx}:{','.join(canons[:8])}"
                for ctx, canons in list(members.items())[:12]
            )
            lines.append(
                f"- {name}: {len(members)} contexts"
                + (f" ({rendered})" if rendered else " ✓")
            )
        else:
            lines.append(f"- {name}: {len(members)}" + (f" ({', '.join(members[:12])})" if members else " ✓"))
    lines += ["", "## Detector coverage (R15: a detector that did not run is UNKNOWN, not PASS)", ""]
    for name, cov in result["detector_coverage"].items():
        lines.append(f"- {name}: ran={cov['ran']} coverage={cov['coverage']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/promote_or_delete_report.md")
    parser.add_argument("--json", default="build/promote_or_delete_report.json")
    # R15-INC-042/258: default is strict — any non-empty hard release invariant
    # returns 1 so CI can never be faked green.
    parser.add_argument("--no-strict", action="store_true", help="exit 0 even on violations")
    args = parser.parse_args()
    result = audit_all_registered_operators()
    out = Path(args.out)
    write_report(result, out)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report: {out}")
    print(f"json:   {args.json}")
    for key, value in result["counts"].items():
        print(f"  {key} = {value}")

    # R16-008 (R19-009): release-blocking invariants are marked in the result,
    # and strict mode fails on ANY of them — there is no second ``_HARD_INVARIANTS``
    # list to keep in sync.
    release_blocking = set(result.get("release_blocking") or ())
    # R16-007: a mineable/production canonical whose dead-param probe raised an
    # AUDIT_ERROR is itself a release blocker (it was never audited).
    dead_cov = result.get("detector_coverage", {}).get("DEAD_SEARCHABLE_PARAMS", {})
    dead_audit_errors = dead_cov.get("audit_errors") or []
    if not args.no_strict:
        violations = [
            name for name in sorted(release_blocking)
            if result["invariants"].get(name)
        ]
        if violations:
            print(f"\nSTRICT FAIL: release-blocking invariants non-empty: {violations}")
            return 1
        if dead_audit_errors:
            print(
                f"\nSTRICT FAIL: {len(dead_audit_errors)} dead-param probe AUDIT_ERRORs "
                f"(first: {dead_audit_errors[0]})"
            )
            return 1
        # R19-013: parameter-level coverage gates — every searchable param must
        # be either probed or honestly declared probeable; an error or an
        # untested searchable param blocks release.
        if dead_cov.get("probe_error_params"):
            print(
                f"\nSTRICT FAIL: {dead_cov['probe_error_params']} searchable params "
                f"raised a probe error (first: {dead_cov.get('audit_errors', [{}])[0] if dead_cov.get('audit_errors') else 'n/a'})"
            )
            return 1
        if dead_cov.get("untested_params"):
            print(
                f"\nSTRICT FAIL: {dead_cov['untested_params']} searchable params untested "
                f"(first: {dead_cov.get('not_probeable', [{}])[0] if dead_cov.get('not_probeable') else 'n/a'})"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
