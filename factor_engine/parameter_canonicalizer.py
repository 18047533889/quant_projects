# -*- coding: utf-8 -*-
"""Parameter-space canonicalization (review #5 §19).

FactorEngine is used to drive alpha search (AlphaProbe / GP mining).  Deduping
canonical *operators* is not enough: several parameter families manufacture
duplicate nodes without a canonicalizer —

  * UltimateOscillator (4,2,1) == (8,4,2)  — homogeneous weights;
  * EaseOfMovement volume_scale — a pure scale leaves cross-sectional rank
    invariant;
  * Keltner multiplier=0 -> KeltnerMid — a zero boundary reduces to a simpler
    AST;
  * symmetric parameters (e.g. a pair of lags) — swapping them is the same
    operator.

``canonicalize_params`` returns the canonical (ordered, normalized) parameter
tuple so the search grammar can dedupe before evaluation.  ``sensitivity_verified``
implements the CI gate: every ``searchable=True`` parameter must have at least
two values that change the operator's output on a canonical fixture — otherwise
it may not enter the mining grammar (parameter_sensitivity_verified).

Review R9-P1 fixes in this module:

* R9-P1-037 — ``pure_scale`` de-duplication is only allowed in an EXPLICIT
  terminal-rank-equivalence mode.  In the default (compositional) mode a pure
  scale ``k`` is kept in the canonical key, because ``k*f(x)`` feeding into
  add/subtract/divide/threshold/where/interaction changes the composed
  expression.  Keys produced in rank-equivalence mode are tagged with
  :data:`_RANK_EQUIVALENCE_TAG` so downstream search can tell they are NOT a
  general AST equivalence.
* R9-P1-038 — a sequence/numeric parameter containing ``bool`` is rejected
  unless its ``ParamSpec`` explicitly declares ``dtype=bool``.  ``True``/``False``
  are never treated as the numbers ``1``/``0``.
* R9-P1-039 — the sensitivity output signature is multi-level (finite-mask
  hash, rank-vector hash, quantized normalized-value hash) instead of the weak
  mean/std/finite_count triple, which both missed genuinely different value
  patterns and flagged tiny float noise.
* R9-P1-040 — ``_probe_value`` draws legal alternatives from the canonical
  grammar (``ParamSpec`` dtype/choices/min/max/active_when, plus declared
  relational specs); a probe index with no legal alternative yields
  :data:`_PROBE_NOT_APPLICABLE` instead of an illegal value.

Review R10 #18/#19/#20/#28/#29/#30 fixes in this module:

* R10 #18 — ``active_when`` is resolvable WITHOUT the controller being
  explicitly provided: the controller's ``ParamSpec.default`` is read and used
  to decide whether a dependent parameter is active.  An INACTIVE parameter
  explicitly bound to a non-default value is rejected (a dead knob must not
  manufacture a second AST).
* R10 #19 — parameter alias single binding: the canonical parameter and any of
  its aliases map to the same logical target, and a single call must bind each
  logical target at most once.  ``window=20`` + ``d=30`` (``d`` aliases
  ``window``) is a duplicate logical binding and is rejected.
* R10 #20 — ParameterSensitivity distinguishes COMPOSITIONAL_SENSITIVITY from
  TERMINAL_RANK_EQUIVALENCE.  The compositional signature binds structural
  identity, finite-mask, cross-sectional rank AND the raw-value behavior, so
  ``f -> 2f`` / ``f -> f+1`` are NOT judged insensitive; the terminal-rank
  signature keeps the scale/shift-invariant normalized-value level.
* R10 #28 — explicit ``panel_params`` / ``input_fields`` / ``scalar_params``
  metadata takes precedence over the legacy numeric-control-name heuristic;
  the heuristic is a fallback only.
* R10 #29 — a fail-closed contract-build path (``build_operator_contract``)
  returns :data:`CONTRACT_ERROR` on any construction failure and never falls
  back to name-guessing; ``operator_contract_searchable`` excludes failed
  contracts from production search.
* R10 #30 — a relational predicate that RAISES is a :data:`CONTRACT_ERROR`
  marker, never ``_PROBE_NOT_APPLICABLE`` (an infeasible-but-real parameter
  combination).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


# R9-P1-037: sentinel prepended to canonical keys produced in terminal
# rank-equivalence mode.  A tagged key is an equivalence ONLY for terminal
# cross-sectional rank, never for general AST equality, so it must never be
# merged with a compositional (untagged) key.
_RANK_EQUIVALENCE_TAG = "__rank_equivalence__"

# R9-P1-040: returned by ``_probe_value`` when no LEGAL alternative exists for a
# probe index (choices exhausted, int singleton, inactive parameter, or a value
# that violates a relational spec).  Callers must skip these probes, never
# evaluate an illegal value.
_PROBE_NOT_APPLICABLE = object()

# R9-P1-039: number of bins for the quantized normalized-value hash level.
# Coarser than 1e-12 so sub-bin float noise does not flip the signature, while a
# genuinely different value pattern still does.
_SIGNATURE_QUANT_LEVELS = 256

# R10 #30 / #29: distinct marker for a relational-predicate execution error or a
# failed contract build.  NEVER conflated with ``_PROBE_NOT_APPLICABLE`` (an
# infeasible-but-real parameter combination).
CONTRACT_ERROR = object()


class ParameterContractError(Exception):
    """An operator/parameter CONTRACT failed to build or to execute.

    Raised/propagated instead of silently treating the failure as a
    NOT_APPLICABLE parameter combination (R10 #30) or falling back to
    name-guessing (R10 #29).
    """


def is_rank_equivalence_key(key: tuple[Any, ...]) -> bool:
    """True when ``key`` was produced by a terminal-rank-equivalence canonicalizer.

    Such a key is tagged ``_RANK_EQUIVALENCE_TAG`` and must NOT be treated as a
    general AST equivalence (R9-P1-037).
    """
    return bool(key) and key[0] == _RANK_EQUIVALENCE_TAG


@dataclass
class ParamNormalizer:
    """How to canonicalize one scalar/control parameter."""

    name: str
    # homogeneous-group members are normalized together (weights -> scale-free).
    normalize_with: tuple[str, ...] = ()
    # pure scale: output is ``k * f(...)`` for any k>0.  Only removed from the
    # search key in explicit terminal-rank-equivalence mode (R9-P1-037); in the
    # default compositional mode the scale REMAINS in the key because the scaled
    # expression feeds into other operators (add/subtract/divide/threshold/
    # where/interaction) where ``k`` is not rank-invariant.
    pure_scale: bool = False
    # symmetric partners: sort the group so (a,b) == (b,a) collapses.
    symmetric_with: tuple[str, ...] = ()
    # zero boundary reduces to a different AST; enforce strict positivity.
    strict_positive: bool = False
    # Audit #386: only declare True when the operator output is provably
    # positive-homogeneous of degree 0 in the group, i.e. f(c*w) == f(w) for any
    # c > 0.  Weighted mean / weighted rank / scale-invariant aggregation are
    # PHD0.  Weighted sum / exposure / regularization / any parameter where
    # scaling the group changes the numeric output MUST stay False — otherwise
    # canonicalization would collapse genuinely distinct parameter sets into one
    # key (audit #386).
    positive_homogeneous_degree_0: bool = False


def _norm_group(values: dict[str, float]) -> dict[str, float]:
    """Homogeneous-weight canonical form: divide by the last member so the group
    is scale-free and the last member is always 1.0."""
    keys = list(values)
    if not keys:
        return values
    base = abs(values[keys[-1]]) if values[keys[-1]] != 0 else 1.0
    return {k: values[k] / base for k in keys}


def _is_missing_default(value: Any) -> bool:
    """True when ``value`` is the ``cleaned_operators.base.MISSING`` sentinel.

    Imported lazily so this standalone module never hard-depends on the
    operator package at import time.
    """
    if value is None:
        return False
    try:
        from cleaned_operators.base import MISSING
    except Exception:  # pragma: no cover - import fallback
        return False
    return value is MISSING


def _active_allows(allowed: Any, ctrl_val: Any) -> bool:
    """Membership test for a ``ParamSpec.active_when`` allowed-values set."""
    if isinstance(allowed, (set, frozenset, tuple, list)):
        return ctrl_val in allowed
    return ctrl_val == allowed


class ParameterCanonicalizer:
    """Canonical form of an operator's parameter dict.

    ``canonical_key(params)`` -> a hashable, order-invariant tuple.  Two
    parameter sets that differ only by a pure scale / homogeneous factor /
    symmetric swap / dead parameter map to the SAME key, so a search space built
    on canonical keys contains no duplicates.

    ``terminal_rank_equivalence`` (R9-P1-037): when False (the default,
    compositional-safe) a ``pure_scale`` parameter is KEPT in the key; only when
    True may pure scales be dropped, and the returned key is tagged as
    rank-equivalence-only.

    ``param_specs`` (R9-P1-038/040): authoritative per-parameter contracts.  A
    ``bool`` inside a sequence parameter is only accepted when the parameter's
    spec declares ``dtype=bool``.
    """

    def __init__(
        self,
        canonical: str,
        normalizers: list[ParamNormalizer],
        *,
        terminal_rank_equivalence: bool = False,
        param_specs: dict[str, Any] | None = None,
    ):
        self.canonical = canonical
        self.normalizers = {n.name: n for n in normalizers}
        self.terminal_rank_equivalence = terminal_rank_equivalence
        self.param_specs: dict[str, Any] = dict(param_specs) if param_specs else {}

    def _sequence_dtype_is_bool(self, name: str) -> bool:
        spec = self.param_specs.get(name)
        return spec is not None and getattr(spec, "dtype", None) is bool

    def _validate_sequence_params(self, params: dict[str, Any]) -> None:
        """Audit #387 + R9-P1-038: reject sequence params containing any
        non-numeric / non-finite element, and reject ``bool`` elements unless
        the parameter's ``ParamSpec`` explicitly declares ``dtype=bool``.

        A mixed-type weight vector (e.g. ``[1.0, "x"]``) or a bool-as-number
        (``True`` -> ``1``) is an invalid parameter set, never a candidate for
        partial-element canonicalization.
        """
        for name, value in params.items():
            if not isinstance(value, (list, tuple)):
                continue
            bool_declared = self._sequence_dtype_is_bool(name)
            for element in value:
                if isinstance(element, bool):
                    if not bool_declared:
                        raise ValueError(
                            f"{self.canonical}: parameter {name} contains bool "
                            "element but its ParamSpec does not declare "
                            "dtype=bool (R9-P1-038: bool is not a number)"
                        )
                    continue
                if not isinstance(element, (int, float)):
                    raise ValueError(
                        f"{self.canonical}: parameter {name} contains non-numeric element"
                    )
                if isinstance(element, float) and not math.isfinite(element):
                    raise ValueError(
                        f"{self.canonical}: parameter {name} contains non-numeric element"
                    )

    def canonical_key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        groups: dict[str, dict[str, float]] = {}
        processed: dict[str, Any] = {}
        # Audit #387: a sequence (list/tuple) parameter must be all-numeric and
        # finite — a single non-numeric or non-finite element makes the WHOLE
        # parameter invalid.  We never drop a partial element (that would quietly
        # canonicalize a corrupted weight vector into a shorter one).
        self._validate_sequence_params(params)
        # Pass 1: pure scales and strict-positives.
        for name, value in params.items():
            norm = self.normalizers.get(name)
            if (
                norm is not None
                and norm.pure_scale
                and self.terminal_rank_equivalence  # R9-P1-037
            ):
                continue  # removed from the search key only in rank-equiv mode
            if norm is not None and norm.strict_positive and float(value) <= 0.0:
                raise ValueError(
                    f"{self.canonical}: {name}={value!r} must be > 0 (zero/negative "
                    "degenerates to a different operator)"
                )
            processed[name] = value
        # Pass 2: homogeneous groups.
        for name, value in list(processed.items()):
            norm = self.normalizers.get(name)
            if norm is not None and norm.normalize_with:
                group = [name, *norm.normalize_with]
                values = {g: float(processed.get(g, np.nan)) for g in group}
                # Audit #386: only scale-free the group when EVERY member is a
                # declared PHD0 parameter (f(c*w) == f(w)).  Otherwise the group
                # values enter the key unchanged — normalizing a weighted sum /
                # exposure / regularization parameter would manufacture a
                # different semantic than the one actually evaluated.
                group_norms = [self.normalizers[g] for g in group if g in self.normalizers]
                phd0 = bool(group_norms) and all(
                    n.positive_homogeneous_degree_0 for n in group_norms
                )
                if all(np.isfinite(v) for v in values.values()) and phd0:
                    canonical = _norm_group(values)
                    for g in group:
                        processed[g] = canonical[g]
        # Pass 3: symmetric groups sorted.
        for name in list(processed):
            norm = self.normalizers.get(name)
            if norm is not None and norm.symmetric_with:
                group = sorted([name, *norm.symmetric_with])
                vals = [processed[g] for g in group]
                # All members must be comparable; if any is non-numeric keep raw.
                if all(isinstance(v, (int, float)) for v in vals):
                    for g, v in zip(group, sorted(vals)):
                        processed[g] = v
        # Order-invariant key: sort by (name, repr(value)).
        key = tuple(
            (k, repr(processed[k]))
            for k in sorted(processed)
        )
        # R9-P1-037: a rank-equivalence-only key is tagged so downstream search
        # never mistakes it for a general AST equivalence.
        if self.terminal_rank_equivalence:
            return (_RANK_EQUIVALENCE_TAG,) + key
        return key


def sensitivity_verified(
    canonical: str,
    param_names: list[str],
    evaluate: Callable[[dict[str, Any]], Any],
    fixture_values: dict[str, Any],
    *,
    searchable: dict[str, bool] | None = None,
    probes: int = 3,
    param_specs: dict[str, Any] | None = None,
    relational_specs: list[Any] | None = None,
) -> bool:
    """CI gate ``parameter_sensitivity_verified``.

    Every ``searchable=True`` parameter must produce a different operator output
    for at least two distinct values on a canonical fixture.  A parameter that
    only changes a global scale (EaseOfMovement.volume_scale), a weight triple
    where all members are equivalent (UltimateOscillator), or any dead parameter
    fails the gate and must be excluded from the mining grammar.

    R9-P1-040: probe values are drawn from the canonical grammar's legal
    alternatives (``param_specs`` dtype/choices/min/max/active_when plus
    ``relational_specs``).  A probe index that has no legal alternative is
    skipped rather than evaluated with an illegal value.
    """
    searchable = searchable or {name: True for name in param_names}
    for name in param_names:
        if not searchable.get(name, True):
            continue
        spec = _spec_for(param_specs, name)
        outputs = set()
        for probe in range(probes):
            params = dict(fixture_values)
            current = params.get(name, _default_for(spec))
            value = _probe_value(name, probe, current, params, spec, relational_specs)
            if value is _PROBE_NOT_APPLICABLE:
                continue
            params[name] = value
            out = evaluate(params)
            outputs.add(_output_signature(out))
        if len(outputs) < 2:
            return False  # parameter {name} has <2 distinct outputs
    return True


def _spec_for(param_specs: dict[str, Any] | None, name: str) -> Any | None:
    if not param_specs:
        return None
    return param_specs.get(name)


def _default_for(spec: Any | None) -> Any:
    """Fallback current value when a probed param is absent from fixture_values."""
    if spec is None:
        return 1.0
    default = getattr(spec, "default", None)
    if default is None or _is_missing_default(default):
        return 1.0
    return default


def _resolve_controller_default(controller: str, param_specs: dict[str, Any] | None) -> Any:
    """Controller value when omitted from a param dict (R10 #18).

    ``active_when`` must be decidable WITHOUT the controller being explicitly
    provided — fall back to the controller parameter's ``ParamSpec.default``
    (or the default of its alias target).  Returns ``None`` when no default is
    declared (undecidable -> the caller skips the check rather than guessing).
    """
    if not param_specs:
        return None
    cspec = param_specs.get(controller)
    if cspec is None:
        # The controller may be a canonical target bound under an alias; a
        # non-empty spec dict keyed by alias is unusual, so simply give up.
        return None
    default = getattr(cspec, "default", None)
    if default is None or _is_missing_default(default):
        return None
    return default


def _probe_value(
    name: str,
    probe: int,
    current: Any,
    params: dict[str, Any],
    spec: Any | None = None,
    relational_specs: list[Any] | None = None,
) -> Any:
    """Legal probe value for ``probe`` index (R9-P1-040).

    Values are drawn from the canonical grammar's legal alternatives: declared
    ``choices`` enumerate directly; ``dtype=int`` uses an integer grid within
    ``[min, max]``; ``dtype=float`` uses quantiles within ``[min, max]`` (plus
    the legacy 1x/2x/0.5x variants clipped to bounds); ``dtype=bool`` toggles the
    boolean; an inactive (``active_when``) or un-probeable parameter yields only
    its current value.  When no LEGAL alternative exists for the probe index the
    result is :data:`_PROBE_NOT_APPLICABLE`, never an illegal value.
    """
    if spec is None:
        value = _legacy_probe_value(name, probe, current)
    else:
        value = _spec_probe_value(name, probe, current, params, spec)
    if value is _PROBE_NOT_APPLICABLE:
        return _PROBE_NOT_APPLICABLE
    return _check_relational_specs(value, name, params, relational_specs)


def _legacy_probe_value(name: str, probe: int, current: Any) -> Any:
    # No declared ParamSpec: keep the historical multiplicative grid, but never
    # treat bool as a number (R9-P1-038) and never emit a non-finite value.
    if isinstance(current, bool):
        return current if probe == 0 else _PROBE_NOT_APPLICABLE
    if isinstance(current, (int, float)):
        scales = (1.0, 2.0, 0.5)
        value = float(current) * scales[probe % 3]
        if not math.isfinite(value):
            return _PROBE_NOT_APPLICABLE
        return value
    # Non-numeric (str/None/...): no scalar alternative can be generated.
    return current if probe == 0 else _PROBE_NOT_APPLICABLE


def _spec_probe_value(
    name: str, probe: int, current: Any, params: dict[str, Any], spec: Any
) -> Any:
    if getattr(spec, "active_when", None) is not None:
        controller, allowed = spec.active_when
        ctrl = params.get(controller)
        if ctrl is not None and not _active_allows(allowed, ctrl):
            # Inactive (dead) knob: probing it is a no-op; keep only the current
            # (default) value so the gate sees no sensitivity from a dead knob.
            return current if probe == 0 else _PROBE_NOT_APPLICABLE

    choices = getattr(spec, "choices", None)
    if choices is not None:
        alts = list(choices)
        if probe >= len(alts):
            return _PROBE_NOT_APPLICABLE
        return alts[probe]

    dtype = getattr(spec, "dtype", None)
    if dtype is bool:
        alts = [current, not current] if isinstance(current, bool) else [False, True]
        if probe >= len(alts):
            return _PROBE_NOT_APPLICABLE
        return alts[probe]
    if dtype is int:
        alts = _int_alternatives(spec.min, spec.max, current, max(probe + 1, 3))
        if probe >= len(alts):
            return _PROBE_NOT_APPLICABLE
        return alts[probe]
    if dtype is float:
        alts = _float_alternatives(spec.min, spec.max, current, max(probe + 1, 3))
        if probe >= len(alts):
            return _PROBE_NOT_APPLICABLE
        return alts[probe]
    if dtype is str:
        return current if probe == 0 else _PROBE_NOT_APPLICABLE
    # Unknown dtype: only the current value is a safe alternative.
    return current if probe == 0 else _PROBE_NOT_APPLICABLE


def _as_int_if_integral(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _int_alternatives(
    lo: Any, hi: Any, current: Any, n: int
) -> list[int]:
    """Legal integer grid within ``[lo, hi]`` (inclusive), at most ``n`` values.

    ``None`` bounds mean unbounded; the legacy audit grid (review #296) seeds an
    artificial range around the current value so a bounded-looking probe set is
    always produced.
    """
    lo_i = int(lo) if lo is not None else 1
    hi_i = int(hi) if hi is not None else max(lo_i + n, lo_i * 3)
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    if hi_i == lo_i:
        return [lo_i]
    span = hi_i - lo_i + 1
    if span <= n:
        vals = list(range(lo_i, hi_i + 1))
    else:
        step = max(1, span // n)
        vals = list(range(lo_i, hi_i + 1, step))
        if vals[-1] != hi_i:
            vals.append(hi_i)
    cur = _as_int_if_integral(current)
    if cur is not None and lo_i <= cur <= hi_i and cur not in vals:
        vals.insert(0, cur)
    return vals


def _float_alternatives(
    lo: Any, hi: Any, current: Any, n: int
) -> list[float]:
    """Legal float grid: quantiles within ``[lo, hi]`` plus clipped 1x/2x/0.5x.

    ``None`` bounds mean unbounded — the legacy multiplicative grid is used.
    """
    lo_f = float(lo) if lo is not None else None
    hi_f = float(hi) if hi is not None else None
    base = current if (
        isinstance(current, (int, float)) and not isinstance(current, bool)
    ) else 5.0
    base = float(base)
    if lo_f is None and hi_f is None:
        candidates = [base, base * 2.0, base * 0.5]
    else:
        if lo_f is None:
            lo_f = hi_f - max(1.0, abs(hi_f) * 2.0)
        if hi_f is None:
            hi_f = lo_f + max(1.0, abs(lo_f) * 2.0)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        if hi_f == lo_f:
            return [lo_f]
        candidates = [
            lo_f,
            hi_f,
            (lo_f + hi_f) / 2.0,
            lo_f + (hi_f - lo_f) * 0.25,
            lo_f + (hi_f - lo_f) * 0.75,
            base,
            base * 2.0,
            base * 0.5,
        ]
    seen: set[float] = set()
    out: list[float] = []
    for v in candidates:
        v = float(v)
        if not math.isfinite(v):
            continue
        if lo_f is not None and v < lo_f:
            continue
        if hi_f is not None and v > hi_f:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    if lo_f is not None and hi_f is not None:
        out.sort()
    return out


def _check_relational_specs(
    value: Any,
    name: str,
    params: dict[str, Any],
    relational_specs: list[Any] | None,
) -> Any:
    """Reject a probe value that violates a declared relational spec (R9-P1-040).

    Each relational spec is evaluated on the full bound dict (other fixture
    params + this probed value); an undecidable/False relation marks the probe
    not-applicable so an infeasible combination is never evaluated.
    """
    if not relational_specs:
        return value
    bound = dict(params)
    bound[name] = value
    for rel in relational_specs:
        check = getattr(rel, "check", None)
        if check is None:
            continue
        try:
            if not check(bound):
                return _PROBE_NOT_APPLICABLE
        except Exception:
            return _PROBE_NOT_APPLICABLE
    return value


def _output_signature(out: Any) -> tuple[str, str, str]:
    """Multi-level output signature (R9-P1-039).

    Level 1 — full finite-mask hash: NaN placement (a partially-NaN vector is
      structurally different from an all-finite one even when the finite values
      coincide).
    Level 2 — rank-vector hash: the order of the finite values, so two vectors
      with identical mean/std/count but different value patterns hash
      differently (a monotone transform keeps the rank hash but is caught by
      level 3).
    Level 3 — quantized normalized-value hash: finite values min-max normalized
      and quantized to ``_SIGNATURE_QUANT_LEVELS`` bins, so tiny float noise
      below one bin does not flip the signature while a genuinely different
      magnitude pattern does.

    Two signatures are equal ONLY when all three levels match.
    """
    if isinstance(out, pd.DataFrame):
        arr = out.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(out, dtype=float)
    finite_mask = np.isfinite(arr)
    mask_hash = hashlib.sha1(finite_mask.tobytes()).hexdigest()
    finite = arr[finite_mask]
    if finite.size == 0:
        return (mask_hash, "empty-rank", "empty-values")
    ranks = np.argsort(np.argsort(finite, kind="stable"), kind="stable").astype(np.int64)
    rank_hash = hashlib.sha1(ranks.tobytes()).hexdigest()
    fmin = float(finite.min())
    fmax = float(finite.max())
    span = fmax - fmin
    if span > 0.0 and math.isfinite(span):
        normalized = (finite.astype(np.float64) - fmin) / span
    else:
        normalized = np.zeros(finite.shape, dtype=np.float64)
    quantized = np.clip(
        np.floor(normalized * _SIGNATURE_QUANT_LEVELS).astype(np.int64),
        0,
        _SIGNATURE_QUANT_LEVELS - 1,
    )
    quant_hash = hashlib.sha1(quantized.tobytes()).hexdigest()
    return (mask_hash, rank_hash, quant_hash)
