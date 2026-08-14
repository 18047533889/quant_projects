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

FE-P0-021 (2026-08-14) — production active_when must fail-closed:

* When a ``ParamSpec`` declares ``active_when`` but the controller parameter is
  missing and has no decidable default, production mode
  (``production_mode=True``) raises :class:`ParameterContractError`
  (fail-closed), while research mode (``production_mode=False``, the default)
  skips the check (fail-open, legacy behavior).  This ensures production
  canonicalization never silently accepts parameter combinations whose
  active/inactive state is undecidable.
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
        param_aliases: dict[str, str] | None = None,
        production_mode: bool = False,
    ):
        self.canonical = canonical
        self.normalizers = {n.name: n for n in normalizers}
        self.terminal_rank_equivalence = terminal_rank_equivalence
        self.param_specs: dict[str, Any] = dict(param_specs) if param_specs else {}
        # R10 #19: alias -> canonical-target map (from the operator's declared
        # ``OperatorMetadata.param_aliases``).  The canonical parameter and any
        # of its aliases must map to the SAME logical target, bound at most once
        # per call.
        self.param_aliases: dict[str, str] = dict(param_aliases) if param_aliases else {}
        # FE-P0-021: production canonicalizer must fail-closed when active_when
        # controller is missing or undecidable; research mode preserves legacy
        # fail-open behavior for backward compatibility.
        self.production_mode = production_mode

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

    def _logical_name(self, name: str) -> str:
        """Alias -> canonical-target name (R10 #19)."""
        return self.param_aliases.get(name, name)

    def _validate_no_duplicate_logical_binding(self, params: dict[str, Any]) -> None:
        """R10 #19: a single call must bind each logical parameter at most once.

        The canonical parameter and any of its aliases map to the same logical
        target.  Binding two spellings of one logical parameter (``window=20``
        and ``d=30`` when ``d`` aliases ``window``) is a duplicate logical
        binding and is rejected.
        """
        if not self.param_aliases:
            return
        bound_logical: dict[str, str] = {}
        for name in params:
            target = self._logical_name(name)
            prior = bound_logical.get(target)
            if prior is not None:
                raise ValueError(
                    f"{self.canonical}: duplicate logical binding for parameter "
                    f"{target!r}: both {prior!r} and {name!r} are bound (R10 #19 "
                    "— a single call must bind each logical parameter once)"
                )
            bound_logical[target] = name

    def _validate_active_when(self, params: dict[str, Any]) -> None:
        """R10 #18 + FE-P0-021: enforce ``ParamSpec.active_when`` with controller defaults.

        When the controller is omitted from ``params`` its ``ParamSpec.default``
        is used to decide whether the dependent parameter is active.  An
        INACTIVE parameter that is explicitly bound to a NON-default value is
        rejected — a dead knob must not manufacture a second AST.

        FE-P0-021: In production mode, if the controller is missing and has no
        decidable default, the canonicalizer raises ParameterContractError
        (fail-closed). In research/legacy mode, undecidable controllers are
        skipped (fail-open) to preserve backward compatibility with legacy
        operators that lack complete active_when specifications.
        """
        for name, spec in self.param_specs.items():
            if spec is None:
                continue
            active_when = getattr(spec, "active_when", None)
            if active_when is None:
                continue
            controller, allowed = active_when
            ctrl_val = params.get(controller)
            if ctrl_val is None:
                ctrl_val = _resolve_controller_default(controller, self.param_specs)
            if ctrl_val is None:
                # FE-P0-021: controller undecidable (no default, no explicit value).
                # Production mode: fail-closed with ParameterContractError.
                # Research mode: fail-open (skip check) for legacy compatibility.
                if self.production_mode:
                    raise ParameterContractError(
                        f"{self.canonical}: active_when controller {controller!r} "
                        f"for parameter {name!r} is missing and has no decidable "
                        f"default (FE-P0-021: production mode requires explicit "
                        f"controller values or ParamSpec defaults for all active_when "
                        f"controllers; cannot judge whether {name!r} is active)"
                    )
                continue  # research mode: controller undecidable -> fail-open (cannot judge)
            if _active_allows(allowed, ctrl_val):
                continue  # active
            # INACTIVE: only tolerate an omitted parameter or its canonical
            # default.
            if name not in params:
                continue
            provided = params[name]
            if not _is_missing_default(spec.default) and provided == spec.default:
                continue
            raise ValueError(
                f"{self.canonical}: parameter {name!r} is inactive when "
                f"{controller}={ctrl_val!r} (allowed: {allowed!r}); explicitly "
                f"bound to non-default {provided!r} (R10 #18 — a dead knob must "
                "not create a second AST)"
            )

    def canonical_key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        groups: dict[str, dict[str, float]] = {}
        processed: dict[str, Any] = {}
        # Audit #387: a sequence (list/tuple) parameter must be all-numeric and
        # finite — a single non-numeric or non-finite element makes the WHOLE
        # parameter invalid.  We never drop a partial element (that would quietly
        # canonicalize a corrupted weight vector into a shorter one).
        self._validate_sequence_params(params)
        # R10 #19: reject duplicate logical bindings (canonical + alias, or two
        # aliases of the same canonical) BEFORE any alias remap could hide them.
        self._validate_no_duplicate_logical_binding(params)
        # R10 #18: reject an inactive parameter bound to a non-default value.
        self._validate_active_when(params)
        # R10 #19: map aliases onto their canonical targets so ``d=30`` and
        # ``window=30`` produce the SAME canonical key.
        if self.param_aliases:
            params = {self._logical_name(k): v for k, v in params.items()}
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
    sensitivity_mode: str = "compositional",
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

    R10 #20: ``sensitivity_mode`` distinguishes the two sensitivity policies.

    * ``"compositional"`` (default): the signature binds structural identity,
      finite-mask, cross-sectional rank AND raw-value behavior — a pure scale
      (``f -> 2f``) or additive shift (``f -> f+1``) feeding into
      add/subtract/divide/threshold/where/interaction is NOT judged insensitive.
    * ``"terminal_rank_equivalence"``: only the terminal cross-sectional rank /
      normalized-value pattern matters — a pure scale feeding only into a
      terminal rank is correctly judged insensitive.

    R10 #30: a relational predicate that RAISES while probing is a
    :class:`ParameterContractError` — never silently downgraded to a
    NOT_APPLICABLE combination.
    """
    if sensitivity_mode not in ("compositional", "terminal_rank_equivalence"):
        raise ValueError(
            f"unknown sensitivity_mode {sensitivity_mode!r}; expected "
            "'compositional' or 'terminal_rank_equivalence'"
        )
    searchable = searchable or {name: True for name in param_names}
    for name in param_names:
        if not searchable.get(name, True):
            continue
        spec = _spec_for(param_specs, name)
        outputs = set()
        for probe in range(probes):
            params = dict(fixture_values)
            current = params.get(name, _default_for(spec))
            value = _probe_value(
                name, probe, current, params, spec, relational_specs, param_specs
            )
            if value is _PROBE_NOT_APPLICABLE:
                continue
            if value is CONTRACT_ERROR:
                raise ParameterContractError(
                    f"{canonical}: relational-predicate execution error while "
                    f"probing parameter {name!r} (R10 #30 — a raising relation "
                    "is a CONTRACT_ERROR, not a NOT_APPLICABLE combination)"
                )
            params[name] = value
            out = evaluate(params)
            outputs.add(_output_signature(out, sensitivity_mode=sensitivity_mode))
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
    param_specs: dict[str, Any] | None = None,
) -> Any:
    """Legal probe value for ``probe`` index (R9-P1-040).

    Values are drawn from the canonical grammar's legal alternatives: declared
    ``choices`` enumerate directly; ``dtype=int`` uses an integer grid within
    ``[min, max]``; ``dtype=float`` uses quantiles within ``[min, max]`` (plus
    the legacy 1x/2x/0.5x variants clipped to bounds); ``dtype=bool`` toggles the
    boolean; an inactive (``active_when``) or un-probeable parameter yields only
    its current value.  When no LEGAL alternative exists for the probe index the
    result is :data:`_PROBE_NOT_APPLICABLE`, never an illegal value.

    R10 #18: ``active_when`` is decidable WITHOUT the controller being
    explicitly provided — the controller's ``ParamSpec.default`` is consulted
    (via ``param_specs``).  R10 #30: a relational predicate that RAISES returns
    :data:`CONTRACT_ERROR`, never ``_PROBE_NOT_APPLICABLE``.
    """
    if spec is None:
        value = _legacy_probe_value(name, probe, current)
    else:
        value = _spec_probe_value(name, probe, current, params, spec, param_specs)
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
    name: str,
    probe: int,
    current: Any,
    params: dict[str, Any],
    spec: Any,
    param_specs: dict[str, Any] | None = None,
) -> Any:
    if getattr(spec, "active_when", None) is not None:
        controller, allowed = spec.active_when
        ctrl = params.get(controller)
        if ctrl is None:
            # R10 #18: decidable WITHOUT the controller being explicitly
            # provided — read the controller parameter's ParamSpec.default.
            ctrl = _resolve_controller_default(controller, param_specs)
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
            satisfied = check(bound)
        except Exception:
            # R10 #30: a relational predicate that RAISES is a CONTRACT_ERROR (a
            # broken / unsupported relation), NEVER a "nonexistent parameter
            # combination".  Undecidable relations are already converted to
            # ``False`` inside ``RelationalParamSpec.check``; an exception that
            # escapes is a genuine contract violation.
            return CONTRACT_ERROR
        if not satisfied:
            return _PROBE_NOT_APPLICABLE
    return value


def _output_signature(
    out: Any, *, sensitivity_mode: str = "compositional"
) -> tuple[str, ...]:
    """Multi-level output signature (R9-P1-039, R10 #20).

    ``sensitivity_mode`` selects the sensitivity policy:

    * ``"compositional"`` (default) — the signature binds structural identity
      (shape + index + columns), the finite-mask, the cross-sectional rank AND
      the quantized RAW-value behavior (12 significant digits).  A pure scale
      (``f -> 2f``) or additive shift (``f -> f+1``) changes the raw-value hash,
      so such a parameter is judged SENSITIVE — it feeds into
      add/subtract/divide/threshold/where/interaction where the composed
      expression changes (R10 #20).
    * ``"terminal_rank_equivalence"`` — the signature binds structural identity,
      the finite-mask, the cross-sectional rank and the quantized
      NORMALIZED-value behavior (min-max, scale/shift-invariant).  ``f -> 2f``
      and ``f -> f+1`` collapse to the same signature because only the terminal
      cross-sectional rank / normalized pattern matters.

    Two signatures are equal ONLY when every level matches.
    """
    if sensitivity_mode not in ("compositional", "terminal_rank_equivalence"):
        raise ValueError(f"unknown sensitivity_mode {sensitivity_mode!r}")
    is_frame = isinstance(out, pd.DataFrame)
    arr = (
        out.to_numpy(dtype=float, copy=False)
        if is_frame
        else np.asarray(out, dtype=float)
    )
    # Level 1 — structural identity (shape / index / columns) so two outputs
    # that differ only in their axes never collide on values alone.
    # SHA-1 used only for parameter equivalence cache keys, not cryptographic security
    structural = _frame_structural_hash(out) if is_frame else hashlib.sha1(
        repr(arr.shape).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    # Level 2 — full finite-mask hash: NaN placement (a partially-NaN vector is
    # structurally different from an all-finite one even when the finite values
    # coincide).
    finite_mask = np.isfinite(arr)
    mask_hash = hashlib.sha1(finite_mask.tobytes(), usedforsecurity=False).hexdigest()
    finite = arr[finite_mask]
    if finite.size == 0:
        return (structural, mask_hash, "empty-rank", "empty-values")
    # Level 3 — cross-sectional rank (for a panel: rank each row across
    # instruments; for a flat array: rank the flattened finite values).
    rank_hash = _cross_sectional_rank_hash(out, arr, finite_mask)
    # Level 4 — value behavior: raw (compositional) vs normalized (rank-equiv).
    if sensitivity_mode == "compositional":
        value_hash = _rounded_raw_hash(finite)
    else:
        value_hash = _quantized_normalized_hash(finite)
    return (structural, mask_hash, rank_hash, value_hash)


def _frame_structural_hash(frame: pd.DataFrame) -> str:
    """Hash of a panel's structural identity (shape + index + columns)."""
    # SHA-1 used only for parameter equivalence cache keys, not cryptographic security
    h = hashlib.sha1(usedforsecurity=False)
    h.update(repr(frame.shape).encode("utf-8"))
    try:
        h.update(repr(list(frame.index)).encode("utf-8"))
    except Exception:  # pragma: no cover - exotic index, cosmetic
        pass
    try:
        h.update(repr(list(frame.columns)).encode("utf-8"))
    except Exception:  # pragma: no cover - exotic columns, cosmetic
        pass
    return h.hexdigest()


def _cross_sectional_rank_hash(out: Any, arr: np.ndarray, finite_mask: np.ndarray) -> str:
    """Cross-sectional rank hash of the finite output cells.

    For a panel, ranks each ROW across columns (instruments) — the A-share
    convention.  For a flat array, ranks the flattened finite values.
    """
    if isinstance(out, pd.DataFrame):
        ranks = out.rank(axis=1, method="average").to_numpy(dtype=float)
        finite_ranks = ranks[finite_mask]
    else:
        finite = arr[finite_mask]
        order = np.argsort(finite, kind="stable")
        finite_ranks = np.empty(finite.size, dtype=np.int64)
        finite_ranks[order] = np.arange(finite.size)
    # SHA-1 used only for parameter equivalence cache keys, not cryptographic security
    return hashlib.sha1(np.asarray(finite_ranks).tobytes(), usedforsecurity=False).hexdigest()


def _rounded_raw_hash(finite: np.ndarray) -> str:
    """Hash of the RAW value behavior (12 significant digits).

    Rounding to 12 significant digits absorbs sub-1e-12 relative float noise
    while any real scale/shift difference — ``2*f(x)``, ``f(x)+1`` — changes the
    hash.  This is the compositional-mode value level (R10 #20).
    """
    vals = finite.astype(np.float64)
    rounded = np.asarray([_round_sig(v, 12) for v in vals], dtype=np.float64)
    # SHA-1 used only for parameter equivalence cache keys, not cryptographic security
    return hashlib.sha1(rounded.tobytes(), usedforsecurity=False).hexdigest()


def _quantized_normalized_hash(finite: np.ndarray) -> str:
    """Hash of the min-max NORMALIZED value pattern (scale/shift-invariant)."""
    vals = finite.astype(np.float64)
    fmin = float(vals.min())
    fmax = float(vals.max())
    span = fmax - fmin
    if span > 0.0 and math.isfinite(span):
        normalized = (vals - fmin) / span
    else:
        normalized = np.zeros(vals.shape, dtype=np.float64)
    quantized = np.clip(
        np.floor(normalized * _SIGNATURE_QUANT_LEVELS).astype(np.int64),
        0,
        _SIGNATURE_QUANT_LEVELS - 1,
    )
    # SHA-1 used only for parameter equivalence cache keys, not cryptographic security
    return hashlib.sha1(quantized.tobytes(), usedforsecurity=False).hexdigest()


def _round_sig(value: float, digits: int) -> float:
    """Round ``value`` to ``digits`` significant digits (stable for log10)."""
    if value == 0.0 or not math.isfinite(value):
        return value
    try:
        shift = digits - int(math.floor(math.log10(abs(value)))) - 1
    except (ValueError, OverflowError):
        return value
    factor = 10.0 ** shift
    return math.floor(value * factor + 0.5) / factor


# ---------------------------------------------------------------------------
# R10 #28 — explicit panel metadata beats the legacy name heuristic
# ---------------------------------------------------------------------------

# Legacy name-heuristic fallback: numeric-control-looking parameter names are
# treated as SCALAR knobs ONLY when no explicit panel metadata declares them
# panel inputs.  Explicit ``panel_params`` / ``input_fields`` / ``scalar_params``
# always win over this heuristic (R10 #28).
_LEGACY_NUMERIC_CONTROL_PARAMS: frozenset[str] = frozenset({
    "window", "min_periods", "order", "lag", "periods", "bins",
    "coefficient_index", "max_q", "block", "level", "band", "d", "k", "q",
    "short_periods", "long_periods", "span", "alpha", "lower", "upper",
    "threshold", "min_bin_count", "n_quantiles", "cap", "floor",
    "body_window", "shadow_window", "penetration", "ratio", "pct", "shift",
    "max_iter", "n_iter", "segments", "min_segments", "max_segments",
    "include_current", "left", "right", "up_count", "down_count",
    "n_levels", "n_buckets", "preset", "side",
})


def classify_panel_scalar_params(
    all_params: list[str] | tuple[str, ...] | None,
    *,
    panel_params: tuple[str, ...] | list[str] | None = None,
    input_fields: tuple[str, ...] | list[str] | None = None,
    scalar_params: tuple[str, ...] | list[str] | None = None,
    numeric_control_params: frozenset[str] = _LEGACY_NUMERIC_CONTROL_PARAMS,
) -> tuple[list[str], list[str]]:
    """Split ``all_params`` into ``(scalar, panel)`` (R10 #28).

    Explicit metadata takes precedence over the name heuristic:

      1. ``panel_params`` — authoritative PANEL inputs; a name declared here is
         NEVER reclassified by the numeric-control-name heuristic (even
         ``level`` / ``ratio`` / ``band`` / ``side``).
      2. ``input_fields`` — authoritative PANEL inputs (legacy spelling).
      3. ``scalar_params`` — authoritative SCALAR knobs when declared.
      4. The name heuristic is applied ONLY to names with no explicit
         declaration: a numeric-control-looking name -> scalar; everything else
         -> panel (legacy fallback).

    The legacy fallback never overrides an explicit declaration.
    """
    all_params = list(all_params or [])
    if not all_params:
        return [], []
    declared_panel = set(panel_params or ())
    declared_inputs = set(input_fields or ())
    declared_scalar = set(scalar_params or ())
    panel: set[str] = set()
    scalar: set[str] = set()
    for p in all_params:
        if p in declared_panel or p in declared_inputs:
            panel.add(p)
        elif p in declared_scalar:
            scalar.add(p)
    for p in all_params:
        if p in panel or p in scalar:
            continue
        if p in numeric_control_params:
            scalar.add(p)
        else:
            panel.add(p)
    return (
        [p for p in all_params if p in scalar],
        [p for p in all_params if p in panel],
    )


# ---------------------------------------------------------------------------
# R10 #29 — fail-closed contract build (exclude failed contracts from search)
# ---------------------------------------------------------------------------


def build_operator_contract(operator: Any) -> dict[str, Any]:
    """Build a structured operator contract (fail-closed, R10 #29).

    Returns a contract dict on success, or :data:`CONTRACT_ERROR` when the
    operator's contract cannot be built — the operator must then be excluded
    from production search.  There is NO silent fallback to name-guessing.
    """
    try:
        meta = getattr(operator, "metadata", None)
        if meta is None:
            raise ValueError("operator has no metadata contract")
        canonical = str(getattr(meta, "name", None) or "")
        if not canonical:
            raise ValueError("operator metadata declares no name")
        param_names = tuple(getattr(meta, "param_names", None) or ())
        param_specs = dict(getattr(meta, "param_specs", None) or {})
        param_aliases = dict(getattr(meta, "param_aliases", None) or {})
        panel_params = tuple(getattr(meta, "panel_params", None) or ())
        input_fields = tuple(getattr(meta, "input_fields", None) or ())
        scalar_params = tuple(getattr(meta, "scalar_params", None) or ())
        _validate_contract_invariants(canonical, param_names, param_specs, param_aliases)
        return {
            "canonical": canonical,
            "param_names": param_names,
            "param_specs": param_specs,
            "param_aliases": param_aliases,
            "panel_params": panel_params,
            "input_fields": input_fields,
            "scalar_params": scalar_params,
        }
    except Exception as exc:  # noqa: BLE001 - fail-closed by contract
        return CONTRACT_ERROR


def _validate_contract_invariants(
    canonical: str,
    param_names: tuple[str, ...],
    param_specs: dict[str, Any],
    param_aliases: dict[str, str],
) -> None:
    """Internal-consistency checks for a built contract (R10 #29)."""
    names = set(param_names)
    alias_keys = set(param_aliases)
    for pname in param_specs:
        if pname not in names and pname not in alias_keys:
            raise ValueError(
                f"{canonical}: ParamSpec key {pname!r} is not a declared "
                "param_name or alias (R10 #29 — a dead spec cannot build a "
                "searchable contract)"
            )
    for alias, target in param_aliases.items():
        if target not in names:
            raise ValueError(
                f"{canonical}: alias {alias!r} -> {target!r}, but {target!r} is "
                "not a declared param_name (R10 #29)"
            )


def operator_contract_searchable(contract: Any) -> bool:
    """An operator is production-searchable only when its contract built cleanly.

    ``CONTRACT_ERROR`` / ``None`` (no contract) exclude the operator from
    production search (R10 #29).
    """
    return contract is not CONTRACT_ERROR and contract is not None
