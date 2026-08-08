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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class ParamNormalizer:
    """How to canonicalize one scalar/control parameter."""

    name: str
    # homogeneous-group members are normalized together (weights -> scale-free).
    normalize_with: tuple[str, ...] = ()
    # pure scale: output is ``k * f(...)`` for any k>0 — remove from the search
    # key (cross-sectional rank is invariant).
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


class ParameterCanonicalizer:
    """Canonical form of an operator's parameter dict.

    ``canonical_key(params)`` -> a hashable, order-invariant tuple.  Two
    parameter sets that differ only by a pure scale / homogeneous factor /
    symmetric swap / dead parameter map to the SAME key, so a search space built
    on canonical keys contains no duplicates.
    """

    def __init__(self, canonical: str, normalizers: list[ParamNormalizer]):
        self.canonical = canonical
        self.normalizers = {n.name: n for n in normalizers}

    def _validate_sequence_params(self, params: dict[str, Any]) -> None:
        """Audit #387: reject sequence params containing any non-numeric or
        non-finite element.  A mixed-type weight vector (e.g. ``[1.0, "x"]``) is
        an invalid parameter set, never a candidate for partial-element
        canonicalization."""
        for name, value in params.items():
            if not isinstance(value, (list, tuple)):
                continue
            for element in value:
                if not isinstance(element, (int, float, bool)):
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
            if norm is not None and norm.pure_scale:
                continue  # removed from the search key entirely
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
        return tuple(
            (k, repr(processed[k]))
            for k in sorted(processed)
        )


def sensitivity_verified(
    canonical: str,
    param_names: list[str],
    evaluate: Callable[[dict[str, Any]], Any],
    fixture_values: dict[str, Any],
    *,
    searchable: dict[str, bool] | None = None,
    probes: int = 3,
) -> bool:
    """CI gate ``parameter_sensitivity_verified``.

    Every ``searchable=True`` parameter must produce a different operator output
    for at least two distinct values on a canonical fixture.  A parameter that
    only changes a global scale (EaseOfMovement.volume_scale), a weight triple
    where all members are equivalent (UltimateOscillator), or any dead parameter
    fails the gate and must be excluded from the mining grammar.
    """
    searchable = searchable or {name: True for name in param_names}
    for name in param_names:
        if not searchable.get(name, True):
            continue
        outputs = set()
        for probe in range(probes):
            params = dict(fixture_values)
            params[name] = _probe_value(name, probe, params.get(name, 1.0))
            out = evaluate(params)
            outputs.add(_output_signature(out))
        if len(outputs) < 2:
            return False  # parameter {name} has <2 distinct outputs
    return True


def _probe_value(name: str, probe: int, current: Any) -> Any:
    # probe values are deliberately separated in log-space so a scale-only or
    # near-degenerate parameter is exposed (1x, 2x, 0.5x of the default).
    if isinstance(current, (int, float)) and not isinstance(current, bool):
        scales = (1.0, 2.0, 0.5)
        return float(current) * scales[probe % 3]
    return current


def _output_signature(out: Any) -> str:
    if isinstance(out, pd.DataFrame):
        arr = out.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(out, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "all-nan"
    return f"{float(finite.mean()):.12f}|{float(finite.std()):.12f}|{finite.size}"
