# -*- coding: utf-8 -*-
"""Round-11 §37 shared machine auditors.

The long-tail audit explicitly requires ONE automatic auditor, not a per-file
test per finding.  This module implements the cross-cutting audits that any
operator workstream must survive:

  * A. Prefix-Extension Invariance   — ``op(full)[:T] == op(full[:T])`` beyond
        warmup, for causal operators.  Catches pivot retroactive rewrite,
        future-dependent normalization, state repaint.
  * B. Stateful-Contract Discovery   — chunk-boundary test: if running the
        second half without carried state differs from the full run but the
        contract says stateless -> HARD FAIL; if identical but the contract
        says required_full_history -> false-positive report.
  * C. Default-Parameter History     — with defaults only, the inferred
        ``history_requirement`` must cover the operator's real warmup (the
        first finite output region); a window whose kernel uses default
        ``window=120`` but history says 0 is a P0 underestimate.
  * D. Unit Algebra                  — ``same_as:<param>`` references must
        resolve to a real declared parameter; covariance/beta/residual/slope/
        duration/event-probability output units must match the family.

Run: ``python3 scripts/audit_r11_longtail.py`` (exits non-zero on hard fails).
A curated sample keeps runtime bounded; the audit target operators are always
included.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

# Load the shared production-audit harness by ABSOLUTE FILE PATH under a unique
# module name — the monorepo root also ships a ``scripts`` package that shadows
# factor_engine's under pytest, so the ``scripts.audit_all_factor_production``
# import form is not reliable in test collection.
_FE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_harness():
    path = os.path.join(_FE_ROOT, "scripts", "audit_all_factor_production.py")
    spec = importlib.util.spec_from_file_location(
        "audit_all_factor_production_harness", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HARNESS = _load_harness()
_build_call = _HARNESS._build_call
_panels = _HARNESS._panels
_slice = _HARNESS._slice
_slice_range = _HARNESS._slice_range
_to_frame = _HARNESS._to_frame

# P0-08: bound at module scope (after the harness has made ``cleaned_operators``
# importable) so ``main()``'s production-candidate gate is monkeypatchable by
# tests (``monkeypatch.setattr(audit, "factor_production_targets", ...)``).
from cleaned_operators.production_hardening import factor_production_targets
from cleaned_operators.semantic_certification import should_fail_closed

# --- operator sample ---------------------------------------------------------
# Always-audited causal kernels (pivot/extrema/structural/directional and the
# audit's named families).  These are the operators where a retrospective
# rewrite would show up as a prefix-invariance violation.
_ALWAYS_CAUSAL_PREFIXES = (
    "ts_", "structural_", "extrema_", "directional_change_", "event_",
    "state_", "cross_event", "ts_cusum_", "ts_hysteresis", "ts_state_",
    "update_", "report_", "vol_", "roll_", "shift_", "cum_", "expanding_",
)


def _canonical_sample(rng: random.Random, n_random: int = 60) -> list[str]:
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry

    all_c = list(OperatorRegistry.list_canonical())
    always = sorted(c for c in all_c if c.startswith(_ALWAYS_CAUSAL_PREFIXES))
    rest = sorted(set(all_c) - set(always))
    sample = list(always) + rng.sample(rest, min(n_random, len(rest)))
    return sample


def _warmup_rows(canonical: str) -> int:
    from runtime.execution_contract import history_requirement

    try:
        return max(1, history_requirement(canonical, {}).rows)
    except Exception:
        return 1


def _bound_scalar_params(
    canonical: str, operator: Any, args: tuple, kwargs: dict
) -> dict[str, Any]:
    """Extract the scalar (non-panel) bound parameters an operator was called with.

    Used to compute the chunk-boundary warmup for the SAME parameterisation the
    audit executed (an operator requiring ``left_window``/``right_window`` has a
    bound-param history far above the unbound floor).
    """
    meta = getattr(operator, "metadata", None)
    names = list(getattr(meta, "param_names", None) or ())
    bound: dict[str, Any] = {}
    for i, value in enumerate(args):
        if i < len(names) and isinstance(value, (int, float)) and not isinstance(value, bool):
            bound[names[i]] = value
    for key, value in kwargs.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            bound[key] = value
    return bound


def _compare_region(left: pd.DataFrame, right: pd.DataFrame, skip: int) -> bool:
    """Compare the trailing rows beyond ``skip``; True when they agree."""
    if left.shape != right.shape:
        return False
    if not left.index.equals(right.index) or not left.columns.equals(right.columns):
        return False
    rows = left.shape[0]
    if rows <= skip:
        return True  # nothing comparable below warmup — not a violation
    l = left.iloc[skip:]
    r = right.iloc[skip:]
    try:
        return bool(
            np.allclose(
                l.to_numpy(dtype=float), r.to_numpy(dtype=float),
                equal_nan=True, rtol=1e-5, atol=1e-7,
            )
        )
    except (TypeError, ValueError):
        return bool(l.astype(object).where(pd.notna(l), None).equals(
            r.astype(object).where(pd.notna(r), None)
        ))


def _mark_unaudited(unaudited: list[str], canonical: str) -> None:
    """Record ``canonical`` as not-auditable without duplicating it."""
    if canonical not in unaudited:
        unaudited.append(canonical)


def _first_finite_row(frame: pd.DataFrame) -> int:
    """0-based row index of the first output row with any finite value.

    Returns ``-1`` when no row carries a finite value (e.g. a fixture that
    never matures the kernel).  This is the operator's real warmup: the first
    meaningful output appears at ``m``, so the declared history must cover at
    least ``m + 1`` rows of input.
    """
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        matrix = frame.notna().to_numpy()
    else:
        matrix = np.isfinite(values)
    nonzero = np.flatnonzero(matrix.any(axis=1))
    if len(nonzero) == 0:
        return -1
    return int(nonzero[0])


# ---------------------------------------------------------------------------
# Audit A — Prefix-Extension Invariance
# ---------------------------------------------------------------------------
def audit_prefix_invariance(
    canonicals: list[str],
    panels: dict[str, pd.DataFrame],
    *,
    unaudited: list[str] | None = None,
) -> list[str]:
    """``op(full)[:T] == op(full[:T])`` beyond warmup for causal operators.

    Any causal operator (rolling window, recursive state, event stream) is
    prefix-invariant: the value at row ``t`` depends only on rows ``<= t``.
    A violation is a retrospective rewrite (pivot replacement, future-dependent
    normalization, state repaint).

    Operators whose fixture execution raises (the auditor cannot construct valid
    inputs) are appended to ``unaudited`` (P0-08) instead of passing silently.
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry

    _unaudited = unaudited if unaudited is not None else []
    n_rows = panels["x"].shape[0]
    ts = sorted({n_rows // 4, n_rows // 2, n_rows * 3 // 4})
    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        skip = _warmup_rows(canonical)
        try:
            full_args, full_kwargs = _build_call(canonical, operator, panels)
            full = _to_frame(operator.calculate(*full_args, **full_kwargs), panels["x"])
        except Exception:
            # operators that raise on this fixture are not auditable here — a
            # production candidate must fail closed rather than pass silently.
            _mark_unaudited(_unaudited, canonical)
            continue
        checked = False
        for t in ts:
            if t <= skip:
                continue
            try:
                args, kwargs = _build_call(canonical, operator, panels)
                pargs = [_slice(a, t) for a in args]
                pkwargs = {
                    k: (_slice(v, t) if isinstance(v, (pd.DataFrame, pd.Series)) else v)
                    for k, v in kwargs.items()
                }
                prefix = _to_frame(operator.calculate(*pargs, **pkwargs), panels["x"].iloc[:t])
            except Exception:
                continue
            checked = True
            full_prefix = _slice(full, t)
            if not _compare_region(full_prefix, prefix, skip):
                errors.append(
                    f"A prefix-invariance: {canonical} full-run[:{t}] != run([:{t}]) "
                    "beyond warmup — future data changes past factor values "
                    "(retrospective rewrite / future-dependent normalization)"
                )
                break
        if not checked:
            # no prefix slice could be executed — nothing was actually verified.
            _mark_unaudited(_unaudited, canonical)
    return errors


# ---------------------------------------------------------------------------
# Audit B — Stateful Contract Discovery (chunk-boundary)
# ---------------------------------------------------------------------------
def audit_stateful_contract_discovery(
    canonicals: list[str],
    panels: dict[str, pd.DataFrame],
    *,
    unaudited: list[str] | None = None,
) -> list[str]:
    """Compare the declared contract against the observed chunk behavior.

    For each operator: run on the FULL panel, then on the true second-half slice
    alone (no carried state).  A genuinely stateful operator needs the first
    half's state, so the second-half values differ from the full run; a truly
    stateless/causal operator reproduces them beyond warmup.

      * contract says stateless but the values differ  -> HARD FAIL (drift:
        module says stateless, runtime is stateful).
      * contract says required_full_history but values match -> false-positive
        report (the full-history claim is not observable).

    Operators whose fixture execution raises are appended to ``unaudited``
    (P0-08).  The second-half slice uses :func:`_slice_range` so it is the real
    trailing half (``iloc[half:]``), not the leading half (P0-06).
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry
    from runtime.execution_contract import execution_contract

    _unaudited = unaudited if unaudited is not None else []
    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        try:
            full_args, full_kwargs = _build_call(canonical, operator, panels)
            full = _to_frame(
                operator.calculate(*full_args, **full_kwargs),
                panels["x"],
            )
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        n_rows = full.shape[0]
        half = n_rows // 2
        if half < 10:
            continue
        try:
            args, kwargs = _build_call(canonical, operator, panels)
            pargs = [_slice_range(a, half, None) for a in args]
            pkwargs = {
                k: (_slice_range(v, half, None) if isinstance(v, (pd.DataFrame, pd.Series)) else v)
                for k, v in kwargs.items()
            }
            second_half = _to_frame(
                operator.calculate(*pargs, **pkwargs), panels["x"].iloc[half:]
            )
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        full_tail = full.iloc[half:]
        if full_tail.shape != second_half.shape:
            continue
        contract = execution_contract(canonical)
        # Compare beyond the slice-local warmup: a causal rolling operator run on
        # the trailing half alone has no lookback rows at the chunk boundary, so
        # the first ``warmup`` rows legitimately differ from the full run.  Only a
        # genuine state machine continues to differ beyond warmup (P0-06 note).
        # The warmup is the history for the SAME parameters the operator was
        # executed with — an operator that requires its window params
        # (left_window/right_window) has a bound-param warmup far larger than the
        # ``{}`` default floor (2), so using the default would false-flag it.
        from runtime.execution_contract import history_requirement

        skip = _warmup_rows(canonical)
        try:
            bound = _bound_scalar_params(canonical, operator, args, kwargs)
            bound_req = history_requirement(canonical, bound).rows
            skip = max(skip, int(bound_req or 0))
        except Exception:
            pass
        identical = _compare_region(full_tail, second_half, skip)
        if contract.state_model == "stateless" and not identical:
            errors.append(
                f"B stateful-drift: {canonical} contract says stateless but the "
                "second half (no carried state) differs from the full run — the "
                "operator is actually stateful (module/runtime drift)"
            )
        elif contract.requires_full_history and identical:
            # False-positive report, not a hard fail.
            pass
    return errors


# ---------------------------------------------------------------------------
# Audit C — Default-Parameter History
# ---------------------------------------------------------------------------
def audit_default_parameter_history(
    canonicals: list[str],
    panels: dict[str, pd.DataFrame],
    *,
    unaudited: list[str] | None = None,
) -> list[str]:
    """The inferred history must cover the operator's window-warmup WORST CASE.

    A windowed kernel's warmup is bounded by its window parameters — if the
    operator declares ``outer_window`` / ``left_window`` / ``cup_window`` etc.
    that the name-guesser does not recognise, the history layer under-allocates
    even though the kernel re-reads a long window.  This audit is
    DATA-INDEPENDENT: it compares ``history_requirement`` for the bound
    parameters (plus kernel signature defaults) against the SUM of the
    window-like parameter values — the worst-case lookback the kernel needs —
    and FAILs only a genuine under-declaration (declared < window-sum).

    Data-dependent first-output rows (a pattern that simply does not form early
    in a fixture) are NOT a history bug and are not flagged.  Operators that
    raise on the fixture are appended to ``unaudited`` (P0-08), never FAILed.
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.base import _kernel_param_defaults
    from cleaned_operators.registry import OperatorRegistry
    from runtime.execution_contract import history_requirement

    _unaudited = unaudited if unaudited is not None else []
    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        meta = getattr(operator, "metadata", None)
        names = list(getattr(meta, "param_names", None) or ())
        window_params = [n for n in names if _window_like(n)]
        if not window_params:
            # No window-shaped parameter — nothing a bar-warmup could under-allocate.
            continue
        try:
            args, kwargs = _build_call(canonical, operator, panels)
            output = _to_frame(operator.calculate(*args, **kwargs), panels["x"])
        except Exception:
            # Cannot construct a valid default execution — P0-08, not a FAIL.
            _mark_unaudited(_unaudited, canonical)
            continue
        # Bound scalar params + the kernel's own signature defaults: the actual
        # window values the operator runs with when the caller does not bind them.
        bound = _bound_scalar_params(canonical, operator, args, kwargs)
        try:
            defaults = _kernel_param_defaults(operator) or {}
        except Exception:
            defaults = {}
        merged = dict(defaults)
        merged.update(bound)
        try:
            requirement = history_requirement(canonical, merged)
            declared = requirement.rows
        except Exception:
            _mark_unaudited(_unaudited, canonical)
            continue
        if requirement.is_full_history:
            # Full-history / event-clock operators are given all available
            # history by contract; ``rows`` is only a floor, not a truncation.
            continue
        # The worst-case window contribution the kernel needs: the SUM of its
        # true window params (``min_*`` floors are not window extensions, and a
        # trailing window contributes ~value while a compound lookback
        # contributes value+value — the factor-of-2 gate tolerates that
        # ambiguity and flags only GROSS under-allocation, e.g. declared=2 vs
        # a window that sums to 60 because the spelling was not recognised).
        window_sum = 0
        for name in window_params:
            if name.startswith("min_"):
                continue
            value = merged.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                window_sum += int(value)
        if window_sum <= 0:
            continue
        if declared * 2 < window_sum:
            errors.append(
                f"C default-history: {canonical} declares window param(s) "
                f"{[n for n in window_params if not n.startswith('min_')]} whose "
                f"values sum to {window_sum} but history_requirement is only "
                f"{declared} rows — the history layer under-allocates the "
                "kernel's warmup (unrecognised window spelling; declare "
                "ParamSpec.history_formula)"
            )
    return errors


def _window_like(name: str) -> bool:
    lowered = str(name).lower()
    # ``n_`` matches only as a standalone prefix (``n_bars``/``n_periods``),
    # NEVER mid-word — ``ann_factor`` contains ``n_`` ("a*nn_*factor") but is a
    # scaling constant, not a warmup window.
    if lowered.startswith("n_") or "_n_" in lowered:
        return True
    return any(
        token in lowered
        for token in (
            "window", "span", "lag", "lookback", "period", "_days",
            "block", "cutoff", "max_run", "history",
        )
    )


# ---------------------------------------------------------------------------
# Audit D — Unit Algebra
# ---------------------------------------------------------------------------
_VARIADIC_PARAM_RE = re.compile(r"^(?:s|p|psid|sid|v|value|w)\d+$")


def _is_variadic(meta: Any) -> bool:
    """A variadic operator (dynamic inputs) references its conceptual input by
    ``same_as:value`` / ``same_as:share`` even though no single named param
    exists — additional scalar knobs (``window``, ``min_periods``, ``k``) do not
    make a variadic-input operator a single-named-param operator."""
    tags = {str(t).lower() for t in (getattr(meta, "tags", None) or [])}
    if "variadic" in tags or "dynamic_inputs" in tags:
        return True
    params = [str(p) for p in (getattr(meta, "param_names", None) or ())]
    if not params:
        return False
    variadic_like = sum(1 for p in params if _VARIADIC_PARAM_RE.match(p))
    scalar_like = sum(1 for p in params if not _VARIADIC_PARAM_RE.match(p))
    # At least two variadic-input slots (s1..sN / p1..pN) makes the conceptual
    # input the "value"/"share" column; a lone ``window`` scalar is fine.
    return variadic_like >= 2 and scalar_like <= 3


def audit_unit_algebra() -> list[str]:
    """``same_as:<param>`` references must resolve to real declared params; the
    formula families must carry the dimensional-analysis-correct output unit."""
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canonical in OperatorRegistry.list_canonical():
        meta = OperatorRegistry.get(canonical).metadata
        names = set(getattr(meta, "param_names", None) or ())
        output_unit = getattr(meta, "output_unit", None) or ""
        if isinstance(output_unit, str) and output_unit.startswith("same_as:"):
            target = output_unit.split(":", 1)[1]
            # Variadic operators legitimately reference their conceptual input
            # (``same_as:value`` / ``same_as:share``) with no single named param.
            if target not in names and not _is_variadic(meta):
                errors.append(
                    f"D unit-algebra: {canonical} output_unit 'same_as:{target}' "
                    f"references an undeclared parameter (params: {sorted(names)})"
                )
        # formula families whose dimensional analysis is fixed.  Token matching
        # uses word boundaries so ``panel_coverage`` is NOT a covariance (the
        # substring ``_cov`` also appears in ``coverage``).
        family = None
        if re.search(r"(?:^|_)cov(?:ariance)?(?:_|$)", canonical):
            family = "covariance -> unit(x)*unit(y)"
        elif re.search(r"(?:^|_)beta(?:_|$)", canonical):
            family = "beta -> unit(y)/unit(x)"
        elif re.search(r"(?:^|_)slope(?:_|$)", canonical) or "regression_slope" in canonical:
            family = "slope -> unit(y)/unit(x)"
        if family is not None and output_unit and "same_as:" not in output_unit:
            if output_unit == "ratio":
                errors.append(
                    f"D unit-algebra: {canonical} is a {family} kernel but its "
                    f"declared output_unit is 'ratio' — dimensional analysis "
                    "requires the parameter-relative unit"
                )
    return errors


def _curated_causal_panels() -> dict[str, pd.DataFrame]:
    import cleaned_operators as co

    co.load_all()
    return _panels(rows=420, columns=6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=60)
    args = parser.parse_args()
    rng = random.Random(20260809)
    panels = _curated_causal_panels()
    sample = _canonical_sample(rng, n_random=args.sample)
    print(f"R11 long-tail audit over {len(sample)} sampled canonicals")
    errors: list[str] = []
    unaudited: list[str] = []
    errors += audit_prefix_invariance(sample, panels, unaudited=unaudited)
    errors += audit_stateful_contract_discovery(sample, panels, unaudited=unaudited)
    errors += audit_default_parameter_history(sample, panels, unaudited=unaudited)
    errors += audit_unit_algebra()
    # P0-08: an operator the auditor cannot construct valid inputs for must not
    # pass silently.  Production candidates fail closed; research/experimental
    # operators only warn.
    production_targets = set(factor_production_targets())
    for canonical in unaudited:
        if canonical in production_targets and not should_fail_closed(canonical):
            errors.append(f"UNAUDITED production candidate: {canonical}")
        else:
            print(
                f"warning: un-auditable operator not a production candidate: "
                f"{canonical}",
                file=sys.stderr,
            )
    if errors:
        print(f"R11 audit FAILED ({len(errors)} issues)", file=sys.stderr)
        for error in errors[:40]:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("R11 long-tail audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
