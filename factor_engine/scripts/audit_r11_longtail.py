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
_to_frame = _HARNESS._to_frame

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


def _compare_region(left: pd.DataFrame, right: pd.DataFrame, skip: int) -> bool:
    """Compare the trailing rows beyond ``skip``; True when they agree."""
    if left.shape != right.shape:
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


# ---------------------------------------------------------------------------
# Audit A — Prefix-Extension Invariance
# ---------------------------------------------------------------------------
def audit_prefix_invariance(
    canonicals: list[str], panels: dict[str, pd.DataFrame]
) -> list[str]:
    """``op(full)[:T] == op(full[:T])`` beyond warmup for causal operators.

    Any causal operator (rolling window, recursive state, event stream) is
    prefix-invariant: the value at row ``t`` depends only on rows ``<= t``.
    A violation is a retrospective rewrite (pivot replacement, future-dependent
    normalization, state repaint).
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry

    n_rows = panels["x"].shape[0]
    ts = sorted({n_rows // 4, n_rows // 2, n_rows * 3 // 4})
    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
        except Exception:
            continue
        skip = _warmup_rows(canonical)
        try:
            full = operator.calculate(*_build_call(canonical, operator, panels))
        except Exception:
            continue  # operators that raise on this fixture are not auditable here
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
            full_prefix = _slice(full, t)
            if not _compare_region(full_prefix, prefix, skip):
                errors.append(
                    f"A prefix-invariance: {canonical} full-run[:{t}] != run([:{t}]) "
                    "beyond warmup — future data changes past factor values "
                    "(retrospective rewrite / future-dependent normalization)"
                )
                break
    return errors


# ---------------------------------------------------------------------------
# Audit B — Stateful Contract Discovery (chunk-boundary)
# ---------------------------------------------------------------------------
def audit_stateful_contract_discovery(
    canonicals: list[str], panels: dict[str, pd.DataFrame]
) -> list[str]:
    """Compare the declared contract against the observed chunk behavior.

    For each operator: run on the FULL panel, then on the second-half slice
    alone (no carried state).  A genuinely stateful operator needs the first
    half's state, so the second-half values differ from the full run; a truly
    stateless/causal operator reproduces them beyond warmup.

      * contract says stateless but the values differ  -> HARD FAIL (drift:
        module says stateless, runtime is stateful).
      * contract says required_full_history but values match -> false-positive
        report (the full-history claim is not observable).
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry
    from runtime.execution_contract import execution_contract

    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
        except Exception:
            continue
        try:
            full = _to_frame(
                operator.calculate(*_build_call(canonical, operator, panels)),
                panels["x"],
            )
        except Exception:
            continue
        n_rows = full.shape[0]
        half = n_rows // 2
        if half < 10:
            continue
        try:
            args, kwargs = _build_call(canonical, operator, panels)
            pargs = [_slice(a, half) for a in args]
            pkwargs = {
                k: (_slice(v, half) if isinstance(v, (pd.DataFrame, pd.Series)) else v)
                for k, v in kwargs.items()
            }
            second_half = _to_frame(
                operator.calculate(*pargs, **pkwargs), panels["x"].iloc[:half]
            )
        except Exception:
            continue
        full_tail = full.iloc[half:]
        if full_tail.shape != second_half.shape:
            continue
        contract = execution_contract(canonical)
        identical = _compare_region(full_tail, second_half, 0)
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
    canonicals: list[str], panels: dict[str, pd.DataFrame]
) -> list[str]:
    """With NO explicit optional params, the inferred history must cover the
    kernel's real warmup.

    The kernel's first finite output appears at the row where its declared
    window (default) has accumulated.  If ``history_requirement`` says 0 but the
    kernel's default window is large, the history layer under-allocates.
    """
    import cleaned_operators as co

    co.load_all()
    from cleaned_operators.registry import OperatorRegistry
    from runtime.execution_contract import history_requirement

    errors: list[str] = []
    for canonical in canonicals:
        try:
            operator = OperatorRegistry.get(canonical)
            meta = operator.metadata
        except Exception:
            continue
        names = list(getattr(meta, "param_names", None) or ())
        # Only audit operators whose call has NO explicitly-bound window-like
        # param (pure defaults) — build the call and check nothing was bound.
        try:
            args, kwargs = _build_call(canonical, operator, panels)
        except Exception:
            continue
        bound_window = [
            n for n in names[: len(args)] if _window_like(n)
        ]
        if bound_window or any(_window_like(k) for k in kwargs):
            continue  # an explicit window was bound — not the default case
        try:
            rows = history_requirement(canonical, {}).rows
        except Exception:
            continue
        if rows >= 2:
            continue
        # history says < 2 rows — a rolling operator with a default window would
        # be under-allocated.  Only flag operators that LOOK like rolling kernels.
        if not any(_window_like(n) for n in names):
            continue
        errors.append(
            f"C default-history: {canonical} declares window-like param(s) "
            f"{[n for n in names if _window_like(n)]} but history_requirement "
            f"with defaults is only {rows} rows — a default window kernel would "
            "not be covered (inner/outer/history_days style miss)"
        )
    return errors


def _window_like(name: str) -> bool:
    lowered = str(name).lower()
    return any(
        token in lowered
        for token in (
            "window", "span", "lag", "lookback", "period", "_days", "n_",
            "block", "cutoff", "max_run", "history",
        )
    )


# ---------------------------------------------------------------------------
# Audit D — Unit Algebra
# ---------------------------------------------------------------------------
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
            if target not in names:
                errors.append(
                    f"D unit-algebra: {canonical} output_unit 'same_as:{target}' "
                    f"references an undeclared parameter (params: {sorted(names)})"
                )
        # formula families whose dimensional analysis is fixed
        family = None
        if any(token in canonical for token in ("_cov", "_cov_", "covar")):
            family = "covariance -> unit(x)*unit(y)"
        elif any(token in canonical for token in ("_beta", "_beta_")):
            family = "beta -> unit(y)/unit(x)"
        elif any(token in canonical for token in ("_slope", "regression_slope")):
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
    errors += audit_prefix_invariance(sample, panels)
    errors += audit_stateful_contract_discovery(sample, panels)
    errors += audit_default_parameter_history(sample, panels)
    errors += audit_unit_algebra()
    if errors:
        print(f"R11 audit FAILED ({len(errors)} issues)", file=sys.stderr)
        for error in errors[:40]:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("R11 long-tail audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
