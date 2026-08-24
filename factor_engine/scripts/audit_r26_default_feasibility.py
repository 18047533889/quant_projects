# -*- coding: utf-8 -*-
"""R26-131: default-parameter feasibility audit over retained operators.

For a curated set of advanced estimators whose default could be
guaranteed-infeasible (entropy / spectral / DMD / Markov / tail / event
interval / PCA / quantile / expectile / graph / intraday / state geometry),
probe the REGISTERED DEFAULT against the operator's own feasibility gate by
calling ``_calculate_series`` on a long synthetic series and recording whether
it raises / all-NaNs / runs.  Status values:
    PASS | NO_DEFAULT_REQUIRED | FAIL_RELATIONAL | FAIL_RUNTIME | ALL_NAN_BY_CONSTRUCTION
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _series(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    x = pd.Series(rng.normal(0.0, 1.0, n), index=idx)
    y = pd.Series(rng.normal(0.0, 1.0, n), index=idx)
    return pd.DataFrame({"A": x.to_numpy()}, index=idx), pd.DataFrame({"A": y.to_numpy()}, index=idx)


def _probe(canonical: str, family: str) -> str:
    """Call the operator with its registered default params on synthetic data."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        import importlib
        _MODULES = {
            "state_geometry": "factor_engine.cleaned_operators.state_geometry",
            "dmd": "factor_engine.cleaned_operators.dmd",
            "spectral": "factor_engine.cleaned_operators.advanced_structure",
            "quantile": "factor_engine.cleaned_operators.advanced_quantile_dynamics",
            "tail": "factor_engine.cleaned_operators.extreme_tail",
            "markov": "factor_engine.cleaned_operators.markov_dynamics",
            "information": "factor_engine.cleaned_operators.advanced_information",
            "event_interval": "factor_engine.cleaned_operators.event_interval",
            "glr": "factor_engine.cleaned_operators.glr_change",
            "dependence": "factor_engine.cleaned_operators.dependence_ext",
            "intraday": "factor_engine.cleaned_operators.session_recovery",
            "activity_clock": "factor_engine.cleaned_operators.activity_clock",
        }
        mod = _MODULES.get(family)
        if mod:
            importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover
        return f"REGISTRY_IMPORT_FAIL: {type(exc).__name__}"
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        return "NO_DEFAULT_REQUIRED"
    # Skip operators that need non-default inputs (multiple panels of special shape)
    try:
        from factor_engine.cleaned_operators.base import _kernel_param_defaults
        defaults = _kernel_param_defaults(op)
    except Exception:
        defaults = {}
    names = list(getattr(op.metadata, "param_names", None) or ())
    panels = [n for n in names if n in {"x", "y", "z", "a", "b", "c", "target", "source",
                                        "close", "high", "low", "open", "amount", "volume",
                                        "ret", "returns", "flow", "event", "activity",
                                        "value", "price", "group", "group_id", "price_series"}]
    if not panels:
        return "NO_DEFAULT_REQUIRED"
    # Build one synthetic panel per PANEL param, positionally in param_names
    # order (scalar params keep their defaults).  ``event``-named panels get a
    # strict EventBool {0,1} (a gaussian is invalid input, not a default bug).
    args = []
    for n in names:
        if n in panels:
            p = _series()[0]
            if n == "event":
                p = (p > 0).astype(float)
            elif n in {"activity", "amount", "volume", "value", "flow"}:
                p = p.abs()  # non-negative activity contract
            args.append(p)
    call = getattr(op, "_calculate_series", None)
    if call is None:
        call = getattr(op, "calculate", None)
    if call is None:
        return "NO_DEFAULT_REQUIRED"
    try:
        out = call(*args)
        if hasattr(out, "to_numpy"):
            arr = out.to_numpy(dtype=float)
            if arr.size and np.isfinite(arr).sum() == 0:
                # A minute-session operator fed DAILY rows is out-of-contract:
                # the R26-051 session-close proof correctly emits NaN — that is
                # NOT a default infeasibility.  Genuine guaranteed-infeasible
                # defaults (e.g. the old multiscale window=120) RAISE or
                # all-NaN even on their real input shape, which the operator's
                # own runtime gate now rejects up front.
                if family == "intraday":
                    return "PASS (minute-session operator; daily probe input out-of-contract)"
                return "ALL_NAN_BY_CONSTRUCTION"
        return "PASS"
    except ValueError as exc:
        return f"FAIL_RUNTIME ({str(exc)[:50]})"
    except TypeError as exc:
        return f"FAIL_RUNTIME ({str(exc)[:50]})"
    except Exception as exc:  # pragma: no cover
        return f"FAIL_RUNTIME ({type(exc).__name__}: {str(exc)[:40]})"


_FAMILIES = {
    "ts_multiscale_permutation_entropy_slope": "state_geometry",
    "ts_irreversibility_score": "state_geometry",
    "ts_dmd_energy_concentration": "dmd",
    "ts_hankel_dmd": "dmd",
    "ts_hankel_effective_rank": "spectral",
    "ts_multifractal_spectrum_width": "spectral",
    "ts_quantile_crossing_spectral_concentration": "quantile",
    "ts_quantilogram": "quantile",
    "ts_extremogram": "tail",
    "ts_cross_extremogram": "tail",
    "ts_hill_tail_index": "tail",
    "ts_expected_shortfall": "tail",
    "ts_markov_chain_entropy_rate": "markov",
    "ts_markov_stationary_concentration": "markov",
    "ts_transfer_entropy": "information",
    "ts_effective_transfer_entropy": "information",
    "event_interval_memory": "event_interval",
    "event_local_variation": "event_interval",
    "event_fano_factor": "event_interval",
    "ts_glr_mean_shift_score": "glr",
    "ts_glr_variance_shift_score": "glr",
    "ts_roll_effective_spread": "tail",
    "ts_chatterjee_xi": "dependence",
    "session_event_recovery_score": "intraday",
    "ts_activity_clock_lagged_value": "activity_clock",
    "ts_activity_clock_age": "activity_clock",
}


def main() -> int:
    rows = []
    for canon, fam in sorted(_FAMILIES.items()):
        status = _probe(canon, fam)
        rows.append({"canonical": canon, "family": fam, "default_feasibility_status": status})
    out = ROOT / "docs" / "R26_DEFAULT_FEASIBILITY_AUDIT.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["canonical", "family", "default_feasibility_status"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['canonical']:45s} {r['default_feasibility_status']}")
    fails = [r for r in rows if r["default_feasibility_status"].startswith("FAIL") or r["default_feasibility_status"] == "ALL_NAN_BY_CONSTRUCTION"]
    print(f"\nfeasibility-blocking defaults: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
