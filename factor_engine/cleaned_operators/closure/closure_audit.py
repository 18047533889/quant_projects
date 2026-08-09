# -*- coding: utf-8 -*-
"""Operator Semantic Closure audit (Master Spec Part BY).

Mechanically reports every release invariant across the FULL public catalog —
the spec's "必须满足" list.  The audit is registry-level (iterates the whole
catalog) as opposed to the per-operator behavioral audits in
``cleaned_operators.closure_audit`` / ``semantic_audit``.

Each invariant reports ``(count, violations)``; a release gate requires all of
them to be 0 (or an allowlisted diagnostics set).  Source-scan invariants
(silent clamp, int truncation) sweep the operator ``.py`` files with a
conservative regex and report suspected lines for manual review — the regex can
not prove intent, so a hit is a *candidate*, and the audit's job is to drive
the count to 0 by replacing the pattern with a strict validator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from cleaned_operators.closure.axis_contract import has_axis_contract
from cleaned_operators.closure.missing_policy import (
    MissingPolicy,
    missing_policy_for,
)
from cleaned_operators.closure.window_semantics import window_semantics_for

# Panel-positional parameter names (excluded from the "scalar parameter"
# analysis — they are the data inputs, not tunable knobs).
_PANEL_PARAMS = frozenset(
    {
        "x", "y", "z", "a", "b", "c", "v", "u", "w", "p", "q",
        "price", "event", "x_df", "y_df", "returns", "return", "values",
        "group", "mask", "market", "weights", "weight",
        # OHLCV panel inputs are data, never scalar knobs
        "open", "high", "low", "close", "volume", "amount", "vwap",
        "open_p", "high_p", "low_p", "close_p", "volume_p", "amount_p",
        "source", "target", "a_", "b_",
    }
)

# Parameter names that are unambiguously numerical-policy / estimator knobs and
# MUST carry an explicit ParamRole under Part A-2 (never default ECONOMIC).
_ESTIMATOR_KNOB_PREFIXES = (
    "bins", "n_seg", "n_bin", "n_surr", "surrogate", "seed", "theiler",
    "embedding", "dim", "k_max", "min_periods", "min_pairs", "min_nodes",
    "min_cells", "min_coverage", "n_component", "max_iter", "n_iter", "tol",
    "eps", "epsilon", "delta", "n_scale", "max_scale", "k_", "lag", "delay",
    "max_gap", "purge", "block", "order", "degree", "decimals", "trim",
    "n_levels", "cutoff", "min_obs",
)
_ESTIMATOR_KNOB_EXACT = {
    "bins", "k", "seed", "purge_gap", "n_segments", "theiler_window",
    "embedding_dim", "dim", "max_iter", "n_iter", "min_periods", "min_pairs",
    "min_nodes", "min_cells_ratio", "min_coverage_fraction", "n_surrogates",
    "epsilon", "tol", "scale", "max_scale", "n_scales", "order", "degree",
    "ddof", "block", "max_gap", "lag", "delay", "min_obs", "min_count",
}


@dataclass
class ClosureInvariant:
    """One release invariant with its current violation list."""

    name: str
    description: str
    violations: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.violations)


@dataclass
class ClosureReport:
    invariants: dict[str, ClosureInvariant] = field(default_factory=dict)

    def add(self, name: str, description: str) -> ClosureInvariant:
        inv = ClosureInvariant(name, description)
        self.invariants[name] = inv
        return inv

    def summary(self) -> dict[str, int]:
        return {k: v.count for k, v in self.invariants.items()}

    def release_safe(self, allowlist: set[str] | None = None) -> bool:
        allow = allowlist or set()
        return all(v.count == 0 or v.name in allow for v in self.invariants.values())


# ---------------------------------------------------------------------------
# operator metadata helpers
# ---------------------------------------------------------------------------

def _is_production(entry: dict[str, Any]) -> bool:
    return entry.get("status") in ("production", "implemented")


def _scalar_params(entry: dict[str, Any], specs: dict[str, Any]) -> list[str]:
    """Non-panel parameter names for an operator, preserving order."""
    names = list(entry.get("param_names") or [])
    return [p for p in names if p not in _PANEL_PARAMS]


def _specs_of(entry: dict[str, Any]) -> dict[str, Any]:
    return dict(entry.get("param_specs") or {})


def _multi_input(entry: dict[str, Any]) -> bool:
    names = list(entry.get("param_names") or [])
    panels = [p for p in names if p in _PANEL_PARAMS]
    # operator has more than one panel input (x,y / x,z / price+event ...)
    return len(panels) >= 2


# ---------------------------------------------------------------------------
# source-scan invariants
# ---------------------------------------------------------------------------

# Part A-4: silent ``int(...)`` cast of a runtime parameter value.  Conservative
# regex — matches ``int(<name>)`` / ``int(<expr>)`` where the argument reads a
# parameter (not a literal).  Only flagged inside operator kernel files.
_INT_TRUNC_RE = re.compile(
    r"(?<!def\s)\bint\s*\(\s*"
    r"(?!['\"])([a-zA-Z_][a-zA-Z0-9_]*|self\.\w+|kwargs?\.\w+)"
    r"\s*\)"
)

# Part A-5: silent clamp ``max(2, int(window))`` / ``min(1, int(lag))``.
_CLAMP_RE = re.compile(
    r"\b(?:max|min)\s*\(\s*[0-9.]+\s*,\s*int\s*\("
)


def _scan_source_files() -> dict[str, list[tuple[int, str]]]:
    """Scan operator .py files for suspected clamp / int-truncation lines."""
    root = Path(__file__).resolve().parent.parent
    results: dict[str, list[tuple[int, str]]] = {"clamp": [], "int_trunc": []}
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        if path.name in {
            "base.py", "registry.py", "operator_audits.py", "closure_audit.py",
            "semantic_audit.py", "operator_spec.py", "operator_surface.py",
            "operator_cost_model.py", "parameter_validation.py", "safe_ops.py",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _CLAMP_RE.search(line):
                results["clamp"].append((f"{path.name}:{lineno}", line.strip()[:140]))
            if _INT_TRUNC_RE.search(line):
                results["int_trunc"].append((f"{path.name}:{lineno}", line.strip()[:140]))
    return results


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def run_semantic_closure_audit(
    catalog: dict[str, Any] | None = None,
    *,
    include_behavioral: bool = True,
) -> ClosureReport:
    """Run every registry-level Part-BY invariant over the public catalog.

    ``catalog`` may be passed pre-loaded (e.g. a filtered subset for testing);
    when ``None`` the full loaded catalog is used.
    """
    from cleaned_operators.registry import OperatorRegistry

    if catalog is None:
        from cleaned_operators import load_all

        load_all()
        catalog = dict(OperatorRegistry.catalog())
    # Ensure the round-14 statistical-family declarations are present so the
    # UNDECLARED_* invariants reflect the committed contracts.
    from cleaned_operators.closure.declared_policies import declare_all

    declare_all()
    from cleaned_operators import operator_surface as _surface
    from mining import operator_catalog as _mining

    report = ClosureReport()
    report.add(
        "UNCLASSIFIED_PUBLIC_OPERATOR",
        "Registered operator not classified by any mining surface",
    )
    report.add("UNUSED_PUBLIC_OPERATOR", "Registered operator never reachable from mining")
    report.add(
        "UNDECLARED_PARAM_ROLE",
        "Searchable scalar parameter without an explicit ParamRole "
        "(Part A-2: undeclared role must fail production certification)",
    )
    report.add(
        "UNDECLARED_MISSING_POLICY",
        "Statistical/time-geometry operator without a declared MissingPolicy "
        "(Part C-10)",
    )
    report.add(
        "UNDECLARED_WINDOW_SEMANTICS",
        "Windowed operator without a declared WindowSemantics (Part D-16)",
    )
    report.add(
        "MULTI_INPUT_WITHOUT_AXIS_CONTRACT",
        "Multi-input operator without a declared SameAxis contract (Part BM-259)",
    )
    report.add(
        "GLOBAL_STATE_AS_ALPHA_TERMINAL",
        "Global-state operator marked terminal-allowed as a factor (Part BR-269)",
    )
    report.add(
        "MINUTE_TO_DAILY_WITHOUT_SESSION_CONTRACT",
        "Frequency-changing operator without declared input/output grain "
        "(Part WS)",
    )
    report.add(
        "OPERATOR_WITH_SILENT_PARAM_CLAMP",
        "max/min over int(...) candidate — silent clamp of a legal parameter "
        "(Part A-5)",
    )
    report.add(
        "OPERATOR_WITH_RUNTIME_INT_TRUNCATION",
        "int(<param>) candidate — silent float truncation (Part A-4)",
    )

    inv_und_class = report.invariants["UNCLASSIFIED_PUBLIC_OPERATOR"]
    inv_unused = report.invariants["UNUSED_PUBLIC_OPERATOR"]
    inv_role = report.invariants["UNDECLARED_PARAM_ROLE"]
    inv_mp = report.invariants["UNDECLARED_MISSING_POLICY"]
    inv_ws = report.invariants["UNDECLARED_WINDOW_SEMANTICS"]
    inv_axis = report.invariants["MULTI_INPUT_WITHOUT_AXIS_CONTRACT"]
    inv_gs = report.invariants["GLOBAL_STATE_AS_ALPHA_TERMINAL"]
    inv_m2d = report.invariants["MINUTE_TO_DAILY_WITHOUT_SESSION_CONTRACT"]

    research_tools = set()
    try:
        from research_tools.registry import ResearchToolRegistry

        research_tools = set(ResearchToolRegistry.list_canonical())
    except Exception:  # pragma: no cover
        pass

    for canonical, entry in sorted(catalog.items()):
        status = entry.get("status", "")
        specs = _specs_of(entry)
        scalar = _scalar_params(entry, specs)

        # surface classification
        try:
            surface = _surface.classify_canonical(canonical)
        except Exception:  # pragma: no cover
            surface = "unclassified"
        if surface == "unclassified" and status not in (
            "research", "deprecated",
        ):
            inv_und_class.violations.append(canonical)

        # mining reachability
        try:
            eligible = _mining.mining_eligible(canonical, catalog=entry)
        except Exception:  # pragma: no cover
            eligible = False
        if not eligible and canonical not in research_tools:
            inv_unused.violations.append(canonical)

        # scalar params: declared role required (Part A-2 fail-closed)
        for p in scalar:
            spec = specs.get(p)
            role = getattr(spec, "param_role", None)
            searchable = getattr(spec, "searchable", True)
            if searchable and role is None and status != "deprecated":
                inv_role.violations.append(f"{canonical}.{p}")

        # MissingPolicy / WindowSemantics declaration (statistical families)
        if status not in ("research", "deprecated"):
            if missing_policy_for(canonical) is None:
                inv_mp.violations.append(canonical)
            if window_semantics_for(canonical) is None:
                inv_ws.violations.append(canonical)

        # multi-input axis contract
        if _multi_input(entry) and not has_axis_contract(canonical):
            inv_axis.violations.append(canonical)

        # global-state as alpha terminal: the mining layer is the authority on
        # terminal-ness.  A metadata ``role=global_state`` is only a violation if
        # mining ALSO assigns it a terminal role — if mining assigns
        # GLOBAL_STATE (regime/condition only) the invariant is satisfied.
        role_field = entry.get("role")
        if role_field == "global_state":
            try:
                mining_role = _mining.assign_mining_role(canonical, entry)
                mining_role_value = (
                    mining_role.value
                    if hasattr(mining_role, "value")
                    else str(mining_role)
                )
            except Exception:  # pragma: no cover
                mining_role_value = None
            if mining_role_value in ("alpha", "intraday_eod"):
                inv_gs.violations.append(
                    f"{canonical} (metadata role=global_state but mining role "
                    f"{mining_role_value} is a terminal alpha slot)"
                )

        # minute -> daily session contract
        g_in, g_out = entry.get("input_grain"), entry.get("output_grain")
        if g_in and g_out and g_in != g_out and not entry.get("same_session_usable"):
            inv_m2d.violations.append(
                f"{canonical} ({g_in}->{g_out}, same_session_usable unset)"
            )

    # source-scan invariants
    scan = _scan_source_files()
    inv_clamp = report.invariants["OPERATOR_WITH_SILENT_PARAM_CLAMP"]
    inv_int = report.invariants["OPERATOR_WITH_RUNTIME_INT_TRUNCATION"]
    for loc, line in scan["clamp"]:
        inv_clamp.violations.append(f"{loc}: {line}")
    for loc, line in scan["int_trunc"]:
        inv_int.violations.append(f"{loc}: {line}")

    return report


def format_report(report: ClosureReport, *, limit: int = 8) -> str:
    """Human-readable summary of a closure audit run."""
    lines = [f"Semantic Closure audit — {len(report.invariants)} invariants"]
    for name, inv in sorted(report.invariants.items()):
        lines.append(f"  {name}: {inv.count}")
        for v in inv.violations[:limit]:
            lines.append(f"      - {v}")
        if inv.count > limit:
            lines.append(f"      … (+{inv.count - limit} more)")
    lines.append(f"release_safe: {report.release_safe()}")
    return "\n".join(lines)
