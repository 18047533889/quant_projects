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
    """One release invariant with its current violation list.

    R16-035: every detector carries a five-state outcome
    PASS / FAIL / N/A / NOT_RUN / AUDIT_ERROR.  A detector that did NOT run
    (e.g. behavioral source-scan disabled) is NOT_RUN — an empty finding list
    must never be mistaken for PASS.
    """

    name: str
    description: str
    violations: list[str] = field(default_factory=list)
    outcome: str = "PASS"  # PASS | FAIL | N/A | NOT_RUN | AUDIT_ERROR
    required: bool = True  # a required detector in NOT_RUN blocks release

    @property
    def count(self) -> int:
        return len(self.violations)

    def finalize(self) -> None:
        """Compute the outcome from the violation list."""
        if self.outcome == "NOT_RUN":
            return
        if self.outcome == "AUDIT_ERROR":
            return
        self.outcome = "FAIL" if self.violations else "PASS"


@dataclass
class ClosureReport:
    invariants: dict[str, ClosureInvariant] = field(default_factory=dict)

    def add(self, name: str, description: str, *, required: bool = True) -> ClosureInvariant:
        inv = ClosureInvariant(name, description, required=required)
        self.invariants[name] = inv
        return inv

    def summary(self) -> dict[str, int]:
        return {k: v.count for k, v in self.invariants.items()}

    def release_safe(self, allowlist: set[str] | None = None) -> bool:
        allow = allowlist or set()
        for v in self.invariants.values():
            if v.name in allow:
                continue
            if v.outcome in ("FAIL", "AUDIT_ERROR"):
                return False
            # R16-035: a REQUIRED detector that did not run (NOT_RUN) blocks.
            if v.required and v.outcome == "NOT_RUN":
                return False
            if v.count and v.outcome == "PASS":
                return False
        return True


# ---------------------------------------------------------------------------
# operator metadata helpers
# ---------------------------------------------------------------------------

def _is_production(entry: dict[str, Any]) -> bool:
    return entry.get("status") in ("production", "implemented")


def _panel_params_typed(entry: dict[str, Any]) -> tuple[str, ...]:
    """R16-036: the typed ``panel_params`` contract is authoritative.  The
    legacy ``_PANEL_PARAMS`` name whitelist (group_id/high_limit/activity/f1/
    benchmark … are easily missed) is only a fallback for operators that never
    declared the typed contract."""
    declared = entry.get("panel_params")
    if isinstance(declared, (list, tuple, frozenset, set)) and declared:
        return tuple(str(p) for p in declared)
    return ()


def _scalar_params(entry: dict[str, Any], specs: dict[str, Any]) -> list[str]:
    """Non-panel parameter names for an operator, preserving order."""
    names = list(entry.get("param_names") or [])
    typed = _panel_params_typed(entry)
    if typed:
        return [p for p in names if p not in typed]
    return [p for p in names if p not in _PANEL_PARAMS]


def _specs_of(entry: dict[str, Any]) -> dict[str, Any]:
    return dict(entry.get("param_specs") or {})


def _multi_input(entry: dict[str, Any]) -> bool:
    names = list(entry.get("param_names") or [])
    typed = _panel_params_typed(entry)
    panels = [p for p in names if p in typed] if typed else [
        p for p in names if p in _PANEL_PARAMS
    ]
    # operator has more than one panel input (x,y / x,z / price+event ...)
    return len(panels) >= 2


def _has_session_contract(canonical: str) -> bool:
    """R16-031: does the canonical declare a SessionContract / session calendar?"""
    try:
        from cleaned_operators.operator_spec import build_operator_spec

        spec = build_operator_spec(canonical)
        sc = getattr(spec, "session_contract", None)
        if sc:
            return True
    except Exception:
        pass
    try:
        from cleaned_operators.registry import OperatorRegistry

        entry = OperatorRegistry._catalog.get(canonical, {})
        return bool(
            entry.get("session_contract")
            or entry.get("session_calendar")
            or entry.get("available_at")
        )
    except Exception:
        return False


def _is_minute_family(canonical: str) -> bool:
    """R16-032: intraday / session / volume-clock family canonicals are minute
    INPUT by construction even before a grain contract is declared."""
    return canonical.startswith(
        ("intraday_", "intra_", "session_", "volume_clock", "minute_")
    ) or "_session" in canonical or "_intraday" in canonical


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


# R15-INC-063: source-scan invariant over shared helper modules too.  The
# recursive sweep below covers every ``.py`` under ``cleaned_operators/`` —
# helper modules carry the same clamp / int-truncation risk as operator bodies,
# so a scan limited to top-level ``cleaned_operators/*.py`` was a blind spot.
_SKIP_SOURCE_FILES = frozenset(
    {
        "base.py", "base_polars.py", "registry.py", "operator_audits.py",
        "closure_audit.py", "semantic_audit.py", "operator_spec.py",
        "operator_surface.py", "operator_cost_model.py",
        "parameter_validation.py", "safe_ops.py",
    }
)


@dataclass
class SourceScan:
    """R15-INC-062: recursive source scan with full coverage accounting."""

    scanned_files: list[str]
    skipped_files: list[str]
    scanned_canonicals: set[str]
    clamp: list[str]
    int_trunc: list[str]


def _canonical_module_file(canonical: str) -> str | None:
    """Real canonical → implementation file (relative to cleaned_operators).

    Uses the pandas backend operator class's ``__module__``/``__file__`` — the
    mapping R15-INC-062 demands (registry canonical → source file), never a
    filename or regex guess.  Returns ``None`` when the implementation does not
    live inside the tree (external tool, dynamically constructed class, test
    fixture).
    """
    try:
        from cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry._operators.get(canonical, {}).get("pandas_numpy")
        if operator is None:
            return None
        module_name = type(operator).__module__
        import sys

        module = sys.modules.get(module_name)
        if module is None:
            return None
        file_path = getattr(module, "__file__", None)
        if not file_path:
            return None
        root = Path(__file__).resolve().parent.parent
        path = Path(file_path).resolve()
        try:
            return str(path.relative_to(root))
        except ValueError:
            return None
    except Exception:
        return None


def _scan_source_files() -> SourceScan:
    """Recursively scan every operator/helper .py for suspected clamp /
    int-truncation lines and record which registered canonicals the scanned
    files actually declare (R15-INC-062: a registry-wide audit must prove it
    scanned the whole tree, not just the top-level files)."""
    root = Path(__file__).resolve().parent.parent
    scanned: list[str] = []
    skipped: list[str] = []
    scanned_canonicals: set[str] = set()
    clamp: list[str] = []
    int_trunc: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        # R16-033: ``_*.py`` PRIVATE SHARED KERNELS (``_rolling_fast.py`` /
        # ``_numpy_kernels.py`` / ``_dedupe.py``) ARE production source and must
        # be scanned — a filename rule silently excluded them.  Only generated /
        # vendor / cache and the infra-metadata whitelist are skipped.
        if path.name in _SKIP_SOURCE_FILES:
            skipped.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            skipped.append(rel)
            continue
        for m in re.finditer(
            r'@register_operator\s*\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']',
            text,
        ):
            scanned_canonicals.add(m.group(1))
        for lineno, line in enumerate(text.splitlines(), 1):
            if _CLAMP_RE.search(line):
                clamp.append(f"{rel}:{lineno}: {line.strip()[:140]}")
            if _INT_TRUNC_RE.search(line):
                int_trunc.append(f"{rel}:{lineno}: {line.strip()[:140]}")
        scanned.append(rel)
    return SourceScan(
        scanned_files=scanned,
        skipped_files=skipped,
        scanned_canonicals=scanned_canonicals,
        clamp=clamp,
        int_trunc=int_trunc,
    )


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

    ``include_behavioral`` (R15-INC-071) is NOT a dead option: when False the
    expensive recursive source-scan invariants (int-trunc / clamp / unscanned-
    canonical) are skipped — a caller that only wants the contract invariants
    pays none of the source-scan cost.
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
    report.add(
        "UNSCANNED_REGISTERED_CANONICALS",
        "R15-INC-062: a registered canonical whose implementation file was not "
        "covered by the recursive source scan (a registry-wide audit must prove "
        "full-tree coverage; == ∅ is a release gate)",
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

        # mining reachability — R16-030: INTRINSIC only.  ``mining_eligible``
        # with no source context fails closed to UNKNOWN, which would flag
        # every operator as UNUSED.  Closure checks intrinsic reachability
        # (certification + mineable role + cost contract); REAL source
        # eligibility is a MiningContext decision, never a closure invariant.
        try:
            intrinsic = (
                bool(entry.get("production_certified"))
                and _mining.assign_mining_role(canonical, entry)
                in _mining._MINEABLE_ROLES
                and _mining.cost_contract_declared(canonical, entry)
            )
        except Exception:  # pragma: no cover
            intrinsic = False
        if not intrinsic and canonical not in research_tools:
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

        # minute <-> daily session contract (R16-031/032).
        g_in, g_out = entry.get("input_grain"), entry.get("output_grain")
        str_in = str(g_in or "").lower()
        str_out = str(g_out or "").lower()
        eod_ok = (
            entry.get("available_at") == "session_close"
            and entry.get("same_session_usable") is False
        )
        session_contract = _has_session_contract(canonical)
        # R16-032: a minute-input canonical MUST declare a COMPLETE grain
        # contract — comparing only when both sides exist hid the missing-
        # declaration case.
        if str_in == "minute" or _is_minute_family(canonical):
            if not g_in or not g_out:
                inv_m2d.violations.append(
                    f"{canonical} incomplete GrainContract (in={g_in!r}, out={g_out!r})"
                )
            elif eod_ok:
                pass  # R16-031: EOD factor (available_at=session_close,
                      # same_session_usable=False) is the CORRECT state.
            elif str_in != str_out and entry.get("same_session_usable") and not session_contract:
                inv_m2d.violations.append(
                    f"{canonical} ({str_in}->{str_out}) marked same_session_usable=True "
                    "without a SessionContract"
                )

    # source-scan invariants (recursive; NOT_RUN when include_behavioral=False)
    scan = None
    if include_behavioral:
        scan = _scan_source_files()
    inv_clamp = report.invariants["OPERATOR_WITH_SILENT_PARAM_CLAMP"]
    inv_int = report.invariants["OPERATOR_WITH_RUNTIME_INT_TRUNCATION"]
    inv_unscanned = report.invariants["UNSCANNED_REGISTERED_CANONICALS"]
    if scan is None:
        # R16-035: disabling the behavioral detectors is NOT a PASS — they are
        # required detectors and report NOT_RUN, which blocks release_safe.
        inv_clamp.outcome = "NOT_RUN"
        inv_int.outcome = "NOT_RUN"
        inv_unscanned.outcome = "NOT_RUN"
    else:
        for loc in scan.clamp:
            inv_clamp.violations.append(loc)
        for loc in scan.int_trunc:
            inv_int.violations.append(loc)
        # R15-INC-062: build the REAL canonical → implementation-file map (from
        # the operator class's module) instead of guessing by filename or a
        # name-extraction regex.  R16-034: a canonical whose implementation
        # file was NOT scanned — whether skipped or unresolved — is UNSCANNED
        # (NOT_RUN), never silently covered.
        for canonical in sorted(catalog):
            file = _canonical_module_file(canonical)
            if file is None:
                continue  # no in-tree implementation file (external/dynamic)
            if file not in scan.scanned_files:
                inv_unscanned.violations.append(f"{canonical} ({file})")

    # finalize every invariant into its five-state outcome (R16-035)
    for inv in report.invariants.values():
        inv.finalize()
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
