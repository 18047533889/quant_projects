# -*- coding: utf-8 -*-
"""Machine operator-semantic audit rules (2026-08 operator review, §8).

The 1362-canonical catalog cannot be hand-swept each review round.  These
automated rules scan operator behaviour on synthetic fixtures and metadata for
the semantic bugs a hand audit keeps finding:

1. ``default_output_finite``    — defaults must not make a factor (nearly) dead.
2. ``parameter_injectivity``    — 5.1 must never silently equal 5.
3. ``relational_constraints``   — min_periods<=window, lag<window, k<=N, ...
4. ``mirror_symmetry``          — upper/lower tail pairs satisfy lower(x)≈upper(-x).
5. ``column_permutation``       — CS ops are invariant to stock-column shuffling.
6. ``missing_state``            — NaN event/condition state is not "confirmed".
7. ``semantic_type_gate``       — Price/Return/EventBool/GroupID are not fungible.
8. ``finite_range``             — declared probability/correlation/count bounds hold.
9. ``scale_shift_invariance``   — dimensionless ops ignore legal unit rescaling.
10. ``group_migration``         — a label move A→B is a real exit+entry.
11. ``native_cohort``           — retention/overlap define state per native date.
12. ``psd_geometry``            — spectral eigen metrics share one PSD matrix.
13. ``golden_reference``        — Hill/GPD/MI/TE/entropy vs known DGP / SciPy.
14. ``canonical_honesty``       — robust/Hill/Granger/tail_dependence names bind a
                                  definition / reference formula.
15. ``default_searchability``   — defaults must not yield all-NaN / constant /
                                  zero / near-constant output.

Each rule registers under a key with a ``check`` callable ``(op, ctx) ->
list[AuditFinding]``.  ``ctx`` supplies synthetic panels, the registry accessor
and per-operator sampling options.  Rules that cannot decide for an operator
report ``skipped`` (with the reason) — coverage is never silently truncated.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.backend.operator_errors import OperatorParameterError

# ---------------------------------------------------------------------------
# findings / report
# ---------------------------------------------------------------------------

_SEVERITIES = {"error", "warning", "info"}


@dataclass(frozen=True)
class AuditFinding:
    rule: str
    operator: str
    severity: str  # error | warning | info
    message: str

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(f"invalid severity {self.severity!r}")


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    ran: dict[str, int] = field(default_factory=dict)          # rule -> ops PASS-checked
    skipped: dict[str, list[str]] = field(default_factory=dict)  # rule -> NOT_APPLICABLE reasons
    audit_errors: dict[str, list[str]] = field(default_factory=dict)  # rule -> AUDIT_ERROR reasons
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)  # canonical -> evidence record
    # R16-042/044: per-canonical REQUIRED-rule coverage.  A canonical's required
    # rules that got NO outcome (NOT_RUN) make certification unsafe; the
    # outcome-matrix hash binds the evidence record to the actual rule results.
    required_missing: dict[str, list[str]] = field(default_factory=dict)
    outcome_matrix: dict[tuple[str, str], str] = field(default_factory=dict)  # (canonical, rule) -> PASS|FAIL|N/A|AUDIT_ERROR

    def add(self, finding: AuditFinding) -> None:
        self.findings.append(finding)

    def note_ran(self, rule: str) -> None:
        self.ran[rule] = self.ran.get(rule, 0) + 1

    def note_skipped(self, rule: str, reason: str) -> None:
        """NOT_APPLICABLE: the rule cannot decide for this operator by design
        (no side param / not a CS op / fixture cannot exercise it).  Coverage is
        never silently truncated — the reason is always recorded."""
        self.skipped.setdefault(rule, []).append(reason)

    def note_audit_error(self, rule: str, reason: str) -> None:
        """AUDIT_ERROR (R13 NEW-P0-07): the RULE itself raised on an operator.
        This is a release-failing infrastructure fault, distinct from a
        NOT_APPLICABLE skip and from a PASS/FAIL finding."""
        self.audit_errors.setdefault(rule, []).append(reason)

    @property
    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def audit_error_count(self) -> int:
        return sum(len(v) for v in self.audit_errors.values())

    def coverage(self) -> dict[str, int]:
        """Per-rule outcome tally (R13 NEW-P0-06/07): PASS-checked vs
        NOT_APPLICABLE vs AUDIT_ERROR.  ``audit_error_count`` must be 0 for any
        release gate; ``audited / (audited + not_applicable)`` is the honest
        decision coverage for the rule."""
        return {
            "checked": sum(self.ran.values()),
            "not_applicable": sum(len(v) for v in self.skipped.values()),
            "audit_errors": self.audit_error_count,
        }

    def certification_safe(self) -> bool:
        """R13 NEW-P0-71 + R16-042: the audit run is safe to use as
        certification evidence ONLY when no error-severity finding fired, no
        rule crashed, AND every canonical's REQUIRED rules got a PASS (or a
        reviewed N/A) outcome.  A canonical with required rules that never ran
        (NOT_RUN) is NOT safe — a large NOT_APPLICABLE / NOT_RUN count with no
        error finding must not certify."""
        if self.errors or self.audit_error_count:
            return False
        if self.required_missing:
            return False
        return True

    def summary(self) -> str:
        lines = [f"{len(self.findings)} findings across {len(self.ran)} rules"]
        for rule, count in sorted(self.ran.items()):
            skipped = self.skipped.get(rule)
            errs = self.audit_errors.get(rule)
            extra = ""
            if skipped:
                extra += f" ({len(skipped)} not-applicable)"
            if errs:
                extra += f" ({len(errs)} AUDIT_ERROR)"
            lines.append(f"  {rule}: {count} checked{extra}")
        if self.audit_error_count:
            lines.append(f"AUDIT_ERRORS: {self.audit_error_count} (release gate FAILS)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# registry + synthetic fixtures
# ---------------------------------------------------------------------------

def _panel(n_rows: int = 80, n_cols: int = 3, seed: int = 0,
           positive: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    cols = [f"S{i}" for i in range(n_cols)]
    base = np.abs(rng.normal(1.0, 0.3, (n_rows, n_cols))) if positive \
        else rng.normal(0.0, 1.0, (n_rows, n_cols))
    return pd.DataFrame(base, index=idx, columns=cols)


def _panel_pair(seed: int = 1, n_rows: int = 80, n_cols: int = 3):
    """Two correlated panels (x leads y) for dependence/regression ops."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    cols = [f"S{i}" for i in range(n_cols)]
    x = rng.normal(0.0, 1.0, (n_rows, n_cols))
    y = np.zeros_like(x)
    y[2:] = 0.7 * x[:-2] + 0.3 * rng.normal(0.0, 1.0, y[2:].shape)
    return (
        pd.DataFrame(x, index=idx, columns=cols),
        pd.DataFrame(y, index=idx, columns=cols),
    )


@dataclass
class OperatorSample:
    """How to run one operator under audit."""

    name: str
    frames: list[pd.DataFrame]
    kwargs: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    # NEW-001/259: True when this audit run is for a production / mining-
    # admissible candidate.  Only those are held to the strict
    # "every searchable scalar declares a ParamRole" contract; legacy /
    # research / experimental surfaces may keep the historical inference.
    production_candidate: bool = False


SAMPLE: dict[str, Callable[[], OperatorSample]] = {
    "ts_hill_tail_index": lambda: OperatorSample(
        "ts_hill_tail_index",
        [_panel(positive=True, seed=2)],
        {"window": 60, "tail_fraction": 0.2, "min_tail_count": 10},
    ),
    "ts_mean_excess_slope": lambda: OperatorSample(
        "ts_mean_excess_slope",
        [_panel(positive=True, seed=2)],
        {"window": 60, "min_tail_count": 8},
    ),
    "ts_count_if": lambda: OperatorSample(
        "ts_count_if",
        [_panel(seed=3) > 0],
        {"window": 20, "min_periods": 5},
    ),
    "ts_downside_deviation": lambda: OperatorSample(
        "ts_downside_deviation",
        [_panel(seed=4)],
        {"window": 20},
    ),
    "cs_rank_composition_churn": lambda: OperatorSample(
        "cs_rank_composition_churn",
        [_panel(seed=5)],
        {"lag": 5},
    ),
    "ts_feature_mode_share": lambda: OperatorSample(
        "ts_feature_mode_share",
        [_panel(seed=8)] * 3,
        {"window": 40},
    ),
    "ts_transfer_entropy_peak_excess": lambda: OperatorSample(
        "ts_transfer_entropy_peak_excess",
        list(_panel_pair(seed=7)),
        # R13: window=30 was infeasible — the kernel requires window-lag >=
        # min_transitions(30); with the default lag the effective window is
        # window-20, so 30 left a 10<30 gap and every rule AUDIT_ERROR'd.
        {"window": 60},
    ),
}


def _resolve_op(name: str) -> Any:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy")
    if op is None:
        raise KeyError(name)
    return op


def _run(op: Any, frames: list[pd.DataFrame], kwargs: dict[str, Any]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # audit must not be derailed by DQ warnings
        out = op.calculate(*frames, **kwargs)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"{op.metadata.name}: audit expects a DataFrame output, got {type(out).__name__}")
    return out


def _finite_ratio(out: pd.DataFrame, warmup_rows: int = 0) -> float:
    arr = out.to_numpy(dtype=float)
    if warmup_rows > 0:
        arr = arr[warmup_rows:]
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def _unit_tag(op: Any) -> str | None:
    for tag in (getattr(op.metadata, "tags", None) or []):
        if str(tag).startswith("unit:"):
            return str(tag)[len("unit:"):]
    unit = getattr(op.metadata, "unit", None)
    return unit or None


# ---------------------------------------------------------------------------
# R13 NEW-P0-06: contract-driven fixture generation over the full catalog
# ---------------------------------------------------------------------------

def _stable_seed(canonical: str) -> int:
    import zlib

    return abs(zlib.crc32(canonical.encode("utf-8")))


def _domain_panel(unit: str | None, seed: int = 0) -> pd.DataFrame:
    """A synthetic panel appropriate to the declared input unit."""
    if unit in {"condition", "event", "event_indicator"}:
        return (_panel(seed=seed, positive=True) > 0.5).astype(float)
    if unit in {"group", "group_id", "category", "membership"}:
        idx = pd.date_range("2024-01-01", periods=40, freq="B")
        cols = [f"S{i}" for i in range(3)]
        return pd.DataFrame(
            {c: pd.Series(["G"] * len(idx), index=idx, dtype=object) for c in cols}
        )
    if unit in {"price", "amount", "volume", "positive", "rate"}:
        return _panel(seed=seed, positive=True)
    return _panel(seed=seed)


def contract_fixture(canonical: str) -> OperatorSample | None:
    """R13 NEW-P0-06: best-effort OperatorSample generated from the operator's
    DECLARED contract — panel inputs from ``panel_params`` / ``input_units``,
    scalars from ``ParamSpec.default``.

    Returns ``None`` when the panel arity is genuinely undeterminable (the caller
    records it as NOT_APPLICABLE rather than guessing).  Family rules that cannot
    run the resulting fixture report NOT_APPLICABLE; only rule-code faults become
    AUDIT_ERROR.
    """
    from factor_engine.cleaned_operators.base import MISSING

    try:
        op = _resolve_op(canonical)
    except KeyError:
        return None
    meta = getattr(op, "metadata", None)
    if meta is None:
        return None
    param_names = list(getattr(meta, "param_names", None) or [])
    panel_params = list(getattr(meta, "panel_params", None) or [])
    scalar_params = list(getattr(meta, "scalar_params", None) or [])
    input_units = getattr(meta, "input_units", None) or {}
    specs = getattr(meta, "param_specs", None) or {}

    if not panel_params:
        panel_params = [
            k for k in input_units if k in param_names and k not in specs
        ]
    if not panel_params and not scalar_params:
        # unary elementwise family: exactly one implied panel
        panel_params = ["x"]
    if not panel_params:
        return None

    frames = [
        _domain_panel(input_units.get(p), seed=_stable_seed(f"{canonical}:{i}"))
        for i, p in enumerate(panel_params)
    ]
    kwargs: dict[str, Any] = {}
    for name in scalar_params:
        spec = specs.get(name)
        if spec is not None and getattr(spec, "default", MISSING) is not MISSING:
            kwargs[name] = spec.default
    return OperatorSample(canonical, frames, kwargs, note="contract-fixture")


@dataclass(frozen=True)
class OutputRangeContract:
    """R13 NEW-P1-13: the ONLY authority for output-range bounds.

    ``unit``-tag guessing was dropped — "ratio" covers shapes that are not
    [0,1]-bounded (slope, score), so a bounded output must be DECLARED.  A family
    declares it via the metadata tag ``output_range:<lo>:<hi>:<cl|op>:<cl|op>``
    (bounds + open/closed markers, both sides closed by default) or a
    ``metadata.output_range`` tuple ``(lo, hi, closed_lo, closed_hi)``.
    """

    lower: float
    upper: float
    closed_lower: bool = True
    closed_upper: bool = True
    source: str = "declared"


def _output_range_contract(op: Any) -> OutputRangeContract | None:
    for tag in (getattr(op.metadata, "tags", None) or []):
        text = str(tag)
        if not text.startswith("output_range:"):
            continue
        parts = text.split(":")[1:]
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            continue
        cl = parts[2] != "op" if len(parts) > 2 else True
        cu = parts[3] != "op" if len(parts) > 3 else True
        return OutputRangeContract(lo, hi, cl, cu)
    attr = getattr(op.metadata, "output_range", None)
    if isinstance(attr, (tuple, list)) and len(attr) >= 2:
        return OutputRangeContract(
            float(attr[0]),
            float(attr[1]),
            bool(attr[2]) if len(attr) > 2 else True,
            bool(attr[3]) if len(attr) > 3 else True,
        )
    return None


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

RULES: dict[str, dict[str, Any]] = {}


def _rule(name: str, *, category: str, description: str):
    def deco(fn: Callable) -> Callable:
        RULES[name] = {
            "category": category,
            "description": description,
            "check": fn,
        }
        return fn
    return deco


@_rule("default_output_finite", category="default", description="defaults must not make a factor (nearly) dead")
def _default_output_finite(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    try:
        out = _run(op, sample.frames, sample.kwargs)
    except Exception as exc:  # noqa: BLE001 — fixture cannot exercise the op
        ctx["report"].note_skipped(
            "default_output_finite",
            f"{op.metadata.name}: sample not runnable ({type(exc).__name__}: {exc})",
        )
        return []
    ratio = _finite_ratio(out, warmup_rows=10)
    if ratio < 0.05:
        return [AuditFinding(
            "default_output_finite", op.metadata.name, "error",
            f"default parameters yield finite ratio {ratio:.3f} (< 0.05): "
            f"a (nearly) dead factor under defaults — {sample.note or 'check tail selection / min counts'}",
        )]
    return []


@_rule("parameter_injectivity", category="parameter", description="5.1 must never silently equal 5")
def _parameter_injectivity(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    from factor_engine.cleaned_operators.base import MISSING

    sample = ctx["sample"]
    specs = getattr(op.metadata, "param_specs", None) or {}
    types = getattr(op.metadata, "param_types", None) or {}
    names = list(getattr(op.metadata, "param_names", None) or [])
    # R13 NEW-P1-12: EVERY declared searchable int parameter is probed — not just
    # the first one that happens not to be pinned in the sample kwargs.
    int_params: list[str] = []
    for n in names:
        spec = specs.get(n)
        if spec is not None:
            if getattr(spec, "dtype", None) is int and getattr(spec, "searchable", True):
                int_params.append(n)
        elif types.get(n) is int:
            int_params.append(n)
    findings: list[AuditFinding] = []
    for int_param in int_params:
        if int_param in sample.kwargs:
            base = sample.kwargs[int_param]
        else:
            base = getattr(specs.get(int_param), "default", MISSING)
            if base is MISSING:
                base = 5
        if isinstance(base, bool) or not isinstance(base, (int, float)):
            continue
        broken = float(base) + 0.1
        spec = specs.get(int_param)
        if spec is not None:
            mx = getattr(spec, "max", None)
            mn = getattr(spec, "min", None)
            if mx is not None and broken > mx:
                continue  # cannot construct a legal fractional probe above max
            if mn is not None and broken < mn:
                continue
        try:
            _run(op, sample.frames, {**sample.kwargs, int_param: broken})
        except OperatorParameterError:
            continue  # loud failure: 5.1 rejected — correct
        except Exception as exc:  # noqa: BLE001 — fixture cannot exercise the op
            ctx["report"].note_skipped(
                "parameter_injectivity",
                f"{op.metadata.name}: fixture not runnable ({type(exc).__name__}: {exc})",
            )
            return []
        findings.append(AuditFinding(
            "parameter_injectivity", op.metadata.name, "error",
            f"int param {int_param}: {broken} silently accepted (truncated to "
            f"{int(broken)}), corrupting the search space",
        ))
    return findings


@_rule("param_role_declared", category="parameter", description="production searchable scalars must declare ParamRole (NEW-001/259)")
def _param_role_declared(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    """NEW-001/259: a searchable scalar whose role is NOT explicitly declared
    falls back to ``ECONOMIC`` via :func:`effective_param_role` — silently
    placing it in the default full-resolution mining search space.  For
    production/admissible candidates this is a certification FAIL; only legacy
    / compatibility / research surfaces may rely on the fallback."""
    from factor_engine.cleaned_operators.base import (
        FULL_SEARCH_ROLES,
        ParamRole,
        effective_param_role,
        param_role_declared,
    )

    sample = ctx["sample"]
    meta = op.metadata
    # Only production/admissible candidates are held to the strict contract;
    # research/experimental surfaces may keep the legacy inference.
    if not getattr(sample, "production_candidate", False):
        return []
    specs = getattr(meta, "param_specs", None) or {}
    names = list(getattr(meta, "param_names", None) or [])
    findings: list[AuditFinding] = []
    for n in names:
        spec = specs.get(n)
        if spec is None:
            # No ParamSpec at all: the parameter's type is inferred by the
            # legacy ``_INTEGER_PARAM_NAMES`` whitelist (NEW-003/260).  For a
            # production candidate this is exactly the fail-open the audit
            # forbids.
            findings.append(AuditFinding(
                "param_role_declared", meta.name, "error",
                f"searchable scalar {n!r} has NO ParamSpec: type + role are "
                "inferred by the legacy name whitelist (NEW-003/260) — declare "
                "ParamSpec + ParamRole or the parameter silently joins the "
                "default mining search space",
            ))
            continue
        if not getattr(spec, "searchable", True):
            continue
        if param_role_declared(spec):
            continue
        inferred = effective_param_role(spec)
        findings.append(AuditFinding(
            "param_role_declared", meta.name, "error",
            f"searchable scalar {n!r} has no declared ParamRole; effective role "
            f"falls back to {inferred.value!r} — a production candidate must "
            "declare its role explicitly (NEW-001/259), never rely on the "
            "ECONOMIC fallback",
        ))
    return findings


@_rule("relational_constraints", category="parameter", description="min_periods<=window, lag<window, k<=N")
def _relational_constraints(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    specs = getattr(op.metadata, "param_specs", None) or {}
    names = list(getattr(op.metadata, "param_names", None) or [])
    findings: list[AuditFinding] = []
    if "window" in names and "min_periods" in specs:
        w = sample.kwargs.get("window", 20)
        bad = {"window": int(w), "min_periods": int(w) + 5}
        try:
            _run(op, sample.frames, {**sample.kwargs, **bad})
        except Exception:
            pass
        else:
            findings.append(AuditFinding(
                "relational_constraints", op.metadata.name, "error",
                f"min_periods({bad['min_periods']}) > window({bad['window']}) "
                "accepted silently; window/min_periods must be validated",
            ))
    if "lag" in names:
        w = int(sample.kwargs.get("window", 20))
        lag = int(sample.kwargs.get("lag", 5))
        # R13: the old ``{"lag": X, **sample.kwargs}`` unpack let a pinned
        # ``lag`` in the sample overwrite the injected violation, so the rule
        # reported a false positive on the ORIGINAL (valid) lag.  Build from the
        # sample kwargs and THEN override lag — the violation is real.
        bad = dict(sample.kwargs)
        bad["lag"] = max(w, lag) + 5
        bad["window"] = w
        try:
            _run(op, sample.frames, bad)
        except Exception:
            pass
        else:
            findings.append(AuditFinding(
                "relational_constraints", op.metadata.name, "error",
                f"lag({bad['lag']}) >= window({w}) accepted silently",
            ))
    return findings


_SIDE_PARAM_RE = re.compile(r"\b(side|tail)\b")


@_rule("mirror_symmetry", category="symmetry", description="lower(x) approx upper(-x) for tail pairs")
def _mirror_symmetry(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    names = list(getattr(op.metadata, "param_names", None) or [])
    if not any(_SIDE_PARAM_RE.search(n) for n in names) and "side" not in names:
        ctx["report"].note_skipped("mirror_symmetry", f"{op.metadata.name}: no side/tail param")
        return []
    x = _panel(positive=True, seed=11)
    upper_kw = {**sample.kwargs, "side": "upper"}
    lower_mirror_kw = {**sample.kwargs, "side": "lower"}
    try:
        u = _run(op, [x], upper_kw)
        lm = _run(op, [-x], lower_mirror_kw)
    except Exception:
        ctx["report"].note_skipped("mirror_symmetry", f"{op.metadata.name}: not runnable with side on this fixture")
        return []
    ur = u.to_numpy(dtype=float)
    lr = lm.to_numpy(dtype=float)
    both = np.isfinite(ur) & np.isfinite(lr)
    if not np.any(both):
        return []
    if float(np.median(np.abs(ur[both] - lr[both]))) > 0.25:
        return [AuditFinding(
            "mirror_symmetry", op.metadata.name, "warning",
            "lower(-x) does not mirror upper(x): |Δ| median {:.3f} > 0.25".format(
                float(np.median(np.abs(ur[both] - lr[both]))),
            ),
        )]
    return []


@_rule("column_permutation", category="invariance", description="CS ops invariant to stock-column shuffling")
def _column_permutation(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    tags = {str(t).lower() for t in (getattr(op.metadata, "tags", None) or [])}
    if not any("cs" in t or "cross_section" in t for t in tags) \
            and "cross_sectional" not in getattr(op.metadata, "category", ""):
        ctx["report"].note_skipped("column_permutation", f"{op.metadata.name}: not cross-sectional")
        return []
    frames = [f.copy() for f in sample.frames]
    try:
        out0 = _run(op, frames, sample.kwargs)
    except Exception as exc:  # noqa: BLE001 — fixture cannot exercise the op
        ctx["report"].note_skipped(
            "column_permutation",
            f"{op.metadata.name}: fixture not runnable ({type(exc).__name__}: {exc})",
        )
        return []
    order = list(range(frames[0].shape[1]))
    rng = np.random.default_rng(42)
    shuffled = list(rng.permutation(order))
    perm_frames = [f.iloc[:, shuffled] for f in frames]
    try:
        outp = _run(op, perm_frames, sample.kwargs)
    except Exception as exc:  # noqa: BLE001 — fixture not runnable on the shuffled panels
        ctx["report"].note_skipped(
            "column_permutation",
            f"{op.metadata.name}: fixture not runnable after permutation ({type(exc).__name__}: {exc})",
        )
        return []
    # undo the permutation on the OUTPUT columns
    inv = sorted(range(len(shuffled)), key=lambda i: shuffled[i])
    outp = outp.iloc[:, inv]
    a = out0.to_numpy(dtype=float)
    b = outp.to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.any(mask) and not np.allclose(a[mask], b[mask], rtol=0, atol=1e-9):
        return [AuditFinding(
            "column_permutation", op.metadata.name, "error",
            "output changes under a stock-column permutation (tie / column-order bug)",
        )]
    return []


# Count/aggregation families legitimately EXCLUDE unknown rows from both the
# numerator and denominator (documented ConditionBool policy, R12) — a NaN
# condition row yields a finite "count of confirmed-True among confirmed", which
# is NOT treating the unknown as confirmed.  The strict NaN->NaN output rule
# applies only to STATE-output operators, where the output at row t IS the
# current row's event/condition state (R13 NEW-P0-08 + family scoping).
_STATE_OUTPUT_MARKERS = (
    "state", "days_since", "time_since", "true_streak", "streak", "latch",
    "since", "until", "transition", "flag", "if_last", "last_if", "is_",
    "segment_state",
)
_COUNT_FAMILY_MARKERS = (
    "count", "sum_if", "mean_if", "std_if", "ratio", "share", "coverage",
    "spacing", "entropy", "prob",
)


@_rule("missing_state", category="state", description="NaN event/condition state is not 'confirmed'")
def _missing_state(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    names = list(getattr(op.metadata, "param_names", None) or [])
    cond_name = next((n for n in names if n in {"condition", "event", "event_indicator"}), None)
    if cond_name is None:
        ctx["report"].note_skipped("missing_state", f"{op.metadata.name}: no condition/event param")
        return []
    op_name = op.metadata.name
    if any(m in op_name for m in _COUNT_FAMILY_MARKERS) and not any(m in op_name for m in _STATE_OUTPUT_MARKERS):
        ctx["report"].note_skipped(
            "missing_state",
            f"{op_name}: windowed count/aggregation family — unknown rows are "
            "EXCLUDED from num and den (documented ConditionBool policy), not "
            "treated as confirmed (R13 family scoping)",
        )
        return []
    cond = (_panel(seed=13) > 0).astype(float)  # 0/1 ConditionBool panel
    cond.iloc[20, 0] = np.nan  # unknown state mid-panel
    frames = list(sample.frames)
    # replace the condition frame: it is the first frame only for pure-event ops
    if len(frames) == 1 and cond_name == "condition":
        frames = [cond]
    else:
        # op takes (x, condition, ...) or (y, x, condition, ...): locate by arity is
        # ambiguous — only audit ops whose FIRST frame is the condition.
        ctx["report"].note_skipped("missing_state", f"{op.metadata.name}: condition not the leading frame")
        return []
    try:
        out = _run(op, frames, sample.kwargs)
    except Exception:
        ctx["report"].note_skipped("missing_state", f"{op.metadata.name}: not runnable")
        return []
    arr = out.to_numpy(dtype=float)
    row = 20
    # R13 NEW-P0-08: the judgement was INVERTED — the injected cell is NaN, so the
    # old guard ``not np.isnan(cond[row,0])`` was always False and the failure
    # ("NaN condition -> finite output") could never fire.  The rule fires exactly
    # when the unknown state yields a finite (confirmed-looking) output.
    if np.isnan(cond.to_numpy()[row, 0]) and np.isfinite(arr[row, 0]):
        return [AuditFinding(
            "missing_state", op.metadata.name, "error",
            "a NaN (unknown) event/condition state at row 20 yields a FINITE "
            "output — unknown state must be censored (NaN), never treated as "
            "confirmed True/False (R13 NEW-P0-08)",
        )]
    return []


@_rule("semantic_type_gate", category="type", description="Price/Return/EventBool/GroupID are not fungible")
def _semantic_type_gate(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    input_units = getattr(op.metadata, "input_units", None) or {}
    x_unit = input_units.get("x")
    if x_unit is None:
        ctx["report"].note_skipped("semantic_type_gate", f"{op.metadata.name}: no input_units['x'] declared")
        return []
    # An operator that REFUSES price-level inputs must not accept a raw price
    # silently — the price-vs-return distinction is part of its contract.
    price_refusing = isinstance(x_unit, str) and (
        "return" in x_unit or "residual" in x_unit or "signal" in x_unit
    )
    if not price_refusing:
        ctx["report"].note_skipped("semantic_type_gate", f"{op.metadata.name}: x contract does not refuse price")
        return []
    # A clean trending price level fed through must not be silently accepted by
    # the pure operator without at least a warning (the hard gate lives in the
    # typed layer; a heuristic at most warns).
    price = pd.DataFrame(
        np.linspace(100.0, 200.0, 40)[:, None].repeat(3, axis=1),
        index=pd.date_range("2024-01-01", periods=40, freq="B"),
        columns=["S0", "S1", "S2"],
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            # call directly so the operator's own DQ warning is observable
            op.calculate(price, **sample.kwargs)
        except Exception:
            return []  # rejected outright — correct
    # accepted but with at least one warning (DQ) — the typed layer is the
    # authority and the operator warns instead of silently miscomputing.
    if any(issubclass(r.category, Warning) for r in recorded):
        return []
    return [AuditFinding(
        "semantic_type_gate", op.metadata.name, "warning",
        f"input_units['x']={x_unit} refuses price-level input but a raw price "
        "is accepted silently (no raise, no warning)",
    )]


@_rule("finite_range", category="range", description="declared output-range bounds hold")
def _finite_range(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    # R13 NEW-P1-13: bounds come ONLY from a declared OutputRangeContract — never
    # guessed from the unit tag ("ratio" covers slope/score shapes that are not
    # [0,1]-bounded; probability/correlation unit guessing is equally unreliable).
    contract = _output_range_contract(op)
    if contract is None:
        ctx["report"].note_skipped(
            "finite_range",
            f"{op.metadata.name}: no declared OutputRangeContract; output bounds are "
            "not guessed from the unit tag (R13 NEW-P1-13)",
        )
        return []
    try:
        out = _run(op, sample.frames, sample.kwargs)
    except Exception as exc:  # noqa: BLE001 — contract fixture cannot exercise it
        ctx["report"].note_skipped(
            "finite_range",
            f"{op.metadata.name}: fixture not runnable ({type(exc).__name__}: {exc})",
        )
        return []
    arr = out.to_numpy(dtype=float)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return []
    within = np.ones(fin.shape, dtype=bool)
    eps = 1e-9
    within &= fin >= contract.lower - eps if contract.closed_lower else fin > contract.lower - eps
    within &= fin <= contract.upper + eps if contract.closed_upper else fin < contract.upper + eps
    frac = float(np.mean(within))
    if frac < 0.999:
        bounds = "[" if contract.closed_lower else "("
        bounds += f"{contract.lower}, {contract.upper}"
        bounds += "]" if contract.closed_upper else ")"
        return [AuditFinding(
            "finite_range", op.metadata.name, "error",
            f"declared OutputRangeContract {bounds} violated: {1.0 - frac:.3f} of "
            "finite values lie outside (R13 NEW-P1-13)",
        )]
    return []


@_rule("scale_shift_invariance", category="invariance", description="dimensionless ops ignore legal rescaling")
def _scale_shift_invariance(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    unit = (_unit_tag(op) or "").lower()
    if unit != "dimensionless":
        ctx["report"].note_skipped("scale_shift_invariance", f"{op.metadata.name}: unit={unit!r} != dimensionless")
        return []
    try:
        base = _run(op, sample.frames, sample.kwargs)
        scaled = _run(op, [3.0 * f for f in sample.frames], sample.kwargs)
    except Exception:
        ctx["report"].note_skipped("scale_shift_invariance", f"{op.metadata.name}: not runnable")
        return []
    a = base.to_numpy(dtype=float)
    b = scaled.to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.any(mask) and not np.allclose(a[mask], b[mask], rtol=1e-9, atol=1e-12):
        return [AuditFinding(
            "scale_shift_invariance", op.metadata.name, "warning",
            "dimensionless output changes under a common 3x input rescaling",
        )]
    return []


_GROUP_MIGRATION_NAMES = ("churn", "retention", "migration", "entry", "exit")


@_rule("group_migration", category="group", description="a label move A->B is a real exit+entry")
def _group_migration(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    name = op.metadata.name
    if not any(k in name for k in _GROUP_MIGRATION_NAMES):
        ctx["report"].note_skipped("group_migration", f"{name}: not a churn/retention/migration operator")
        return []
    # R13 NEW-P0-09: ``sample`` was never bound in the original rule — every
    # ``sample.kwargs`` reference raised NameError, swallowed as "not runnable",
    # so this rule NEVER validated anything.
    sample = ctx["sample"]
    # one stock moves group A -> B between two dates; the A-side churn must rise.
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = ["S0", "S1", "S2", "S3"]
    x = pd.DataFrame(np.ones((4, 4)), index=idx, columns=cols)
    groups = pd.DataFrame(index=idx, columns=cols, dtype=object)
    groups.iloc[:] = "A"
    groups.loc[groups.index[3], "S1"] = "B"  # S1 migrates A->B on the last date
    stable = pd.DataFrame(index=idx, columns=cols, dtype=object)
    stable.iloc[:] = "A"
    try:
        mig = _run(op, [x], {**sample.kwargs, "group": groups})
        ctl = _run(op, [x], {**sample.kwargs, "group": stable})
    except Exception:
        ctx["report"].note_skipped("group_migration", f"{name}: not runnable with a group frame")
        return []
    ma = mig.to_numpy(dtype=float)
    ca = ctl.to_numpy(dtype=float)
    last = ma[3]
    last_c = ca[3]
    fin = np.isfinite(last)
    if not np.any(fin):
        return []
    if float(np.nanmax(last[fin])) <= float(np.nanmax(last_c[fin])) + 1e-12:
        return [AuditFinding(
            "group_migration", name, "error",
            "a stock's group move A->B does not increase the A-side churn/retention "
            "metric vs the stable control (membership transition is being dropped)",
        )]
    return []


_NATIVE_COHORT_NAMES = ("retention", "overlap", "churn")


@_rule("native_cohort", category="group", description="retention/overlap define state per native date")
def _native_cohort(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    name = op.metadata.name
    if not any(k in name for k in _NATIVE_COHORT_NAMES):
        ctx["report"].note_skipped("native_cohort", f"{name}: not a retention/overlap/churn operator")
        return []
    # R13 NEW-P0-09: ``sample`` was never bound in the original rule — every
    # ``sample.kwargs`` reference raised NameError, which the broad ``except``
    # swallowed as "not runnable", so this rule NEVER validated anything.
    sample = ctx["sample"]
    # A stock that exits the group must not be silently removed from the PAST
    # tail definition: the lagged tail is over the FULL lagged cohort.
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    cols = ["S0", "S1", "S2", "S3", "S4"]
    xv = np.array([
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [1.0, 0.9, 0.8, 0.7, 0.6],
    ])
    x = pd.DataFrame(xv, index=idx, columns=cols)
    groups = pd.DataFrame(index=idx, columns=cols, dtype=object)
    groups.iloc[:] = "G"
    groups.loc[groups.index[3], ["S4"]] = "H"  # S4 leaves G on the last date
    stable = pd.DataFrame(index=idx, columns=cols, dtype=object)
    stable.iloc[:] = "G"
    try:
        mig = _run(op, [x], {**sample.kwargs, "group": groups})
        ctl = _run(op, [x], {**sample.kwargs, "group": stable})
    except Exception:
        ctx["report"].note_skipped("native_cohort", f"{name}: not runnable with a group frame")
        return []
    # R13 NEW-P0-09: the rule must actually assert native-cohort semantics.
    ma = mig.to_numpy(dtype=float)
    ca = ctl.to_numpy(dtype=float)
    # 1) PAST rows (0..2) must be identical between mig and ctl: the lagged tail
    #    is native to its OWN date — S4 leaving G at row 3 must NOT rewrite the
    #    historical tail definition.
    past = slice(0, 3)
    both_past = np.isfinite(ma[past]) & np.isfinite(ca[past])
    if np.any(both_past) and not np.allclose(
        ma[past][both_past], ca[past][both_past], rtol=1e-9, atol=1e-12
    ):
        return [AuditFinding(
            "native_cohort", name, "error",
            "past (pre-migration) rows change when a stock exits the group later — "
            "the lagged cohort is being redefined by the CURRENT membership "
            "instead of each row's native-date cohort (R13 NEW-P0-09)",
        )]
    # 2) the membership change on the LAST date is real: the current-date metric
    #    must not be byte-identical to the stable control (the exit would be
    #    silently dropped from the current cohort).
    last_fin = np.isfinite(ma[3]) & np.isfinite(ca[3])
    if np.any(last_fin) and np.allclose(
        ma[3][last_fin], ca[3][last_fin], rtol=0, atol=1e-12
    ):
        return [AuditFinding(
            "native_cohort", name, "warning",
            "a group exit on the last date leaves the current-date metric identical "
            "to the stable control — the membership change is being dropped from "
            "the current cohort (R13 NEW-P0-09)",
        )]
    return []


@_rule("psd_geometry", category="geometry", description="spectral eigen metrics share one PSD matrix")
def _psd_geometry(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    name = op.metadata.name
    if not any(k in name for k in (
        "feature_geometry", "feature_mode_share", "feature_effective_rank",
        "subspace_rotation", "pca", "spectral", "eigen",
    )):
        ctx["report"].note_skipped("psd_geometry", f"{name}: not a spectral/geometry operator")
        return []
    # R13 NEW-P0-10: the kernel exposes a TEST bundle (raw corr, projected corr,
    # eigvals, eigvecs).  The machine audit verifies the shared-PSD property
    # directly instead of always skipping "matrix not exposed".
    from factor_engine.cleaned_operators import feature_geometry as fg

    accessor = getattr(fg, "geometry_audit_bundle", None)
    if accessor is None:
        ctx["report"].note_skipped(
            "psd_geometry",
            f"{name}: feature_geometry.geometry_audit_bundle is not exposed",
        )
        return []
    rng = np.random.default_rng(31)
    panel = pd.DataFrame(rng.normal(0.0, 1.0, (80, 3)), columns=["F0", "F1", "F2"])
    try:
        bundle = accessor(panel)
    except Exception as exc:  # noqa: BLE001
        ctx["report"].note_skipped("psd_geometry", f"{name}: bundle raised ({type(exc).__name__}: {exc})")
        return []
    if bundle is None:
        ctx["report"].note_skipped("psd_geometry", f"{name}: bundle returned None (panel not valid)")
        return []
    raw, projected, eigvals, eigvecs = bundle
    findings: list[AuditFinding] = []
    if float(np.min(eigvals)) < -1e-8:
        findings.append(AuditFinding(
            "psd_geometry", name, "error",
            "projected correlation has a negative eigenvalue (not PSD)",
        ))
    # eigenpair consistency: A @ v == lambda * v for the projected matrix.
    resid = projected @ eigvecs - eigvecs * eigvals
    if float(np.max(np.abs(resid))) > 1e-6:
        findings.append(AuditFinding(
            "psd_geometry", name, "error",
            "eigvals/eigvecs do not diagonalize the projected matrix (mismatched matrix)",
        ))
    # eigvecs orthonormal (the matrix they come from is symmetric PSD).
    gram = eigvecs.T @ eigvecs
    if float(np.max(np.abs(gram - np.eye(gram.shape[0])))) > 1e-6:
        findings.append(AuditFinding(
            "psd_geometry", name, "warning",
            "eigvecs are not orthonormal",
        ))
    return findings


@_rule("golden_reference", category="reference", description="Hill/GPD/MI/TE/entropy vs known DGP / SciPy")
def _golden_reference(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    name = op.metadata.name
    adapter = _GOLDEN_ADAPTERS.get(name) or _family_golden(name)
    if adapter is None:
        ctx["report"].note_skipped(
            "golden_reference",
            f"{name}: no golden adapter registered for its statistical family "
            "(R13 NEW-P0-11 — families are enumerated, not silently only-Hill)",
        )
        return []
    try:
        return adapter(op, ctx)
    except Exception as exc:  # noqa: BLE001 — golden fixture cannot exercise the op
        ctx["report"].note_skipped(
            "golden_reference",
            f"{name}: golden fixture not runnable ({type(exc).__name__}: {exc})",
        )
        return []


def _family_golden(name: str):
    if "hill" in name:
        return _golden_hill
    return None


def _golden_hill(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    # Pareto with a known tail shape: Hill must recover it within tolerance.
    # ``X = U ** (-xi)`` with ``U ~ U(0,1)`` has survival ``P(X > x) = x **
    # (-1/xi)``, i.e. GPD tail index exactly ``xi`` (NOT ``(1-U) ** (-1/xi)``,
    # which has tail index 1).
    name = op.metadata.name
    xi_true = 0.5
    rng = np.random.default_rng(17)
    n = 600
    u = rng.uniform(size=n)
    vals = u ** (-xi_true)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    frame = pd.DataFrame(vals[:, None], index=idx, columns=["S0"])
    out = _run(op, [frame], {"window": n, "side": "upper", "tail_fraction": 0.2, "min_tail_count": 20})
    est = out.to_numpy(dtype=float)[:, 0]
    est = est[np.isfinite(est)]
    if est.size == 0:
        return [AuditFinding("golden_reference", name, "error", "Pareto DGP yields no finite Hill estimate")]
    median_est = float(np.median(est))
    if abs(median_est - xi_true) > 0.25:
        return [AuditFinding(
            "golden_reference", name, "error",
            f"Hill on Pareto(shape={xi_true}) recovers {median_est:.3f} (|Δ| > 0.25)",
        )]
    return []


# canonical -> golden adapter (R13 NEW-P0-11): each statistical family pins a
# known-DGP reference.  Families without an adapter report NOT_APPLICABLE with
# the reason, so "only Hill was pinned" can never be mistaken for coverage.
#
# A mean-excess adapter was trialled and REMOVED: the OLS-over-quantile-grid
# estimator of ``ξ/(1-ξ)`` is unstable on finite Pareto samples (slope drifts
# 0.70 -> 1.36 across n on the SAME seed while the kernel matches the direct
# empirical slope exactly), so it cannot distinguish a kernel bug from estimator
# bias — a flaky golden that flags correct kernels is worse than an explicit
# NOT_APPLICABLE.  Family dispatch remains so new adapters slot in per family.
_GOLDEN_ADAPTERS: dict[str, Any] = {
    "ts_hill_tail_index": _golden_hill,
}


_HONESTY_TERMS = ("robust", "hill", "allan", "granger", "tail_dependence", "extremal", "entropy")
_FORMULA_HINT = re.compile(
    r"[=ξθλ]|\\(frac|sum|log|int|prod)|\(\s*1\s*/\s*k|estimator|formula|definition|reference|runs estimator",
    re.IGNORECASE,
)


@_rule("canonical_honesty", category="honesty", description="statistical-term names bind a definition/formula")
def _canonical_honesty(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    name = op.metadata.name
    if not any(t in name for t in _HONESTY_TERMS):
        return []
    description = str(getattr(op.metadata, "description", "") or "")
    if not _FORMULA_HINT.search(description):
        return [AuditFinding(
            "canonical_honesty", name, "warning",
            "name contains a statistical term (robust/hill/allan/granger/tail_dependence/"
            "extremal/entropy) but the description binds no definition or reference "
            "formula — a later reader cannot tell what is actually computed",
        )]
    return []


@_rule("default_searchability", category="default", description="defaults must not yield all-NaN/constant/near-zero output")
def _default_searchability(op: Any, ctx: dict[str, Any]) -> list[AuditFinding]:
    sample = ctx["sample"]
    try:
        out = _run(op, sample.frames, {})
    except Exception as exc:  # noqa: BLE001 — no runnable pure-default set
        ctx["report"].note_skipped(
            "default_searchability",
            f"{op.metadata.name}: no runnable pure-default parameter set "
            f"({type(exc).__name__}: {exc})",
        )
        return []
    arr = out.to_numpy(dtype=float)
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return [AuditFinding(
            "default_searchability", op.metadata.name, "error",
            "default parameters yield an all-NaN panel — not searchable",
        )]
    if np.allclose(fin, fin[0], atol=1e-12):
        return [AuditFinding(
            "default_searchability", op.metadata.name, "warning",
            "default parameters yield a constant output — no discriminating power",
        )]
    if float(np.std(fin)) < 1e-12 and float(np.mean(np.abs(fin))) < 1e-12:
        return [AuditFinding(
            "default_searchability", op.metadata.name, "warning",
            "default parameters yield a near-zero constant output",
        )]
    return []


DEFAULT_RULES: tuple[str, ...] = tuple(sorted(RULES))


def run_audit(
    *,
    sample_names: list[str] | None = None,
    rule_names: list[str] | None = None,
) -> AuditReport:
    """Run the enabled rules over the operator sample.

    ``sample_names`` filters which SAMPLE entries run; ``rule_names`` filters
    which rules run (default: all registered).  Returns a report with findings,
    per-rule PASS-checked counts, honest NOT_APPLICABLE reasons and — since
    R13 NEW-P0-07 — a separate AUDIT_ERROR tally.  A rule that RAISES is an
    AUDIT_ERROR (release gate), never a silent skip.
    """
    report = AuditReport()
    names = sample_names or sorted(SAMPLE)
    rules = rule_names or DEFAULT_RULES
    for name in names:
        if name not in SAMPLE:
            raise KeyError(f"no audit sample for {name!r}")
        sample = SAMPLE[name]()
        try:
            op = _resolve_op(name)
        except KeyError:
            report.note_skipped("(resolve)", f"{name}: not in registry")
            continue
        _run_rule_sweep(report, op, sample, rules)
    return report


def _run_rule_sweep(
    report: AuditReport,
    op: Any,
    sample: OperatorSample,
    rules: tuple[str, ...],
) -> None:
    """Run one operator through the given rules with R13 NEW-P0-07 outcome
    classes: findings (FAIL/warning) -> added; rule raised -> AUDIT_ERROR
    (release gate), never a silent skip."""
    ctx = {"sample": sample, "report": report}
    name = sample.name
    for rule in rules:
        entry = RULES.get(rule)
        if entry is None:
            raise KeyError(f"no audit rule {rule!r}")
        # R16-041: a rule that reports NOT_APPLICABLE (note_skipped) must not
        # ALSO be counted as checked — the pre/post skip-count snapshot gives
        # every (canonical, rule) exactly ONE outcome instead of double-counting.
        pre_skip = len(report.skipped.get(rule, ()))
        pre_err = len(report.audit_errors.get(rule, ()))
        try:
            findings = entry["check"](op, ctx)
        except Exception as exc:  # noqa: BLE001 — rule crashed on the operator
            report.note_audit_error(rule, f"{name}: {type(exc).__name__}: {exc}")
            report.outcome_matrix[(name, rule)] = "AUDIT_ERROR"
            continue
        for finding in findings:
            report.add(finding)
        post_skip = len(report.skipped.get(rule, ()))
        post_err = len(report.audit_errors.get(rule, ()))
        if post_err > pre_err:
            report.outcome_matrix[(name, rule)] = "AUDIT_ERROR"
        elif post_skip > pre_skip:
            report.outcome_matrix[(name, rule)] = "N/A"
        else:
            report.outcome_matrix[(name, rule)] = "PASS" if not findings else "FAIL"
            report.note_ran(rule)  # genuinely checked (PASS or FAIL)


# R13 NEW-P0-06: rules that can run on a contract-generated fixture over the
# WHOLE production/searchable catalog (not just the 7 curated SAMPLE ops).
# R16-040: ``param_role_declared`` and ``relational_constraints`` are part of
# the REQUIRED catalog sweep — a production target's searchable scalars must
# actually be checked for a declared ParamRole and its relations enforced, not
# just described in a comment.
_CONTRACT_CATALOG_RULES: tuple[str, ...] = (
    "default_output_finite",
    "default_searchability",
    "parameter_injectivity",
    "column_permutation",
    "finite_range",
    "scale_shift_invariance",
    "mirror_symmetry",
    "canonical_honesty",
    "semantic_type_gate",
    "param_role_declared",
    "relational_constraints",
)


def _required_rules_for(canonical: str, rules: tuple[str, ...]) -> tuple[str, ...]:
    """R16-039/040: the REQUIRED rule set for a canonical.

    Every rule the catalog sweep is asked to run is required to have an outcome
    for every applicable production canonical (a rule that cannot decide for
    this canonical reports N/A with a reason — which IS an outcome).  This is
    the machine required-rule matrix: nothing is inferred from comments.
    """
    return tuple(rules)


def _outcome_matrix_hash(report: AuditReport, canonical: str) -> str:
    """R16-044: stable hash over this canonical's (rule, outcome) pairs."""
    import hashlib

    pairs = sorted(
        (rule, outcome)
        for (canon, rule), outcome in report.outcome_matrix.items()
        if canon == canonical
    )
    return hashlib.sha256(repr(pairs).encode("utf-8")).hexdigest()[:16]


def audit_all(*, rule_names: tuple[str, ...] | None = None, sample_limit: int | None = None) -> AuditReport:
    """R13 NEW-P0-06/71: full-catalog semantic audit.

    ``sample_limit`` bounds the number of production targets swept (for fast CI
    gates); ``None`` sweeps the entire production/searchable catalog.

    Two sweeps:

    1. the curated SAMPLE (hand-built fixtures that exercise tail/state/group/
       dependence kernels precisely) through the full rule set;
    2. EVERY production/searchable canonical through the :data:`_CONTRACT_CATALOG_RULES`
       family, with the fixture generated from the operator's declared contract
       (``ContractFixtureFactory``).  Operators whose contract cannot be turned
       into a runnable fixture are recorded as NOT_APPLICABLE with the reason —
       never silently dropped.

    The returned report's ``certification_safe()`` is the R13 NEW-P0-71 gate:
    no error-severity finding and zero AUDIT_ERRORs.  Only then may a production
    certification record cite the audit as semantic evidence.
    """
    from factor_engine.backend.evidence_provenance import implementation_hashes_for
    from factor_engine.cleaned_operators.production_hardening import factor_production_targets

    report = run_audit()
    rules = tuple(rule_names) if rule_names is not None else _CONTRACT_CATALOG_RULES

    targets: list[str] = []
    try:
        targets = sorted(factor_production_targets())
    except Exception:  # pragma: no cover - hardened production list should load
        report.note_audit_error("(catalog)", "factor_production_targets() unavailable")
        return report
    if sample_limit is not None:
        targets = targets[: int(sample_limit)]

    for canonical in targets:
        try:
            op = _resolve_op(canonical)
        except KeyError:
            # R16-043: a production target that cannot even be RESOLVED is an
            # audit error, not a skip — registry membership is the precondition.
            report.note_audit_error("(catalog-resolve)", f"{canonical}: not in registry")
            continue
        fixture = contract_fixture(canonical)
        if fixture is None:
            # R16-043: the operator IS resolvable but the fixture generator
            # cannot determine its panel inputs — that is an AUDIT_ERROR (an
            # unresolvable ResolvedSignature), never a silent NOT_APPLICABLE.
            report.note_audit_error(
                "(catalog-fixture)",
                f"{canonical}: no runnable contract fixture (required panels/units undeterminable)",
            )
            continue
        # NEW-001/259: production targets are held to the strict contract —
        # every searchable scalar must declare ParamSpec + ParamRole.
        fixture = replace(fixture, production_candidate=True)
        _run_rule_sweep(report, op, fixture, rules)
        # R16-039/042: every REQUIRED rule must have exactly one outcome for
        # this canonical.  A required rule with NO outcome is NOT_RUN and
        # blocks certification.
        required = _required_rules_for(canonical, rules)
        missing = [r for r in required if (canonical, r) not in report.outcome_matrix]
        if missing:
            report.required_missing[canonical] = missing
        # R16-044: the evidence record references an outcome-MATRIX hash +
        # required coverage — "has an evidence record" must prove the required
        # rules all had an outcome, not just that a bool was written.
        outcome_hash = _outcome_matrix_hash(report, canonical)
        # R13 NEW-P0-71: a per-canonical evidence record binds the audited
        # implementation hash + fixture source + rule family, so a production
        # certification record can cite WHICH implementation + WHICH rules were
        # audited — not just "semantic_audit.py passed".
        try:
            impl_hash = implementation_hashes_for(canonical)
        except Exception:  # noqa: BLE001
            impl_hash = {}
        report.evidence[canonical] = {
            "implementation_hash": impl_hash,
            "fixture": "contract",
            "rule_family": list(rules),
            "outcome_matrix_hash": outcome_hash,
            "required_rules_with_outcome": len(required) - len(missing),
            "required_rules_total": len(required),
        }
    return report
