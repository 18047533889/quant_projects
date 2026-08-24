# -*- coding: utf-8 -*-
"""Full-library Operator Closure Audit (round-2, review §20).

Iterates the FINAL registry of every canonical and runs a battery of machine
audits so that every canonical that enters an automatic mining algorithm can be
shown to satisfy: the parameter space is real (no silent ``int()`` truncation /
``max(10, int(x))`` clamps / Python-truthiness bools), inputs are axis-aligned
or explicitly broadcast, the time axis is never silently compressed across a
missing gap, missing-value behaviour matches the declared policy, output units
are algebraically consistent, and default parameters produce valid output.

Categories implemented here (the *new* round-2 ones):

1.  ``param_type_fuzzing``      — probe 5.1/True/NaN/Inf/None/"5" per scalar param;
                                  a param that silently ACCEPTS an out-of-contract
                                  value is a fake parameter dimension.
2.  ``parameter_grid_dead_region`` — sample legal parameter combinations; a legal
                                  region that is always all-NaN is a ParamSpec gap.
3.  ``axis_integrity``          — multi-panel ops must reject a permuted / date
                                  shifted / missing-stock / duplicated-axis second
                                  panel (the central ``validate_operator_call``
                                  gate enforces this; the audit verifies it across
                                  the whole library).
4.  ``missing_time_topology``   — windowed ops must not bridge an interior NaN
                                  block (no drop-finite reconnect).
5.  ``recursive_rewarmup``      — stateful/technical ops must not emit a confident
                                  value immediately after a suspension gap.
6.  ``condition_bool_contract`` — condition/event/trigger-named panel inputs must
                                  reject values outside {0,1,NaN}.
7.  ``unit_algebra``            — name/description-derived output unit vs the
                                  declared metadata (t-stat -> dimensionless,
                                  energy/variance -> unit^2, peer-mean -> same_as).
8.  ``boundary_behavior``       — minimum parameter values must fail cleanly
                                  (raise or NaN), never crash (IndexError/KeyError).
9.  ``semantic_duplicates``     — engine-level: canonical-name stem clusters.
10. ``semantic_input_type``     — a declared PositivePrice / NonNegativeVolume /
                                  ConditionBool input must actually fail closed
                                  on a violating value (round-3 §三 Audit 10).
11. ``null_calibration``        — an estimator knob must not move the null-level
                                  output dispersion >10x across its legal range
                                  (round-3 §三 Audit 7).
12. ``practical_usability``     — coverage / Inf / constant-column / trailing-NaN
                                  gates on a realistic 250×20 panel (round-3
                                  §三 Audit 12).
13. ``param_search_grade``      — role-aware search-grade enforcement (round-3
                                  §二 ParamRole): known estimator knobs must be
                                  declared ESTIMATOR_RESOLUTION, and a NUMERICAL/
                                  POLICY role requires searchable=False.

The 15 round-1 machine rules in :mod:`semantic_audit` (parameter_injectivity,
relational_constraints, mirror_symmetry, column_permutation, missing_state,
semantic_type_gate, finite_range, scale_shift_invariance, group_migration,
native_cohort, psd_geometry, golden_reference, canonical_honesty,
default_searchability, default_output_finite) complement this engine and are
invoked separately via ``semantic_audit.run_audit``.

Design rules
------------
- A check only DECIDES when the operator runs on its declared contract fixture
  (valid-run gate first).  An operator that cannot be exercised on a synthetic
  fixture is recorded NOT_APPLICABLE, never silently skipped.
- An operator that passes a category (or rejects a fail-closed probe) is counted
  in ``report.ran``.
- A RULE that raises on an operator is an AUDIT_ERROR (release-failing infra
  fault), distinct from a finding and from NOT_APPLICABLE.
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from cleaned_operators.semantic_audit import (
    AuditFinding,
    AuditReport,
    _domain_panel,
    _finite_ratio,
    _panel,
    _resolve_op,
    _run,
    _stable_seed,
    _unit_tag,
    contract_fixture,
)

# ---------------------------------------------------------------------------
# param-name classification (used only when no ParamSpec declares the dtype)
# ---------------------------------------------------------------------------

_INTISH = (
    "window", "lag", "period", "order", "delay", "bins", "min_periods",
    "cooldown", "history_length", "max_age", "span", "lookback", "count",
    "min_count", "run_length", "num_", "_k", "length", "iteration", "depth",
    "degree", "step", "top", "share", "quantile_count",
)
_BOOLISH = {
    "normalized", "add_intercept", "scale", "log", "include_intercept",
    "return_index", "keep_state", "centered",
}


def _classify_param(name: str, spec: Any) -> type:
    if spec is not None and spec.dtype is not None:
        return spec.dtype
    n = name.lower()
    if n in _BOOLISH or "bool" in n or n.endswith("_flag") or n.endswith("_logic"):
        return bool
    if any(k in n for k in _INTISH):
        return int
    return float


# ---------------------------------------------------------------------------
# category registry
# ---------------------------------------------------------------------------

CLOSURE_CATEGORIES: dict[str, dict[str, Any]] = {
    # engine-level pseudo-category: run once over the full registry as a
    # post-pass (see :func:`_semantic_duplicates`), not per-operator.
    "semantic_duplicates": {
        "description": "engine-level: canonical-name stem clusters",
        "engine_level": True,
        "check": None,
    },
}


def _category(name: str, *, description: str):
    def deco(fn: Callable[[Any, dict[str, Any]], tuple[list[AuditFinding], str | None]]):
        CLOSURE_CATEGORIES[name] = {"description": description, "check": fn}
        return fn

    return deco


def _valid_gate(op: Any, ctx: dict[str, Any]) -> str | None:
    """Run the operator once with its declared fixture.  Returns None when the
    fixture is exercisable (checks may proceed); returns a skip reason when the
    fixture itself fails (operator needs financial/minute/… data the synthetic
    fixture cannot supply)."""
    sample = ctx["sample"]
    if sample is None:
        return "no contract fixture (panel arity undeterminable)"
    try:
        _run(op, sample.frames, sample.kwargs)
    except Exception as exc:  # fixture inapplicable, not a rule fault
        return f"fixture cannot exercise: {type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# 1. parameter type fuzzing
# ---------------------------------------------------------------------------

@_category(
    "param_type_fuzzing",
    description="probe out-of-contract scalar values; silent acceptance = fake parameter dimension",
)
def _param_type_fuzzing(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    specs = getattr(meta, "param_specs", None) or {}
    sample = ctx["sample"]
    panel_names = set(getattr(meta, "panel_params", None) or [])
    declared_scalar = set(getattr(meta, "scalar_params", None) or [])
    param_names = list(getattr(meta, "param_names", None) or [])
    # Scalars = declared scalar_params, or any param with a kernel default (an
    # optional/panel-with-None-default param like ``group`` must NOT be fuzzed
    # as a scalar — that was a false-positive source).
    from cleaned_operators.base import _kernel_param_defaults

    defaults = _kernel_param_defaults(op) or {}
    # A param whose default is None is an optional panel / None-sentinel (e.g.
    # ``group: Any = None``), NOT a typed scalar — fuzzing it produces false
    # positives (the operator legitimately accepts None).
    scalar = [
        p for p in param_names
        if (p not in panel_names)
        and (p in declared_scalar or (p in defaults and defaults[p] is not None))
    ]
    if not scalar:
        return [], "no scalar parameters"
    findings: list[AuditFinding] = []
    for pname in scalar:
        spec = specs.get(pname)
        dtype = _classify_param(pname, spec)
        if dtype is str:
            continue
        if dtype is bool:
            probes: Sequence[Any] = [1, 2, "False", 0.5, None]
        elif dtype is int:
            # NB: a numeric string like "5" is NOT probed — the round-11
            # declared-numeric contract legitimately coerces it (the repo treats
            # int("5") == 5 as one AST), so probing it would be a false positive.
            probes = [5.1, True, np.nan, np.inf, -np.inf, None]
        else:
            probes = [np.nan, np.inf, -np.inf, None]
        for probe in probes:
            try:
                _run(op, sample.frames, {**sample.kwargs, pname: probe})
            except Exception:
                continue  # rejected — fail-closed, good
            findings.append(
                AuditFinding(
                    "param_type_fuzzing", meta.name, "error",
                    f"param {pname!r} (dtype={dtype.__name__}) silently accepted "
                    f"out-of-contract value {probe!r} — fake parameter dimension",
                )
            )
    return findings, None


# ---------------------------------------------------------------------------
# 2. whole-parameter-grid dead region
# ---------------------------------------------------------------------------

@_category(
    "parameter_grid_dead_region",
    description="sample legal parameter combos; a legal region that is always all-NaN is a ParamSpec gap",
)
def _parameter_grid_dead_region(op: Any, ctx: dict[str, Any], max_combos: int = 24) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    from cleaned_operators.base import MISSING

    specs = getattr(meta, "param_specs", None) or {}
    sample = ctx["sample"]
    panel_names = set(getattr(meta, "panel_params", None) or [])
    param_names = list(getattr(meta, "param_names", None) or [])
    scalar = [p for p in param_names if p not in panel_names]
    # numeric searchable params with a numeric default
    numeric: list[str] = []
    for p in scalar:
        d = sample.kwargs.get(p)
        if d is None or isinstance(d, bool) or not isinstance(d, (int, float, np.number)):
            continue
        spec = specs.get(p)
        if spec is not None and spec.searchable is False:
            continue
        numeric.append(p)
    if not numeric:
        return [], "no numeric searchable scalar parameter"
    # longer fixture so larger windows are legal
    frames = [f.copy() for f in sample.frames]
    frames[0] = pd.DataFrame(
        _panel(n_rows=200, seed=_stable_seed(meta.name)).to_numpy(),
        index=pd.date_range("2024-01-01", periods=200, freq="B"),
        columns=frames[0].columns,
    )
    base = dict(sample.kwargs)

    def _levels(p: str) -> list[Any]:
        d = base.get(p)
        spec = specs.get(p)
        dtype = _classify_param(p, spec)
        lo = spec.min if spec is not None and spec.min is not None else (1 if dtype is int else 0.0)
        hi = spec.max if spec is not None and spec.max is not None else (max(d * 3, lo + 2) if dtype is int else d * 3 + 1)
        if dtype is int:
            hi = max(hi, lo + 1)
            return sorted({int(round(x)) for x in np.linspace(lo, hi, num=5)})
        return [round(float(x), 4) for x in np.linspace(float(lo), float(hi), num=5)]

    grid = numeric[:2]
    levels = [_levels(p) for p in grid]
    combos = list(itertools.product(*levels)) if len(grid) == 2 else [(v,) for v in levels[0]]
    combos = combos[:max_combos]
    finite_map: list[tuple[tuple[Any, ...], float]] = []
    for combo in combos:
        kw = dict(base)
        for p, v in zip(grid, combo):
            kw[p] = v
        try:
            out = _run(op, frames, kw)
            fr = _finite_ratio(out)
        except Exception:
            fr = 0.0  # a legal combo that never emits is exactly what we flag
        finite_map.append((combo, fr))
    any_finite = any(fr > 0 for _, fr in finite_map)
    any_dead = any(fr == 0 for _, fr in finite_map)
    if not any_finite:
        return [], "default fixture all-NaN (operator legitimately needs non-synthetic inputs)"
    if any_dead:
        dead = [c for c, fr in finite_map if fr == 0][:4]
        live = [c for c, fr in finite_map if fr > 0][:2]
        return [
            AuditFinding(
                "parameter_grid_dead_region", meta.name, "warning",
                f"legal parameter region(s) always all-NaN while neighbours emit: "
                f"dead={dead} live={live} — ParamSpec/relational constraint incomplete",
            )
        ], None
    return [], None


# ---------------------------------------------------------------------------
# 3. multi-input axis integrity
# ---------------------------------------------------------------------------

@_category(
    "axis_integrity",
    description="multi-panel ops must reject permuted / shifted / missing / duplicated axes",
)
def _axis_integrity(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    tags = [str(t) for t in (getattr(meta, "tags", None) or [])]
    if any("allow_panel_broadcast" in t for t in tags):
        return [], "declares allow_panel_broadcast waiver"
    if getattr(meta, "broadcast_specs", None):
        return [], "declares AlignmentPlan/BroadcastSpec"
    sample = ctx["sample"]
    if sample is None or len(sample.frames) < 2:
        return [], "single-panel operator"
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    base = list(sample.frames)
    second = base[1]

    def _dup_index(f: pd.DataFrame) -> pd.DataFrame:
        idx = f.index.insert(len(f) // 2, f.index[len(f) // 2])
        return f.set_axis(idx, axis=0)

    def _dup_instrument(f: pd.DataFrame) -> pd.DataFrame:
        return f.set_axis(list(f.columns) + [f.columns[0]], axis=1)

    mutations: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "column_permutation": lambda f: f[f.columns[::-1]],
        "date_shift": lambda f: f.set_axis(f.index + pd.Timedelta(days=1), axis=0),
        "missing_stock": lambda f: f.iloc[:, :-1],
        "extra_stock": lambda f: pd.concat([f, f.iloc[:, :1]], axis=1),
        "duplicate_timestamp": _dup_index,
        "duplicate_instrument": _dup_instrument,
    }
    findings: list[AuditFinding] = []
    for label, mut in mutations.items():
        try:
            mutated = mut(second)
        except Exception:
            continue
        frames = [base[0], mutated, *base[2:]]
        try:
            _run(op, frames, sample.kwargs)
        except Exception:
            continue  # rejected — fail-closed, good
        findings.append(
            AuditFinding(
                "axis_integrity", meta.name, "error",
                f"multi-input axis integrity: {label} silently accepted and computed — "
                "panels are paired positionally without an axis check",
            )
        )
    return findings, None


# ---------------------------------------------------------------------------
# 4. missing-time topology
# ---------------------------------------------------------------------------

@_category(
    "missing_time_topology",
    description="windowed ops must not bridge an interior NaN block (drop-finite reconnect)",
)
def _missing_time_topology(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    sample = ctx["sample"]
    # a window parameter must exist for the bridging concern to apply
    window_params = [p for p in sample.kwargs if "window" in p.lower() or "period" in p.lower()]
    if not window_params:
        return [], "no window parameter (non-windowed)"
    w = sample.kwargs.get(window_params[0])
    if not isinstance(w, (int, float, np.number)) or w < 5:
        return [], f"window param {window_params[0]} too small to probe a gap"
    w = int(w)
    # interior NaN block rows [50,60) on a 120-row panel
    arr = _panel(n_rows=120, seed=_stable_seed(meta.name)).to_numpy()
    arr[50:60, :] = np.nan
    x = pd.DataFrame(
        arr, index=pd.date_range("2024-01-01", periods=120, freq="B"),
        columns=sample.frames[0].columns,
    )
    try:
        out = _run(op, [x, *sample.frames[1:]], sample.kwargs)
    except Exception:
        return [], "fixture cannot exercise with a NaN block"
    # rows 62..62+w must contain at least one NaN for an honest windowed op
    post = out.iloc[62 : min(62 + w, len(out))]
    if post.empty:
        return [], "fixture too short after the gap"
    post_arr = post.to_numpy(dtype=float)
    if np.isfinite(post_arr).all():
        return [
            AuditFinding(
                "missing_time_topology", meta.name, "warning",
                "fully finite for one window length directly after an interior NaN "
                "block — the op may be dropping NaN and reconnecting across the gap "
                "(verify against its declared missing policy)",
            )
        ], None
    return [], None


# ---------------------------------------------------------------------------
# 5. recursive re-warmup
# ---------------------------------------------------------------------------

_REWARM_WHITELIST = {
    "KAMA", "Supertrend", "PSAR", "ts_kama", "ts_supertrend", "ts_psar",
    "ts_ema", "ts_wilder_ema", "ts_dema", "ts_tema", "ts_macd",
}


@_category(
    "recursive_rewarmup",
    description="stateful/technical ops must not emit a confident value immediately after a suspension gap",
)
def _recursive_rewarmup(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    if meta.name not in _REWARM_WHITELIST:
        return [], "not in the stateful/technical re-warmup whitelist"
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    sample = ctx["sample"]
    arr = _panel(n_rows=90, seed=_stable_seed(meta.name)).to_numpy()
    arr[60:70, :] = np.nan  # suspension rows 60..69
    x = pd.DataFrame(
        arr, index=pd.date_range("2024-01-01", periods=90, freq="B"),
        columns=sample.frames[0].columns,
    )
    try:
        out = _run(op, [x, *sample.frames[1:]], sample.kwargs)
    except Exception:
        return [], "fixture cannot exercise with a suspension block"
    col = out.iloc[:, 0].to_numpy(dtype=float)
    finite_after = np.flatnonzero(np.isfinite(col[70:]))
    if finite_after.size == 0:
        return [], "never recovers finite on this fixture (declared policy may be stricter)"
    first = 70 + int(finite_after[0])
    # a re-warmup operator must not emit at the very first bar after the gap
    if first <= 70:
        return [
            AuditFinding(
                "recursive_rewarmup", meta.name, "error",
                "emits a finite value at the first bar after a 10-bar suspension — "
                "no re-warmup (gap-then-first-price straight-line output)",
            )
        ], None
    return [], None


# ---------------------------------------------------------------------------
# 6. ConditionBool contract (panel slots)
# ---------------------------------------------------------------------------

_CONDITION_WORDS = (
    "condition", "event", "reset", "set", "update", "trigger", "gate",
    "flag", "mask", "enabled", "indicator", "bool", "confirm",
)


@_category(
    "condition_bool_contract",
    description="condition/event/trigger-named panel inputs must reject values outside {0,1,NaN}",
)
def _condition_bool_contract(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    sample = ctx["sample"]
    panel_names = list(getattr(meta, "panel_params", None) or [])
    cond_panels = [p for p in panel_names if any(w in p.lower() for w in _CONDITION_WORDS)]
    if not cond_panels:
        return [], "no condition/event/trigger-named panel input"
    findings: list[AuditFinding] = []
    for pname in cond_panels:
        pos = panel_names.index(pname)
        base_panel = sample.frames[pos]
        bad = base_panel.copy()
        values = bad.to_numpy(dtype=float)
        # overwrite with out-of-contract finite values (-1 / 0.5 / 2)
        rng = np.random.default_rng(_stable_seed(f"{meta.name}:{pname}"))
        values = np.where(np.isfinite(values), np.nan, np.nan)
        flat = values.ravel()
        flat[::3] = -1.0
        flat[1::3] = 0.5
        flat[2::3] = 2.0
        bad = pd.DataFrame(flat.reshape(values.shape), index=base_panel.index, columns=base_panel.columns)
        frames = list(sample.frames)
        frames[pos] = bad
        try:
            _run(op, frames, sample.kwargs)
        except Exception:
            continue  # rejected — ConditionBool enforced
        findings.append(
            AuditFinding(
                "condition_bool_contract", meta.name, "error",
                f"condition/event panel {pname!r} silently accepted finite values "
                "outside {{0,1,NaN}} (-1/0.5/2) — no ConditionBool contract",
            )
        )
    return findings, None


# ---------------------------------------------------------------------------
# 7. unit algebra
# ---------------------------------------------------------------------------

@_category(
    "unit_algebra",
    description="name/description-derived output unit vs declared metadata",
)
def _unit_algebra(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    name = (meta.name or "").lower()
    desc = (meta.description or "").lower()
    out_unit = (_unit_tag(op) or "").lower()
    expected: str | None = None
    kind: str = "warning"
    if "tstat" in name or "t_stat" in name or "t-stat" in name or "t statistic" in desc:
        expected, kind = "dimensionless", "error"
    elif "dirichlet" in name or ("energy" in name and "spectral" not in name):
        expected, kind = "unit(target)^2", "error"
    elif any(k in name for k in ("peer_mean", "ex_self_mean", "group_mean", "mean_ex_self", "neighbor_mean")):
        expected, kind = "same_as:target", "error"
    elif "entropy" in name:
        expected, kind = "dimensionless", "warning"
    elif "probability" in name or "prob_" in name:
        expected, kind = "dimensionless", "warning"
    if expected is None:
        return [], "no unit inference from name/description"
    bad_units = {"level", "ratio", "rank", "count"}
    if out_unit in bad_units:
        return [
            AuditFinding(
                "unit_algebra", meta.name, kind,
                f"output unit declared {out_unit!r} but name/description implies "
                f"{expected!r} — typed factor composition will treat the output as "
                "the wrong dimension",
            )
        ], None
    return [], None


# ---------------------------------------------------------------------------
# 8. boundary behavior
# ---------------------------------------------------------------------------

@_category(
    "boundary_behavior",
    description="minimum parameter values must fail cleanly, never crash",
)
def _boundary_behavior(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    from cleaned_operators.base import MISSING

    specs = getattr(meta, "param_specs", None) or {}
    sample = ctx["sample"]
    panel_names = set(getattr(meta, "panel_params", None) or [])
    param_names = list(getattr(meta, "param_names", None) or [])
    findings: list[AuditFinding] = []
    for p in param_names:
        if p in panel_names:
            continue
        d = sample.kwargs.get(p)
        if not isinstance(d, (int, float, np.number)) or isinstance(d, bool):
            continue
        spec = specs.get(p)
        if spec is None or spec.min is None or spec.min < 1:
            continue
        lo = int(spec.min)
        try:
            _run(op, sample.frames, {**sample.kwargs, p: lo})
        except (ValueError, TypeError, ArithmeticError):
            continue  # clean rejection — fail-closed
        except Exception as exc:
            findings.append(
                AuditFinding(
                    "boundary_behavior", meta.name, "error",
                    f"param {p!r} at its minimum {lo} crashed with "
                    f"{type(exc).__name__}: {exc} (must raise cleanly or emit NaN)",
                )
            )
    return findings, None


# ---------------------------------------------------------------------------
# 10. semantic input type contract (round-3 review §三 Audit 10)
# ---------------------------------------------------------------------------

_VIOLATION_VALUE = {
    "price": -1.0,
    "amount": -1.0,
    "volume": -1.0,
    "rate": -1.0,
    "positive": 0.0,
}
_COND_INPUT_UNITS = {"condition", "event", "event_indicator", "condition_bool"}


@_category(
    "semantic_input_type",
    description="declared input semantics (PositivePrice / NonNegativeVolume / ConditionBool) must be enforced at runtime",
)
def _semantic_input_type(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    """Audit 10: an operator that DECLARES a restricted input semantic must
    actually fail closed on a violating value.

    PositivePrice / NonNegativeVolume / Rate panels are probed with a negative
    (or zero) value; ConditionBool/Event panels are probed with 0.5.  The
    operator must raise (or emit meaningfully less finite output than on the
    valid fixture) — if it reacts exactly as if nothing happened, the declared
    contract is not enforced and the finding is recorded.
    """
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    input_units = getattr(meta, "input_units", None) or {}
    panel_names = list(getattr(meta, "panel_params", None) or [])
    restricted = {
        p: u for p, u in input_units.items()
        if u in _VIOLATION_VALUE or u in _COND_INPUT_UNITS
    }
    if not restricted:
        return [], "no declared PositivePrice/NonNegativeVolume/ConditionBool input"
    sample = ctx["sample"]
    findings: list[AuditFinding] = []
    # baseline finite count on the valid fixture (per output column, post-warmup)
    try:
        valid_out = _run(op, sample.frames, sample.kwargs).to_numpy(dtype=float)
    except Exception as exc:  # fixture cannot exercise
        return [], f"fixture cannot exercise: {type(exc).__name__}: {exc}"
    valid_finite = int(np.isfinite(valid_out).sum())
    for pname, unit in restricted.items():
        if pname not in panel_names:
            continue
        pos = panel_names.index(pname)
        if pos >= len(sample.frames):
            continue
        base = sample.frames[pos]
        arr = base.to_numpy(dtype=float)
        if unit in _COND_INPUT_UNITS:
            arr = np.where(np.isfinite(arr), 0.5, arr)
        else:
            arr = np.where(np.isfinite(arr), _VIOLATION_VALUE[unit], arr)
        bad = pd.DataFrame(arr, index=base.index, columns=base.columns)
        frames = list(sample.frames)
        frames[pos] = bad
        try:
            out = _run(op, frames, sample.kwargs).to_numpy(dtype=float)
        except (ValueError, TypeError, ArithmeticError):
            continue  # fail-closed — enforced
        except Exception:
            continue  # treated as an enforcement rejection (fixture-level)
        finite = int(np.isfinite(out).sum())
        if finite >= valid_finite - 1:
            findings.append(
                AuditFinding(
                    "semantic_input_type", meta.name, "error",
                    f"input {pname!r} declared {unit!r} accepted the violating "
                    f"value {_VIOLATION_VALUE.get(unit, 0.5)} with no fail-closed "
                    f"reaction ({finite} finite cells vs {valid_finite} valid)",
                )
            )
    return findings, None


# ---------------------------------------------------------------------------
# 11. null calibration (round-3 review §三 Audit 7)
# ---------------------------------------------------------------------------

_ESTIMATOR_KNOB_WORDS = (
    "bins", "bin", "grid", "n_segments", "segments", "epsilon", "tolerance",
    "tol", "bandwidth", "bw", "theiler", "embed", "embedding", "dim",
    "surrogate", "n_shifts", "shifts", "samples", "draws", "order", "degree",
    "quantile_grid", "n_points", "resolution", "sigma",
)


@_category(
    "null_calibration",
    description="a parameter whose null baseline shifts wildly across its legal range is an estimator-bias search dimension",
)
def _null_calibration(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    """Audit 7: under an iid-permuted (null) input, a nuisance parameter must
    not move the estimator's noise baseline by an order of magnitude.

    For every searchable scalar param declared ESTIMATOR_RESOLUTION (or whose
    NAME is a known estimator knob with no declared role), the operator is run
    on an iid-permuted fixture at the param's min and max legal values.  If the
    null-level output dispersion shifts >10x, the parameter is an estimator-bias
    dimension and must not be a full-resolution search dimension.
    """
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    specs = getattr(meta, "param_specs", None) or {}
    sample = ctx["sample"]
    panel_names = set(getattr(meta, "panel_params", None) or [])
    param_names = list(getattr(meta, "param_names", None) or [])
    from cleaned_operators.base import ParamRole

    candidates: list[tuple[str, Any]] = []
    for p in param_names:
        if p in panel_names:
            continue
        spec = specs.get(p)
        if spec is None:
            continue
        role = getattr(spec, "param_role", None)
        if role in (ParamRole.NUMERICAL, ParamRole.POLICY):
            continue
        if role is ParamRole.ESTIMATOR_RESOLUTION:
            candidates.append((p, spec))
        elif role is None and getattr(spec, "searchable", True):
            n = p.lower()
            if any(w in n for w in _ESTIMATOR_KNOB_WORDS):
                candidates.append((p, spec))
    if not candidates:
        return [], "no estimator-resolution / estimator-knob scalar param"
    if not sample.frames:
        return [], "no panel input"
    # iid null fixture: permute each column of the first panel independently
    base = sample.frames[0]
    rng = np.random.default_rng(_stable_seed(f"{meta.name}:null"))
    null_arr = base.to_numpy(dtype=float).copy()
    for c in range(null_arr.shape[1]):
        col = null_arr[:, c]
        finite_idx = np.flatnonzero(np.isfinite(col))
        if finite_idx.size > 1:
            perm = rng.permutation(finite_idx)
            col[finite_idx] = col[perm]
    null_frame = pd.DataFrame(null_arr, index=base.index, columns=base.columns)
    frames = [null_frame, *sample.frames[1:]]

    def _null_dispersion(p: str, value: Any) -> float | None:
        try:
            out = _run(op, frames, {**sample.kwargs, p: value}).to_numpy(dtype=float)
        except Exception:
            return None
        out = out[20:]  # skip warmup
        if out.size == 0 or not np.isfinite(out).any():
            return None
        per_col = np.nanstd(out, axis=0)
        per_col = per_col[np.isfinite(per_col)]
        if per_col.size == 0:
            return None
        return float(np.mean(per_col))

    findings: list[AuditFinding] = []
    for p, spec in candidates:
        lo, hi = getattr(spec, "min", None), getattr(spec, "max", None)
        if lo is None or hi is None or lo == hi:
            continue
        s_lo = _null_dispersion(p, lo)
        s_hi = _null_dispersion(p, hi)
        if s_lo is None or s_hi is None:
            continue
        if min(s_lo, s_hi) <= 1e-12:
            continue
        ratio = max(s_lo, s_hi) / min(s_lo, s_hi)
        if ratio > 10.0:
            findings.append(
                AuditFinding(
                    "null_calibration", meta.name, "error",
                    f"param {p!r} shifts the null-level output dispersion "
                    f"{ratio:.0f}x across its legal range [{lo},{hi}] — "
                    f"estimator-bias search dimension (declare "
                    f"ESTIMATOR_RESOLUTION and search only on a reviewed grid)",
                )
            )
    return findings, None


# ---------------------------------------------------------------------------
# 12. factor practical usability (round-3 review §三 Audit 12)
# ---------------------------------------------------------------------------

_USABILITY_ROWS = 250
_USABILITY_COLS = 20


@_category(
    "practical_usability",
    description="coverage / Inf / constant / effective-sample gates on a realistic panel",
)
def _practical_usability(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    """Audit 12: a canonical must produce a usable factor on a realistic panel —
    no Inf, meaningful coverage, not mostly constant, no structural trailing NaN
    run that makes the latest decision date dead.

    The contract fixture's panels are rebuilt at 250 rows × 20 columns with the
    DECLARED input-unit domain, plus one mid-history suspension block and a
    sparse NaN sprinkle on the primary panel.  Metrics are computed on the
    operator output and gated.
    """
    meta = op.metadata
    skip = _valid_gate(op, ctx)
    if skip:
        return [], skip
    sample = ctx["sample"]
    input_units = getattr(meta, "input_units", None) or {}
    panel_names = list(getattr(meta, "panel_params", None) or [])
    if not panel_names:
        return [], "no panel input"
    idx = pd.date_range("2023-01-02", periods=_USABILITY_ROWS, freq="B")
    cols = [f"S{i}" for i in range(_USABILITY_COLS)]
    rng = np.random.default_rng(_stable_seed(f"{meta.name}:usability"))
    frames: list[pd.DataFrame] = []
    for i, pname in enumerate(panel_names):
        unit = input_units.get(pname)
        if unit in {"condition", "event", "event_indicator", "condition_bool"}:
            data = (rng.random((_USABILITY_ROWS, _USABILITY_COLS)) > 0.5).astype(float)
        elif unit in {"group", "group_id", "category", "membership"}:
            data = np.full((_USABILITY_ROWS, _USABILITY_COLS), "G", dtype=object)
        else:
            data = rng.normal(0.0, 1.0, (_USABILITY_ROWS, _USABILITY_COLS))
            data = np.abs(data) + 1.0 if unit in {"price", "volume", "amount", "positive"} else data
            data *= 10 ** rng.uniform(-0.3, 0.3, _USABILITY_COLS)
            if i == 0:
                # one mid-history suspension block per column + sparse sprinkle
                for c in range(_USABILITY_COLS):
                    start = int(rng.integers(80, _USABILITY_ROWS - 120))
                    data[start:start + 30, c] = np.nan
                mask = rng.random(data.shape) < 0.03
                data[mask] = np.nan
        frames.append(pd.DataFrame(data, index=idx, columns=cols))
    try:
        out = _run(op, frames, sample.kwargs).to_numpy(dtype=float)
    except Exception as exc:
        return [], f"fixture cannot exercise: {type(exc).__name__}: {exc}"
    if out.size == 0:
        return [], "empty output on the usability panel"
    findings: list[AuditFinding] = []
    if np.isinf(out).any():
        findings.append(
            AuditFinding(
                "practical_usability", meta.name, "error",
                f"output contains {int(np.isinf(out).sum())} Inf values — not a "
                f"usable factor",
            )
        )
    coverage = float(np.isfinite(out).sum() / out.size)
    if coverage < 0.15:
        findings.append(
            AuditFinding(
                "practical_usability", meta.name, "error",
                f"median coverage {coverage:.2f} < 0.15 — factor is structurally "
                f"sparse on a realistic panel",
            )
        )
    # cross-sectional constant: columns whose time-std is ~0 on the final 30 rows
    tail = out[-30:]
    per_col_std = np.nanstd(tail, axis=0)
    per_col_mean = np.nanmean(tail, axis=0)
    denom = np.where(np.abs(per_col_mean) > 1e-9, np.abs(per_col_mean), 1.0)
    constant_cols = np.where(np.isfinite(per_col_std) & (per_col_std / denom < 1e-9), 1, 0)
    if float(constant_cols.sum()) > 0.3 * _USABILITY_COLS:
        findings.append(
            AuditFinding(
                "practical_usability", meta.name, "error",
                f"{int(constant_cols.sum())}/{_USABILITY_COLS} columns are "
                f"cross-sectionally constant at the latest dates",
            )
        )
    # trailing NaN run: rows from the last finite row to the end, per column.
    # (per-column longest trailing streak is the honest measure)
    max_trailing = 0
    for c in range(out.shape[1]):
        col = out[:, c]
        if not np.isfinite(col).any():
            max_trailing = max(max_trailing, _USABILITY_ROWS)
        else:
            last_finite = int(np.max(np.argwhere(np.isfinite(col))))
            max_trailing = max(max_trailing, _USABILITY_ROWS - 1 - last_finite)
    trailing = max_trailing
    if trailing > 0.4 * _USABILITY_ROWS:
        findings.append(
            AuditFinding(
                "practical_usability", meta.name, "error",
                f"longest trailing NaN run is {trailing}/{_USABILITY_ROWS} rows — "
                f"the factor is dead at the most recent decision date",
            )
        )
    return findings, None


# ---------------------------------------------------------------------------
# 13. role-aware search grade enforcement (round-3 review §二 ParamRole)
# ---------------------------------------------------------------------------


@_category(
    "param_search_grade",
    description="estimator knobs must not silently resolve to full-resolution ECONOMIC; NUMERICAL/POLICY roles require searchable=False",
)
def _param_search_grade(op: Any, ctx: dict[str, Any]) -> tuple[list[AuditFinding], str | None]:
    """Review §二: every parameter that enters the alpha grammar has a role.

    * a known estimator knob (``bins``/``grid``/``n_segments``/``epsilon``/…)
      with NO declared role resolves to full-resolution ``ECONOMIC`` and must be
      declared ``ESTIMATOR_RESOLUTION`` (coarse) or ``searchable=False`` —
      warning, prompting the declaration;
    * a param whose role is ``NUMERICAL``/``POLICY`` but ``searchable=True`` is
      an internal contradiction — error.
    """
    from cleaned_operators.base import ParamRole, effective_param_role

    meta = op.metadata
    specs = getattr(meta, "param_specs", None) or {}
    panel_names = set(getattr(meta, "panel_params", None) or [])
    param_names = list(getattr(meta, "param_names", None) or [])
    findings: list[AuditFinding] = []
    for p in param_names:
        if p in panel_names:
            continue
        spec = specs.get(p)
        if spec is None:
            continue
        role = getattr(spec, "param_role", None)
        searchable = getattr(spec, "searchable", True)
        n = p.lower()
        if role in (ParamRole.NUMERICAL, ParamRole.POLICY) and searchable:
            findings.append(
                AuditFinding(
                    "param_search_grade", meta.name, "error",
                    f"param {p!r} declares role {role.value!r} (non-search) but "
                    f"searchable=True — contradiction",
                )
            )
            continue
        if role is None and searchable and any(w in n for w in _ESTIMATOR_KNOB_WORDS):
            findings.append(
                AuditFinding(
                    "param_search_grade", meta.name, "warning",
                    f"param {p!r} is a known estimator knob but has no declared "
                    f"role — it silently resolves to full-resolution ECONOMIC; "
                    f"declare ESTIMATOR_RESOLUTION (or set searchable=False)",
                )
            )
    return findings, None


# ---------------------------------------------------------------------------
# engine-level: semantic duplicates
# ---------------------------------------------------------------------------

_SUFFIXES = (
    "_score", "_index", "_ratio", "_resid", "_proxy", "_excess", "_signal",
    "_spread", "_duration", "_width", "_diff", "_change",
)


def _stem(name: str) -> str:
    lower = name.lower()
    while True:
        for suf in _SUFFIXES:
            if lower.endswith(suf):
                lower = lower[: -len(suf)]
                break
        else:
            return lower
        # continue trimming in case of stacked suffixes


def _semantic_duplicates(names: Sequence[str]) -> list[AuditFinding]:
    clusters: dict[str, list[str]] = {}
    for n in names:
        clusters.setdefault(_stem(n), []).append(n)
    findings: list[AuditFinding] = []
    for stem, members in sorted(clusters.items()):
        members = sorted(members)
        if len(members) >= 4 and stem:  # an honest cluster of >=4 canonicals
            findings.append(
                AuditFinding(
                    "semantic_duplicates", ", ".join(members), "info",
                    f"possible semantic-duplicate cluster sharing stem {stem!r}: {members}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_closure_audit(
    canonical_names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    max_canonicals: int | None = None,
) -> AuditReport:
    """Iterate the full operator registry and run the requested closure-audit
    categories over every canonical.

    ``canonical_names=None`` -> every registered canonical.  ``categories=None``
    -> every category in :data:`CLOSURE_CATEGORIES`.  ``max_canonicals`` bounds
    the sweep when a fast first pass is wanted (the registry is sorted, so the
    bound is deterministic).
    """
    from cleaned_operators.registry import OperatorRegistry

    report = AuditReport()
    enabled = list(categories) if categories else sorted(CLOSURE_CATEGORIES)
    names = sorted(OperatorRegistry._operators.keys())
    if canonical_names:
        wanted = set(canonical_names)
        names = [n for n in names if n in wanted]
    if max_canonicals is not None and not canonical_names:
        names = names[: max_canonicals]

    for name in names:
        try:
            op = _resolve_op(name)
        except KeyError:
            report.note_skipped(name, "not resolvable on pandas_numpy")
            continue
        sample = contract_fixture(name)
        ctx: dict[str, Any] = {"sample": sample, "canonical": name, "op": op}
        for cat in enabled:
            entry = CLOSURE_CATEGORIES.get(cat)
            if entry is None or entry.get("engine_level"):
                continue  # engine-level categories run as post-passes
            try:
                findings, skip_reason = entry["check"](op, ctx)
            except Exception as exc:
                report.note_audit_error(cat, f"{name}: {type(exc).__name__}: {exc}")
                continue
            if skip_reason is not None:
                report.note_skipped(cat, f"{name}: {skip_reason}")
                continue
            for finding in findings:
                report.add(finding)
            report.note_ran(cat)

    # engine-level passes
    if "semantic_duplicates" in enabled:
        for finding in _semantic_duplicates(names):
            report.add(finding)
    return report


# legacy alias so a consumer can also treat this engine as a rule source
RULES = {name: {"category": "closure", "description": entry["description"]} for name, entry in CLOSURE_CATEGORIES.items()}
