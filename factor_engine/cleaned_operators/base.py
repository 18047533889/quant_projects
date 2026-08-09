# -*- coding: utf-8 -*-
"""Pandas/NumPy operator base classes and registration helpers.

Runtime contracts are enforced centrally:
- only parameters declared as integer controls are normalised to ``int``;
- booleans cannot masquerade as windows/lags;
- all panel axes must be unique;
- multi-panel inputs must have exactly matching index/columns unless an
  operator explicitly declares the ``allow_panel_broadcast`` tag;
- ``validate_params`` is executed for every operator call.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from backend.operator_errors import OperatorParameterError

class _MissingDefaultType:
    """Sentinel: ``ParamSpec.default`` was NOT declared (vs explicitly None)."""

    _instance: "_MissingDefaultType | None" = None

    def __new__(cls) -> "_MissingDefaultType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "MISSING"


# R7-222: ``ParamSpec.default = MISSING`` distinguishes "no default declared"
# from an explicitly-declared ``None`` default.  All default checks must test
# ``spec.default is MISSING``, never ``spec.default is None`` (an explicit None
# default is a legitimate contract, e.g. ``center=None`` in a z-score).
MISSING = _MissingDefaultType()


_INTEGER_PARAM_NAMES = frozenset(
    {
        # legacy core names
        "window", "period", "periods", "d", "lag", "n", "m", "k",
        "min_periods", "max_periods", "max_lookback", "periods_per_year",
        "fast_period", "slow_period", "signal_period", "bins", "buckets",
        "order", "degree", "ddof",
        # window-ish names (review P0-06): without central validation these were
        # silently ``int(value)``-truncated inside operator kernels, so
        # fast_window=5.1 / 5.2 / 5.9 all compiled to the same formula — a false
        # search space.  `param_types` in OperatorMetadata is authoritative when
        # set; this name whitelist is the fallback for operators that do not
        # declare it.
        "fast_window", "slow_window", "signal_window", "short_window",
        "long_window", "medium_window", "tenkan_window", "kijun_window",
        "senkou_b_window", "er_window", "atr_window", "ema_window", "adl_window",
        "left_window", "right_window", "outer_window", "inner_window",
        "recent_window", "prior_window", "reference_window", "history_window",
        "old_window", "lookback_window", "baseline_window", "smooth_window",
        "scale_window", "path_window", "window_periods", "average_periods",
        # periods-ish
        "short_periods", "long_periods", "growth_periods", "compare_periods",
        # lag / count / history
        "max_lag", "event_lag", "match_lag", "fit_lag", "lookback_days",
        "history_days", "max_gap", "max_shift", "max_interval", "n_updates",
        "min_updates", "min_transitions", "min_events", "min_peers", "min_pairs",
        "min_patterns", "min_obs", "min_scale", "max_scale", "n_scales", "k_max",
        "min_valid_lags", "min_reference_days", "max_run",
        # structure / search
        "n_components", "embedding_dim", "bucket_count", "n_bins", "n_slots",
        "n_segments", "n_patterns", "steps", "cooldown", "max_spacing",
        "min_spacing", "body_window", "shadow_window", "points", "min_count",
        "top_k", "delay", "grid", "sampling",
        # review #4 integer-parameter sweep: names that kernels were still
        # ``int(value)``-truncating without central validation.  5.9 -> 5 and
        # 3.4 -> 3 compiled to the SAME factor, manufacturing a false search
        # space.  A kernel that legitimately needs a fractional value for one of
        # these must declare ``param_types`` (float) / ``ParamSpec(dtype=float)``.
        #
        # review #5 R5-07: ``tolerance``/``tau`` are removed from the name
        # whitelist — both are float-ratio semantics in pattern / expectile /
        # decay operators (``pattern_double_top(tolerance=0.02)``,
        # ``_expectile_chunk(..., tau=0.1)``).  Guessing their type from the
        # *name* was exactly the failure the whitelist was meant to fix in one
        # direction; operators that use ``tau`` as an integer embedding lag or
        # ``tolerance``/``confirmation`` as integer counts now declare
        # ``ParamSpec(dtype=int)`` explicitly.
        "confirmation", "horizon", "min_anchors",
        "min_tail_count", "min_bin_count", "cutoff", "block",
        "max_iter", "n_iter", "segments", "min_segments", "max_segments",
        "left", "right", "up_count", "down_count", "n_levels", "n_buckets",
    }
)
_NONNEGATIVE_INTEGER_PARAMS = frozenset(
    {"lag", "periods", "d", "ddof", "max_lag", "event_lag", "match_lag",
     "fit_lag", "delay", "max_shift", "max_gap", "lookback_days", "block"}
)


@dataclass(frozen=True)
class ParamSpec:
    """Authoritative contract for a single operator parameter (review #4 R4-01).

    Replaces the name-whitelist heuristics: every operator entering runtime is
    validated against its declared ``ParamSpec``, so a ``5.9`` never silently
    truncates to ``5`` and an inactive parameter never shows up in the search
    grammar.  Falls back to the legacy name whitelist when a parameter has no
    spec (operators that do not declare contracts yet).
    """

    dtype: type | None = None             # int / float / str / bool
    min: float | int | None = None
    max: float | int | None = None
    choices: tuple | None = None          # EnumSpec: canonical allowed values
    searchable: bool = True               # False -> excluded from AlphaProbe/GP grammar
    active_when: tuple | None = None      # (param_name, allowed_values): conditional activation
    # round-11 #13: machine-readable HISTORY semantics replace the name-guessing
    # window/span/lookback whitelist.  ``history_semantics`` classifies THIS
    # parameter's history kind: exact_rows (lag: value IS the row count),
    # max_rows (trailing window re-read: value - 1), finite_observations /
    # trailing_contiguous (need N finite/contiguous bars: value), or an
    # event-clock kind — report_events / session_slots / event_count /
    # report_count / session_count — which a bar-window warmup cannot derive.
    history_semantics: str | None = None
    # round-11 #14: COMPOUND history as a machine-readable formula over declared
    # parameter names (e.g. ``"outer_window + inner_window"``, ``"2 * window"``,
    # ``"window + max_pre_window_age"``).  Evaluated by the restricted arithmetic
    # interpreter against bound+default values; an unresolvable value makes the
    # operator's history UNKNOWN (conservative full history, never truncated).
    history_formula: str | None = None
    # round-7: the canonical default value.  Required for ``active_when`` runtime
    # enforcement — an INACTIVE parameter (its controller is not in the allowed
    # values) must be unprovided or exactly equal to this default, otherwise the
    # call is rejected (a dead knob that silently changes nothing must not create
    # a second AST).  ``MISSING`` (the sentinel) means "no default declared";
    # ``None`` is a legitimate declared default.
    default: Any = MISSING


# R6-24 (RelationalParamSpec): cross-parameter feasibility constraints that
# would otherwise be discovered only at runtime (``window >= 4*k+1``,
# ``min_periods <= window-1``, ``min_line < embedding_count``, …).  Declaring
# them lets the compiler/search grammar filter infeasible combinations BEFORE
# spending expression budget, instead of each kernel hand-rolling a
# ``return NaN`` guard.  ``expression`` is evaluated in the namespace of the
# bound parameters via :func:`eval_relational_expression`.

# Review-8 #462: relational expressions are parsed with :mod:`ast` and executed
# by a restricted interpreter — never free-form ``eval``.  The allowed grammar
# is: numeric literals, parameter names, arithmetic (``+ - * / **``), comparison
# (``< <= > >= == !=``), boolean (``and/or/not``) and unary sign.  Function
# calls, attributes, subscripting, comprehensions and container literals are
# rejected at parse time.  ``search`` grammar and runtime share this object.
_ALLOWED_REL_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)
_ALLOWED_REL_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_ALLOWED_REL_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_REL_UNARYOPS = (ast.USub, ast.UAdd, ast.Not)


def parse_relational_expression(expression: str) -> tuple[ast.AST, frozenset[str]]:
    """Parse + validate a restricted predicate, returning ``(tree, params)``.

    Raises :class:`ValueError` for any operator, literal or name usage outside
    the allowlist — a malformed relation must fail loudly at registration time
    instead of being ``eval``-ed at runtime.
    """
    text = str(expression).strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"invalid relational expression {expression!r}: {exc.msg}"
        ) from exc

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        # Operator node classes (Add/GtE/And/USub/...) are validated by their
        # PARENT (BinOp/BoolOp/Compare/UnaryOp) below; walk visits them as bare
        # ``_operator`` instances, so pass them through untouched.
        if isinstance(node, (ast.operator, ast.cmpop, ast.boolop, ast.unaryop)):
            continue
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                continue
            raise ValueError(
                f"relational expression {expression!r} contains unsupported "
                f"literal {node.value!r} (only numeric literals are allowed)"
            )
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_REL_BINOPS:
                raise ValueError(
                    f"relational expression {expression!r} uses unsupported "
                    f"binary operator {type(node.op).__name__}"
                )
            continue
        if isinstance(node, ast.BoolOp):
            if type(node.op) not in _ALLOWED_REL_BOOLOPS:
                raise ValueError(
                    f"relational expression {expression!r} uses unsupported "
                    f"boolean operator {type(node.op).__name__}"
                )
            continue
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_REL_UNARYOPS:
                raise ValueError(
                    f"relational expression {expression!r} uses unsupported "
                    f"unary operator {type(node.op).__name__}"
                )
            continue
        if isinstance(node, ast.Compare):
            if any(type(op) not in _ALLOWED_REL_CMPOPS for op in node.ops):
                raise ValueError(
                    f"relational expression {expression!r} uses an unsupported "
                    f"comparison operator"
                )
            continue
        raise ValueError(
            f"relational expression {expression!r} uses unsupported node "
            f"{type(node).__name__} (function calls / attributes / subscripting "
            f"are not allowed)"
        )
    return tree, frozenset(referenced)


def _eval_rel_ast(node: ast.AST, ns: dict[str, Any]) -> Any:
    """Restricted interpreter over a validated relational predicate."""
    if isinstance(node, ast.Expression):
        return _eval_rel_ast(node.body, ns)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return ns[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_rel_ast(node.left, ns)
        right = _eval_rel_ast(node.right, ns)
        op = type(node.op)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / right
        if op is ast.FloorDiv:
            return left // right
        if op is ast.Pow:
            return left ** right
        raise ValueError(f"unsupported binop {op.__name__}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_rel_ast(node.operand, ns)
        op = type(node.op)
        if op is ast.USub:
            return -value
        if op is ast.UAdd:
            return +value
        if op is ast.Not:
            return not value
        raise ValueError(f"unsupported unaryop {op.__name__}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not _eval_rel_ast(value, ns):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if _eval_rel_ast(value, ns):
                    return True
            return False
        raise ValueError("unsupported boolop")
    if isinstance(node, ast.Compare):
        left = _eval_rel_ast(node.left, ns)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_rel_ast(comparator, ns)
            optype = type(op)
            if optype is ast.Lt:
                if not (left < right):
                    return False
            elif optype is ast.LtE:
                if not (left <= right):
                    return False
            elif optype is ast.Gt:
                if not (left > right):
                    return False
            elif optype is ast.GtE:
                if not (left >= right):
                    return False
            elif optype is ast.Eq:
                if not (left == right):
                    return False
            elif optype is ast.NotEq:
                if not (left != right):
                    return False
            else:  # pragma: no cover - rejected at parse time
                raise ValueError(f"unsupported comparison {optype.__name__}")
            left = right
        return True
    raise ValueError(f"unsupported node {type(node).__name__}")


@dataclass
class RelationalParamSpec:
    """A declared cross-parameter feasibility constraint.

    ``expression`` is a restricted expression over parameter names, e.g.
    ``"window >= 4*k + 1"`` or ``"min_periods <= window - 1"``.  Parameters are
    bound positionally+by-name exactly like ``_normalise_call`` does, so every
    relation is checked on the SAME values the kernel will receive.

    ``message`` (optional) replaces the default "parameter relation violated"
    text; it may interpolate the bound values with ``{window}`` etc.

    Review-8 #462: the expression is parsed once (register/construct time) into
    a restricted predicate AST — never free-form ``eval``.  A relation that
    cannot be parsed by :func:`parse_relational_expression` raises at
    construction.
    """

    expression: str
    message: str | None = None

    def __post_init__(self) -> None:
        tree, params = parse_relational_expression(self.expression)
        object.__setattr__(self, "_rel_tree", tree)
        object.__setattr__(self, "_rel_param_names", params)

    @property
    def param_names(self) -> frozenset[str]:
        """The parameter names referenced by this relation (validated at parse)."""
        return self._rel_param_names

    def check(self, bound: dict[str, Any]) -> bool:
        """Evaluate the relation against a bound-parameter dict."""
        ns = {name: bound[name] for name in self._rel_param_names if name in bound}
        try:
            return bool(_eval_rel_ast(self._rel_tree, ns))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # NaN/None/absent parameters: relation undecidable -> treat as
            # unmet so search/replanning is forced to a feasible combination.
            return False

    def describe(self, bound: dict[str, Any]) -> str:
        if self.message:
            try:
                return self.message.format(**bound)
            except (KeyError, IndexError, ValueError):
                return self.message
        return f"parameter relation violated: {self.expression}"


def eval_relational_expression(expression: str, bound: dict[str, Any]) -> bool:
    """Evaluate one relational expression against bound parameters (helper)."""
    return RelationalParamSpec(expression=expression).check(bound)


# Typed broadcast tags (review #4 R4-99): a bare ``allow_panel_broadcast`` is a
# blanket waiver of multi-panel axis parity.  Operators that legitimately
# broadcast must declare the SPECIFIC shape they need so a daily scalar cannot
# silently stretch across symbols/sessions.
_TYPED_BROADCAST_TAGS = frozenset(
    {
        "daily_to_minute_broadcast",       # daily scalar/row -> minute panel, same day
        "scalar_to_cross_section_broadcast",  # scalar -> every instrument column
        "same_trading_date_broadcast",     # row aligned on the trading-date level only
        "session_boundary_broadcast",      # session summary row -> each minute slot
    }
)


# R11 P0-09: structured broadcast declaration replacing the bare blanket waiver.
# A ``BroadcastSpec`` names the exact mapping (``mode`` + ``date_mapping`` +
# ``instrument_policy`` + ``timezone``) the operator needs, so a broadcast
# satisfies a PROVABLE contract instead of an unverifiable tag.  The bare
# ``allow_panel_broadcast`` tag remains accepted as the legacy research waiver,
# but production admission is gated on a declared spec (see
# ``verify_broadcast`` / the registration audit).
@dataclass(frozen=True)
class BroadcastSpec:
    """Structured multi-panel broadcast declaration (R11 P0-09).

    ``mode`` is one of the supported mapping shapes; ``date_mapping`` describes
    how the broadcast input's dates align to the base panel's (``exact`` for
    same-row, ``trading_date`` for same-trading-day, ``session`` for the owning
    session); ``instrument_policy`` is ``exact`` / ``subset`` /
    ``independent``; ``timezone`` anchors date normalisation.  Any axis the spec
    does not cover (unknown index type, unknown grain, unknown mapping) must
    fail closed at runtime rather than silently pass.
    """

    mode: str  # one of the _TYPED_BROADCAST_TAGS (sans _broadcast suffix) or a registered alias
    date_mapping: str = "trading_date"
    instrument_policy: str = "subset"
    timezone: str | None = None

    def __post_init__(self) -> None:
        allowed_modes = {"daily_to_minute", "scalar_to_cross_section",
                         "same_trading_date", "session_boundary"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"BroadcastSpec.mode={self.mode!r} is not a supported broadcast "
                f"mode {sorted(allowed_modes)} (R11 P0-09 fail-closed)"
            )
        if self.date_mapping not in {"exact", "trading_date", "session"}:
            raise ValueError(
                f"BroadcastSpec.date_mapping={self.date_mapping!r} is unknown "
                "(R11 P0-09 fail-closed)"
            )
        if self.instrument_policy not in {"exact", "subset", "independent"}:
            raise ValueError(
                f"BroadcastSpec.instrument_policy={self.instrument_policy!r} is "
                "unknown (R11 P0-09 fail-closed)"
            )


@dataclass
class OperatorMetadata:
    name: str
    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    param_types: Dict[str, type] = field(default_factory=dict)
    return_type: str = "series"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    # Optional field-semantic metadata.  Existing operators may omit these.
    input_fields: List[str] = field(default_factory=list)
    output_field: str | None = None
    input_units: Dict[str, str] = field(default_factory=dict)
    output_unit: str | None = None
    compatible_units: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    # R6-24: declared cross-parameter feasibility constraints.  Evaluated in
    # validate_operator_call on the SAME bound values the kernel receives, so a
    # guaranteed-NaN combination is rejected at the call boundary, not after an
    # expensive rolling loop.  Search grammar consumers read this list to prune
    # infeasible regions before generation.
    relational_specs: List[RelationalParamSpec] = field(default_factory=list)
    # review #4 R4-01: per-parameter authoritative contracts (dtype/min/max/
    # choices/searchable/active_when/history_semantics).  When a name has a spec
    # here, it OVERRIDES the legacy name whitelist in both directions — a
    # declared float stays float even for a window-ish name, and a spec'd int is
    # validated even for a name outside the whitelist.
    param_specs: Dict[str, ParamSpec] = field(default_factory=dict)
    # review #4 R4-95: meaning of the window-like parameters in this operator.
    window_semantics: str | None = None
    # review #5 R5-06: declared keyword aliases -> canonical parameter.  Kernels
    # historically read both ``d`` and ``window`` (or ``p``/``q``,
    # ``std_dev``/``k``) for the same positional slot; those aliases are now an
    # explicit, catalog-visible declaration so the strict unknown-kwarg gate can
    # reject genuinely hidden parameters without breaking documented aliases.
    param_aliases: Dict[str, str] = field(default_factory=dict)
    # WS4 P0-07: time-frequency grain contract.  A ``SessionAggregationOperator``
    # (or any operator that changes frequency — e.g. minute panel -> daily panel)
    # declares its input/output grain here so the catalog and policy layer can
    # see the frequency change instead of guessing from the return shape.  ``None``
    # means same-grain / undeclared.
    input_grain: str | None = None
    output_grain: str | None = None
    # R6-196: machine-readable availability contract for EOD-realised / session-
    # realised operators.  ``available_at`` is when the value becomes usable
    # (``session_close`` / ``report_date`` / ``next_open`` …); ``same_session_usable``
    # is False for operators whose output at minute ``t`` depends on the rest of
    # the session (impact paths, daily aggregates) and must never feed an
    # intra-session decision.  These are fields, not just docstring text, so the
    # execution layer can refuse same-session misuse mechanically.
    available_at: str | None = None
    same_session_usable: bool | None = None
    # R9-OP-024: machine-readable operator ROLE.  ``global_state`` marks an
    # operator whose output is a single value per date spread identically over
    # every instrument (e.g. ``cs_hartigan_dip``): it can never be a terminal
    # cross-sectional alpha (CS IC is meaningless on a constant cross-section)
    # and is only valid as a regime/condition input.  This is a FIELD, not a
    # tag/comment, so the search grammar can gate terminal generation
    # mechanically instead of guessing from docstrings.
    role: str | None = None
    # R9-P1-047: declared cost contract ``cost_model(params, shape) ->
    # (runtime_cost, memory_cost)`` in units of one 120-row rolling op.  When
    # set, ``operator_cost_model.runtime_cost/memory_cost`` use it instead of
    # the prefix-name complexity table — the operator declares its true
    # parameter/panel complexity rather than being guessed from its name.
    # ``None`` = no explicit contract (prefix fallback applies).
    cost_model: Callable[[dict[str, Any], tuple[int, int] | None], tuple[float, float]] | None = None
    # R11 P0-09: structured broadcast declarations.  When non-empty, the typed
    # broadcast verification runs against these specs (each input index must
    # satisfy the declared mapping or fail closed); a bare ``allow_panel_broadcast``
    # tag without a spec remains the legacy research waiver.
    broadcast_specs: tuple["BroadcastSpec", ...] = ()
    # round-7: explicit positional PANEL-input arity for zero-parameter operators
    # whose panel contract is not expressible in ``param_names`` (``log``=1,
    # ``add``=2).  ``None`` = kernel-implied (legacy).  When set, the
    # extra-positional gate uses it instead of ``len(param_names)`` so an
    # over-long call is rejected even for an op with ``param_names=[]``.
    input_arity: int | None = None
    # round-7: declared PANEL-input names (vs scalar parameters).  Empty = the
    # manifest layer falls back to kernel-signature inference (params without a
    # default are panel inputs).  Lets the machine manifest distinguish
    # ``input_fields`` from ``scalar_parameters`` instead of listing ``window`` /
    # ``lag`` as data fields.
    panel_params: tuple[str, ...] = ()
    # R7-224: explicit panel-input arity, distinct from total positional arity.
    # ``panel_arity`` counts ONLY the panel/Series arguments (``ts_corr(x, y,
    # window)`` -> 2); ``total_positional_arity`` counts every positional
    # argument INCLUDING scalar params (``ts_corr(x, y, window)`` -> 3).  When
    # ``total_positional_arity`` is set it replaces ``input_arity`` in the
    # extra-positional gate; when only ``panel_arity`` is set, ``scalar_params``
    # provides the count of trailing scalar slots.  ``None`` = kernel-implied
    # (legacy inference from param_names/defaults).
    panel_arity: int | None = None
    total_positional_arity: int | None = None
    # R7-224: declared scalar (non-panel) parameter names — the complement of
    # ``panel_params`` within ``param_names``.  Used to derive
    # ``total_positional_arity = len(panel_params) + len(scalar_params)`` when
    # the explicit arity fields are not set.
    scalar_params: tuple[str, ...] = ()


def _normalise_integer(
    value: Any,
    name: str,
    declared_type: type | None = None,
    spec: ParamSpec | None = None,
) -> Any:
    """Validate one parameter value against its authoritative contract.

    Resolution order (review #4 R4-01): ``ParamSpec.dtype`` first, then
    ``OperatorMetadata.param_types``, then the legacy name whitelist.  A spec
    may also carry ``min``/``max``/``choices`` that are enforced regardless of
    dtype.

    R7-219: a spec is the sole authority for bounds.  ``ParamSpec(dtype=int)``
    with ``min=None`` means *truly unbounded* (``-1``, ``0`` and ``1`` are all
    legal), NOT "fall back to the legacy ≥1 lower bound".  The name-whitelist
    lower bound applies only when there is NO spec at all — i.e. only legacy
    operators without a declared contract use the name heuristic.
    """
    lower_default = 0 if name in _NONNEGATIVE_INTEGER_PARAMS else 1
    if spec is not None:
        lower = spec.min  # None => unbounded (declaration is authoritative)
        upper = spec.max  # None => unbounded
    else:
        lower = lower_default
        upper = None
    choices = spec.choices if spec is not None and spec.choices else None

    def _check_int(result: int) -> int:
        if lower is not None and result < lower:
            raise OperatorParameterError(f"{name} must be >= {lower}")
        if upper is not None and result > upper:
            raise OperatorParameterError(f"{name} must be <= {upper}")
        if choices is not None and result not in choices:
            raise OperatorParameterError(
                f"{name}={result} is not an allowed choice {list(choices)}"
            )
        return result

    # R7-220: every declared ``ParamSpec`` — int/float/bool/str/choices — routes
    # through the single strict validator below.  Only operators WITHOUT a spec
    # keep the legacy name-whitelist / ``param_types`` heuristics.
    if spec is not None and spec.dtype is not None:
        return _validate_param_spec(value, name, spec, lower, upper)

    is_int_declared = declared_type is int
    if is_int_declared:
        # Authoritative source: a param declared int is validated regardless of
        # its name, so `int(5.9) -> 5` can never slip through (review P0-06 / R4-01).
        if isinstance(value, (bool, np.bool_)):
            raise OperatorParameterError(f"{name} must be an integer, not bool")
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)) or float(value) != float(int(value)):
                raise OperatorParameterError(f"{name} must be an integer")
            return _check_int(int(value))
        # round-7 P0: a declared-int parameter receiving a non-numeric (e.g. the
        # string ``"20"``) is a contract violation, NOT a silent ``int("20")``.
        # Numeric string literals are converted to numbers at the DSL parser
        # layer; the runtime never guesses.
        raise OperatorParameterError(
            f"{name} must be an integer, not {type(value).__name__} ({value!r})"
        )
    if spec is not None and spec.dtype is float:
        # Explicitly declared float: validate numeric and bounds, never truncate.
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise OperatorParameterError(f"{name} must be finite")
            if lower is not None and numeric < lower:
                raise OperatorParameterError(f"{name} must be >= {lower}")
            if upper is not None and numeric > upper:
                raise OperatorParameterError(f"{name} must be <= {upper}")
            # round-7 P0: ``choices`` must be enforced for EVERY dtype.  The
            # old flow only checked choices on the no-dtype EnumSpec branch, so
            # ``ParamSpec(dtype=float, choices=(0.05, 0.1, 0.2))`` accepted any
            # float (``tail_fraction`` / ``quantile`` / ``alpha`` / ``bandwidth`` /
            # ``split_quantile`` — a false search space).
            if choices is not None and numeric not in choices:
                raise OperatorParameterError(
                    f"{name}={numeric} is not an allowed choice {list(choices)}"
                )
            return value
        raise OperatorParameterError(
            f"{name} must be a real number, not {type(value).__name__} ({value!r})"
        )
    if spec is not None and spec.choices is not None and name not in _INTEGER_PARAM_NAMES:
        # EnumSpec: string/bool/numeric choices are canonicalized and checked.
        if isinstance(value, (bool, np.bool_)) and True not in choices and False not in choices:
            raise OperatorParameterError(f"{name} must be one of {list(choices)}")
        if value not in choices:
            raise OperatorParameterError(
                f"{name}={value!r} is not an allowed choice {list(choices)}"
            )
        return value
    if name not in _INTEGER_PARAM_NAMES:
        return value
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if not np.isfinite(float(value)) or float(value) != float(int(value)):
        raise OperatorParameterError(f"{name} must be an integer")
    return _check_int(int(value))


def _validate_param_spec(
    value: Any,
    name: str,
    spec: ParamSpec,
    lower: Any,
    upper: Any,
) -> Any:
    """Single strict entry point for a declared ``ParamSpec`` (R7-220).

    Every declared dtype is enforced here — including ``bool`` (``type(x) is
    bool``, never a truthy int) and ``str`` — instead of each caller branch
    re-implementing its own acceptance rules.  ``int`` accepts ``Integral`` but
    not ``bool``; ``float`` accepts ``Real`` but not ``bool`` and must be
    finite; ``choices`` requires exact membership.
    """
    dtype = spec.dtype
    choices = spec.choices
    # R7-222: an explicitly-declared ``None`` default (``default=None``, NOT
    # ``MISSING``) makes ``None`` a legal value regardless of dtype — e.g.
    # ``center=None`` (no centering) in a z-score.  Without this, the strict
    # type gate would reject the very value the contract declares as default.
    if value is None and spec.default is None:
        return value
    if dtype is bool:
        # Strict: ``type(value) is bool`` — a truthy ``1``/``1.0`` is a contract
        # violation, not a usable boolean.
        if type(value) is not bool:
            raise OperatorParameterError(
                f"{name} must be a boolean, not {type(value).__name__} ({value!r})"
            )
        return value
    if dtype is str:
        if not isinstance(value, str):
            raise OperatorParameterError(
                f"{name} must be a string, not {type(value).__name__} ({value!r})"
            )
        if choices is not None and value not in choices:
            raise OperatorParameterError(
                f"{name}={value!r} is not an allowed choice {list(choices)}"
            )
        return value
    if dtype is int:
        if isinstance(value, (bool, np.bool_)):
            raise OperatorParameterError(f"{name} must be an integer, not bool")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise OperatorParameterError(
                f"{name} must be an integer, not {type(value).__name__} ({value!r})"
            )
        if not np.isfinite(float(value)) or float(value) != float(int(value)):
            raise OperatorParameterError(f"{name} must be an integer")
        result = int(value)
        if lower is not None and result < lower:
            raise OperatorParameterError(f"{name} must be >= {lower}")
        if upper is not None and result > upper:
            raise OperatorParameterError(f"{name} must be <= {upper}")
        if choices is not None and result not in choices:
            raise OperatorParameterError(
                f"{name}={result} is not an allowed choice {list(choices)}"
            )
        return result
    if dtype is float:
        if isinstance(value, (bool, np.bool_)):
            raise OperatorParameterError(f"{name} must be a real number, not bool")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise OperatorParameterError(
                f"{name} must be a real number, not {type(value).__name__} ({value!r})"
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise OperatorParameterError(f"{name} must be finite")
        if lower is not None and numeric < lower:
            raise OperatorParameterError(f"{name} must be >= {lower}")
        if upper is not None and numeric > upper:
            raise OperatorParameterError(f"{name} must be <= {upper}")
        if choices is not None and numeric not in choices:
            raise OperatorParameterError(
                f"{name}={numeric} is not an allowed choice {list(choices)}"
            )
        return value
    # No recognized dtype: if choices are declared, require exact membership.
    if choices is not None:
        if value not in choices:
            raise OperatorParameterError(
                f"{name}={value!r} is not an allowed choice {list(choices)}"
            )
        return value
    return value


# review #5 R5-06 / round-7 P0: legacy keyword aliases that kernels read for the
# same positional slot (``d``/``window``, ``p``/``q``, ``std_dev``/``k`` …).
# Each alias maps to the CANONICAL parameter names it is a synonym for.  A legacy
# alias kwarg is accepted ONLY when the operator actually declares one of those
# canonical targets in ``param_names`` / ``param_aliases`` — so
# ``operator(x, alpha=0.1)`` is rejected when the operator has no ``alpha`` /
# ``halflife`` / ``lambda_param`` parameter (the kernel would silently swallow it
# and ``alpha=0.1`` vs ``alpha=0.9`` would manufacture two identical ASTs).
# New operators must declare ``param_aliases`` instead of relying on this set.
_LEGACY_KERNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "d": ("window", "lag", "periods", "delay", "horizon", "lookback"),
    "window": ("window", "period", "span", "n"),
    "p": ("order", "window", "period", "p"),
    "q": ("q", "order", "max_q"),
    "n": ("window", "n", "period", "min_periods"),
    "k": ("window", "k", "n_components", "top_k", "order"),
    "m": ("window", "m", "order"),
    "min": ("min", "lower", "lo"),
    "max": ("max", "upper", "hi"),
    "span": ("span", "window", "halflife"),
    "lo": ("lo", "lower", "min"),
    "hi": ("hi", "upper", "max"),
    "std_dev": ("std_dev", "k", "window"),
    "fast": ("fast", "fast_period", "fast_window"),
    "slow": ("slow", "slow_period", "slow_window"),
    "signal": ("signal", "signal_period", "signal_window"),
    "alpha": ("alpha", "halflife", "lambda_param", "decay"),
    "lambda_param": ("lambda_param", "alpha", "halflife"),
    "method": ("method", "strategy"),
    "field": ("field", "column", "value"),
    "value": ("value", "field", "threshold"),
    "strategy": ("strategy", "method"),
}


def _coerce_declared_numeric_string(
    value: Any,
    name: str,
    declared_type: type | None = None,
    spec: ParamSpec | None = None,
) -> Any:
    """Controlled numeric-string coercion for a declared NUMERIC parameter.

    Round-11 #16: the DSL parser no longer converts numeric-looking strings
    (``"000001"`` must stay a string for a string/code parameter).  A STRING
    literal reaching the binder is coerced to ``int``/``float`` ONLY when the
    parameter's declared contract is numeric (``ParamSpec(dtype=int|float)`` or a
    numeric ``param_types`` entry).  A string parameter — or a parameter with no
    numeric contract — keeps the exact string.  An unparseable string for a
    numeric contract is a hard error, never a silent pass-through.
    """
    if not isinstance(value, str):
        return value
    if spec is not None:
        dtype = spec.dtype
    else:
        dtype = declared_type
    if dtype in (str, bool) or dtype is None:
        # String-contract (or undeclared) parameters never convert: ``"000001"``
        # stays ``"000001"``, a category ID stays its string.
        return value
    try:
        numeric_dtype = dtype in (int, np.integer)
    except TypeError:  # dtype may be a non-type (unlikely) — be safe
        return value
    if not numeric_dtype and dtype not in (float, np.floating):
        return value
    stripped = value.strip()
    if not stripped:
        raise OperatorParameterError(
            f"{name}: empty numeric string cannot be bound to a numeric parameter"
        )
    try:
        parsed = int(stripped) if numeric_dtype else float(stripped)
    except ValueError as exc:
        raise OperatorParameterError(
            f"{name}: numeric string {value!r} cannot be converted to "
            f"{dtype.__name__ if isinstance(dtype, type) else 'numeric'}"
        ) from exc
    if not np.isfinite(float(parsed)):
        raise OperatorParameterError(f"{name}: numeric string {value!r} is non-finite")
    return parsed


def _kernel_param_defaults(operator: Any) -> dict[str, Any]:
    """Canonical default values from the operator kernel signature.

    Used for ``ParamSpec.active_when`` runtime enforcement: an INACTIVE
    parameter is only tolerated when it equals its canonical default (the "dead
    knob set to its no-op value" case).  Fall back to ``ParamSpec.default`` when
    the kernel signature is not introspectable.

    R6-24: the ``register_dual`` bridge binds the real kernel as the ``_fn``
    default of ``_calculate_series(*args, _fn=fn, **kwargs)`` — signature
    introspection of the bridge itself yields only ``_fn``.  Resolve through the
    ``_fn`` default so relational specs see the kernel's real per-parameter
    defaults (window/dim/delay/min_line/…), not just the bridge.
    """
    import inspect

    fn = getattr(operator, "_calculate_series", None) or getattr(operator, "calculate", None)
    if fn is None:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    params = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    # Resolve through a ``_fn`` bridge default when present.
    bridged = params.get("_fn")
    if bridged is not None and callable(bridged):
        try:
            bsig = inspect.signature(bridged)
        except (TypeError, ValueError):
            bsig = None
        if bsig is not None:
            for name, param in bsig.parameters.items():
                if param.default is not inspect.Parameter.empty and name not in params:
                    params[name] = param.default
    return params


def _active_allows(allowed: Any, ctrl_val: Any) -> bool:
    """Membership test for a ``ParamSpec.active_when`` allowed-values set."""
    if isinstance(allowed, (set, frozenset, tuple, list)):
        return ctrl_val in allowed
    return ctrl_val == allowed


def _enforce_active_when(
    metadata: OperatorMetadata,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    defaults: dict[str, Any] | None,
) -> None:
    """Round-7 P0: enforce ``ParamSpec.active_when`` at runtime.

    A parameter is INACTIVE when its controlling parameter's value is not in the
    declared allowed set.  An inactive parameter must be unprovided or exactly
    equal to its canonical default — otherwise the call is rejected, because
    varying a dead knob creates two ASTs with identical output (a false search
    space).  When the controller is unbound, the judge is skipped (fail-open).
    """
    names = list(getattr(metadata, "param_names", None) or [])
    specs = getattr(metadata, "param_specs", None) or {}
    if not specs:
        return
    bound: dict[str, Any] = {name: args[index] for index, name in enumerate(names[: len(args)])}
    bound.update(kwargs)
    defaults = defaults or {}
    for pname, spec in specs.items():
        if spec is None or spec.active_when is None:
            continue
        controller, allowed = spec.active_when
        ctrl_val = bound.get(controller)
        if ctrl_val is None:
            continue  # controller unbound -> cannot judge; fail-open
        if _active_allows(allowed, ctrl_val):
            continue  # active
        # INACTIVE: only tolerate unprovided, or equal to the canonical default.
        if pname not in bound:
            continue
        provided = bound[pname]
        # R7-222: ``MISSING`` = no default declared -> fall back to the kernel
        # signature default; an explicit ``None`` default is a real default that
        # an inactive parameter may legitimately be pinned to.
        if spec.default is not MISSING:
            canonical_default = spec.default
        else:
            canonical_default = defaults.get(pname, MISSING)
        if provided == canonical_default:
            continue
        raise OperatorParameterError(
            f"{metadata.name}: parameter {pname!r} is inactive when "
            f"{controller}={ctrl_val!r} (allowed: {sorted(allowed) if isinstance(allowed, (tuple, list, set, frozenset)) else allowed}); "
            "provide only its canonical default or omit it (round-7 P0 — a dead "
            "knob must not create a second AST)"
        )


def _normalise_call(
    metadata: OperatorMetadata,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
):
    # ``getattr`` keeps this compatible with the parallel polars metadata class
    # (base_polars.OperatorMetadata has no param_specs / param_types on every
    # instance); both classes share the param_names contract.
    names = list(getattr(metadata, "param_names", None) or [])
    types = getattr(metadata, "param_types", None) or {}
    specs = getattr(metadata, "param_specs", None) or {}
    aliases = set(getattr(metadata, "param_aliases", None) or {})
    tags = {str(t).lower() for t in (getattr(metadata, "tags", None) or [])}
    variadic = "variadic" in tags or "dynamic_inputs" in tags
    # R5-06: reject extra positional arguments past the declared contract unless
    # the operator is explicitly variadic.  A fifth panel silently accepted by a
    # four-parameter kernel is a contract violation, not a feature.
    #
    # R6 P0-03/P0-25 refinement: an operator that declares NO named parameters
    # (``param_names=[]`` — legacy unary/binary elementwise like ``log``,
    # ``add``, ``subtract``) has no declared positional contract to violate; its
    # panel arguments are implied by the kernel.  Enforcing ``len(args) >
    # len(names)`` there would reject every valid ``log(x)`` call.  The strict
    # extra-positional gate applies only to operators that DO declare a
    # parameter contract; zero-parameter ops keep their implicit panel arity
    # (an over-long call still fails inside the kernel).
    #
    # round-7: an operator that explicitly declares ``input_arity`` gets an exact
    # positional contract regardless of ``param_names`` (``log``=1, ``add``=2),
    # so a zero-param op with a ``*args`` kernel cannot silently swallow extra
    # panels.
    if not variadic:
        # R7-224: total positional arity is the panel arity + scalar params.  An
        # explicit ``total_positional_arity`` wins; then ``input_arity`` (the
        # round-7 panel-input count); then panel_params + scalar_params when both
        # are declared; then the legacy ``len(param_names)``.
        declared_arity = getattr(metadata, "total_positional_arity", None)
        if declared_arity is None:
            declared_arity = getattr(metadata, "input_arity", None)
        if declared_arity is None:
            _pp = getattr(metadata, "panel_params", None) or ()
            _sp = getattr(metadata, "scalar_params", None) or ()
            if _pp or _sp:
                declared_arity = len(_pp) + len(_sp)
        if declared_arity is not None:
            if len(args) != int(declared_arity):
                raise OperatorParameterError(
                    f"{metadata.name}: declares input_arity={declared_arity} but "
                    f"received {len(args)} positional arguments"
                )
        elif names and len(args) > len(names):
            raise OperatorParameterError(
                f"{metadata.name}: received {len(args)} positional arguments but "
                f"declares {len(names)} parameters {names}; extra positional "
                "arguments are rejected unless the operator declares variadic"
            )
    processed_args = [
        _normalise_integer(
            _coerce_declared_numeric_string(
                value,
                names[index] if index < len(names) else "",
                types.get(names[index]) if index < len(names) else None,
                specs.get(names[index]) if index < len(names) else None,
            ),
            names[index] if index < len(names) else "",
            types.get(names[index]) if index < len(names) else None,
            specs.get(names[index]) if index < len(names) else None,
        )
        for index, value in enumerate(args)
    ]
    # R7-223: alias -> canonical param-name map.  A keyword given under an alias
    # spelling is validated against the CANONICAL target's ParamSpec/type/bounds
    # — never the alias's own (usually empty) spec — so ``d=5`` on an operator
    # whose canonical ``window`` is declared ``ParamSpec(dtype=int, min=2)`` is
    # validated as ``window=5``.  ``metadata.param_aliases`` (explicit
    # declarations) wins over the legacy kernel-alias table.
    explicit_aliases = getattr(metadata, "param_aliases", None) or {}
    alias_target: dict[str, str] = {}
    for key in list(kwargs.keys()):
        if key in explicit_aliases:
            target = explicit_aliases[key]
            if target not in names:
                raise OperatorParameterError(
                    f"{metadata.name}: alias {key!r} -> {target!r}, but {target!r} "
                    "is not a declared parameter (R7-223); the alias must point at "
                    "a canonical param_names entry"
                )
            alias_target[key] = target
        elif key in _LEGACY_KERNEL_ALIASES and not variadic:
            targets = _LEGACY_KERNEL_ALIASES[key]
            matched = [t for t in targets if t in names or t in aliases]
            if matched:
                # Prefer the first declared canonical target, mirroring the
                # kernel's own lookup order.
                alias_target[key] = matched[0]
    processed_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        # Round-11 #16: controlled numeric-string coercion against the declared
        # contract happens once per kwarg, before any alias resolution — the
        # canonical target's spec (via alias_target) is the authority below.
        if key in names:
            coerced = _coerce_declared_numeric_string(
                value, key, types.get(key), specs.get(key)
            )
            processed_kwargs[key] = _normalise_integer(
                coerced, key, types.get(key), specs.get(key)
            )
            continue
        if key in alias_target:
            # R7-223: validate against the canonical target's contract.
            canon = alias_target[key]
            coerced = _coerce_declared_numeric_string(
                value, canon, types.get(canon), specs.get(canon)
            )
            processed_kwargs[key] = _normalise_integer(
                coerced, canon, types.get(canon), specs.get(canon)
            )
            continue
        if key in aliases:
            # Legacy alias declared without a param_aliases entry but present in
            # the alias set: keep the alias spelling as the kwarg key but validate
            # against the (empty) alias slot — accepted for backward compat.
            coerced = _coerce_declared_numeric_string(
                value, key, types.get(key), specs.get(key)
            )
            processed_kwargs[key] = _normalise_integer(
                coerced, key, types.get(key), specs.get(key)
            )
            continue
        if variadic:
            coerced = _coerce_declared_numeric_string(
                value, key, types.get(key), specs.get(key)
            )
            processed_kwargs[key] = _normalise_integer(
                coerced, key, types.get(key), specs.get(key)
            )
            continue
        raise OperatorParameterError(
            f"{metadata.name}: undeclared keyword parameter {key!r}; parameters "
            f"are {names}.  A hidden keyword that changes results without being "
            "visible in the catalog is rejected (R5-06 / round-7 P0); declare it "
            "in param_names / param_aliases or tag the operator variadic."
        )
    _enforce_active_when(metadata, args, kwargs, defaults)
    return tuple(processed_args), processed_kwargs


def _frame_instruments(frame: Any) -> set | None:
    """Instrument set of a panel, or None when there is no instrument level."""
    index = getattr(frame, "index", None)
    if index is not None and isinstance(index, pd.MultiIndex) and "instrument" in index.names:
        return set(index.get_level_values("instrument"))
    # Polars wide panels carry the instrument identity as a metadata column.
    for key in ("inst", "instrument", "stock_code"):
        if key in getattr(frame, "columns", ()) or key in getattr(frame, "schema", ()):
            series = frame[key]
            try:
                values = list(series.to_list())
            except AttributeError:  # pandas column
                values = list(series)
            return set(values)
    return None


def _is_panel(value: Any) -> bool:
    """A multi-column panel (pandas wide, polars wide/long) rather than a scalar."""
    if isinstance(value, (int, float, str, bool, np.number)) or value is None:
        return False
    columns = getattr(value, "columns", None)
    if columns is None:
        return False
    try:
        n = len(list(columns))
    except TypeError:
        return False
    return n > 0


def _verify_typed_broadcast_axes(
    metadata: OperatorMetadata, tags: set[str], frames: list[Any]
) -> None:
    """Round-7 P0: real timestamp/session verification for typed broadcasts.

    The typed broadcast tags declare the SPECIFIC mapping the operator needs; a
    broadcast must satisfy that mapping, not just the structural column checks.
    Checks are best-effort — they run only when both axes are ``DatetimeIndex``
    and skip otherwise (exotic MultiIndex / polars long axes keep the structural
    guards already applied).

    R11 P0-09 fail-closed: when a TYPED broadcast tag is declared but the base
    axis is not a ``DatetimeIndex``, the date mapping CANNOT be verified — the
    old ``return`` silently let an unverifiable broadcast through.  An unknown
    axis must be rejected, not passed.
    """
    if len(frames) < 2:
        return
    typed = bool(tags & _TYPED_BROADCAST_TAGS)
    base_idx = getattr(frames[0], "index", None)
    if not isinstance(base_idx, pd.DatetimeIndex):
        if typed:
            raise ValueError(
                f"{metadata.name}: declares a typed broadcast but the base panel "
                f"index {type(base_idx).__name__} is not a DatetimeIndex — the "
                f"date mapping cannot be verified (R11 P0-09 fail-closed)"
            )
        return

    def _dates(idx: Any):
        try:
            return idx.normalize()
        except (TypeError, ValueError):
            return None

    base_dates = _dates(base_idx)
    if base_dates is None:
        return

    if "same_trading_date_broadcast" in tags:
        for position, frame in enumerate(frames[1:], start=1):
            other_idx = getattr(frame, "index", None)
            if isinstance(other_idx, pd.DatetimeIndex) and len(other_idx) == len(base_idx):
                other_dates = _dates(other_idx)
                if other_dates is not None and not other_dates.equals(base_dates):
                    raise ValueError(
                        f"{metadata.name}: same_trading_date_broadcast input panel "
                        f"{position} is NOT row-aligned on the same trading dates as "
                        "the base — broadcasting a value onto a different day is a "
                        "silent look-ahead (round-7 P0)"
                    )

    if "daily_to_minute_broadcast" in tags:
        base_set = set(base_dates)
        for position, frame in enumerate(frames[1:], start=1):
            other_idx = getattr(frame, "index", None)
            if not isinstance(other_idx, pd.DatetimeIndex):
                continue
            other_dates = _dates(other_idx)
            if other_dates is None:
                continue
            missing = base_set - set(other_dates)
            if missing:
                raise ValueError(
                    f"{metadata.name}: daily_to_minute_broadcast daily base carries "
                    f"trading dates absent from the minute panel "
                    f"({sorted(missing)[:3]} ...) — a daily row must map to minute "
                    "slots on the SAME trading date (round-7 P0)"
                )


def _verify_broadcast_specs(
    metadata: OperatorMetadata, specs: tuple[BroadcastSpec, ...], frames: list[Any]
) -> None:
    """Verify non-base panels against a declared :class:`BroadcastSpec`.

    R11 P0-09 fail-closed: the spec names the mapping the operator needs.  For
    each non-base panel we require its axis to be *interpretable* — a
    ``DatetimeIndex`` so the date mapping can actually be checked — otherwise
    the broadcast is unverifiable and must be rejected, not passed.  This closes
    the old fail-open where an unknown index type silently bypassed the date
    alignment gate.
    """
    base = frames[0]
    base_idx = getattr(base, "index", None)
    if not isinstance(base_idx, pd.DatetimeIndex):
        raise ValueError(
            f"{metadata.name}: declares BroadcastSpec {[s.mode for s in specs]} but "
            f"the base panel index {type(base_idx).__name__} is not a DatetimeIndex "
            "— the broadcast date mapping cannot be verified (R11 P0-09 fail-closed)"
        )
    base_dates = base_idx.normalize()
    for position, frame in enumerate(frames[1:], start=1):
        other_idx = getattr(frame, "index", None)
        if not isinstance(other_idx, pd.DatetimeIndex):
            raise ValueError(
                f"{metadata.name}: broadcast input panel {position} index "
                f"{type(other_idx).__name__} is not a DatetimeIndex — cannot verify "
                "the declared BroadcastSpec mapping (R11 P0-09 fail-closed)"
            )
        other_dates = other_idx.normalize()
        for spec in specs:
            if spec.date_mapping == "exact":
                if not other_dates.equals(base_dates):
                    raise ValueError(
                        f"{metadata.name}: BroadcastSpec(mode={spec.mode!r}, "
                        f"date_mapping='exact') input panel {position} dates are "
                        "not row-aligned with the base (R11 P0-09)"
                    )
            elif spec.date_mapping == "trading_date":
                if not set(other_dates).issubset(set(base_dates)):
                    raise ValueError(
                        f"{metadata.name}: BroadcastSpec(mode={spec.mode!r}, "
                        f"date_mapping='trading_date') input panel {position} "
                        "carries trading dates absent from the base panel "
                        "(R11 P0-09 fail-closed)"
                    )


def _validate_panel_axes(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    frames = [value for value in (*args, *kwargs.values()) if _is_panel(value)]
    if not frames:
        return
    for position, frame in enumerate(frames):
        columns = list(frame.columns)
        if len(columns) != len(set(columns)):
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate columns")
        index = getattr(frame, "index", None)
        if index is not None and not index.is_unique:
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate index values")
    if len(frames) < 2:
        return
    tags = set(metadata.tags or [])
    # R11 P0-09: a declared BroadcastSpec is the structured, provable contract.
    # When present it drives verification (each non-base panel must satisfy the
    # declared date mapping or fail closed); a bare ``allow_panel_broadcast``
    # tag without any spec remains the legacy research waiver below.
    specs = tuple(getattr(metadata, "broadcast_specs", None) or ())
    if specs:
        _verify_broadcast_specs(metadata, specs, frames)
        return
    if tags & _TYPED_BROADCAST_TAGS or "allow_panel_broadcast" in tags:
        # Review #4 R4-99 / review #5 R5-05: a bare ``allow_panel_broadcast`` is
        # the legacy blanket waiver; typed tags declare the SPECIFIC shape.
        # Beyond column *count* and instrument identity, the column *labels* are
        # now compared too — a wide panel whose columns are [A,B] must never be
        # silently broadcast onto a same-width panel whose columns are [B,C].
        base_cols = frames[0].shape[1]
        base_columns = list(frames[0].columns)
        base_instruments = _frame_instruments(frames[0])
        for position, frame in enumerate(frames[1:], start=1):
            if frame.shape[1] != base_cols:
                raise ValueError(
                    f"{metadata.name}: broadcast input panel {position} has "
                    f"{frame.shape[1]} columns != base {base_cols}"
                )
            if list(frame.columns) != base_columns:
                raise ValueError(
                    f"{metadata.name}: broadcast input panel {position} columns "
                    f"{list(frame.columns)} != base {base_columns} — same-width "
                    "panels with different instrument/column identities cannot be "
                    "broadcast (R5-05)"
                )
            if base_instruments is not None:
                other = _frame_instruments(frame)
                if other is not None and not other.issubset(base_instruments):
                    raise ValueError(
                        f"{metadata.name}: broadcast input panel {position} carries "
                        f"instruments outside the base panel (daily->minute or "
                        f"cross-symbol broadcast across different symbols is not allowed)"
                    )
        # round-7 P0: the typed broadcast tags are NOT a waiver — verify the
        # actual date/session mapping, not just the tag name.  A daily value
        # broadcast to minute t+1 (or a same_trading_date row shifted a day) is a
        # silent look-ahead.  Checks are best-effort: they fire only when both
        # axes are ``DatetimeIndex`` and skip gracefully on exotic axes.
        _verify_typed_broadcast_axes(metadata, tags, frames)
        return
    base = frames[0]
    base_columns = list(base.columns)
    base_index = getattr(base, "index", None)
    for position, frame in enumerate(frames[1:], start=1):
        if list(frame.columns) != base_columns:
            raise ValueError(f"{metadata.name}: input panel {position} columns are misaligned")
        other_index = getattr(frame, "index", None)
        if base_index is not None and other_index is not None and not other_index.equals(base_index):
            raise ValueError(f"{metadata.name}: input panel {position} index is misaligned")


def validate_operator_call(
    operator: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Central logical-call validator (review #4 R4-02).

    Every operator — ``SeriesOperator``/``PandasOperator``, the Polars bridge,
    the DuckDB operator, and modules that implement ``calculate`` directly —
    must pass through this gate so integer validation (``ParamSpec`` /
    ``param_types`` / name whitelist), panel-axis alignment (incl. typed
    broadcast), ``validate_params`` and common parameter relations are enforced
    uniformly.  Registry/dispatch layers call this instead of trusting each
    module to remember to call ``_prepare_call``.
    """
    metadata = getattr(operator, "metadata", None)
    if metadata is None:
        raise ValueError(f"{operator!r} has no metadata; cannot validate call")
    processed_args, processed_kwargs = _normalise_call(
        metadata, args, kwargs, defaults=_kernel_param_defaults(operator)
    )
    _validate_panel_axes(metadata, processed_args, processed_kwargs)
    _validate_common_integer_relations(metadata, processed_args, processed_kwargs)
    _validate_relational_specs(metadata, operator, processed_args, processed_kwargs)
    valid = operator.validate_params(*processed_args, **processed_kwargs)
    if valid is False:
        raise ValueError(f"{metadata.name}: parameter validation failed")
    return processed_args, processed_kwargs


def _validate_common_integer_relations(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    names = list(metadata.param_names or [])
    bound = {name: args[index] for index, name in enumerate(names[: len(args)])}
    bound.update(kwargs)
    window = bound.get("window")
    minimum = bound.get("min_periods")
    if isinstance(window, int) and isinstance(minimum, int) and minimum > window:
        raise ValueError(f"{metadata.name}: min_periods must not exceed window")
    k = bound.get("k")
    if isinstance(window, int) and isinstance(k, int) and k > window:
        raise ValueError(f"{metadata.name}: k must not exceed window")


def _validate_relational_specs(
    metadata: OperatorMetadata,
    operator: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Evaluate every declared RelationalParamSpec on the bound parameter values.

    R6-24: rejects guaranteed-NaN parameter combinations at the call boundary
    (e.g. ``window < 4*k+1`` for Pickands, ``min_periods == window`` for a
    regression with N-1 pairs) so search/AlphaProbe never spends expression
    budget on a combination the kernel can only fail.
    """
    specs = getattr(metadata, "relational_specs", None) or []
    if not specs:
        return
    names = list(metadata.param_names or [])
    bound: dict[str, Any] = {
        name: args[index] for index, name in enumerate(names[: len(args)])
    }
    bound.update(kwargs)
    # R6-24: a relation like ``min_line < window - (dim-1)*delay`` must be
    # evaluated on the SAME values the kernel receives — i.e. with the canonical
    # defaults merged in for any parameter the caller did not bind explicitly.
    # ``_kernel_param_defaults`` resolves the real kernel's default arguments
    # (the operator's ``_calculate_series`` is often a ``*args, **kwargs``
    # bridge, so signature introspection alone would not find them).
    defaults = _kernel_param_defaults(operator) or {}
    for name in names:
        if name not in bound and name in defaults:
            bound[name] = defaults[name]
    for spec in specs:
        if not spec.check(bound):
            raise ValueError(f"{metadata.name}: {spec.describe(bound)}")


class Operator(ABC):
    metadata: OperatorMetadata

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def validate_params(self, *args, **kwargs) -> bool:
        return True

    def _prepare_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        return validate_operator_call(self, args, kwargs)

    def __repr__(self):
        return f"<Operator: {self.metadata.name}>"

    def __str__(self):
        return f"{self.metadata.name}: {self.metadata.description}"


class SeriesOperator(Operator):
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_series(*processed_args, **processed_kwargs)

    @abstractmethod
    def _calculate_series(self, *args, **kwargs) -> pd.DataFrame:
        pass


class ScalarOperator(Operator):
    def calculate(self, *args, **kwargs) -> Any:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_scalar(*processed_args, **processed_kwargs)

    @abstractmethod
    def _calculate_scalar(self, *args, **kwargs) -> Any:
        pass


class TransformOperator(Operator):
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call((x,), kwargs)
        return self._calculate_series(processed_args[0], **processed_kwargs)


class TwoVarOperator(Operator):
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call((x, y), kwargs)
        return self._calculate_series(processed_args[0], processed_args[1], **processed_kwargs)


def register_operator(
    name: str = None,
    category: str = "general",
    business_category: str = "",
    canonical: str = "",
    source: str = "",
    backend: str | None = None,
    status: str = "implemented",
    replace: bool = False,
    replacement_reason: str = "",
    expected_old_source: str = "",
):
    """Instantiate and register an operator class.

    ``replace``/``replacement_reason`` are the explicit escape hatch for the
    registry's silent-overwrite ban (P0-31): a compatibility/override layer
    re-registering an existing canonical+backend must declare it, or load_all
    raises.  ``expected_old_source`` pins the exact source an override replaces
    (round-7 P0) so an override *chain* is independent of import order.
    """

    def decorator(cls):
        instance = cls()
        if name:
            instance.metadata.name = name
        if category:
            instance.metadata.category = category
        if business_category:
            instance.metadata.business_category = business_category
        if backend is not None:
            effective_backend = backend
            backend_explicit = True
        elif "Polars" in cls.__name__:
            effective_backend = "polars"
            backend_explicit = False
        else:
            effective_backend = "pandas_numpy"
            backend_explicit = False
        canon = canonical or (name if name else cls.__name__)
        from cleaned_operators.registry import OperatorRegistry

        name_aliases = [name] if name and name != canon else None
        OperatorRegistry.register(
            instance,
            canonical=canon,
            backend=effective_backend,
            source=source or "factor_dsl_np",
            aliases=name_aliases,
            status=status,
            backend_explicit=backend_explicit,
            replace=replace,
            replacement_reason=replacement_reason,
            expected_old_source=expected_old_source,
        )
        return cls

    return decorator
