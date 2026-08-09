# -*- coding: utf-8 -*-
"""ExecutionContract + HistoryRequirement — the SINGLE authority for
statefulness / chunking / checkpoint / lookback-warmup (WS-D findings
#256-#258).

Every layer that used to consult a scattered stateful-name list or a
``_LOOKBACK_PARAM_PRIORITY`` name-guessing tuple now routes through here:

* ``execution_contract(canonical)``  — state_model / chunking / checkpoint_schema.
* ``history_requirement(canonical, params)`` — the ONE lookback authority,
  returning a ``HistoryRequirement`` instead of an integer sentinel.
* ``factor_history_requirement(ir)`` — combined requirement for a whole factor IR.

The legacy ``FULL_HISTORY_LOOKBACK_SENTINEL`` integer survives ONLY as the
serialized-analysis encoding that ``ir.analyzer`` writes for ``requires_full_history``
factors; it is no longer the authority for anything.  ``warmup_service``,
``composite_lowering`` and the analyzer integration read the dataclasses below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

# Serialized-analysis encoding only.  ``analysis.lookback`` carries this value
# when ``requires_full_history`` was True at analysis time (see
# ``cleaned_operators.production_policy_extensions_v2``).  Consumers must treat
# it as a compatibility signal, never as a window size.
FULL_HISTORY_LOOKBACK_SENTINEL = 1_000_000_000

# The stateful seed (WS-D #256).  This replaces ``contract_hardening._STATEFUL_CANONICALS``
# and any ``_STATE_MODEL_*`` lists: nothing else may be consulted for statefulness /
# chunking.  The segmented (checkpoint-capable) members are registered in
# ``stateful_contract.StatefulCheckpointRegistry``; the rest are recursive /
# full-history-replay operators with no checkpoint-restore.
_STATEFUL_CANONICALS: frozenset[str] = frozenset({
    # --- checkpoint-capable (segmented) recursive kernels -------------------
    "ts_ema",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_ewm_cov",
    "ts_ewm_corr",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
    # --- recursive stateful operators without a checkpoint restore -----------
    # (from contract_hardening's list; ts_max_run / ts_min_run are NOT present
    # in the registry as of round-7, so they are intentionally absent here).
    "ts_sma_cn",
    "MACD",
    "KAMA",
    "Supertrend",
    "SupertrendDirection",
    "PSAR",
    "DMI_plus",
    "DMI_minus",
    "DX",
    "NATR",
    "PPO",
    "PPO_signal",
    "PPO_hist",
    "PVO",
    "PVO_signal",
    "PVO_hist",
    "KeltnerMid",
    "KeltnerUpper",
    "KeltnerLower",
    "KeltnerPosition",
    "TSI",
    "TSI_signal",
    "DEMA",
    "TEMA",
    "ChaikinOscillator",
    "ForceIndex",
    "expanding_rank",
    "hump_decay",
    "state_latch",
    "state_hold",
    "state_slew_limit",
    "state_deadband",
    "state_ewm_if",
    "state_since_reduce",
    "event_refractory",
    "directional_change_state",
    "directional_change_extent",
    "state_since_trend_tstat",
    "ts_cusum_break_score",
    "ts_threshold_cycle_period",
    "ts_threshold_cycle_asymmetry",
    "state_episode_mfe",
    "state_episode_mae",
    "state_episode_efficiency",
    "state_episode_retrace_ratio",
    "state_episode_excursion_balance",
    "ts_interval_nesting_depth",
    "candle_gap_atr",
})

_MACD_FAMILY = frozenset({"MACD", "MACD_line", "MACD_signal", "MACD_hist"})

# Window-like parameter names used ONLY as the CONSERVATIVE DEFAULT fallback
# (``_default_window_extension``) for operators with no explicit
# ``_HISTORY_TRANSFORMS`` entry (R9-P0-006).  The default treats every name as
# ``value - 1``; lag-like semantics (``value``, never ``value - 1``) are only
# applied for operators explicitly declared in ``_HISTORY_TRANSFORMS``.
# Restricted to the operator's declared parameters inside
# ``_default_window_extension``.
_WINDOW_LIKE_PARAM_NAMES = frozenset({
    "window", "span", "period", "periods", "d", "n", "lag", "lookback",
    "max_lookback", "fast", "slow", "signal", "fast_period", "slow_period",
    "signal_period", "signal_span", "fast_window", "slow_window",
    "signal_window", "short_window", "medium_window", "long_window",
    "ema_window", "atr_window", "er_window", "vol_window",
    "tenkan_window", "kijun_window", "senkou_b_window",
    "baseline_window", "price_window", "volume_window", "turnover_window",
})


@dataclass(frozen=True)
class ExecutionContract:
    """Three-axis execution contract for one canonical (WS-D #256).

    Attributes:
        state_model: ``stateless`` | ``recursive`` | ``episode`` | ``session_state``
            | ``unknown``.
        chunking: ``independent`` | ``checkpoint`` | ``required_full_history`` |
            ``unknown``.
        checkpoint_schema: opaque schema id (``None`` when no checkpoint exists).
        resolution_error: set to ``"UNKNOWN_EXECUTION_CONTRACT"`` (R10 #4) when a
            contract lookup failed and research fell back — NEVER silently
            ``stateless``/``independent`` on an internal registry error.
    """

    state_model: str = "stateless"
    chunking: str = "independent"
    checkpoint_schema: str | None = None
    resolution_error: str | None = None

    @property
    def is_stateful(self) -> bool:
        return self.state_model != "stateless"

    @property
    def requires_full_history(self) -> bool:
        # R10 #4: UNKNOWN chunking is fail-closed (treated as full-history
        # recompute), never a finite window and never a silent stateless skip.
        return self.chunking in {"required_full_history", "unknown"}


class ExecutionContractResolutionError(RuntimeError):
    """Raised in production when an execution contract cannot be resolved.

    An internal registry / checkpoint-contract lookup failure must NEVER fall a
    stateful operator back to ``stateless``/``independent`` (R10 #4).  Production
    hard-fails instead; only research falls back, and only with the explicit
    ``UNKNOWN_EXECUTION_CONTRACT`` marker.
    """


@dataclass(frozen=True)
class HistoryRequirement:
    """Machine-readable lookback requirement (WS-D #257).

    ``kind`` is one of ``finite`` / ``full_history`` / ``fiscal_period`` /
    ``session`` / ``event_count`` / ``report_count`` / ``session_count``; ``rows``
    is the minimum warm-up rows required before the first meaningful output
    (never the 1e9 sentinel).

    Event-clock kinds (``event_count`` / ``report_count`` / ``session_count``)
    count OBSERVATIONS of an event/report/session, NOT trading bars — a bar-window
    warmup cannot derive them, so ``history_requirement`` reports them as
    ``full_history`` (conservative) unless a declared ``rows`` floor exists.
    ``count`` records the declared event/report/session count when known.
    """

    kind: str = "finite"  # finite | full_history | fiscal_period | session |
    #                       # event_count | report_count | session_count
    rows: int = 2
    count: int | None = None  # event/report/session observation count (event-clock kinds)
    # R11 P1-13: the exact lookback FORMULA as a recoverable expression, e.g.
    # ``"window - 1 + lag"`` with ``semantics="exact_rows"``.  The catalog's
    # legacy ``parameters=["window", "lag"]`` only names the involved params; it
    # cannot independently recover the true warm-up rows the planner/evidence/
    # incremental runtime must use.  ``expression`` is the authoritative formula,
    # ``semantics`` describes how it maps to rows (``exact_rows`` /
    # ``parameterized_rows`` / ``event_count`` / ``full_history``).  ``None``
    # means the ``rows`` floor is authoritative and no formula is declared.
    expression: str | None = None
    semantics: str | None = None  # exact_rows | parameterized_rows | event_count | full_history

    @property
    def is_full_history(self) -> bool:
        # Event-clock kinds (event_count / report_count / session_count) are
        # measured in observations, not bars — a bar-window warmup cannot derive
        # them, so they behave as full-history for DAG composition (conservative).
        return self.kind in {
            "full_history", "event_count", "report_count", "session_count",
        }

    @property
    def is_event_clock(self) -> bool:
        return self.kind in {"event_count", "report_count", "session_count"}


# Round-11 #12: per-operator execution-contract declaration.  Operators declare
# their OWN statefulness / chunking / history-clock instead of being patched into
# the hand-maintained ``_STATEFUL_CANONICALS`` name set.  ``execution_contract()``
# resolves in order: checkpoint registry -> declared contract -> legacy seed.
# ``_STATEFUL_CANONICALS`` is kept ONLY as the legacy fallback for operators not
# yet migrated to a declared contract; new stateful operators must declare.
_DECLARED_STATEFUL: dict[str, dict[str, Any]] = {}


def declare_stateful(
    canonical: str,
    *,
    state_model: str,
    chunking: str,
    checkpoint_schema: str | None = None,
    minimum_history: int = 0,
    history_kind: str = "full_history",
    history_count: int | None = None,
) -> None:
    """Declare an operator's execution contract at its own module.

    Round-11 #12: this is the ONLY way new stateful operators enter the runtime —
    never via an edited ``_STATEFUL_CANONICALS`` list.  ``state_model`` is
    ``recursive`` / ``episode`` / ``session_state`` (never ``stateless`` here);
    ``chunking`` is ``checkpoint`` / ``required_full_history``.  ``history_kind``
    is ``full_history`` (default) or an event-clock kind (``event_count`` /
    ``report_count`` / ``session_count``) for operators whose history is measured
    in observations, not bars; ``history_count`` is the declared count.

    Re-declaring a canonical is an error (a second declaration is a drift, not a
    refinement) — use ``execution_contract_overrides()`` to inspect.
    """
    if canonical in _DECLARED_STATEFUL:
        raise ExecutionContractResolutionError(
            f"duplicate declare_stateful for {canonical!r} — a canonical may "
            "declare its execution contract exactly once"
        )
    if state_model == "stateless":
        raise ExecutionContractResolutionError(
            f"declare_stateful called for {canonical!r} with stateless; "
            "stateless operators declare nothing (remove the legacy seed entry)"
        )
    if chunking not in {"checkpoint", "required_full_history"}:
        raise ExecutionContractResolutionError(
            f"declare_stateful for {canonical!r}: chunking must be checkpoint or "
            f"required_full_history, got {chunking!r}"
        )
    _DECLARED_STATEFUL[canonical] = {
        "state_model": state_model,
        "chunking": chunking,
        "checkpoint_schema": checkpoint_schema,
        "minimum_history": int(minimum_history or 0),
        "history_kind": history_kind,
        "history_count": history_count,
    }


def declared_stateful_canonicals() -> frozenset[str]:
    """All canonicals with an operator-declared execution contract."""
    return frozenset(_DECLARED_STATEFUL)


def execution_contract_overrides() -> dict[str, dict[str, Any]]:
    """Snapshot of operator-declared execution contracts (for audits / CI)."""
    return {k: dict(v) for k, v in _DECLARED_STATEFUL.items()}


# R9-P0-004: sentinel for "a bound parameter exists but its value is not an
# integral row count" (e.g. a fractional ``window=5.9``).  The history layer
# must NEVER truncate ``int(5.9) -> 5``; a fractional window means the operator's
# history is UNKNOWN and must be treated conservatively as full history.
_UNKNOWN = object()


@dataclass(frozen=True)
class HistoryTransform:
    """Declared per-operator history-extension rule (R9-P0-005/P0-006).

    How an operator extends its children's MAX history along a DAG path:
    ``H(node) = own_history_transform(max(H(children)))``.

    ``kind``:
      * ``identity`` — pointwise / cross-sectional: adds 0 rows.
      * ``window``   — re-reads the input over a trailing window: adds ``W - 1``.
      * ``lag``      — shifts the input back ``d`` bars: adds ``d`` (NOT ``d - 1``).
      * ``compound`` — explicit ``fn(canonical, params) -> int`` for composite
        lookbacks (MACD, autocorr+lag, pattern sums, …).

    ``params`` lists the declared parameter names to read; for ``window`` /
    ``lag`` the MAX bound value is used.  ``fixed`` provides a constant lag for
    operators with no window parameter (``prev``, ``ts_ratio``).  ``fn`` is only
    meaningful for ``kind == "compound"`` and must return ``_UNKNOWN`` when any
    referenced bound parameter is un-resolvable (fractional).
    """

    kind: str = "identity"
    params: tuple[str, ...] = ()
    fixed: int | None = None
    fn: Any = None


def _resolve(canonical: str, *, strict: bool = False) -> str:
    """Resolve DSL alias -> canonical (lazy; safe before the registry is built).

    R10 #4: when ``strict`` (production) and the registry itself errors, raise
    instead of silently returning the unresolved name — alias resolution failure
    can mask a stateful operator as a different one.
    """
    if not canonical:
        return canonical
    try:
        from cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(canonical)
    except Exception as exc:
        if strict:
            raise ExecutionContractResolutionError(
                f"production: alias resolution failed for {canonical!r}: {exc}"
            ) from exc
        return canonical


def _metadata(canonical: str, *, strict: bool = False) -> Any | None:
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        return getattr(op, "metadata", None)
    except Exception as exc:
        if strict:
            raise ExecutionContractResolutionError(
                f"production: operator metadata lookup failed for {canonical!r}: {exc}"
            ) from exc
        return None


def _checkpoint_spec(canonical: str, *, strict: bool = False) -> Any | None:
    try:
        from stateful_contract import StatefulCheckpointRegistry

        return StatefulCheckpointRegistry.get(canonical)
    except Exception as exc:
        if strict:
            raise ExecutionContractResolutionError(
                f"production: checkpoint-contract lookup failed for {canonical!r}: {exc}"
            ) from exc
        return None


def execution_contract(canonical: str, *, production: bool = False) -> ExecutionContract:
    """Return the single authority execution contract for ``canonical``.

    Resolution order: the checkpoint registry decides ``checkpoint`` vs
    ``required_full_history``; the stateful seed covers operators without a
    checkpoint spec; everything else is stateless/independent.

    R10 #4 (fail-closed): an *internal* lookup failure (registry / checkpoint
    registry raising) is NOT a resolution to stateless.  Production raises
    :class:`ExecutionContractResolutionError`; research falls back with the
    explicit ``resolution_error="UNKNOWN_EXECUTION_CONTRACT"`` marker, and
    ``requires_full_history`` treats that as full-history (conservative).
    """
    resolved = _resolve(canonical, strict=production)
    lookup_failed = False
    try:
        from stateful_contract import StatefulCheckpointRegistry

        spec = StatefulCheckpointRegistry.get(resolved)
    except Exception as exc:
        if production:
            raise ExecutionContractResolutionError(
                f"production: checkpoint-contract lookup failed for {resolved!r}: {exc}"
            ) from exc
        spec = None
        lookup_failed = True  # research: remember it was an ERROR, not a miss
    if spec is not None:
        if spec.segmented_execution_supported:
            return ExecutionContract(
                state_model="recursive",
                chunking="checkpoint",
                checkpoint_schema=spec.state_schema_version,
            )
        return ExecutionContract(
            state_model="recursive",
            chunking="required_full_history",
            checkpoint_schema=spec.state_schema_version,
        )
    if resolved in _DECLARED_STATEFUL:
        # Round-11 #12: the operator's own declaration is authoritative — it
        # carries the REAL state_model (episode / recursive / session_state) and
        # chunking instead of a generic recursive/full-history fallback.
        declared = _DECLARED_STATEFUL[resolved]
        return ExecutionContract(
            state_model=declared["state_model"],
            chunking=declared["chunking"],
            checkpoint_schema=declared["checkpoint_schema"],
        )
    if resolved in _STATEFUL_CANONICALS:
        return ExecutionContract(
            state_model="recursive",
            chunking="required_full_history",
            checkpoint_schema=None,
        )
    if lookup_failed:
        # R10 #4: research falls back, but explicitly marked — never silently
        # stateless/independent on an internal registry error.  chunking=unknown
        # makes ``requires_full_history`` True (fail-closed full recompute).
        return ExecutionContract(
            state_model="unknown",
            chunking="unknown",
            checkpoint_schema=None,
            resolution_error="UNKNOWN_EXECUTION_CONTRACT",
        )
    return ExecutionContract(state_model="stateless", chunking="independent")


def _as_row_count(value: Any) -> int | None:
    """Coerce an already-validated typed param to a NON-NEGATIVE row count
    WITHOUT truncation (R9-P0-004).

    Accepts ``int`` (never ``bool``) and an integral ``float`` (``20.0`` -> 20),
    returning ``max(0, value)`` so a bound ``0`` / negative contributes zero rows
    (a zero lag ``ts_delay(x, 0)`` is valid and needs no prior bars).  A
    fractional float (``5.9``), ``bool`` or non-numeric value returns ``None``:
    such a value is NOT a window row count and must never be silently floored.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if value.is_integer():
            return max(0, int(value))
        return None  # fractional window -> unknown history, never truncate
    return None


def _kernel_signature_default(canonical: str, name: str) -> int | None:
    """Default for an unbound parameter from the operator's real kernel signature
    (P0-04).

    ``def op(x, window=120)``: when ``window`` is omitted the REAL runtime value
    is 120 — that must be what the history layer assumes, NOT ``ParamSpec.min``
    (a validity-domain boundary, never a runtime default).  Uses
    ``cleaned_operators.base._kernel_param_defaults`` so ``register_dual``
    bridges resolve through their ``_fn`` kernel.  ``None`` when the signature
    has no default for ``name`` or the operator cannot be inspected.
    """
    try:
        from cleaned_operators.base import _kernel_param_defaults
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical)
        if op is None:
            return None
        defaults = _kernel_param_defaults(op)
        if name not in defaults:
            return None
        return _as_row_count(defaults[name])
    except Exception:
        return None


def _bound_param(
    params: Mapping[str, Any],
    name: str,
    specs: Mapping[str, Any],
    canonical: str = "",
) -> int | object | None:
    """Resolve one parameter to a positive int row count, else its declared
    default, else the kernel signature default, else None.

    R9-P0-004: the history layer is a CONSUMER of already-validated bound
    params — it must never coerce a value itself.  Returns:
      * ``int``  — a genuine integral row count (int, or integral float 20.0).
      * ``_UNKNOWN`` — the param is BOUND but not an integral row count (e.g.
        ``window=5.9``).  This is UNKNOWN history; the caller must treat it
        conservatively as full history instead of truncating to 5.
      * ``None`` — param not bound and no usable default (P0-04: ``ParamSpec.min``
        is a legal-domain boundary, NEVER a runtime default, so it is not used).
    """
    if name in params and params.get(name) is not None:
        value = params[name]
        resolved = _as_row_count(value)
        if resolved is not None:
            return resolved
        # Bound to a value we cannot turn into a row count — do NOT truncate.
        return _UNKNOWN
    spec = specs.get(name)
    if spec is not None:
        candidate = getattr(spec, "default", None)
        if candidate is not None and not isinstance(candidate, bool):
            resolved = _as_row_count(candidate)
            if resolved is not None:
                return resolved
    if canonical:
        resolved = _kernel_signature_default(canonical, name)
        if resolved is not None:
            return resolved
    return None


def _specs(canonical: str) -> Mapping[str, Any]:
    meta = _metadata(canonical)
    if meta is None:
        return {}
    return getattr(meta, "param_specs", None) or {}


def _w1(params: Mapping[str, Any], specs: Mapping[str, Any], name: str, default: int = 0, *, canonical: str = "") -> int | object:
    """Contribution ``max(0, bound - 1)`` for a window param; ``_UNKNOWN``
    propagates (a fractional bound value is unknown history)."""
    value = _bound_param(params, name, specs, canonical=canonical)
    if value is _UNKNOWN:
        return _UNKNOWN
    if value is None:
        return default
    return max(0, value - 1)


def _w0(params: Mapping[str, Any], specs: Mapping[str, Any], name: str, default: int = 0, *, canonical: str = "") -> int | object:
    """Contribution of a raw window/lag value; ``_UNKNOWN`` propagates."""
    value = _bound_param(params, name, specs, canonical=canonical)
    if value is _UNKNOWN:
        return _UNKNOWN
    if value is None:
        return default
    return max(0, value)


# --- compound history-extension formulas (mirror ``ir.analyzer``) ------------
def _macd_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    fast = _bound_param(params, "fast", specs, canonical=canonical)
    slow = _bound_param(params, "slow", specs, canonical=canonical)
    signal = _bound_param(params, "signal", specs, canonical=canonical)
    if _UNKNOWN in (fast, slow, signal):
        return _UNKNOWN
    fast = fast or 12
    slow = slow or 26
    signal = signal or 9
    return max(fast, slow) - 1 + max(signal - 1, 0)


def _ppo_pvo_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    slow = _w1(params, specs, "slow_window", canonical=canonical)
    signal = _w1(params, specs, "signal_window", canonical=canonical)
    if _UNKNOWN in (slow, signal):
        return _UNKNOWN
    return slow + signal


def _tsi_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    long_w = _w1(params, specs, "long_window", canonical=canonical)
    short_w = _w1(params, specs, "short_window", canonical=canonical)
    signal_w = _w1(params, specs, "signal_window", canonical=canonical)
    if _UNKNOWN in (long_w, short_w, signal_w):
        return _UNKNOWN
    return long_w + short_w + signal_w


def _dema_tema_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    window = _bound_param(params, "window", specs, canonical=canonical)
    if window is _UNKNOWN:
        return _UNKNOWN
    multiplier = 3 if canonical == "TEMA" else 2
    return multiplier * max(0, (window or 1) - 1)


def _ichimoku_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    parts = []
    for name in ("tenkan_window", "kijun_window", "senkou_b_window"):
        value = _bound_param(params, name, specs, canonical=canonical)
        if value is _UNKNOWN:
            return _UNKNOWN
        parts.append(max(0, (value or 1) - 1))
    return max(parts)


def _window_plus_lag_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """``(window - 1) + lag`` for autocorr / regression / event kernels."""
    specs = _specs(canonical)
    w = _w1(params, specs, "window", canonical=canonical)
    lag = _w0(params, specs, "lag", canonical=canonical)
    if _UNKNOWN in (w, lag):
        return _UNKNOWN
    return w + lag


def _event_response_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    w = _w1(params, specs, "history_window", canonical=canonical)
    horizon = _w0(params, specs, "horizon", canonical=canonical)
    if _UNKNOWN in (w, horizon):
        return _UNKNOWN
    return w + horizon


def _stochastic_d_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    window = _bound_param(params, "window", specs, canonical=canonical)
    if window is _UNKNOWN:
        return _UNKNOWN
    return (window or 0) + 1


def _ulcer_index_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    window = _bound_param(params, "window", specs, canonical=canonical)
    if window is _UNKNOWN:
        return _UNKNOWN
    return 2 * max(0, (window or 1) - 1)


def _candlestick_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    body = _bound_param(params, "body_window", specs, canonical=canonical)
    shadow = _bound_param(params, "shadow_window", specs, canonical=canonical)
    if _UNKNOWN in (body, shadow):
        return _UNKNOWN
    return max(body or 1, shadow or 1) + 4


def _sum_two_extension(canonical: str, params: Mapping[str, Any], n1: str, n2: str) -> int | object:
    specs = _specs(canonical)
    a = _w0(params, specs, n1, canonical=canonical)
    b = _w0(params, specs, n2, canonical=canonical)
    if _UNKNOWN in (a, b):
        return _UNKNOWN
    return a + b


def _flag_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    return _sum_two_extension(canonical, params, "impulse_window", "flag_window")


def _pennant_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    return _sum_two_extension(canonical, params, "impulse_window", "pennant_window")


def _cup_handle_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    return _sum_two_extension(canonical, params, "cup_window", "handle_window")


def _retest_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    w = _w0(params, specs, "window", canonical=canonical)
    wait = _w0(params, specs, "max_wait", canonical=canonical)
    if _UNKNOWN in (w, wait):
        return _UNKNOWN
    return w + wait


def _impulse_volume_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    w = _w0(params, specs, "window", canonical=canonical)
    base = _w0(params, specs, "baseline_window", canonical=canonical)
    if _UNKNOWN in (w, base):
        return _UNKNOWN
    return w + base


def _channel_width_slope_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    specs = _specs(canonical)
    h = _w1(params, specs, "history_window", canonical=canonical)
    left = _w0(params, specs, "left_window", canonical=canonical)
    right = _w0(params, specs, "right_window", canonical=canonical)
    w = _w1(params, specs, "window", canonical=canonical)
    if _UNKNOWN in (h, left, right, w):
        return _UNKNOWN
    return h + left + right + w


def _window_transform(*params: str) -> HistoryTransform:
    return HistoryTransform(kind="window", params=params)


def _lag_transform(*params: str) -> HistoryTransform:
    return HistoryTransform(kind="lag", params=params)


def _compound_transform(fn: Any) -> HistoryTransform:
    return HistoryTransform(kind="compound", fn=fn)


# R9-P0-006: per-operator HistoryTransform declarations — the ONLY place lag is
# treated as ``value`` (never ``value - 1``) and compound lookbacks are composed.
# Operators NOT listed here fall back to ``_default_window_extension`` (the
# conservative ``value - 1`` name-based default for window-like params).
_HISTORY_TRANSFORMS: dict[str, HistoryTransform] = {}

# Rolling window operators: re-read the input over ``W`` bars -> W - 1 prior.
for _canon in (
    "ts_mean", "ts_std", "ts_var", "ts_zscore", "ts_rank", "ts_max", "ts_min",
    "ts_sum", "ts_median", "ts_skew", "ts_kurt", "ts_product", "ts_corr",
    "ts_cov", "ts_beta", "ts_sharpe", "ts_mad", "ts_decay_linear",
    "ts_sum_decay", "WMA", "ts_regression",
):
    _HISTORY_TRANSFORMS[_canon] = _window_transform("window")
# EMA-family uses ``span``; argmax/argmin use ``d``.
for _canon in ("ts_ema", "EMA"):
    _HISTORY_TRANSFORMS[_canon] = _window_transform("span")
for _canon in ("ts_argmax", "ts_argmin"):
    _HISTORY_TRANSFORMS[_canon] = _window_transform("d")

# Lag / delay operators: a pure lag of ``d`` needs ``d`` prior bars, never d-1.
for _canon in ("ts_delay", "delay", "ts_delta"):
    _HISTORY_TRANSFORMS[_canon] = _lag_transform("n", "d", "lag", "periods", "window")
for _canon in ("ts_pct", "ts_log_return"):
    _HISTORY_TRANSFORMS[_canon] = _lag_transform("d", "n", "lag", "periods", "window")
for _canon in ("MOM", "ROC"):
    _HISTORY_TRANSFORMS[_canon] = _lag_transform("window", "d", "n")
for _canon in ("prev", "ts_ratio"):
    _HISTORY_TRANSFORMS[_canon] = HistoryTransform(kind="lag", fixed=1)
# Round-11 #11: cross_event is a pure lag-1 crossing detector (x_t vs x_{t-1}
# and y_t vs y_{t-1}) — STATELESS, needs exactly 1 prior bar, and was wrongly
# listed in the stateful seed.  A finite lag transform gives it its 1-row warmup.
_HISTORY_TRANSFORMS["cross_event"] = HistoryTransform(kind="lag", fixed=1)

# Window + lag kernels.
for _canon in ("volume_autocorr", "turnover_autocorr", "ts_autocorr"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_window_plus_lag_extension)
for _canon in (
    "event_historical_response_mean",
    "event_historical_response_sign_balance",
):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_event_response_extension)

# MACD / PPO / PVO / TSI / DEMA / TEMA / ichimoku compound families.
for _canon in _MACD_FAMILY:
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_macd_extension)
for _canon in ("PPO", "PPO_signal", "PPO_hist", "PVO", "PVO_signal", "PVO_hist"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_ppo_pvo_extension)
for _canon in ("TSI", "TSI_signal"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_tsi_extension)
for _canon in ("DEMA", "TEMA"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_dema_tema_extension)

# Pattern / structure families.
_HISTORY_TRANSFORMS["StochasticD"] = _compound_transform(_stochastic_d_extension)
_HISTORY_TRANSFORMS["ulcer_index"] = _compound_transform(_ulcer_index_extension)
_HISTORY_TRANSFORMS["candlestick_pattern"] = _compound_transform(_candlestick_extension)
_HISTORY_TRANSFORMS["pattern_cup_handle"] = _compound_transform(_cup_handle_extension)
for _canon in ("pattern_bull_flag", "pattern_bear_flag"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_flag_extension)
for _canon in ("pattern_bull_pennant", "pattern_bear_pennant"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_pennant_extension)
for _canon in ("pattern_breakout_retest", "pattern_breakdown_retest"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_retest_extension)
_HISTORY_TRANSFORMS["ts_impulse_volume"] = _compound_transform(_impulse_volume_extension)
_HISTORY_TRANSFORMS["ts_channel_width_slope"] = _compound_transform(
    _channel_width_slope_extension
)


def _nested_pca_resid_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """R10-P0-008: ``panel_rolling_pca_resid_vol`` / ``_momentum`` are TWO
    stacked rolling windows — a rolling PCA of ``window`` rows followed by a
    rolling ``window`` std/sum of the residuals.  The maturity history is
    ~``2 * (window - 1)`` bars, not ``window``.  A generic parser reading one
    ``window`` param would underestimate the warm-up by half.
    """
    specs = _specs(canonical)
    w = _bound_param(params, "window", specs, canonical=canonical)
    if w is _UNKNOWN:
        return _UNKNOWN
    if w is None:
        w = 120
    return max(2, 2 * (w - 1))


for _canon in ("panel_rolling_pca_resid_vol", "panel_rolling_pca_resid_momentum"):
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_nested_pca_resid_extension)

# Report-period operators: their warm-up is a DATA-DRIVEN fiscal-event lookback
# (a report-period calendar), never a bar-window guess.  Ported from the
# analyzer's ``_financial_lookback`` so the history layer is the single
# authority (P0-03): production with an unavailable fiscal calendar fails
# closed (conservative full history) instead of silently under-allocating.
_FIN_REPORT_PERIOD_CANONICALS = frozenset({
    "fin_lag", "fin_diff", "fin_pct_change", "fin_log_change", "fin_qoq",
    "fin_yoy", "fin_ttm", "fin_ttm_quarterly", "fin_quarter_from_cumulative",
    "fin_ttm_cumulative", "fin_average_balance", "fin_growth", "fin_cagr",
    "fin_growth_acceleration", "fin_growth_change", "fin_growth_volatility",
    "fin_growth_stability", "fin_growth_persistence", "fin_std", "fin_mad",
    "fin_cv", "fin_stability", "fin_range", "fin_zscore_history",
    "fin_percentile_history", "fin_trend_slope", "fin_trend_r2",
    "fin_trend_tstat", "fin_trend_acceleration", "fin_monotonicity",
    "fin_positive_streak", "fin_negative_streak", "fin_sign_change_count",
    "fin_turnover", "fin_divergence", "fin_working_capital_change",
    "fin_beat_streak", "fin_miss_streak",
})


def _report_period_count(canonical: str, params: Mapping[str, Any]) -> int:
    """Number of report periods a fin_* operator traverses (mirrors the
    analyzer's ``_report_period_count``; the warm-up must span them)."""
    def value(name: str, default: int = 0) -> int:
        raw = params.get(name)
        if isinstance(raw, bool):
            return default
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    if canonical == "fin_growth_change":
        return value("growth_periods", 4) + value("compare_periods", 1)
    if canonical in {
        "fin_growth_volatility", "fin_growth_stability", "fin_growth_persistence",
    }:
        return value("growth_periods", 1) + value("window_periods", 8)
    if canonical == "fin_trend_acceleration":
        return max(value("short_periods", 4), value("long_periods", 8))
    if canonical == "fin_turnover":
        return value("average_periods", 2)
    if canonical in {"fin_positive_streak", "fin_negative_streak", "fin_beat_streak", "fin_miss_streak"}:
        return value("max_periods", 8)
    if canonical in {"fin_yoy", "fin_ttm", "fin_ttm_quarterly", "fin_ttm_cumulative"}:
        return value("periods_per_year", 4)
    if canonical == "fin_quarter_from_cumulative":
        return 2
    return max(
        1, value("periods"), value("window_periods"), value("average_periods"),
        value("long_periods"), value("max_periods"),
    )


def _financial_report_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """Data-driven fiscal-event lookback for report-period operators (P0-03).

    Prefers the active financial source's period calendar; a missing/errored
    source or period calendar is UNKNOWN history (conservative).  The
    production fail-closed / research heuristic split happens in
    ``history_requirement`` / ``own_history_requirement`` — NEVER a silent
    80-rows/period heuristic inside the single-authority path for production.
    """
    n_events = _report_period_count(canonical, params)
    try:
        from backend.financial_semantics import (
            fiscal_event_lookback,
            get_active_financial_source,
        )

        data_driven = fiscal_event_lookback(get_active_financial_source(), None, n_events)
    except Exception:
        data_driven = None
    if data_driven is not None:
        try:
            return max(0, int(data_driven))
        except (TypeError, ValueError):
            return _UNKNOWN
    return _UNKNOWN


def _financial_research_heuristic(
    canonical: str, params: Mapping[str, Any]
) -> int:
    """Research-only fallback: 80 trading rows per report period (explicit,
    never silent — production must fail closed instead)."""
    import os

    try:
        rows_per_period = (
            int(os.environ.get("FACTOR_ENGINE_REPORT_PERIOD_LOOKBACK_ROWS", "80")) or 80
        )
    except ValueError:
        rows_per_period = 80
    return max(1, rows_per_period) * max(1, _report_period_count(canonical, params))


for _canon in _FIN_REPORT_PERIOD_CANONICALS:
    _HISTORY_TRANSFORMS[_canon] = _compound_transform(_financial_report_extension)


def _apply_transform(
    transform: HistoryTransform,
    canonical: str,
    params: Mapping[str, Any],
) -> int | object:
    if transform.kind == "identity":
        return 0
    if transform.kind == "lag":
        if transform.fixed is not None:
            return transform.fixed
        specs = _specs(canonical)
        best = 0
        for name in transform.params:
            value = _bound_param(params, name, specs, canonical=canonical)
            if value is _UNKNOWN:
                return _UNKNOWN
            if value is not None:
                best = max(best, value)
        return best
    if transform.kind == "window":
        specs = _specs(canonical)
        best = 0
        for name in transform.params:
            value = _bound_param(params, name, specs, canonical=canonical)
            if value is _UNKNOWN:
                return _UNKNOWN
            if value is not None:
                best = max(best, value - 1)
        return best
    if transform.kind == "compound":
        return transform.fn(canonical, params)
    return 0


def _default_window_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """Conservative DEFAULT history extension for operators WITHOUT an explicit
    ``_HISTORY_TRANSFORMS`` declaration (R9-P0-006).

    Only parameters the operator declares in its ``param_names`` are considered,
    each is bounded by its declared ParamSpec default/min, and every window-like
    name contributes ``value - 1``.  A bound but fractional window value is
    UNKNOWN history (``_UNKNOWN``) — never silently truncated to ``value - 1``.
    """
    meta = _metadata(canonical)
    if meta is None:
        return 0
    specs = getattr(meta, "param_specs", None) or {}
    param_names = tuple(getattr(meta, "param_names", None) or ())
    rows = 0
    for name in param_names:
        if name not in _WINDOW_LIKE_PARAM_NAMES:
            continue
        value = _bound_param(params, name, specs, canonical=canonical)
        if value is _UNKNOWN:
            return _UNKNOWN
        if value is not None:
            rows = max(rows, value - 1)
    return rows


def _declared_history_extension(
    canonical: str, params: Mapping[str, Any]
) -> int | object | None:
    """Round-11 #13/#14: declared ``ParamSpec.history_semantics`` / ``history_formula``.

    The machine-readable replacement for the ``window/span/lookback`` name
    guessing: a parameter declares how it contributes history.  Returns:
      * ``int``  — declared compound/row history (authoritative).
      * ``_UNKNOWN`` — a declared event-clock semantics (report_events /
        session_slots / event_count / report_count / session_count) that a
        bar-window warmup cannot derive, or a formula that references an
        unresolvable bound value (conservative full history).
      * ``None`` — no declared history contract; caller falls back.
    """
    meta = _metadata(canonical)
    if meta is None:
        return None
    specs = getattr(meta, "param_specs", None) or {}
    param_names = tuple(getattr(meta, "param_names", None) or ())
    formulas: list[tuple[str, str]] = []
    event_clock = False
    semantics_rows = 0
    semantics_known = False
    for name in param_names:
        spec = specs.get(name)
        if spec is None:
            continue
        formula = getattr(spec, "history_formula", None)
        if formula:
            formulas.append((name, str(formula)))
            continue
        sem = getattr(spec, "history_semantics", None)
        if not sem:
            continue
        semantics_known = True
        if sem in {
            "report_events", "session_slots",
            "event_count", "report_count", "session_count",
        }:
            event_clock = True
            continue
        if sem not in {"exact_rows", "max_rows", "finite_observations", "trailing_contiguous"}:
            continue
        value = _bound_param(params, name, specs, canonical=canonical)
        if value is _UNKNOWN:
            return _UNKNOWN
        if value is None:
            continue
        rows = int(value) if sem in {"exact_rows", "finite_observations", "trailing_contiguous"} \
            else max(0, int(value) - 1)
        semantics_rows = max(semantics_rows, rows)
    if not formulas and not semantics_known:
        return None
    if formulas:
        from cleaned_operators.base import _eval_rel_ast, parse_relational_expression

        best = 0
        for _name, formula in formulas:
            try:
                tree, referenced = parse_relational_expression(formula)
            except ValueError:
                return _UNKNOWN
            ns: dict[str, Any] = {}
            for pname in referenced:
                if pname not in param_names:
                    return _UNKNOWN  # formula references an undeclared param
                value = _bound_param(params, pname, specs)
                if value is _UNKNOWN or value is None:
                    return _UNKNOWN
                ns[pname] = value
            try:
                result = _eval_rel_ast(tree, ns)
            except Exception:
                return _UNKNOWN
            if isinstance(result, bool) or not isinstance(result, (int, float)):
                return _UNKNOWN
            if not math.isfinite(float(result)):
                return _UNKNOWN
            best = max(best, max(0, int(result)))
        if event_clock:
            return _UNKNOWN
        return best
    if event_clock:
        return _UNKNOWN
    return semantics_rows


def _own_history_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """The operator's OWN prior-bar extension beyond its children's max history.

    R9-P0-005/P0-006: an explicit per-operator ``HistoryTransform`` is
    authoritative; round-11 #13/#14 declared ``ParamSpec`` history semantics /
    formulas come next (machine-readable, no name guessing); the conservative
    default fallback scans declared window-like params with ``value - 1``.
    Returns ``_UNKNOWN`` for an un-resolvable (fractional) bound window value or
    an event-clock history.
    """
    transform = _HISTORY_TRANSFORMS.get(canonical)
    if transform is not None:
        return _apply_transform(transform, canonical, params)
    if canonical.startswith("ichimoku_"):
        return _ichimoku_extension(canonical, params)
    declared = _declared_history_extension(canonical, params)
    if declared is not None:
        return declared
    return _default_window_extension(canonical, params)


# ---------------------------------------------------------------------------
# P0-01 forward impact: how many FUTURE output bars a single changed input bar
# affects.  This is the second half of the temporal dependency —
# ``input[t]`` changes -> output in ``[t, t + forward_impact]`` (the scheduler
# then expands BACKWARD history for that output window).  For most finite
# operators forward impact equals backward history (``ts_mean(w)`` reads W rows,
# so a change at ``t`` reaches output ``t..t+W-1``).  Recursive / stateful /
# event-clock operators are UNBOUNDED: a change propagates to the end of the
# series.  ``forward_impact()`` returns ``None`` for unbounded.
# ---------------------------------------------------------------------------
_UNBOUNDED_FORWARD = object()


def _event_response_forward_extension(canonical: str, params: Mapping[str, Any]) -> int | object:
    """Forward impact of an event-response kernel = the response ``horizon``
    only (the event at ``t`` is answered over the NEXT ``horizon`` bars; the
    backward ``history_window`` is warm-up, not forward reach)."""
    specs = _specs(canonical)
    horizon = _w0(params, specs, "horizon", canonical=canonical)
    if horizon is _UNKNOWN:
        return _UNKNOWN
    return max(0, int(horizon or 0))


_FORWARD_IMPACT_FNS: dict[str, Any] = {
    "event_historical_response_mean": _event_response_forward_extension,
    "event_historical_response_sign_balance": _event_response_forward_extension,
}


def _own_forward_impact(canonical: str, params: Mapping[str, Any]) -> int | object:
    """One node's own forward impact; ``_UNBOUNDED_FORWARD`` when a change
    propagates to the end of the series (recursive / stateful / event-clock /
    unresolvable window)."""
    contract = execution_contract(canonical)
    if contract.requires_full_history or contract.state_model != "stateless":
        return _UNBOUNDED_FORWARD
    fn = _FORWARD_IMPACT_FNS.get(canonical)
    if fn is not None:
        result = fn(canonical, params or {})
        return _UNBOUNDED_FORWARD if result is _UNKNOWN else result
    ext = _own_history_extension(canonical, params or {})
    if ext is _UNKNOWN:
        return _UNBOUNDED_FORWARD
    return ext


def forward_impact(
    canonical: str,
    params: Mapping[str, Any] | None = None,
    *,
    production: bool = False,
) -> int | None:
    """Finite future-bar impact of one changed input bar on this operator's
    output; ``None`` = unbounded (the change reaches every later output)."""
    resolved = _resolve(canonical, strict=production)
    contract = execution_contract(resolved, production=production)
    if contract.requires_full_history or contract.state_model != "stateless":
        return None
    ext = _own_forward_impact(resolved, params or {})
    if ext is _UNKNOWN or ext is _UNBOUNDED_FORWARD:
        return None
    return max(0, int(ext))


def factor_forward_impact(ir: Any | None) -> int | None:
    """Combined forward impact of a whole factor IR (P0-01).

    Composes along DAG paths like ``factor_history_requirement``:
    ``F(node) = own_forward(node) + max(F(children))`` — ``ts_delay(ts_mean(x,20),5)``
    reaches ``19 + 5 = 24`` future bars.  A single unbounded node (recursive /
    stateful / event-clock) makes the whole factor unbounded (``None``).
    """
    if ir is None:
        return 0

    def walk(node: Any) -> int | None:
        if node is None:
            return 0
        children = tuple(getattr(node, "inputs", ()) or ())
        child_impacts = [walk(child) for child in children]
        if any(impact is None for impact in child_impacts):
            return None
        child_max = max(child_impacts, default=0)
        op = getattr(node, "op", None)
        if not op:
            # Column / literal / opaque node: no forward reach of its own.
            return child_max
        own = forward_impact(str(op), _node_params(node))
        if own is None:
            return None
        return child_max + own

    return walk(ir)


def _minimum_warmup_rows(
    canonical: str, params: Mapping[str, Any] | None, *, production: bool = False
) -> tuple[int, bool]:
    """Return ``(rows, unknown)`` for a single operator.

    ``rows`` is the finite minimum warm-up estimate (``>= 2``, incl. the
    checkpoint's ``minimum_history``); ``unknown`` is True when a declared
    window param had an un-resolvable fractional value and the operator's
    history must be treated as UNKNOWN (conservative full history).
    """
    spec = _checkpoint_spec(canonical, strict=production)
    min_rows = int(getattr(spec, "minimum_history", 0) or 0) if spec is not None else 0
    declared = _own_history_extension(canonical, params or {})
    if declared is _UNKNOWN:
        return max(2, min_rows), True
    return max(2, min_rows, declared), False


def history_requirement(
    canonical: str,
    params: Mapping[str, Any] | None = None,
    *,
    production: bool = False,
) -> HistoryRequirement:
    """THE one lookback authority for a single operator.

    Returns a ``HistoryRequirement`` for a canonical — never a raw integer and
    never the 1e9 sentinel.  Recursive / full-history-replay operators report
    ``kind='full_history'``; a bound-but-unresolvable (fractional) window param
    is UNKNOWN history and is also reported as ``kind='full_history'``
    (conservative, never truncated).  Everything else reports a finite row count
    derived from the operator's declared history contract (safe default: finite,
    2 rows).

    R10 #4: when ``production``, an internal contract lookup failure raises
    :class:`ExecutionContractResolutionError` instead of silently resolving to a
    finite/stateless contract.
    """
    resolved = _resolve(canonical, strict=production)
    contract = execution_contract(resolved, production=production)
    rows, unknown = _minimum_warmup_rows(resolved, params, production=production)
    declared = _DECLARED_STATEFUL.get(resolved)
    if declared is not None and declared.get("history_kind") in {
        "event_count", "report_count", "session_count",
    }:
        # Round-11 #7/#8/#71: an event-clock operator's history is measured in
        # observations (updates / reports / sessions), not trading bars — a
        # bar-window warmup cannot derive it.  Conservative full-history, with
        # the observation count exposed for the planner to reason about.
        return HistoryRequirement(
            kind=declared["history_kind"],
            rows=max(rows, int(declared.get("minimum_history") or 0)),
            count=declared.get("history_count"),
        )
    if contract.requires_full_history or unknown:
        if (
            resolved in _FIN_REPORT_PERIOD_CANONICALS
            and unknown
            and not production
        ):
            # P0-03 / R11-P1-12: a report-period operator whose data-driven
            # fiscal calendar is unavailable FAILS CLOSED in production
            # (full history).  Research gets the explicit 80-rows/period
            # heuristic so fin factors do not silently full-replay.
            return HistoryRequirement(
                kind="finite",
                rows=max(2, _financial_research_heuristic(resolved, params or {})),
            )
        return HistoryRequirement(kind="full_history", rows=rows)
    return HistoryRequirement(kind="finite", rows=rows)


def _own_history_requirement(canonical: str, params: Mapping[str, Any]) -> HistoryRequirement:
    """History requirement contribution of ONE IR node for DAG-path composition.

    R9-P0-005: ``H(node) = own_history_transform(max(H(children)))``.  This
    returns the node's OWN contribution (finite row extension, or a
    full-history marker for recursive / unknown operators) — it does NOT include
    the children's rows; the caller composes them.
    """
    resolved = _resolve(canonical)
    contract = execution_contract(resolved)
    if contract.requires_full_history:
        rows, _ = _minimum_warmup_rows(resolved, params)
        declared = _DECLARED_STATEFUL.get(resolved)
        if declared is not None and declared.get("history_kind") in {
            "event_count", "report_count", "session_count",
        }:
            return HistoryRequirement(
                kind=declared["history_kind"],
                rows=rows,
                count=declared.get("history_count"),
            )
        return HistoryRequirement(kind="full_history", rows=rows)
    extension = _own_history_extension(resolved, params or {})
    if extension is _UNKNOWN:
        rows, _ = _minimum_warmup_rows(resolved, params)
        return HistoryRequirement(kind="full_history", rows=rows)
    return HistoryRequirement(kind="finite", rows=extension)


def own_history_requirement(
    canonical: str, params: Mapping[str, Any] | None = None
) -> HistoryRequirement:
    """Public per-node history requirement: ONE node's OWN contribution
    (excluding children) for DAG-path composition (P0-03).

    This is the single per-node history authority the analyzer delegates to —
    ``H(node) = own(node) composed over max(H(children))`` is done by
    ``factor_history_requirement`` / the analyzer's DAG walk.  A report-period
    operator whose data-driven fiscal calendar is unavailable gets the explicit
    research heuristic here (DAG composition is mode-agnostic); the production
    fail-closed lives in ``history_requirement``.
    """
    resolved = _resolve(canonical)
    req = _own_history_requirement(resolved, params or {})
    if req.is_full_history and resolved in _FIN_REPORT_PERIOD_CANONICALS:
        return HistoryRequirement(
            kind="finite",
            rows=max(2, _financial_research_heuristic(resolved, params or {})),
        )
    return req


def _node_params(node: Any) -> dict[str, Any]:
    """Extract bound operator params from an IRNode (kwargs attrs + positional
    literal inputs mapped through the operator's declared ``param_names``)."""
    params: dict[str, Any] = {}
    if getattr(node, "attrs", None):
        params.update(dict(node.attrs))
    meta = _metadata(getattr(node, "op", ""))
    if meta is None:
        return params
    param_names = tuple(getattr(meta, "param_names", None) or ())
    for index, child in enumerate(getattr(node, "inputs", ()) or ()):
        if getattr(child, "op", None) == "literal":
            name = param_names[index] if index < len(param_names) else None
            if name:
                params.setdefault(name, getattr(child, "attrs", {}).get("value"))
    return params


def factor_history_requirement(ir: Any | None) -> HistoryRequirement:
    """Combined history requirement for a whole factor IR.

    R9-P0-005: requirements compose along DAG PATHS (bottom-up), never
    max-over-tree.  ``H(node) = own_history_transform(max(H(children)))`` — so
    ``ts_mean(ts_delay(x, 20), 60)`` needs ``20 + (60 - 1) = 79`` prior bars,
    NOT ``max(20, 59) = 59``.  Parallel branches take the max; a full-history
    branch anywhere (or a recursive / unknown operator) makes the whole factor
    full-history.  Leaf columns/literals contribute 0.  The finite result keeps
    the historic ``>= 2`` floor so warmup/planner never under-allocate.
    """
    if ir is None:
        return HistoryRequirement(kind="finite", rows=2)

    def walk(node: Any) -> HistoryRequirement:
        if node is None:
            return HistoryRequirement(kind="finite", rows=0)
        children = tuple(getattr(node, "inputs", ()) or ())
        child_reqs = [walk(child) for child in children]
        child_rows = max((req.rows for req in child_reqs), default=0)
        any_full = any(req.is_full_history for req in child_reqs)
        op = getattr(node, "op", None)
        if not op:
            # Column / literal / opaque node: contributes no history of its own.
            return HistoryRequirement(
                kind="full_history" if any_full else "finite", rows=child_rows
            )
        own = _own_history_requirement(str(op), _node_params(node))
        if own.is_full_history or any_full:
            return HistoryRequirement(
                kind="full_history", rows=max(child_rows, own.rows)
            )
        return HistoryRequirement(kind="finite", rows=child_rows + own.rows)

    result = walk(ir)
    if not result.is_full_history:
        result = HistoryRequirement(kind="finite", rows=max(2, result.rows))
    return result


def is_full_history_lookback(lookback: int) -> bool:
    """True when a raw ``analysis.lookback`` encodes full-history replay via the
    legacy serialized sentinel.  Kept only for serialized-analysis compatibility;
    the authority is ``history_requirement`` / ``factor_history_requirement``."""
    try:
        return int(lookback) >= FULL_HISTORY_LOOKBACK_SENTINEL
    except (TypeError, ValueError):
        return False


__all__ = [
    "ExecutionContract",
    "HistoryRequirement",
    "FULL_HISTORY_LOOKBACK_SENTINEL",
    "declared_stateful_canonicals",
    "declare_stateful",
    "execution_contract",
    "execution_contract_overrides",
    "factor_forward_impact",
    "factor_history_requirement",
    "forward_impact",
    "history_requirement",
    "is_full_history_lookback",
    "own_history_requirement",
]
